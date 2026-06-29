import torch
from torch import nn
from transformers import VJEPA2ForVideoClassification
import logging
import copy


def build_vjepa2(
    num_classes,
    model_name,
    trainable_parameters_configuration="last_block+predictor+pool+head",
    pooling_mode="attentive",
    predict_future_temporal_steps=0,
    prediction_future_frames=0,
    classify_on_predicted_only=False,
):
    """
    Build a VJEPA2 model with configurable pooling and classification head.

    Args:
        num_classes: Number of output classes
        model_name: HuggingFace model name (e.g., "facebook/vjepa2-vitl-fpc16-256-ssv2")
        trainable_parameters_configuration: Freezing trainable_parameters_configuration
            - "pooler+head": Pooler + classifier trainable
            - "last_block+pool+head": Last encoder block + LayerNorm + Pooler + classifier trainable (predictor frozen)
            - "last_block+predictor+pool+head": Last encoder block + LayerNorm + Predictor + Pooler + classifier trainable
        pooling_mode: How to pool sequence features
            - "attentive": Full VJEPA2AttentivePooler (3 self-attn + 1 cross-attn with MLP)
            - "mean": Simple mean pooling over sequence
        predict_future_temporal_steps: Number of future temporal positions to predict (default: 0, disabled).
            Each temporal step covers tubelet_size frames (typically 2). For example, to predict 1 second
            of future at 16fps with tubelet_size=2, use predict_future_temporal_steps=8.
            The predictor generates latent representations for these future positions,
            which are concatenated with the encoder output before pooling.
        prediction_future_frames: Number of actual future frames loaded alongside the context (default: 0).
            The dataset must load num_frames + prediction_future_frames total frames. The encoder processes
            all of them; the classifier/pooler receives only the context portion (first num_frames).
            The predictor is trained to predict the future portion from context via prediction loss.

    Returns:
        VJEPA2Adapted model instance
    """
    accepted_pretrained_models = ["facebook/vjepa2-vitl-fpc16-256-ssv2"]

    if model_name not in accepted_pretrained_models:
        raise ValueError(f"Unsupported VJEPA2 model_name: {model_name}. Only {accepted_pretrained_models} are supported.")

    model = VJEPA2Adapted(
        num_classes=num_classes,
        model_name=model_name,
        pooling_mode=pooling_mode,
        predict_future_temporal_steps=predict_future_temporal_steps,
        prediction_future_frames=prediction_future_frames,
        classify_on_predicted_only=classify_on_predicted_only,
    )
    model.set_trainable_parameters(trainable_parameters_configuration)
    logging.info(
        f"Built VJEPA2 with trainable_parameters_configuration='{trainable_parameters_configuration}', "
        f"pooling='{pooling_mode}', classes={num_classes}, "
        f"predict_future_temporal_steps={predict_future_temporal_steps}, "
        f"prediction_future_frames={prediction_future_frames}, "
        f"classify_on_predicted_only={classify_on_predicted_only}"
    )
    return model




# ============================================================================
# Main Model Classes
# ============================================================================

class VJEPA2Wrap(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.model = VJEPA2ForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label={i: f"CLASS_{i}" for i in range(num_classes)},
            label2id={f"CLASS_{i}": i for i in range(num_classes)},
        )
        self.hidden_size = self.model.config.hidden_size

        # Set pooler attention layers to use eager attention (required for attention weight extraction)
        self._set_pooler_attn_implementation("eager")

    def _set_pooler_attn_implementation(self, attn_implementation: str):
        """
        Set the attention implementation for the pooler's cross-attention layer only.
        This allows using SDPA for the backbone and pooler self-attention while using
        eager for cross-attention to enable attention weight extraction.
        """
        pooler = self.model.pooler

        # Create a modified config for the specified attention implementation
        eager_config = copy.deepcopy(self.model.config)
        eager_config._attn_implementation = attn_implementation

        # Only update cross-attention layer in pooler (where we extract attention weights)
        if hasattr(pooler, 'cross_attention_layer') and hasattr(pooler.cross_attention_layer, 'cross_attn'):
            pooler.cross_attention_layer.cross_attn.config = eager_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        return outputs


class VJEPA2Adapted(VJEPA2Wrap):
    """
    Extended VJEPA2 wrapper with configurable pooling.

    Pooling modes:
        - "attentive": Full VJEPA2AttentivePooler (3 self-attn + cross-attn with MLP)
        - "mean": Simple mean pooling over sequence
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        pooling_mode: str = "attentive",  # "attentive", "mean"
        predict_future_temporal_steps: int = 0,  # Number of future temporal positions to predict with the predictor
        prediction_future_frames: int = 0,  # Number of actual future frames used as prediction targets (future_frames mode)
        classify_on_predicted_only: bool = False,  # If True, pooler/classifier only see predictor output, not encoder context
    ) -> None:
        self.pooling_mode = pooling_mode
        assert pooling_mode in ["attentive", "mean"], \
            f"Invalid pooling_mode: {pooling_mode}. Choose 'attentive' or 'mean'."

        self.predict_future_temporal_steps = predict_future_temporal_steps
        self.prediction_future_frames = prediction_future_frames

        if classify_on_predicted_only and predict_future_temporal_steps == 0 and prediction_future_frames == 0:
            raise ValueError(
                "classify_on_predicted_only=True requires either predict_future_temporal_steps > 0 "
                "or prediction_future_frames > 0."
            )
        self.classify_on_predicted_only = classify_on_predicted_only

        super().__init__(model_name=model_name, num_classes=num_classes)

        self._num_classes = num_classes

        # Replace the pooler for mean pooling (must be done after super().__init__)
        if pooling_mode == "mean":
            self.model.pooler = nn.Identity()  # Mean pooling will be handled in forward()
            self.fc_norm = nn.LayerNorm(self.model.config.hidden_size)  # Normalize after mean pooling

        if self.predict_future_temporal_steps > 0:
            logging.info(
                f"Future prediction enabled: {self.predict_future_temporal_steps} temporal steps "
                f"will be predicted by the predictor and concatenated with encoder output before pooling."
            )
        if self.prediction_future_frames > 0:
            logging.info(
                f"Prediction future-frames mode enabled: {self.prediction_future_frames} actual future frames "
                f"will be encoded as prediction targets. Classifier/pooler receives only the context portion."
            )


    @property
    def classifier(self):
        return self.model.classifier

    @property
    def predictor(self):
        """Expose the VJEPA2 predictor for external use (e.g., auxiliary prediction loss)."""
        return self.model.vjepa2.predictor

    def set_trainable_parameters(self, trainable_parameters_configuration="pooler+head"):
        """
        Sets the trainable parameters trainable_parameters_configuration for the VJEPA2ForVideoClassification model.

        Args:
            trainable_parameters_configuration (str): One of:
                - "pooler+head": Pooler + classifier trainable (backbone + predictor frozen)
                - "last_block+pool+head": Last encoder block + LayerNorm + Pooler + classifier trainable (predictor frozen)
                - "last_block+predictor+pool+head": Last encoder block + LayerNorm + Predictor + Pooler + classifier trainable
        """
        # 1. Reset: make everything trainable
        for param in self.model.parameters():
            param.requires_grad = True

        if trainable_parameters_configuration == "pooler+head":
            # Freeze backbone (encoder + predictor)
            for param in self.model.vjepa2.parameters():
                param.requires_grad = False
            # Pooler and classifier remain trainable
            print("trainable_parameters_configuration: 'pooler+head' -> Pooler + classifier trainable.")

        elif trainable_parameters_configuration in ("last_block+pool+head", "last_block+predictor+pool+head"):
            # Freeze entire backbone first
            for param in self.model.vjepa2.parameters():
                param.requires_grad = False

            # Unfreeze last encoder transformer block
            last_block = self.model.vjepa2.encoder.layer[-1]
            for param in last_block.parameters():
                param.requires_grad = True

            # Unfreeze final LayerNorm
            for param in self.model.vjepa2.encoder.layernorm.parameters():
                param.requires_grad = True

            # Unfreeze predictor for 'last_block+predictor+pool+head'
            if trainable_parameters_configuration == "last_block+predictor+pool+head":
                for param in self.model.vjepa2.predictor.parameters():
                    param.requires_grad = True

            # Pooler and classifier remain trainable (already set to True above)
            trainable_desc = "Last encoder block + LayerNorm + Pooler + classifier"
            if trainable_parameters_configuration == "last_block+predictor+pool+head":
                trainable_desc += " + Predictor"
            print(f"trainable_parameters_configuration: '{trainable_parameters_configuration}' -> {trainable_desc} trainable.")

        else:
            raise ValueError(
                f"Unknown trainable_parameters_configuration: {trainable_parameters_configuration}. "
                "Choose 'pooler+head', 'last_block+pool+head', or 'last_block+predictor+pool+head'."
            )

        # Print parameter counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
        print("Trainable parameters have been set.")
        print(f"Trainable parameters configuration: {trainable_parameters_configuration}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")

    def load_pretrained_weights(self, path: str, map_location: str = 'cpu') -> dict:
        """
        Load pretrained weights from `path`, match them to this model's state dict and load only
        the parameters that have matching names and shapes.

        Handles common checkpoint wrappers (a top-level 'state_dict') and 'module.' prefixes.
        Returns a summary dict with counts and a short list of mismatched/missing keys.
        """
        try:
            pretrained = torch.load(path, map_location=map_location)
            if isinstance(pretrained, dict) and 'state_dict' in pretrained:
                pretrained = pretrained['state_dict']

            # Remove 'module.' prefix if present (e.g., saved from DataParallel)
            normalized = {}
            for k, v in pretrained.items():
                nk = k[len('module.'):] if k.startswith('module.') else k
                normalized[nk] = v

            model_dict = self.state_dict()
            matched = {}
            mismatched_keys = []
            missing_keys = []

            for k, v in normalized.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        matched[k] = v
                    else:
                        mismatched_keys.append(
                            f"{k}: pretrained {tuple(v.shape)} vs model {tuple(model_dict[k].shape)}"
                        )
                else:
                    missing_keys.append(k)

            model_dict.update(matched)
            self.load_state_dict(model_dict)

            unmatched_model_keys = [k for k in model_dict.keys() if k not in normalized]

            print(f"Successfully loaded {len(matched)}/{len(normalized)} parameters from pretrained file: {path}")
            if mismatched_keys:
                print(f"Skipped {len(mismatched_keys)} parameters due to shape mismatch:")
                for key in mismatched_keys:
                    print(f"  - {key}")
            if missing_keys:
                print(f"Parameters present in pretrained file but not in current model: {len(missing_keys)}")
                for key in missing_keys:
                    print(f"  - {key}")
            if unmatched_model_keys:
                print(f"Model parameters without corresponding pretrained weights: {len(unmatched_model_keys)}")
                for key in unmatched_model_keys:
                    print(f"  - {key}")

            return {
                "matched": len(matched),
                "total_pretrained": len(normalized),
                "mismatched": len(mismatched_keys),
                "missing": len(missing_keys),
                "mismatched_keys": mismatched_keys,
                "missing_keys": missing_keys,
                "unmatched_model_keys": unmatched_model_keys,
            }
        except Exception as e:
            print(f"Error loading pretrained weights from {path}: {e}")
            return {"error": str(e)}

    def _get_future_patches(self) -> int:
        """
        Compute the number of future patch tokens from self.prediction_future_frames.

        Returns:
            N_future: number of patch tokens corresponding to prediction_future_frames video frames.
        """
        config = self.model.config
        patches_per_step = (config.image_size // config.patch_size) ** 2
        tubelet_size = getattr(config, 'tubelet_size', 2)
        return (self.prediction_future_frames // tubelet_size) * patches_per_step


    def _predict_future(self, encoder_output: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """
        Use the pretrained predictor to predict latent representations for future temporal positions.

        The predictor uses all encoder patches as context and generates learnable mask tokens
        at future positions. RoPE decomposes flat patch indices into (frame, height, width),
        so future positions naturally extend the temporal dimension.

        Args:
            encoder_output: Encoder hidden states [B, N, D] where N = T_enc * S
                (T_enc = num_frames / tubelet_size temporal positions,
                 S = (image_size / patch_size)^2 spatial patches per position)
            num_steps: Number of future temporal steps to predict. Defaults to
                self.predict_future_temporal_steps when None.

        Returns:
            Predicted future representations [B, N_future, D] where
            N_future = num_steps * S
        """
        if num_steps is None:
            num_steps = self.predict_future_temporal_steps
        B, N, D = encoder_output.shape
        config = self.model.config
        patches_per_step = (config.image_size // config.patch_size) ** 2
        N_future = num_steps * patches_per_step

        device = encoder_output.device
        context_mask = [torch.arange(N, device=device).unsqueeze(0).expand(B, -1)]
        target_mask = [torch.arange(N, N + N_future, device=device).unsqueeze(0).expand(B, -1)]

        predictor_output = self.model.vjepa2.predictor(
            encoder_hidden_states=encoder_output,
            context_mask=context_mask,
            target_mask=target_mask,
        )
        return predictor_output.last_hidden_state  # [B, N_future, D]

    def _encode_masked(self, pixel_values: torch.Tensor, visible_patch_indices: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder on a subset of patches with correct RoPE positional information.

        This replicates the encoder forward pass but:
        1. Computes patch embeddings for the full video
        2. Selects only visible patches (drops masked ones)
        3. Passes original flat patch indices as position_mask so RoPE
           decomposes them into correct (frame, height, width) coordinates

        Args:
            pixel_values: Input video tensor [B, T, C, H, W]
            visible_patch_indices: [B, N_visible] original flat indices of visible patches

        Returns:
            Encoder output for visible patches only [B, N_visible, D]
        """
        encoder = self.model.vjepa2.encoder

        # 1. Compute all patch embeddings
        all_embeddings = encoder.embeddings(pixel_values)  # [B, N, D]

        # 2. Select only visible patches
        B, N_visible = visible_patch_indices.shape
        D = all_embeddings.shape[-1]
        gather_idx = visible_patch_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, N_visible, D]
        hidden_states = torch.gather(all_embeddings, dim=1, index=gather_idx)  # [B, N_visible, D]

        # 3. Run through encoder layers with position_mask for correct RoPE
        for layer_module in encoder.layer:
            layer_outputs = layer_module(hidden_states, visible_patch_indices, None, False)
            hidden_states = layer_outputs[0]

        # 4. Final layer norm
        hidden_states = encoder.layernorm(hidden_states)

        return hidden_states  # [B, N_visible, D]

    def _build_sequence_for_classifier(self, context_feats, predicted_future=None):
        """Select what the pooler/classifier sees based on classify_on_predicted_only."""
        if predicted_future is None:
            return context_feats
        if self.classify_on_predicted_only:
            return predicted_future
        return torch.cat([context_feats, predicted_future], dim=1)

    def forward(
        self, pixel_values: torch.Tensor,
        compute_prediction_loss: bool = False,
    ):
        """
        Args:
            pixel_values: [B, T, C, H, W]. For prediction training, T = T_context + T_future.
            compute_prediction_loss: If True, return (cls_logits, pooled, predicted_targets, encoder_targets).

        Returns:
            (classification_logits, pooled_output) — plus prediction loss tensors when requested.
        """
        encoder_output = self.model.vjepa2(pixel_values, skip_predictor=True).last_hidden_state

        prediction_result = None

        if compute_prediction_loss and self.prediction_future_frames > 0:
            N_future = self._get_future_patches()
            B, N_total, D = encoder_output.shape
            N_context = N_total - N_future
            device = encoder_output.device

            ctx_indices = torch.arange(N_context, device=device).unsqueeze(0).expand(B, -1)
            fut_indices = torch.arange(N_context, N_total, device=device).unsqueeze(0).expand(B, -1)

            context_feats = self._encode_masked(pixel_values, ctx_indices)
            predicted_future = self.model.vjepa2.predictor(
                encoder_hidden_states=context_feats,
                context_mask=[ctx_indices],
                target_mask=[fut_indices],
            ).last_hidden_state

            sequence_output = self._build_sequence_for_classifier(context_feats, predicted_future)
            prediction_result = (predicted_future, encoder_output[:, N_context:, :])

        elif self.predict_future_temporal_steps > 0:
            predicted_future = self._predict_future(encoder_output)
            sequence_output = self._build_sequence_for_classifier(encoder_output, predicted_future)

        else:
            sequence_output = encoder_output

        if self.pooling_mode == "attentive":
            pooled_output = self.model.pooler(sequence_output)
        elif self.pooling_mode == "mean":
            pooled_output = self.fc_norm(sequence_output.mean(dim=1))

        cls_logits = self.model.classifier(pooled_output)

        if prediction_result is not None:
            return (cls_logits, pooled_output) + prediction_result
        return (cls_logits, pooled_output)