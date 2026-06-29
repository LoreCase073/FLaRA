import os
import math
from typing import Dict, Optional, Tuple
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import numpy as np

from dataset.nexar_dataset import NexarDataset
from models.model import build_vjepa2

import time




class WarmupCosineAnnealingLR:
    """
    Learning rate scheduler with warmup phase followed by cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup phase (typically 20% of total epochs)
        total_epochs: Total number of training epochs
        warmup_start_lr: Starting learning rate for warmup (e.g., 1e-6)
        base_lr: Target learning rate after warmup (same as optimizer's initial lr)
        eta_min: Minimum learning rate for cosine annealing phase
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=1e-6, base_lr=None, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr if base_lr is not None else optimizer.param_groups[0]['lr']
        self.eta_min = eta_min
        self.current_epoch = 0

        # Store each group's target peak LR before resetting to warmup_start_lr.
        # This supports differential LR: groups can have different 'lr' values set
        # by the caller (e.g. head at base_lr, backbone at base_lr * scale).
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
            param_group['lr'] = self.warmup_start_lr

    def step(self):
        """Update learning rate based on current epoch, respecting per-group peak LRs."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase from warmup_start_lr to each group's initial_lr
            factor = 1.0 if self.warmup_epochs == 1 else self.current_epoch / (self.warmup_epochs - 1)
            for param_group in self.optimizer.param_groups:
                initial = param_group['initial_lr']
                param_group['lr'] = self.warmup_start_lr + (initial - self.warmup_start_lr) * factor
        else:
            # Cosine annealing phase: decay from initial_lr to eta_min (scaled per group)
            cosine_epochs = self.total_epochs - self.warmup_epochs
            cosine_epoch = self.current_epoch - self.warmup_epochs
            cosine_factor = (1 + math.cos(math.pi * cosine_epoch / cosine_epochs)) / 2
            for param_group in self.optimizer.param_groups:
                initial = param_group['initial_lr']
                # Scale eta_min proportionally so all groups decay by the same relative amount
                eta = self.eta_min * (initial / self.base_lr)
                param_group['lr'] = eta + (initial - eta) * cosine_factor

        self.current_epoch += 1

    def get_last_lr(self):
        """Get current learning rate (for logging)."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class ModelManager:
    """Handles model creation, loading, and saving operations."""

    @staticmethod
    def create_model(args) -> torch.nn.Module:
        """
        Create and initialize the model based on arguments.

        Args:
            args: Training arguments

        Returns:
            Initialized model
        """
        print("Initializing VJEPA2 model...")
        model = build_vjepa2(
            num_classes=args.num_classes,
            model_name=args.model_name,
            trainable_parameters_configuration=args.trainable_parameters_configuration,
            pooling_mode=args.pooling_mode,
            predict_future_temporal_steps=args.predict_future_temporal_steps,
            prediction_future_frames=args.prediction_future_frames,
            classify_on_predicted_only=args.classify_on_predicted_only,
        )
        print(f"VJEPA2 model initialized with {args.num_classes} classes from {args.model_name}")
        print("")

        return model

    @staticmethod
    def save_model(model: torch.nn.Module, save_path: str) -> None:
        """
        Save model state dict to file.

        Args:
            model: Model to save
            save_path: Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    @staticmethod
    def load_model(model: torch.nn.Module, load_path: str, device: torch.device) -> bool:
        """
        Load model state dict from file.

        Args:
            model: Model to load into
            load_path: Path to load from
            device: Device to load on

        Returns:
            True if loaded successfully, False otherwise
        """
        if os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=device))
            return True
        else:
            print(f"WARNING: Model file not found: {load_path}")
            return False


class DatasetManager:
    """Handles dataset creation and data loader setup."""

    @staticmethod
    def create_datasets(args) -> Tuple["NexarDataset", "NexarDataset"]:
        """
        Create train and validation datasets.

        Args:
            args: Training arguments

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        anticipation_offset_range = (args.anticipation_offset_min, args.anticipation_offset_max)

        train_dataset = NexarDataset(
            csv_path=args.train_csv,
            data_root=args.data_root,
            label_mapping_path=args.label_mapping_path,
            mean=mean,
            std=std,
            duration=args.duration,
            num_frames=args.num_frames,
            fps=args.fps,
            frame_size=args.frame_size,
            split='train',
            anticipation_offset_range=anticipation_offset_range,
            use_fixed_fps=args.use_fixed_fps,
            use_time_of_alert_offset=args.use_time_of_alert_offset,
            seed=args.seed,
            prediction_future_frames=args.prediction_future_frames,
        )

        # Handle validation split
        if not args.val_csv or args.val_csv == "":
            args.val_csv = args.test_csv
            print("No validation CSV provided, since in Nexar there is no validation split, using test CSV for validation.")
            val_split = 'test'
        else:
            print("Validation CSV provided, but note that in Nexar there is no validation split, a custom split was provided.")
            val_split = 'val'

        val_dataset = NexarDataset(
            csv_path=args.val_csv,
            data_root=args.data_root,
            label_mapping_path=args.label_mapping_path,
            mean=mean,
            std=std,
            duration=args.duration,
            num_frames=args.num_frames,
            fps=args.fps,
            frame_size=args.frame_size,
            split=val_split,
            anticipation_offset_range=anticipation_offset_range,
            use_fixed_fps=args.use_fixed_fps,
            seed=args.seed,
        )

        print("Dataset sizes:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")

        return train_dataset, val_dataset

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """
        Initialize worker processes with deterministic seeds for reproducibility.
        Seeds Python random, numpy, and scipy random states for each worker.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # Seed scipy's random state (used by scipy.stats distributions like rv_histogram.rvs())
        try:
            import scipy
            scipy.random.seed(worker_seed)
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def create_data_loaders(train_dataset: "NexarDataset", val_dataset: "NexarDataset",
                          batch_size: int, num_workers: int,
                          seed: int = 42) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for train and validation datasets with reproducibility support.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for all loaders
            num_workers: Number of worker processes
            seed: Random seed for reproducible shuffling (default: 42)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        g = torch.Generator()
        g.manual_seed(seed)

        print(f"Creating DataLoaders with batch size {batch_size} and {num_workers} workers (seed={seed})")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=NexarDataset.collate_fn,
            worker_init_fn=DatasetManager._worker_init_fn,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=NexarDataset.collate_fn,
            worker_init_fn=DatasetManager._worker_init_fn,
            generator=g,
        )

        return train_loader, val_loader


class MetricsCalculator:
    """Handles computation of various training and evaluation metrics."""

    @staticmethod
    def compute_classification_metrics(predictions_probs: torch.Tensor,
                                     true_labels: torch.Tensor,
                                     num_classes: int) -> Dict[str, float]:
        """
        Compute classification metrics including accuracy and average precision.

        Args:
            predictions_probs: Predicted probabilities (N, num_classes)
            true_labels: True labels (N,)
            num_classes: Number of classes

        Returns:
            Dictionary containing computed metrics
        """
        metrics = {}

        # Compute accuracy
        pred_labels = predictions_probs.argmax(axis=1)
        accuracy = (pred_labels == true_labels).mean()
        metrics['accuracy'] = accuracy

        # Compute Average Precision
        if num_classes == 2:
            # Binary classification
            avg_prec = average_precision_score(
                true_labels,
                predictions_probs[:, 1]  # Probability of positive class
            )
            metrics['ap'] = avg_prec

            # Per-class AP for binary
            for i in range(num_classes):
                ap_i = average_precision_score(
                    (true_labels == i).astype(int),
                    predictions_probs[:, i]
                )
                
                metrics[f'ap_class_{i}'] = ap_i
        else:
            # Multiclass classification
            true_labels_onehot = label_binarize(
                true_labels,
                classes=list(range(num_classes))
            )

            # Compute macro-averaged AP
            avg_prec = average_precision_score(
                true_labels_onehot,
                predictions_probs,
                average='macro'
            )
            metrics['ap_macro'] = avg_prec

            # Compute per-class AP
            for i in range(num_classes):
                ap_i = average_precision_score(
                    true_labels_onehot[:, i],
                    predictions_probs[:, i]
                )
                metrics[f'ap_class_{i}'] = ap_i

        return metrics

class TrainingEvaluator:
    """Handles training and evaluation loops."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 criterion_cls: torch.nn.Module,
                 lambda_cls: float,
                 lambda_prediction: float = 0.0,
                 gradient_accumulation_steps: int = 1,
                 num_frames: int = 16):
        """
        Initialize the training evaluator.

        Args:
            model: The model to train/evaluate
            device: Device to run on
            criterion_cls: Classification loss function
            lambda_cls: Weight for classification loss
            lambda_prediction: Weight for auxiliary prediction loss (predictor regularization)
            gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
            num_frames: Number of input video frames (for spatiotemporal info)
        """
        self.model = model
        self.device = device
        self.criterion_cls = criterion_cls
        self.lambda_cls = lambda_cls
        self.lambda_prediction = lambda_prediction
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_frames = num_frames

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_prediction_loss = 0.0

        need_prediction_loss = self.lambda_prediction > 0
        accum_steps = self.gradient_accumulation_steps

        num_batches = len(train_loader)
        start_time = time.time()

        optimizer.zero_grad()

        for i, (video_frames, labels_cls) in enumerate(train_loader):
            # ----- PROGRESS + ETA every 10 batches -----
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / i
                remaining_batches = num_batches - i
                eta_seconds = remaining_batches * avg_time_per_batch

                # Convert ETA to min:sec format
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)

                pct = (i / num_batches) * 100
                print(f"[{i}/{num_batches}] {pct:.1f}% done | ETA: {eta_min}m {eta_sec}s")

            inputs = video_frames.to(self.device)
            labels_cls = labels_cls.to(self.device)

            # Forward pass
            predicted_targets = None
            encoder_targets = None
            result = self.model(
                inputs,
                compute_prediction_loss=need_prediction_loss,
            )
            # Unpack: base (2) + optional prediction (2)
            outputs_cls = result[0]
            idx = 2
            if need_prediction_loss:
                predicted_targets, encoder_targets = result[idx], result[idx + 1]
                idx += 2

            # Compute loss
            loss_cls = self._compute_cls_loss(outputs_cls, labels_cls)
            loss = self.lambda_cls * loss_cls

            # Compute auxiliary prediction loss: stop-gradient on encoder targets
            prediction_loss = torch.tensor(0.0, device=self.device)
            if need_prediction_loss and predicted_targets is not None:
                prediction_loss = F.smooth_l1_loss(predicted_targets, encoder_targets.detach())
                prediction_loss = prediction_loss * self.lambda_prediction
                loss += prediction_loss

            # Scale loss for gradient accumulation
            loss = loss / accum_steps

            # Backward pass
            loss.backward()

            # Optimizer step every accum_steps or at end of epoch
            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate losses (unscaled for logging)
            total_loss += loss.item() * accum_steps
            total_cls_loss += loss_cls.item() if self.model.classifier is not None else 0.0
            if need_prediction_loss:
                total_prediction_loss += prediction_loss.item()

        # Compute averages
        metrics = {
            'loss': total_loss / len(train_loader),
            'cls_loss': total_cls_loss / len(train_loader) if self.model.classifier is not None else 0.0,
            'prediction_loss': total_prediction_loss / len(train_loader) if self.lambda_prediction > 0 else 0.0,
        }

        return metrics

    def evaluate(self, data_loader: DataLoader, num_classes: int) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: Data loader for evaluation
            num_classes: Number of classes

        Returns:
            Tuple of (metrics_dict, cls_predictions_probs, cls_true_labels)
        """
        self.model.eval()
        total_loss = 0.0

        predictions_cls = []
        true_labels_cls = []

        start_time = time.time()
        num_batches = len(data_loader)

        with torch.no_grad():
            for i, (video_frames, labels_cls) in enumerate(data_loader):
                # Skip empty batches (all samples were corrupted)
                if video_frames.numel() == 0:
                    print(f"WARNING: Skipping empty batch {i} (all samples corrupted)")
                    continue

                # ----- PROGRESS + ETA every 10 batches -----
                if i % 10 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    avg_time_per_batch = elapsed / i
                    remaining_batches = num_batches - i
                    eta_seconds = remaining_batches * avg_time_per_batch

                    # Convert ETA to min:sec format
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)

                    pct = (i / num_batches) * 100
                    print(f"[{i}/{num_batches}] {pct:.1f}% done | ETA: {eta_min}m {eta_sec}s")

                inputs = video_frames.to(self.device)
                labels_cls = labels_cls.to(self.device)

                # Forward pass
                outputs_cls, _ = self.model(inputs)

                # Compute loss
                loss_cls = self._compute_cls_loss(outputs_cls, labels_cls)
                loss = self.lambda_cls * loss_cls

                # Collect predictions and labels
                if self.model.classifier is not None:
                    predictions_cls.append(outputs_cls.detach().cpu())
                    true_labels_cls.append(labels_cls.detach().cpu())

                total_loss += loss.item()

        # Process collected predictions
        metrics = {'loss': total_loss / len(data_loader)}

        cls_predictions_probs = None
        cls_true_labels = None

        if self.model.classifier is not None and len(predictions_cls) > 0:
            predictions_cls = torch.cat(predictions_cls).numpy()
            true_labels_cls = torch.cat(true_labels_cls).numpy()

            # Convert to probabilities
            cls_predictions_probs = F.softmax(torch.tensor(predictions_cls), dim=1).numpy()
            cls_true_labels = true_labels_cls

            # Compute classification metrics
            cls_metrics = MetricsCalculator.compute_classification_metrics(
                cls_predictions_probs, true_labels_cls, num_classes
            )
            metrics.update({f'cls_{k}': v for k, v in cls_metrics.items()})

        return metrics, cls_predictions_probs, cls_true_labels

    def _compute_cls_loss(self, outputs_cls: torch.Tensor, labels_cls: torch.Tensor) -> torch.Tensor:
        """
        Compute the classification loss.

        Args:
            outputs_cls: Classification outputs
            labels_cls: Classification labels

        Returns:
            Classification loss
        """
        if self.model.classifier is not None and labels_cls is not None:
            return self.criterion_cls(outputs_cls, labels_cls)
        return torch.tensor(0.0).to(self.device)

class ExperimentManager:
    """Handles experiment setup, logging, and results saving."""

    @staticmethod
    def save_test_results(output_dir: str, num_classes: int,
                         test_predictions_cls_probs: torch.Tensor = None,
                         test_true_labels_cls: torch.Tensor = None,
                         test_metrics: Dict[str, float] = None,
                         epoch: str = 'final') -> None:
        """
        Save test results including predictions and metrics.

        Args:
            output_dir: Directory to save results
            num_classes: Number of classes
            test_predictions_cls_probs: Classification prediction probabilities
            test_true_labels_cls: True classification labels
            test_metrics: Test metrics dictionary
            epoch: Epoch identifier for saving
        """
        from utils.utils import save_metrics_to_csv

        os.makedirs(output_dir, exist_ok=True)

        # Save classification predictions
        if test_predictions_cls_probs is not None and test_true_labels_cls is not None:
            cls_csv_path = os.path.join(output_dir, 'test_predictions_classification.csv')
            ExperimentManager._save_classification_predictions(
                cls_csv_path, test_predictions_cls_probs, test_true_labels_cls, num_classes
            )

        # Save test metrics
        if test_metrics is not None:
            metrics_csv_path = os.path.join(output_dir, 'test_metrics.csv')
            save_metrics_to_csv(metrics_csv_path, epoch=epoch, metrics_dict=test_metrics)
            print(f"Test metrics saved to {metrics_csv_path}")

    @staticmethod
    def _save_classification_predictions(file_path: str, predictions_probs: torch.Tensor,
                                       true_labels: torch.Tensor, num_classes: int) -> None:
        """Save classification predictions to CSV."""
        with open(file_path, 'w') as f:
            header = [f'pred_prob_class_{i}' for i in range(num_classes)]
            header += ['true_class']
            f.write(','.join(header) + '\n')

            num_samples = predictions_probs.shape[0]
            for i in range(num_samples):
                row = [str(predictions_probs[i, j]) for j in range(num_classes)]
                row += [str(true_labels[i])]
                f.write(','.join(row) + '\n')
        print(f"Test classification predictions saved to {file_path}")

    @staticmethod
    def setup_scheduler(optimizer: torch.optim.Optimizer, lr_scheduler: str,
                       learning_rate: float, epochs: int) -> Optional[WarmupCosineAnnealingLR]:
        """
        Setup learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule
            lr_scheduler: Scheduler type
            learning_rate: Base learning rate
            epochs: Total epochs

        Returns:
            Scheduler instance or None
        """
        if lr_scheduler == "cosine":
            # Calculate warmup epochs (20% of total epochs)
            warmup_epochs = max(1, int(0.2 * epochs))
            print(f"Using Warmup + Cosine Annealing LR scheduler.")
            print(f"Warmup phase: {warmup_epochs} epochs (1e-6 -> {learning_rate})")
            print(f"Cosine annealing: {epochs - warmup_epochs} epochs ({learning_rate} -> {learning_rate * 1e-2})")

            scheduler = WarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs,
                warmup_start_lr=1e-6,
                base_lr=learning_rate,
                eta_min=learning_rate * 1e-2
            )
            return scheduler
        elif lr_scheduler == "constant":
            print(f"Using constant learning rate: {learning_rate}")
            print(f"Learning rate will remain fixed at {learning_rate} throughout all {epochs} epochs")
            return None
        else:
            print(f"WARNING: Unknown scheduler '{lr_scheduler}'. Using constant learning rate: {learning_rate}")
            return None
        


    