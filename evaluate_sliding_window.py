#!/usr/bin/env python3
import os
import argparse
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from utils.utils import set_random_seed, str2bool
from utils.eval_metrics import evaluation
from utils.video_utils import build_eval_transforms, extract_sliding_windows
from dataset.nexar_dataset import NexarDataset
from dataset.dad_dataset import DadDataset
from dataset.dada2000_dataset import Dada2000Dataset
from dataset.dota_dataset import DotaDataset


# ---------------------------------------------------------------------------
# Dataset configurations — sourced from each dataset class's EVAL_CONFIG
# ---------------------------------------------------------------------------
DATASET_CLASSES = {
    'nexar': NexarDataset,
    'dad': DadDataset,
    'dada2000': Dada2000Dataset,
    'dota': DotaDataset,
}

DATASET_CONFIGS = {name: cls.EVAL_CONFIG for name, cls in DATASET_CLASSES.items()}


# ---------------------------------------------------------------------------
# Dataset metadata loaders
# ---------------------------------------------------------------------------
def load_dataset_metadata(dataset_name: str, args) -> Optional[pd.DataFrame]:
    """
    Load dataset metadata (video paths, labels, accident times) without
    creating the full dataset object. Returns a DataFrame with columns:
        video_path, cls_label, time_of_collision, time_to_accident, video_id
    """
    config = DATASET_CONFIGS[dataset_name]

    csv_path = getattr(args, f'{dataset_name}_csv', config['csv_default'])
    data_root = getattr(args, f'{dataset_name}_data_root', config['data_root_default'])

    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Filter to test split
    if 'split' in df.columns:
        df = df[df['split'] == 'test'].copy()
    elif dataset_name in ['dad', 'dada2000', 'dota']:
        # These datasets may not have a split column if csv is already test-only
        pass

    if len(df) == 0:
        print(f"Warning: No test samples found in {csv_path}")
        return None

    # Build video paths and extract metadata
    records = []
    label_col = config['label_col']
    toc_col = config['time_of_collision_col']
    positive_labels = config['positive_labels']
    dataset_cls = DATASET_CLASSES[dataset_name]

    for _, row in df.iterrows():
        # Determine binary label
        raw_label = row[label_col]
        is_positive = raw_label in positive_labels or str(raw_label) in [str(x) for x in positive_labels]
        cls_label = 1 if is_positive else 0

        # Time of collision
        toc = row.get(toc_col, None)
        if pd.isna(toc):
            toc = None
        else:
            toc = float(toc)

        # Time to accident (Nexar-specific: gap between video end and actual collision)
        tta_gap = row.get('time_to_accident', None)
        if pd.isna(tta_gap):
            tta_gap = None
        else:
            tta_gap = float(tta_gap)

        # Build video path and ID (dataset-specific)
        video_path = dataset_cls.build_eval_video_path(row, data_root)
        if video_path is None:
            continue

        vid = dataset_cls.get_eval_video_id(row)

        records.append({
            'video_path': str(video_path),
            'cls_label': cls_label,
            'time_of_collision': toc,
            'time_to_accident': tta_gap,
            'video_id': vid,
        })

    return pd.DataFrame(records)



# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def load_model(args, device):
    """Load model and pretrained weights."""
    from utils.train_utils import ModelManager
    model = ModelManager.create_model(args)
    model.to(device)

    # Load weights
    if hasattr(model, 'load_pretrained_weights'):
        load_result = model.load_pretrained_weights(args.model_path, map_location=device)
        if isinstance(load_result, dict) and load_result.get('error') is not None:
            print(f"load_pretrained_weights failed: {load_result['error']}, falling back to ModelManager.load_model")
            success = ModelManager.load_model(model, args.model_path, device)
            if not success:
                raise RuntimeError(f"Failed to load model from {args.model_path}")
    else:
        success = ModelManager.load_model(model, args.model_path, device)
        if not success:
            raise RuntimeError(f"Failed to load model from {args.model_path}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model loaded from {args.model_path}")
    return model


def run_model_forward_batch(model, clips_batch: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Run a batch of clips through the model and return collision probabilities.

    Args:
        model: The loaded model.
        clips_batch: Tensor of shape (B, T, C, H, W).
        num_classes: Number of output classes.

    Returns:
        numpy array of shape (B,) with collision probabilities in [0, 1].
    """
    B = clips_batch.shape[0]

    outputs_cls, _ = model(clips_batch)

    if outputs_cls is None:
        return np.zeros(B, dtype=np.float64)

    pred_probs = F.softmax(outputs_cls, dim=1).cpu().numpy()  # (B, num_classes)

    # Collision probability: sum of all non-safe classes (class 0 = safe)
    if num_classes >= 3:
        collision_probs = np.sum(pred_probs[:, 1:], axis=1)
    elif num_classes == 2:
        collision_probs = pred_probs[:, 1]
    else:
        collision_probs = pred_probs[:, 0]

    return collision_probs.astype(np.float64)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
class SlidingWindowEvaluator:
    """Evaluates models using sliding window inference with AP/AUC/mTTA/TTA@R80 metrics."""

    def __init__(self, args):
        self.args = args

        # Device
        if args.gpu_id == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{args.gpu_id}')

        set_random_seed(args.seed)
        os.makedirs(args.output_dir, exist_ok=True)

        # Load model
        self.model = load_model(args, self.device)

        # Build transforms
        self.transform = build_eval_transforms(frame_size=args.frame_size)

    def evaluate_dataset(self, dataset_name: str) -> Dict:
        """Run sliding window evaluation on a single dataset."""
        config = DATASET_CONFIGS[dataset_name]
        print(f"\n{'=' * 60}")
        print(f"EVALUATING {config['display_name'].upper()} - SLIDING WINDOW")
        print(f"{'=' * 60}")

        # Load metadata
        metadata_df = load_dataset_metadata(dataset_name, self.args)
        if metadata_df is None or len(metadata_df) == 0:
            print(f"No data found for {dataset_name}, skipping.")
            return {'error': 'No data found'}

        print(f"Found {len(metadata_df)} test videos")
        print(f"  Positive: {(metadata_df['cls_label'] == 1).sum()}")
        print(f"  Negative: {(metadata_df['cls_label'] == 0).sum()}")
        print(f"  Sliding window stride: {self.args.sliding_window_stride} frames")
        print(f"  Window: {self.args.num_frames} frames at {self.args.fps} FPS = {self.args.num_frames / self.args.fps:.1f}s")

        # Run inference on all videos
        all_video_preds = []     # List of 1D arrays (variable length per video)
        all_labels = []
        all_num_valid = []       # Number of valid windows (end before collision)
        all_vid_dur_seconds = [] # Actual video file duration in seconds
        all_toc_seconds = []    # Time of collision in seconds
        all_stride_seconds = [] # Per-video stride in seconds
        all_win_dur_seconds = []# Per-video window duration in seconds
        all_video_ids = []
        skipped = 0

        # Window duration is constant in target-FPS space (for clip-based models)
        default_window_duration_sec = self.args.num_frames / self.args.fps

        print("\nRunning sliding window inference...")
        with torch.no_grad():
            for idx in tqdm(range(len(metadata_df)), desc=f"  {config['display_name']}"):
                row = metadata_df.iloc[idx]
                video_path = row['video_path']
                cls_label = row['cls_label']
                toc = row['time_of_collision']
                tta_gap = row.get('time_to_accident', None)
                if pd.isna(tta_gap):
                    tta_gap = None
                video_id = row['video_id']

                if not os.path.exists(video_path):
                    skipped += 1
                    continue

                # Per-window clip inference
                clips, video_fps, total_frames = extract_sliding_windows(
                    video_path=video_path,
                    num_frames=self.args.num_frames,
                    sampling_fps=self.args.fps,
                    sliding_window_stride=self.args.sliding_window_stride,
                    transform=self.transform,
                )

                if clips is None or len(clips) == 0:
                    skipped += 1
                    continue

                num_windows = len(clips)
                video_duration = (total_frames - 1) / video_fps
                stride_sec = self.args.sliding_window_stride / video_fps
                window_duration_sec = default_window_duration_sec

                # Run clips through the model in batches
                window_preds = np.zeros(num_windows, dtype=np.float64)

                batch_size = self.args.inference_batch_size
                for batch_start in range(0, num_windows, batch_size):
                    batch_end = min(batch_start + batch_size, num_windows)
                    batch_clips = torch.stack(
                        clips[batch_start:batch_end], dim=0
                    ).to(self.device)  # (B, T, C, H, W)

                    probs = run_model_forward_batch(
                        self.model, batch_clips, self.args.num_classes
                    )
                    window_preds[batch_start:batch_end] = probs

                # Free clip tensors from CPU memory
                del clips

                # Determine toc_seconds and num_valid_windows.
                # A window is valid if its END is before the collision:
                #   window_end_time = w * stride_sec + window_duration_sec <= toc_sec
                #   => w <= (toc_sec - window_duration_sec) / stride_sec

                if cls_label == 1 and config['video_has_accident'] and toc is not None:
                    # DAD/DADA/DoTA: collision at toc seconds within the video
                    toc_sec = float(toc)
                    max_valid_w = (toc_sec - window_duration_sec) / stride_sec
                    n_valid = min(int(max_valid_w) + 1, num_windows)
                    n_valid = max(n_valid, 0)
                elif cls_label == 1 and not config['video_has_accident'] and tta_gap is not None:
                    # Nexar positive: video ends before collision
                    toc_sec = video_duration + float(tta_gap)
                    # All windows end within the video, which is before the collision
                    n_valid = num_windows
                    # For rmTTA, the conceptual "full event" extends to toc
                    video_duration = toc_sec
                else:
                    # Negative video
                    neg_cutoff = config.get('negative_temporal_cutoff', None)
                    if neg_cutoff is not None and cls_label == 0:
                        # DADA-2000/DoTA: negative samples share the same video as
                        # positives, so restrict evaluation to the first neg_cutoff
                        # seconds.
                        toc_sec = min(neg_cutoff, video_duration)
                        max_valid_w = (toc_sec - window_duration_sec) / stride_sec
                        n_valid = min(int(max_valid_w) + 1, num_windows)
                        n_valid = max(n_valid, 0)
                    else:
                        # DAD/Nexar negatives: separate videos, use full duration
                        toc_sec = video_duration
                        n_valid = num_windows

                all_video_preds.append(window_preds)
                all_labels.append(cls_label)
                all_num_valid.append(n_valid)
                all_toc_seconds.append(toc_sec)
                all_stride_seconds.append(stride_sec)
                all_win_dur_seconds.append(window_duration_sec)
                all_vid_dur_seconds.append(video_duration)
                all_video_ids.append(video_id)

        if skipped > 0:
            print(f"\nSkipped {skipped} videos (missing/corrupted)")

        if len(all_video_preds) == 0:
            print("No valid videos processed!")
            return {'error': 'No valid videos'}

        # Build (N x T_max) prediction matrix with 0-padding
        T_max = max(len(p) for p in all_video_preds)
        N = len(all_video_preds)
        all_pred = np.zeros((N, T_max), dtype=np.float64)
        for i, preds in enumerate(all_video_preds):
            all_pred[i, :len(preds)] = preds

        all_labels = np.array(all_labels, dtype=np.float64)
        all_num_valid = np.array(all_num_valid, dtype=np.int64)
        all_toc_seconds = np.array(all_toc_seconds, dtype=np.float64)
        all_stride_seconds = np.array(all_stride_seconds, dtype=np.float64)
        all_win_dur_seconds = np.array(all_win_dur_seconds, dtype=np.float64)
        all_vid_dur_seconds = np.array(all_vid_dur_seconds, dtype=np.float64)

        print(f"\nPrediction matrix shape: ({N}, {T_max})")
        print(f"Positive videos: {int(all_labels.sum())}, Negative: {int((all_labels == 0).sum())}")

        # Run evaluation
        print(f"\n--- {config['display_name']} Results ---")
        metrics = evaluation(
            all_pred, all_labels, all_num_valid,
            all_toc_seconds, all_stride_seconds, all_win_dur_seconds,
            all_vid_dur_seconds,
        )

        # Build full results
        results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_path': self.args.model_path,
                'model_type': 'vjepa2',
                'dataset_name': dataset_name,
                'num_videos': N,
                'num_positive': int(all_labels.sum()),
                'num_negative': int((all_labels == 0).sum()),
                'skipped_videos': skipped,
                'max_windows': int(T_max),
                'sliding_window_stride': self.args.sliding_window_stride,
                'num_frames': self.args.num_frames,
                'sampling_fps': self.args.fps,
                'frame_size': self.args.frame_size,
            },
            'metrics': metrics,
        }

        return results

    def evaluate(self) -> Dict:
        """Run evaluation on all specified datasets."""
        all_results = {}

        for dataset_name in self.args.datasets:
            if dataset_name not in DATASET_CONFIGS:
                print(f"Warning: Unknown dataset '{dataset_name}', skipping.")
                continue

            try:
                results = self.evaluate_dataset(dataset_name)
                all_results[dataset_name] = results
                self.save_results(results, dataset_name)
            except Exception as e:
                print(f"EVALUATION FAILED for {dataset_name}: {e}")
                traceback.print_exc()
                all_results[dataset_name] = {'error': str(e)}

        return all_results

    def save_results(self, results: Dict, dataset_name: str):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            self.args.output_dir,
            f"sliding_window_{dataset_name}_{timestamp}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sliding Window Evaluation for AP, AUC, mTTA, TTA@R80",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument("--datasets", type=str, nargs='+', default=['nexar'],
                        choices=['nexar', 'dad', 'dada2000', 'dota'],
                        help="Datasets to evaluate on")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model_name", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2",
                        help="Pretrained model name")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes")

    # Model-specific args needed by ModelManager.create_model
    parser.add_argument("--trainable_parameters_configuration", type=str,
                        default="last_block+predictor+pool+head",
                        help="Trainable parameters configuration (required by ModelManager.create_model)")
    parser.add_argument("--pooling_mode", type=str, default="attentive")
    parser.add_argument("--predict_future_temporal_steps", type=int, default=0)
    parser.add_argument("--prediction_future_frames", type=int, default=0)
    parser.add_argument("--classify_on_predicted_only", type=str2bool, default=False)

    # Video processing
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per sliding window")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Sampling FPS within each window")
    parser.add_argument("--frame_size", type=int, default=256,
                        help="Frame size (H=W)")

    # Sliding window
    parser.add_argument("--sliding_window_stride", type=int, default=8,
                        help="Stride between windows in original video frames")

    # Inference
    parser.add_argument("--inference_batch_size", type=int, default=16,
                        help="Number of clips to batch on GPU (per video).")

    # Dataset paths (override defaults)
    parser.add_argument("--nexar_csv", type=str, default=DATASET_CONFIGS['nexar']['csv_default'])
    parser.add_argument("--nexar_data_root", type=str, default=DATASET_CONFIGS['nexar']['data_root_default'])
    parser.add_argument("--dad_csv", type=str, default=DATASET_CONFIGS['dad']['csv_default'])
    parser.add_argument("--dad_data_root", type=str, default=DATASET_CONFIGS['dad']['data_root_default'])
    parser.add_argument("--dada2000_csv", type=str, default=DATASET_CONFIGS['dada2000']['csv_default'])
    parser.add_argument("--dada2000_data_root", type=str, default=DATASET_CONFIGS['dada2000']['data_root_default'])
    parser.add_argument("--dota_csv", type=str, default=DATASET_CONFIGS['dota']['csv_default'])
    parser.add_argument("--dota_data_root", type=str, default=DATASET_CONFIGS['dota']['data_root_default'])

    # System
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU device ID or 'cpu'")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="sliding_window_results",
                        help="Directory to save results")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("SLIDING WINDOW EVALUATION - AP / AUC / mTTA / TTA@R80")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Model type: vjepa2")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Frame size: {args.frame_size}x{args.frame_size}")
    if args.predict_future_temporal_steps > 0:
        print(f"Predicting future {args.predict_future_temporal_steps} temporal steps")
    print(f"Window: {args.num_frames} frames at {args.fps} FPS = {args.num_frames / args.fps:.1f}s")
    print(f"Stride: {args.sliding_window_stride} frames")
    print(f"Num classes: {args.num_classes}")
    print(f"Device: {'cuda:' + args.gpu_id if torch.cuda.is_available() and args.gpu_id != 'cpu' else 'cpu'}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    try:
        evaluator = SlidingWindowEvaluator(args)
        all_results = evaluator.evaluate()

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        for dataset_name, results in all_results.items():
            if 'error' in results:
                print(f"  {dataset_name}: FAILED - {results['error']}")
            elif 'metrics' in results:
                m = results['metrics']
                print(f"  {DATASET_CONFIGS[dataset_name]['display_name']}:")
                print(f"    AP={m['AP']:.4f}  AUC={m['AUC']:.4f}  mTTA={m['mTTA']:.2f}s  TTA@R80={m['TTA_R80']:.2f}s")
                print(f"    rmTTA={m['rmTTA']:.4f}  rTTA@R80={m['rTTA_R80']:.4f}")

        return 0

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
