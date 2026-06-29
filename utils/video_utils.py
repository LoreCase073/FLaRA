"""
Video processing utilities for sliding window evaluation.

Provides frame extraction, sliding window clip construction, and
evaluation transforms.
"""

import traceback
import numpy as np
from typing import List, Optional, Tuple

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from decord import VideoReader, cpu
import cv2


def build_eval_transforms(frame_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Build standard evaluation transforms."""
    return A.Compose([
        A.Resize(frame_size, frame_size, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(frame_size, frame_size),
        A.Normalize(mean=list(mean), std=list(std)),
        ToTensorV2(),
    ])


def extract_sliding_windows(
    video_path: str,
    num_frames: int,
    sampling_fps: float,
    sliding_window_stride: int,
    transform: A.Compose,
) -> Tuple[Optional[List[torch.Tensor]], Optional[float], Optional[int]]:
    """
    Extract all sliding window clips from a video.

    Optimized: reads all unique frames in a single batch call and applies
    transforms once per unique frame, then assembles windows from the
    pre-transformed frames. This avoids redundant decoding and transforms
    for overlapping windows.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames per window (e.g., 16).
        sampling_fps: Target sampling FPS within each window (e.g., 4.0).
        sliding_window_stride: Stride in original video frames between window starts.
        transform: Albumentations transform to apply to each frame.

    Returns:
        (clips, video_fps, total_frames) where:
        - clips: List of tensors, each (num_frames, C, H, W)
        - video_fps: Actual FPS of the video
        - total_frames: Total number of frames in the video
        Returns (None, None, None) on error.
    """
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            print(f"Warning: Video has no frames: {video_path}")
            return None, None, None

        video_fps = float(vr.get_avg_fps())
        if video_fps <= 0:
            print(f"Warning: Invalid FPS for {video_path}, cannot process.")
            return None, None, None

        # Window size in original frames
        window_duration_seconds = num_frames / sampling_fps
        window_size_orig = int(window_duration_seconds * video_fps)

        if window_size_orig > total_frames:
            # Video too short for even one window - try to make one window from all frames
            if total_frames >= num_frames:
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames = vr.get_batch(frame_indices.tolist()).asnumpy()
                processed = [transform(image=f)['image'] for f in frames]
                clip = torch.stack(processed, dim=0)
                del vr
                return [clip], video_fps, total_frames
            else:
                print(f"Warning: Video too short ({total_frames} frames) for window of {num_frames} frames: {video_path}")
                del vr
                return None, None, None

        # --- Compute all window starts and their frame indices ---
        window_starts = list(range(0, total_frames - window_size_orig + 1, sliding_window_stride))
        if len(window_starts) == 0:
            del vr
            return None, None, None

        window_frame_indices = []
        unique_indices_set = set()
        for start in window_starts:
            window_end = start + window_size_orig - 1
            indices = np.linspace(start, window_end, num_frames, dtype=int)
            window_frame_indices.append(indices)
            unique_indices_set.update(indices.tolist())

        # --- Read all unique frames in one batch call ---
        sorted_unique = sorted(unique_indices_set)
        index_to_pos = {idx: pos for pos, idx in enumerate(sorted_unique)}

        all_frames_raw = vr.get_batch(sorted_unique).asnumpy()
        del vr

        # --- Transform each unique frame once ---
        transformed = [transform(image=f)['image'] for f in all_frames_raw]
        del all_frames_raw

        # --- Assemble clips from pre-transformed frames ---
        clips = []
        for indices in window_frame_indices:
            frame_tensors = [transformed[index_to_pos[int(idx)]] for idx in indices]
            clip = torch.stack(frame_tensors, dim=0)  # (num_frames, C, H, W)
            clips.append(clip)

        del transformed
        return clips, video_fps, total_frames

    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ['corrupt', 'decode', 'invalid', 'h264', 'damaged']):
            print(f"Skipping corrupted video {video_path}: {e}")
        else:
            print(f"Error extracting windows from {video_path}: {e}")
            traceback.print_exc()
        return None, None, None