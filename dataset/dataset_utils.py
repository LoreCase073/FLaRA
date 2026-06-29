"""
Utility functions for the dataset classes.

This module contains helper functions for video frame selection, data validation,
label mapping, and other utilities to support the main dataset functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class DataValidator:
    """Handles data validation and preprocessing for the dataset."""

    @staticmethod
    def filter_dataframe_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        """
        Filter dataframe by data split.

        Args:
            df: Input dataframe
            split: Split name ('train', 'val', 'test')

        Returns:
            Filtered dataframe

        Raises:
            ValueError: If split is not valid
        """
        valid_splits = ['train', 'val', 'test']
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")

        return df[df['split'] == split].copy()


class FrameProcessor:
    """Handles video frame processing and extraction logic."""

    @staticmethod
    def compute_frame_indices_around_peak(
        event_timestamp: float,
        duration: float,
        fps: float,
        original_fps: float = 30.0,
        buffer_after_peak: float = 1.0,
        anticipation_offset: Optional[float] = None,
        total_frames: Optional[int] = None,
        prediction_future_frames: int = 0,
    ) -> Tuple[List[int], int]:
        """
        Compute frame indices around the event timestamp.

        Args:
            event_timestamp: Time of the event (e.g., collision) in seconds
            duration: Duration of video segment to extract in seconds
            fps: Target sampling fps
            original_fps: Original video fps
            buffer_after_peak: Buffer time after peak in seconds
            anticipation_offset: If provided, shifts the window backwards by this amount.
                               Window becomes (peak - anticipation_offset - duration, peak - anticipation_offset)
            total_frames: Total frames in video
            prediction_future_frames: If > 0, appends this many additional frames that come
                immediately after the context window (using the same sampling step).
                Used to provide actual future frames as prediction reconstruction targets.

        Returns:
            Tuple of (frame_indices, num_frames_selected)
        """
        if total_frames is None:
            raise ValueError("total_frames must be provided")

        # Calculate end time and frame based on mode
        if anticipation_offset is not None:
            # Anticipation mode: window is (peak - offset - duration, peak - offset)
            end_time = event_timestamp - anticipation_offset
            start_time = end_time - duration
        else:
            # Original mode: window ends buffer_after_peak seconds after the peak
            end_time = event_timestamp + buffer_after_peak
            end_time = max(end_time, duration)  # Ensure we have at least 'duration' seconds
            start_time = end_time - duration

        # Convert to frame indices
        end_frame = int(end_time * original_fps)
        start_frame_ideal = int(start_time * original_fps)

        # Ensure frame indices are within bounds
        end_frame = min(end_frame, total_frames - 1)
        start_frame_ideal = max(0, start_frame_ideal)

        # Calculate sampling step and number of frames
        step = max(1, int(original_fps / fps))  # Ensure step is at least 1
        num_frames = int(duration * fps)

        if anticipation_offset is not None:
            # Anticipation mode: work backwards from end_frame, pad at beginning if needed
            start_frame = max(0, end_frame - num_frames * step)
            indices = list(range(start_frame, end_frame, step))

            # If we don't have enough frames, pad at the BEGINNING with earliest available frame
            while len(indices) < num_frames:
                earliest_frame = indices[0] if indices else 0
                indices.insert(0, max(0, earliest_frame))

            # Ensure we have exactly num_frames
            indices = indices[:num_frames]
        else:
            # Original mode: work backwards from end_frame, pad at end if needed
            start_frame = max(0, end_frame - num_frames * step)
            indices = list(range(start_frame, end_frame, step))

            # If we don't have enough frames, pad with the last available frame
            while len(indices) < num_frames:
                indices.append(min(total_frames - 1, indices[-1] if indices else 0))

            # Ensure we have exactly num_frames
            indices = indices[:num_frames]

        # Append actual future frames (for future-frames mode).
        # Future indices start at end_frame (the first frame not included in context)
        # and continue forward with the same sampling step.
        if prediction_future_frames > 0:
            future_indices = [
                min(end_frame + i * step, total_frames - 1)
                for i in range(prediction_future_frames)
            ]
            indices = indices + future_indices

        return indices, len(indices)


def load_label_mapping(label_mapping_path: str) -> Dict[str, int]:
    """
    Load label mapping from JSON file.

    Args:
        label_mapping_path: Path to the label mapping JSON file

    Returns:
        Dictionary mapping label names to integer indices
    """
    with open(label_mapping_path, 'r') as f:
        mapping = json.load(f)
    d = {k: int(v) for k, v in mapping.items()}
    return d

def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


