import torch
import numpy as np
import random
from decord import VideoReader, cpu
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
from typing import Optional, Tuple

from .dataset_utils import (
    DataValidator, FrameProcessor,
    load_label_mapping, ensure_directory_exists
)



class NexarDataset(torch.utils.data.Dataset):
    """
    Nexar video dataset for accident anticipation.

    This dataset loads video frames,
    applies transformations, and provides data for training/validation/testing.
    """

    EVAL_CONFIG = {
        'csv_default': 'data/nexar_all_merged_metadata.csv',
        'data_root_default': '',
        'label_mapping_default': 'dataset/nexar_label_mapping.json',
        'display_name': 'Nexar',
        'default_fps': 30,
        'time_of_collision_col': 'time_of_event',
        'label_col': 'label',
        'positive_labels': ['crash'],
        'video_has_accident': False,
    }

    @staticmethod
    def build_eval_video_path(row, data_root):
        data_root = Path(data_root)
        file_name = row['file_name']
        vid = file_name.replace('.mp4', '')
        label = row['label']
        test_split = row.get('test_split', 'public')
        label_dir = 'positive' if label == 'crash' else 'negative'
        return data_root / f"test-{test_split}" / label_dir / f"{vid}.mp4"

    @staticmethod
    def get_eval_video_id(row):
        return row['file_name'].replace('.mp4', '')

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        label_mapping_path: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        num_frames: int = 16,
        fps: float = 4.0,
        duration: float = 4.0,
        frame_size: int = 224,
        resampling: str = 'bilinear',
        split: str = 'train',
        anticipation_offset_range: Optional[Tuple[float, float]] = None,
        num_classes: int = 2,
        use_fixed_fps: bool = False,
        use_time_of_alert_offset: bool = False,
        seed: int = 42,
        prediction_future_frames: int = 0,
    ):
        """
        Initialize the NexarDataset.

        Args:
            csv_path: Path to the CSV file containing dataset metadata
            data_root: Root directory for caching videos
            label_mapping_path: Path to JSON file mapping labels to indices
            mean: Mean values for image normalization
            std: Standard deviation values for image normalization
            num_frames: Number of frames to extract per video
            fps: Target sampling FPS for video frames (e.g., 4.0 means sample at 4 FPS)
            frame_size: Target frame size (height and width)
            resampling: Resampling method ('bilinear' or 'bicubic')
            split: Data split ('train', 'val', or 'test')
            anticipation_offset_range: Tuple of (min, max) offset range for anticipation mode.
                                     If provided, enables anticipation mode where offset is randomly
                                     sampled from [min, max] for each video. Should be in range [0, 1].
            num_classes: Number of classification classes
            use_fixed_fps: If True, use DEFAULT_FPS instead of extracting FPS from each video
            seed: Random seed for reproducibility
            prediction_future_frames: Number of additional future frames to load after the context window.
                These frames are appended to the context frames (same sampling step) and used as
                prediction reconstruction targets. The returned video tensor will have shape
                [num_frames + prediction_future_frames, C, H, W]. Default 0 (disabled).
        """
        # Store initialization parameters
        self.csv_path = csv_path
        self.data_root = data_root
        self.num_frames = num_frames
        self.fps = fps
        self.frame_size = frame_size
        self.resampling = resampling
        self.split = split
        self.anticipation_offset_range = anticipation_offset_range
        self.duration = duration
        self.num_classes = num_classes

        self.seed = seed # For reproducibility of albumentationsx transforms

        self.use_fixed_fps = use_fixed_fps
        self.use_time_of_alert_offset = use_time_of_alert_offset
        self.prediction_future_frames = prediction_future_frames

        self.DEFAULT_CONFIG = {
            'DEFAULT_FPS': 30,
            'INTERPOLATION_MAPPING': {'bilinear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC},
        }


        # Setup video cache directory
        self.video_cache_dir = ensure_directory_exists(self.data_root)

        self.sampling_fps = int(self.fps)

        expected_frames = int(self.duration * self.fps)
        if expected_frames != self.num_frames:
            raise ValueError(
                f"duration ({self.duration}) * fps ({self.fps}) = {expected_frames} "
                f"frames, but num_frames={self.num_frames}. These must match."
            )

        # Load and process dataset
        self.df, _ = self._load_and_process_dataframe()

        # Extract dataset components
        self._extract_dataset_components()

        # Load label mapping and labels
        self.label_mapping = load_label_mapping(label_mapping_path)
        self._process_labels()

        # Setup image transformations
        self._setup_transforms(mean, std)

        print(f"Dataset initialized with {len(self)} samples for split '{split}'")


    def _load_and_process_dataframe(self) -> pd.DataFrame:
        """Load CSV file and apply initial filtering."""
        df = pd.read_csv(self.csv_path)


        # Filter by split
        assert self.split in ['train', 'val', 'test'], f"Invalid split: {self.split}. Only 'train', 'val', and 'test' are supported for NexarDataset."
        # Note: there is not validation split in Nexar, so we treat val as test
        if self.split == 'val':
            self.split = 'test'
            print("Note: 'val' split is treated as 'test' split for NexarDataset.")
        df = DataValidator.filter_dataframe_by_split(df, self.split)
        
        return df, None


    def _extract_dataset_components(self) -> None:
        """Extract video IDs and paths from dataframe."""
        self.video_id = self.df['file_name'].tolist()
        self.video_id = [vid.replace('.mp4', '') for vid in self.video_id]
        # Modify to load train/test videos accordingly to the original path structure of Nexar
        split_path_modifier = "train" if self.split == "train" else "test"
        # If test, have to determine if the video comes from the public or private test set, select according to the csv_info
        if self.split == "test" or self.split == "val":
            test_split_modifier = self.df['test_split'].tolist()
            # Check that all values are either 'public' or 'private'
            for val in test_split_modifier:
                if val not in ['public', 'private']:
                    raise ValueError(f"Unexpected test_split value: {val}. Expected 'public' or 'private'.")
        
        # Modify path also according to the label, if crash positive, normal_driving negative
        label_modifier = self.df['label'].tolist()
        for i in range(len(label_modifier)):
            if label_modifier[i] == 'crash':
                label_modifier[i] = 'positive'
            elif label_modifier[i] == 'normal_driving':
                label_modifier[i] = 'negative'
            else:
                raise ValueError(f"Unexpected label value: {label_modifier[i]}. Expected 'crash' or 'normal_driving'.")

        self.video_path = []

        for i in range(len(self.video_id)):
            if self.split == "train":
                video_subpath = self.video_cache_dir / f"{split_path_modifier}/{label_modifier[i]}/{self.video_id[i]}.mp4"
            else:
                video_subpath = self.video_cache_dir / f"{split_path_modifier}-{test_split_modifier[i]}/{label_modifier[i]}/{self.video_id[i]}.mp4"
            self.video_path.append(video_subpath)

    def _process_labels(self) -> None:
        """Process classification and regression labels."""
        # Classification labels
        raw_cls_labels = self.df['label'].tolist()
        self.cls_labels = [self.label_mapping[label] for label in raw_cls_labels]

        # Event timestamps (time of collision/event in seconds)
        assert 'time_of_event' in self.df.columns, \
            "Column 'time_of_event' not found in dataframe."
        raw_timestamps = self.df['time_of_event'].tolist()
        self.acc_timestamps = [float(ts) for ts in raw_timestamps]

        # Per-sample anticipation time: (time_of_event - time_of_alert) for crash samples, None for normal
        if self.use_time_of_alert_offset:
            assert 'time_of_alert' in self.df.columns, \
                "Column 'time_of_alert' not found in CSV, required when use_time_of_alert_offset=True"
            assert 'time_of_event' in self.df.columns, \
                "Column 'time_of_event' not found in CSV, required when use_time_of_alert_offset=True"
            self.per_sample_anticipation_time = []
            for i in range(len(self.df)):
                if self.cls_labels[i] == 1:  # crash
                    toa = self.df.iloc[i]['time_of_alert']
                    toe = self.df.iloc[i]['time_of_event']
                    self.per_sample_anticipation_time.append(toe - toa)
                else:
                    self.per_sample_anticipation_time.append(None)


    def _setup_transforms(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Setup image transformations based on split."""
        mean_list = mean.tolist() if isinstance(mean, torch.Tensor) else mean
        std_list = std.tolist() if isinstance(std, torch.Tensor) else std

        interp_method = self.DEFAULT_CONFIG['INTERPOLATION_MAPPING'][self.resampling]

        if self.split == 'train':
            self.transform = A.Compose([
                A.VerticalFlip(p=0.2),

                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(p=0.3),
                A.CLAHE(p=0.2),

                A.RandomFog(p=0.15),
                A.RandomRain(p=0.15),
                A.RandomSunFlare(p=0.1),
                A.RandomSnow(p=0.2),
                A.RandomShadow(p=0.2),

                A.MotionBlur(blur_range=(3, 3), p=0.2),
                A.GaussNoise(p=0.2),

                A.Resize(self.frame_size, self.frame_size, interpolation=interp_method),
                A.Normalize(mean=mean_list, std=std_list),
                ToTensorV2(),
            ], seed=self.seed)
        else:
            # Validation/test: deterministic transforms
            self.transform = A.Compose([
                A.Resize(self.frame_size, self.frame_size, interpolation=interp_method),
                A.Normalize(mean=mean_list, std=std_list),
                ToTensorV2(),
            ], seed=self.seed)


    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.video_path)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Get a single sample from the dataset.

        Returns:
            (video_tensor, cls_label)
            - video_tensor: Shape (T, C, H, W) where T=num_frames (e.g., 16)
        """
        try:
            return self._get_standard_item(idx)

        except Exception as e:
            vid = self.video_id[idx]
            error_msg = str(e).lower()
            if (isinstance(e, RuntimeError) or
                any(keyword in error_msg for keyword in [
                    'invalid nal unit', 'h264', 'corrupt', 'decode', 'damaged',
                    'invalid', 'failed to create videoreader'
                ])):
                print(f"Skipping corrupted video idx {idx}, video_id {vid}: {e}")
                return None
            else:
                raise e
            


    def _get_standard_item(self, idx: int) -> Optional[Tuple]:
        """Original __getitem__ logic for training."""
        # Get sample metadata
        video_id = self.video_id[idx]
        video_path = self.video_path[idx]
        cls_label = self.cls_labels[idx]
        acc_timestamp = self.acc_timestamps[idx]

        # Ensure video is available locally
        if not self.ensure_video_available(idx):
            print(f"Could not obtain video for idx {idx}")
            return None

        # Compute per-sample anticipation offset range if using time_of_alert
        sample_offset_range = None
        if self.use_time_of_alert_offset and cls_label == 1:
            anticipation_time = self.per_sample_anticipation_time[idx]
            sample_offset_range = (
                self.anticipation_offset_range[0],
                max(anticipation_time, self.anticipation_offset_range[1])
            )

        video_tensor = self._load_and_process_video(
            video_path, video_id, acc_timestamp,
            anticipation_offset_range_override=sample_offset_range
        )

        return video_tensor, cls_label

    def _load_and_process_video(self, video_path: Path, video_id: str, acc_timestamp: float, anticipation_offset_range_override: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Load video and extract/process frames.

        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            acc_timestamp: Event/collision timestamp in seconds
            anticipation_offset_range_override: If provided, overrides self.anticipation_offset_range for this sample

        Returns:
            Processed video tensor of shape (T, C, H, W)

        Raises:
            RuntimeError: If video cannot be loaded or processed
        """
        vr = None
        try:
            # Load video with error handling
            try:
                vr = VideoReader(str(video_path), ctx=cpu(0))
            except Exception as e:
                raise RuntimeError(f"Failed to create VideoReader for {video_path}: {e}")

            # Get total frames with validation
            try:
                total_frames = len(vr)
                if total_frames <= 0:
                    raise RuntimeError(f"Video has no frames: {video_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to get video length for {video_path}: {e}")

            # Quick validation: try to read the first frame to detect early corruption
            try:
                test_frame = vr[0].asnumpy()
                if test_frame is None or test_frame.size == 0:
                    raise RuntimeError("Test frame is empty - video likely corrupted")
            except Exception as e:
                raise RuntimeError(f"Failed to read test frame (video corruption): {e}")
            
            # Extract fps of video or use fixed value
            if self.use_fixed_fps:
                original_fps = float(self.DEFAULT_CONFIG['DEFAULT_FPS'])
            else:
                try:
                    original_fps = float(vr.get_avg_fps())
                    if original_fps <= 0:
                        raise RuntimeError(f"Invalid original FPS: {original_fps}")
                except Exception as e:
                    print(f"Warning: Failed to extract original FPS for {video_path}, defaulting to {self.DEFAULT_CONFIG['DEFAULT_FPS']}: {e}")
                    original_fps = float(self.DEFAULT_CONFIG['DEFAULT_FPS'])

            # NOTE: 
            # In NexarDataset, there is no acc_timestamp if the video is normal driving
            # Sample a timestamp randomly in the middle of the video for normal driving videos
            # However, make sure that there is enough buffer to extract frames around it
            # I want to do this only for train split, for test/val the videos are already cut before the event.
            # In this case, the acc_timestamp should be at the end of the video
            if acc_timestamp is None or np.isnan(acc_timestamp):
                if self.split == 'train':
                    # Compute video duration
                    video_duration = (total_frames - 1) / original_fps
                    # Define safe range for sampling timestamp
                    min_safe_time = self.duration / 2
                    max_safe_time = video_duration - (self.duration / 2)
                    if max_safe_time <= min_safe_time:
                        raise RuntimeError(f"Video too short to sample safe timestamp: {video_path}")
                    # Sample uniformly within safe range
                    acc_timestamp = random.uniform(min_safe_time, max_safe_time)
                else:
                    # For val/test, set acc_timestamp to the end of the video
                    acc_timestamp = (total_frames - 1) / original_fps

            # Compute frame indices
            try:
                if self.split == 'train':
                    offset_range = anticipation_offset_range_override or self.anticipation_offset_range
                    anticipation_offset = random.uniform(
                        offset_range[0],
                        offset_range[1]
                    )
                else:
                    anticipation_offset = 0

                # In the test split, videos are pre-cut, so no random offsetting. Also the acc_timestamp is not aligned with the cut.
                # Make it so that the video end corresponds exactly at acc_timestamp
                if self.split in ['val', 'test']:
                    # NOTE: just for precaution, it should already be 0 from above logic
                    anticipation_offset = 0
                    # Compute timestamp from the fps and total frames
                    acc_timestamp = (total_frames - 1) / original_fps

                frame_indices, _ = FrameProcessor.compute_frame_indices_around_peak(
                    event_timestamp=acc_timestamp,
                    duration=self.duration,
                    fps=float(self.sampling_fps),
                    original_fps=float(original_fps),
                    buffer_after_peak=0.0,
                    anticipation_offset=anticipation_offset,
                    total_frames=total_frames,
                    prediction_future_frames=self.prediction_future_frames,
                )

                total_expected_frames = self.num_frames + self.prediction_future_frames
                if len(frame_indices) != total_expected_frames:
                    if len(frame_indices) < total_expected_frames:
                        raise RuntimeError(f"Expected {total_expected_frames} frames, but got only {len(frame_indices)} frames. Video too short around the selected timestamp.")
                    elif len(frame_indices) > total_expected_frames:
                        # Trim excess context frames from the front, keep all future frames at the end
                        frame_indices = frame_indices[-total_expected_frames:]

            except Exception as e:
                raise RuntimeError(f"Failed to compute frame indices for {video_path}: {e}")

            # Extract frames
            try:
                frames = vr.get_batch(frame_indices).asnumpy()  # (num_frames, H, W, C)

                # Validate extracted frames
                if frames is None or frames.size == 0:
                    raise RuntimeError("Extracted frames are empty")

                total_expected_frames = self.num_frames + self.prediction_future_frames
                expected_shape = (total_expected_frames, None, None, 3)  # (T, H, W, C)
                if len(frames.shape) != 4 or frames.shape[0] != total_expected_frames or frames.shape[3] != 3:
                    raise RuntimeError(f"Invalid frame shape: {frames.shape}, expected: {expected_shape}")

            except Exception as e:
                raise RuntimeError(f"Failed to extract frames from {video_path}: {e}")

            # Apply transformations
            try:
                video_tensor = self.transform(images=frames)['images']  # (T, C, H, W)

                # Final validation
                if video_tensor.numel() == 0:
                    raise RuntimeError("Final video tensor is empty")

                return video_tensor

            except Exception as e:
                raise RuntimeError(f"Failed to process frames for {video_path}: {e}")

        except Exception as e:
            # Clean up VideoReader if it was created
            if vr is not None:
                try:
                    del vr
                except:
                    pass
            # Re-raise the error to be caught by __getitem__
            raise e

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle None values from corrupted videos.

        Args:
            batch: List of samples. Each sample is either:
                - None (corrupted)
                - (video, cls_label)

        Returns:
            Tuple of (videos, cls_labels), or None if the whole batch is corrupted.
        """
        # Filter out None values
        batch = [item for item in batch if item is not None]

        if len(batch) == 0:
            # If all samples in batch are corrupted, signal an empty batch
            print("All samples in batch are corrupted, returning empty batch")
            return None

        # Separate the components from batched samples
        video_tensors = []
        cls_labels = []
        for video_tensor, cls_label in batch:
            video_tensors.append(video_tensor)
            cls_labels.append(cls_label)

        collated_videos = torch.stack(video_tensors, dim=0)
        collated_cls_labels = torch.as_tensor(cls_labels, dtype=torch.long)

        return collated_videos, collated_cls_labels



    def ensure_video_available(self, idx: int) -> bool:
        """
        Check that the video is available locally.

        Args:
            idx: Dataset index

        Returns:
            True if the local video file exists and is non-empty, False otherwise
        """
        local_path = self.video_path[idx]
        return local_path.exists() and local_path.stat().st_size > 0
