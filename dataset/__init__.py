"""
Video Dataset Module

This module provides dataset classes and utilities for loading and processing
video data for accident anticipation.
"""

from .nexar_dataset import NexarDataset
from .dad_dataset import DadDataset
from .dada2000_dataset import Dada2000Dataset
from .dota_dataset import DotaDataset
from .dataset_utils import (
    DataValidator,
    FrameProcessor,
    load_label_mapping,
    ensure_directory_exists,
)
__all__ = [
    'NexarDataset',
    'DadDataset',
    'Dada2000Dataset',
    'DotaDataset',
    'DataValidator',
    'FrameProcessor',
    'load_label_mapping',
    'ensure_directory_exists',
]