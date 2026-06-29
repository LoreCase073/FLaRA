"""
DoTA dataset configuration for sliding window evaluation.
"""

from pathlib import Path


class DotaDataset:
    """DoTA dataset — eval metadata and video path construction."""

    EVAL_CONFIG = {
        'csv_default': 'data/dota_test.csv',
        'data_root_default': '',
        'label_mapping_default': 'dataset/dota_label_mapping.json',
        'display_name': 'DoTA',
        'default_fps': 10,
        'time_of_collision_col': 'Time-of-collision',
        'label_col': 'label',
        'positive_labels': [1, '1'],
        'video_has_accident': True,
        'negative_temporal_cutoff': 4.0,
    }

    @staticmethod
    def build_eval_video_path(row, data_root):
        data_root = Path(data_root)
        vid = str(row['id'])
        vid_stem = vid.replace('.mp4', '')
        return data_root / "dota_annotated" / vid_stem / f"{vid_stem}.mp4"

    @staticmethod
    def get_eval_video_id(row):
        return str(row['id']).replace('.mp4', '').replace('.avi', '')