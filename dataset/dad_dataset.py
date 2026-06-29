"""
DAD dataset configuration for sliding window evaluation.
"""

from pathlib import Path


class DadDataset:
    """DAD dataset — eval metadata and video path construction."""

    EVAL_CONFIG = {
        'csv_default': 'data/dad_test.csv',
        'data_root_default': '',
        'label_mapping_default': 'dataset/dad_label_mapping.json',
        'display_name': 'DAD',
        'default_fps': 25,
        'time_of_collision_col': 'Time-of-collision',
        'label_col': 'label',
        'positive_labels': ['positive'],
        'video_has_accident': True,
    }

    @staticmethod
    def build_eval_video_path(row, data_root):
        data_root = Path(data_root)
        vid = str(row['id'])
        label = row['label']
        label_dir = 'positive' if label in ['positive'] else 'negative'
        if not vid.endswith('.mp4'):
            vid = vid + '.mp4'
        return data_root / "testing" / label_dir / vid

    @staticmethod
    def get_eval_video_id(row):
        return str(row['id']).replace('.mp4', '').replace('.avi', '')