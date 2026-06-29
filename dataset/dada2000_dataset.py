"""
DADA-2000 dataset configuration for sliding window evaluation.
"""

from pathlib import Path


class Dada2000Dataset:
    """DADA-2000 dataset — eval metadata and video path construction."""

    EVAL_CONFIG = {
        'csv_default': 'data/dada2000_small_test.csv',
        'data_root_default': '',
        'label_mapping_default': 'dataset/dada2000_label_mapping.json',
        'display_name': 'DADA-2000',
        'default_fps': 30,
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
        if not vid.endswith('.avi') and not vid.endswith('.mp4'):
            vid_stem = vid
        else:
            vid_stem = vid.rsplit('.', 1)[0]
        return data_root / "videos" / f"images_{vid_stem}.avi"

    @staticmethod
    def get_eval_video_id(row):
        return str(row['id']).replace('.mp4', '').replace('.avi', '')