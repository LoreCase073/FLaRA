"""
Evaluation metrics for sliding window collision prediction.

Computes AP, AUC, mTTA, TTA@R80, rmTTA, rTTA@R80 from per-video
temporal prediction series.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluation(all_pred, all_labels, num_valid_windows, toc_seconds,
               stride_seconds_arr, window_duration_seconds_arr,
               video_duration_seconds_arr):
    """
    Compute AP, AUC, mTTA, rmTTA, and TTA@R80 from per-video temporal prediction series.

    All time computations use seconds directly — no intermediate conversion to
    window indices — to avoid quantization artifacts.

    Args:
        all_pred: (N x T) prediction matrix. Entry [i, w] is the collision
                  probability of window w for video i. Padded with 0.0.
        all_labels: (N,) binary labels (1=positive, 0=negative).
        num_valid_windows: (N,) int array. Number of valid windows per video
                          (windows whose END is before the collision for positives,
                          or all windows for negatives).
        toc_seconds: (N,) float array. Time of collision in seconds from video
                    start. For Nexar: video_duration + tta_gap. For negatives:
                    can be any value (not used for TTA).
        stride_seconds_arr: (N,) float array. Stride in seconds per video
                           (= sliding_window_stride / video_fps).
        window_duration_seconds_arr: (N,) float array. Window duration in seconds
                                    per video (= num_frames / sampling_fps).
        video_duration_seconds_arr: (N,) float array. Actual video file duration
                                   in seconds (= (total_frames - 1) / video_fps).
                                   Used for rmTTA: (T-t)/T * video_duration per video.

    Returns:
        dict with keys: 'AP', 'AUC', 'mTTA', 'TTA_R80', 'rmTTA', 'rTTA_R80'
    """
    # --- Filter videos: extract valid prediction slices ---
    preds_eval = []
    labels_eval = []
    toc_eval = []
    stride_eval = []
    win_dur_eval = []
    vid_dur_eval = []
    min_pred = np.inf
    n_frames = 0

    for idx in range(len(all_labels)):
        n_valid = int(num_valid_windows[idx])
        pred = all_pred[idx, :n_valid]
        if len(pred) == 0:
            continue
        min_pred = min(min_pred, np.min(pred))
        preds_eval.append(pred)
        labels_eval.append(all_labels[idx])
        toc_eval.append(toc_seconds[idx])
        stride_eval.append(stride_seconds_arr[idx])
        win_dur_eval.append(window_duration_seconds_arr[idx])
        vid_dur_eval.append(video_duration_seconds_arr[idx])
        n_frames += len(pred)

    labels_eval = np.array(labels_eval)
    toc_eval = np.array(toc_eval)
    stride_eval = np.array(stride_eval)
    win_dur_eval = np.array(win_dur_eval)
    vid_dur_eval = np.array(vid_dur_eval)

    if n_frames == 0 or len(preds_eval) == 0:
        print("Warning: No valid predictions to evaluate.")
        return {'AP': 0.0, 'AUC': 0.0, 'mTTA': 0.0, 'TTA_R80': 0.0,
                'rmTTA': 0.0, 'rTTA_R80': 0.0}

    total_positives = np.sum(labels_eval)
    total_negatives = len(labels_eval) - total_positives

    # --- AUC via sklearn ---
    max_scores = np.array([np.max(p) for p in preds_eval])
    if total_positives > 0 and total_negatives > 0:
        AUC = roc_auc_score(labels_eval, max_scores)
    else:
        AUC = 0.0

    # --- Sweep thresholds (for AP, mTTA, rmTTA) ---
    thresholds = np.arange(max(min_pred, 0), 1.0, 0.001)
    n_thresholds = len(thresholds)
    Precision = np.zeros(n_thresholds)
    Recall = np.zeros(n_thresholds)
    TTA = np.zeros(n_thresholds)
    rTTA = np.zeros(n_thresholds)
    cnt = 0

    for Th in thresholds:
        Tp = 0.0
        Tp_Fp = 0.0
        tta_sum = 0.0
        rtta_sum = 0.0
        tta_count = 0.0

        for i in range(len(preds_eval)):
            # True positive: positive video where at least one prediction >= Th
            if labels_eval[i] > 0:
                tp = np.where(preds_eval[i] >= Th)
                Tp += float(len(tp[0]) > 0)

                if len(tp[0]) > 0:
                    # TTA in seconds: toc - window_end_time of first detection
                    first_det_window = tp[0][0]
                    detection_end_time = first_det_window * stride_eval[i] + win_dur_eval[i]
                    tta_val = max(toc_eval[i] - detection_end_time, 0.0)
                    tta_sum += tta_val
                    # Relative TTA scaled by video duration: (T-t)/T * video_duration
                    if toc_eval[i] > 0:
                        rtta_sum += (tta_val / toc_eval[i]) * vid_dur_eval[i]
                    tta_count += 1

            # Any video (positive or negative) with at least one prediction >= Th
            detected = float(len(np.where(preds_eval[i] >= Th)[0]) > 0)
            Tp_Fp += detected

        if Tp_Fp == 0:
            continue
        Precision[cnt] = Tp / Tp_Fp

        if total_positives == 0:
            continue
        Recall[cnt] = Tp / total_positives

        if tta_count > 0:
            TTA[cnt] = tta_sum / tta_count
            rTTA[cnt] = rtta_sum / tta_count
        cnt += 1

    if cnt == 0:
        print("Warning: Could not compute meaningful metrics.")
        return {'AP': 0.0, 'AUC': 0.0, 'mTTA': 0.0, 'TTA_R80': 0.0,
                'rmTTA': 0.0, 'rTTA_R80': 0.0}

    # --- AP (area under Precision-Recall curve) ---
    # Sort by recall ascending
    pr_index = np.argsort(Recall[:cnt])
    Precision_sorted = Precision[:cnt][pr_index]
    Recall_sorted = Recall[:cnt][pr_index]
    TTA_sorted = TTA[:cnt][pr_index]
    rTTA_sorted = rTTA[:cnt][pr_index]

    # Deduplicate recall values (keep max Precision / max TTA per unique recall)
    unique_recall, rec_rep_index = np.unique(Recall_sorted, return_index=True)
    # Skip recall=0 entries (if present), but keep all positive recall values
    nonzero_mask = unique_recall > 0
    unique_recall = unique_recall[nonzero_mask]
    rec_rep_index = rec_rep_index[nonzero_mask]

    if len(rec_rep_index) == 0:
        print("Warning: Could not compute meaningful P-R curve.")
        return {'AP': 0.0, 'AUC': 0.0, 'mTTA': 0.0, 'TTA_R80': 0.0,
                'rmTTA': 0.0, 'rTTA_R80': 0.0}

    new_Precision = np.zeros(len(rec_rep_index))
    new_TTA = np.zeros(len(rec_rep_index))
    new_rTTA = np.zeros(len(rec_rep_index))
    for i in range(len(rec_rep_index) - 1):
        new_Precision[i] = np.max(Precision_sorted[rec_rep_index[i]:rec_rep_index[i + 1]])
        new_TTA[i] = np.max(TTA_sorted[rec_rep_index[i]:rec_rep_index[i + 1]])
        new_rTTA[i] = np.max(rTTA_sorted[rec_rep_index[i]:rec_rep_index[i + 1]])
    new_Precision[-1] = Precision_sorted[rec_rep_index[-1]]
    new_TTA[-1] = TTA_sorted[rec_rep_index[-1]]
    new_rTTA[-1] = rTTA_sorted[rec_rep_index[-1]]
    new_Recall = Recall_sorted[rec_rep_index]

    # Trapezoidal AP
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    # --- mTTA and TTA@R80 (already in seconds) ---
    mTTA = np.mean(new_TTA)

    sort_idx = np.argsort(new_Recall)
    sort_recall = new_Recall[sort_idx]
    sort_tta = new_TTA[sort_idx]
    TTA_R80 = sort_tta[np.argmin(np.abs(sort_recall - 0.8))]

    # --- rmTTA and rTTA@R80 (relative, in seconds: (T-t)/T * video_duration) ---
    rmTTA = np.mean(new_rTTA)

    sort_rtta = new_rTTA[sort_idx]
    rTTA_R80 = sort_rtta[np.argmin(np.abs(sort_recall - 0.8))]

    print(f"Average Precision= {AP:.4f}, AUC= {AUC:.4f}, mean Time to accident= {mTTA:.4f}s")
    print(f"Recall@80%, Time to accident= {TTA_R80:.4f}s")
    print(f"relative mTTA= {rmTTA:.4f}s, relative TTA@R80= {rTTA_R80:.4f}s")

    return {'AP': float(AP), 'AUC': float(AUC),
            'mTTA': float(mTTA), 'TTA_R80': float(TTA_R80),
            'rmTTA': float(rmTTA), 'rTTA_R80': float(rTTA_R80)}