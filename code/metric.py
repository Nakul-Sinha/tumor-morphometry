"""Exact replication of MorphometryScore: mean of 5 clipped Kendall tau-b."""
import numpy as np
from scipy.stats import kendalltau

COLS = ["cellularity", "tumor_frac", "size_spread", "elongation", "dispersion"]


def morphometry_score(pred_df, gt_df):
    """pred_df, gt_df aligned on id (caller responsibility). Returns (mean, per-target dict)."""
    per = {}
    for c in COLS:
        t = kendalltau(pred_df[c].values, gt_df[c].values).statistic
        per[c] = float(np.clip(0.0 if np.isnan(t) else t, 0.0, 1.0))
    return float(np.mean(list(per.values()))), per
