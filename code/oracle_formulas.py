"""Milestone 1a: recompute the 5 targets from train_cells.csv + PNG dims.

Tests several formula variants per target to find the EXACT GT arithmetic
(needed so the decode mirrors label generation bit-for-bit).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, spearmanr

DATA = Path(r"G:\Datacurve\Latest_Chals\challenge 2\dataset")


def clipped_tau(a, b):
    t = kendalltau(a, b).statistic
    return float(np.clip(0.0 if np.isnan(t) else t, 0.0, 1.0))


def main():
    targets = pd.read_csv(DATA / "train_targets.csv")
    cells = pd.read_csv(DATA / "train_cells.csv")

    # image dims (PIL header read only - fast)
    dims = {}
    for iid in targets["image_id"]:
        with Image.open(DATA / "train_images" / f"{iid}.png") as im:
            dims[iid] = im.size  # (W, H)

    rows = []
    for iid, g in cells.groupby("image_id"):
        W, H = dims[iid]
        n = len(g)
        area = g["area"].values.astype(float)
        elo = g["elongation"].values.astype(float)
        x = g["x"].values.astype(float)
        y = g["y"].values.astype(float)

        rec = {"image_id": iid, "n": n, "W": W, "H": H}
        rec["cellularity"] = 1e5 * n / (W * H)
        rec["tumor_frac"] = float((g["cell_type"] == "tumor").mean())
        m = area.mean()
        rec["ss_pop"] = area.std(ddof=0) / m       # population CV
        rec["ss_samp"] = area.std(ddof=1) / m if n > 1 else 0.0  # sample CV
        rec["elongation"] = elo.mean()

        # Clark-Evans variants
        if n >= 2:
            # variant 1: normalized [0,1]^2 coords, area=1
            pts = np.column_stack([x, y])
            d1 = cKDTree(pts).query(pts, k=2)[0][:, 1]
            rec["ce_norm"] = d1.mean() * 2.0 * np.sqrt(n)
            # variant 2: pixel coords, area = W*H
            ptsp = np.column_stack([x * W, y * H])
            d2 = cKDTree(ptsp).query(ptsp, k=2)[0][:, 1]
            rec["ce_pix"] = d2.mean() * 2.0 * np.sqrt(n / (W * H))
            # variant 3: pixel coords with (n-1)/n density (unbiased-ish)
            rec["ce_pix_n1"] = d2.mean() * 2.0 * np.sqrt((n - 1) / (W * H))
        else:
            rec["ce_norm"] = rec["ce_pix"] = rec["ce_pix_n1"] = np.nan
        rows.append(rec)

    df = pd.DataFrame(rows).merge(targets, on="image_id", suffixes=("_hat", ""))

    print("=== exactness checks (max abs diff, spearman, clipped tau) ===")
    checks = [
        ("cellularity", "cellularity_hat"),
        ("tumor_frac", "tumor_frac_hat"),
        ("size_spread", "ss_pop"),
        ("size_spread", "ss_samp"),
        ("elongation", "elongation_hat"),
        ("dispersion", "ce_norm"),
        ("dispersion", "ce_pix"),
        ("dispersion", "ce_pix_n1"),
    ]
    for gt_col, hat_col in checks:
        a = df[hat_col].values
        b = df[gt_col].values
        ok = np.isfinite(a)
        mad = np.max(np.abs(a[ok] - b[ok]))
        sp = spearmanr(a[ok], b[ok]).statistic
        ct = clipped_tau(a[ok], b[ok])
        print(f"{gt_col:12s} <- {hat_col:16s} maxdiff={mad:.6g}  spearman={sp:.6f}  tau={ct:.6f}")

    df.to_csv(Path(__file__).parent / "oracle_formula_recon.csv", index=False)
    print("\nsaved oracle_formula_recon.csv")


if __name__ == "__main__":
    main()
