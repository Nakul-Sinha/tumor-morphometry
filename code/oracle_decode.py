"""Milestone 1b: render GT maps from train_cells.csv, run the real decode,
compare decoded targets to train_targets.csv. Validates the whole decode chain
(peaks, top-K, Clark-Evans pixel-coord formula, attribute sampling) at the
oracle ceiling (perfect maps). Gate: mean tau >= 0.995... realistically the
attribute/peak steps lose a little; per design doc gate is >= 0.995 for the
formula path and we report the rendered-decode ceiling honestly.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from maps import CLASSES, render_maps, decode_tile
from metric import morphometry_score, COLS

DATA = Path(r"G:\ml\Latest_Chals\challenge 2\dataset")
N_TILES = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all


def main():
    targets = pd.read_csv(DATA / "train_targets.csv")
    cells = pd.read_csv(DATA / "train_cells.csv")
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    cells["ci"] = cells["cell_type"].map(cls_idx)

    ids = list(targets["image_id"])
    if N_TILES:
        rng = np.random.RandomState(0)
        ids = list(rng.choice(ids, N_TILES, replace=False))
    tset = targets.set_index("image_id").loc[ids]

    grouped = {k: v for k, v in cells.groupby("image_id")}
    t0 = time.time()
    rows = []
    for j, iid in enumerate(ids):
        with Image.open(DATA / "train_images" / f"{iid}.png") as im:
            W, H = im.size
        g = grouped[iid]
        cell_rows = list(zip(g["x"], g["y"], g["ci"], g["area"], g["elongation"]))
        m = render_maps(cell_rows, H, W)
        dec = decode_tile(m["dens"], m["heat"], m["amap"], m["emap"], H, W)
        dec["image_id"] = iid
        rows.append(dec)
        if (j + 1) % 200 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    pred = pd.DataFrame(rows).set_index("image_id").loc[ids]
    mean_tau, per = morphometry_score(pred, tset)
    print(f"\n=== ORACLE DECODE (rendered GT maps, n={len(ids)}) ===")
    for c in COLS:
        print(f"  {c:12s} tau={per[c]:.4f}")
    print(f"  MEAN tau = {mean_tau:.4f}")

    # alternative estimators
    for alt in ["size_spread", "elongation"]:
        p2 = pred.copy()
        p2[alt] = pred[f"{alt}_alt"]
        _, per2 = morphometry_score(p2, tset)
        print(f"  [alt] {alt:12s} tau={per2[alt]:.4f}")

    pred.to_csv(Path(__file__).parent / "oracle_decode_recon.csv")


if __name__ == "__main__":
    main()
