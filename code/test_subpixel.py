"""Oracle test: does 3x3 center-of-mass subpixel peak refinement improve the
dispersion decode ceiling?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from maps import CLASSES, render_maps, nms_peaks, STRIDE
from scipy.stats import kendalltau

DATA = Path(r"G:\Datacurve\Latest_Chals\challenge 2\dataset")


def ct(a, b):
    t = kendalltau(a, b).statistic
    return float(np.clip(0.0 if np.isnan(t) else t, 0.0, 1.0))


def refine(heat, ys, xs):
    """3x3 center-of-mass refinement around each peak."""
    H, W = heat.shape
    ry, rx = ys.astype(np.float64).copy(), xs.astype(np.float64).copy()
    for i, (y, x) in enumerate(zip(ys, xs)):
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        p = np.maximum(heat[y0:y1, x0:x1], 0)
        s = p.sum()
        if s > 0:
            gy, gx = np.mgrid[y0:y1, x0:x1]
            ry[i] = (p * gy).sum() / s
            rx[i] = (p * gx).sum() / s
    return ry, rx


def ce_from_pts(py, px, W, H):
    k = len(py)
    pts = np.column_stack([py, px]) * STRIDE
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    return float(nn.mean() * 2.0 * np.sqrt(k / (W * H)))


def main():
    targets = pd.read_csv(DATA / "train_targets.csv")
    cells = pd.read_csv(DATA / "train_cells.csv")
    cells["ci"] = cells["cell_type"].map({c: i for i, c in enumerate(CLASSES)})
    ids = list(targets["image_id"])
    rng = np.random.RandomState(2)
    ids = list(rng.choice(ids, 500, replace=False))
    tset = targets.set_index("image_id").loc[ids]
    grouped = {k: v for k, v in cells.groupby("image_id")}

    d_int, d_sub = [], []
    for iid in ids:
        with Image.open(DATA / "train_images" / f"{iid}.png") as im:
            W, H = im.size
        g = grouped[iid]
        m = render_maps(list(zip(g["x"], g["y"], g["ci"], g["area"], g["elongation"])), H, W)
        n = len(g)
        k = int(max(5, round(n)))
        ys, xs, sc = nms_peaks(m["heat"], k)
        keep = sc > 0.05
        if keep.sum() >= 3:
            ys, xs, sc = ys[keep], xs[keep], sc[keep]
        d_int.append(ce_from_pts(ys.astype(float), xs.astype(float), W, H))
        ry, rx = refine(m["heat"], ys, xs)
        d_sub.append(ce_from_pts(ry, rx, W, H))

    gt = tset["dispersion"].values
    print(f"integer peaks : tau = {ct(d_int, gt):.4f}")
    print(f"subpixel peaks: tau = {ct(d_sub, gt):.4f}")


if __name__ == "__main__":
    main()
