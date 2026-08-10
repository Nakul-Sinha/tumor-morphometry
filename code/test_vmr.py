"""Oracle test: quadrat-count VMR from rendered GT density maps as a
peak-free dispersion estimator. Also tests it from raw GT points (upper bound)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from maps import CLASSES, render_maps, STRIDE
from metric import COLS
from scipy.stats import kendalltau

DATA = Path(r"G:\Datacurve\Latest_Chals\challenge 2\dataset")


def ct(a, b):
    t = kendalltau(a, b).statistic
    return float(np.clip(0.0 if np.isnan(t) else t, 0.0, 1.0))


def vmr_from_density(dtot, n_hat, H, W, factors=(0.7, 1.0, 1.4)):
    """dtot: total density map (integral ~= N) at half-res. Returns feature list."""
    mh, mw = dtot.shape
    lam = max(n_hat, 1.0) / (H * W)          # cells per full-res px^2
    nn_scale_px = 0.5 / np.sqrt(lam)         # expected NN dist (px)
    feats = []
    for f in factors:
        q = max(2, int(round(f * nn_scale_px / STRIDE)))  # quadrat side in map px
        gh, gw = max(1, mh // q), max(1, mw // q)
        c = dtot[:gh * q, :gw * q].reshape(gh, q, gw, q).sum(axis=(1, 3))
        m = c.mean()
        feats.append(-np.log(c.var() / m + 1e-6) if m > 0 else 0.0)
    return feats


def main():
    targets = pd.read_csv(DATA / "train_targets.csv")
    cells = pd.read_csv(DATA / "train_cells.csv")
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    cells["ci"] = cells["cell_type"].map(cls_idx)
    ids = list(targets["image_id"])
    rng = np.random.RandomState(1)
    ids = list(rng.choice(ids, 500, replace=False))
    tset = targets.set_index("image_id").loc[ids]

    grouped = {k: v for k, v in cells.groupby("image_id")}
    feats_all = []
    for iid in ids:
        with Image.open(DATA / "train_images" / f"{iid}.png") as im:
            W, H = im.size
        g = grouped[iid]
        m = render_maps(list(zip(g["x"], g["y"], g["ci"], g["area"], g["elongation"])), H, W)
        dtot = m["dens"].sum(axis=0)
        feats_all.append(vmr_from_density(dtot, len(g), H, W))
    F = np.array(feats_all)

    gt = tset["dispersion"].values
    for i, f in enumerate([0.7, 1.0, 1.4]):
        print(f"VMR scale {f}: tau vs dispersion = {ct(F[:, i], gt):.4f}")
    print(f"VMR mean-of-scales: tau = {ct(F.mean(1), gt):.4f}")


if __name__ == "__main__":
    main()
