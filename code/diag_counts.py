"""Diagnostic: predicted vs true cell counts per val tile, slide-clustered error.
Usage: python diag_counts.py --ckpt runs/full1/best.pt [--seed 42] [--tta 1]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_decode import get_val_ids, DATA
from infer import predict_maps
from model import UNetR18

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tta", type=int, default=1)
args = ap.parse_args()

va_ids, targets = get_val_ids(args.seed)
index = pd.read_csv(DATA / "train_index.csv").set_index("image_id")
cells = pd.read_csv(DATA / "train_cells.csv")
true_n = cells.groupby("image_id").size()
true_tumor = cells[cells.cell_type == "tumor"].groupby("image_id").size()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetR18(pretrained=False).to(device)
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
print(f"ckpt ep {ck['epoch']} val {ck['mean_tau']:.4f}")

rows = []
for iid in va_ids:
    with Image.open(DATA / "train_images" / f"{iid}.png") as im:
        img = np.asarray(im.convert("RGB"))
    dens, heat, amap, emap = predict_maps(model, img, device, tta=args.tta)
    n_hat = float(dens.sum())
    nt_hat = float(dens[0].sum())
    rows.append({"image_id": iid, "group": index.loc[iid, "group"],
                 "n_true": int(true_n.get(iid, 0)), "n_hat": n_hat,
                 "t_true": int(true_tumor.get(iid, 0)), "t_hat": nt_hat,
                 "side": np.sqrt(img.shape[0] * img.shape[1])})
df = pd.DataFrame(rows)
df["ratio"] = df.n_hat / df.n_true.clip(lower=1)
print("\nper-slide count ratio (pred/true):")
g = df.groupby("group").agg(n_tiles=("ratio", "count"), mean_ratio=("ratio", "mean"),
                            sd=("ratio", "std"), mean_side=("side", "mean")).round(3)
print(g.to_string())
print(f"\noverall ratio mean {df.ratio.mean():.3f} sd {df.ratio.std():.3f}")
print(f"within-slide sd of ratio (mean) {g.sd.mean():.3f} vs between-slide sd {g.mean_ratio.std():.3f}")
df.to_csv(Path(args.ckpt).parent / "diag_counts.csv", index=False)
