"""Generate submission.csv from a trained checkpoint.

Usage: python predict.py --data <dataset_dir> --ckpt runs/X/best.pt --out submission.csv
       [--ckpt2 runs/Y/best.pt] [--tta 4] [--use-alt-ss] [--use-alt-el]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from infer import predict_maps
from maps import decode_tile
from metric import COLS
from model import UNetR18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt2", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--use-alt-ss", action="store_true")
    ap.add_argument("--use-alt-el", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)

    targets = pd.read_csv(data / "train_targets.csv")
    train_stats = {
        "dispersion_mean": float(targets["dispersion"].mean()),
        "size_spread_med": float(targets["size_spread"].median()),
        "elongation_med": float(targets["elongation"].median()),
    }
    queries = pd.read_csv(data / "test_queries.csv")

    models = []
    for p in [args.ckpt, args.ckpt2]:
        if p:
            m = UNetR18(pretrained=False).to(device)
            ck = torch.load(p, map_location=device, weights_only=False)
            m.load_state_dict(ck["model"])
            m.eval()
            models.append(m)
            print(f"loaded {p} (val mean_tau {ck.get('mean_tau'):.4f} @ ep {ck.get('epoch')})")

    rows = []
    t0 = time.time()
    for i, qid in enumerate(queries["query_id"]):
        with Image.open(data / "test_images" / f"{qid}.png") as im:
            img = np.asarray(im.convert("RGB"))
        H, W = img.shape[:2]
        acc = None
        for m in models:
            dens, heat, amap, emap = predict_maps(m, img, device, tta=args.tta)
            cur = (dens, heat, amap, emap)
            acc = cur if acc is None else tuple(a + c for a, c in zip(acc, cur))
        dens, heat, amap, emap = (a / len(models) for a in acc)
        d = decode_tile(dens, heat, amap, emap, H, W, train_stats=train_stats)
        if args.use_alt_ss:
            d["size_spread"] = d["size_spread_alt"]
        if args.use_alt_el:
            d["elongation"] = d["elongation_alt"]
        rows.append({"id": qid, **{c: d[c] for c in COLS}})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/384  {time.time()-t0:.0f}s", flush=True)

    sub = pd.DataFrame(rows)[["id"] + COLS]
    vals = sub[COLS].values
    assert np.isfinite(vals).all(), "non-finite predictions"
    assert len(sub) == len(queries) and set(sub["id"]) == set(queries["query_id"])
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
