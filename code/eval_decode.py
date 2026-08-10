"""Dev tool: dump predicted maps for val tiles from a checkpoint, then sweep
decode variants offline (no retraining).

Usage:
  python eval_decode.py dump --ckpt runs/full1/best.pt --out runs/full1/valmaps [--seed 42] [--tta 4]
  python eval_decode.py sweep --maps runs/full1/valmaps
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from maps import decode_tile, nms_peaks, STRIDE, _sample3
from metric import COLS, morphometry_score
from scipy.stats import kendalltau

DATA = Path(r"G:\Datacurve\Latest_Chals\challenge 2\dataset")


def ct(a, b):
    t = kendalltau(a, b).statistic
    return float(np.clip(0.0 if np.isnan(t) else t, 0.0, 1.0))


def get_val_ids(seed=42):
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    targets = pd.read_csv(DATA / "train_targets.csv")
    index = pd.read_csv(DATA / "train_index.csv")
    ids_all = list(targets["image_id"])
    groups = index.set_index("image_id").loc[ids_all, "group"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr, va = next(gss.split(ids_all, groups=groups))
    return [ids_all[i] for i in va], targets


def dump(args):
    import torch
    from PIL import Image
    from infer import predict_maps
    from model import UNetR18

    va_ids, targets = get_val_ids(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetR18(pretrained=False).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, iid in enumerate(va_ids):
        with Image.open(DATA / "train_images" / f"{iid}.png") as im:
            img = np.asarray(im.convert("RGB"))
        dens, heat, amap, emap = predict_maps(model, img, device, tta=args.tta, raw=True)
        np.savez_compressed(outdir / f"{iid}.npz", dens=dens.astype(np.float32),
                            heat=heat.astype(np.float16), amap=amap.astype(np.float16),
                            emap=emap.astype(np.float16),
                            hw=np.array(img.shape[:2]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(va_ids)}", flush=True)
    print(f"dumped {len(va_ids)} val map sets to {outdir}")


def decode_variant(dens, heat, amap, emap, H, W, ts, nms_thr=0.05, kmul=1.0,
                   sample_center=False, count_mode="clip", cthr=0.0):
    if count_mode == "raw":
        per_class = dens.reshape(4, -1).sum(axis=1)
        dens = np.maximum(dens, 0.0)  # for downstream weighting only
    elif count_mode == "thr":
        d = np.where(dens > cthr, dens, 0.0)
        per_class = d.reshape(4, -1).sum(axis=1)
        dens = d
    else:  # clip
        dens = np.maximum(dens, 0.0)
        per_class = dens.reshape(4, -1).sum(axis=1)
    n_hat = float(per_class.sum())
    out = {"cellularity": 1e5 * n_hat / (W * H),
           "tumor_frac": float(per_class[0] / (n_hat + 1e-6))}
    k = int(max(5, round(kmul * n_hat)))
    ys, xs, sc = nms_peaks(heat, k)
    keep = sc > nms_thr
    if keep.sum() >= 3:
        ys, xs, sc = ys[keep], xs[keep], sc[keep]
    kk = len(ys)
    if kk >= 5:
        pts = np.column_stack([ys, xs]).astype(np.float64) * STRIDE
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        out["dispersion"] = float(np.sqrt(d2.min(axis=1)).mean() * 2.0 * np.sqrt(kk / (W * H)))
    else:
        out["dispersion"] = ts["dispersion_mean"]
    if kk >= 3:
        if sample_center:
            a_pk = np.exp(amap[ys, xs].astype(np.float64))
            e_pk = emap[ys, xs].astype(np.float64)
        else:
            a_pk = np.exp(_sample3(amap, ys, xs).astype(np.float64))
            e_pk = _sample3(emap, ys, xs).astype(np.float64)
        m = a_pk.mean()
        out["size_spread"] = float(a_pk.std(ddof=0) / m) if m > 0 else ts["size_spread_med"]
        out["elongation"] = float(e_pk.mean())
    else:
        out["size_spread"] = ts["size_spread_med"]
        out["elongation"] = ts["elongation_med"]

    # peak-based alternates for count-derived targets
    out["cellularity_pk"] = 1e5 * kk / (W * H)
    if kk >= 1:
        cls_at_pk = np.stack([_sample3(dens[c], ys, xs) for c in range(4)])  # (4,kk)
        votes = cls_at_pk.argmax(axis=0)
        out["tumor_frac_pk"] = float((votes == 0).mean())
        wsum = cls_at_pk.sum(axis=0) + 1e-9
        out["tumor_frac_soft"] = float((cls_at_pk[0] / wsum).mean())
    else:
        out["tumor_frac_pk"] = out["tumor_frac"]
        out["tumor_frac_soft"] = out["tumor_frac"]
    return out


def sweep(args):
    mdir = Path(args.maps)
    files = sorted(mdir.glob("*.npz"))
    targets = pd.read_csv(DATA / "train_targets.csv").set_index("image_id")
    ts = {"dispersion_mean": 1.2634, "size_spread_med": 0.5097, "elongation_med": 1.3185}
    cache = []
    for f in files:
        z = np.load(f)
        cache.append((f.stem, z["dens"].astype(np.float32), z["heat"].astype(np.float32),
                      z["amap"].astype(np.float32), z["emap"].astype(np.float32),
                      int(z["hw"][0]), int(z["hw"][1])))
    ids = [c[0] for c in cache]
    gt = targets.loc[ids]

    variants = [dict(count_mode="thr", cthr=3e-4, kmul=1.0)]

    for v in variants:
        rows = []
        for (iid, dens, heat, amap, emap, H, W) in cache:
            d = decode_variant(dens, heat, amap, emap, H, W, ts, **v)
            d["image_id"] = iid
            rows.append(d)
        pred = pd.DataFrame(rows).set_index("image_id").loc[ids]
        mean_tau, per = morphometry_score(pred, gt)
        print(f"{v}: mean {mean_tau:.4f} | " +
              " ".join(f"{c[:4]}={per[c]:.3f}" for c in COLS), flush=True)
        for alt_col, base in [("cellularity_pk", "cellularity"),
                              ("tumor_frac_pk", "tumor_frac"),
                              ("tumor_frac_soft", "tumor_frac")]:
            print(f"    {alt_col}: tau {ct(pred[alt_col].values, gt[base].values):.4f} "
                  f"(base {ct(pred[base].values, gt[base].values):.4f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "sweep"])
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--maps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "dump":
        dump(args)
    else:
        sweep(args)
