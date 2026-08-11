"""Milestone 5: simulate the official eval end-to-end.

Master split: GroupShuffleSplit(test_size=0.28, random_state=0) on slides ->
pseudo-test (~380 tiles, unseen slides). On the inner 72%: train model A
(seed 42 inner split) + model B (seed 43), estimator selection on A's val —
exactly solution.py's flow — then predict the pseudo-test with flip-TTA and
score with the exact metric.

Usage: python sim_eval.py --out runs/sim0 [--epochs 40] [--cpi 4] [--tta 4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
DATA = Path(r"G:\ml\Latest_Chals\challenge 2\dataset")

# reuse solution.py internals (import-safe)
sys.path.insert(0, str(Path(__file__).parent.parent))
import solution as S
from metric import morphometry_score, COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cpi", type=int, default=4)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--stop-a", type=float, default=90.0)
    ap.add_argument("--stop-all", type=float, default=180.0)
    ap.add_argument("--single", action="store_true", help="model A only (CPU recipe)")
    ap.add_argument("--device", default=None, help="dev override: cuda for fast sim")
    ap.add_argument("--data", default=None, help="dataset dir override")
    args = ap.parse_args()
    global DATA
    if args.data:
        DATA = Path(args.data)
    if args.device:
        S.DEVICE = args.device  # dev-only; deliverable path is always cpu
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    S.EPOCHS = args.epochs
    S.CPI = args.cpi
    t0 = time.time()

    targets = pd.read_csv(DATA / "train_targets.csv")
    index = pd.read_csv(DATA / "train_index.csv")
    cells = pd.read_csv(DATA / "train_cells.csv")
    ids_all = list(targets["image_id"])

    cls_idx = {c: i for i, c in enumerate(S.CLASSES)}
    cells["ci"] = cells["cell_type"].map(cls_idx).astype(int)
    cells_by = {}
    for iid, g in cells.groupby("image_id"):
        cells_by[iid] = {"x": g["x"].values.astype(np.float64),
                         "y": g["y"].values.astype(np.float64),
                         "ci": g["ci"].values.astype(np.int64),
                         "area": g["area"].values.astype(np.float64),
                         "elo": g["elongation"].values.astype(np.float64)}

    from PIL import Image
    print("loading images...", flush=True)
    images = {}
    for iid in ids_all:
        with Image.open(DATA / "train_images" / f"{iid}.png") as im:
            images[iid] = np.asarray(im.convert("RGB"))

    # master split
    from sklearn.model_selection import GroupShuffleSplit
    groups_all = index.set_index("image_id").loc[ids_all, "group"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.28, random_state=0)
    inner_idx, ptest_idx = next(gss.split(ids_all, groups=groups_all))
    inner_ids = [ids_all[i] for i in inner_idx]
    ptest_ids = [ids_all[i] for i in ptest_idx]
    print(f"inner {len(inner_ids)} tiles / pseudo-test {len(ptest_ids)} tiles", flush=True)

    groups_inner = index.set_index("image_id").loc[inner_ids, "group"].values

    def split(seed):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        tr, va = next(gss.split(inner_ids, groups=groups_inner))
        return [inner_ids[i] for i in tr], [inner_ids[i] for i in va]

    tr_a, va_a = split(S.SEED)
    tr_tgt = targets.set_index("image_id").loc[tr_a]
    train_stats = {"dispersion_mean": float(tr_tgt["dispersion"].mean()),
                   "size_spread_med": float(tr_tgt["size_spread"].median()),
                   "elongation_med": float(tr_tgt["elongation"].median())}

    model_a, tau_a = S.train_one(images, cells_by, tr_a, va_a, targets, train_stats,
                                 S.SEED, stop_min=args.stop_a, epochs=args.epochs, tag="A")
    torch.save({"model": model_a.state_dict(), "mean_tau": tau_a, "epoch": -1},
               out / "sim_A.pt")
    models = [model_a]
    model_b, tau_b = None, -1.0
    if not args.single:
        tr_b, va_b = split(S.SEED + 1)
        model_b, tau_b = S.train_one(images, cells_by, tr_b, va_b, targets, train_stats,
                                     S.SEED + 1, stop_min=args.stop_all,
                                     epochs=args.epochs, tag="B")
        torch.save({"model": model_b.state_dict(), "mean_tau": tau_b, "epoch": -1},
                   out / "sim_B.pt")
        if tau_b > 0.25:
            models.append(model_b)

    # cthr + estimator selection on A's val (mirrors solution.py)
    va_maps = {}
    for iid in va_a:
        img = images[iid]
        va_maps[iid] = (S.predict_maps(model_a, img, tta=1), img.shape[:2])
    gt_val = targets.set_index("image_id").loc[va_a]

    def decode_val(cthr):
        rows = []
        for iid in va_a:
            (dens, heat, amap, emap), (H, W) = va_maps[iid]
            d = S.decode_tile(dens, heat, amap, emap, H, W, train_stats, cthr=cthr)
            d["image_id"] = iid
            rows.append(d)
        return pd.DataFrame(rows).set_index("image_id").loc[va_a]

    best_cthr, best_ct_tau, va_pred = S.DENS_THR, -1.0, None
    for cthr in [1e-4, 2e-4, 3e-4, 5e-4]:
        p = decode_val(cthr)
        mt, _ = morphometry_score(p, gt_val)
        print(f"[select] cthr={cthr:g}: val mean_tau {mt:.4f}", flush=True)
        if mt > best_ct_tau:
            best_cthr, best_ct_tau, va_pred = cthr, mt, p
    print(f"[select] chose cthr={best_cthr:g}", flush=True)

    use_alt = {}
    for c in ["size_spread", "elongation"]:
        t_pri = S.clipped_tau(va_pred[c], gt_val[c])
        t_alt = S.clipped_tau(va_pred[f"{c}_alt"], gt_val[c])
        use_alt[c] = bool(t_alt > t_pri)
        print(f"[select] {c}: primary {t_pri:.4f} alt {t_alt:.4f} -> "
              f"{'alt' if use_alt[c] else 'primary'}", flush=True)

    # pseudo-test inference: single models and ensemble
    def infer(mods, tta):
        rows = []
        for iid in ptest_ids:
            d = S.predict_tile_multi(mods, images[iid], train_stats, tta=tta,
                                     cthr=best_cthr)
            rec = {"image_id": iid}
            for c in COLS:
                rec[c] = d[f"{c}_alt"] if use_alt.get(c, False) else d[c]
            rows.append(rec)
        return pd.DataFrame(rows).set_index("image_id").loc[ptest_ids]

    gt_p = targets.set_index("image_id").loc[ptest_ids]
    results = {}
    configs = [("A_tta1", [model_a], 1), ("A_tta2", [model_a], 2),
               ("A_tta4", [model_a], args.tta)]
    if model_b is not None:
        configs += [("B_tta4", [model_b], args.tta), ("ENS_tta4", models, args.tta)]
    for name, mods, tta in configs:
        pred = infer(mods, tta)
        mean_tau, per = morphometry_score(pred, gt_p)
        results[name] = {"mean": mean_tau, **per}
        print(f"[ptest] {name}: mean {mean_tau:.4f} | " +
              " ".join(f"{c[:4]}={per[c]:.3f}" for c in COLS), flush=True)
        pred.to_csv(out / f"ptest_pred_{name}.csv")

    (out / "sim_results.json").write_text(json.dumps(
        {"results": results, "use_alt": use_alt, "tau_a": tau_a, "tau_b": tau_b,
         "minutes": (time.time() - t0) / 60}, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
