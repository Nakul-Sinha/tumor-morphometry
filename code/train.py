"""Dev trainer with grouped validation + metric-aligned checkpointing.

Usage: python train.py --data <dataset_dir> --out <run_dir> [--epochs 40] ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CropDataset, load_images_and_cells
from infer import predict_tile
from metric import COLS, morphometry_score
from model import UNetR18


def val_metric(model, images, ids, targets_df, device, train_stats, tta=1):
    model.eval()
    rows = []
    for iid in ids:
        d = predict_tile(model, images[iid], device, tta=tta, train_stats=train_stats)
        d["image_id"] = iid
        rows.append(d)
    pred = pd.DataFrame(rows).set_index("image_id").loc[ids]
    gt = targets_df.set_index("image_id").loc[ids]
    mean_tau, per = morphometry_score(pred, gt)
    # also score alt estimators
    alts = {}
    for c in ["size_spread", "elongation"]:
        p2 = pred.copy()
        p2[c] = pred[f"{c}_alt"]
        _, per2 = morphometry_score(p2, gt)
        alts[c] = per2[c]
    return mean_tau, per, alts, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cpi", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subset", type=int, default=0, help="limit #train images (smoke)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=1e9)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    targets = pd.read_csv(data / "train_targets.csv")
    index = pd.read_csv(data / "train_index.csv")
    ids_all = list(targets["image_id"])

    # grouped split
    from sklearn.model_selection import GroupShuffleSplit
    groups = index.set_index("image_id").loc[ids_all, "group"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(ids_all, groups=groups))
    tr_ids = [ids_all[i] for i in tr_idx]
    va_ids = [ids_all[i] for i in va_idx]
    if args.subset:
        tr_ids = tr_ids[:args.subset]
        va_ids = va_ids[:max(40, args.subset // 5)]
    print(f"train {len(tr_ids)} tiles / val {len(va_ids)} tiles "
          f"({len(set(groups[tr_idx]))} vs {len(set(groups[va_idx]))} slides)", flush=True)

    # train-only stats for decode fallbacks
    tr_tgt = targets.set_index("image_id").loc[tr_ids]
    train_stats = {
        "dispersion_mean": float(tr_tgt["dispersion"].mean()),
        "size_spread_med": float(tr_tgt["size_spread"].median()),
        "elongation_med": float(tr_tgt["elongation"].median()),
    }

    print("loading images...", flush=True)
    images, cells_by = load_images_and_cells(data, ids_all)
    print(f"loaded {len(images)} images in {time.time()-t0:.0f}s", flush=True)

    ds = CropDataset(images, cells_by, tr_ids, crops_per_image=args.cpi, train=True, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=(device == "cuda"), drop_last=True,
                    persistent_workers=(args.workers > 0))

    model = UNetR18(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    iters = max(1, len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * iters, eta_min=args.lr / 10)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best = -1.0
    hist = []
    for ep in range(args.epochs):
        if (time.time() - t0) / 60 > args.time_budget_min:
            print(f"[time guard] stopping training at epoch {ep}", flush=True)
            break
        ds.set_epoch(ep)
        model.train()
        agg = {"dens": 0.0, "cnt": 0.0, "heat": 0.0, "attr": 0.0, "n": 0}
        for x, tgt, counts in dl:
            x, tgt, counts = x.to(device), tgt.to(device), counts.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                pred = model(x)
                l_dens = F.mse_loss(pred[:, :4], tgt[:, :4])
                pred_counts = pred[:, :4].sum(dim=(2, 3)) / 100.0
                l_cnt = F.l1_loss(pred_counts, counts)
                l_heat = F.mse_loss(pred[:, 4], tgt[:, 4])
                mask = (tgt[:, 5] > 0.5).float()
                msum = mask.sum() + 1e-6
                l_a = (torch.abs(pred[:, 5] - tgt[:, 5]) * mask).sum() / msum
                l_e = (torch.abs(pred[:, 6] - tgt[:, 6]) * mask).sum() / msum
                loss = l_dens + 0.01 * l_cnt + l_heat + 0.5 * (l_a + l_e)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg["dens"] += l_dens.item(); agg["cnt"] += l_cnt.item()
            agg["heat"] += l_heat.item(); agg["attr"] += (l_a.item() + l_e.item())
            agg["n"] += 1
        n = max(agg["n"], 1)
        msg = (f"ep {ep:02d} dens {agg['dens']/n:.4f} cnt {agg['cnt']/n:.3f} "
               f"heat {agg['heat']/n:.4f} attr {agg['attr']/n:.3f} "
               f"lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/60:.1f}m")
        if ep % args.val_every == args.val_every - 1 or ep == args.epochs - 1:
            mean_tau, per, alts, _ = val_metric(model, images, va_ids, targets, device,
                                               train_stats, tta=1)
            msg += f" | VAL mean_tau {mean_tau:.4f} " + \
                   " ".join(f"{c[:4]}={per[c]:.3f}" for c in COLS) + \
                   f" altss={alts['size_spread']:.3f} alte={alts['elongation']:.3f}"
            hist.append({"epoch": ep, "mean_tau": mean_tau, **per,
                         "alt_ss": alts["size_spread"], "alt_el": alts["elongation"]})
            if mean_tau > best:
                best = mean_tau
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "mean_tau": mean_tau, "per": per}, out / "best.pt")
                msg += " *"
        print(msg, flush=True)

    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"BEST val mean_tau {best:.4f}  ({(time.time()-t0)/60:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
