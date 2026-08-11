"""Base training run for the Ch2 improvement campaign.

Reproduces the reference recipe's model A training EXACTLY (same RNG consumption
as solution.train_one), additionally tracks the top-2 validation checkpoints, and
dumps per-view D4 TTA raw output maps for the selection-val tiles and the
pseudo-test tiles at BOTH checkpoints so every decode-side experiment can be run
offline later without re-running the network.

Usage:
  python exp_base.py --data <dataset_dir> --out <out_dir> [--epochs 20] [--smoke]
      [--train-seed 43] [--w-attr 0.5] [--w-heat 1.0] [--dump-best-only]

The optional knobs all default to the reference behaviour: --train-seed defaults
to solution.SEED (the tr/va split is ALWAYS split(solution.SEED) so every run
shares identical ids), --w-attr/--w-heat default to the reference loss weights,
and --dump-best-only only skips the second-checkpoint map dump (ckpt_second.pt
is still written).

All files (solution.py, metric.py, exp_base.py, tta8_geom_test.py) live flat in
the same directory.
"""
import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# flat-layout import: solution.py + metric.py sit next to this file
sys.path.insert(0, str(Path(__file__).parent))
import solution as S  # noqa: E402
from metric import morphometry_score, COLS  # noqa: E402

T0 = time.time()


def em():
    """Elapsed minutes since script start."""
    return (time.time() - T0) / 60.0


def P(msg):
    print(f"{msg}  [{em():.1f}m]", flush=True)


# ----------------------------------------------------------------------------
# D4 TTA view dumping
# ----------------------------------------------------------------------------
# Views 0-3 are exactly solution.predict_maps' flip views, in the same order:
#   0 identity, 1 flip-W ("h"), 2 flip-H ("v"), 3 flip-HW ("hv")
# Views 4-7 compose a rot90 on top of views 0-3:
#   input  = R(F(x))   with R = torch.rot90(., 1, dims=[2, 3])
#   output canonicalization = F_inv(R_inv(y)), R_inv = rot90(., -1) on the map
#   spatial axes (1, 2) of the (7, h, w) numpy output.
#
# DOWNSTREAM RECONSTRUCTION CONTRACT
# ----------------------------------
#   v    = np.load(f"{iid}.npz")["v"]              # (8, 7, mh, mw) float16
#   m    = v[sel].astype(np.float32).mean(axis=0)  # sel = list of view indices
#   dens = np.maximum(m[:4], 0.0) / S.DENS_SCALE
#   heat, amap, emap = m[4], m[5], m[6]
# Averaging saved views [0] / [0,1] / [0,1,2,3] reproduces
# S.predict_maps(model, img, tta=1 / 2 / 4) up to float16 rounding.
# Stored values are the RAW head outputs (7 channels, no clipping, no
# DENS_SCALE division), cropped to [:mh, :mw] AFTER canonicalization.
_FLIP_TAGS = [None, "h", "v", "hv"]


def _flip_in(x, tag):
    if tag == "h":
        return torch.flip(x, [3])
    if tag == "v":
        return torch.flip(x, [2])
    if tag == "hv":
        return torch.flip(x, [2, 3])
    return x


def _flip_out(out, tag):
    # out: (7, h, w) numpy, canonicalization inverse of _flip_in
    if tag == "h":
        return out[:, :, ::-1]
    if tag == "v":
        return out[:, ::-1, :]
    if tag == "hv":
        return out[:, ::-1, ::-1]
    return out


@torch.no_grad()
def dump_views(model, img):
    """Return (8, 7, mh, mw) float16 array of canonicalized raw head outputs."""
    H, W = img.shape[:2]
    mh, mw = S.map_shape(H, W)
    x = S._prep(img).to(S.DEVICE)
    views = np.empty((8, 7, mh, mw), np.float16)
    for i, tag in enumerate(_FLIP_TAGS):
        out = model(_flip_in(x, tag))[0].float().cpu().numpy()
        out = _flip_out(out, tag)
        views[i] = out[:, :mh, :mw].astype(np.float16)
    for i, tag in enumerate(_FLIP_TAGS):
        xr = torch.rot90(_flip_in(x, tag), 1, dims=[2, 3])
        out = model(xr)[0].float().cpu().numpy()      # (7, Wp/2, Hp/2)
        out = np.rot90(out, -1, axes=(1, 2))          # (7, Hp/2, Wp/2)
        out = _flip_out(out, tag)
        views[4 + i] = out[:, :mh, :mw].astype(np.float16)
    return views


def maps_from_views(v, sel):
    """Reconstruct (dens, heat, amap, emap) from saved views (see contract)."""
    m = np.asarray(v)[list(sel)].astype(np.float32).mean(axis=0)
    dens = np.maximum(m[:4], 0.0) / S.DENS_SCALE
    return dens, m[4], m[5], m[6]


# ----------------------------------------------------------------------------
# training (verbatim S.train_one + top-2 checkpoint tracking + val history)
# ----------------------------------------------------------------------------
def train_top2(images, cells_by, tr_ids, va_ids, targets, train_stats, seed,
               stop_min, epochs, tag="A", w_attr=0.5, w_heat=1.0):
    """VERBATIM copy of solution.train_one, plus:
      - keeps the top-2 validation checkpoints (deepcopy only; no RNG use)
      - records the full validation history
      - loss weights w_heat / w_attr are configurable; the defaults
        (1.0 / 0.5) reproduce the reference expression exactly.
    Nothing that consumes RNG is changed, so the returned best model is
    bit-comparable with what S.train_one would produce for the same inputs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = S.CropDataset(images, cells_by, tr_ids, S.CPI, seed)
    workers = 2 if (os.name == "posix") else 0
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=S.BATCH, shuffle=True, num_workers=workers,
                    pin_memory=False, drop_last=True, generator=gen,
                    persistent_workers=(workers > 0))
    model = S.UNetR18(pretrained=bool(S.PRETRAINED)).to(S.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=S.LR, weight_decay=S.WD)
    iters = max(1, len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * iters,
                                                       eta_min=S.LR / 10)
    gt_val = targets.set_index("image_id").loc[va_ids]

    best_tau, best_state = -1.0, None
    keep = []       # [(mean_tau, epoch, state_dict)] pruned to the top 2
    history = []    # every validated epoch: {"epoch", "mean_tau", "per": {...}}
    ep_t = time.time()
    for ep in range(epochs):
        if S.elapsed_min() > stop_min:
            print(f"[guard] model {tag}: stop training at epoch {ep} "
                  f"({S.elapsed_min():.1f} min)", flush=True)
            break
        ds.set_epoch(ep)
        model.train()
        for x, tgt, counts in dl:
            x, tgt, counts = x.to(S.DEVICE), tgt.to(S.DEVICE), counts.to(S.DEVICE)
            pred = model(x)
            l_dens = F.mse_loss(pred[:, :4], tgt[:, :4])
            l_cnt = F.l1_loss(pred[:, :4].sum(dim=(2, 3)) / S.DENS_SCALE, counts)
            l_heat = F.mse_loss(pred[:, 4], tgt[:, 4])
            mask = (tgt[:, 5] > 0.5).float()
            msum = mask.sum() + 1e-6
            l_a = (torch.abs(pred[:, 5] - tgt[:, 5]) * mask).sum() / msum
            l_e = (torch.abs(pred[:, 6] - tgt[:, 6]) * mask).sum() / msum
            loss = l_dens + 0.01 * l_cnt + w_heat * l_heat + w_attr * (l_a + l_e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        do_val = (ep % S.VAL_EVERY == S.VAL_EVERY - 1) or (ep == epochs - 1) \
            or (S.elapsed_min() > stop_min * 0.92)
        msg = f"[{tag}] ep {ep:02d} loss {loss.item():.4f} " \
              f"({(time.time()-ep_t):.0f}s/ep, {em():.1f}m)"
        ep_t = time.time()
        if do_val:
            model.eval()
            rows = []
            for iid in va_ids:
                d = S.predict_tile_multi([model], images[iid], train_stats, tta=1)
                d["image_id"] = iid
                rows.append(d)
            pred = pd.DataFrame(rows).set_index("image_id").loc[va_ids]
            mean_tau, per = S.morph_score(pred, gt_val)
            msg += f" VAL {mean_tau:.4f} " + " ".join(f"{c[:4]}={per[c]:.3f}" for c in COLS)
            # --- addition: track every validated checkpoint (deepcopy, no RNG) ---
            st = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            keep.append((float(mean_tau), int(ep), st))
            keep.sort(key=lambda t: -t[0])
            keep = keep[:2]          # bounded memory; identical top-2 outcome
            history.append({"epoch": int(ep), "mean_tau": float(mean_tau),
                            "per": {c: float(per[c]) for c in COLS}})
            # --- end addition ---
            if mean_tau > best_tau:
                best_tau = mean_tau
                best_state = deepcopy({k: v.detach().cpu() for k, v in
                                       model.state_dict().items()})
                msg += " *"
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"[{tag}] best val mean_tau {best_tau:.4f}", flush=True)
    return model, best_tau, keep, history


# ----------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch intra-op threads (0 = leave solution.py's pin)")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="training seed (default: solution.SEED); the tr/va "
                         "split always stays at solution.SEED")
    ap.add_argument("--w-attr", type=float, default=0.5,
                    help="weight on the (area + elongation) L1 terms")
    ap.add_argument("--w-heat", type=float, default=1.0,
                    help="weight on the heat MSE term")
    ap.add_argument("--dump-best-only", action="store_true",
                    help="dump maps for the best checkpoint only (halves disk);"
                         " ckpt_second.pt is still written")
    args = ap.parse_args()

    # solution.py pins torch to min(10, ncores) at import; experiment boxes are wider.
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    train_seed = S.SEED if args.train_seed is None else args.train_seed

    DATA = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "maps_best").mkdir(exist_ok=True)
    if not args.dump_best_only:
        (out / "maps_second").mkdir(exist_ok=True)
    epochs = 2 if args.smoke else args.epochs

    P(f"exp_base data={DATA} out={out} epochs={epochs} smoke={args.smoke}")
    P(f"train_seed={train_seed} (split_seed={S.SEED}) w_attr={args.w_attr} "
      f"w_heat={args.w_heat} dump_best_only={bool(args.dump_best_only)}")
    P(f"device={S.DEVICE} threads={torch.get_num_threads()} torch={torch.__version__}")

    # ---------------- data (mirrors sim_eval.py exactly) ----------------
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

    P("loading images...")
    images = {}
    for iid in ids_all:
        with Image.open(DATA / "train_images" / f"{iid}.png") as im:
            images[iid] = np.asarray(im.convert("RGB"))
    P(f"loaded {len(images)} train images")

    from sklearn.model_selection import GroupShuffleSplit
    groups_all = index.set_index("image_id").loc[ids_all, "group"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.28, random_state=0)
    inner_idx, ptest_idx = next(gss.split(ids_all, groups=groups_all))
    inner_ids = [ids_all[i] for i in inner_idx]
    ptest_ids = [ids_all[i] for i in ptest_idx]
    print(f"inner {len(inner_ids)} tiles / pseudo-test {len(ptest_ids)}", flush=True)

    groups_inner = index.set_index("image_id").loc[inner_ids, "group"].values

    def split(seed):
        g = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        tr, va = next(g.split(inner_ids, groups=groups_inner))
        return [inner_ids[i] for i in tr], [inner_ids[i] for i in va]

    tr_a, va_a = split(S.SEED)
    if args.smoke:
        tr_a, va_a, ptest_ids = tr_a[:60], va_a[:16], ptest_ids[:12]
    P(f"train {len(tr_a)} / val {len(va_a)} / ptest {len(ptest_ids)}")

    tr_tgt = targets.set_index("image_id").loc[tr_a]
    train_stats = {"dispersion_mean": float(tr_tgt["dispersion"].mean()),
                   "size_spread_med": float(tr_tgt["size_spread"].median()),
                   "elongation_med": float(tr_tgt["elongation"].median())}
    P(f"train_stats {train_stats}")

    # ---------------- train ----------------
    model_best, tau_a, keep, history = train_top2(
        images, cells_by, tr_a, va_a, targets, train_stats, train_seed,
        stop_min=10000, epochs=epochs, tag="A",
        w_attr=args.w_attr, w_heat=args.w_heat)

    if len(keep) == 0:
        print("[FATAL] no validated epoch", flush=True)
        sys.exit(1)
    b_tau, b_ep, b_state = keep[0]
    if len(keep) > 1:
        s_tau, s_ep, s_state = keep[1]
    else:
        print("[warn] only one validated epoch: second == best", flush=True)
        s_tau, s_ep, s_state = b_tau, b_ep, b_state
    torch.save({"model": b_state, "epoch": b_ep, "mean_tau": b_tau}, out / "ckpt_best.pt")
    torch.save({"model": s_state, "epoch": s_ep, "mean_tau": s_tau}, out / "ckpt_second.pt")
    P(f"ckpt best ep{b_ep} tau {b_tau:.4f} | second ep{s_ep} tau {s_tau:.4f}")

    meta = {"va_ids": _jsonable(va_a), "ptest_ids": _jsonable(ptest_ids),
            "train_stats": train_stats, "val_history": history,
            "best": {"epoch": b_ep, "mean_tau": b_tau},
            "second": {"epoch": s_ep, "mean_tau": s_tau},
            "epochs": epochs, "smoke": bool(args.smoke),
            "train_seed": int(train_seed), "w_attr": float(args.w_attr),
            "w_heat": float(args.w_heat),
            "dump_best_only": bool(args.dump_best_only)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    model_second = deepcopy(model_best)
    model_second.load_state_dict(s_state)
    model_second.eval()
    model_best.eval()

    # ---------------- dump D4 views ----------------
    dump_ids = list(va_a) + list(ptest_ids)
    dump_specs = [(model_best, "maps_best")]
    if not args.dump_best_only:
        dump_specs.append((model_second, "maps_second"))
    else:
        P("dump_best_only: skipping maps_second")
    for mdl, sub in dump_specs:
        P(f"dumping {len(dump_ids)} tiles -> {sub}")
        t_d = time.time()
        for i, iid in enumerate(dump_ids):
            v = dump_views(mdl, images[iid])
            np.savez_compressed(out / sub / f"{iid}.npz", v=v)
            if (i + 1) % 50 == 0 or (i + 1) == len(dump_ids):
                P(f"  {sub} {i+1}/{len(dump_ids)} "
                  f"({(time.time()-t_d)/60:.1f}m in dump)")

    # ---------------- gate (a): parity vs S.predict_maps ----------------
    # Per-channel tolerance.  These are float16 STORAGE-quantization budgets,
    # not geometry budgets: dens/heat sit on a small numeric range, while amap
    # carries log-area (~0..7.6) where fp16 eps alone is ~4e-3.  A genuine view
    # or geometry bug produces O(1) error and still trips any of these.
    GATE_TOL = {"dens": 3e-3, "heat": 3e-3, "amap": 8e-3, "emap": 8e-3}
    CHANS = ["dens", "heat", "amap", "emap"]
    P("=== GATE A: view-reconstruction parity ===")
    gate_ids = list(va_a[:8]) + list(ptest_ids[:8])
    sels = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}
    ok_all = True
    for tta, sel in sels.items():
        worst = {c: 0.0 for c in CHANS}
        where = {c: "" for c in CHANS}
        for iid in gate_ids:
            v = np.load(out / "maps_best" / f"{iid}.npz")["v"]
            r = maps_from_views(v, sel)
            ref = S.predict_maps(model_best, images[iid], tta=tta)
            for name, a, b in zip(CHANS, r, ref):
                d = float(np.max(np.abs(np.asarray(a, np.float64) -
                                        np.asarray(b, np.float64))))
                if d > worst[name]:
                    worst[name], where[name] = d, iid
        ok = all(worst[c] < GATE_TOL[c] for c in CHANS)
        ok_all = ok_all and ok
        detail = " ".join(f"{c}={worst[c]:.3e}/{GATE_TOL[c]:g}({where[c]})"
                          for c in CHANS)
        print(f"[gate] tta={tta} views={sel} {detail} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok_all:
        print("[gate] PARITY FAIL", flush=True)
        sys.exit(1)
    P("GATE A PASS")

    # ---------------- gate (b): reference score from saved maps ----------------
    P("=== GATE B: reference sim_eval score from saved best maps ===")
    va_cache = {}
    for iid in va_a:
        v = np.load(out / "maps_best" / f"{iid}.npz")["v"]
        va_cache[iid] = (maps_from_views(v, [0]), images[iid].shape[:2])
    gt_val = targets.set_index("image_id").loc[va_a]

    def decode_val(cthr):
        rows = []
        for iid in va_a:
            (dens, heat, amap, emap), (H, W) = va_cache[iid]
            d = S.decode_tile(dens, heat, amap, emap, H, W, train_stats, cthr=cthr)
            d["image_id"] = iid
            rows.append(d)
        return pd.DataFrame(rows).set_index("image_id").loc[va_a]

    best_cthr, best_ct_tau, va_pred = S.DENS_THR, -1.0, None
    for cthr in [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3]:
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

    rows = []
    for iid in ptest_ids:
        v = np.load(out / "maps_best" / f"{iid}.npz")["v"]
        dens, heat, amap, emap = maps_from_views(v, [0, 1, 2, 3])
        H, W = images[iid].shape[:2]
        d = S.decode_tile(dens, heat, amap, emap, H, W, train_stats, cthr=best_cthr)
        rec = {"image_id": iid}
        for c in COLS:
            rec[c] = d[f"{c}_alt"] if use_alt.get(c, False) else d[c]
        rows.append(rec)
    pred = pd.DataFrame(rows).set_index("image_id").loc[ptest_ids]
    gt_p = targets.set_index("image_id").loc[ptest_ids]
    mean_tau, per = morphometry_score(pred, gt_p)
    print(f"[REF] ptest A_tta4 mean={mean_tau:.4f} cell={per['cellularity']:.4f} "
          f"tumor={per['tumor_frac']:.4f} size={per['size_spread']:.4f} "
          f"elong={per['elongation']:.4f} disp={per['dispersion']:.4f}", flush=True)
    pred.to_csv(out / "ref_ptest_pred_A_tta4.csv")

    meta["ref_tta4"] = {"mean": float(mean_tau), **{k: float(v) for k, v in per.items()},
                        "cthr": float(best_cthr), "use_alt": use_alt,
                        "val_cthr_tau": float(best_ct_tau)}
    (out / "meta.json").write_text(json.dumps(_jsonable(meta), indent=1))

    print(f"EXP_BASE DONE {em():.1f}", flush=True)


if __name__ == "__main__":
    main()
