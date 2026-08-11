"""Offline decode-variant sweep over dumped D4-TTA map stacks (no model, no torch).

Consumes a RUN dir produced by the map-dump job:
  RUN/meta.json          {"va_ids", "ptest_ids", "train_stats", "sizes"?, ...}
  RUN/maps_best/{iid}.npz    key "v": float16 (8, 7, mh, mw) RAW head outputs
  RUN/maps_second/{iid}.npz  same, second checkpoint (optional)

View order (already canonicalized to the identity orientation):
  [0..3] = identity, hflip, vflip, hvflip      -> averaging = solution.py tta1/2/4
  [4..7] = the same four composed with rot90   -> [0..7] = new TTA8

Reconstruction contract (mirrors solution.predict_maps):
  m    = maps[views].astype(float32).mean(0)
  dens = max(m[:4], 0) / DENS_SCALE ; heat = m[4] ; amap = m[5] ; emap = m[6]

decode_variant() is a generalized re-implementation of solution.decode_tile: with
DEFAULT_CFG it reproduces it bit for bit (see test_decode_sweep_synth.py).

Usage:
  python decode_sweep.py --run RUN_DIR --data DATASET_DIR --out RESULTS_JSON [--quick]
  python decode_sweep.py --run RUN_DIR --targets-csv T.csv --out R.json [--quick]

Pure numpy + scipy.ndimage; no RNG anywhere, fully deterministic.
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import morphometry_score, COLS  # noqa: E402

# ----------------------------- constants (mirror solution.py) -----------------------------
STRIDE = 2
DENS_SCALE = 100.0
ALT_COLS = ["size_spread", "elongation"]
OUT_KEYS = ["cellularity", "tumor_frac", "dispersion", "size_spread", "elongation",
            "size_spread_alt", "elongation_alt"]

# protocol grid (solution.py main(); sim_eval.py uses a 4-point prefix of this)
CTHR_GRID = [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3]
CTHR_GRID_QUICK = [1e-4, 3e-4, 8e-4]
CTHR_TABLE = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3, 1.6e-3, 2e-3]
CTHR_TABLE_QUICK = [1e-4, 3e-4, 8e-4, 2e-3]

TTA1, TTA2, TTA4 = (0,), (0, 1), (0, 1, 2, 3)
TTA8 = (0, 1, 2, 3, 4, 5, 6, 7)

DEFAULT_CFG = {
    "views": TTA4,        # view subset to average
    "cthr": 3e-4,         # density noise floor (solution.DENS_THR)
    "nms_radius": 1,      # 1 -> 3x3 NMS (current), 2 -> 5x5
    "peak_thr": 0.05,     # peak score cutoff
    "k_mult": 1.0,        # k = max(5, round(k_mult * n_hat))
    "readout_r": 1,       # attribute readout window radius (1 -> _sample3)
    "disp_est": "peaks",  # "peaks" | "soft" | "quadrat"
    "soft_sigma": 0.0,    # extra gaussian smoothing (map px) for "soft"
    "quadrat_g": 16,      # block size (map px) for "quadrat"
    "alt_mask_thr": 0.5,  # amap threshold for the *_alt estimators
}
RECON_GROUPS = 4  # (source, views) reconstruction caches held in RAM at once


def make_cfg(**kw):
    cfg = dict(DEFAULT_CFG)
    for k, v in kw.items():
        if k not in DEFAULT_CFG:
            raise KeyError(f"unknown cfg key {k}")
        cfg[k] = v
    cfg["views"] = tuple(int(v) for v in cfg["views"])
    return cfg


def cfg_key(cfg):
    return tuple(sorted(cfg.items()))


def clipped_tau(a, b):
    """Identical to solution.clipped_tau."""
    t = kendalltau(np.asarray(a, float), np.asarray(b, float)).statistic
    return float(np.clip(0.0 if (t is None or np.isnan(t)) else t, 0.0, 1.0))


def map_shape(H, W):
    return (H + STRIDE - 1) // STRIDE, (W + STRIDE - 1) // STRIDE


# ----------------------------- generalized decode primitives -----------------------------
def nms_peaks_r(heat, k, r=1):
    """r=1 is byte-identical to solution.nms_peaks (padded-stack local-max trick)."""
    H, W = heat.shape
    n = 2 * r + 1
    p = np.pad(heat, r, mode="constant", constant_values=-1)
    stack = np.stack([p[dy:dy + H, dx:dx + W] for dy in range(n) for dx in range(n)])
    is_max = heat >= stack.max(axis=0) - 1e-9
    ys, xs = np.nonzero(is_max)
    scores = heat[ys, xs]
    order = np.argsort(-scores)[:k]
    return ys[order], xs[order], scores[order]


def sample_r(m, ys, xs, r=1):
    """r=1 is byte-identical to solution._sample3 (float32 accumulation kept)."""
    H, W = m.shape
    out = np.empty(len(ys), np.float32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        out[i] = m[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)].mean()
    return out


def _disp_peaks(ys, xs, kk, H, W, ts):
    """Clark-Evans R from NMS peak coords, in pixel units (solution.py behavior)."""
    if kk >= 5:
        pts = np.column_stack([ys, xs]).astype(np.float64) * STRIDE
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn_d = np.sqrt(d2.min(axis=1))
        return float(nn_d.mean() * 2.0 * np.sqrt(kk / (W * H)))
    return float(ts["dispersion_mean"])


def _disp_soft(dtot, n_hat, H, W, ts, sigma):
    """Density-weighted Clark-Evans surrogate: no peak detection at all.

    lam(x) = smoothed total density / STRIDE^2   [cells per px^2]
    local CSR expected NN distance = 0.5 / sqrt(lam)
    softNN = sum(d * 0.5/sqrt(lam)) / sum(d)     [d = thresholded total density]
    R_soft = softNN * 2 * sqrt(N_hat / (W*H)),   clipped to [0, 2.5]
    """
    tot = float(dtot.sum())
    if not np.isfinite(tot) or tot <= 1e-12:
        return float(ts["dispersion_mean"])
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        lam_src = gaussian_filter(dtot.astype(np.float64), float(sigma), mode="nearest")
    else:
        lam_src = dtot.astype(np.float64)
    lam = np.maximum(lam_src, 0.0) / float(STRIDE * STRIDE)
    m = dtot > 0
    if not m.any():
        return float(ts["dispersion_mean"])
    d = dtot[m].astype(np.float64)
    e_nn = 0.5 / np.sqrt(np.maximum(lam[m], 1e-12))
    ws = d.sum()
    if not np.isfinite(ws) or ws <= 1e-12:
        return float(ts["dispersion_mean"])
    soft_nn = float((d * e_nn).sum() / ws)
    r = soft_nn * 2.0 * np.sqrt(max(n_hat, 0.0) / (W * H))
    if not np.isfinite(r):
        return float(ts["dispersion_mean"])
    return float(np.clip(r, 0.0, 2.5))


def _disp_quadrat(dtot, ts, g):
    """Variance-to-mean ratio of density mass in g x g map-px blocks (edge blocks
    dropped). R_q = 1/sqrt(VMR): monotone in the same direction as Clark-Evans R
    (clustered -> VMR>1 -> R_q<1), which is all Kendall tau needs."""
    g = int(g)
    mh, mw = dtot.shape
    nby, nbx = mh // g, mw // g
    if nby * nbx < 4:
        return float(ts["dispersion_mean"])
    blk = dtot[:nby * g, :nbx * g].astype(np.float64)
    c = blk.reshape(nby, g, nbx, g).sum(axis=(1, 3)).ravel()
    mean = float(c.mean())
    if not np.isfinite(mean) or mean < 1e-9:
        return float(ts["dispersion_mean"])
    vmr = float(c.var(ddof=0)) / mean
    if not np.isfinite(vmr):
        return float(ts["dispersion_mean"])
    return float(1.0 / np.sqrt(max(vmr, 1e-12)))


def decode_maps(dens, heat, amap, emap, H, W, train_stats, cfg=None):
    """Generalized solution.decode_tile. With DEFAULT_CFG: bit-identical output."""
    c = DEFAULT_CFG if cfg is None else cfg
    dens = np.where(dens > c["cthr"], dens, 0.0)
    per_class = dens.reshape(4, -1).sum(axis=1)
    n_hat = float(per_class.sum())
    out = {}
    out["cellularity"] = 1e5 * n_hat / (W * H)
    out["tumor_frac"] = float(per_class[0] / (n_hat + 1e-6))

    # fail-closed peak selection: only confident peaks count; if too few,
    # the per-target fallbacks below use train statistics instead of junk.
    k = int(max(5, round(c["k_mult"] * n_hat)))
    ys, xs, scores = nms_peaks_r(heat, k, c["nms_radius"])
    keep = scores > c["peak_thr"]
    ys, xs, scores = ys[keep], xs[keep], scores[keep]
    kk = len(ys)

    ts = train_stats
    dtot = dens.sum(axis=0)
    de = c["disp_est"]
    if de == "peaks":
        out["dispersion"] = _disp_peaks(ys, xs, kk, H, W, ts)
    elif de == "soft":
        out["dispersion"] = _disp_soft(dtot, n_hat, H, W, ts, c["soft_sigma"])
    elif de == "quadrat":
        out["dispersion"] = _disp_quadrat(dtot, ts, c["quadrat_g"])
    else:
        raise ValueError(f"unknown disp_est {de!r}")

    rr = c["readout_r"]
    if kk >= 3:
        a_pk = np.exp(sample_r(amap, ys, xs, rr).astype(np.float64))
        m = a_pk.mean()
        out["size_spread"] = float(a_pk.std(ddof=0) / m) if m > 0 else ts["size_spread_med"]
        out["elongation"] = float(sample_r(emap, ys, xs, rr).mean())
    else:
        out["size_spread"] = ts["size_spread_med"]
        out["elongation"] = ts["elongation_med"]

    mask = amap > c["alt_mask_thr"]
    if mask.sum() > 4 and dtot[mask].sum() > 1e-6:
        w = dtot[mask]
        a = np.exp(amap[mask].astype(np.float64))
        e1 = float((w * a).sum() / w.sum())
        e2 = float((w * a * a).sum() / w.sum())
        var = max(e2 - e1 * e1, 0.0)
        out["size_spread_alt"] = float(np.sqrt(var) / e1) if e1 > 0 else out["size_spread"]
        we = emap[mask].astype(np.float64)
        out["elongation_alt"] = float((w * we).sum() / w.sum())
    else:
        out["size_spread_alt"] = out["size_spread"]
        out["elongation_alt"] = out["elongation"]
    return out


def reconstruct(maps_arr, views):
    """(8,7,mh,mw) raw head stack + view subset -> (dens, heat, amap, emap) float32."""
    m = np.asarray(maps_arr)[list(views)].astype(np.float32).mean(axis=0)
    return np.maximum(m[:4], 0.0) / DENS_SCALE, m[4], m[5], m[6]


def decode_variant(maps_arr, H, W, train_stats, cfg=None):
    """Reconstruct the chosen views then decode. cfg=None/{} -> current behavior.
    cfg may be partial (missing keys fall back to DEFAULT_CFG)."""
    c = make_cfg(**cfg) if cfg else DEFAULT_CFG
    dens, heat, amap, emap = reconstruct(maps_arr, c["views"])
    return decode_maps(dens, heat, amap, emap, H, W, train_stats, c)


# ----------------------------- run loading -----------------------------
def load_sizes(meta, ids, data_dir):
    """meta['sizes'] if present, else PNG *headers* only (PIL does not decode on open)."""
    ms = meta.get("sizes") or {}
    sizes, missing = {}, []
    for iid in ids:
        hw = ms.get(iid, ms.get(str(iid)))  # json object keys are always strings
        if hw is not None:
            sizes[iid] = (int(hw[0]), int(hw[1]))
        else:
            missing.append(iid)
    if missing:
        if data_dir is None:
            raise SystemExit(f"meta.json lacks 'sizes' for {len(missing)} ids and no "
                             f"--data given (e.g. {missing[:3]})")
        from PIL import Image
        print(f"[sizes] meta lacks {len(missing)} ids -> PNG header fallback", flush=True)
        for iid in missing:
            p = None
            for sub in ("train_images", "test_images"):
                q = Path(data_dir) / sub / f"{iid}.png"
                if q.exists():
                    p = q
                    break
            if p is None:
                raise SystemExit(f"no PNG found for {iid} under {data_dir}")
            with Image.open(p) as im:
                w, h = im.size  # header-only read
            sizes[iid] = (int(h), int(w))
    return sizes


def load_maps(mdir, ids):
    out = {}
    for iid in ids:
        f = Path(mdir) / f"{iid}.npz"
        with np.load(f) as z:
            key = "v" if "v" in z.files else z.files[0]
            a = np.asarray(z[key], dtype=np.float16)
        if a.ndim != 4 or a.shape[0] != 8 or a.shape[1] != 7:
            raise SystemExit(f"{f}: expected (8,7,mh,mw) got {a.shape}")
        out[iid] = a
    return out


def build_ctx(args):
    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text())
    va_ids = list(meta["va_ids"])
    pt_ids = list(meta["ptest_ids"])
    if args.quick:
        va_ids, pt_ids = va_ids[:40], pt_ids[:40]
    ids = va_ids + pt_ids
    ts = {k: float(v) for k, v in meta["train_stats"].items()}

    tcsv = Path(args.targets_csv) if args.targets_csv else Path(args.data) / "train_targets.csv"
    targets = pd.read_csv(tcsv).set_index("image_id")
    miss = [i for i in ids if i not in targets.index]
    if miss:
        raise SystemExit(f"{len(miss)} ids missing from {tcsv} (e.g. {miss[:3]})")

    sizes = load_sizes(meta, ids, args.data)

    t0 = time.time()
    mb = load_maps(run / "maps_best", ids)
    ms = None
    if (run / "maps_second").is_dir():
        try:
            ms = load_maps(run / "maps_second", ids)
        except Exception as e:  # tolerate a partial second dump
            print(f"[load] maps_second unusable ({e}) -> snapshot variants skipped", flush=True)
            ms = None
    else:
        print("[load] no maps_second/ -> snapshot variants skipped", flush=True)
    nbytes = sum(a.nbytes for a in mb.values()) + (sum(a.nbytes for a in ms.values()) if ms else 0)
    print(f"[load] {len(ids)} tiles ({len(va_ids)} va / {len(pt_ids)} ptest), "
          f"{nbytes/2**20:.0f} MiB fp16, {time.time()-t0:.1f}s", flush=True)

    bad = [iid for iid in ids if tuple(mb[iid].shape[2:]) != map_shape(*sizes[iid])]
    if bad:
        print(f"[warn] {len(bad)} tiles: map shape != map_shape(H,W) from sizes "
              f"(e.g. {bad[0]} {mb[bad[0]].shape[2:]} vs {map_shape(*sizes[bad[0]])})", flush=True)

    return {"meta": meta, "va_ids": va_ids, "ptest_ids": pt_ids, "ts": ts, "sizes": sizes,
            "gt_va": targets.loc[va_ids], "gt_pt": targets.loc[pt_ids],
            "maps": {"best": mb, "second": ms},
            "cthr_grid": CTHR_GRID_QUICK if args.quick else CTHR_GRID,
            "cthr_table": CTHR_TABLE_QUICK if args.quick else CTHR_TABLE,
            "recon": OrderedDict(), "sel_memo": {}, "res_memo": {}, "n_decode": [0]}


# ----------------------------- cached reconstruction + decode -----------------------------
def _group(ctx, key):
    cache = ctx["recon"]
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    g = {}
    cache[key] = g
    while len(cache) > RECON_GROUPS:
        cache.popitem(last=False)
    return g


def get_recon(ctx, iid, source, views):
    g = _group(ctx, (source, views))
    m = g.get(iid)
    if m is None:
        vi = list(views)
        if source == "avg":
            a = ctx["maps"]["best"][iid][vi].astype(np.float32).mean(axis=0)
            b = ctx["maps"]["second"][iid][vi].astype(np.float32).mean(axis=0)
            mm = (a + b) / 2.0
        else:
            mm = ctx["maps"][source][iid][vi].astype(np.float32).mean(axis=0)
        m = (np.maximum(mm[:4], 0.0) / DENS_SCALE, mm[4], mm[5], mm[6])
        g[iid] = m
    return m


def decode_one(ctx, iid, source, cfg):
    dens, heat, amap, emap = get_recon(ctx, iid, source, cfg["views"])
    H, W = ctx["sizes"][iid]
    ctx["n_decode"][0] += 1
    return decode_maps(dens, heat, amap, emap, H, W, ctx["ts"], cfg)


def predict_set(ctx, ids, cfg, snapshot):
    if snapshot == "best":
        return [decode_one(ctx, i, "best", cfg) for i in ids]
    if snapshot == "avg_maps":
        return [decode_one(ctx, i, "avg", cfg) for i in ids]
    if snapshot == "avg_decode":
        # mirrors solution.predict_tile_multi: average every raw key, then select
        rows = []
        for i in ids:
            d1 = decode_one(ctx, i, "best", cfg)
            d2 = decode_one(ctx, i, "second", cfg)
            rows.append({k: float(np.mean([d1[k], d2[k]])) for k in d1})
        return rows
    raise ValueError(snapshot)


def _full_df(rows, ids):
    return pd.DataFrame(rows, index=pd.Index(ids, name="image_id"))


def _sel_df(rows, ids, use_alt):
    recs = [{c: d[f"{c}_alt"] if use_alt.get(c, False) else d[c] for c in COLS} for d in rows]
    return pd.DataFrame(recs, index=pd.Index(ids, name="image_id"))


def _n_nonfinite(rows):
    return int(sum(1 for d in rows for k in OUT_KEYS if not np.isfinite(d[k])))


# ----------------------------- selection protocol -----------------------------
def select_on_va(ctx, cfg):
    """Mirrors sim_eval.py / solution.py main(): cthr grid then per-target alt choice,
    both on va with maps_best and views=(0,) only (the tta1 selection pass)."""
    scfg = dict(cfg)
    scfg["views"] = TTA1
    scfg["cthr"] = None
    key = cfg_key(scfg)
    if key in ctx["sel_memo"]:
        return ctx["sel_memo"][key]

    best_cthr, best_tau, va_pred = None, -1.0, None
    table = []
    for cthr in ctx["cthr_grid"]:
        c2 = dict(scfg)
        c2["cthr"] = cthr
        rows = predict_set(ctx, ctx["va_ids"], c2, "best")
        p = _full_df(rows, ctx["va_ids"])
        mt, _ = morphometry_score(p, ctx["gt_va"])
        table.append({"cthr": cthr, "va_mean": mt})
        if mt > best_tau:  # strict >: first best wins (matches solution.py)
            best_cthr, best_tau, va_pred = cthr, mt, p

    use_alt, alt_taus = {}, {}
    for c in ALT_COLS:
        t_pri = clipped_tau(va_pred[c].values, ctx["gt_va"][c].values)
        t_alt = clipped_tau(va_pred[f"{c}_alt"].values, ctx["gt_va"][c].values)
        use_alt[c] = bool(t_alt > t_pri)
        alt_taus[c] = {"primary": t_pri, "alt": t_alt}
    res = {"best_cthr": best_cthr, "sel_va_mean": best_tau, "use_alt": use_alt,
           "alt_taus": alt_taus, "cthr_grid_va": table}
    ctx["sel_memo"][key] = res
    return res


def run_protocol(ctx, cfg, snapshot):
    """Full protocol for one decode config. Memoized on (cfg, snapshot)."""
    mkey = (cfg_key(cfg), snapshot)
    if mkey in ctx["res_memo"]:
        return ctx["res_memo"][mkey]
    if snapshot != "best" and ctx["maps"]["second"] is None:
        res = {"skipped": "maps_second missing"}
        ctx["res_memo"][mkey] = res
        return res

    sel = select_on_va(ctx, cfg)
    fcfg = dict(cfg)
    fcfg["cthr"] = sel["best_cthr"]

    pt_rows = predict_set(ctx, ctx["ptest_ids"], fcfg, snapshot)
    pt_mean, pt_per = morphometry_score(_sel_df(pt_rows, ctx["ptest_ids"], sel["use_alt"]),
                                        ctx["gt_pt"])
    va_rows = predict_set(ctx, ctx["va_ids"], fcfg, snapshot)
    va_mean, va_per = morphometry_score(_sel_df(va_rows, ctx["va_ids"], sel["use_alt"]),
                                        ctx["gt_va"])
    res = {"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in fcfg.items()},
           "snapshot": snapshot, "best_cthr": sel["best_cthr"], "use_alt": sel["use_alt"],
           "alt_taus": sel["alt_taus"], "sel_va_mean": sel["sel_va_mean"],
           "va": {"mean": va_mean, **va_per}, "ptest": {"mean": pt_mean, **pt_per},
           "n_nonfinite": _n_nonfinite(pt_rows) + _n_nonfinite(va_rows)}
    ctx["res_memo"][mkey] = res
    return res


def log_record(rec):
    if "skipped" in rec:
        print(f"[var] {rec['name']:<26s} SKIPPED ({rec['skipped']})", flush=True)
        return
    alt = "".join("A" if rec["use_alt"][c] else "-" for c in ALT_COLS)
    per = " ".join(f"{c[:4]}={rec['ptest'][c]:.3f}" for c in COLS)
    print(f"[var] {rec['name']:<26s} va {rec['va']['mean']:.4f} ptest {rec['ptest']['mean']:.4f}"
          f" | {per} | cthr={rec['best_cthr']:.1e} alt[{alt}]"
          + (f" NONFINITE={rec['n_nonfinite']}" if rec["n_nonfinite"] else ""), flush=True)


# ----------------------------- experiment grid -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="RUN dir with meta.json + maps_best/")
    ap.add_argument("--data", default=None, help="dataset dir (targets + PNG size fallback)")
    ap.add_argument("--targets-csv", default=None, help="override GT csv path")
    ap.add_argument("--out", required=True, help="results json")
    ap.add_argument("--quick", action="store_true", help="40+40 tiles, coarse grid")
    args = ap.parse_args()
    if args.data is None and args.targets_csv is None:
        ap.error("need --data or --targets-csv")

    t0 = time.time()
    ctx = build_ctx(args)
    records = []

    def add(name, cfg, snapshot, phase):
        rec = {"name": name, "phase": phase}
        rec.update(run_protocol(ctx, cfg, snapshot))
        if "skipped" not in rec:
            rec["views"] = list(cfg["views"])
        records.append(rec)
        log_record(rec)
        return rec

    # --- phase 0: reference (current solution.py behavior) ---
    print("\n=== phase 0: reference ===", flush=True)
    add("REF_tta4_best", make_cfg(), "best", "reference")

    # --- phase 1: TTA view subsets ---
    print("\n=== phase 1: TTA view subsets ===", flush=True)
    tta_sets = [TTA1, TTA2, TTA4, TTA8, (0, 4), (0, 1, 4, 5)]
    if args.quick:
        tta_sets = [TTA1, TTA4, TTA8, (0, 4)]
    tta_recs = [add(f"TTA_v{''.join(map(str, v))}", make_cfg(views=v), "best", "tta")
                for v in tta_sets]
    # honest selection: by va mean tau; ties -> the larger view set (more averaging)
    def _rank(r):
        return (r["va"]["mean"], len(r["views"]))
    best_views = tuple(max(tta_recs, key=_rank)["views"])
    best_pt = tuple(max(tta_recs, key=lambda r: (r["ptest"]["mean"], len(r["views"])))["views"])
    print(f"[tta] best by va: {best_views}  (best by ptest, for info: {best_pt})", flush=True)

    # --- phase 2: snapshot combination ---
    print("\n=== phase 2: snapshot combination ===", flush=True)
    for v, tag in [(TTA4, "tta4"), (TTA8, "tta8")]:
        for snap in ["best", "avg_decode", "avg_maps"]:
            add(f"SNAP_{tag}_{snap}", make_cfg(views=v), snap, "snapshot")

    # --- phase 3: cthr sensitivity tables (report only) ---
    print("\n=== phase 3: cthr sensitivity (report only) ===", flush=True)
    cthr_tables = {}
    for tag, v in [("tta8", TTA8), ("tta1_selection_pass", TTA1)]:
        rows = []
        for cthr in ctx["cthr_table"]:
            c = make_cfg(views=v, cthr=cthr)
            p = _full_df(predict_set(ctx, ctx["va_ids"], c, "best"), ctx["va_ids"])
            mt, per = morphometry_score(p, ctx["gt_va"])
            rows.append({"cthr": cthr, "va_mean": mt, **per})
            print(f"[cthr:{tag}] {cthr:.1e}: va mean {mt:.4f} | "
                  + " ".join(f"{c2[:4]}={per[c2]:.3f}" for c2 in COLS), flush=True)
        cthr_tables[tag] = rows

    # --- phase 4: NMS / peak knobs at the best TTA ---
    print(f"\n=== phase 4: NMS/peak knobs (views={best_views}, snapshot=best) ===", flush=True)
    base = dict(views=best_views)
    knob_grids = {"nms_radius": [1, 2], "peak_thr": [0.03, 0.05, 0.08, 0.12],
                  "k_mult": [0.8, 1.0, 1.2]}
    if args.quick:
        knob_grids["peak_thr"] = [0.03, 0.05, 0.12]
    knob_best = {}
    for knob, vals in knob_grids.items():
        rr = []
        for val in vals:
            rr.append((val, add(f"{knob}={val}", make_cfg(**base, **{knob: val}), "best",
                                "peaks_knob")))
        knob_best[knob] = max(rr, key=lambda t: t[1]["va"]["mean"])[0]
    print(f"[knobs] best-by-va one-at-a-time: {knob_best}", flush=True)
    add("JOINT_peaks_best", make_cfg(**base, **knob_best), "best", "peaks_knob")

    # --- phase 5: attribute readout window ---
    print("\n=== phase 5: readout window ===", flush=True)
    for r in [1, 2]:
        add(f"readout_r={r}", make_cfg(**base, readout_r=r), "best", "readout")

    # --- phase 6: alt-estimator mask threshold ---
    print("\n=== phase 6: alt mask threshold ===", flush=True)
    for t in [0.3, 0.5, 1.0]:
        add(f"alt_mask_thr={t}", make_cfg(**base, alt_mask_thr=t), "best", "alt_mask")

    # --- phase 7: dispersion estimators ---
    print("\n=== phase 7: dispersion estimators ===", flush=True)
    add("disp=peaks", make_cfg(**base), "best", "dispersion")
    soft_sigmas = [0, 2] if args.quick else [0, 1, 2, 4]
    for s in soft_sigmas:
        add(f"disp=soft_s{s}", make_cfg(**base, disp_est="soft", soft_sigma=float(s)),
            "best", "dispersion")
    quad_gs = [8, 16] if args.quick else [8, 16, 32]
    for g in quad_gs:
        add(f"disp=quadrat_g{g}", make_cfg(**base, disp_est="quadrat", quadrat_g=g),
            "best", "dispersion")

    # --- write results ---
    scored = [r for r in records if "skipped" not in r]
    ranked = sorted(scored, key=lambda r: -r["ptest"]["mean"])
    ref = next(r for r in records if r["name"] == "REF_tta4_best")
    out = {"meta": {"run": str(Path(args.run).resolve()), "quick": bool(args.quick),
                    "n_va": len(ctx["va_ids"]), "n_ptest": len(ctx["ptest_ids"]),
                    "has_second": ctx["maps"]["second"] is not None,
                    "cthr_grid": ctx["cthr_grid"], "best_views_by_va": list(best_views),
                    "train_stats": ctx["ts"], "n_decode_calls": ctx["n_decode"][0],
                    "minutes": (time.time() - t0) / 60.0},
           "reference": ref, "records": records, "cthr_tables": cthr_tables,
           "ranked": [{"name": r["name"], "ptest_mean": r["ptest"]["mean"],
                       "va_mean": r["va"]["mean"],
                       "delta_vs_ref": r["ptest"]["mean"] - ref["ptest"]["mean"]}
                      for r in ranked]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"\n=== ranked by ptest mean tau (REF = {ref['ptest']['mean']:.4f}) ===", flush=True)
    for r in ranked[:15]:
        print(f"  {r['name']:<26s} ptest {r['ptest']['mean']:.4f} "
              f"({r['ptest']['mean']-ref['ptest']['mean']:+.4f})  va {r['va']['mean']:.4f}",
              flush=True)
    print(f"wrote {args.out} ({len(records)} records, {ctx['n_decode'][0]} decodes, "
          f"{(time.time()-t0)/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
