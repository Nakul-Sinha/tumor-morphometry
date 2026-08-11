"""Self-test for decode_sweep.py on a synthetic RUN dir (no real map dumps needed).

Builds ~31 fake tiles, renders maps with solution.render_maps, stacks them into the
(8, 7, mh, mw) fp16 per-view contract with small per-view noise, and checks:
  a. decode_variant(default cfg) == solution.decode_tile, bit for bit  [critical]
  b. the sweep runs end to end under --quick and writes a sane results json
  c. TTA8 decodes differ from TTA4 but the ptest tau moves by < 0.2
  d. all three dispersion estimators are finite on every tile, degenerate included
  e. the PNG-header size fallback reproduces meta['sizes']

Usage: python test_decode_sweep_synth.py [--keep]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import solution as S  # noqa: E402
import decode_sweep as DS  # noqa: E402
from metric import COLS  # noqa: E402

SCRATCH = Path(r"C:\Users\nakul\AppData\Local\Temp\claude"
               r"\G--Datacurve-Latest-Chals\da66ee4a-e36e-4362-b628-ae671430fee1\scratchpad")
RUN = SCRATCH / "synth_run"
DATA = SCRATCH / "synth_data"

N_TILES = 31
DEG_IDX = (0, 16)          # degenerate all-zero tiles (one in va, one in ptest)
VA_N = 16                  # first 16 ids -> va, rest -> ptest
TS0 = {"dispersion_mean": 1.26, "size_spread_med": 0.51, "elongation_med": 1.32}
# per-channel absolute noise std (channels are RAW head units: dens*100, heat, amap, emap)
NOISE = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01], np.float32)


# ----------------------------- synthetic run -----------------------------
def gen_tile(rng, iid, degenerate):
    H = int(rng.randint(180, 321))
    W = int(rng.randint(180, 321))
    mh, mw = S.map_shape(H, W)
    if degenerate:
        return H, W, np.zeros((7, mh, mw), np.float32)
    n = int(rng.randint(40, 201))
    x = rng.uniform(0.02, 0.98, n)
    y = rng.uniform(0.02, 0.98, n)
    ci = rng.choice(4, size=n, p=[0.45, 0.25, 0.2, 0.1])
    area = np.exp(rng.normal(4.4, 0.45, n))          # ~80 px^2 median
    elo = 1.0 + rng.gamma(2.0, 0.35, n)              # ~1..3
    dens, heat, amap, emap = S.render_maps(
        list(zip(x, y, ci, area, elo)), H, W)
    raw = np.concatenate([dens * S.DENS_SCALE, heat[None], amap[None], emap[None]], 0)
    return H, W, raw.astype(np.float32)


def stack_views(raw, rng, bias=0.0):
    """(7,mh,mw) -> (8,7,mh,mw) fp16: the same map 8x with tiny per-view noise.
    Views 4..7 get more noise so the TTA8 mean is genuinely different from TTA4."""
    if not raw.any():
        return np.zeros((8,) + raw.shape, np.float16)
    out = np.empty((8,) + raw.shape, np.float32)
    for v in range(8):
        amp = NOISE[:, None, None] * (1.0 + 0.4 * v)
        out[v] = raw + bias * NOISE[:, None, None] + rng.normal(0, 1, raw.shape) * amp
    return out.astype(np.float16)


def build_synth():
    rng = np.random.RandomState(20260811)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "maps_best").mkdir(exist_ok=True)
    (RUN / "maps_second").mkdir(exist_ok=True)
    (DATA / "train_images").mkdir(parents=True, exist_ok=True)
    from PIL import Image

    ids, sizes, rows = [], {}, []
    for i in range(N_TILES):
        iid = f"syn{i:03d}"
        ids.append(iid)
        H, W, raw = gen_tile(rng, iid, i in DEG_IDX)
        sizes[iid] = [H, W]
        np.savez_compressed(RUN / "maps_best" / f"{iid}.npz", v=stack_views(raw, rng))
        np.savez_compressed(RUN / "maps_second" / f"{iid}.npz",
                            v=stack_views(raw, rng, bias=0.6))
        # GT targets: decode the CLEAN rendered maps (taus stay high but < 1)
        dens = np.maximum(raw[:4], 0.0) / S.DENS_SCALE
        d = S.decode_tile(dens, raw[4], raw[5], raw[6], H, W, TS0, cthr=S.DENS_THR)
        rows.append({"image_id": iid, **{c: d[c] for c in COLS}})
        # header-only-readable PNG for the size-fallback path (uniform -> tiny file)
        Image.new("RGB", (W, H), (128, 128, 128)).save(DATA / "train_images" / f"{iid}.png")

    tgt = pd.DataFrame(rows)
    tgt.to_csv(RUN / "targets.csv", index=False)
    tgt.to_csv(DATA / "train_targets.csv", index=False)

    ok = tgt[~tgt["image_id"].isin([ids[i] for i in DEG_IDX])]
    train_stats = {"dispersion_mean": float(ok["dispersion"].mean()),
                   "size_spread_med": float(ok["size_spread"].median()),
                   "elongation_med": float(ok["elongation"].median())}
    meta = {"va_ids": ids[:VA_N], "ptest_ids": ids[VA_N:], "train_stats": train_stats,
            "sizes": sizes, "val_history": [], "best": {"tag": "synthetic_best"},
            "second": {"tag": "synthetic_second"}, "ref_tta4": {}}
    (RUN / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[synth] {N_TILES} tiles -> {RUN}", flush=True)
    print(f"[synth] train_stats {train_stats}", flush=True)
    return meta, ids, sizes


def load_arr(iid, which="maps_best"):
    with np.load(RUN / which / f"{iid}.npz") as z:
        return np.asarray(z["v"], np.float16)


# ----------------------------- checks -----------------------------
def check_parity(meta, ids, sizes):
    """(a) default cfg must reproduce solution.decode_tile bit for bit."""
    ts = meta["train_stats"]
    n_ok, shown = 0, False
    for iid in ids[:10]:
        arr = load_arr(iid)
        H, W = sizes[iid]
        dens, heat, amap, emap = DS.reconstruct(arr, DS.TTA4)
        ref = S.decode_tile(dens, heat, amap, emap, H, W, ts, cthr=S.DENS_THR)
        got = DS.decode_variant(arr, H, W, ts, {})
        assert set(got) == set(ref) == set(DS.OUT_KEYS), (sorted(got), sorted(ref))
        for k in ref:
            assert got[k] == ref[k], f"{iid}.{k}: {got[k]!r} != {ref[k]!r}"
        n_ok += 1
        if not shown and iid != ids[DEG_IDX[0]]:
            shown = True
            print(f"[parity] verbatim, tile {iid} (H={H} W={W}, views=TTA4, cthr=3e-04):",
                  flush=True)
            for k in DS.OUT_KEYS:
                print(f"    {k:<17s} sweep={got[k]!r}  solution={ref[k]!r}  "
                      f"exact={got[k] == ref[k]}", flush=True)
    # also parity for a non-default cthr and for the tta1 selection pass
    for iid in ids[:10]:
        arr = load_arr(iid)
        H, W = sizes[iid]
        for views, cthr in [(DS.TTA4, 8e-4), (DS.TTA1, 3e-4), (DS.TTA2, 1.2e-3)]:
            m = DS.reconstruct(arr, views)
            ref = S.decode_tile(*m, H, W, ts, cthr=cthr)
            got = DS.decode_variant(arr, H, W, ts, {"views": views, "cthr": cthr})
            for k in ref:
                assert got[k] == ref[k], f"{iid}.{k} views={views} cthr={cthr}"
    print(f"[parity] OK: {n_ok}/10 tiles bit-identical to S.decode_tile "
          f"(+ tta1/tta2/tta4 x cthr {{3e-4, 8e-4, 1.2e-3}} cross-checks)", flush=True)


def check_disp_finite(meta, ids, sizes):
    """(d) every dispersion estimator finite on every tile, degenerate included."""
    ts = meta["train_stats"]
    variants = ([{"disp_est": "peaks"}]
                + [{"disp_est": "soft", "soft_sigma": float(s)} for s in (0, 1, 2, 4)]
                + [{"disp_est": "quadrat", "quadrat_g": g} for g in (8, 16, 32)])
    deg_ids = [ids[i] for i in DEG_IDX]
    seen_fallback = 0
    for iid in ids:
        arr = load_arr(iid)
        H, W = sizes[iid]
        for v in variants:
            d = DS.decode_variant(arr, H, W, ts, dict(v, views=DS.TTA8))
            for k in DS.OUT_KEYS:
                assert np.isfinite(d[k]), f"{iid} {v} {k} = {d[k]}"
            if iid in deg_ids:
                assert d["dispersion"] == ts["dispersion_mean"], (iid, v, d["dispersion"])
                assert d["size_spread"] == ts["size_spread_med"]
                assert d["elongation"] == ts["elongation_med"]
                assert d["cellularity"] == 0.0 and d["tumor_frac"] == 0.0
                seen_fallback += 1
    print(f"[disp] OK: {len(variants)} estimators x {len(ids)} tiles all finite; "
          f"{seen_fallback} degenerate-tile fallbacks hit train_stats", flush=True)
    # sanity: the estimators are not all the same constant
    ranks = {}
    for v in variants:
        ranks[str(v)] = [DS.decode_variant(load_arr(i), *sizes[i], ts,
                                           dict(v, views=DS.TTA8))["dispersion"]
                         for i in ids if i not in deg_ids]
    for name, vals in ranks.items():
        assert np.ptp(vals) > 0, f"{name} constant across tiles"
        print(f"    {name:<45s} range [{min(vals):.4f}, {max(vals):.4f}]", flush=True)


def check_tta(meta, ids, sizes):
    """(c) TTA8 output differs from TTA4 (per-view noise) on real tiles."""
    ts = meta["train_stats"]
    diff_tiles = 0
    for iid in ids:
        if iid in [ids[i] for i in DEG_IDX]:
            continue
        arr = load_arr(iid)
        H, W = sizes[iid]
        d4 = DS.decode_variant(arr, H, W, ts, {"views": DS.TTA4})
        d8 = DS.decode_variant(arr, H, W, ts, {"views": DS.TTA8})
        if any(d4[k] != d8[k] for k in DS.OUT_KEYS):
            diff_tiles += 1
    assert diff_tiles > 0, "TTA8 identical to TTA4 on every tile"
    print(f"[tta] OK: TTA8 decode differs from TTA4 on {diff_tiles}/"
          f"{len(ids)-len(DEG_IDX)} non-degenerate tiles", flush=True)


def check_size_fallback(meta, ids, sizes):
    """(e) PNG-header fallback when meta has no 'sizes'."""
    m2 = {k: v for k, v in meta.items() if k != "sizes"}
    got = DS.load_sizes(m2, ids, DATA)
    for iid in ids:
        assert got[iid] == tuple(sizes[iid]), (iid, got[iid], sizes[iid])
    print(f"[sizes] OK: PNG-header fallback matches meta['sizes'] on {len(ids)} tiles",
          flush=True)


def check_sweep(out_json):
    """(b) end-to-end --quick run."""
    cmd = [sys.executable, str(HERE / "decode_sweep.py"), "--run", str(RUN),
           "--targets-csv", str(RUN / "targets.csv"), "--out", str(out_json), "--quick"]
    print(f"[sweep] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise AssertionError(f"decode_sweep.py exited {p.returncode}")
    res = json.loads(Path(out_json).read_text())

    names = [r["name"] for r in res["records"]]
    assert "REF_tta4_best" in names, names
    assert res["reference"]["name"] == "REF_tta4_best"
    assert len(names) == len(set(names)), "duplicate record names"
    scored = [r for r in res["records"] if "skipped" not in r]
    assert len(scored) == len(res["records"]), "some variants were skipped"
    for r in scored:
        assert r["n_nonfinite"] == 0, (r["name"], r["n_nonfinite"])
        for side in ("va", "ptest"):
            for c in ["mean"] + COLS:
                t = r[side][c]
                assert np.isfinite(t) and 0.0 <= t <= 1.0, (r["name"], side, c, t)
        assert r["best_cthr"] in res["meta"]["cthr_grid"], (r["name"], r["best_cthr"])
        assert set(r["use_alt"]) == {"size_spread", "elongation"}
    for tag, rows in res["cthr_tables"].items():
        for row in rows:
            assert 0.0 <= row["va_mean"] <= 1.0, (tag, row)
    assert res["meta"]["has_second"] is True
    for snap in ("best", "avg_decode", "avg_maps"):
        assert f"SNAP_tta8_{snap}" in names

    d = {r["name"]: r["ptest"]["mean"] for r in scored}
    gap = abs(d["TTA_v01234567"] - d["TTA_v0123"])
    assert gap < 0.2, f"tta8 vs tta4 ptest tau gap {gap:.4f} >= 0.2"
    print(f"[sweep] OK: {len(scored)} records, tta8-vs-tta4 ptest gap {gap:.4f}, "
          f"{time.time()-t0:.1f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="reuse an existing synth run")
    args = ap.parse_args()

    if args.keep and (RUN / "meta.json").exists():
        meta = json.loads((RUN / "meta.json").read_text())
        ids = meta["va_ids"] + meta["ptest_ids"]
        sizes = {k: tuple(v) for k, v in meta["sizes"].items()}
        print(f"[synth] reusing {RUN}", flush=True)
    else:
        meta, ids, sz = build_synth()
        sizes = {k: tuple(v) for k, v in sz.items()}

    check_parity(meta, ids, sizes)
    check_disp_finite(meta, ids, sizes)
    check_tta(meta, ids, sizes)
    check_size_fallback(meta, ids, sizes)
    res = check_sweep(SCRATCH / "synth_results.json")

    ref = res["reference"]
    print("\n[summary] top rows by ptest mean tau:", flush=True)
    for r in res["ranked"][:8]:
        print(f"    {r['name']:<26s} ptest {r['ptest_mean']:.4f} "
              f"({r['delta_vs_ref']:+.4f})  va {r['va_mean']:.4f}", flush=True)
    print(f"[summary] REF_tta4_best ptest {ref['ptest']['mean']:.4f} "
          f"cthr={ref['best_cthr']:.1e} use_alt={ref['use_alt']}", flush=True)
    print("\nALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
