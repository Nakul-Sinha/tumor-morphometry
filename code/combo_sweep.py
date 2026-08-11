"""Interaction/combination sweep over the winning decode levers from the phase sweep.

Reuses decode_sweep's verified protocol (va-based cthr + alt selection, honest
ptest eval, memoized). Grid: views x k_mult x alt_mask_thr, with an EXTENDED
cthr selection grid, plus one-at-a-time nms/readout rechecks at the best combo.

Usage: python combo_sweep.py --run RUN_DIR --data DATASET_DIR --out RESULTS_JSON
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import decode_sweep as DS

CTHR_EXT = [1e-4, 2e-4, 3e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3, 1.2e-3]

VIEWS = [(0, 4), (0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 2, 3, 4, 5, 6, 7)]
KMULT = [0.6, 0.7, 0.8, 0.9, 1.0]
ALTM = [0.3, 0.5]


def vtag(v):
    return "v" + "".join(map(str, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--targets-csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ctx = DS.build_ctx(args)
    ctx["cthr_grid"] = CTHR_EXT  # extended selection grid (also implementable in-script)
    records = []

    def add(name, cfg):
        rec = {"name": name}
        rec.update(DS.run_protocol(ctx, cfg, "best"))
        records.append(rec)
        DS.log_record(rec)
        return rec

    print("=== reference (tta4, defaults, extended grid) ===", flush=True)
    ref = add("REF_tta4_extgrid", DS.make_cfg())

    print("\n=== combo grid: views x k_mult x alt_mask_thr ===", flush=True)
    for v, km, am in itertools.product(VIEWS, KMULT, ALTM):
        add(f"{vtag(v)}_k{km:g}_a{am:g}", DS.make_cfg(views=v, k_mult=km, alt_mask_thr=am))

    best = max(records, key=lambda r: r["va"]["mean"])
    bcfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in best["cfg"].items()}
    print(f"\n[combo] best by va: {best['name']} va {best['va']['mean']:.4f} "
          f"ptest {best['ptest']['mean']:.4f}", flush=True)

    print("\n=== rechecks at best combo ===", flush=True)
    base = dict(views=bcfg["views"], k_mult=bcfg["k_mult"], alt_mask_thr=bcfg["alt_mask_thr"])
    for name, kw in [("nms_radius=2", {"nms_radius": 2}),
                     ("readout_r=2", {"readout_r": 2}),
                     ("peak_thr=0.03", {"peak_thr": 0.03}),
                     ("peak_thr=0.08", {"peak_thr": 0.08})]:
        add(f"BEST+{name}", DS.make_cfg(**base, **kw))

    ranked_va = sorted(records, key=lambda r: -r["va"]["mean"])
    ranked_pt = sorted(records, key=lambda r: -r["ptest"]["mean"])
    out = {"meta": {"run": str(Path(args.run).resolve()), "cthr_grid": CTHR_EXT,
                    "minutes": (time.time() - t0) / 60.0,
                    "n_decode_calls": ctx["n_decode"][0]},
           "reference": ref, "records": records,
           "ranked_by_va": [{"name": r["name"], "va": r["va"]["mean"],
                             "ptest": r["ptest"]["mean"], "cthr": r["best_cthr"],
                             "use_alt": r["use_alt"]} for r in ranked_va],
           "ranked_by_ptest": [{"name": r["name"], "ptest": r["ptest"]["mean"],
                                "va": r["va"]["mean"]} for r in ranked_pt]}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"\n=== top 12 by va (adoption view) ===", flush=True)
    for r in ranked_va[:12]:
        print(f"  {r['name']:<24s} va {r['va']['mean']:.4f}  ptest {r['ptest']['mean']:.4f}"
              f"  cthr={r['best_cthr']:.0e} alt={r['use_alt']}", flush=True)
    print(f"\n=== top 12 by ptest (evidence view) ===", flush=True)
    for r in ranked_pt[:12]:
        print(f"  {r['name']:<24s} ptest {r['ptest']['mean']:.4f}  va {r['va']['mean']:.4f}",
              flush=True)
    print(f"wrote {args.out} ({len(records)} records, {ctx['n_decode'][0]} decodes, "
          f"{(time.time()-t0)/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
