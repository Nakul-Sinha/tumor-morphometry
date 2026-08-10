# Eris Challenge 2 — Tumor Microenvironment Morphometry

Goal: beat AI baseline 0.46 (mean of 5 clipped Kendall taus over 384 test tiles).

## Approach
Dense-supervision reframing: train a ResNet18-UNet (in-script) on per-cell
annotations to predict 7 stride-2 maps per tile (4 class-density, 1 center
heatmap, 2 attribute maps: log-area + elongation). Decode the five scalar
targets from predicted maps with the EXACT label-generation arithmetic.

## Measured facts (train set)
- Targets are exact aggregates of `train_cells.csv` (all tau >= 0.9999):
  - `cellularity = 1e5 * N / (W*H)`
  - `tumor_frac = n_tumor / N`
  - `size_spread = std(area, ddof=0) / mean(area)` (population CV)
  - `elongation = mean(elongation)`
  - `dispersion = mean_NN_dist_PIXELS * 2 * sqrt(N/(W*H))` (Clark-Evans;
    pixel coords, NOT normalized coords — normalized only reaches tau 0.959)
- Oracle decode ceiling (decode run on GT-rendered maps, n=1358):
  mean 0.9918 — cell 1.000 / tumor 0.998 / size 0.996 / elong 0.998 / disp 0.967.
- Rejected by measurement: quadrat-VMR dispersion from smoothed density
  (oracle tau only 0.25–0.46); subpixel peak refinement (no change, 0.9655).

## Final recipe (CPU-only reference env: 10 cores, 62GB, 90-min limit)
Single ResNet18-UNet (ImageNet encoder, random-init fallback), 20 epochs x
4 crops/image (256px), bs 16, AdamW 3e-4 cosine, val every 3 epochs on a 15%
slide-grouped holdout with metric-aligned checkpointing; in-script selection of
the density noise-floor threshold (cthr grid) and size/elongation estimator
(peak vs density-weighted) on the same held-out slides; flip-TTA4 per-sample
inference. device is hardcoded cpu; threads = min(10, cores); fixed seeds.
Emergency-only guards (train stop 68 min, budget 82 min) never fire on
reference hardware.

## Estimated scores (28% slide-grouped pseudo-test, 402 tiles, exact metric)
- CPU recipe (the deliverable): **0.5663** mean tau
  (cell 0.742, tumor 0.591, size 0.507, elong 0.561, disp 0.431)
- Upside (40-epoch 2-model ensemble, if a GPU were available): 0.5914
- AI baseline to beat: 0.46

## Layout
- `solution.py` — self-contained Eris submission (`python3 solution.py <public_dir> <out.csv>`);
  trains model A (+B if time), metric-aligned checkpointing on slide-grouped
  holdout, in-script estimator selection, flip-TTA per-sample inference,
  time-guarded (55 min train stop), defensive CSV write.
- `code/` — dev modules and tests:
  - `metric.py` exact MorphometryScore; `maps.py` render/decode core
  - `oracle_formulas.py` / `oracle_decode.py` — milestone-1 verification
  - `dataset.py`, `model.py`, `train.py`, `infer.py`, `predict.py` — dev training
  - `eval_decode.py` — offline decode-variant sweeps from dumped val maps
  - `test_pipeline.py` — integral/shape checks; `test_vmr.py`, `test_subpixel.py`

## Validation discipline
Slide-grouped only (`group` column): GroupShuffleSplit 15% for training runs,
GroupKFold k=4 for the headline estimate. Test slides are fully held out, so
tile-level splits would leak.
