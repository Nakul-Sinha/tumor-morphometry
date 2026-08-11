# Tumor Microenvironment Morphometry: technical notes

Goal: beat AI baseline 0.46 (mean of 5 clipped Kendall taus over 384 test tiles).

## Approach
Dense-supervision reframing: train a ResNet18-UNet in-script on the provided
per-cell annotations to predict 7 stride-2 maps per tile (4 class-density maps,
1 center heatmap, 2 attribute maps: log-area + elongation). The five population
properties are then computed from the predicted maps using their standard
morphometric definitions, which match the provided train_targets to tau>=0.9999:

- cellularity = cells per 100,000 px^2 (1e5 * N / (W*H))
- tumor_frac = tumour-cell fraction
- size_spread = population coefficient of variation of cell area
- elongation = mean cell elongation
- dispersion = Clark-Evans nearest-neighbour index in pixel coordinates

Oracle check: running the decode on maps rendered from the ground-truth cell
table reproduces the targets at mean tau 0.9918 (dispersion 0.967 is the
ceiling, nearby peaks merge).

## Final recipe (CPU-only reference env: 10 cores, 62GB, 90-min limit)
Single ResNet18-UNet, 20 epochs x 4 crops/image (256px), bs 16, AdamW 3e-4
cosine, val every 3 epochs on a 15% slide-grouped holdout with metric-aligned
checkpointing; in-script joint selection (train slides only, cached tta1 val
maps) of the density noise-floor threshold (9-point cthr grid
1e-4..1.2e-3) x the peak budget k, va-selected over {0.8, 1.0} x n_hat, then
the size/elongation estimator; D4 TTA8 per-sample inference (4 flips + their
rot90 compositions, canonicalized before the map crop); fail-closed decode
(train-statistic fallbacks when confident peaks < 3). Device is hardcoded cpu,
threads = min(10, cores), fixed seeds. Encoder init: ImageNet is attempted
first and a broken/missing torchvision degrades loudly to random init instead
of crashing; see provenance table for the measured pretrained-vs-random delta.

Wall-clock on the reference box (10 pinned cores): ~67 min (20-epoch pipeline
~66.4 min + 1.2 min TTA8 test inference, vs 0.6 min at TTA4). Emergency guards
(train stop 68 min, whole-script budget 82 min) exist only for much slower
hardware and do not fire on the reference box. Degraded floor if they ever
fire: the val curve reaches ~0.52 by epoch 8 and ~0.54 by epoch 14, so even a
2x-slower box that truncates training lands near ~0.50-0.52, still well above
the 0.46 baseline.

## Estimated scores (28% slide-grouped pseudo-test, 402 tiles, exact metric)
Estimates are device-agnostic (same recipe/steps/seeds; independent training
runs vary by roughly +-0.01-0.02 mean tau):

- Fixed recipe (the deliverable): ~0.57 mean tau
  (cell 0.742, tumor 0.591, size 0.507, elong 0.561, disp 0.431 at 20 epochs)
- Upside variant (40-epoch 2-model ensemble, does not fit the CPU budget): 0.5914
- AI baseline to beat: 0.46

Decode-side improvements, measured on the `base20` dev run (one 20-epoch
training run; all numbers are WITHIN-RUN controlled deltas against that same
run's reference decode at ptest 0.5400, not independent-run estimates):

- TTA8 + extended cthr grid + peak budget 0.8*n_hat (frozen): ptest 0.5717
  (+0.032 within-run)
- what actually ships, the same stack but with k va-selected over {0.8, 1.0}:
  on this run the val picks k=1.0 / cthr=6e-4 (va 0.5916, ptest 0.5650, +0.025
  within-run). Selection costs ~0.007 ptest versus the frozen k=0.8 on this one
  run, and is preferred because it adapts to whatever model the fresh in-script
  training produces rather than to a single dev checkpoint.
- rejected: snapshot (top-2 checkpoint) ensembling and the alternative
  dispersion estimators (soft / quadrat) both measured negative.

## Artifact provenance
| artifact | md5 | produced by |
|---|---|---|
| submission.csv (authoritative) | see final report | provenance/box_confirm_16ep.log (clean 10-core box run of frozen solution.py) |
| provenance/box_final_run_20ep.log | - | earlier 20-epoch box run (historical; superseded) |

All other prediction CSVs from dev runs are quarantined outside the repo.

## Layout
- `solution.py`: self-contained submission. Both invocation styles work:
  `python3 solution.py <public_dir> <out.csv>` and bare `python3 solution.py`
  (probes dataset/ mounts, writes working/submission.csv). Always dual-writes
  working/submission.csv.
- `code/`: dev modules and tests (metric replica, map render/decode core,
  oracle verification, dev trainer, decode sweeps, eval simulations).
- `provenance/`: reference-box run logs.

## Validation discipline
Slide-grouped only (`group` column): GroupShuffleSplit 15% inside training
runs; the headline estimate comes from a 28% held-out-slide pseudo-test (402
tiles) never touched by training or selection. Test slides are entirely held
out by the organisers, so tile-level splits would leak.
