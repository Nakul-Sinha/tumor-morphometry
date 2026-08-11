# Tumor Microenvironment Morphometry

## The problem

I get histology tiles of tumor tissue and have to predict five population level
properties of the cells in each tile: cellularity, tumor cell fraction, size
spread, mean elongation, and spatial dispersion. Scoring is the mean of five
clipped Kendall taus over 384 test tiles, so only the ordering of each property
matters, not the absolute value. The baseline to beat is 0.46.

## What I did

I deliberately did not regress the five numbers straight off the image. The
training split ships per-cell annotations, so instead I train a ResNet18-UNet to
predict seven dense maps per tile: four class density maps, a cell center
heatmap, and two attribute maps for log area and elongation. The five properties
are then computed from those predicted maps using their ordinary morphometric
definitions.

The reason this is worth the detour is that the reframing is close to lossless.
When I render maps from the ground truth cell table and run the same decode, I
recover the official targets at mean tau 0.9918. So the model only has to learn
the maps, where the supervision is dense, instead of five noisy scalars.

It runs CPU only, about 67 minutes on 10 cores, inside the grader budget.

## Layout

`solution.py` is the self contained entry point. `TECHNICAL.md` has the full
recipe, the ablations, and the timing breakdown. Datasets are not committed.
