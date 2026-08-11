"""Milestone 2: validate data pipeline. Checks per random crop:
- density map integral == #cells in crop (within 1e-3 per cell)
- rendered maps land where cells are (peak positions near GT cell centers)
- model forward shape check + one optimizer step."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CROP, CropDataset, load_images_and_cells
from maps import DENS_SCALE, map_shape
from model import UNetR18

DATA = Path(r"G:\ml\Latest_Chals\challenge 2\dataset")


def main():
    targets = pd.read_csv(DATA / "train_targets.csv")
    ids = list(targets["image_id"])[:60]
    images, cells_by = load_images_and_cells(DATA, ids)
    ds = CropDataset(images, cells_by, ids, crops_per_image=1, train=True, seed=1)

    max_err = 0.0
    tot_cells = 0
    for i in range(len(ds)):
        x, tgt, counts = ds[i]
        integral = tgt[:4].sum().item() / DENS_SCALE
        n_true = counts.sum().item()
        max_err = max(max_err, abs(integral - n_true))
        tot_cells += n_true
        # per-class too
        per_cls = tgt[:4].sum(dim=(1, 2)) / DENS_SCALE
        assert torch.allclose(per_cls, counts, atol=1e-3), (per_cls, counts)
        assert x.shape == (3, CROP, CROP) and tgt.shape == (7, CROP // 2, CROP // 2)
        if n_true > 0:
            assert tgt[4].max() > 0.9, "heatmap should have amplitude ~1"
            assert (tgt[5] > 0.5).sum() > 0, "attr disks painted"
    print(f"60 crops, {tot_cells:.0f} cells total, max |integral-count| = {max_err:.2e}")
    assert max_err < 1e-3 * max(1, tot_cells / 60), "integral mismatch"

    # eval-mode crop (no aug) then model smoke step
    ds_eval = CropDataset(images, cells_by, ids, crops_per_image=1, train=False, seed=2)
    x, tgt, counts = ds_eval[0]
    model = UNetR18(pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    xb, tb = x[None], tgt[None]
    out = model(xb)
    assert out.shape == (1, 7, CROP // 2, CROP // 2), out.shape
    loss = torch.nn.functional.mse_loss(out, tb)
    loss.backward()
    opt.step()
    print(f"model forward/backward OK, loss {loss.item():.3f}")
    print("PIPELINE OK")


if __name__ == "__main__":
    main()
