"""Training dataset: random 256x256 crops with scale jitter, dihedral aug,
color jitter, on-the-fly GT map rendering (see maps.py for conventions)."""
import numpy as np
import torch
from torch.utils.data import Dataset

from maps import CLASSES, DENS_SCALE, render_maps

CROP = 256


class CropDataset(Dataset):
    def __init__(self, images, cells_by_img, ids, crops_per_image=4, train=True, seed=0):
        """images: dict id -> HxWx3 uint8; cells_by_img: dict id -> dict of arrays
        (x, y, ci, area, elo)."""
        self.images = images
        self.cells = cells_by_img
        self.ids = list(ids)
        self.cpi = crops_per_image
        self.train = train
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.ids) * self.cpi

    def __getitem__(self, idx):
        rng = np.random.RandomState((self.seed + 977 * self.epoch + idx) % (2 ** 31))
        iid = self.ids[idx % len(self.ids)]
        img = self.images[iid]
        c = self.cells[iid]
        H, W = img.shape[:2]

        # --- scale jitter: source window C = CROP/s resized to CROP ---
        s = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)))) if self.train else 1.0
        src = int(round(CROP / s))
        s_eff_x = s_eff_y = CROP / src

        # pad (zeros) if image smaller than src window
        ph, pw = max(0, src - H), max(0, src - W)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
        Hp, Wp = img.shape[:2]
        y0 = rng.randint(0, Hp - src + 1)
        x0 = rng.randint(0, Wp - src + 1)
        crop = img[y0:y0 + src, x0:x0 + src]
        if src != CROP:
            import cv2
            crop = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_LINEAR)

        # cells inside window -> crop pixel coords (scaled)
        px = c["x"] * W - x0
        py = c["y"] * H - y0
        inside = (px >= 0) & (px < src) & (py >= 0) & (py < src)
        px = px[inside] * s_eff_x
        py = py[inside] * s_eff_y
        ci = c["ci"][inside]
        area = c["area"][inside] * (s_eff_x * s_eff_y)  # s^2 area correction
        elo = c["elo"][inside]

        # --- dihedral ---
        if self.train:
            k = rng.randint(4)
            fl = rng.randint(2)
            if k:
                crop = np.rot90(crop, k)
                for _ in range(k):  # (x,y) -> (y, C-1-x)  [rot90 CCW on array]
                    px, py = py, CROP - 1 - px
            if fl:
                crop = crop[:, ::-1]
                px = CROP - 1 - px

            # --- color jitter (numpy): brightness/contrast/saturation +-0.15, per-ch +-0.05 ---
            f = crop.astype(np.float32)
            f *= rng.uniform(0.85, 1.15)
            mean = f.mean()
            f = (f - mean) * rng.uniform(0.85, 1.15) + mean
            gray = f.mean(axis=2, keepdims=True)
            f = gray + (f - gray) * rng.uniform(0.85, 1.15)
            f *= rng.uniform(0.95, 1.05, size=(1, 1, 3))
            crop = np.clip(f, 0, 255).astype(np.uint8)

        maps = render_maps(list(zip(px / CROP, py / CROP, ci,
                                    np.maximum(area, 1.0), elo)), CROP, CROP)
        dens = maps["dens"] * DENS_SCALE
        tgt = np.concatenate([dens,
                              maps["heat"][None],
                              maps["amap"][None],
                              maps["emap"][None]], axis=0)
        counts = np.bincount(ci, minlength=len(CLASSES)).astype(np.float32)

        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
            torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.from_numpy(np.ascontiguousarray(tgt)), torch.from_numpy(counts)


def load_images_and_cells(data_dir, ids):
    """Preload uint8 images + per-image cell arrays."""
    import pandas as pd
    from PIL import Image

    cells = pd.read_csv(data_dir / "train_cells.csv")
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    cells["ci"] = cells["cell_type"].map(cls_idx)
    by = {}
    for iid, g in cells.groupby("image_id"):
        by[iid] = {"x": g["x"].values.astype(np.float64),
                   "y": g["y"].values.astype(np.float64),
                   "ci": g["ci"].values.astype(np.int64),
                   "area": g["area"].values.astype(np.float64),
                   "elo": g["elongation"].values.astype(np.float64)}
    images = {}
    for iid in ids:
        with Image.open(data_dir / "train_images" / f"{iid}.png") as im:
            images[iid] = np.asarray(im.convert("RGB"))
        if iid not in by:
            empty = {k: np.zeros(0) for k in ["x", "y", "area", "elo"]}
            empty["ci"] = np.zeros(0, np.int64)
            by[iid] = empty
    return images, by
