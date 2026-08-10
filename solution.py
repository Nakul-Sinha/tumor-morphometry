"""Tumor Microenvironment Morphometry — end-to-end Eris submission.

Approach: reframe the 5 scalar targets as dense per-cell prediction. A ResNet18-UNet
is trained IN THIS SCRIPT on the provided tiles + per-cell annotations to predict:
  - 4 class density maps (sum-normalized Gaussians -> channel integral = class count)
  - 1 center heatmap (adaptive-sigma Gaussians, for peak decoding)
  - 2 attribute maps (log-area, elongation painted in disks at cell centers)
The five targets are then computed from the predicted maps with the EXACT arithmetic
of the label generation (verified tau>=0.9999 on train):
  cellularity = 1e5 * N / (W*H)          N = density integral
  tumor_frac  = N_tumor / N
  size_spread = population CV of per-cell area (cells = decoded peaks)
  elongation  = mean per-cell elongation
  dispersion  = Clark-Evans ratio in pixel coords: mean_NN_px * 2*sqrt(N/(W*H))

Validation inside the script: slide-grouped holdout (GroupShuffleSplit on `group`),
checkpoint selection and estimator selection by the actual metric (mean clipped
Kendall tau) on held-out slides. Strictly per-sample test inference (flip TTA only).

Usage: python3 solution.py <public_dir> <submission_out>
"""
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

T0 = time.time()


def elapsed_min():
    return (time.time() - T0) / 60.0


# ----------------------------- config -----------------------------
def _env(name, default, cast):
    try:
        return cast(os.environ.get(name, default))
    except Exception:
        return default


# ---- FIXED CPU RECIPE (reference env: 10 CPU cores, 62GB RAM, 90-min limit) ----
# Measured on a 10-core-pinned Zen5 box: 158 s/epoch, ~0.1 s/tile forward.
# 20 epochs x 4 crops/image + val every 3 + TTA4 inference ~= 63-65 min wall.
SEED = 42
EPOCHS = _env("CH2_EPOCHS", 20, int)
CPI = _env("CH2_CPI", 4, int)                       # crops per image per epoch
BATCH = 16
LR = 3e-4
WD = 1e-4
VAL_EVERY = 3
CROP = 256
# Emergency-only wall-clock guards: must NEVER fire on reference hardware.
TRAIN_STOP_MIN = _env("CH2_TRAIN_STOP_MIN", 68.0, float)
TOTAL_BUDGET_MIN = _env("CH2_TOTAL_BUDGET_MIN", 82.0, float)
MODEL_A_STOP_MIN = _env("CH2_MODEL_A_STOP_MIN", 68.0, float)
TWO_MODELS = _env("CH2_TWO_MODELS", 0, int)         # single model A (CPU budget)
SUBSET = _env("CH2_SUBSET", 0, int)                 # dev-only: limit train tiles
TTA = _env("CH2_TTA", 4, int)
LIMIT_TEST = _env("CH2_LIMIT_TEST", 0, int)         # dev-only: limit test tiles

STRIDE = 2
DENS_SIGMA = 4.0
ATTR_RADIUS = 4
HEAT_SIGMA_MIN, HEAT_SIGMA_MAX = 1.5, 6.0
CLASSES = ["tumor", "immune", "stromal", "other"]
DENS_SCALE = 100.0
COLS = ["cellularity", "tumor_frac", "size_spread", "elongation", "dispersion"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# CPU is THE reference device (identical behavior everywhere; no cuda branch).
# CH2_DEVICE is a dev-only escape hatch; the eval default is always cpu.
DEVICE = _env("CH2_DEVICE", "cpu", str)
try:
    _N_CPU = len(os.sched_getaffinity(0))  # respects core pinning (Linux)
except AttributeError:
    _N_CPU = os.cpu_count() or 4
torch.set_num_threads(min(10, max(1, _N_CPU)))


# ----------------------------- metric -----------------------------
from scipy.stats import kendalltau  # noqa: E402


def clipped_tau(a, b):
    t = kendalltau(np.asarray(a, float), np.asarray(b, float)).statistic
    return float(np.clip(0.0 if (t is None or np.isnan(t)) else t, 0.0, 1.0))


def morph_score(pred_df, gt_df):
    per = {c: clipped_tau(pred_df[c].values, gt_df[c].values) for c in COLS}
    return float(np.mean(list(per.values()))), per


# ----------------------------- map render/decode -----------------------------
def map_shape(H, W):
    return (H + STRIDE - 1) // STRIDE, (W + STRIDE - 1) // STRIDE


def _gauss_patch(sigma, radius):
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    return np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2.0 * sigma ** 2))


_DENS_PATCH = _gauss_patch(DENS_SIGMA, int(np.ceil(3 * DENS_SIGMA)))
_r = ATTR_RADIUS
_ax = np.arange(-_r, _r + 1)
_DISK = (_ax[None, :] ** 2 + _ax[:, None] ** 2) <= _r * _r


def _stamp(canvas, cy, cx, patch, mode="add", normalize=False):
    r = patch.shape[0] // 2
    H, W = canvas.shape
    y0, y1, x0, x1 = cy - r, cy + r + 1, cx - r, cx + r + 1
    py0, px0 = max(0, -y0), max(0, -x0)
    py1 = patch.shape[0] - max(0, y1 - H)
    px1 = patch.shape[1] - max(0, x1 - W)
    if py1 <= py0 or px1 <= px0:
        return
    sub = patch[py0:py1, px0:px1]
    if normalize:
        s = sub.sum()
        if s <= 0:
            return
        sub = sub / s
    ys, xs = slice(max(0, y0), min(H, y1)), slice(max(0, x0), min(W, x1))
    if mode == "add":
        canvas[ys, xs] += sub
    else:
        np.maximum(canvas[ys, xs], sub, out=canvas[ys, xs])


def render_maps(cell_rows, H, W):
    mh, mw = map_shape(H, W)
    dens = np.zeros((len(CLASSES), mh, mw), np.float32)
    heat = np.zeros((mh, mw), np.float32)
    amap = np.zeros((mh, mw), np.float32)
    emap = np.zeros((mh, mw), np.float32)
    for (x, y, ci, area, elo) in cell_rows:
        cx = min(max(int(round(x * W / STRIDE)), 0), mw - 1)
        cy = min(max(int(round(y * H / STRIDE)), 0), mh - 1)
        _stamp(dens[ci], cy, cx, _DENS_PATCH, mode="add", normalize=True)
        sig = float(np.clip(0.25 * np.sqrt(max(area, 1.0) / np.pi),
                            HEAT_SIGMA_MIN, HEAT_SIGMA_MAX))
        _stamp(heat, cy, cx, _gauss_patch(sig, int(np.ceil(3 * sig))), mode="max")
        y0, y1 = max(0, cy - _r), min(mh, cy + _r + 1)
        x0, x1 = max(0, cx - _r), min(mw, cx + _r + 1)
        sub = _DISK[_r - (cy - y0): _r + (y1 - cy), _r - (cx - x0): _r + (x1 - cx)]
        amap[y0:y1, x0:x1][sub] = np.log(max(area, 1.0))
        emap[y0:y1, x0:x1][sub] = elo
    return dens, heat, amap, emap


def nms_peaks(heat, k):
    H, W = heat.shape
    p = np.pad(heat, 1, mode="constant", constant_values=-1)
    stack = np.stack([p[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])
    is_max = heat >= stack.max(axis=0) - 1e-9
    ys, xs = np.nonzero(is_max)
    scores = heat[ys, xs]
    order = np.argsort(-scores)[:k]
    return ys[order], xs[order], scores[order]


def _sample3(m, ys, xs):
    H, W = m.shape
    out = np.empty(len(ys), np.float32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        out[i] = m[max(0, y - 1):min(H, y + 2), max(0, x - 1):min(W, x + 2)].mean()
    return out


DENS_THR = 3e-4  # default noise-floor cutoff; re-selected in-script on val slides


def decode_tile(dens, heat, amap, emap, H, W, train_stats, cthr=DENS_THR):
    # zero sub-threshold densities: kills ReLU background-noise integral bias
    dens = np.where(dens > cthr, dens, 0.0)
    per_class = dens.reshape(4, -1).sum(axis=1)
    n_hat = float(per_class.sum())
    out = {}
    out["cellularity"] = 1e5 * n_hat / (W * H)
    out["tumor_frac"] = float(per_class[0] / (n_hat + 1e-6))

    k = int(max(5, round(n_hat)))
    ys, xs, scores = nms_peaks(heat, k)
    keep = scores > 0.05
    if keep.sum() >= 3:
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
    kk = len(ys)

    ts = train_stats
    if kk >= 5:
        pts = np.column_stack([ys, xs]).astype(np.float64) * STRIDE
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn_d = np.sqrt(d2.min(axis=1))
        out["dispersion"] = float(nn_d.mean() * 2.0 * np.sqrt(kk / (W * H)))
    else:
        out["dispersion"] = ts["dispersion_mean"]

    if kk >= 3:
        a_pk = np.exp(_sample3(amap, ys, xs).astype(np.float64))
        m = a_pk.mean()
        out["size_spread"] = float(a_pk.std(ddof=0) / m) if m > 0 else ts["size_spread_med"]
        out["elongation"] = float(_sample3(emap, ys, xs).mean())
    else:
        out["size_spread"] = ts["size_spread_med"]
        out["elongation"] = ts["elongation_med"]

    dtot = dens.sum(axis=0)
    mask = amap > 0.5
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


# ----------------------------- dataset -----------------------------
class CropDataset(Dataset):
    def __init__(self, images, cells_by_img, ids, crops_per_image, seed):
        self.images, self.cells, self.ids = images, cells_by_img, list(ids)
        self.cpi, self.seed, self.epoch = crops_per_image, seed, 0

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

        s = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
        src = int(round(CROP / s))
        s_eff = CROP / src
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

        px = c["x"] * W - x0
        py = c["y"] * H - y0
        inside = (px >= 0) & (px < src) & (py >= 0) & (py < src)
        px, py = px[inside] * s_eff, py[inside] * s_eff
        ci = c["ci"][inside]
        area = c["area"][inside] * (s_eff * s_eff)
        elo = c["elo"][inside]

        k = rng.randint(4)
        fl = rng.randint(2)
        if k:
            crop = np.rot90(crop, k)
            for _ in range(k):
                px, py = py, CROP - 1 - px
        if fl:
            crop = crop[:, ::-1]
            px = CROP - 1 - px

        f = crop.astype(np.float32)
        f *= rng.uniform(0.85, 1.15)
        mean = f.mean()
        f = (f - mean) * rng.uniform(0.85, 1.15) + mean
        gray = f.mean(axis=2, keepdims=True)
        f = gray + (f - gray) * rng.uniform(0.85, 1.15)
        f *= rng.uniform(0.95, 1.05, size=(1, 1, 3))
        crop = np.clip(f, 0, 255).astype(np.uint8)

        dens, heat, amap, emap = render_maps(
            list(zip(px / CROP, py / CROP, ci, np.maximum(area, 1.0), elo)), CROP, CROP)
        tgt = np.concatenate([dens * DENS_SCALE, heat[None], amap[None], emap[None]], 0)
        counts = np.bincount(ci, minlength=len(CLASSES)).astype(np.float32)

        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
            torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.from_numpy(np.ascontiguousarray(tgt)), torch.from_numpy(counts)


# ----------------------------- model -----------------------------
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


class UNetR18(nn.Module):
    def __init__(self, out_ch=7, pretrained=True):
        super().__init__()
        import torchvision
        try:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            enc = torchvision.models.resnet18(weights=w)
        except Exception as e:
            print(f"[model] pretrained download failed ({e}); random init", flush=True)
            enc = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.d4, self.d3 = _block(512 + 256, 256), _block(256 + 128, 128)
        self.d2, self.d1 = _block(128 + 64, 64), _block(64 + 64, 32)
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.l1(self.pool(c1))
        c3, c4, c5 = self.l2(c2), None, None
        c4 = self.l3(c3)
        c5 = self.l4(c4)
        u = F.interpolate(c5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d4(torch.cat([u, c4], 1))
        u = F.interpolate(u, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d3(torch.cat([u, c3], 1))
        u = F.interpolate(u, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d2(torch.cat([u, c2], 1))
        u = F.interpolate(u, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d1(torch.cat([u, c1], 1))
        return self.head(u)


# ----------------------------- inference -----------------------------
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _prep(img):
    H, W = img.shape[:2]
    Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
    f = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    f = np.pad(f, ((0, Hp - H), (0, Wp - W), (0, 0)))
    return torch.from_numpy(f).permute(2, 0, 1)[None]


@torch.no_grad()
def predict_maps(model, img, tta=TTA):
    H, W = img.shape[:2]
    mh, mw = map_shape(H, W)
    x = _prep(img).to(DEVICE)
    views = [(x, None)]
    if tta >= 2:
        views.append((torch.flip(x, [3]), "h"))
    if tta >= 4:
        views += [(torch.flip(x, [2]), "v"), (torch.flip(x, [2, 3]), "hv")]
    acc = None
    for v, tag in views:
        out = model(v)[0].float().cpu().numpy()
        if tag == "h":
            out = out[:, :, ::-1]
        elif tag == "v":
            out = out[:, ::-1, :]
        elif tag == "hv":
            out = out[:, ::-1, ::-1]
        acc = out if acc is None else acc + out
    out = acc / len(views)
    out = out[:, :mh, :mw]
    return np.maximum(out[:4], 0.0) / DENS_SCALE, out[4], out[5], out[6]


def predict_tile_multi(models, img, train_stats, tta=TTA, cthr=DENS_THR):
    """Average raw per-target predictions across models (units are physical)."""
    H, W = img.shape[:2]
    decs = []
    for m in models:
        dens, heat, amap, emap = predict_maps(m, img, tta=tta)
        decs.append(decode_tile(dens, heat, amap, emap, H, W, train_stats, cthr=cthr))
    keys = decs[0].keys()
    return {k: float(np.mean([d[k] for d in decs])) for k in keys}


# ----------------------------- training -----------------------------
def train_one(images, cells_by, tr_ids, va_ids, targets, train_stats, seed,
              stop_min, epochs=EPOCHS, tag="A"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = CropDataset(images, cells_by, tr_ids, CPI, seed)
    workers = 2 if (os.name == "posix") else 0
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=workers,
                    pin_memory=False, drop_last=True, generator=gen,
                    persistent_workers=(workers > 0))
    model = UNetR18(pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    iters = max(1, len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * iters,
                                                       eta_min=LR / 10)
    gt_val = targets.set_index("image_id").loc[va_ids]

    best_tau, best_state = -1.0, None
    ep_t = time.time()
    for ep in range(epochs):
        if elapsed_min() > stop_min:
            print(f"[guard] model {tag}: stop training at epoch {ep} "
                  f"({elapsed_min():.1f} min)", flush=True)
            break
        ds.set_epoch(ep)
        model.train()
        for x, tgt, counts in dl:
            x, tgt, counts = x.to(DEVICE), tgt.to(DEVICE), counts.to(DEVICE)
            pred = model(x)
            l_dens = F.mse_loss(pred[:, :4], tgt[:, :4])
            l_cnt = F.l1_loss(pred[:, :4].sum(dim=(2, 3)) / DENS_SCALE, counts)
            l_heat = F.mse_loss(pred[:, 4], tgt[:, 4])
            mask = (tgt[:, 5] > 0.5).float()
            msum = mask.sum() + 1e-6
            l_a = (torch.abs(pred[:, 5] - tgt[:, 5]) * mask).sum() / msum
            l_e = (torch.abs(pred[:, 6] - tgt[:, 6]) * mask).sum() / msum
            loss = l_dens + 0.01 * l_cnt + l_heat + 0.5 * (l_a + l_e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        do_val = (ep % VAL_EVERY == VAL_EVERY - 1) or (ep == epochs - 1) \
            or (elapsed_min() > stop_min * 0.92)
        msg = f"[{tag}] ep {ep:02d} loss {loss.item():.4f} " \
              f"({(time.time()-ep_t):.0f}s/ep, {elapsed_min():.1f}m)"
        ep_t = time.time()
        if do_val:
            model.eval()
            rows = []
            for iid in va_ids:
                d = predict_tile_multi([model], images[iid], train_stats, tta=1)
                d["image_id"] = iid
                rows.append(d)
            pred = pd.DataFrame(rows).set_index("image_id").loc[va_ids]
            mean_tau, per = morph_score(pred, gt_val)
            msg += f" VAL {mean_tau:.4f} " + " ".join(f"{c[:4]}={per[c]:.3f}" for c in COLS)
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
    return model, best_tau


# ----------------------------- main -----------------------------
def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE} threads={torch.get_num_threads()}", flush=True)

    targets = pd.read_csv(public_dir / "train_targets.csv")
    index = pd.read_csv(public_dir / "train_index.csv")
    cells = pd.read_csv(public_dir / "train_cells.csv")
    queries = pd.read_csv(public_dir / "test_queries.csv")
    ids_all = list(targets["image_id"])

    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    cells["ci"] = cells["cell_type"].map(cls_idx).fillna(3).astype(int)
    cells_by = {}
    for iid, g in cells.groupby("image_id"):
        cells_by[iid] = {"x": g["x"].values.astype(np.float64),
                         "y": g["y"].values.astype(np.float64),
                         "ci": g["ci"].values.astype(np.int64),
                         "area": g["area"].values.astype(np.float64),
                         "elo": g["elongation"].values.astype(np.float64)}
    empty = {"x": np.zeros(0), "y": np.zeros(0), "ci": np.zeros(0, np.int64),
             "area": np.zeros(0), "elo": np.zeros(0)}
    for iid in ids_all:
        cells_by.setdefault(iid, empty)

    print("loading train images...", flush=True)
    images = {}
    for iid in ids_all:
        with Image.open(public_dir / "train_images" / f"{iid}.png") as im:
            images[iid] = np.asarray(im.convert("RGB"))
    print(f"loaded {len(images)} train images ({elapsed_min():.1f}m)", flush=True)

    # grouped splits
    from sklearn.model_selection import GroupShuffleSplit
    groups = index.set_index("image_id").loc[ids_all, "group"].values

    def split(seed):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        tr, va = next(gss.split(ids_all, groups=groups))
        return [ids_all[i] for i in tr], [ids_all[i] for i in va]

    tr_a, va_a = split(SEED)
    if SUBSET:
        tr_a, va_a = tr_a[:SUBSET], va_a[:max(40, SUBSET // 5)]

    tr_tgt = targets.set_index("image_id").loc[tr_a]
    train_stats = {"dispersion_mean": float(tr_tgt["dispersion"].mean()),
                   "size_spread_med": float(tr_tgt["size_spread"].median()),
                   "elongation_med": float(tr_tgt["elongation"].median())}

    # ---- train model A (+ optional B) ----
    model_a, tau_a = train_one(images, cells_by, tr_a, va_a, targets, train_stats,
                               SEED, stop_min=min(MODEL_A_STOP_MIN, TRAIN_STOP_MIN),
                               tag="A")
    models = [model_a]
    if TWO_MODELS and elapsed_min() < TRAIN_STOP_MIN - 12:
        tr_b, va_b = split(SEED + 1)
        if SUBSET:
            tr_b, va_b = tr_b[:SUBSET], va_b[:max(40, SUBSET // 5)]
        model_b, tau_b = train_one(images, cells_by, tr_b, va_b, targets, train_stats,
                                   SEED + 1, stop_min=TRAIN_STOP_MIN, tag="B")
        if tau_b > 0.25:  # sanity: only ensemble a functional model
            models.append(model_b)

    # ---- in-script selection on model A's held-out slides (train data only) ----
    model_a.eval()
    va_maps = {}
    for iid in va_a:
        img = images[iid]
        va_maps[iid] = (predict_maps(model_a, img, tta=1), img.shape[:2])
    gt_val = targets.set_index("image_id").loc[va_a]

    def decode_val(cthr):
        rows = []
        for iid in va_a:
            (dens, heat, amap, emap), (H, W) = va_maps[iid]
            d = decode_tile(dens, heat, amap, emap, H, W, train_stats, cthr=cthr)
            d["image_id"] = iid
            rows.append(d)
        return pd.DataFrame(rows).set_index("image_id").loc[va_a]

    best_cthr, best_ct_tau, va_pred = DENS_THR, -1.0, None
    for cthr in [1e-4, 2e-4, 3e-4, 5e-4]:
        p = decode_val(cthr)
        mt, _ = morph_score(p, gt_val)
        print(f"[select] cthr={cthr:g}: val mean_tau {mt:.4f}", flush=True)
        if mt > best_ct_tau:
            best_cthr, best_ct_tau, va_pred = cthr, mt, p
    print(f"[select] chose cthr={best_cthr:g}", flush=True)

    use_alt = {}
    for c in ["size_spread", "elongation"]:
        t_pri = clipped_tau(va_pred[c], gt_val[c])
        t_alt = clipped_tau(va_pred[f"{c}_alt"], gt_val[c])
        use_alt[c] = bool(t_alt > t_pri)
        print(f"[select] {c}: primary {t_pri:.4f} vs alt {t_alt:.4f} "
              f"-> {'alt' if use_alt[c] else 'primary'}", flush=True)

    # ---- test inference (strictly per-sample; flip TTA only) ----
    print(f"test inference on {len(queries)} tiles ({elapsed_min():.1f}m)...", flush=True)
    out_rows = []
    q_iter = list(queries["query_id"])
    if LIMIT_TEST:
        q_iter = q_iter[:LIMIT_TEST]
    # adaptive budget: measure per-tile cost on first tiles, degrade TTA/models
    # if the projection exceeds the remaining wall budget (per-sample only).
    tta_use, models_use = TTA, models
    probe_n = min(4, len(q_iter))
    tp0 = time.time()
    for qid in q_iter[:probe_n]:
        with Image.open(public_dir / "test_images" / f"{qid}.png") as im:
            img = np.asarray(im.convert("RGB"))
        predict_tile_multi(models_use, img, train_stats, tta=tta_use, cthr=best_cthr)
    cost = (time.time() - tp0) / max(probe_n, 1)
    remain_s = max((TOTAL_BUDGET_MIN - elapsed_min()) * 60.0, 60.0)
    allowed = 0.9 * remain_s / max(len(q_iter), 1)
    if cost > allowed and tta_use > 2:
        tta_use = 2
        cost /= 2
    if cost > allowed and tta_use > 1:
        tta_use = 1
        cost /= 2
    if cost > allowed and len(models_use) > 1:
        models_use = models_use[:1]
        cost /= 2
    print(f"[budget] per-tile {cost:.2f}s allowed {allowed:.2f}s -> "
          f"tta={tta_use} models={len(models_use)}", flush=True)
    for i, qid in enumerate(q_iter):
        with Image.open(public_dir / "test_images" / f"{qid}.png") as im:
            img = np.asarray(im.convert("RGB"))
        d = predict_tile_multi(models_use, img, train_stats, tta=tta_use, cthr=best_cthr)
        rec = {"id": qid}
        for c in COLS:
            key = f"{c}_alt" if use_alt.get(c, False) else c
            rec[c] = d[key]
        out_rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} ({elapsed_min():.1f}m)", flush=True)

    sub = pd.DataFrame(out_rows)[["id"] + COLS]
    # defensive: finite, complete, exactly the query ids in order
    sub = sub.set_index("id").reindex(queries["query_id"]).reset_index() \
        .rename(columns={"query_id": "id"})
    med = targets[COLS].median()
    for c in COLS:
        v = pd.to_numeric(sub[c], errors="coerce")
        sub[c] = v.where(np.isfinite(v), med[c])
    assert len(sub) == len(queries)
    assert np.isfinite(sub[COLS].values).all()
    sub.to_csv(submission_out, index=False)
    try:  # challenge text also names ./working/submission.csv
        Path("working").mkdir(parents=True, exist_ok=True)
        sub.to_csv(Path("working") / "submission.csv", index=False)
    except Exception as e:
        print(f"[warn] secondary write failed: {e}", flush=True)
    print(f"wrote {submission_out} ({len(sub)} rows) at {elapsed_min():.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
