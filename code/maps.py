"""Shared rendering + decoding for the density/heatmap/attribute-map approach.

Conventions (fixed across training, oracle test, and inference):
- Maps live at stride S=2 (half resolution). Map size = ceil(H/S) x ceil(W/S).
- Cell center (x,y) normalized in [0,1] -> full-res pixel (x*W, y*H) -> map (x*W/S, y*H/S).
- Channels: 4 class density maps (sum-normalized Gaussians, integral == class count),
  1 center heatmap (amplitude-1 adaptive Gaussians, max-combined),
  1 attribute map A = log(area) painted in disks, 1 attribute map E = elongation in disks.
- GT arithmetic (verified exact vs train_targets, tau >= 0.9999):
    cellularity = 1e5 * N / (W*H)
    tumor_frac  = n_tumor / N
    size_spread = std(area, ddof=0) / mean(area)
    elongation  = mean(elongation)
    dispersion  = mean_NN_dist_pixels * 2 * sqrt(N / (W*H))
"""
import numpy as np

STRIDE = 2
DENS_SIGMA = 4.0          # half-res px, fixed, for class density maps
ATTR_RADIUS = 4           # half-res px, disk radius for attribute maps
HEAT_SIGMA_MIN, HEAT_SIGMA_MAX = 1.5, 6.0
CLASSES = ["tumor", "immune", "stromal", "other"]
DENS_SCALE = 100.0        # density maps are trained scaled by this


def map_shape(H, W, stride=STRIDE):
    return (H + stride - 1) // stride, (W + stride - 1) // stride


def _gauss_patch(sigma, radius):
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2.0 * sigma ** 2))
    return g


def _stamp(canvas, cy, cx, patch, mode="add", normalize=False):
    """Stamp patch centered at (cy,cx) int coords; clips at borders.
    normalize=True renormalizes the clipped patch to sum 1 (for density maps)."""
    r = patch.shape[0] // 2
    H, W = canvas.shape
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
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
    elif mode == "max":
        np.maximum(canvas[ys, xs], sub, out=canvas[ys, xs])
    else:
        raise ValueError(mode)


_DENS_PATCH = _gauss_patch(DENS_SIGMA, int(np.ceil(3 * DENS_SIGMA)))
_DISK = None


def _disk_mask():
    global _DISK
    if _DISK is None:
        r = ATTR_RADIUS
        ax = np.arange(-r, r + 1)
        _DISK = (ax[None, :] ** 2 + ax[:, None] ** 2) <= r * r
    return _DISK


def render_maps(cell_rows, H, W, stride=STRIDE):
    """cell_rows: structured arrays/lists of (x, y, class_idx, area, elongation).
    Returns dict of float32 maps at map_shape(H, W).
    Density maps are NOT pre-scaled here (integral == count); scale in the dataset."""
    mh, mw = map_shape(H, W, stride)
    dens = np.zeros((len(CLASSES), mh, mw), np.float32)
    heat = np.zeros((mh, mw), np.float32)
    amap = np.zeros((mh, mw), np.float32)
    emap = np.zeros((mh, mw), np.float32)
    disk = _disk_mask()
    r = ATTR_RADIUS

    for (x, y, ci, area, elo) in cell_rows:
        cx = int(round(x * W / stride))
        cy = int(round(y * H / stride))
        cx = min(max(cx, 0), mw - 1)
        cy = min(max(cy, 0), mh - 1)
        _stamp(dens[ci], cy, cx, _DENS_PATCH, mode="add", normalize=True)
        sig = float(np.clip(0.25 * np.sqrt(max(area, 1.0) / np.pi), HEAT_SIGMA_MIN, HEAT_SIGMA_MAX))
        _stamp(heat, cy, cx, _gauss_patch(sig, int(np.ceil(3 * sig))), mode="max")
        # attribute disks (overwrite where overlapping)
        y0, y1 = max(0, cy - r), min(mh, cy + r + 1)
        x0, x1 = max(0, cx - r), min(mw, cx + r + 1)
        sub = disk[r - (cy - y0): r + (y1 - cy), r - (cx - x0): r + (x1 - cx)]
        amap[y0:y1, x0:x1][sub] = np.log(max(area, 1.0))
        emap[y0:y1, x0:x1][sub] = elo
    return {"dens": dens, "heat": heat, "amap": amap, "emap": emap}


def nms_peaks(heat, k):
    """3x3 max NMS, return top-k (y, x, score). Pure numpy."""
    H, W = heat.shape
    p = np.pad(heat, 1, mode="constant", constant_values=-1)
    stack = np.stack([p[dy:dy + H, dx:dx + W]
                      for dy in range(3) for dx in range(3)])
    is_max = heat >= stack.max(axis=0) - 1e-9
    ys, xs = np.nonzero(is_max)
    scores = heat[ys, xs]
    order = np.argsort(-scores)[:k]
    return ys[order], xs[order], scores[order]


def _sample3(m, ys, xs):
    """3x3 mean around each (y,x)."""
    H, W = m.shape
    out = np.empty(len(ys), np.float32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        out[i] = m[y0:y1, x0:x1].mean()
    return out


def decode_tile(dens, heat, amap, emap, H, W, stride=STRIDE,
                train_stats=None):
    """dens: (4, mh, mw) unscaled (integral == count). Returns dict of 5 targets
    plus alternative estimators (keys with suffix _alt)."""
    dens = np.maximum(dens, 0.0)
    per_class = dens.reshape(4, -1).sum(axis=1)
    n_hat = float(per_class.sum())
    out = {}
    out["cellularity"] = 1e5 * n_hat / (W * H)
    out["tumor_frac"] = float(per_class[0] / (n_hat + 1e-6))

    k = int(max(5, round(n_hat)))
    ys, xs, scores = nms_peaks(heat, k)
    # keep peaks with a minimal score to avoid reading noise floor
    keep = scores > 0.05
    if keep.sum() >= 3:
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
    kk = len(ys)

    ts = train_stats or {}
    # dispersion: Clark-Evans in PIXEL coords (exact GT arithmetic)
    if kk >= 5:
        pts = np.column_stack([ys, xs]).astype(np.float64) * stride
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(d2.min(axis=1))
        out["dispersion"] = float(nn.mean() * 2.0 * np.sqrt(kk / (W * H)))
    else:
        out["dispersion"] = ts.get("dispersion_mean", 1.26)

    # attributes sampled at peaks
    if kk >= 3:
        a_pk = np.exp(_sample3(amap, ys, xs).astype(np.float64))
        m = a_pk.mean()
        out["size_spread"] = float(a_pk.std(ddof=0) / m) if m > 0 else ts.get("size_spread_med", 0.51)
        out["elongation"] = float(_sample3(emap, ys, xs).mean())
    else:
        out["size_spread"] = ts.get("size_spread_med", 0.51)
        out["elongation"] = ts.get("elongation_med", 1.32)

    # alternative estimators: density-weighted moments masked to painted regions
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
