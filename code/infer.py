"""Full-tile inference: pad to /32, forward, per-sample flip TTA, decode."""
import numpy as np
import torch

from maps import DENS_SCALE, map_shape, decode_tile

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _prep(img):
    H, W = img.shape[:2]
    Hp = (H + 31) // 32 * 32
    Wp = (W + 31) // 32 * 32
    f = img.astype(np.float32) / 255.0
    f = (f - _MEAN) / _STD
    f = np.pad(f, ((0, Hp - H), (0, Wp - W), (0, 0)))
    return torch.from_numpy(f).permute(2, 0, 1)[None]


@torch.no_grad()
def predict_maps(model, img, device, tta=4):
    """img: HxWx3 uint8. Returns dens (4,mh,mw) unscaled, heat, amap, emap."""
    H, W = img.shape[:2]
    mh, mw = map_shape(H, W)
    x = _prep(img).to(device)
    views = [(x, None)]
    if tta >= 2:
        views.append((torch.flip(x, [3]), "h"))
    if tta >= 4:
        views.append((torch.flip(x, [2]), "v"))
        views.append((torch.flip(x, [2, 3]), "hv"))
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
    dens = np.maximum(out[:4], 0.0) / DENS_SCALE
    return dens, out[4], out[5], out[6]


def predict_tile(model, img, device, tta=4, train_stats=None):
    H, W = img.shape[:2]
    dens, heat, amap, emap = predict_maps(model, img, device, tta)
    return decode_tile(dens, heat, amap, emap, H, W, train_stats=train_stats)
