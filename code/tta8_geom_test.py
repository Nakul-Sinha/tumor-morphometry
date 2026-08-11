"""Stub-model D4 geometry verification for exp_base.dump_views (no training).

A stub "model" maps the normalized input (1,3,Hp,Wp) to (1,7,Hp/2,Wp/2) by
average-pooling the RGB mean by 2 -- i.e. the exact spatial contract of the real
UNetR18 head (stride 2), with perfect equivariance. An impulse at input pixel
(y, x) must therefore land at map pixel (y//2, x//2) in EVERY canonicalized
view. Any error in the flip/rot90 inverse composition shows up as a displaced
peak.

Usage: python tta8_geom_test.py     (exits nonzero on any FAIL)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from exp_base import dump_views  # noqa: E402


class StubModel:
    """Call-compatible with dump_views' model argument."""

    def __call__(self, x):
        g = x.mean(dim=1, keepdim=True)              # (1,1,Hp,Wp)
        g = F.avg_pool2d(g, 2)                       # (1,1,Hp/2,Wp/2)
        return g.repeat(1, 7, 1, 1)                  # (1,7,Hp/2,Wp/2)


def main():
    H, W = 300, 260                                  # neither is a multiple of 32
    img = np.full((H, W, 3), 128, np.uint8)          # mid-gray background
    impulses = [(101, 87), (7, 5), (295, 255)]
    for (y, x) in impulses:
        img[y, x] = 255                              # strict max after normalize

    views = dump_views(StubModel(), img)             # (8,7,mh,mw) float16
    mh, mw = views.shape[2], views.shape[3]
    print(f"image {H}x{W} -> map {mh}x{mw}, views {views.shape} {views.dtype}",
          flush=True)

    ok_all = True
    print("view |            impulse (y,x) -> expected | found | d")
    for v in range(8):
        ch0 = views[v, 0].astype(np.float32)
        gy, gx = np.unravel_index(int(np.argmax(ch0)), ch0.shape)
        rowbits, ok_v = [], True
        for (y, x) in impulses:
            ey, ex = int(round(y / 2)), int(round(x / 2))
            y0, y1 = max(0, ey - 2), min(mh, ey + 3)
            x0, x1 = max(0, ex - 2), min(mw, ex + 3)
            win = ch0[y0:y1, x0:x1]
            wy, wx = np.unravel_index(int(np.argmax(win)), win.shape)
            fy, fx = y0 + wy, x0 + wx
            dy, dx = fy - ey, fx - ex
            good = abs(dy) <= 1 and abs(dx) <= 1     # inside the center 3x3
            ok_v = ok_v and good
            rowbits.append(f"({y},{x})->({ey},{ex}) got({fy},{fx}) d=({dy},{dx})"
                           f"{'' if good else ' XX'}")
        ok_all = ok_all and ok_v
        print(f"v{v}  globalmax=({gy},{gx})  " + " | ".join(rowbits) +
              f"   {'PASS' if ok_v else 'FAIL'}", flush=True)

    print("TTA8 GEOMETRY " + ("PASS" if ok_all else "FAIL"), flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
