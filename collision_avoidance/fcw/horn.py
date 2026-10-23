"""Horn direct time-to-contact, as integer sums the PL engine can produce.

Model for a flat object facing the camera (Horn, Fang & Masaki 2007):
    A*Ex + B*Ey + C*G + Et = 0,   G = x*Ex + y*Ey,
with x, y measured from the principal point. Moving the origin only
reparameterises A and B, so every box can share per-pixel products.

Integer definitions (the HLS engine must match these bit for bit):
    S  = prev + curr                       (uint8 + uint8, 9 bit)
    Ex = Sobel_x(S), Ey = Sobel_y(S)       (3x3, 12 bit signed; interior pixels only)
    Et = curr - prev                       (9 bit signed)
Sobel of the frame sum is 16x the mean-intensity derivative, so the fitted C'
is 1/16 of the expansion rate per frame: C = 16 * C'.
The 10 sums per box: Ex^2, ExEy, ExG, Ey^2, EyG, G^2, ExEt, EyEt, GEt, Et^2.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .measurement import Measurement

HORN_TERMS = ("ExEx", "ExEy", "ExG", "EyEy", "EyG", "GG", "ExEt", "EyEt", "GEt", "EtEt")
SOBEL_GAIN = 16


@dataclass(frozen=True)
class HornConfig:
    grad_threshold_l1: int = 64     # |Ex| + |Ey| must exceed this (Sobel-of-sum units)
    shrink: float = 0.10            # use (1 - shrink) of the box, to stay on the object
    max_condition: float = 1e4      # of the column-normalised normal matrix
    max_rel_residual: float = 0.5   # residual / sum(Et^2)
    min_pixels: int = 100


def gradients(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ex, Ey, Et as int64 arrays; Ex/Ey are 0 on the 1-pixel border."""
    s = prev.astype(np.int64) + curr.astype(np.int64)
    ex = np.zeros_like(s)
    ey = np.zeros_like(s)
    ex[1:-1, 1:-1] = (s[:-2, 2:] + 2 * s[1:-1, 2:] + s[2:, 2:]) - (s[:-2, :-2] + 2 * s[1:-1, :-2] + s[2:, :-2])
    ey[1:-1, 1:-1] = (s[2:, :-2] + 2 * s[2:, 1:-1] + s[2:, 2:]) - (s[:-2, :-2] + 2 * s[:-2, 1:-1] + s[:-2, 2:])
    return ex, ey, curr.astype(np.int64) - prev.astype(np.int64)


def inner_box(box, shrink: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Integer pixel range [x1, x2) x [y1, y2) of the shrunk box, clipped to the Sobel interior."""
    x1, y1, x2, y2 = map(float, box)
    mx, my = shrink / 2 * (x2 - x1), shrink / 2 * (y2 - y1)
    ix1, iy1 = max(1, int(np.ceil(x1 + mx))), max(1, int(np.ceil(y1 + my)))
    ix2, iy2 = min(width - 1, int(np.floor(x2 - mx))), min(height - 1, int(np.floor(y2 - my)))
    return (ix1, iy1, ix2, iy2) if ix2 > ix1 and iy2 > iy1 else None


def horn_sums(prev: np.ndarray, curr: np.ndarray, box, principal_point, cfg: HornConfig = HornConfig()) -> tuple[np.ndarray, int] | None:
    """The 10 int64 sums of HORN_TERMS over the box, and the pixel count."""
    h, w = curr.shape
    rng = inner_box(box, cfg.shrink, w, h)
    if rng is None:
        return None
    x1, y1, x2, y2 = rng
    ex, ey, et = gradients(prev[y1 - 1:y2 + 1, x1 - 1:x2 + 1], curr[y1 - 1:y2 + 1, x1 - 1:x2 + 1])
    ex, ey, et = ex[1:-1, 1:-1], ey[1:-1, 1:-1], et[1:-1, 1:-1]
    px, py = int(round(principal_point[0])), int(round(principal_point[1]))
    ys, xs = np.mgrid[y1:y2, x1:x2]
    g = (xs - px) * ex + (ys - py) * ey
    keep = (np.abs(ex) + np.abs(ey)) > cfg.grad_threshold_l1
    ex, ey, et, g = ex[keep], ey[keep], et[keep], g[keep]
    sums = np.array([(ex * ex).sum(), (ex * ey).sum(), (ex * g).sum(), (ey * ey).sum(), (ey * g).sum(),
                     (g * g).sum(), (ex * et).sum(), (ey * et).sum(), (g * et).sum(), (et * et).sum()], dtype=np.int64)
    return sums, int(keep.sum())


def horn_solve(sums: np.ndarray, n: int, dt: float, cfg: HornConfig = HornConfig()) -> Measurement | None:
    """Inverse TTC from the box sums; None when the fit is unreliable."""
    if n < cfg.min_pixels:
        return None
    s = sums.astype(np.float64)
    M = np.array([[s[0], s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], s[5]]])
    q = np.array([s[6], s[7], s[8]])
    diag = np.sqrt(np.diag(M))
    if np.any(diag <= 0):
        return None
    Mn = M / np.outer(diag, diag)
    if np.linalg.cond(Mn) > cfg.max_condition:
        return None
    p = np.linalg.solve(M, -q)                       # (A', B', C')
    residual = s[9] + 2 * p @ q + p @ M @ p          # sum of squared equation errors
    if s[9] <= 0 or residual / s[9] > cfg.max_rel_residual:
        return None
    sigma2 = max(residual, 0.0) / max(n - 3, 1)
    var_c = sigma2 * np.linalg.inv(M)[2, 2]
    eta = SOBEL_GAIN * p[2] / dt
    return Measurement("horn", float(eta), float(SOBEL_GAIN**2 * var_c / dt**2))


def choose_level(eta_estimate: float | None, box, center_speed_px_s: float, dt: float, levels: int = 3) -> int:
    """Pyramid level at which the motion inside the box stays under about 1 px."""
    half_diag = 0.5 * float(np.hypot(box[2] - box[0], box[3] - box[1]))
    motion = (abs(eta_estimate) if eta_estimate else 0.0) * dt * half_diag + center_speed_px_s * dt
    level = 0
    while motion / 2**level > 1.0 and level < levels - 1:
        level += 1
    return level


def downsample(gray: np.ndarray, level: int) -> np.ndarray:
    """Gaussian pyramid level, replicated borders: exactly the fabric's Vitis pyrDown (hls/preprocess)."""
    for _ in range(level):
        gray = cv2.pyrDown(gray, borderType=cv2.BORDER_REPLICATE)
    return gray


def horn_at_level(prev: np.ndarray, curr: np.ndarray, box, principal_point, level: int, dt: float,
                  cfg: HornConfig = HornConfig()) -> Measurement | None:
    """horn_sums + horn_solve on pyramid level `level` (box and principal point scaled to it)."""
    f = 2.0**-level
    found = horn_sums(downsample(prev, level), downsample(curr, level), np.asarray(box, float) * f,
                      (principal_point[0] * f, principal_point[1] * f), cfg)
    return None if found is None else horn_solve(*found, dt, cfg)
