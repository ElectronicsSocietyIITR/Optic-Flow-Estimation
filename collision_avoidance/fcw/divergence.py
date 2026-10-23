"""Flow divergence per box from an affine fit, after removing camera rotation.

Inside a box, u = a0 + a1*x + a2*y and v = b0 + b1*x + b2*y; div = a1 + b2.
For an object approaching at constant speed, the translational flow of a
fronto-parallel surface has div = 2 / TTC_frames, so eta = div / (2 * k * dt).

The fit needs only 12 sums per box (FLOW_TERMS), which the PL engine
accumulates in the same pass as the Horn sums.
"""

from dataclasses import dataclass

import numpy as np

from .measurement import Measurement

FLOW_TERMS = ("n", "x", "y", "xx", "xy", "yy", "u", "v", "xu", "yu", "xv", "yv")


@dataclass(frozen=True)
class DivergenceConfig:
    shrink: float = 0.10
    flow_noise_px: float = 0.3   # assumed per-pixel flow noise, sets the variance
    inlier_px: float = 1.0       # robust refit: flow residual that still counts as the object
    ransac_iterations: int = 64
    ransac_stride: int = 2       # subsample pixels for the RANSAC hypotheses
    min_pixels: int = 50
    min_inlier_share: float = 0.5  # predicted-fit mask below this share of the box: fall back to RANSAC
    max_fit_age: int = 1           # frames: only last frame's fit predicts this frame's flow


def flow_moments(flow: np.ndarray, box, principal_point, mask: np.ndarray | None = None, shrink: float = 0.10) -> np.ndarray | None:
    """The 12 FLOW_TERMS sums over the shrunk box (float64), or None if empty."""
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    mx, my = shrink / 2 * (x2 - x1), shrink / 2 * (y2 - y1)
    ix1, iy1 = max(0, int(np.ceil(x1 + mx))), max(0, int(np.ceil(y1 + my)))
    ix2, iy2 = min(w, int(np.floor(x2 - mx))), min(h, int(np.floor(y2 - my)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    ys, xs = np.mgrid[iy1:iy2, ix1:ix2]
    x = (xs - principal_point[0]).astype(np.float64)
    y = (ys - principal_point[1]).astype(np.float64)
    u = flow[iy1:iy2, ix1:ix2, 0].astype(np.float64)
    v = flow[iy1:iy2, ix1:ix2, 1].astype(np.float64)
    keep = np.ones(x.shape, bool) if mask is None else mask[iy1:iy2, ix1:ix2]
    x, y, u, v = x[keep], y[keep], u[keep], v[keep]
    return np.array([x.size, x.sum(), y.sum(), (x * x).sum(), (x * y).sum(), (y * y).sum(),
                     u.sum(), v.sum(), (x * u).sum(), (y * u).sum(), (x * v).sum(), (y * v).sum()])


FLOW_SCALE = 64  # PL engine flow format: int16 in 1/64 px (+-512 px)


def quantise_flow(flow: np.ndarray) -> np.ndarray:
    """Flow (H, W, 2) in px -> int16 in 1/FLOW_SCALE px, as streamed to the PL engine."""
    return np.clip(np.round(flow * FLOW_SCALE), -32768, 32767).astype(np.int16)


def flow_moments_fixed(flow_q: np.ndarray, box, principal_point, shrink: float = 0.10,
                       mask: np.ndarray | None = None) -> np.ndarray | None:
    """The 12 FLOW_TERMS as exact int64 sums (u, v in 1/FLOW_SCALE px; x, y from the rounded principal point).

    Golden model of the PL engine's flow-moment accumulators. Divide the u/v terms
    (indices 6..11) by FLOW_SCALE to get flow_moments() of the dequantised flow.
    """
    h, w = flow_q.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    mx, my = shrink / 2 * (x2 - x1), shrink / 2 * (y2 - y1)
    ix1, iy1 = max(0, int(np.ceil(x1 + mx))), max(0, int(np.ceil(y1 + my)))
    ix2, iy2 = min(w, int(np.floor(x2 - mx))), min(h, int(np.floor(y2 - my)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    px, py = int(round(principal_point[0])), int(round(principal_point[1]))
    ys, xs = np.mgrid[iy1:iy2, ix1:ix2]
    x, y = (xs - px).astype(np.int64), (ys - py).astype(np.int64)
    u = flow_q[iy1:iy2, ix1:ix2, 0].astype(np.int64)
    v = flow_q[iy1:iy2, ix1:ix2, 1].astype(np.int64)
    if mask is not None:
        keep = mask[iy1:iy2, ix1:ix2]
        x, y, u, v = x[keep], y[keep], u[keep], v[keep]
    return np.array([x.size, x.sum(), y.sum(), (x * x).sum(), (x * y).sum(), (y * y).sum(),
                     u.sum(), v.sum(), (x * u).sum(), (y * u).sum(), (x * v).sum(), (y * v).sum()], dtype=np.int64)


def dequantise_moments(m_fixed: np.ndarray) -> np.ndarray:
    """flow_moments_fixed -> flow_moments units (u, v terms back to px)."""
    m = m_fixed.astype(np.float64)
    m[6:] /= FLOW_SCALE
    return m


FIT_SHIFT = 16  # fraction bits of the fixed-point affine fit
FitQ = tuple[int, int, int, int, int, int]


def fit_to_fixed(a: np.ndarray, b: np.ndarray) -> FitQ:
    """Affine fit (a0, a1, a2), (b0, b1, b2) in px and px/px -> int64 in 1/FLOW_SCALE px, scaled by 2^FIT_SHIFT."""
    s = FLOW_SCALE * (1 << FIT_SHIFT)
    return tuple(int(np.round(float(c) * s)) for c in (*a, *b))


def predicted_mask_fixed(flow_q: np.ndarray, rect, principal_point_int, fit_q: FitQ, inlier_q: int) -> np.ndarray:
    """Inliers of a predicted affine flow over rect = (x1, y1, x2, y2), exactly as the PL engine decides them.

    Prediction P = a0 + a1 x + a2 y in int64 (row by row it only grows by a1: adders in
    hardware); residual = ((u << FIT_SHIFT) - P) >> FIT_SHIFT in 1/FLOW_SCALE px, clamped to
    int16; inlier when du^2 + dv^2 < inlier_q^2 (the same 1 px test as robust_mask).
    """
    x1, y1, x2, y2 = rect
    ys, xs = np.mgrid[y1:y2, x1:x2]
    x, y = (xs - principal_point_int[0]).astype(np.int64), (ys - principal_point_int[1]).astype(np.int64)
    a0, a1, a2, b0, b1, b2 = (np.int64(c) for c in fit_q)
    u = flow_q[y1:y2, x1:x2, 0].astype(np.int64) << FIT_SHIFT
    v = flow_q[y1:y2, x1:x2, 1].astype(np.int64) << FIT_SHIFT
    du = np.clip((u - (a0 + a1 * x + a2 * y)) >> FIT_SHIFT, -32767, 32767)
    dv = np.clip((v - (b0 + b1 * x + b2 * y)) >> FIT_SHIFT, -32767, 32767)
    return du * du + dv * dv < np.int64(inlier_q) * inlier_q


def divergence_mask(flow: np.ndarray, flow_q: np.ndarray, box, principal_point, prev_fit, fit_age: int,
                    cfg: DivergenceConfig = DivergenceConfig()) -> tuple[np.ndarray, bool]:
    """Object pixels in the box: last frame's fit predicts them; RANSAC when there is no usable fit.

    Returns (mask over the frame, whether the prediction was used).
    """
    h, w = flow.shape[:2]
    rect = flow_rect(box, cfg.shrink, w, h)
    if prev_fit is not None and fit_age <= cfg.max_fit_age and rect is not None:
        pp = (int(round(principal_point[0])), int(round(principal_point[1])))
        inside = predicted_mask_fixed(flow_q, rect, pp, fit_to_fixed(*prev_fit), int(round(cfg.inlier_px * FLOW_SCALE)))
        if inside.mean() >= cfg.min_inlier_share:
            mask = np.zeros((h, w), bool)
            mask[rect[1]:rect[3], rect[0]:rect[2]] = inside
            return mask, True
    return robust_mask(flow, box, principal_point, cfg), False


def flow_rect(box, shrink: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Integer [x1, x2) x [y1, y2) that flow_moments / flow_moments_fixed sum over."""
    x1, y1, x2, y2 = map(float, box)
    mx, my = shrink / 2 * (x2 - x1), shrink / 2 * (y2 - y1)
    ix1, iy1 = max(0, int(np.ceil(x1 + mx))), max(0, int(np.ceil(y1 + my)))
    ix2, iy2 = min(width, int(np.floor(x2 - mx))), min(height, int(np.floor(y2 - my)))
    return (ix1, iy1, ix2, iy2) if ix2 > ix1 and iy2 > iy1 else None


def affine_from_moments(m: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(a0, a1, a2), (b0, b1, b2) of the least-squares affine flow."""
    n, sx, sy, sxx, sxy, syy, su, sv, sxu, syu, sxv, syv = m
    A = np.array([[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]])
    if n < 3 or abs(np.linalg.det(A)) < 1e-9:
        return None
    return np.linalg.solve(A, [su, sxu, syu]), np.linalg.solve(A, [sv, sxv, syv])


def divergence_ttc(m: np.ndarray, box, rotation_div: float, k: int, dt: float,
                   cfg: DivergenceConfig = DivergenceConfig()) -> Measurement | None:
    """Inverse TTC from flow moments; rotation_div is subtracted (see ego_rotation.rotation_divergence)."""
    if m is None or m[0] < cfg.min_pixels:
        return None
    fit = affine_from_moments(m)
    if fit is None:
        return None
    a, b = fit
    div = a[1] + b[2] - rotation_div
    w, h = box[2] - box[0], box[3] - box[1]
    var_div = cfg.flow_noise_px**2 * 12.0 / m[0] * (1.0 / w**2 + 1.0 / h**2)
    span = 2.0 * k * dt
    return Measurement("divergence", float(div / span), float(var_div / span**2))


_LCG_MASK = (1 << 64) - 1


def ransac_samples(n: int, iterations: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """Index triples from a fixed 64-bit LCG (Knuth MMIX constants).

    Deliberately simple so the C++ port and the HLS engine draw exactly the same
    samples as this reference: numpy's generator cannot be reproduced there.
    """
    state = (seed ^ 0x853C49E6748FEA9B) & _LCG_MASK
    out = []
    for _ in range(iterations):
        triple: list[int] = []
        while len(triple) < 3:
            state = (state * 6364136223846793005 + 1442695040888963407) & _LCG_MASK
            value = (state >> 33) % n
            if value not in triple:
                triple.append(value)
        out.append(tuple(triple))
    return out


def robust_mask(flow: np.ndarray, box, principal_point, cfg: DivergenceConfig = DivergenceConfig()) -> np.ndarray:
    """Pixels that follow the dominant affine flow in the box (RANSAC; CPU-side refit).

    A plain least-squares fit is dragged towards background pixels leaking into
    the box before any residual test can reject them, so hypotheses come from
    random 3-pixel samples and the one with most inliers wins.
    """
    h, w = flow.shape[:2]
    mask = np.ones((h, w), bool)
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return mask
    ys, xs = np.mgrid[y1:y2, x1:x2]
    x, y = xs - principal_point[0], ys - principal_point[1]
    u, v = flow[y1:y2, x1:x2, 0], flow[y1:y2, x1:x2, 1]
    s = cfg.ransac_stride
    X = np.stack([np.ones(x[::s, ::s].size), x[::s, ::s].ravel(), y[::s, ::s].ravel()], axis=1)
    U, V = u[::s, ::s].ravel(), v[::s, ::s].ravel()
    best, best_count = None, -1
    for sample in ransac_samples(len(X), cfg.ransac_iterations):
        idx = list(sample)
        try:
            a = np.linalg.solve(X[idx], U[idx])
            b = np.linalg.solve(X[idx], V[idx])
        except np.linalg.LinAlgError:
            continue
        count = int((np.hypot(X @ a - U, X @ b - V) < cfg.inlier_px).sum())
        if count > best_count:
            best, best_count = (a, b), count
    if best is None:
        return mask
    a, b = best
    r = np.hypot(u - (a[0] + a[1] * x + a[2] * y), v - (b[0] + b[1] * x + b[2] * y))
    mask[y1:y2, x1:x2] = r < cfg.inlier_px
    return mask
