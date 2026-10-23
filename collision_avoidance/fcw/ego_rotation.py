"""Camera rotation and heading (focus of expansion) from a dense flow field.

Only rotation is ever removed from the flow: forward translation is what makes
obstacles ahead expand, so it is the collision signal and must stay.

Rotation R maps camera coordinates prev -> curr (see synth.py for conventions).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EgoRotationConfig:
    method: str = "linear"            # "linear": rotation + FOE least squares; "ransac": essential matrix / homography
    grid_step_px: int = 16
    box_margin: float = 0.10          # enlarge excluded boxes by this fraction per side
    hood_rows: int = 0                # bottom image rows hidden by the car's hood
    ransac_threshold_px: float = 1.0
    ransac_confidence: float = 0.999
    rotation_only_tol: float = 0.01   # max |singular value - 1| of K^-1 H K for a pure rotation
    rotation_only_share: float = 0.9  # homography must explain this share of E's inliers
    min_inlier_ratio: float = 0.5
    max_jump_rad: float = np.radians(1.0)  # per-frame change of rotation angle accepted
    min_points: int = 30
    min_foe_tz: float = 0.5           # |t_z| of the unit translation needed for a usable FOE
    max_iters: int = 200              # RANSAC cap: no accuracy loss on KITTI, 2.5x faster there
    pose_points: int = 64             # evenly spread inliers given to recoverPose (0 = all): it only picks 1 of 4 poses
    foe_smoothing: float = 0.05       # EMA weight of the newest FOE (~0.7 s): follows bends, not per-frame jitter
    linear_rounds: int = 5            # Gauss-Newton rounds of the linear method
    huber_px: float = 1.0             # robust (Cauchy) weight scale and inlier threshold (px)
    stationary_px: float = 0.15       # flow left after removing rotation below this (at the quantile): stopped
    stationary_quantile: float = 0.75 # not the median: distant background alone can be half the points
    refit_px: float = 0.3             # final rounds: plain least squares over points within this residual
    refit_rounds: int = 2


@dataclass
class EgoMotion:
    R: np.ndarray                 # 3x3, prev -> curr
    rvec: np.ndarray              # Rodrigues vector of R, radians over the frame pair
    foe: np.ndarray | None        # (x, y) px; None when stationary or not moving forward
    inlier_ratio: float
    stationary: bool
    valid: bool                   # False: low confidence, R held from an earlier frame

    def omega_rps(self, dt: float) -> np.ndarray:
        return self.rvec / dt

    @staticmethod
    def identity() -> "EgoMotion":
        return EgoMotion(np.eye(3), np.zeros(3), None, 0.0, True, False)


def grid_samples(flow: np.ndarray, exclude_boxes, cfg: EgoRotationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Matched points (prev, curr) on a regular grid, skipping boxes and the hood."""
    h, w = flow.shape[:2]
    s = cfg.grid_step_px
    ys, xs = np.mgrid[s // 2:h - cfg.hood_rows:s, s // 2:w:s]
    keep = np.ones(xs.shape, bool)
    for x1, y1, x2, y2 in exclude_boxes:
        mx, my = cfg.box_margin * (x2 - x1), cfg.box_margin * (y2 - y1)
        keep &= ~((xs >= x1 - mx) & (xs <= x2 + mx) & (ys >= y1 - my) & (ys <= y2 + my))
    xs, ys = xs[keep], ys[keep]
    prev = np.stack([xs, ys], axis=1).astype(np.float64)
    return prev, prev + flow[ys, xs].astype(np.float64)


def _rotation_from_homography(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, float]:
    """Nearest rotation to K^-1 H K and how far that matrix is from a rotation."""
    A = np.linalg.inv(K) @ H @ K
    A = A / np.cbrt(np.linalg.det(A))
    U, sv, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return R, float(np.abs(sv - 1.0).max())


def _angle(R: np.ndarray) -> float:
    return float(np.linalg.norm(cv2.Rodrigues(R)[0]))


def estimate(prev_pts: np.ndarray, curr_pts: np.ndarray, K: np.ndarray, cfg: EgoRotationConfig) -> EgoMotion | None:
    """One unsmoothed estimate, or None when the points cannot support one."""
    n = len(prev_pts)
    if n < cfg.min_points:
        return None
    E, e_mask = cv2.findEssentialMat(prev_pts, curr_pts, K, cv2.RANSAC, cfg.ransac_confidence, cfg.ransac_threshold_px,
                                     maxIters=cfg.max_iters)
    e_ratio = 0.0 if E is None else float(e_mask.sum()) / n
    # Pure rotation (or a stopped car) is degenerate for E: prefer the rotation
    # homography when it explains about as many points as E does.
    H, h_mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, cfg.ransac_threshold_px, maxIters=cfg.max_iters)
    if H is not None:
        R_h, deviation = _rotation_from_homography(H, K)
        h_ratio = float(h_mask.sum()) / n
        if deviation < cfg.rotation_only_tol and h_ratio >= cfg.rotation_only_share * e_ratio:
            return EgoMotion(R_h, cv2.Rodrigues(R_h)[0].ravel(), None, h_ratio, True, True)
    if E is None:
        return None
    E = E[:3]  # findEssentialMat may stack several solutions
    # recoverPose's own mask also drops points beyond 50 baselines (far background),
    # so the RANSAC inlier share of E is the confidence measure.
    inliers = np.flatnonzero(e_mask.ravel())
    if cfg.pose_points and len(inliers) > cfg.pose_points:
        # Triangulating every inlier was half the frame time; a spread subset decides as well.
        sel = inliers[np.linspace(0, len(inliers) - 1, cfg.pose_points).astype(int)]
        _, R, t, _ = cv2.recoverPose(E, prev_pts[sel], curr_pts[sel], K)
    else:
        _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, K, mask=e_mask.copy())
    ratio = e_ratio
    t = t.ravel() / np.linalg.norm(t)
    foe = None
    if abs(t[2]) >= cfg.min_foe_tz:
        p = K @ t
        foe = p[:2] / p[2]
    return EgoMotion(R, cv2.Rodrigues(R)[0].ravel(), foe, ratio, False, True)


def _rotation_jacobian(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d(flow)/d(omega) for a small rotation, normalised coordinates: (n, 3) rows for u and for v."""
    one = np.ones_like(x)
    return np.stack([-x * y, one + x * x, -y], 1), np.stack([-(one + y * y), x * y, x], 1)


def _rotate(R: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.stack([x, y, np.ones_like(x)], 1) @ R.T
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]


def _cauchy(r: np.ndarray, k: float) -> np.ndarray:
    """Redescending weights: bad flow (object edges, horizon) gets almost none, unlike Huber."""
    return 1.0 / (1.0 + (r / k) ** 2)


def estimate_linear(prev_pts: np.ndarray, curr_pts: np.ndarray, K: np.ndarray, cfg: EgoRotationConfig,
                    foe_init=None) -> EgoMotion | None:
    """Rotation and FOE by robust least squares (Bruss-Horn style), no RANSAC.

    Once the rotation is removed, a static point moves straight away from the FOE:
    (flow - rotational flow) x (p - FOE) = 0. That residual is linear in a small
    rotation update and in the FOE, so a few Gauss-Newton rounds over the grid solve
    both (5 unknowns). If removing the rotation alone leaves almost no flow, the car
    is stopped: no FOE. Cauchy weights keep bad flow (object edges, horizon) from pulling
    the fit. Points are in px; the maths runs in normalised coordinates.
    """
    n = len(prev_pts)
    if n < cfg.min_points:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y = (prev_pts[:, 0] - cx) / fx, (prev_pts[:, 1] - cy) / fy
    xc, yc = (curr_pts[:, 0] - cx) / fx, (curr_pts[:, 1] - cy) / fy
    Ju, Jv = _rotation_jacobian(x, y)
    px = 0.5 * (fx + fy)                     # normalised -> px for thresholds

    # Rotation only (also the stationary test): flow = rotational flow.
    R = np.eye(3)
    for _ in range(cfg.linear_rounds):
        xr, yr = _rotate(R, x, y)
        ru, rv = xc - xr, yc - yr
        w = np.tile(_cauchy(np.hypot(ru, rv) * px, cfg.huber_px), 2)
        A, b = np.vstack([Ju, Jv]), np.concatenate([ru, rv])
        d = np.linalg.solve((A * w[:, None]).T @ A + 1e-12 * np.eye(3), (A * w[:, None]).T @ b)
        R = cv2.Rodrigues(d)[0] @ R
    xr, yr = _rotate(R, x, y)
    left = np.hypot(xc - xr, yc - yr) * px
    if np.quantile(left, cfg.stationary_quantile) < cfg.stationary_px:
        return EgoMotion(R, cv2.Rodrigues(R)[0].ravel(), None, float((left < cfg.huber_px).mean()), True, True)

    # Driving: rotation update and FOE together.
    R = np.eye(3)
    e = np.zeros(2) if foe_init is None else np.array([(foe_init[0] - cx) / fx, (foe_init[1] - cy) / fy])
    for round_ in range(cfg.linear_rounds + cfg.refit_rounds):
        xr, yr = _rotate(R, x, y)
        ru, rv = xc - xr, yc - yr
        dx, dy = xr - e[0], yr - e[1]
        dist = np.maximum(np.hypot(dx, dy), 0.05)
        res = (ru * dy - rv * dx) / dist                          # flow across the radial line (normalised)
        J = np.column_stack([(-Ju * dy[:, None] + Jv * dx[:, None]), rv, -ru]) / dist[:, None]
        if round_ < cfg.linear_rounds:
            w = _cauchy(res * px, cfg.huber_px)
        else:                                                     # RANSAC's last step: refit on clear inliers only
            w = (np.abs(res) * px < cfg.refit_px).astype(float)
            if w.sum() < cfg.min_points:
                break
        step = np.linalg.solve((J * w[:, None]).T @ J + 1e-12 * np.eye(5), -(J * w[:, None]).T @ res)
        R = cv2.Rodrigues(step[:3])[0] @ R
        e = e + step[3:]
    xr, yr = _rotate(R, x, y)
    ru, rv = xc - xr, yc - yr
    dx, dy = xr - e[0], yr - e[1]
    across = np.abs(ru * dy - rv * dx) / np.maximum(np.hypot(dx, dy), 0.05) * px
    ratio = float((across < cfg.huber_px).mean())
    forward = float(np.sum(ru * dx + rv * dy)) > 0                # expansion, not contraction
    max_e = np.sqrt(1.0 / cfg.min_foe_tz ** 2 - 1.0)            # same heading limit as |t_z| >= min_foe_tz
    foe = np.array([e[0] * fx + cx, e[1] * fy + cy]) if forward and np.hypot(*e) <= max_e else None
    return EgoMotion(R, cv2.Rodrigues(R)[0].ravel(), foe, ratio, False, True)


class EgoRotationEstimator:
    """Per-frame estimate with outlier holding and FOE smoothing."""

    def __init__(self, K: np.ndarray, cfg: EgoRotationConfig = EgoRotationConfig()):
        self.K = K
        self.cfg = cfg
        self.last = EgoMotion.identity()

    def update(self, flow: np.ndarray, exclude_boxes) -> EgoMotion:
        prev_pts, curr_pts = grid_samples(flow, exclude_boxes, self.cfg)
        if self.cfg.method == "linear":
            est = estimate_linear(prev_pts, curr_pts, self.K, self.cfg, self.last.foe)
        else:
            est = estimate(prev_pts, curr_pts, self.K, self.cfg)
        if est is None or est.inlier_ratio < self.cfg.min_inlier_ratio:
            return self._hold()
        if self.last.valid and abs(_angle(est.R) - _angle(self.last.R)) > self.cfg.max_jump_rad:
            return self._hold()
        if est.foe is not None and self.last.foe is not None:
            a = self.cfg.foe_smoothing
            est.foe = a * est.foe + (1 - a) * self.last.foe
        self.last = est
        return est

    def _hold(self) -> EgoMotion:
        held = EgoMotion(self.last.R, self.last.rvec, self.last.foe, 0.0, self.last.stationary, False)
        return held


def rotation_divergence(R: np.ndarray, K: np.ndarray, center_xy) -> float:
    """Divergence (per frame) that the rotation alone adds to the flow at a pixel.

    Computed from the exact rotation homography H = K R K^-1 by central
    differences. For a small rotation vector w it equals
    3 * (x_n * w_y - y_n * w_x) with normalised coordinates x_n, y_n.
    """
    H = K @ R @ np.linalg.inv(K)
    cx, cy = map(float, center_xy)

    def mapped(x, y):
        p = H @ np.array([x, y, 1.0])
        return p[:2] / p[2]

    d = 1.0
    dgx = (mapped(cx + d, cy)[0] - mapped(cx - d, cy)[0]) / (2 * d)
    dgy = (mapped(cx, cy + d)[1] - mapped(cx, cy - d)[1]) / (2 * d)
    return float(dgx + dgy - 2.0)
