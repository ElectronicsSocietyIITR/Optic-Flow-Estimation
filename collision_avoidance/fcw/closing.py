"""Closing-speed ratio: does an object close faster than we drive?

With camera rotation removed, a static road pixel's flow points away from the FOE
with magnitude |p - FOE| / TTC per frame. On a flat road depth is proportional to
1 / (y - y_horizon), and when we drive forward the horizon is the FOE's row, so

    C = TTC_road(y) * (y - y_FOE)            [frames * px]

is the same for every road pixel: our speed over the camera height, in image
terms. C is measured where road flow is large and accurate (the band between the
horizon and the hood), then gives the static TTC at any object's contact row:
TTC_static = C / (y_bottom - y_FOE). No calibration, no camera height.

kappa = TTC_static / TTC_object = closing speed / our speed: about 1 for parked
objects, below 1 for traffic going our way, well above 1 for oncoming traffic.
Forward-collision warning targets traffic going our way in our lane (the NHTSA
scenarios), so clearly oncoming objects are never the lead vehicle.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClosingConfig:
    enabled: bool = True
    band: tuple[float, float] = (0.25, 0.75)  # road rows, as fractions of the image below the FOE row
    grid_step_px: tuple[int, int] = (8, 4)    # (x, y) sampling of the road band: within 5 % of 4 x 2, 4x cheaper
    min_points: int = 30
    min_radius_px: float = 5.0                # ignore pixels too close to the FOE
    min_outward_px: float = 0.05              # per frame: slower pixels (hood, stopped car) carry no timing
    min_contact_px: float = 3.0               # object's contact row must be this far below the horizon
    oncoming_kappa: float = 1.8               # above this an object is treated as oncoming
    history: int = 8                          # frames in the medians of C and of each track's kappa


def road_constant(flow: np.ndarray, foe, R: np.ndarray, K: np.ndarray, exclude_boxes=(),
                  cfg: ClosingConfig = ClosingConfig()) -> float | None:
    """C = TTC_road(y) * (y - y_FOE) from the road band (frames * px), or None when unmeasurable."""
    h, w = flow.shape[:2]
    yh = float(foe[1])
    y1, y2 = int(yh + cfg.band[0] * (h - yh)), int(yh + cfg.band[1] * (h - yh))
    if y2 - y1 < 2 or yh >= h:
        return None
    ys, xs = np.mgrid[max(y1, 0):min(y2, h):cfg.grid_step_px[1], 0:w:cfg.grid_step_px[0]]
    keep = np.ones(xs.shape, bool)
    for x1, by1, x2, by2 in exclude_boxes:
        keep &= ~((xs >= x1) & (xs <= x2) & (ys >= by1) & (ys <= by2))
    xs, ys = xs[keep], ys[keep]
    if xs.size < cfg.min_points:
        return None
    pts = np.stack([xs, ys, np.ones(xs.size)]).astype(np.float64)
    rot = K @ R @ np.linalg.inv(K) @ pts
    translational = flow[ys, xs].astype(np.float64) - (rot[:2] / rot[2] - pts[:2]).T
    radial = pts[:2].T - np.asarray(foe, float)
    dist = np.linalg.norm(radial, axis=1)
    outward = np.einsum("ij,ij->i", translational, radial / np.maximum(dist, 1e-9)[:, None])
    ok = (dist > cfg.min_radius_px) & (outward > cfg.min_outward_px)
    if ok.sum() < cfg.min_points:
        return None
    return float(np.median(dist[ok] / outward[ok] * (pts[1, ok] - yh)))


def static_ttc(c: float, box, foe, cfg: ClosingConfig = ClosingConfig()) -> float | None:
    """TTC (frames) we would have to a parked object standing where this box does."""
    contact = float(box[3]) - float(foe[1])
    return c / contact if contact >= cfg.min_contact_px else None


class ClosingTracker:
    """Medians over recent frames: of C (scene-wide) and of each track's kappa (single frames are noisy)."""

    def __init__(self, cfg: ClosingConfig = ClosingConfig()):
        self.cfg = cfg
        self.c_history: deque = deque(maxlen=cfg.history)
        self.samples: dict[int, deque] = {}

    def update_c(self, c: float | None) -> float | None:
        if c is not None and np.isfinite(c) and c > 0:
            self.c_history.append(c)
        return float(np.median(self.c_history)) if self.c_history else None

    def update(self, track_id: int, kappa: float | None) -> float | None:
        hist = self.samples.setdefault(track_id, deque(maxlen=self.cfg.history))
        if kappa is not None and np.isfinite(kappa):
            hist.append(kappa)
        return float(np.median(hist)) if hist else None

    def retain(self, track_ids) -> None:
        keep = set(track_ids)
        self.samples = {tid: h for tid, h in self.samples.items() if tid in keep}
