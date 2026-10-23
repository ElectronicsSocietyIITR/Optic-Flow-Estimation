"""Forward-collision-warning pipeline: the PC reference of the on-board app.

frame -> resize/gray -> flow (DIS on PC, Vitis LK on the board) + detections
      -> ScaleTracker -> ego-rotation -> 4 TTC methods per track -> fusion
      -> course check -> warning FSM
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from ..tracking import Detection
from .closing import ClosingConfig, ClosingTracker, road_constant, static_ttc
from .collision import CLASS_WIDTH_M, CourseChecker, CourseConfig, CourseResult, Level, WarningConfig, WarningFsm
from .divergence import (DivergenceConfig, affine_from_moments, dequantise_moments, divergence_mask, divergence_ttc,
                         flow_moments_fixed, quantise_flow)
from .ego_rotation import EgoMotion, EgoRotationConfig, EgoRotationEstimator, rotation_divergence
from .fusion import FusionConfig, TtcEstimate, TtcFusion
from .horn import HornConfig, choose_level, horn_at_level
from .looming import looming
from .measurement import Measurement
from .scale_search import ScaleSearchConfig, choose_gap, scale_ttc
from .synth import intrinsics
from .tracker import HISTORY, ScaleTrack, ScaleTracker, TrackerConfig

FlowSource = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (prev_gray, curr_gray) -> (H, W, 2) px/frame
DetectorFn = Callable[[np.ndarray], list[Detection]]          # resized BGR frame -> detections


@dataclass(frozen=True)
class FcwConfig:
    frame_width: int = 512
    frame_height: int = 384
    fx: float = 500.0
    fy: float | None = None          # None: square pixels
    default_fps: float = 30.0        # used for the first dt only
    max_ttc_tracks: int = 16         # tracks that get TTC per frame (PL engine capacity)
    heading_from_foe: bool = True    # course reference: smoothed FOE (follows bends); False = principal point
    scale_enabled: bool = True       # scale-ratio search (needs up to 8 past frames)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ego: EgoRotationConfig = field(default_factory=EgoRotationConfig)
    scale: ScaleSearchConfig = field(default_factory=ScaleSearchConfig)
    horn: HornConfig = field(default_factory=HornConfig)
    divergence: DivergenceConfig = field(default_factory=DivergenceConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    course: CourseConfig = field(default_factory=CourseConfig)
    warning: WarningConfig = field(default_factory=WarningConfig)
    closing: ClosingConfig = field(default_factory=ClosingConfig)

    @property
    def K(self) -> np.ndarray:
        return intrinsics(self.fx, self.frame_width, self.frame_height, self.fy)


def processing_size(width: int, height: int, target_width: int = 512) -> tuple[int, int]:
    """Processing frame for a source: target_width wide, the source aspect, height a multiple of 32 in [64, 512].

    Multiples of 32 suit YOLO's strides; keeping the aspect keeps objects undistorted
    (a 16:9 camera squeezed into 512x384 stretches everything 1.33x vertically).
    """
    h = int(round(target_width * height / width / 32)) * 32
    return target_width, min(max(h, 64), 512)


def config_for_source(width: int, height: int, fx: float, **overrides) -> "FcwConfig":
    """FcwConfig at processing_size(width, height); fx in px of the processing frame's width, square source pixels."""
    w, h = processing_size(width, height)
    fy = fx * (h / height) / (w / width)   # rounding h to 32 px leaves pixels slightly non-square
    return FcwConfig(frame_width=w, frame_height=h, fx=fx, fy=fy, **overrides)


@dataclass
class ObjectResult:
    track_id: int
    bbox: np.ndarray
    class_name: str
    estimate: TtcEstimate | None
    course: CourseResult | None
    level: Level
    measurements: list[Measurement] = field(default_factory=list)  # everything produced, before gating
    kappa: float | None = None    # closing speed / our speed (median of recent frames); > ~1.8 means oncoming


@dataclass
class FcwFrame:
    index: int
    time_s: float
    frame: np.ndarray            # resized BGR
    flow: np.ndarray
    ego: EgoMotion
    heading: tuple[float, float]  # FOE, or the principal point when there is none
    objects: list[ObjectResult]
    timings_ms: dict[str, float] = field(default_factory=dict)
    dt_s: float = 0.0             # time step used for this frame

    @property
    def threat(self) -> ObjectResult | None:
        """The on-course object with the smallest TTC."""
        candidates = [o for o in self.objects if o.course and o.course.on_course and o.estimate and o.estimate.ttc_s]
        return min(candidates, key=lambda o: o.estimate.ttc_s, default=None)

    @property
    def level(self) -> Level:
        return max((o.level for o in self.objects), default=Level.NONE)


class DisFlow:
    """PC stand-in for the board's dense LK: OpenCV DIS optical flow."""

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
        self.dis = cv2.DISOpticalFlow_create(preset)

    def __call__(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        return self.dis.calc(prev, curr, None)


class FcwPipeline:
    def __init__(self, cfg: FcwConfig, detector: DetectorFn, flow_source: FlowSource):
        self.cfg = cfg
        self.detector = detector
        self.flow_source = flow_source
        self.K = cfg.K
        self.pp = (float(self.K[0, 2]), float(self.K[1, 2]))
        self.tracker = ScaleTracker(cfg.tracker)
        self.ego = EgoRotationEstimator(self.K, cfg.ego)
        self.fusion = TtcFusion(cfg.fusion)
        self.course = CourseChecker(cfg.course)
        self.fsms: dict[int, WarningFsm] = {}
        self.ring: deque = deque(maxlen=HISTORY)   # (frame_index, time_s, gray)
        self._index = -1
        self._prev_t: float | None = None
        self._on_course: set[int] = set()
        self._in_path_streak: dict[int, int] = {}
        self._entry_streak: dict[int, int] = {}
        self.closing = ClosingTracker(cfg.closing)
        self._fits: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}  # track -> last affine flow fit (a, b, frame)

    def process(self, frame_bgr: np.ndarray, t: float) -> FcwFrame:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        self._index += 1
        dt = t - self._prev_t if self._prev_t is not None else 0.0
        if dt <= 0:  # first frame, or a repeated/out-of-order timestamp
            dt = 1.0 / self.cfg.default_fps
        if self._prev_t is None or t > self._prev_t:  # never move the clock backwards
            self._prev_t = t
        frame = cv2.resize(frame_bgr, (self.cfg.frame_width, self.cfg.frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = self.ring[-1][2] if self.ring else None

        mark = time.perf_counter()
        flow = self.flow_source(prev_gray, gray) if prev_gray is not None else np.zeros((*gray.shape, 2), np.float32)
        timings["flow"] = (time.perf_counter() - mark) * 1e3

        mark = time.perf_counter()
        detections = [d for d in self.detector(frame) if d.class_name in self.cfg.course.class_width_m]
        timings["detection"] = (time.perf_counter() - mark) * 1e3

        mark = time.perf_counter()
        prev_boxes = [tr.bbox for tr in self.tracker.tracks.values()]
        tracks = self.tracker.update(detections, dt, self._index)
        ego = self.ego.update(flow, prev_boxes) if prev_gray is not None else EgoMotion.identity()
        heading = tuple(ego.foe) if self.cfg.heading_from_foe and ego.foe is not None else self.pp
        yaw_rate = float(ego.rvec[1]) / dt
        self.ring.append((self._index, t, gray))
        timings["track_ego"] = (time.perf_counter() - mark) * 1e3

        mark = time.perf_counter()
        selected = self._select(tracks)
        flow_q = quantise_flow(flow)  # divergence runs on the PL engine's flow format
        objects = [self._assess(track, gray, prev_gray, flow, flow_q, ego, heading, yaw_rate, t, dt) for track in selected]
        self._warn(objects, {tr.id: tr for tr in selected}, flow, ego, dt)
        live = list(tracks)
        self.fusion.retain(live)
        self.course.retain(live)
        self.closing.retain(live)
        self._fits = {tid: f for tid, f in self._fits.items() if tid in tracks}
        self.fsms = {tid: f for tid, f in self.fsms.items() if tid in tracks}
        self._in_path_streak = {tid: n for tid, n in self._in_path_streak.items() if tid in tracks}
        self._entry_streak = {tid: n for tid, n in self._entry_streak.items() if tid in tracks}
        self._on_course = {o.track_id for o in objects if o.course and o.course.in_path}
        timings["ttc"] = (time.perf_counter() - mark) * 1e3
        timings["total"] = (time.perf_counter() - t0) * 1e3
        return FcwFrame(self._index, t, frame, flow, ego, heading, objects, timings, dt)

    def _kappa(self, o: ObjectResult, c: float | None, ego: EgoMotion, dt: float) -> float | None:
        """This frame's closing-speed ratio, or None without road timing, a heading or a TTC."""
        if c is None or ego.foe is None or not (o.estimate and o.estimate.ttc_s):
            return None
        static = static_ttc(c, o.bbox, ego.foe, self.cfg.closing)
        return None if static is None else static * dt / o.estimate.ttc_s

    def _select(self, tracks: dict[int, ScaleTrack]) -> list[ScaleTrack]:
        """At most max_ttc_tracks: last frame's on-course tracks first, then the largest boxes."""
        def priority(tr: ScaleTrack):
            x1, y1, x2, y2 = tr.bbox
            return (tr.id not in self._on_course, -(x2 - x1) * (y2 - y1))
        return sorted(tracks.values(), key=priority)[: self.cfg.max_ttc_tracks]

    def _frame_at(self, frame_index: int) -> tuple[float, np.ndarray] | None:
        for index, time_s, gray in self.ring:
            if index == frame_index:
                return time_s, gray
        return None

    def _assess(self, track, gray, prev_gray, flow, flow_q, ego, heading, yaw_rate, t, dt) -> ObjectResult:
        cfg = self.cfg
        box = track.bbox
        prior = self.fusion.filters.get(track.id)
        eta_prior = prior.eta if prior is not None else None
        measurements = [m for m in [looming(track)] if m is not None]
        if prev_gray is not None and not track.lost:
            k = choose_gap(eta_prior, dt, available=len(self.ring) - 1) if cfg.scale_enabled else 0
            past, box_tk = self._frame_at(self._index - k), track.box_at(self._index - k)
            if k and past is not None and box_tk is not None and t > past[0]:
                # The real span, not k * dt: frames can arrive irregularly.
                m = scale_ttc(gray, past[1], box, box_tk, k, (t - past[0]) / k, cfg.scale)
                if m is not None:
                    measurements.append(m)
            level = choose_level(eta_prior, box, float(np.hypot(track.x[4], track.x[5])), dt)
            m = horn_at_level(prev_gray, gray, box, self.pp, level, dt, cfg.horn)
            if m is not None:
                measurements.append(m)
            prev = self._fits.get(track.id)
            mask, _ = divergence_mask(flow, flow_q, box, self.pp, prev[:2] if prev else None,
                                      self._index - prev[2] if prev else 1 << 30, cfg.divergence)
            fixed = flow_moments_fixed(flow_q, box, self.pp, cfg.divergence.shrink, mask)
            moments = None if fixed is None else dequantise_moments(fixed)
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            m = divergence_ttc(moments, box, rotation_divergence(ego.R, self.K, center), 1, dt, cfg.divergence)
            if m is not None:
                measurements.append(m)
                fit = affine_from_moments(moments)
                if fit is not None:
                    self._fits[track.id] = (*fit, self._index)
        estimate = self.fusion.update(track.id, t, dt, measurements)
        ttc = estimate.ttc_s if estimate else None
        course = self.course.update(track.id, t, box, track.class_name, heading[0], heading[1], ttc, yaw_rate,
                                    (gray.shape[1], gray.shape[0]), self.K @ ego.R @ np.linalg.inv(self.K), self.pp[0])
        return ObjectResult(track.id, box, track.class_name, estimate, course, Level.NONE, measurements)

    def _warn(self, objects: list[ObjectResult], tracks: dict[int, ScaleTrack], flow, ego: EgoMotion, dt: float) -> None:
        """Only the lead vehicle may warn: the nearest object in our path or steadily entering it.

        In path: overlaps our width now, for in_path_frames. Entering: predicted in our path at
        contact, moving towards it and within max_entry_m, for entry_frames (crossing pedestrians
        and cut-ins only overlap us shortly before contact). A lost track is a prediction, not an
        observation: never the lead. Neither is an object closing clearly faster than we drive
        (oncoming traffic, see closing.py).
        """
        cc = self.cfg.course
        c = None
        if self.cfg.closing.enabled and ego.foe is not None:
            c = self.closing.update_c(road_constant(flow, ego.foe, ego.R, self.K, [o.bbox for o in objects], self.cfg.closing))
        for o in objects:
            co, lost = o.course, tracks[o.track_id].lost
            in_path = co is not None and co.in_path and not lost
            ttc = o.estimate.ttc_s if o.estimate else None
            entering = (co is not None and not lost and not co.in_path and co.on_course and co.r_contact is not None
                        and ttc and co.width_m is not None and abs(co.r) * co.width_m <= cc.max_entry_m
                        and co.r * (co.r_contact - co.r) < 0           # offset shrinking towards our path
                        and co.entry_speed_mps is not None and co.entry_speed_mps >= cc.min_entry_speed_mps)
            if (in_path or entering) and self.cfg.closing.enabled:
                o.kappa = self.closing.update(o.track_id, self._kappa(o, c, ego, dt))
                if o.kappa is not None and o.kappa > self.cfg.closing.oncoming_kappa:
                    in_path = entering = False
            # Entry is a prediction, so it needs positive evidence the object is not oncoming: with kappa
            # unknown (e.g. we are stopped, so the road does not move) only objects already in our path warn.
            entering = entering and self.cfg.closing.enabled and o.kappa is not None
            self._in_path_streak[o.track_id] = self._in_path_streak.get(o.track_id, 0) + 1 if in_path else 0
            self._entry_streak[o.track_id] = self._entry_streak.get(o.track_id, 0) + 1 if (in_path or entering) else 0
        candidates = [o for o in objects if self._entry_streak[o.track_id] > 0]
        # Nearest = largest metric width in pixels: distance is proportional to W_obj / w.
        lead = min(candidates, key=lambda o: o.course.width_m / max(o.bbox[2] - o.bbox[0], 1.0), default=None)
        for o in objects:
            e = o.estimate
            ok = (lead is not None and o.track_id == lead.track_id and o.course.on_course
                  and (self._in_path_streak[o.track_id] >= cc.in_path_frames or self._entry_streak[o.track_id] >= cc.entry_frames))
            fsm = self.fsms.setdefault(o.track_id, WarningFsm(self.cfg.warning))
            o.level = fsm.step(ok, e.ttc_s if e else None, e.sigma_ttc_s if e else None, len(e.recent_methods) if e else 0)
