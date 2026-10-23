"""Export golden vectors from the Python FCW reference for the C++ parity tests (plan 2).

    python tools/export_golden.py --out host/tests/golden            # every module
    python tools/export_golden.py --out host/tests/golden --only horn

Each module writes host/tests/golden/<module>.yml.gz (OpenCV FileStorage), read
by host/tests/golden.hpp. int64 values are written as decimal strings because
FileStorage integers are 32-bit.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


class Writer:
    def __init__(self, out_dir: Path, name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{name}.yml.gz"
        self.fs = cv2.FileStorage(str(self.path), cv2.FILE_STORAGE_WRITE)

    def mat(self, key: str, value) -> None:
        self.fs.write(key, np.ascontiguousarray(value))

    def num(self, key: str, value) -> None:
        self.fs.write(key, float(value))

    def i64(self, key: str, value) -> None:
        self.fs.write(key, str(int(value)))

    def str(self, key: str, value: str) -> None:
        self.fs.write(key, value)

    def png(self, key: str, image) -> None:
        """Lossless image next to the YAML; the key stores the file name."""
        name = f"{self.path.name.split('.')[0]}_{key}.png"
        cv2.imwrite(str(self.path.parent / name), image)
        self.fs.write(key, name)

    def close(self) -> None:
        self.fs.release()
        print("wrote", self.path)


def export_harness(out: Path) -> None:
    w = Writer(out, "harness")
    w.mat("m", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]]))
    w.num("x", 3.25)
    w.i64("big", 6_000_000_000)
    w.close()


def _dets(boxes, names=None):
    from collision_avoidance.tracking import Detection
    return [Detection(np.asarray(b, float), (names or ["car"] * len(boxes))[i], 0.9) for i, b in enumerate(boxes)]


def _track_state(w, key, tracks):
    ids = sorted(tracks)
    w.mat(f"{key}_ids", np.array(ids, np.float64).reshape(-1, 1) if ids else np.zeros((0, 1)))
    for tid in ids:
        t = tracks[tid]
        w.mat(f"{key}_t{tid}_x", t.x.reshape(-1, 1))
        w.mat(f"{key}_t{tid}_P", t.P)
        w.num(f"{key}_t{tid}_lost", t.lost)
        w.num(f"{key}_t{tid}_age", t.age)
        hist = np.array([[i, *b] for i, b in t.history], np.float64) if t.history else np.zeros((0, 5))
        w.mat(f"{key}_t{tid}_hist", hist)


def export_tracker(out: Path) -> None:
    """Three detection sequences replayed frame by frame; full tracker state after every frame."""
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.looming import looming
    from collision_avoidance.fcw.tracker import ScaleTracker

    dt = 1 / 30
    scene = synth.render_scene(synth.SceneConfig(n_frames=40, z0_m=40.0))
    cases = {
        "approach": [_dets([b]) for b in scene.boxes],
        "two": [_dets([[10 + 2 * i, 10, 50 + 2 * i, 40], [300, 200 - i, 400, 260 - i]], ["car", "truck"]) for i in range(10)],
        "lost": [_dets([[100, 100, 160, 150]])] * 5 + [[]] + [_dets([[100, 100, 160, 150]])],
    }
    w = Writer(out, "tracker")
    w.num("dt", dt)
    for name, frames in cases.items():
        tracker = ScaleTracker()
        w.num(f"{name}_frames", len(frames))
        for i, dets in enumerate(frames):
            w.mat(f"{name}_f{i}_dets", np.array([d.bbox for d in dets], np.float64).reshape(-1, 4))
            w.str(f"{name}_f{i}_cls", ",".join(d.class_name for d in dets))
            tracks = tracker.update(dets, dt, i)
            _track_state(w, f"{name}_f{i}", tracks)
            for tid, t in tracks.items():
                m = looming(t)
                if m is not None:
                    w.num(f"{name}_f{i}_t{tid}_loom_eta", m.eta)
                    w.num(f"{name}_f{i}_t{tid}_loom_var", m.var)
    w.close()


def _ego(w, key, ego):
    w.mat(f"{key}_R", ego.R)
    w.num(f"{key}_valid", ego.valid)
    w.num(f"{key}_stationary", ego.stationary)
    w.num(f"{key}_has_foe", ego.foe is not None)
    w.mat(f"{key}_rvec", np.asarray(ego.rvec, float).reshape(1, 3))
    if ego.foe is not None:
        w.mat(f"{key}_foe", np.asarray(ego.foe, float).reshape(1, 2))


def export_ego(out: Path) -> None:
    """Grid samples per case (all the estimator sees), the resulting EgoMotion, and a small sampling case."""
    import cv2 as _cv2
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.ego_rotation import EgoRotationConfig, EgoRotationEstimator, grid_samples, rotation_divergence

    W, H = 512, 384
    K = synth.intrinsics(500, W, H)
    cfg = EgoRotationConfig()

    def dense(R, t, depth):
        _, f = synth.flow_from_motion(K, R, t, depth, W, H, step=1)
        return f.reshape(H, W, 2).astype(np.float32)

    def road(xs, ys):
        return 8.0 + 60.0 * np.random.default_rng(0).random(xs.shape)

    fwd = np.array([0.0, 0.0, -0.5])
    truck = dense(synth.rotation_y(np.radians(0.3)), fwd, road)
    truck[100:300, 150:350] += np.array([6.0, -2.0], np.float32)
    good = dense(synth.rotation_y(np.radians(0.3)), fwd, road)
    noise = np.random.default_rng(1).normal(0, 5, (H, W, 2)).astype(np.float32)
    # Each case is a sequence of (flow, excluded boxes) fed to one estimator.
    cases = {
        "yaw0": [(dense(synth.rotation_y(0.0), fwd, road), [])],
        "yaw04": [(dense(synth.rotation_y(np.radians(0.4)), fwd, road), [])],
        "yawm08": [(dense(synth.rotation_y(np.radians(-0.8)), fwd, road), [])],
        "pure": [(dense(synth.rotation_y(np.radians(0.5)), np.zeros(3), 30.0), [])],
        "truck": [(truck, [(150, 100, 350, 300)])],
        "hold": [(good, []), (noise, [])],
        "pitch": [(dense(_cv2.Rodrigues(np.array([np.radians(0.5) / 30, np.radians(2.0) / 30, 0.0]))[0], fwd, road), [])],
        "hidden": [(good, []), (good, [(0, 0, 500, 380)])],
        "ransac_yaw04": [(dense(synth.rotation_y(np.radians(0.4)), fwd, road), [])],
        "ransac_pure": [(dense(synth.rotation_y(np.radians(0.5)), np.zeros(3), 30.0), [])],
    }
    w = Writer(out, "ego")
    w.mat("K", K)
    for name, seq in cases.items():
        est = EgoRotationEstimator(K, EgoRotationConfig(method="ransac" if name.startswith("ransac") else "linear"))
        w.num(f"{name}_steps", len(seq))
        for i, (flow, boxes) in enumerate(seq):
            prev, curr = grid_samples(flow, boxes, cfg)
            w.mat(f"{name}_s{i}_prev", prev)
            w.mat(f"{name}_s{i}_curr", curr)
            _ego(w, f"{name}_s{i}", est.update(flow, boxes))
    small = np.random.default_rng(2).normal(0, 1, (48, 64, 2)).astype(np.float32)
    w.mat("small_flow", small)
    prev, curr = grid_samples(small, [(10, 5, 30, 20)], EgoRotationConfig(grid_step_px=8, hood_rows=4))
    w.mat("small_prev", prev)
    w.mat("small_curr", curr)
    for j, c in enumerate([(256, 192), (400, 100), (60, 330)]):
        R = _cv2.Rodrigues(np.array([0.004, -0.006, 0.002]))[0]
        w.mat(f"div{j}_R", R)
        w.mat(f"div{j}_c", np.array([c], float))
        w.num(f"div{j}_value", rotation_divergence(R, K, c))
    w.close()


def export_scale(out: Path) -> None:
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.scale_search import choose_gap, scale_ttc, search_scale

    dt = 1 / 30
    approach = synth.render_scene(synth.SceneConfig(n_frames=25))
    static = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0))
    sideways = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0, lateral_speed_mps=4.0))
    tiny = synth.render_scene(synth.SceneConfig(n_frames=5, z0_m=200.0))
    cases = [("k2", approach, 24, 2), ("k4", approach, 24, 4), ("k8", approach, 24, 8),
             ("static", static, 8, 8), ("sideways", sideways, 8, 8), ("tiny", tiny, 4, 4)]
    w = Writer(out, "scale")
    w.num("dt", dt)
    w.num("cases", len(cases))
    for i, (name, scene, t, k) in enumerate(cases):
        key = f"c{i}"
        w.str(f"{key}_name", name)
        w.png(f"{key}_gray_t", scene.frames[t])
        w.png(f"{key}_gray_tk", scene.frames[t - k])
        w.mat(f"{key}_box_t", np.asarray(scene.boxes[t], float).reshape(1, 4))
        w.mat(f"{key}_box_tk", np.asarray(scene.boxes[t - k], float).reshape(1, 4))
        w.num(f"{key}_k", k)
        found = search_scale(scene.frames[t], scene.frames[t - k], scene.boxes[t], scene.boxes[t - k])
        w.num(f"{key}_found", found is not None)
        if found is not None:
            w.num(f"{key}_s", found[0])
            w.num(f"{key}_peak", found[1])
        m = scale_ttc(scene.frames[t], scene.frames[t - k], scene.boxes[t], scene.boxes[t - k], k, dt)
        w.num(f"{key}_has_m", m is not None)
        if m is not None:
            w.num(f"{key}_eta", m.eta)
            w.num(f"{key}_var", m.var)
    gaps = [(None, 8), (1.0, 8), (0.25, 8), (0.25, 3), (0.25, 0), (0.6, 8), (-0.2, 8)]
    w.mat("gaps_in", np.array([[np.nan if e is None else e, a] for e, a in gaps]))
    w.mat("gaps_out", np.array([[choose_gap(e, dt, available=a)] for e, a in gaps], float))
    w.close()


def export_horn(out: Path) -> None:
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.horn import HornConfig, choose_level, horn_at_level, horn_solve, horn_sums

    dt = 1 / 30
    rng = np.random.default_rng(3)
    rand_prev = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    rand_curr = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    approach = synth.render_scene(synth.SceneConfig(n_frames=21))
    side = synth.render_scene(synth.SceneConfig(n_frames=21, z0_m=15.0, closing_speed_mps=6.0, lateral_m=-1.0, lateral_speed_mps=1.5))
    flat = np.full((384, 512), 128, np.uint8)
    # 2-px checkerboard in both frames: maximal |Ex| and |Ey| everywhere, so the largest sums.
    checker = np.kron((np.indices((192, 256)).sum(axis=0) % 2) * 255, np.ones((2, 2))).astype(np.uint8)
    cases = [
        ("pixel_loop", rand_prev, rand_curr, (10, 8, 30, 28), (25, 20), HornConfig(shrink=0.0, grad_threshold_l1=0), 0),
        ("approach", approach.frames[19], approach.frames[20], tuple(approach.boxes[20]), (256, 192), HornConfig(), 0),
        ("flat", flat, flat, (100, 100, 200, 200), (256, 192), HornConfig(), 0),
        ("saturated", checker, checker, (1, 1, 511, 383), (256, 192), HornConfig(shrink=0.0, grad_threshold_l1=0), 0),
        ("side_level2", side.frames[19], side.frames[20], tuple(side.boxes[20]), (256, 192), HornConfig(), 2),
    ]
    w = Writer(out, "horn")
    w.num("dt", dt)
    w.num("cases", len(cases))
    for i, (name, prev, curr, box, pp, cfg, level) in enumerate(cases):
        k = f"c{i}"
        w.str(f"{k}_name", name)
        w.png(f"{k}_prev", prev)
        w.png(f"{k}_curr", curr)
        w.mat(f"{k}_box", np.array([box], float))
        w.mat(f"{k}_pp", np.array([pp], float))
        w.num(f"{k}_shrink", cfg.shrink)
        w.num(f"{k}_thr", cfg.grad_threshold_l1)
        w.num(f"{k}_level", level)
        if level == 0:
            found = horn_sums(prev, curr, box, pp, cfg)
            w.num(f"{k}_has_sums", found is not None)
            if found is not None:
                sums, n = found
                for j, v in enumerate(sums):
                    w.i64(f"{k}_sum{j}", v)
                w.i64(f"{k}_n", n)
                m = horn_solve(sums, n, dt, cfg)
            else:
                m = None
        else:
            m = horn_at_level(prev, curr, box, pp, level, dt, cfg)
        w.num(f"{k}_has_m", m is not None)
        if m is not None:
            w.num(f"{k}_eta", m.eta)
            w.num(f"{k}_var", m.var)
    levels = [(None, (0, 0, 120, 90), 0.0), (1.0, (0, 0, 120, 90), 0.0), (0.2, (0, 0, 120, 90), 30.0), (3.0, (0, 0, 400, 300), 200.0)]
    w.mat("levels_in", np.array([[np.nan if e is None else e, *b, v] for e, b, v in levels]))
    w.mat("levels_out", np.array([[choose_level(e, b, v, dt)] for e, b, v in levels], float))
    w.close()


def export_divergence(out: Path) -> None:
    import cv2 as _cv2
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.divergence import DivergenceConfig, divergence_ttc, flow_moments, robust_mask
    from collision_avoidance.fcw.ego_rotation import rotation_divergence

    dt, W, H = 1 / 30, 512, 384
    K = synth.intrinsics(500, W, H)
    pp = (K[0, 2], K[1, 2])
    box = (180.0, 140.0, 300.0, 230.0)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)

    def looming(rate, center=(240, 185), lateral=(0.0, 0.0)):
        return np.stack([(xs - center[0]) * rate + lateral[0], (ys - center[1]) * rate + lateral[1]], axis=-1)

    R = _cv2.Rodrigues(np.array([0.01, -0.015, 0.0]))[0]
    _, rot = synth.flow_from_motion(K, R, np.zeros(3), 50.0, W, H, step=1)
    leak = looming(0.02)
    leak[140:160, 180:300] = (5.0, 0.0)
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    cases = [("expanding", looming(0.025, lateral=(2.0, -1.0)), 0.0),
             ("rotation", looming(0.02) + rot.reshape(H, W, 2), rotation_divergence(R, K, center)),
             ("leak", leak, 0.0)]
    cfg = DivergenceConfig()
    x0, y0, x1, y1 = 176, 136, 304, 234   # stored window around the box
    w = Writer(out, "divergence")
    w.num("dt", dt)
    w.mat("pp", np.array([pp]))
    w.mat("box", np.array([box]))
    w.mat("window", np.array([[x0, y0, x1, y1]], float))
    w.num("cases", len(cases))
    for i, (name, flow, rot_div) in enumerate(cases):
        k = f"c{i}"
        flow = flow.astype(np.float32)
        w.str(f"{k}_name", name)
        w.mat(f"{k}_flow", flow[y0:y1, x0:x1].copy())
        w.num(f"{k}_rotdiv", rot_div)
        m = flow_moments(flow, box, pp, shrink=cfg.shrink)
        w.mat(f"{k}_moments", m.reshape(1, -1))
        plain = divergence_ttc(m, box, rot_div, 1, dt, cfg)
        w.num(f"{k}_plain_eta", plain.eta)
        w.num(f"{k}_plain_var", plain.var)
        # RANSAC sample triples exactly as robust_mask draws them (numpy default_rng(0)).
        bx1, by1 = max(0, int(box[0])), max(0, int(box[1]))
        bx2, by2 = min(W, int(np.ceil(box[2]))), min(H, int(np.ceil(box[3])))
        n_sub = len(np.arange(by1, by2)[::cfg.ransac_stride]) * len(np.arange(bx1, bx2)[::cfg.ransac_stride])
        from collision_avoidance.fcw.divergence import ransac_samples
        w.mat(f"{k}_samples", np.array(ransac_samples(n_sub, cfg.ransac_iterations), float))
        mask = robust_mask(flow, box, pp, cfg)
        w.mat(f"{k}_mask", mask[y0:y1, x0:x1].astype(np.uint8))
        robust = divergence_ttc(flow_moments(flow, box, pp, mask, cfg.shrink), box, rot_div, 1, dt, cfg)
        w.num(f"{k}_robust_eta", robust.eta)
        w.num(f"{k}_robust_var", robust.var)
        # Fixed-point path (the PL engine's): quantised flow, last frame's fit as a prediction, masked int64 moments.
        from collision_avoidance.fcw.divergence import (affine_from_moments, divergence_mask, fit_to_fixed, flow_moments_fixed,
                                                        flow_rect, predicted_mask_fixed, quantise_flow)
        flow_q = quantise_flow(flow)
        w.mat(f"{k}_flow_q", flow_q[y0:y1, x0:x1].astype(np.float64))
        fa, fb = affine_from_moments(flow_moments(flow, box, pp, mask, cfg.shrink))
        fit_q = fit_to_fixed(fa, fb)
        w.mat(f"{k}_fit", np.array([[*fa, *fb]]))
        w.mat(f"{k}_fit_q", np.array([fit_q], float))
        rect = flow_rect(box, cfg.shrink, W, H)
        w.mat(f"{k}_rect", np.array([rect], float))
        pred = predicted_mask_fixed(flow_q, rect, (int(round(pp[0])), int(round(pp[1]))), fit_q, 64)
        w.mat(f"{k}_pred_mask", pred.astype(np.uint8))
        dmask, used = divergence_mask(flow, flow_q, box, pp, (fa, fb), 1, cfg)
        w.num(f"{k}_pred_used", used)
        w.mat(f"{k}_fixed_moments", flow_moments_fixed(flow_q, box, pp, cfg.shrink, dmask).astype(np.float64).reshape(1, -1))
        _, used_stale = divergence_mask(flow, flow_q, box, pp, (fa, fb), 2, cfg)
        w.num(f"{k}_stale_used", used_stale)
    w.close()


METHOD_INDEX = {"looming": 0, "scale": 1, "horn": 2, "divergence": 3}


def export_fusion(out: Path) -> None:
    from collision_avoidance.fcw.fusion import FusionConfig, TtcFusion
    from collision_avoidance.fcw.measurement import Measurement

    dt, methods = 1 / 30, list(METHOD_INDEX)
    rng = np.random.default_rng(0)
    scenarios = {}
    steps = []
    for i in range(60):
        tau = 3.0 - i * dt
        steps.append([Measurement("scale", 1 / tau + rng.normal(0, 0.03), 0.03**2), Measurement("horn", 1 / tau + rng.normal(0, 0.05), 0.05**2)])
    scenarios["track"] = (5, steps)
    scenarios["outlier"] = (5, [[Measurement("scale", 0.5, 0.01**2)]] * 20 + [[Measurement("horn", 5.0, 0.05**2)]])
    scenarios["reinit"] = (3, [[Measurement("scale", 0.2, 0.01**2)]] * 10 + [[Measurement("scale", 2.0, 0.01**2)]] * 5)
    scenarios["receding"] = (5, [[Measurement("looming", -0.3, 0.01)]])
    mixed = []
    for i in range(300):
        eta_true = 0.3 + 0.004 * i
        ms = []
        for m in methods:
            if rng.random() < 0.7:
                eta = eta_true + rng.normal(0, 0.05) + (rng.normal(0, 2) if rng.random() < 0.05 else 0.0)
                ms.append(Measurement(m, eta, float(rng.uniform(0.0005, 0.05))))
        mixed.append(ms)
    scenarios["mixed"] = (5, mixed)

    w = Writer(out, "fusion")
    w.num("dt", dt)
    w.str("scenarios", ",".join(scenarios))
    for name, (max_rejects, seq) in scenarios.items():
        fusion = TtcFusion(FusionConfig(max_consecutive_rejects=max_rejects))
        w.num(f"{name}_max_rejects", max_rejects)
        w.num(f"{name}_steps", len(seq))
        for i, ms in enumerate(seq):
            k = f"{name}_s{i}"
            w.mat(f"{k}_in", np.array([[METHOD_INDEX[m.method], m.eta, m.var] for m in ms], float).reshape(-1, 3))
            est = fusion.update(1, i * dt, dt, ms)
            w.num(f"{k}_has", est is not None)
            if est is not None:
                w.mat(f"{k}_out", np.array([[est.eta, est.eta_var,
                                             np.nan if est.ttc_s is None else est.ttc_s,
                                             np.nan if est.sigma_ttc_s is None else est.sigma_ttc_s]]))
                w.mat(f"{k}_accepted", np.array([METHOD_INDEX[m.method] for m in est.measurements], float).reshape(-1, 1))
                w.mat(f"{k}_recent", np.array(sorted(METHOD_INDEX[m] for m in est.recent_methods), float).reshape(-1, 1))
    w.close()


def export_collision(out: Path) -> None:
    from collision_avoidance.fcw.collision import CourseChecker, WarningFsm

    dt = 1 / 30
    rng = np.random.default_rng(1)
    classes = ["car", "truck", "person", "dog", "bicycle"]
    course = []
    for i in range(300):
        x, z = rng.uniform(-6, 6), rng.uniform(5, 60)
        width = {"car": 1.8, "truck": 2.5, "person": 0.5, "dog": 0.6, "bicycle": 0.6}[cls := classes[int(rng.integers(0, 5))]]
        y2 = 192 + rng.uniform(-30, 80)
        ttc = None if rng.random() < 0.1 else float(rng.uniform(0.5, 8))
        t = (i // 3) * dt if rng.random() < 0.9 else ((i // 3) - 1) * dt   # some repeated / earlier times
        course.append((t, (256 + 500 * (x - width / 2) / z, y2 - 40, 256 + 500 * (x + width / 2) / z, y2), cls,
                       256.0 + rng.normal(0, 5), 192.0, ttc, float(rng.normal(0, 0.06))))
    fsm = []
    for i in range(300):
        fsm.append((bool(rng.random() < 0.8), None if rng.random() < 0.05 else float(rng.uniform(0.8, 4.0)),
                    None if rng.random() < 0.1 else float(rng.uniform(0.0, 0.5)), int(rng.integers(0, 5))))
    w = Writer(out, "collision")
    w.str("classes", ",".join(classes))
    K_COLL = np.array([[500.0, 0, 256.0], [0, 500.0, 192.0], [0, 0, 1]])
    checker = CourseChecker()
    w.num("course_steps", len(course))
    for i, (t, box, cls, hx, hy, ttc, yaw) in enumerate(course):
        k = f"course_s{i}"
        tid = i % 3
        w.mat(f"{k}_in", np.array([[tid, t, *box, classes.index(cls), hx, hy, np.nan if ttc is None else ttc, yaw]]))
        a = yaw / 30   # a small rotation per step exercises the derotation of the entry speed
        H = K_COLL @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ np.linalg.inv(K_COLL)
        w.mat(f"{k}_rot", H)
        r = checker.update(tid, t, box, cls, hx, hy, ttc, yaw, (512, 384), H, 256.0)   # some boxes leave the image: clipped
        w.mat(f"{k}_out", np.array([[r.on_course, r.r, np.nan if r.r_contact is None else r.r_contact, r.threshold, r.in_path,
                                     np.nan if r.entry_speed_mps is None else r.entry_speed_mps]], float))
    machine = WarningFsm()
    w.num("fsm_steps", len(fsm))
    w.mat("fsm_in", np.array([[on, np.nan if t is None else t, np.nan if s is None else s, m] for on, t, s, m in fsm], float))
    w.mat("fsm_out", np.array([[int(machine.step(*args))] for args in fsm], float))
    w.close()


def export_pipeline(out: Path) -> None:
    """Whole-pipeline replays. Too large for git: written to golden_local/ next to `out`.

    Flow is DIS, quantised to float16 and stored raw (H*W*2 values, row-major); the
    Python pipeline itself runs on that quantised flow, so both languages see the same input.
    """
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
    from collision_avoidance.tracking import Detection

    local = out.parent / "golden_local"
    local.mkdir(parents=True, exist_ok=True)
    head_on = synth.render_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0))
    scenarios = {
        "head_on": (head_on, list(range(80)), [i / 30 for i in range(80)]),
        "next_lane": (synth.render_scene(synth.SceneConfig(n_frames=60, z0_m=50.0, closing_speed_mps=15.0, lateral_m=3.5)), list(range(60)), [i / 30 for i in range(60)]),
        "static": (synth.render_scene(synth.SceneConfig(n_frames=30, z0_m=20.0, closing_speed_mps=0.0)), list(range(30)), [i / 30 for i in range(30)]),
        "dropped": (head_on, list(range(0, 80, 2)), [i / 30 for i in range(0, 80, 2)]),
        "irregular": (head_on, [i for i in range(60) if i % 3 != 2], [i / 30 for i in range(60) if i % 3 != 2]),
        "out_of_order": (head_on, [0, 1, 2, 3], [0.0, 1 / 30, 0.0, 2 / 30]),
    }
    twenty = [np.array([10 + 30 * (i % 10), 100 + 60 * (i // 10), 35 + 30 * (i % 10), 150 + 60 * (i // 10)], float) for i in range(20)]
    w = Writer(local, "pipeline")
    w.str("scenarios", ",".join([*scenarios, "ground_oncoming", "ground_parked", "ground_crossing", "twenty"]))
    dis = DisFlow()

    def run(name, frames, boxes_per_frame, times, class_name="car"):
        state = {"i": 0}
        w.str(f"{name}_class", class_name)

        def detector(_frame):
            return [Detection(b, class_name, 0.9) for b in boxes_per_frame[state["i"]]]

        def flow_source(prev, curr):
            flow = dis(prev, curr).astype(np.float16)
            fname = f"pipeline_{name}_flow{state['i']}.bin"
            flow.tofile(local / fname)
            w.str(f"{name}_f{state['i']}_flow", fname)
            return flow.astype(np.float32)

        pipe = FcwPipeline(FcwConfig(), detector, flow_source)
        w.num(f"{name}_frames", len(frames))
        for i, (gray, t) in enumerate(zip(frames, times)):
            state["i"] = i
            k = f"{name}_f{i}"
            w.png(k + "_gray", gray)
            w.num(k + "_t", t)
            w.mat(k + "_dets", np.array(boxes_per_frame[i], float).reshape(-1, 4))
            r = pipe.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), t)
            w.num(k + "_dt", r.dt_s)
            w.mat(k + "_heading", np.array([r.heading], float))
            w.num(k + "_ego_valid", r.ego.valid)
            rows = []
            for o in r.objects:
                e = o.estimate
                bits = sum(1 << METHOD_INDEX[m.method] for m in o.measurements)
                rows.append([o.track_id, *o.bbox, int(o.level), e is not None, e.eta if e else np.nan,
                             (e.ttc_s if e and e.ttc_s else np.nan), o.course.on_course if o.course else 0, bits,
                             np.nan if o.kappa is None else o.kappa])
            # id, x1, y1, x2, y2, level, has_est, eta, ttc, on_course, method bits, kappa
            w.mat(k + "_objects", np.array(rows, float).reshape(-1, 12))

    # Road plane + our own motion: exercises the closing-speed (oncoming) test end to end.
    scenarios["ground_oncoming"] = (synth.render_scene(synth.SceneConfig(n_frames=50, ground=True, ego_speed_mps=10.0, closing_speed_mps=25.0, z0_m=70.0)), list(range(50)), [i / 30 for i in range(50)])
    scenarios["ground_parked"] = (synth.render_scene(synth.SceneConfig(n_frames=60, ground=True, ego_speed_mps=10.0, closing_speed_mps=10.0, z0_m=30.0)), list(range(60)), [i / 30 for i in range(60)])
    scenarios["ground_crossing"] = (synth.render_scene(synth.SceneConfig(n_frames=70, ground=True, ego_speed_mps=10.0, closing_speed_mps=10.0, z0_m=30.0, lateral_m=-3.0, lateral_speed_mps=1.0, obj_width_m=0.5, obj_height_m=1.7, obj_y_m=0.2)), list(range(70)), [i / 30 for i in range(70)])
    for name, (scene, idx, times) in scenarios.items():
        run(name, [scene.frames[i] for i in idx], [[scene.boxes[i]] for i in idx], times,
            "person" if name == "ground_crossing" else "car")
    tex = synth.texture(384, 512, 0)
    run("twenty", [tex] * 3, [twenty] * 3, [0.0, 1 / 30, 2 / 30])
    w.close()


def export_hls(out: Path) -> None:
    """Plain files for the HLS testbench (no OpenCV in C-simulation): hls/ttc_engine/golden/<case>/."""
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.divergence import flow_moments_fixed, flow_rect, quantise_flow
    from collision_avoidance.fcw.horn import HornConfig, horn_sums, inner_box
    from collision_avoidance.fcw.pipeline import DisFlow

    root = Path("hls/ttc_engine/golden")
    rng = np.random.default_rng(5)
    rand_prev = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    rand_curr = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    scene = synth.render_scene(synth.SceneConfig(n_frames=21))
    checker = np.kron((np.indices((192, 256)).sum(axis=0) % 2) * 255, np.ones((2, 2))).astype(np.uint8)
    flat = np.full((384, 512), 128, np.uint8)
    overlapping = [(5, 5, 40, 30), (20, 10, 60, 45), (30, 20, 50, 40)]
    rand_flow = quantise_flow(rng.uniform(-5, 5, (48, 64, 2)))
    approach_flow = quantise_flow(DisFlow()(scene.frames[19], scene.frames[20]))
    extreme_flow = np.full((384, 512, 2), 32767, np.int16)          # largest products: overflow check
    extreme_flow[::2, :, 1] = -32768
    cases = {
        "random_overlap": (rand_prev, rand_curr, rand_flow, overlapping, (32, 24), HornConfig(shrink=0.0, grad_threshold_l1=0)),
        "random_threshold": (rand_prev, rand_curr, rand_flow, overlapping, (32, 24), HornConfig(shrink=0.0, grad_threshold_l1=300)),
        "approach": (scene.frames[19], scene.frames[20], approach_flow, [tuple(scene.boxes[20])], (256, 192), HornConfig()),
        "checker": (checker, checker, extreme_flow, [(0, 0, 512, 384)], (256, 192), HornConfig(shrink=0.0, grad_threshold_l1=0)),
        "flat": (flat, flat, np.zeros((384, 512, 2), np.int16), [(100, 100, 200, 200)], (256, 192), HornConfig()),
    }
    for name, (prev, curr, flow_q, boxes, pp, cfg) in cases.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        h, w = curr.shape
        prev.tofile(d / "prev.bin")
        curr.tofile(d / "curr.bin")
        flow_q.tofile(d / "flow.bin")           # int16 u, v interleaved, row-major
        px, py = int(round(pp[0])), int(round(pp[1]))
        (d / "params.txt").write_text(f"{w} {h} {px} {py} {cfg.grad_threshold_l1}\n")
        rect_lines, flow_rect_lines, expected_lines = [], [], []
        for box in boxes:
            rect = inner_box(box, cfg.shrink, w, h)
            sums, n = horn_sums(prev, curr, box, pp, cfg)
            moments = flow_moments_fixed(flow_q, box, pp, cfg.shrink)
            rect_lines.append(" ".join(map(str, rect)))
            flow_rect_lines.append(" ".join(map(str, flow_rect(box, cfg.shrink, w, h))))
            expected_lines.append(" ".join(str(int(v)) for v in sums) + f" {n} " + " ".join(str(int(v)) for v in moments))
        (d / "boxes.txt").write_text("\n".join(rect_lines) + "\n")
        (d / "flow_boxes.txt").write_text("\n".join(flow_rect_lines) + "\n")
        (d / "expected.txt").write_text("\n".join(expected_lines) + "\n")
        print("wrote", d)


YOLO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def export_yolo(out: Path) -> None:
    """Raw YOLOv9t head maps (what the DPU returns) on 3 dash-cam frames, and the six-class decode + NMS.

    The reference scores only the six collision classes, as the board decoder does. Too large
    for git: written to golden_local/.
    """
    import sys as _sys
    import torch
    import torchvision
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "vitis_ai"))
    import yolo_dpu

    local = out.parent / "golden_local"
    local.mkdir(parents=True, exist_ok=True)
    graph = yolo_dpu.load("yolov9t.pt")
    cap = cv2.VideoCapture("data/normal_driving/241220_125301_002_FH.MP4")
    ids = list(YOLO_CLASSES)
    w = Writer(local, "yolo")
    frames = 0
    for i in range(90):
        ok, frame = cap.read()
        if not ok:
            break
        if i % 30:
            continue
        bgr = cv2.resize(frame, (512, 384), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            raw = graph(yolo_dpu.preprocess(bgr))
            pred = yolo_dpu.decode(raw, graph.head)[0].T            # (N, 4 + 80)
        k = f"f{frames}"
        for j, r in enumerate(raw):                                  # NHWC rows: (H*W, 144)
            c, h, wd = r.shape[1:]
            w.mat(f"{k}_head{j}", r[0].permute(1, 2, 0).reshape(h * wd, c).numpy().astype(np.float64))
            w.num(f"{k}_head{j}_h", h)
            w.num(f"{k}_head{j}_w", wd)
        scores, which = pred[:, 4:][:, ids].max(dim=1)
        keep = scores > 0.5
        xy, wh = pred[keep, :2], pred[keep, 2:4]
        boxes = torch.cat((xy - wh / 2, xy + wh / 2), dim=1)
        cls = torch.tensor(ids)[which[keep]]
        idx = torchvision.ops.batched_nms(boxes, scores[keep], cls, 0.7)
        w.mat(f"{k}_dets", torch.cat((boxes[idx], scores[keep][idx, None], cls[idx, None].float()), 1).numpy().astype(np.float64).reshape(-1, 6))
        frames += 1
    w.num("frames", frames)
    w.close()


def export_lk(out: Path) -> None:
    """A real dash-cam frame pair at 512x288 and Python's LK flow (GPU), for the C++ LK. Written to golden_local/."""
    from collision_avoidance.fcw.lk_flow import LkFlow

    local = out.parent / "golden_local"
    local.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture("data/normal_driving/241220_125301_002_FH.MP4")
    grays = []
    for _ in range(2):
        ok, frame = cap.read()
        grays.append(cv2.cvtColor(cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    flow = LkFlow()(grays[0], grays[1])
    w = Writer(local, "lk")
    w.png("prev", grays[0])
    w.png("curr", grays[1])
    w.mat("flow", flow.reshape(-1, 2).astype(np.float64))
    w.close()


EXPORTERS = {"lk": export_lk, "yolo": export_yolo, "hls": export_hls, "harness": export_harness, "tracker": export_tracker, "ego": export_ego, "scale": export_scale, "horn": export_horn,
             "divergence": export_divergence, "fusion": export_fusion, "collision": export_collision, "pipeline": export_pipeline}


def record_detections(video: Path, out_file: Path, max_frames: int, weights: str = "yolov9t.pt") -> None:
    """Per-frame YOLOv9t detections at 512x384, for fcw_app --detections on the PC."""
    from collision_avoidance.fcw.__main__ import make_detector

    detector = make_detector(weights, 0.5, "auto")
    cap = cv2.VideoCapture(str(video))
    fs = cv2.FileStorage(str(out_file), cv2.FILE_STORAGE_WRITE)
    i = 0
    while not max_frames or i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector(cv2.resize(frame, (512, 384)))
        fs.write(f"f{i}_dets", np.array([d.bbox for d in dets], np.float64).reshape(-1, 4))
        fs.write(f"f{i}_cls", ",".join(d.class_name for d in dets))
        i += 1
    fs.write("frames", i)
    fs.release()
    print("wrote", out_file, i, "frames")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("host/tests/golden"))
    p.add_argument("--only", nargs="*", choices=sorted(EXPORTERS))
    p.add_argument("--record-detections", type=Path, help="video to run YOLOv9t on (instead of exporting goldens)")
    p.add_argument("--record-out", type=Path, default=Path("detections.yml.gz"))
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args(argv)
    if args.record_detections:
        record_detections(args.record_detections, args.record_out, args.max_frames)
        return
    for name in args.only or EXPORTERS:
        EXPORTERS[name](args.out)


if __name__ == "__main__":
    main()
