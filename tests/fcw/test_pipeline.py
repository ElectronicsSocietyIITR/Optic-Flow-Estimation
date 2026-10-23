import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
from collision_avoidance.tracking import Detection


def run_scene(scene_cfg: synth.SceneConfig, class_name: str = "car"):
    scene = synth.render_scene(scene_cfg)
    frames = iter(range(len(scene.frames)))
    current = {"i": 0}

    def detector(_frame):
        return [Detection(scene.boxes[current["i"]], class_name, 0.9)]

    pipeline = FcwPipeline(FcwConfig(fx=scene_cfg.fx), detector, DisFlow())
    results = []
    for i in frames:
        current["i"] = i
        results.append(pipeline.process(cv2.cvtColor(scene.frames[i], cv2.COLOR_GRAY2BGR), i / scene_cfg.fps))
    return scene, results


def first_frame_at(results, level):
    return next((r.index for r in results if r.level >= level), None)


def test_head_on_approach_warns_in_time():
    scene, results = run_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0))
    warn = first_frame_at(results, Level.WARNING)
    crit = first_frame_at(results, Level.CRITICAL)
    assert warn is not None and scene.ttc_s[warn] >= 2.0          # NHTSA: warned before TTC 2.0 s
    assert scene.ttc_s[warn] <= 3.0                                # and not absurdly early
    assert crit is not None and scene.ttc_s[crit] >= 1.0
    final = results[-1].objects[0].estimate
    assert final.ttc_s == pytest.approx(scene.ttc_s[-1], rel=0.15)


def test_car_in_the_next_lane_never_warns():
    _, results = run_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0, lateral_m=3.5))
    assert first_frame_at(results, Level.WARNING) is None


def test_static_object_never_warns():
    _, results = run_scene(synth.SceneConfig(n_frames=40, z0_m=20.0, closing_speed_mps=0.0))
    assert first_frame_at(results, Level.WARNING) is None
    assert results[-1].objects[0].estimate.ttc_s is None


def test_dropped_frames_use_real_timestamps():
    scene = synth.render_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)], DisFlow())
    for current in range(0, 80, 2):  # every other frame arrives
        result = pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30)
    assert result.objects[0].estimate.ttc_s == pytest.approx(scene.ttc_s[78], rel=0.15)


def test_repeated_timestamp_does_not_divide_by_zero():
    scene = synth.render_scene(synth.SceneConfig(n_frames=4))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[0], "car", 0.9)], DisFlow())
    for gray in scene.frames:
        result = pipeline.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.0)
    assert all(np.isfinite(v) for v in result.timings_ms.values())


def test_box_leaving_the_frame_is_handled():
    _, results = run_scene(synth.SceneConfig(n_frames=30, z0_m=12.0, closing_speed_mps=3.0, lateral_m=-2.5, lateral_speed_mps=-3.0))
    assert results[-1].objects[0].bbox[0] < 0  # partly outside on the left
    assert first_frame_at(results, Level.WARNING) is None


def test_ttc_is_limited_to_sixteen_tracks():
    boxes = [np.array([10 + 30 * (i % 10), 100 + 60 * (i // 10), 35 + 30 * (i % 10), 150 + 60 * (i // 10)], float) for i in range(20)]
    gray = synth.texture(384, 512, 0)
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(b, "car", 0.9) for b in boxes], DisFlow())
    for i in range(3):
        result = pipeline.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), i / 30)
    assert len(pipeline.tracker.tracks) == 20 and len(result.objects) == 16


def test_scale_search_uses_the_real_time_span_when_frames_arrive_irregularly():
    # Every third frame is lost: gaps alternate between 1/30 s and 2/30 s, so k * (latest dt) is not the span.
    scene = synth.render_scene(synth.SceneConfig(n_frames=60, z0_m=40.0, closing_speed_mps=15.0))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)], DisFlow())
    errors = []
    for current in [i for i in range(60) if i % 3 != 2]:
        result = pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30)
        scale = [m for m in result.objects[0].measurements if m.method == "scale"]
        if current >= 30 and scale:
            errors.append(abs(scale[0].ttc_s - scene.ttc_s[current]) / scene.ttc_s[current])
    assert errors and max(errors) < 0.1


def test_out_of_order_timestamp_does_not_inflate_the_next_step():
    scene = synth.render_scene(synth.SceneConfig(n_frames=4))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[0], "car", 0.9)], DisFlow())
    times = [0.0, 1 / 30, 0.0, 2 / 30]  # the third frame arrives with a stale timestamp
    results = [pipeline.process(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), t) for g, t in zip(scene.frames, times)]
    assert results[2].dt_s == pytest.approx(1 / 30)   # fallback to the nominal period
    assert results[3].dt_s == pytest.approx(1 / 30)   # measured from 1/30, not from the stale 0.0


def test_lost_track_cannot_escalate_or_hold_a_warning():
    # The object is detected until true TTC is ~2 s, then the detector loses it.
    scene = synth.render_scene(synth.SceneConfig(n_frames=70, z0_m=50.0, closing_speed_mps=15.0))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)] if current < 40 else [], DisFlow())
    levels = []
    for current in range(70):
        levels.append(pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30).level)
    assert Level.CRITICAL not in levels[40:]   # no escalation from extrapolated predictions
    assert levels[-1] == Level.NONE            # cleared while the track is lost, before it is deleted


def _shifted(box, dx_widths=0.0, scale=1.0):
    x1, y1, x2, y2 = box
    w, h, cx, cy = x2 - x1, y2 - y1, (x1 + x2) / 2 + dx_widths * (x2 - x1), (y1 + y2) / 2
    return np.array([cx - scale * w / 2, cy - scale * h / 2, cx + scale * w / 2, cy + scale * h / 2])


def _levels_per_track(scene, detections_for):
    pipeline = FcwPipeline(FcwConfig(), lambda f: detections_for(current), DisFlow())
    per_track = {}
    for current in range(len(scene.frames)):
        result = pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30)
        for o in result.objects:
            per_track.setdefault(o.track_id, []).append(o.level)
    return per_track


def test_car_beside_our_path_never_warns():
    # 1.6 m off centre (less than one object width): closing fast, but beside us, not ahead.
    scene = synth.render_scene(synth.SceneConfig(n_frames=70, z0_m=45.0, closing_speed_mps=15.0))
    levels = _levels_per_track(scene, lambda i: [Detection(_shifted(scene.boxes[i], 1.6 / 1.8), "car", 0.9)])
    assert all(lv == Level.NONE for track in levels.values() for lv in track)


def test_only_the_nearest_object_in_our_path_can_warn():
    # A farther car straight ahead (half the image size) closes at the same rate as the lead.
    scene = synth.render_scene(synth.SceneConfig(n_frames=70, z0_m=45.0, closing_speed_mps=15.0))
    levels = _levels_per_track(scene, lambda i: [Detection(scene.boxes[i], "car", 0.9), Detection(_shifted(scene.boxes[i], 0.0, 0.5), "car", 0.9)])
    lead, behind = levels[0], levels[1]
    assert Level.WARNING in lead or Level.CRITICAL in lead
    assert all(lv == Level.NONE for lv in behind)


def _scene_levels(scene_cfg):
    scene = synth.render_scene(scene_cfg)
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)], DisFlow())
    levels = []
    for current in range(len(scene.frames)):
        levels.append(pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30).level)
    return scene, levels


def test_oncoming_car_straight_ahead_does_not_warn():
    # We drive at 10 m/s; it comes at 15 m/s: closing 2.5x our speed, the oncoming-traffic signature.
    _, levels = _scene_levels(synth.SceneConfig(n_frames=70, ground=True, ego_speed_mps=10.0, closing_speed_mps=25.0, z0_m=70.0))
    assert all(lv == Level.NONE for lv in levels)


def test_parked_car_straight_ahead_still_warns():
    scene, levels = _scene_levels(synth.SceneConfig(n_frames=70, ground=True, ego_speed_mps=10.0, closing_speed_mps=10.0, z0_m=30.0))
    first = next(i for i, lv in enumerate(levels) if lv >= Level.WARNING)
    assert scene.ttc_s[first] >= 2.0


def _first_warning_ttc(cls, scene_cfg):
    scene = synth.render_scene(scene_cfg)
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], cls, 0.9)], DisFlow())
    for current in range(len(scene.frames)):
        if pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30).level >= Level.WARNING:
            return scene.ttc_s[current]
    return None


_ROAD = dict(ground=True, ego_speed_mps=10.0, n_frames=85)


def test_pedestrian_crossing_into_our_path_warns_early():
    # Reaches our centre line exactly at contact: must be predicted, not waited for.
    ttc = _first_warning_ttc("person", synth.SceneConfig(**_ROAD, closing_speed_mps=10.0, z0_m=30.0, lateral_m=-3.0,
                                                         lateral_speed_mps=1.0, obj_width_m=0.5, obj_height_m=1.7, obj_y_m=0.2))
    assert ttc is not None and ttc >= 1.5


def test_car_cutting_into_our_lane_warns_early():
    ttc = _first_warning_ttc("car", synth.SceneConfig(**_ROAD, closing_speed_mps=8.0, z0_m=25.0, lateral_m=3.0, lateral_speed_mps=-0.96))
    assert ttc is not None and ttc >= 1.5


def test_pedestrian_who_clears_our_path_does_not_warn():
    # Crosses at 1.6 m/s: 1.8 m right of centre (0.65 m clear) by the time we arrive.
    ttc = _first_warning_ttc("person", synth.SceneConfig(**_ROAD, closing_speed_mps=10.0, z0_m=30.0, lateral_m=-3.0,
                                                         lateral_speed_mps=1.6, obj_width_m=0.5, obj_height_m=1.7, obj_y_m=0.2))
    assert ttc is None


def test_scale_search_can_be_switched_off():
    from dataclasses import replace
    scene = synth.render_scene(synth.SceneConfig(n_frames=12))
    for enabled in (True, False):
        pipeline = FcwPipeline(replace(FcwConfig(), scale_enabled=enabled), lambda f: [Detection(scene.boxes[i], "car", 0.9)], DisFlow())
        methods = set()
        for i in range(len(scene.frames)):
            r = pipeline.process(cv2.cvtColor(scene.frames[i], cv2.COLOR_GRAY2BGR), i / 30)
            methods |= {m.method for o in r.objects for m in o.measurements}
        assert ("scale" in methods) is enabled


@pytest.mark.parametrize("src, expected", [((3840, 2160), (512, 288)), ((1920, 1200), (512, 320)), ((640, 480), (512, 384)),
                                           ((1242, 375), (512, 160)), ((480, 640), (512, 512)), ((100, 10), (512, 64))])
def test_processing_size_keeps_the_source_aspect(src, expected):
    from collision_avoidance.fcw.pipeline import processing_size
    assert processing_size(*src) == expected


def test_config_for_source_keeps_pixels_square():
    from collision_avoidance.fcw.pipeline import config_for_source
    cfg = config_for_source(3840, 2160, fx=200.0)
    assert (cfg.frame_width, cfg.frame_height) == (512, 288)
    assert cfg.K[1, 1] == pytest.approx(200.0)                 # 16:9 into 16:9: square pixels
    kitti = config_for_source(1242, 375, fx=200.0)
    assert kitti.K[1, 1] == pytest.approx(200.0 * (160 / 375) / (512 / 1242))


def test_bgr_to_gray_is_the_fabric_formula():
    # (4899 R + 9617 G + 1868 B + 2^13) >> 14, as the preprocessing kernel computes it; cv2.cvtColor's
    # rounding depends on the OpenCV build (its fast paths), so the PC must not rely on it.
    from collision_avoidance.fcw.pipeline import bgr_to_gray
    bgr = np.random.default_rng(0).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    b, g, r = (bgr[..., i].astype(np.int64) for i in range(3))
    assert np.array_equal(bgr_to_gray(bgr), ((4899 * r + 9617 * g + 1868 * b + 8192) >> 14).astype(np.uint8))
    assert np.abs(bgr_to_gray(bgr).astype(int) - cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)).max() <= 1
    flat = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    assert np.array_equal(bgr_to_gray(flat)[0], np.arange(256))   # gray input stays exactly itself


def test_horn_pyramid_matches_the_fabric_pyrdown():
    # Vitis pyrDown equals cv2.pyrDown with replicated borders (C-simulation, hls/preprocess).
    from collision_avoidance.fcw.horn import downsample
    img = np.random.default_rng(1).integers(0, 256, (96, 128), dtype=np.uint8)
    assert np.array_equal(downsample(img, 1), cv2.pyrDown(img, borderType=cv2.BORDER_REPLICATE))
