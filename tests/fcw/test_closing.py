import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.closing import ClosingTracker, road_constant, static_ttc
from collision_avoidance.fcw.pipeline import DisFlow

DT = 1 / 30
K = synth.intrinsics(500, 512, 384)
PP = (256.0, 192.0)


def kappa_at(closing_speed, frame=20):
    scene = synth.render_scene(synth.SceneConfig(n_frames=frame + 1, ground=True, ego_speed_mps=10.0,
                                                 closing_speed_mps=closing_speed, z0_m=40.0))
    flow = DisFlow()(scene.frames[frame - 1], scene.frames[frame])
    c = road_constant(flow, PP, np.eye(3), K, [scene.boxes[frame]])
    return static_ttc(c, scene.boxes[frame], PP), scene.depth_m[frame], scene.ttc_s[frame]


def test_road_gives_our_own_ttc_to_a_parked_car():
    static_frames, depth, ttc = kappa_at(10.0)         # parked: closes at our speed
    assert static_frames * DT == pytest.approx(depth / 10.0, rel=0.1)
    assert static_frames * DT / ttc == pytest.approx(1.0, abs=0.1)


def test_oncoming_car_closes_faster_than_we_drive():
    static_frames, _, ttc = kappa_at(25.0)              # oncoming at 15 m/s while we do 10
    assert static_frames * DT / ttc == pytest.approx(2.5, rel=0.1)


def test_car_ahead_going_our_way_closes_slower():
    static_frames, _, ttc = kappa_at(5.0)               # it drives away at 5 m/s while we do 10
    assert static_frames * DT / ttc == pytest.approx(0.5, rel=0.15)


def test_no_estimate_without_road_motion_or_below_the_horizon():
    still = np.zeros((384, 512, 2), np.float32)       # stopped: the road does not move
    assert road_constant(still, PP, np.eye(3), K) is None
    assert static_ttc(100.0, (100, 150, 200, 193), PP) is None   # contact row at the horizon


def test_tracker_smooths_with_a_median_and_forgets_tracks():
    tracker = ClosingTracker()
    for k in (1.0, 1.1, 9.0, 0.9, 1.0):
        smoothed = tracker.update(3, k)
    assert smoothed == pytest.approx(1.0)
    assert tracker.update(3, None) == pytest.approx(1.0)  # a missing sample keeps the history
    tracker.retain([])
    assert tracker.update(3, None) is None
    for c in (50.0, None, 60.0, 1000.0):
        smoothed_c = tracker.update_c(c)
    assert smoothed_c == pytest.approx(60.0)


@pytest.mark.parametrize("closing_speed", [10.0, 25.0, 5.0])
def test_coarse_road_grid_keeps_kappa(closing_speed):
    # 8 x 4 px sampling (4x fewer points) reads kappa within 5 % of the dense 4 x 2 grid.
    from collision_avoidance.fcw.closing import ClosingConfig
    assert ClosingConfig().grid_step_px == (8, 4)
    scene = synth.render_scene(synth.SceneConfig(n_frames=21, ground=True, ego_speed_mps=10.0,
                                                 closing_speed_mps=closing_speed, z0_m=40.0))
    flow = DisFlow()(scene.frames[19], scene.frames[20])
    box = scene.boxes[20]
    dense = road_constant(flow, PP, np.eye(3), K, [box], ClosingConfig(grid_step_px=(4, 2)))
    coarse = road_constant(flow, PP, np.eye(3), K, [box])
    assert coarse == pytest.approx(dense, rel=0.05)
