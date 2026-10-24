import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _shifted_pair():
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (384, 512)).astype(np.uint8), (0, 0), 2.0)
    moved = cv2.warpAffine(tex, np.float32([[1, 0, 1.6], [0, 1, -0.7]]), (512, 384), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    return tex, moved


def test_lk_recovers_a_sub_pixel_shift():
    from collision_avoidance.fcw.lk_flow import LkFlow
    tex, moved = _shifted_pair()
    flow = LkFlow()(tex, moved)
    assert flow.shape == (384, 512, 2) and flow.dtype == np.float32
    err = np.hypot(flow[32:352, 32:480, 0] - 1.6, flow[32:352, 32:480, 1] + 0.7)
    assert np.median(err) < 0.05


def test_flat_image_gives_zero_flow():
    # No texture: every pixel's gradient matrix is singular, so nothing moves (as in the C++ and Vitis kernels).
    from collision_avoidance.fcw.lk_flow import LkFlow
    flat = np.full((96, 128), 100, np.uint8)
    assert np.abs(LkFlow(levels=3)(flat, flat)).max() == 0.0


def test_make_flow_reads_lk_settings():
    from collision_avoidance.fcw.pipeline import make_flow

    f = make_flow("lk:levels=4,iters=3,win=7,scale=2")
    assert (f.levels, f.iterations, f.win, f.scale) == (4, 3, 7, 2)


def test_half_resolution_lk_recovers_a_shift_at_full_resolution():
    import cv2
    import numpy as np
    from collision_avoidance.fcw.lk_flow import LkFlow

    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(0, 256, (320, 560)).astype(np.uint8), (0, 0), 2.0)
    prev, curr = tex[16:304, 16:528], tex[16:304, 13:525]  # content moves 3 px right
    flow = LkFlow(levels=4, iterations=5, win=11, scale=2)(prev, curr)
    assert flow.shape == (288, 512, 2)
    core = flow[40:-40, 40:-40]
    assert abs(np.median(core[..., 0]) - 3.0) < 0.1 and abs(np.median(core[..., 1])) < 0.1
