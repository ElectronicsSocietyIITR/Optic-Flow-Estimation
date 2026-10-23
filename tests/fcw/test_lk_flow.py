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
