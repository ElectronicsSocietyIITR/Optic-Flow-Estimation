import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.divergence import divergence_ttc, flow_moments, robust_mask
from collision_avoidance.fcw.ego_rotation import rotation_divergence

DT = 1 / 30
W, H = 512, 384
K = synth.intrinsics(500, W, H)
PP = (K[0, 2], K[1, 2])
BOX = (180, 140, 300, 230)


def looming_flow(rate, center=(240, 185), lateral=(0.0, 0.0)):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.stack([(xs - center[0]) * rate + lateral[0], (ys - center[1]) * rate + lateral[1]], axis=-1)


def test_recovers_ttc_from_expanding_flow():
    rate = 0.025  # per frame -> TTC = 40 frames = 1.333 s
    m = divergence_ttc(flow_moments(looming_flow(rate, lateral=(2.0, -1.0)), BOX, PP), BOX, 0.0, 1, DT)
    assert m.ttc_s == pytest.approx(1 / (rate * 30), rel=1e-6)


def test_camera_rotation_is_removed():
    rate = 0.02
    R = cv2.Rodrigues(np.array([0.01, -0.015, 0.0]))[0]
    _, rot = synth.flow_from_motion(K, R, np.zeros(3), 50.0, W, H, step=1)
    flow = looming_flow(rate) + rot.reshape(H, W, 2)
    center = ((BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2)
    m = divergence_ttc(flow_moments(flow, BOX, PP), BOX, rotation_divergence(R, K, center), 1, DT)
    assert m.ttc_s == pytest.approx(1 / (rate * 30), rel=0.02)
    uncorrected = divergence_ttc(flow_moments(flow, BOX, PP), BOX, 0.0, 1, DT)
    assert abs(uncorrected.ttc_s - 1 / (rate * 30)) > 0.05  # the correction matters here


def test_robust_refit_ignores_background_pixels():
    rate = 0.02
    flow = looming_flow(rate)
    flow[140:160, 180:300] = (5.0, 0.0)  # background leaking into the box top
    mask = robust_mask(flow, BOX, PP)
    m = divergence_ttc(flow_moments(flow, BOX, PP, mask), BOX, 0.0, 1, DT)
    assert m.ttc_s == pytest.approx(1 / (rate * 30), rel=0.02)


def test_empty_box_returns_none():
    assert flow_moments(looming_flow(0.02), (600, 10, 700, 50), PP) is None
    assert divergence_ttc(None, BOX, 0.0, 1, DT) is None


def test_ransac_samples_are_reproducible_across_languages():
    # Fixed 64-bit LCG so the C++ port (and HLS) draw exactly the same RANSAC samples.
    from collision_avoidance.fcw.divergence import ransac_samples
    samples = ransac_samples(1000, 4, seed=0)
    assert samples == ransac_samples(1000, 4, seed=0)
    assert all(len(set(s)) == 3 and all(0 <= i < 1000 for i in s) for s in samples)
    assert samples[0] == (502, 397, 989)   # pinned: the C++ test checks the same first triple


def test_fixed_point_moments_are_exact_for_quantised_flow_and_close_to_float():
    # The PL engine gets flow as int16 in 1/64 px and sums integers; this is its golden model.
    from collision_avoidance.fcw.divergence import FLOW_SCALE, flow_moments_fixed, quantise_flow
    rate = 0.02
    flow = looming_flow(rate, lateral=(1.3, -0.7))
    q = quantise_flow(flow)
    fixed = flow_moments_fixed(q, BOX, (256, 192))
    assert fixed.dtype == np.int64
    as_float = fixed.astype(np.float64)
    as_float[6:] /= FLOW_SCALE                        # u, v terms back to pixels
    np.testing.assert_array_equal(as_float, flow_moments(q.astype(np.float64) / FLOW_SCALE, BOX, (256, 192)))
    m = divergence_ttc(as_float, BOX, 0.0, 1, DT)
    assert m.ttc_s == pytest.approx(1 / (rate * 30), rel=0.01)


def _affine_field(a, b, pp=(256.0, 192.0), h=384, w=512, seed=0):
    ys, xs = np.mgrid[0:h, 0:w]
    x, y = xs - pp[0], ys - pp[1]
    noise = np.random.default_rng(seed).uniform(-0.3, 0.3, (h, w, 2))
    flow = np.stack([a[0] + a[1] * x + a[2] * y, b[0] + b[1] * x + b[2] * y], -1) + noise
    return flow.astype(np.float32)


A_TRUE, B_TRUE = (0.8, 0.02, 0.001), (-0.3, 0.0005, 0.025)
BOX_FIT = (180.0, 120.0, 330.0, 260.0)


def test_predicted_mask_matches_float_residual():
    from collision_avoidance.fcw.divergence import fit_to_fixed, flow_rect, predicted_mask_fixed, quantise_flow
    flow = _affine_field(A_TRUE, B_TRUE)
    flow[150:200, 200:260] += np.array([5.0, -4.0], np.float32)           # background leaking into the box
    flow_q = quantise_flow(flow)
    rect = flow_rect(BOX_FIT, 0.10, 512, 384)
    mask = predicted_mask_fixed(flow_q, rect, (256, 192), fit_to_fixed(np.array(A_TRUE), np.array(B_TRUE)), 64)
    x1, y1, x2, y2 = rect
    ys, xs = np.mgrid[y1:y2, x1:x2]
    x, y = xs - 256, ys - 192
    fq = flow_q[y1:y2, x1:x2] / 64.0
    ref = np.hypot(fq[..., 0] - (A_TRUE[0] + A_TRUE[1] * x + A_TRUE[2] * y),
                   fq[..., 1] - (B_TRUE[0] + B_TRUE[1] * x + B_TRUE[2] * y)) < 1.0
    assert mask.shape == ref.shape and (mask == ref).all()
    assert not mask[150 - y1:200 - y1, 200 - x1:260 - x1].any()


def test_fixed_moments_with_mask_match_float():
    from collision_avoidance.fcw.divergence import flow_moments, flow_moments_fixed, quantise_flow
    flow_q = quantise_flow(_affine_field(A_TRUE, B_TRUE))
    mask = np.random.default_rng(3).random((384, 512)) > 0.3
    fixed = flow_moments_fixed(flow_q, BOX_FIT, (256, 192), 0.10, mask=mask).astype(np.float64)
    fixed[6:] /= 64.0
    ref = flow_moments(flow_q.astype(np.float64) / 64.0, BOX_FIT, (256, 192), mask, 0.10)
    np.testing.assert_allclose(fixed, ref, rtol=1e-9)


def test_previous_fit_is_used_when_fresh():
    from collision_avoidance.fcw.divergence import DivergenceConfig, divergence_mask, quantise_flow
    flow = _affine_field(A_TRUE, B_TRUE)
    mask, used = divergence_mask(flow, quantise_flow(flow), BOX_FIT, (256, 192), (np.array(A_TRUE), np.array(B_TRUE)), 1, DivergenceConfig())
    assert used and mask.mean() > 0


def test_stale_fit_falls_back_to_ransac():
    from collision_avoidance.fcw.divergence import DivergenceConfig, divergence_mask, quantise_flow
    flow = _affine_field(A_TRUE, B_TRUE)
    _, used = divergence_mask(flow, quantise_flow(flow), BOX_FIT, (256, 192), (np.array(A_TRUE), np.array(B_TRUE)), 2, DivergenceConfig())
    assert not used
    _, used = divergence_mask(flow, quantise_flow(flow), BOX_FIT, (256, 192), None, 1, DivergenceConfig())
    assert not used


def test_low_inlier_share_falls_back():
    from collision_avoidance.fcw.divergence import DivergenceConfig, divergence_mask, quantise_flow
    flow = _affine_field(A_TRUE, B_TRUE)
    wrong = (np.array([4.0, 0.0, 0.0]), np.array(B_TRUE))                 # predicts 3+ px off everywhere
    _, used = divergence_mask(flow, quantise_flow(flow), BOX_FIT, (256, 192), wrong, 1, DivergenceConfig())
    assert not used
