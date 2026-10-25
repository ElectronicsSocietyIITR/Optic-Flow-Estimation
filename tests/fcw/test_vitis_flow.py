import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from collision_avoidance.fcw.vitis_flow import VitisFlow, pair_key

KERNEL = Path("hls/lk_ref/build/lk_ref")


def _pair(h=96, w=128, shift=(1.6, -0.7)):
    tex = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, (h, w)).astype(np.uint8), (0, 0), 2.0)
    moved = cv2.warpAffine(tex, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    return tex, moved


def test_cached_pair_is_served_without_running_the_kernel(tmp_path):
    prev, curr = _pair()
    stored = np.random.default_rng(1).normal(size=(*prev.shape, 2)).astype(np.float32)
    stored.tofile(tmp_path / f"{pair_key(prev, curr)}.bin")
    flow = VitisFlow(tmp_path, kernel=tmp_path / "missing")(prev, curr)
    np.testing.assert_array_equal(flow, stored)


def test_key_depends_on_both_frames():
    prev, curr = _pair()
    assert pair_key(prev, curr) != pair_key(curr, prev) != pair_key(prev, prev)


@pytest.mark.skipif(not KERNEL.exists() or shutil.which("wsl") is None, reason="build hls/lk_ref first (make -C hls/lk_ref build/lk_ref in WSL)")
def test_prefill_runs_the_kernel_for_missing_pairs(tmp_path):
    prev, curr = _pair()
    vf = VitisFlow(tmp_path)
    vf.prefill([(prev, curr)], jobs=2)
    assert (tmp_path / f"{pair_key(prev, curr)}.bin").exists()
    flow = vf(prev, curr)
    err = np.hypot(flow[16:-16, 16:-16, 0] - 1.6, flow[16:-16, 16:-16, 1] + 0.7)
    assert np.median(err) < 0.1


def test_make_flow_reads_vitis_settings():
    from collision_avoidance.fcw.pipeline import make_flow

    f = make_flow("vitis:median=0,scale=2")
    assert (f.median, f.scale) == (False, 2)
    assert f.kernel.name == "lk_ref_nomed" and f.cache.name == "vitis_flow_cache_nomed"
    g = make_flow("vitis")
    assert (g.median, g.scale, g.kernel.name, g.cache.name) == (5, 1, "lk_ref", "vitis_flow_cache")
    h = make_flow("vitis:median=3")
    assert (h.median, h.kernel.name, h.cache.name) == (3, "lk_ref_med3", "vitis_flow_cache_med3")


def test_make_flow_reads_the_line_buffer_setting():
    from collision_avoidance.fcw.pipeline import make_flow

    f = make_flow("vitis:median=0,lines=26")
    assert f.lines == 26 and f.kernel.name == "lk_ref_nomed_l26" and f.cache.name == "vitis_flow_cache_nomed_l26"


def test_make_flow_reads_the_3x3_median_with_fewer_lines():
    from collision_avoidance.fcw.pipeline import make_flow

    f = make_flow("vitis:median=3,lines=34")
    assert (f.median, f.lines, f.kernel.name, f.cache.name) == (3, 34, "lk_ref_med3_l34", "vitis_flow_cache_med3_l34")
