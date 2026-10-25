"""The board's flow on the PC: AMD's densePyrOpticalFlow in C-simulation (hls/lk_ref), cached per frame pair.

Our software LK (lk_flow.py) is close to the kernel but not equal to it (median 0.19 px,
90th percentile 1.8 px on dash-cam frames: Gaussian pyramid, per-pass median filter,
fixed-point arithmetic). For results that must hold on the board, use this source.
The kernel takes ~4 s per frame pair in C-simulation, so evaluations fill the cache first
with prefill(), many pairs in parallel, then read it.

Needs WSL with hls/lk_ref built (make -C hls/lk_ref build/lk_ref).
"""

import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WSL_DISTRO = "Ubuntu-24.04"


def pair_key(prev: np.ndarray, curr: np.ndarray) -> str:
    h = hashlib.sha1()
    for a in (prev, curr):
        h.update(np.asarray(a.shape, np.int64).tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _wsl_path(p: Path) -> str:
    p = Path(p).resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}" if p.drive else p.as_posix()


class VitisFlow:
    """FlowSource: (prev_gray, curr_gray) uint8 -> (H, W, 2) float32 px, as the board's LK kernel computes it."""

    def __init__(self, cache_dir: Path | None = None, kernel: Path | None = None, median: int = 5, scale: int = 1,
                 lines: int = 50):
        """median: the per-pass median window on the flow: 5 (AMD's), 3 (lk_ref_med3), 0 = off (lk_ref_nomed); own caches.
        scale 2: run on pyramid level 1 (cv2.pyrDown, replicated borders) and upsample the flow back (bilinear, x2).
        lines: the remap line buffer (NUM_LINES; AMD's example uses 50), with the 3x3 or no median."""
        median = 5 if median is True else 0 if median is False else median
        if lines != 50 and median == 5:
            raise ValueError("lines other than 50 are built only for the 3x3 or no median (lk_ref_med3_l<lines>, lk_ref_nomed_l<lines>)")
        suffix = {5: "", 3: "_med3", 0: "_nomed"}[median] + ("" if lines == 50 else f"_l{lines}")
        self.median, self.scale, self.lines = median, scale, lines
        self.cache = Path(cache_dir or ROOT / "outputs" / f"vitis_flow_cache{suffix}")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.kernel = Path(kernel or ROOT / "hls" / "lk_ref" / "build" / f"lk_ref{suffix}")

    def _down(self, img: np.ndarray) -> np.ndarray:
        for _ in range(self.scale.bit_length() - 1):
            img = cv2.pyrDown(img, borderType=cv2.BORDER_REPLICATE)
        return img

    def __call__(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        p, c = self._down(prev), self._down(curr)
        path = self.cache / f"{pair_key(p, c)}.bin"
        if not path.exists():
            self.prefill([(prev, curr)], jobs=1)
        flow = np.fromfile(path, np.float32).reshape(*p.shape, 2)
        if self.scale > 1:
            flow = self.scale * cv2.resize(flow, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_LINEAR)
        return flow

    def prefill(self, pairs, jobs: int = 12) -> int:
        """Run the kernel for every pair not yet cached, `jobs` at a time. Returns how many ran."""
        todo, lines = {}, []
        for prev, curr in pairs:
            prev, curr = self._down(prev), self._down(curr)
            key = pair_key(prev, curr)
            if key in todo or (self.cache / f"{key}.bin").exists():
                continue
            todo[key] = True
            cv2.imwrite(str(self.cache / f"{key}_prev.png"), prev)
            cv2.imwrite(str(self.cache / f"{key}_curr.png"), curr)
            lines.append(f"{key}_prev.png {key}_curr.png {key}.bin")
        if not lines:
            return 0
        (self.cache / "todo.txt").write_text("\n".join(lines) + "\n", newline="\n")  # no \r in names for WSL's xargs
        cmd = f"cd {_wsl_path(self.cache)} && xargs -P {jobs} -n 3 {_wsl_path(self.kernel)} < todo.txt > /dev/null"
        subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", cmd], check=True)
        for key in todo:
            for suffix in ("_prev.png", "_curr.png"):
                (self.cache / f"{key}{suffix}").unlink(missing_ok=True)
            if not (self.cache / f"{key}.bin").exists():
                raise RuntimeError(f"the LK kernel produced no flow for pair {key}")
        return len(lines)
