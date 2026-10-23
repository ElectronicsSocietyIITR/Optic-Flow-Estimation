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

    def __init__(self, cache_dir: Path = ROOT / "outputs" / "vitis_flow_cache", kernel: Path = ROOT / "hls" / "lk_ref" / "build" / "lk_ref"):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.kernel = Path(kernel)

    def __call__(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        path = self.cache / f"{pair_key(prev, curr)}.bin"
        if not path.exists():
            self.prefill([(prev, curr)], jobs=1)
        return np.fromfile(path, np.float32).reshape(*prev.shape, 2)

    def prefill(self, pairs, jobs: int = 12) -> int:
        """Run the kernel for every pair not yet cached, `jobs` at a time. Returns how many ran."""
        todo, lines = {}, []
        for prev, curr in pairs:
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
