"""Native GPU implementation of the FPGA-accelerated stages, with measured GPU power.

    python eval/bench_gpu.py --source clip.mp4 --frames 30
    python eval/bench_gpu.py --selftest

Times on the GPU what the KV260 design puts in the fabric and the DPU:
preprocessing (upload of the raw frame, resize to 512x384, gray, RGB tensor),
dense pyramidal Lucas-Kanade flow (5 levels, 5 iterations, 11x11 window, the
Vitis kernel's parameters, exact per-window LK batched over the 121 window offsets) and
YOLOv9t. The per-object stages stay on the CPU in a GPU design too (see
host/tools/cpu_baseline.cpp). GPU board power is sampled with nvidia-smi while
the full GPU workload runs flat out and while it is paced at 30 FPS.
"""

import argparse
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from collision_avoidance.fcw.lk_flow import LkFlow  # the board's LK, exact per window (shared with the pipeline)

_LK = LkFlow(device="cuda")
dense_lk = _LK._flow   # (1, 1, H, W) tensors on the GPU -> (1, 2, H, W)


def selftest() -> None:
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (384, 512)).astype(np.uint8), (0, 0), 2.0)
    moved = cv2.warpAffine(tex, np.float32([[1, 0, 1.6], [0, 1, -0.7]]), (512, 384), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    t = lambda a: torch.from_numpy(a).float().cuda()[None, None]
    f = dense_lk(t(tex), t(moved))[0].cpu().numpy()[:, 32:352, 32:480]
    err = np.median(np.hypot(f[0] - 1.6, f[1] + 0.7))
    print(f"selftest: shift (1.6, -0.7) px, median endpoint error {err:.3f} px -> {'ok' if err < 0.05 else 'FAIL'}")


class PowerLog:
    """nvidia-smi power.draw samples every 100 ms while the block runs."""

    def __enter__(self):
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
                                  stdout=subprocess.PIPE, text=True)
        time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        out, _ = self.p.communicate()
        self.samples = [float(v) for v in out.split() if v.replace(".", "", 1).isdigit()]

    @property
    def mean_w(self) -> float:
        return float(np.mean(self.samples[5:] if len(self.samples) > 10 else self.samples))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--seconds", type=float, default=20.0, help="duration of each power measurement")
    p.add_argument("--weights", default="yolov9t.pt")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    if args.selftest:
        selftest()
        return
    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.source)
    raw = []
    while len(raw) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(f)
    model = YOLO(args.weights)
    dev = torch.device("cuda")
    sync = torch.cuda.synchronize

    def preprocess(f):
        x = torch.from_numpy(f).to(dev, non_blocking=False).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(384, 512), mode="area")
        gray = (0.114 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.299 * x[:, 2:3])
        rgb = x[:, [2, 1, 0]] / 255.0
        return gray, rgb

    def detect(rgb):
        return model.predict(rgb, imgsz=(384, 512), device=0, verbose=False)

    def step(i, prev_gray):
        gray, rgb = preprocess(raw[i % len(raw)])
        flow = dense_lk(prev_gray, gray) if prev_gray is not None else None
        detect(rgb)
        return gray, flow

    for i in range(3):  # warm-up: CUDA context, cuDNN autotune
        g, rgb = preprocess(raw[i])
        dense_lk(g, g)
        detect(rgb)
    sync()

    times = {"preprocess": [], "dense_lk": [], "yolo": []}
    prev = None
    for f in raw:
        t0 = time.perf_counter(); gray, rgb = preprocess(f); sync(); times["preprocess"].append(time.perf_counter() - t0)
        if prev is not None:
            t0 = time.perf_counter(); dense_lk(prev, gray); sync(); times["dense_lk"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); detect(rgb); sync(); times["yolo"].append(time.perf_counter() - t0)
        prev = gray
    for k, v in times.items():
        print(f"{k:12s} {np.mean(v) * 1000:8.2f} ms/frame (median {np.median(v) * 1000:.2f})")

    with PowerLog() as idle:
        time.sleep(args.seconds / 2)
    n, prev = 0, None
    with PowerLog() as flat:
        t_end = time.perf_counter() + args.seconds
        t0 = time.perf_counter()
        while time.perf_counter() < t_end:
            prev, _ = step(n, prev); sync(); n += 1
        flat_ms = (time.perf_counter() - t0) / n * 1000
    n, prev = 0, None
    with PowerLog() as paced:  # at most 30 FPS: sleeps only when the GPU is faster than that
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            prev, _ = step(n, prev); sync(); n += 1
            time.sleep(max(0.0, t0 + n / 30 - time.perf_counter()))
        paced_fps = n / (time.perf_counter() - t0)
    print(f"gpu_frame    {flat_ms:8.2f} ms/frame flat out ({1000 / flat_ms:.1f} FPS)")
    print(f"gpu_power    idle {idle.mean_w:.1f} W | flat out {flat.mean_w:.1f} W | capped at 30 FPS {paced.mean_w:.1f} W "
          f"(achieved {paced_fps:.1f} FPS)")
    print(f"gpu_energy   {flat.mean_w * flat_ms / 1000:.2f} J/frame flat out | {paced.mean_w / paced_fps:.2f} J/frame capped "
          f"(GPU board power only; the CPU is not included)")


if __name__ == "__main__":
    main()
