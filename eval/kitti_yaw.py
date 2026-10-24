"""Ego-rotation accuracy: estimated yaw rate vs KITTI raw OXTS (P1 exit: RMS < 1 deg/s).

    python eval/kitti_yaw.py data/kitti/2011_09_26/2011_09_26_drive_0005_sync [more drives...]

Runs at the native KITTI resolution with the rectified camera matrix; flow is
OpenCV DIS (the PC stand-in for the board's dense LK).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from collision_avoidance.fcw import kitti
from collision_avoidance.fcw.ego_rotation import EgoRotationEstimator
from collision_avoidance.fcw.metrics import rms
from collision_avoidance.fcw.pipeline import make_flow, processing_size
from collision_avoidance.telemetry import start_run


def evaluate_drive(drive_dir: Path, flow: str = "dis") -> tuple[np.ndarray, np.ndarray]:
    """(estimated, ground-truth) yaw rates in deg/s for every frame pair with a valid estimate."""
    images, times, K, wu = kitti.load_drive(drive_dir)
    load = lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if flow != "dis":  # the board's flows (vitis, lk*): at the processing size (512 wide), intrinsics scaled
        h0, w0 = load(images[0]).shape
        w, h = processing_size(w0, h0)
        K = np.diag([w / w0, h / h0, 1.0]) @ K
        load = lambda p, w=w, h=h: cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_AREA)
    flow_source, estimator = make_flow(flow), EgoRotationEstimator(K)
    if flow.startswith("vitis"):
        grays = [load(p) for p in images]
        flow_source.prefill(list(zip(grays, grays[1:])))
    est, gt = [], []
    prev = load(images[0])
    for i in range(1, len(images)):
        curr = load(images[i])
        ego = estimator.update(flow_source(prev, curr), [])
        dt = times[i] - times[i - 1]
        if ego.valid and dt > 0:
            est.append(np.degrees(ego.rvec[1] / dt))
            gt.append(np.degrees(0.5 * (wu[i] + wu[i - 1])))
        prev = curr
    return np.array(est), np.array(gt)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("drives", nargs="+", type=Path)
    p.add_argument("--flow", default="dis", help="dis | vitis | lk | lk:levels=N,iters=N,win=N,scale=N")
    args = p.parse_args(argv)
    run = start_run("fcw-eval-kitti-yaw", {"drives": [d.name for d in args.drives]}, tags=["fcw", "eval", "kitti"])
    all_est, all_gt = [], []
    for drive in args.drives:
        est, gt = evaluate_drive(drive, args.flow)
        all_est.append(est)
        all_gt.append(gt)
        print(f"{drive.name:40s} pairs={len(est):5d} rms={rms(est, gt):.3f} deg/s")
    total = rms(np.concatenate(all_est), np.concatenate(all_gt))
    print(f"{'overall':40s} rms={total:.3f} deg/s  (P1 exit: < 1.0)")
    if run is not None:
        run.summary.update({"yaw_rms_deg_s": total})
        run.finish()


if __name__ == "__main__":
    main()
