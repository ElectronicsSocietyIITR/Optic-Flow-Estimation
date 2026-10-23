"""Experiment: closing-speed ratio kappa = TTC_static_road / TTC_object at each warning.

The road patch just below an object's box is static and at the object's depth.
After removing camera rotation, its flow points away from the FOE with magnitude
|p - FOE| / TTC_static (per frame), so TTC_static needs no calibration. kappa is
~1 for parked objects, < 1 for traffic going our way, and well above 1 for
oncoming traffic.

    python eval/kappa_probe.py --video data/normal_driving/<clip> --frames 1186 1209 1389 --fx 200
"""

import argparse

import cv2
import numpy as np

from collision_avoidance.fcw.pipeline import DisFlow, FcwPipeline, config_for_source


def road_ttc_frames(flow, box, foe, R, K, rows=(3, 15)):
    """Median static TTC (frames) of the road strip below the box, or None."""
    h, w = flow.shape[:2]
    x1, x2 = int(max(0, box[0])), int(min(w, box[2]))
    y1, y2 = int(box[3] + rows[0]), int(min(h, box[3] + rows[1]))
    if x2 - x1 < 4 or y2 - y1 < 2:
        return None
    ys, xs = np.mgrid[y1:y2, x1:x2]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]).astype(np.float64)
    H = K @ R @ np.linalg.inv(K)
    rot = H @ pts
    rot_flow = (rot[:2] / rot[2] - pts[:2]).T
    f = flow[y1:y2, x1:x2].reshape(-1, 2) - rot_flow     # translational flow of the road
    radial = pts[:2].T - np.asarray(foe, float)
    dist = np.linalg.norm(radial, axis=1)
    speed = np.einsum("ij,ij->i", f, radial / dist[:, None])  # outward component
    ok = (dist > 5) & (speed > 0.05)
    if ok.sum() < 10:
        return None
    return float(np.median(dist[ok] / speed[ok]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--fx", type=float, default=200.0)
    args = p.parse_args(argv)
    from collision_avoidance.fcw.__main__ import make_detector

    det = make_detector("yolov9t.pt", 0.5, "auto")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe = FcwPipeline(config_for_source(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), args.fx), det, DisFlow())
    wanted, i = set(args.frames), 0
    while i <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        r = pipe.process(frame, i / fps)
        if i in wanted:
            for o in r.objects:
                if not (o.estimate and o.estimate.ttc_s):
                    continue
                static = road_ttc_frames(r.flow, o.bbox, r.heading, r.ego.R, pipe.K)
                kappa = None if static is None else static * r.dt_s / o.estimate.ttc_s
                tag = "<- warning" if o.level else ""
                print(f"frame {i} track {o.track_id} {o.class_name:6s} box {np.round(o.bbox).astype(int)} "
                      f"ttc {o.estimate.ttc_s:5.2f}s road-static ttc {'-' if static is None else f'{static * r.dt_s:5.2f}s'} "
                      f"kappa {'-' if kappa is None else f'{kappa:4.2f}'} {tag}")
        i += 1


if __name__ == "__main__":
    main()
