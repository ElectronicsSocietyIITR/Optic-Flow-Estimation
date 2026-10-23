"""Dump every warning raised on a folder of normal-driving videos, for manual review.

    python eval/review_warnings.py --videos data/normal_driving --fx 200 --out outputs/warning_review

For each NONE -> WARNING/CRITICAL transition it saves the annotated frame
(the raising object in red, other tracks grey) and appends one JSON line with
the numbers behind the decision. contact_sheet.jpg tiles all frames.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.pipeline import FcwPipeline, config_for_source, make_flow

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def review(video: Path, fx: float, detector, out: Path, log, flow: str = "dis") -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cfg = config_for_source(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fx)
    pipeline, prev, thumbs, i = FcwPipeline(cfg, detector, make_flow(flow)), {}, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = pipeline.process(frame, i / fps)
        for o in r.objects:
            if o.level >= Level.WARNING and prev.get(o.track_id, Level.NONE) < Level.WARNING:
                img = r.frame.copy()
                for other in r.objects:
                    x1, y1, x2, y2 = map(int, other.bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
                x1, y1, x2, y2 = map(int, o.bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                hx, hy = map(int, r.heading)
                cv2.drawMarker(img, (hx, hy), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
                e, c = o.estimate, o.course
                label = f"{video.stem[-6:]} f{i} {o.class_name} ttc {e.ttc_s:.2f} r {c.r:.2f} {'CRIT' if o.level == Level.CRITICAL else 'WARN'}"
                cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                name = f"{video.stem}_f{i:05d}.jpg"
                cv2.imwrite(str(out / name), img)
                thumbs.append(cv2.resize(img, (256, 192)))
                log.write(json.dumps({
                    "video": video.name, "frame": i, "image": name, "class": o.class_name, "level": int(o.level),
                    "box": [round(float(v), 1) for v in o.bbox], "ttc": e.ttc_s, "sigma": e.sigma_ttc_s,
                    "r": c.r, "r_contact": c.r_contact, "threshold": c.threshold, "heading": [float(v) for v in r.heading],
                    "in_path": bool(c.in_path), "width_m": c.width_m, "kappa": o.kappa,
                    "in_path_streak": pipeline._in_path_streak.get(o.track_id, 0),
                    "entry_streak": pipeline._entry_streak.get(o.track_id, 0),
                    "ego_valid": r.ego.valid, "ego_stationary": r.ego.stationary, "yaw_rate": float(r.ego.rvec[1] / r.dt_s),
                    "methods": {m.method: (1 / m.eta if m.eta > 0 else None) for m in o.measurements},
                }) + "\n")
                log.flush()
        prev = {o.track_id: o.level for o in r.objects}
        i += 1
    cap.release()
    return thumbs


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--fx", type=float, default=500.0)
    p.add_argument("--flow", choices=["dis", "lk", "vitis"], default="dis")
    p.add_argument("--out", type=Path, default=Path("outputs/warning_review"))
    p.add_argument("--yolo-weights", default="yolov9t.pt")
    args = p.parse_args(argv)
    from collision_avoidance.fcw.__main__ import make_detector

    args.out.mkdir(parents=True, exist_ok=True)
    detector = make_detector(args.yolo_weights, 0.5, "auto")
    thumbs = []
    with open(args.out / "warnings.jsonl", "w") as log:
        for video in sorted(v for v in args.videos.iterdir() if v.suffix.lower() in VIDEO_SUFFIXES):
            thumbs += review(video, args.fx, detector, args.out, log, flow=args.flow)
            print(f"{video.name}: {len(thumbs)} warnings so far", flush=True)
    if thumbs:
        cols = 6
        rows = [np.hstack(thumbs[k:k + cols] + [np.zeros_like(thumbs[0])] * (cols - len(thumbs[k:k + cols]))) for k in range(0, len(thumbs), cols)]
        cv2.imwrite(str(args.out / "contact_sheet.jpg"), np.vstack(rows))
    print(f"{len(thumbs)} warnings -> {args.out}")


if __name__ == "__main__":
    main()
