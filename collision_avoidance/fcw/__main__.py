"""Run the forward-collision-warning pipeline on a video file or webcam.

    python -m collision_avoidance.fcw --source dashcam.mp4 --fx 500
    python -m collision_avoidance.fcw --source 0 --output run.mp4
    python -m collision_avoidance.fcw --source clip.mp4 --no-display --max-frames 300

--fx is the focal length in pixels of the processing frame, which is 512 px
wide with the source's aspect ratio (16:9 -> 512x288). For a camera with
horizontal field of view F: fx = 256 / tan(F / 2).
Keys (display mode): ESC quit, P pause.
"""

import argparse
import time

import cv2
import numpy as np

from .collision import Level
from .metrics import warning_raises
from dataclasses import replace

from .pipeline import FcwPipeline, config_for_source, make_flow
from .render import draw


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path or webcam index")
    p.add_argument("--fx", type=float, default=500.0, help="focal length (px) at the 512 px wide processing frame")
    p.add_argument("--fy", type=float, help="vertical focal length (px) if pixels are not square")
    p.add_argument("--flow", default="dis", help="dis: fast PC stand-in; lk[:levels=N,iters=N,win=N,scale=N]: the board's algorithm; vitis: its kernel")
    p.add_argument("--yolo-weights", default="yolov9t.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", help="write the annotated video here")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args(argv)


def make_detector(weights: str, conf: float, device: str):
    from ..detection import Detector  # imports ultralytics and torch
    from ..flow import resolve_device

    return Detector(weights, conf, resolve_device(device))


def main(argv=None) -> None:
    args = parse_args(argv)
    live = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if live else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source '{args.source}'")
    cfg = config_for_source(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), args.fx)
    if args.fy is not None:
        cfg = replace(cfg, fy=args.fy)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = FcwPipeline(cfg, make_detector(args.yolo_weights, args.conf, args.device), make_flow(args.flow))
    writer, levels, totals = None, [], []
    start = time.perf_counter()
    try:
        while not args.max_frames or len(levels) < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter() - start if live else len(levels) / fps
            result = pipeline.process(frame, t)
            levels.append(result.level)
            totals.append(result.timings_ms["total"])
            if args.no_display and not args.output:
                continue
            image = draw(result, cfg.fx)
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, image.shape[1::-1])
                writer.write(image)
            if not args.no_display:
                cv2.imshow("Forward collision warning", image)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("p"):
                    cv2.waitKey(0)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"frames {len(levels)} | mean {np.mean(totals[1:]) if len(totals) > 1 else 0:.1f} ms/frame | "
          f"warnings raised {warning_raises(levels)} | critical frames {sum(lv == Level.CRITICAL for lv in levels)}")


if __name__ == "__main__":
    main()
