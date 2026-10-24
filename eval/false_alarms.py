"""False-alarm rate on normal driving (P1 exit: < 1 warning raise per 10 minutes).

    python eval/false_alarms.py --videos data/normal_driving --fx 500

Every video in the folder must be free of real near-collisions; each
NONE -> WARNING transition then counts as a false alarm.
"""

import argparse
from pathlib import Path

import cv2

from collision_avoidance.fcw.metrics import warning_raises
from collision_avoidance.fcw.pipeline import FcwPipeline, config_for_source, make_flow
from collision_avoidance.telemetry import start_run

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def count_raises(video: Path, fx: float, detector, flow: str = "dis") -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cfg = config_for_source(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fx)
    pipeline, levels = FcwPipeline(cfg, detector, make_flow(flow)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        levels.append(pipeline.process(frame, len(levels) / fps).level)
    cap.release()
    return warning_raises(levels), len(levels) / fps


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--fx", type=float, default=500.0)
    p.add_argument("--flow", default="dis", help="dis | vitis | lk | lk:levels=N,iters=N,win=N,scale=N")
    p.add_argument("--yolo-weights", default="yolov9t.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from collision_avoidance.fcw.__main__ import make_detector

    detector = make_detector(args.yolo_weights, 0.5, args.device)
    videos = sorted(v for v in args.videos.iterdir() if v.suffix.lower() in VIDEO_SUFFIXES)
    run = start_run("fcw-eval-false-alarms", {"videos": [v.name for v in videos], "fx": args.fx}, tags=["fcw", "eval"])
    total_raises, total_s = 0, 0.0
    for video in videos:
        raises, seconds = count_raises(video, args.fx, detector, flow=args.flow)
        total_raises, total_s = total_raises + raises, total_s + seconds
        print(f"{video.name:40s} {seconds:7.1f} s  raises={raises}")
    rate = total_raises / (total_s / 600.0) if total_s else float("nan")
    print(f"total {total_s / 60:.1f} min, {total_raises} raises -> {rate:.2f} per 10 min  (P1 exit: < 1)")
    if run is not None:
        run.summary.update({"false_alarms_per_10min": rate, "minutes": total_s / 60})
        run.finish()


if __name__ == "__main__":
    main()
