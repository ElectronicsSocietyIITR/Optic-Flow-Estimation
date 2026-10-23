"""TTC accuracy, warning lead time and variance calibration on EvTTC car-rear approaches.

    python -m ttc_esn.evttc --root data/evttc          # download once (upstream tool)
    python eval/evttc_fcw.py --data data/evttc

P1 exits: fused median relative TTC error <= 20 %; >= 90 % of approach events
warned while ground-truth TTC >= 2.0 s. Also prints FusionConfig.var_scale
values calibrated from the per-method errors.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from collision_avoidance.fcw import metrics
from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.measurement import METHODS
from collision_avoidance.fcw.pipeline import FcwConfig, FcwPipeline, make_flow, processing_size
from collision_avoidance.telemetry import start_run
from collision_avoidance.tracking import iou
from ttc_esn import evttc

PANEL_W, PANEL_H = 1920, 1200
MIN_TARGET_IOU = 0.3


def config_for_evttc() -> FcwConfig:
    """EvTTC's 1920x1200 left panel at its own aspect (512x320)."""
    w, h = processing_size(PANEL_W, PANEL_H)
    return FcwConfig(frame_width=w, frame_height=h, fx=evttc.LEFT_RGB_FX * w / PANEL_W, fy=evttc.LEFT_RGB_FX * h / PANEL_H)


def target_object(result, box_512):
    scored = [(iou(o.bbox, box_512), o) for o in result.objects]
    best = max(scored, key=lambda s: s[0], default=(0.0, None))
    return best[1] if best[0] >= MIN_TARGET_IOU else None


def evaluate_sequence(seq_dir: Path, detector, warmup: int = 20, cfg: FcwConfig | None = None, flow: str = "dis") -> dict:
    gt, annotations = evttc.load_gt(seq_dir), evttc.load_annotations(seq_dir)
    if not annotations:  # the target cannot be identified without at least one annotated box
        return {"name": seq_dir.name, "rows": [], "skipped": "no annotations"}
    fps = evttc.video_fps(seq_dir / "video.mp4")
    sync = evttc.estimate_sync_offset(annotations, gt, fps)
    first = math.ceil((gt.t.iloc[0] - sync.offset_s) * fps)
    last = math.floor((gt.t.iloc[-1] - sync.offset_s) * fps)
    cfg = cfg or config_for_evttc()
    scale = np.array([cfg.frame_width / PANEL_W, cfg.frame_height / PANEL_H] * 2)
    anchors = np.array(sorted(annotations))
    pipeline = FcwPipeline(cfg, detector, make_flow(flow))

    rows = []
    for index, panel in evttc.iter_left_frames(seq_dir / "video.mp4"):
        if index < first - warmup:
            continue
        if index > last:
            break
        result = pipeline.process(panel, index / fps)
        if index < first:
            continue
        nearest = int(anchors[np.abs(anchors - index).argmin()])
        obj = target_object(result, annotations[nearest] * scale)
        gt_ttc = float(np.interp(index / fps + sync.offset_s, gt.t, gt.ttc))
        rows.append({
            "gt_ttc": gt_ttc,
            "fused": obj.estimate.ttc_s if obj and obj.estimate and obj.estimate.ttc_s else np.nan,
            "level": obj.level if obj else Level.NONE,
            "measurements": [] if obj is None else [(m.method, m.eta, m.var) for m in obj.measurements],
        })
    return {"name": seq_dir.name, "rows": rows, "skipped": None}


def summarise(results: list[dict]) -> dict:
    gt = np.array([r["gt_ttc"] for res in results for r in res["rows"]])
    fused = np.array([r["fused"] for res in results for r in res["rows"]])
    summary = {"fused_median_rel_err": metrics.median_relative_error(fused, gt), "fused_coverage": metrics.coverage(fused)}
    records = []
    for method in METHODS:
        per = np.full(len(gt), np.nan)
        k = 0
        for res in results:
            for r in res["rows"]:
                for name, eta, var in r["measurements"]:
                    if name == method:
                        per[k] = 1.0 / eta if eta > 0 else np.nan
                        records.append((name, eta, var, 1.0 / r["gt_ttc"]))
                k += 1
        summary[f"{method}_median_rel_err"] = metrics.median_relative_error(per, gt)
        summary[f"{method}_coverage"] = metrics.coverage(per)
    summary["var_scale"] = metrics.variance_scale(records)
    events = [metrics.first_warning_ttc([r["level"] for r in res["rows"]], [r["gt_ttc"] for r in res["rows"]])
              for res in results if res["rows"] and min(r["gt_ttc"] for r in res["rows"]) < 2.0]
    summary["warned_by_2s"] = metrics.warned_in_time(events)
    summary["approach_events"] = len(events)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/evttc"))
    p.add_argument("--yolo-weights", default="yolov9t.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-scale", action="store_true", help="disable the scale-ratio search (ablation)")
    p.add_argument("--flow", choices=["dis", "lk", "vitis"], default="dis")
    args = p.parse_args(argv)
    from dataclasses import replace
    cfg = replace(config_for_evttc(), scale_enabled=not args.no_scale)

    from collision_avoidance.fcw.__main__ import make_detector

    detector = make_detector(args.yolo_weights, 0.5, args.device)
    seq_dirs = sorted(d for d in args.data.iterdir() if (d / "video.mp4").exists() and (d / "gt_ttc.csv").exists())
    run = start_run("fcw-eval-evttc", {"sequences": [d.name for d in seq_dirs]}, tags=["fcw", "eval", "evttc"])
    results = [evaluate_sequence(d, detector, cfg=cfg, flow=args.flow) for d in seq_dirs]
    for res in results:
        if res["skipped"]:
            print(f"{res['name']:24s} skipped: {res['skipped']}")
            continue
        one = summarise([res])
        print(f"{res['name']:24s} fused_err={one['fused_median_rel_err']:.3f} coverage={one['fused_coverage']:.2f}")
    summary = summarise(results)
    print(json.dumps(summary, indent=2))
    print("P1 exits: fused_median_rel_err <= 0.20, warned_by_2s >= 0.90")
    if run is not None:
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        run.finish()


if __name__ == "__main__":
    main()
