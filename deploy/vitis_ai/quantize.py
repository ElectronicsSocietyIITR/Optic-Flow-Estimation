"""Post-training INT8 quantization of PWC-Net and YOLOv9t with Vitis AI 3.5.

Runs inside the Vitis AI PyTorch docker (see README.md). Typical sequence:

    python quantize.py --model pwcnet --mode inspect --videos clips/*.mp4
    python quantize.py --model pwcnet --mode calib   --videos clips/*.mp4
    python quantize.py --model pwcnet --mode test    --videos clips/*.mp4   # evaluates + exports .xmodel

Both models use the pipeline resolution (288x512: the preprocess kernel's tensor for 16:9 sources). Calibration and evaluation
frames are sampled evenly from the given videos (evaluation uses a disjoint
set). PWC-Net consumes consecutive frame pairs, YOLOv9t single frames.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# The pipeline's PWC-Net module only needs torch/numpy, so it imports fine in the docker (py3.8).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collision_avoidance import pwcnet  # noqa: E402

HEIGHT, WIDTH = 288, 512  # fcw processingSize(16:9 source): the DPU input the preprocess kernel writes
TARGET = "DPUCZDX8G_ISA1_B4096"  # KV260; confirm with `xdputil query` on the board


def sample_frames(videos: list[str], count: int, offset: int, pairs: bool) -> list[tuple[np.ndarray, ...]]:
    """Evenly spaced (frame,) or (frame, next_frame) tuples resized to the pipeline size."""
    per_video = max(1, count // len(videos))
    samples = []
    for path in videos:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        starts = np.linspace(0, max(0, total - 2), per_video, dtype=int) + offset
        for start in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(start, total - 2)))
            frames = [cap.read()[1] for _ in range(2 if pairs else 1)]
            if all(f is not None for f in frames):
                samples.append(tuple(cv2.resize(f, (WIDTH, HEIGHT)) for f in frames))
        cap.release()
    return samples[:count]


def detection_f1(pred: np.ndarray, ref: np.ndarray, iou_thr: float = 0.5) -> float:
    """F1 of `pred` detections against `ref` (same class, IoU >= iou_thr, greedy by score)."""
    if len(pred) == 0 and len(ref) == 0:
        return 1.0
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    import torchvision
    ious = torchvision.ops.box_iou(torch.from_numpy(pred[:, :4]), torch.from_numpy(ref[:, :4])).numpy()
    ious[pred[:, 5][:, None] != ref[:, 5][None, :]] = 0.0
    matched = 0
    for i in np.argsort(-pred[:, 4]):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr:
            ious[:, j] = 0.0  # each reference detection matches at most once
            matched += 1
    return 2.0 * matched / (len(pred) + len(ref))


class ModelSpec:
    """Builds the float model, its input tensors and an output-error metric."""

    def __init__(self, name: str, yolo_weights: str):
        self.name = name
        if name == "pwcnet":
            model = pwcnet.load_ptlflow_weights(pwcnet.PWCNetDPU(dc=True), "things")
            self.model = pwcnet.set_leaky_slope(model, pwcnet.DPU_LEAKY_SLOPE)
            self.pairs = True
        else:
            import yolo_dpu  # needs ultralytics inside the docker
            self.model = yolo_dpu.load(yolo_weights)
            self.pairs = False

    def to_input(self, sample) -> torch.Tensor:
        if self.pairs:
            return pwcnet.preprocess(*sample, to_rgb=True)
        import yolo_dpu
        return yolo_dpu.preprocess(sample[0])

    def dummy_input(self) -> torch.Tensor:
        return torch.randn(1, 6 if self.pairs else 3, HEIGHT, WIDTH)

    def error(self, quant_out, float_out) -> dict[str, float]:
        if self.pairs:  # end-point error of the full-resolution flow
            float_flow = pwcnet.postprocess(float_out)
            epe = torch.linalg.vector_norm(pwcnet.postprocess(quant_out) - float_flow, dim=1)
            return {"epe_vs_float": float(epe.mean()), "float_flow_magnitude": float(torch.linalg.vector_norm(float_flow, dim=1).mean())}
        import yolo_dpu
        diffs = [float((q - f).abs().mean()) for q, f in zip(quant_out, float_out)]
        head = self.model.head
        q_det = yolo_dpu.nms(yolo_dpu.decode(quant_out, head))
        f_det = yolo_dpu.nms(yolo_dpu.decode(float_out, head))
        return {"raw_head_mae_vs_float": float(np.mean(diffs)), "detection_f1_vs_float": detection_f1(q_det, f_det)}


def start_tracking(args):
    if os.environ.get("COLLISION_WANDB", "1") == "0":
        return None
    try:
        import wandb
        return wandb.init(project=os.environ.get("COLLISION_WANDB_PROJECT", "kria-collision-avoidance"),
                          job_type=f"quantize-{args.mode}", config=vars(args), tags=["vitis-ai", args.model, args.mode])
    except Exception as exc:
        raise SystemExit(f"W&B unavailable ({exc}); install/login or set COLLISION_WANDB=0 to run untracked.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["pwcnet", "yolov9t"], required=True)
    p.add_argument("--mode", choices=["inspect", "calib", "test"], required=True)
    p.add_argument("--videos", nargs="+", required=True, help="dash-cam clips for calibration/evaluation")
    p.add_argument("--calib-samples", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=50)
    p.add_argument("--yolo-weights", default="../../yolov9t.pt")
    p.add_argument("--output-dir", default="quantized")
    p.add_argument("--fast-finetune", action="store_true",
                   help="AdaQuant-style weight-rounding optimisation during calib; reused automatically in test mode")
    args = p.parse_args()

    from pytorch_nndct.apis import Inspector, torch_quantizer

    spec = ModelSpec(args.model, args.yolo_weights)
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    if args.mode == "inspect":
        Inspector(TARGET).inspect(spec.model, (spec.dummy_input(),), device=device, output_dir=str(out_dir / "inspect"))
        print(f"Operator partition report written to {out_dir / 'inspect'}")
        return

    run = start_tracking(args)
    quantizer = torch_quantizer(args.mode, spec.model, (spec.dummy_input(),), output_dir=str(out_dir), device=device, target=TARGET)
    qmodel = quantizer.quant_model

    def forward_all(model, batch):
        with torch.no_grad():
            for sample in batch:
                model(spec.to_input(sample))

    if args.mode == "calib":
        samples = sample_frames(args.videos, args.calib_samples, offset=0, pairs=spec.pairs)
        if args.fast_finetune:
            quantizer.fast_finetune(forward_all, (qmodel, samples))
        with torch.no_grad():
            for i, sample in enumerate(samples, 1):
                qmodel(spec.to_input(sample))
                if i % 20 == 0:
                    print(f"calibration {i}/{len(samples)}")
        quantizer.export_quant_config()
        if run:
            run.summary.update({"calib_samples": len(samples)})
    else:
        if args.fast_finetune:
            quantizer.load_ft_param()
        # Disjoint from calibration frames (offset by a few frames into each clip segment).
        samples = sample_frames(args.videos, args.eval_samples, offset=7, pairs=spec.pairs)
        metrics = []
        with torch.no_grad():
            for sample in samples:
                x = spec.to_input(sample)
                metrics.append(spec.error(qmodel(x), spec.model(x)))
        summary = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        print("quantized vs float:", summary)
        quantizer.export_xmodel(output_dir=str(out_dir), deploy_check=False)
        if run:
            run.summary.update({**summary, "eval_samples": len(samples)})
    if run:
        run.finish()


if __name__ == "__main__":
    main()
