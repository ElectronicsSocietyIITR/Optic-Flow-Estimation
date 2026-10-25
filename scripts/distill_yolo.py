"""Distil the DPU's Hardswish YOLOv9t from the SiLU original on dash-cam frames.

    python scripts/distill_yolo.py --images "data/kitti/*/*/image_02/data"
    python scripts/distill_yolo.py --videos "clip*.mp4" --every 300 --epochs 1 --dataset /tmp/d    # smoke test

1. Frames: every `--every`-th frame of each clip (plus optional image folders); whole clips
   are held out for validation. The SiLU teacher's detections on them are kept as YOLO labels
   for an mAP-against-the-teacher check (collision_avoidance.yolo_distill).
2. Student: the same weights with SiLU -> Hardswish (the DPU has no SiLU), which alone
   drops agreement with the teacher sharply.
3. Distil: the student learns the teacher's raw Detect-head maps (class scores as soft
   targets, DFL box distributions) at the DPU's 288x512 input. Training on the teacher's
   thresholded boxes with Ultralytics' trainer was tried first and made agreement worse
   (mAP50-95 0.46 -> 0.44), like the earlier COCO fine-tunes in deploy/vitis_ai/README.md.
The epoch with the best detection F1 against the teacher on the held-out clips is written
to --out; quantise it with `quantize.py --model yolov9t --yolo-weights ../../<out>`.
"""

import argparse
import copy
import glob
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "deploy" / "vitis_ai"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

import yolo_dpu  # noqa: E402  (the deploy NMS)
from collision_avoidance import yolo_distill, yolo_pruning  # noqa: E402
from collision_avoidance.telemetry import start_run  # noqa: E402
from prune_yolo import evaluate  # noqa: E402
from quantize import HEIGHT, WIDTH, detection_f1  # noqa: E402  (the DPU input size and the deploy metric)


def write_split(teacher: YOLO, frames, split_dir: Path, args) -> int:
    """Run the teacher on (name, frame) pairs and write images/<split> and labels/<split>."""
    img_dir = split_dir.parent.parent / "images" / split_dir.name
    img_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    batch: list[tuple[str, object]] = []

    def flush():
        nonlocal count
        results = teacher.predict([f for _, f in batch], imgsz=args.imgsz, conf=args.conf, iou=0.7,
                                  device=args.device, verbose=False)
        for (name, frame), r in zip(batch, results):
            cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = yolo_distill.label_lines(r.boxes.xywhn.cpu().numpy(), r.boxes.cls.cpu().numpy())
            (split_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            count += 1
        batch.clear()

    for item in frames:
        batch.append(item)
        if len(batch) == 32:
            flush()
    if batch:
        flush()
    return count


def video_frames(clips, every):
    for clip in clips:
        for index, frame in yolo_distill.sample_video(clip, every):
            yield f"{Path(clip).stem}_{index:06d}", frame


def image_frames(dirs, every):
    for d in dirs:
        tag = "_".join(Path(d).parts[-4:-2])  # e.g. 2011_09_26_drive_0005_sync_image_02
        for i, path in enumerate(sorted(Path(d).glob("*.png"))):
            if i % every == 0:
                yield f"{tag}_{path.stem}", cv2.imread(str(path))


def build_dataset(args, teacher: YOLO) -> Path:
    data_dir = Path(args.dataset)
    data_yaml = data_dir / "data.yaml"
    if data_yaml.exists() and not args.rebuild:
        print(f"reusing {data_dir}")
        return data_yaml
    shutil.rmtree(data_dir, ignore_errors=True)
    clips = sorted(c for pattern in args.videos for c in glob.glob(pattern))
    train_clips, val_clips = yolo_distill.split_clips(clips, args.val_fraction)
    image_dirs = sorted(d for pattern in args.images for d in glob.glob(pattern))
    n_train = write_split(teacher, video_frames(train_clips, args.every), data_dir / "labels" / "train", args)
    n_train += write_split(teacher, image_frames(image_dirs, args.image_every), data_dir / "labels" / "train", args)
    n_val = write_split(teacher, video_frames(val_clips, args.every), data_dir / "labels" / "val", args)
    print(f"pseudo-labels: {n_train} train frames ({len(train_clips)} clips + {len(image_dirs)} image dirs), "
          f"{n_val} val frames (held-out clips: {[Path(c).name for c in val_clips]})")
    data_yaml.write_text(yaml.safe_dump({
        "path": str(data_dir.resolve()), "train": "images/train", "val": "images/val",
        "names": {int(k): v for k, v in teacher.names.items()},
    }))
    return data_yaml


def to_dpu_input(frame: np.ndarray, train: bool, rng: random.Random) -> np.ndarray:
    """BGR frame -> RGB 288x512 float, letterboxed (16:9 dash-cam frames fill it exactly).
    In training: a random 75-100 % crop, horizontal flip and brightness/contrast jitter."""
    if train:
        h, w = frame.shape[:2]
        s = rng.uniform(0.75, 1.0)
        ch, cw = int(h * s), int(w * s)
        y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
        frame = frame[y0:y0 + ch, x0:x0 + cw]
        if rng.random() < 0.5:
            frame = frame[:, ::-1]
    h, w = frame.shape[:2]
    scale = min(WIDTH / w, HEIGHT / h)
    resized = cv2.resize(np.ascontiguousarray(frame), (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((HEIGHT, WIDTH, 3), 114, np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    x = canvas[..., ::-1].astype(np.float32) / 255.0
    if train:
        x = np.clip((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def batches(paths, batch, train, rng):
    order = list(paths)
    if train:
        rng.shuffle(order)
    for i in range(0, len(order), batch):
        yield torch.from_numpy(np.stack([to_dpu_input(cv2.imread(str(p)), train, rng) for p in order[i:i + batch]]))


@torch.no_grad()
def f1_vs_teacher(student, teacher, paths, device, batch=32) -> float:
    """Mean detection F1 of the student against the teacher at the DPU input (as quantize.py scores INT8)."""
    student.eval()
    scores = []
    for x in batches(paths, batch, False, random.Random(0)):
        x = x.to(device)
        s_pred, t_pred = student(x)[0].float().cpu(), teacher(x)[0].float().cpu()
        scores += [detection_f1(yolo_dpu.nms(s_pred[i:i + 1]), yolo_dpu.nms(t_pred[i:i + 1])) for i in range(len(x))]
    return float(np.mean(scores))


def save_checkpoint(model, path: Path):
    """An Ultralytics-loadable checkpoint (YOLO(path) and deploy/vitis_ai/yolo_dpu.load)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": copy.deepcopy(model).half(), "ema": None, "epoch": -1, "train_args": {}}, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(ROOT / "yolov9t.pt"), help="SiLU teacher, also the student's initial weights")
    p.add_argument("--videos", nargs="+", default=[str(ROOT / "data" / "normal_driving" / "*.MP4")])
    p.add_argument("--images", nargs="*", default=[], help="extra training image folders (glob), e.g. KITTI image_02/data")
    p.add_argument("--every", type=int, default=15, help="take every n-th video frame")
    p.add_argument("--image-every", type=int, default=2, help="take every n-th image of --images")
    p.add_argument("--val-fraction", type=float, default=0.2, help="fraction of the clips held out")
    p.add_argument("--conf", type=float, default=0.25, help="teacher confidence threshold for a pseudo-label")
    p.add_argument("--dataset", default=str(ROOT / "outputs" / "distill" / "data"))
    p.add_argument("--rebuild", action="store_true", help="regenerate the frames and pseudo-labels")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640, help="pseudo-labelling and mAP size")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "outputs" / "models" / "yolov9t_hswish_distilled.pt"))
    args = p.parse_args(argv)
    settings.update({"wandb": False})  # we log to our own run below
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    run = start_run("distill", vars(args), tags=["yolov9t", "hardswish", "distillation"])
    log = (lambda d: run.log(d)) if run is not None else (lambda d: None)

    data_yaml = build_dataset(args, YOLO(args.weights))
    data_dir = data_yaml.parent
    train_paths = sorted((data_dir / "images" / "train").glob("*.jpg"))
    val_paths = sorted((data_dir / "images" / "val").glob("*.jpg"))
    device = torch.device(args.device)

    teacher = YOLO(args.weights).model.float().to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = yolo_pruning.replace_silu(copy.deepcopy(teacher)).to(device)
    for q in student.parameters():
        q.requires_grad_(True)
    head = student.model[-1]

    # mAP against the pseudo-labels at the DPU's 512 width (rectangular batches: 512x288), where the student runs.
    before_map = evaluate(_as_yolo(args.weights, student), str(data_yaml), WIDTH, 0 if device.type == "cuda" else "cpu", 16)
    best_f1 = f1_vs_teacher(student, teacher, val_paths, device)
    print(f"hardswish vs teacher, before: F1 {best_f1:.3f} (288x512), {before_map} (mAP vs pseudo-labels)")
    log({"epoch": 0, "val/f1_vs_teacher": best_f1})
    out = Path(args.out)
    save_checkpoint(student, out)

    steps = args.epochs * ((len(train_paths) + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    for epoch in range(1, args.epochs + 1):
        student.train()
        losses = []
        for x in batches(train_paths, args.batch, True, rng):
            x = x.to(device)
            with torch.no_grad():
                t_maps = teacher(x)[1]
            loss = yolo_distill.distillation_loss(student(x), t_maps, nc=head.nc, reg_max=head.reg_max)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        f1 = f1_vs_teacher(student, teacher, val_paths, device)
        print(f"epoch {epoch}/{args.epochs}: loss {np.mean(losses):.4f}, F1 vs teacher {f1:.3f}")
        log({"epoch": epoch, "train/kd_loss": float(np.mean(losses)), "val/f1_vs_teacher": f1})
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(student, out)

    after_map = evaluate(str(out), str(data_yaml), WIDTH, 0 if device.type == "cuda" else "cpu", 16)
    summary = {"before/map50_95": before_map["map50_95"], "after/map50_95": after_map["map50_95"],
               "after/f1_vs_teacher": best_f1, "output": str(out)}
    print("hardswish vs teacher, best:", summary)
    if run is not None:
        run.summary.update(summary)
        run.finish()


def _as_yolo(weights: str, model) -> YOLO:
    yolo = YOLO(weights)
    yolo.model = copy.deepcopy(model).float().cpu()
    return yolo


if __name__ == "__main__":
    main()
