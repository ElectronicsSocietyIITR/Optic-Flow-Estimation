"""Pseudo-label datasets for distilling the DPU's Hardswish YOLOv9t from the SiLU original.

The DPU cannot run SiLU, and swapping in Hardswish without retraining loses much of the
detections (deploy/vitis_ai/README.md). The SiLU model's own detections on dash-cam frames
serve as labels, so the Hardswish copy learns to match it where the system runs, with no
manual annotation. Used by scripts/distill_yolo.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def label_lines(xywhn: np.ndarray, classes: np.ndarray) -> list[str]:
    """YOLO label rows ("cls cx cy w h", normalised) for one image, clipped to [0, 1]."""
    boxes = np.clip(np.asarray(xywhn, dtype=float).reshape(-1, 4), 0.0, 1.0)
    return [f"{int(c)} " + " ".join(f"{v:.6f}" for v in b) for b, c in zip(boxes, np.asarray(classes).reshape(-1))]


def split_clips(clips: list[str], val_fraction: float = 0.2) -> tuple[list[str], list[str]]:
    """Hold out whole clips (the last ones in sorted order), so validation frames are unseen scenes."""
    clips = sorted(clips)
    n_val = max(1, round(len(clips) * val_fraction))
    return clips[:-n_val], clips[-n_val:]


def distillation_loss(student: list[torch.Tensor], teacher: list[torch.Tensor], nc: int = 80, reg_max: int = 16) -> torch.Tensor:
    """Match the teacher's raw Detect-head maps, level by level ([B, 4*reg_max + nc, H, W] each).

    Classes: Bernoulli KL to the teacher's sigmoid scores, summed over classes, so background
    (low scores everywhere) is learned as well as objects. Boxes: KL between the DFL bin
    distributions of the four sides, weighted by the teacher's top class score so that only
    cells the teacher believes in pull the box regression. Zero when the maps are equal.
    """
    total = 0.0
    for s, t in zip(student, teacher):
        t = t.detach()
        s_box, s_cls = s[:, : 4 * reg_max], s[:, 4 * reg_max:]
        t_box, t_cls = t[:, : 4 * reg_max], t[:, 4 * reg_max:]
        p = torch.sigmoid(t_cls)
        # BCE(s, p) - BCE(t, p) is the Bernoulli KL(p || sigmoid(s)); exactly 0 when s == t.
        cls_kl = (F.binary_cross_entropy_with_logits(s_cls, p, reduction="none")
                  - F.binary_cross_entropy_with_logits(t_cls, p, reduction="none"))
        cls_term = cls_kl.sum(1).mean()
        b, _, h, w = s.shape
        log_s = F.log_softmax(s_box.reshape(b, 4, reg_max, h, w), dim=2)
        log_t = F.log_softmax(t_box.reshape(b, 4, reg_max, h, w), dim=2)
        box_kl = (log_t.exp() * (log_t - log_s)).sum(2).mean(1)  # [B, H, W]
        weight = p.max(1).values
        box_term = (weight * box_kl).sum() / weight.sum().clamp(min=1e-6)
        total = total + cls_term + box_term
    return total / len(student)


def sample_video(path: str | Path, every: int, max_width: int = 1280) -> Iterator[tuple[int, np.ndarray]]:
    """Every `every`-th frame of a video, downscaled to at most `max_width` pixels wide."""
    cap = cv2.VideoCapture(str(path))
    index = 0
    while True:
        if not cap.grab():
            break
        if index % every == 0:
            ok, frame = cap.retrieve()
            if ok:
                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                yield index, frame
        index += 1
    cap.release()
