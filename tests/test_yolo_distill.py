"""Pseudo-labels for distilling the Hardswish YOLOv9t from the SiLU one (scripts/distill_yolo.py)."""

import numpy as np
import torch

from collision_avoidance import yolo_distill

NC, REG = 80, 16


def _maps(seed: int, scale: float = 3.0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 4 * REG + NC, h, w, generator=g) * scale for h, w in ((36, 64), (18, 32), (9, 16))]


def test_label_lines_are_yolo_format():
    xywhn = np.array([[0.5, 0.25, 0.1, 0.2], [0.123456, 0.9, 0.05, 0.04]])
    lines = yolo_distill.label_lines(xywhn, np.array([2.0, 0.0]))
    assert lines == ["2 0.500000 0.250000 0.100000 0.200000", "0 0.123456 0.900000 0.050000 0.040000"]


def test_label_lines_clip_boxes_to_the_image():
    lines = yolo_distill.label_lines(np.array([[1.01, -0.01, 0.3, 1.2]]), np.array([7]))
    assert lines == ["7 1.000000 0.000000 0.300000 1.000000"]


def test_no_detections_give_an_empty_label_file():
    assert yolo_distill.label_lines(np.zeros((0, 4)), np.zeros(0)) == []


def test_distillation_loss_is_zero_when_student_matches_teacher():
    teacher = _maps(0)
    loss = yolo_distill.distillation_loss(teacher, teacher, nc=NC, reg_max=REG)
    # BCE against soft targets bottoms out at the targets' entropy; the loss subtracts it.
    assert abs(float(loss)) < 1e-5


def test_distillation_loss_grows_with_the_mismatch():
    teacher = _maps(0)
    near = [t + 0.1 * n for t, n in zip(teacher, _maps(1, 1.0))]
    far = [t + 1.0 * n for t, n in zip(teacher, _maps(1, 1.0))]
    l_near = float(yolo_distill.distillation_loss(near, teacher, nc=NC, reg_max=REG))
    l_far = float(yolo_distill.distillation_loss(far, teacher, nc=NC, reg_max=REG))
    assert 0 < l_near < l_far


def test_distillation_loss_trains_the_student_towards_the_teacher():
    teacher = _maps(0)
    student = [torch.zeros_like(t, requires_grad=True) for t in teacher]
    opt = torch.optim.Adam(student, lr=0.1)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = yolo_distill.distillation_loss(student, teacher, nc=NC, reg_max=REG)
        first = float(loss) if first is None else first
        loss.backward()
        opt.step()
    assert float(loss) < 0.2 * first


def test_split_holds_out_whole_clips():
    clips = [f"clip_{i}.MP4" for i in range(10)]
    train, val = yolo_distill.split_clips(clips, val_fraction=0.2)
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(clips)
    assert val == ["clip_8.MP4", "clip_9.MP4"]  # deterministic: the last clips in sorted order
