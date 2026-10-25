"""YOLOv9t split for DPU deployment.

`YoloDpuGraph` runs the Ultralytics DetectionModel up to and including the
Detect-head convolutions and returns the three raw per-stride maps; that is the
part the DPU can execute. Box decoding (DFL softmax, anchors, sigmoid) and NMS
run on the CPU via `decode` + `nms`.

Input is the pipeline's resized frame (288x512 for 16:9 sources, both multiples of the 32 px
stride), RGB in [0, 1] -- no letterboxing, so boxes are in frame pixels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO


class YoloDpuGraph(nn.Module):
    def __init__(self, detection_model: nn.Module):
        super().__init__()
        self.layers = detection_model.model
        self.save = set(detection_model.save)

    def forward(self, x):
        outputs = []
        for m in self.layers[:-1]:
            if m.f != -1:
                x = outputs[m.f] if isinstance(m.f, int) else [x if j == -1 else outputs[j] for j in m.f]
            x = m(x)
            outputs.append(x if m.i in self.save else None)
        head = self.layers[-1]
        feats = [x if j == -1 else outputs[j] for j in head.f]
        return tuple(torch.cat((head.cv2[i](f), head.cv3[i](f)), 1) for i, f in enumerate(feats))

    @property
    def head(self) -> nn.Module:
        return self.layers[-1]


def load(weights: str = "yolov9t.pt") -> YoloDpuGraph:
    model = YOLO(weights).model.float().fuse().eval()  # fuse Conv+BN before quantization
    return YoloDpuGraph(model).eval()


def preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


@torch.no_grad()
def decode(raw: tuple[torch.Tensor, ...], head: nn.Module) -> torch.Tensor:
    """Raw head maps -> [B, 4 + nc, N] with xywh boxes (pixels) and class scores."""
    return head._inference(list(raw))


def nms(pred: torch.Tensor, conf: float = 0.25, iou: float = 0.45) -> np.ndarray:
    """One image's decoded predictions -> rows of (x1, y1, x2, y2, score, class)."""
    p = pred[0].T
    scores, classes = p[:, 4:].max(dim=1)
    keep = scores > conf
    p, scores, classes = p[keep], scores[keep], classes[keep]
    xy, wh = p[:, :2], p[:, 2:4]
    boxes = torch.cat((xy - wh / 2, xy + wh / 2), dim=1)
    idx = torchvision.ops.batched_nms(boxes, scores, classes, iou)
    return torch.cat((boxes[idx], scores[idx, None], classes[idx, None].float()), dim=1).numpy()
