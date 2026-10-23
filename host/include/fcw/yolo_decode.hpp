// YOLOv9t head decode for the DPU's raw outputs, six collision classes only.
// Checked against ultralytics' decode (tools/export_golden.py, exporter "yolo").
#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "fcw/types.hpp"

namespace fcw {

struct YoloDecodeConfig {
  double conf = 0.5;                            // same as the PC detector
  double iou = 0.7;                             // ultralytics' default NMS overlap, as the PC detector uses
  std::vector<int> classes{0, 1, 2, 3, 5, 7};   // COCO: person, bicycle, car, motorcycle, bus, truck
  int reg_max = 16;                             // DFL bins per box side
};

// raw_heads: one H x W map per stride (finest first), CV_32F with 4*reg_max + 80 channels (NHWC),
// as the DPU returns them after dequantisation. Boxes in input pixels, sorted by score.
// Only the six class logits are tested (in logit space, no sigmoid) before any box decoding.
std::vector<Detection> decodeYolo(const std::vector<cv::Mat>& raw_heads, cv::Size input, const YoloDecodeConfig& cfg = {});

}  // namespace fcw
