// Annotation layer for the board's display (spec section 10, issue 15): boxes, TTC labels and the warning
// banner drawn into an ARGB overlay plane that the DisplayPort hardware blends over the camera frame.
// Builds anywhere (OpenCV only), so it is tested on the PC.
#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "fcw/pipeline.hpp"

namespace fcw {

class OverlayPainter {
 public:
  // overlay: CV_8UC4 (B, G, R, A), alpha 0 = transparent. Objects are in processing-frame pixels and are
  // scaled to `frame` (the camera frame, shown at the overlay's top-left). Each call first clears only what
  // the previous call on this overlay drew, so use one painter per overlay buffer.
  void paint(cv::Mat& overlay, const FcwFrame& f, cv::Size processing, cv::Size frame);

 private:
  std::vector<cv::Rect> drawn_;
};

}  // namespace fcw
