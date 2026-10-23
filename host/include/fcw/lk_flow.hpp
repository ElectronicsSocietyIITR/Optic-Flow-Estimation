// Dense pyramidal Lucas-Kanade with the Vitis Vision kernel's parameters: the board's flow,
// in software. Plain loops on purpose (also the CPU-only baseline, host/tools/cpu_baseline.cpp).
#pragma once

#include <opencv2/core.hpp>

#include "fcw/pipeline.hpp"

namespace fcw {

struct LkParams {
  int levels = 5;      // pyramid levels
  int iterations = 5;  // Gauss-Newton steps per level
  int win = 11;        // window side
};

// prev8, curr8: CV_8U gray. Returns CV_32FC2 flow (prev -> curr) in px.
cv::Mat denseLk(const cv::Mat& prev8, const cv::Mat& curr8, const LkParams& p = {});
FlowSource lkFlow(LkParams p = {});

}  // namespace fcw
