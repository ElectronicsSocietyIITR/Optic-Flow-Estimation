// The per-box sums of the Horn TTC engine (hls/ttc_engine): Horn sums at each box's pyramid level
// and flow moments (masked by last frame's fit when one is given). On the board: one engine call per
// pyramid level in use (Option A, after the tracker); the level-0 pass carries every box's flow moments
// and the level-0 Horn boxes, the level-1/2 passes only their Horn boxes. On the PC,
// SoftwareMomentEngine computes the same numbers (bit-exact with the kernel's C-simulation).
#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

#include "fcw/divergence.hpp"
#include "fcw/horn.hpp"

namespace fcw {

struct EngineBox {
  Box box;                  // level-0 pixels
  int level = 0;            // pyramid level for the Horn sums
  std::optional<FitQ> fit;  // last frame's affine flow fit: moments over its inliers only
};

struct BoxSums {
  std::optional<HornSums> horn;
  std::optional<FixedMoments> flow;  // from the level-0 pass, for every box
};

class MomentEngine {
 public:
  virtual ~MomentEngine() = default;
  // prev, curr: CV_8U gray at level 0; flow_q: CV_16SC2 in 1/64 px. frame_index: curr's frame (the board
  // engine reads the pyramid slots of frame_index - 1 and frame_index from DDR; software ignores it).
  virtual std::vector<BoxSums> run(const cv::Mat& prev, const cv::Mat& curr, const cv::Mat& flow_q, const std::vector<EngineBox>& boxes,
                                   cv::Point2d principal, int frame_index) = 0;
};

class SoftwareMomentEngine : public MomentEngine {
 public:
  SoftwareMomentEngine(HornConfig horn, DivergenceConfig divergence) : horn_(horn), div_(divergence) {}
  std::vector<BoxSums> run(const cv::Mat& prev, const cv::Mat& curr, const cv::Mat& flow_q, const std::vector<EngineBox>& boxes,
                           cv::Point2d principal, int frame_index = 0) override;

 private:
  HornConfig horn_;
  DivergenceConfig div_;
};

}  // namespace fcw
