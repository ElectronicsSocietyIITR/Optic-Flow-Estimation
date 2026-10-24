#include "fcw/moment_engine.hpp"

#include <cmath>
#include <map>

namespace fcw {

std::vector<BoxSums> SoftwareMomentEngine::run(const cv::Mat& prev, const cv::Mat& curr, const cv::Mat& flow_q,
                                               const std::vector<EngineBox>& boxes, cv::Point2d pp, int /*frame_index*/) {
  std::vector<BoxSums> out(boxes.size());
  std::map<int, std::pair<cv::Mat, cv::Mat>> levels;  // one pass per level in use, as the engine runs
  const cv::Point ppi(static_cast<int>(std::nearbyint(pp.x)), static_cast<int>(std::nearbyint(pp.y)));
  const int inlier_q = static_cast<int>(std::nearbyint(div_.inlier_px * kFlowScale));
  for (size_t i = 0; i < boxes.size(); ++i) {
    const EngineBox& e = boxes[i];
    auto it = levels.find(e.level);
    if (it == levels.end()) it = levels.emplace(e.level, std::make_pair(downsample(prev, e.level), downsample(curr, e.level))).first;
    const double f = std::pow(2.0, -e.level);
    out[i].horn = hornSums(it->second.first, it->second.second, {e.box.x1 * f, e.box.y1 * f, e.box.x2 * f, e.box.y2 * f},
                           {pp.x * f, pp.y * f}, horn_);
    // Flow moments come from the level-0 pass for every box (the flow is at level 0), whatever its Horn level.
    cv::Mat mask;
    if (e.fit) {
      if (const auto r = flowRect(e.box, div_.shrink, flow_q.cols, flow_q.rows)) {
        mask = cv::Mat::zeros(flow_q.size(), CV_8U);
        predictedMaskFixed(flow_q, *r, ppi, *e.fit, inlier_q).copyTo(mask(*r));
      }
    }
    out[i].flow = flowMomentsFixed(flow_q, e.box, pp, div_.shrink, mask);
  }
  return out;
}

}  // namespace fcw
