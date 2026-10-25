#include "fcw/moment_engine.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

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

std::vector<EnginePass> planEnginePasses(const std::vector<EngineBox>& boxes, int lanes) {
  std::vector<EnginePass> passes;
  if (boxes.empty()) return passes;
  std::set<int> levels{0};
  for (const auto& b : boxes) levels.insert(b.level);
  for (const int L : levels) {
    std::vector<int> carried;
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i)
      if (L == 0 || boxes[i].level == L) carried.push_back(i);
    for (size_t k = 0; k < carried.size(); k += static_cast<size_t>(lanes))
      passes.push_back({L, std::vector<int>(carried.begin() + k, carried.begin() + std::min(carried.size(), k + lanes))});
  }
  return passes;
}

}  // namespace fcw
