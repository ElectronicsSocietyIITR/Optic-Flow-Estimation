// Closing-speed ratio (oncoming-traffic test). Port of collision_avoidance/fcw/closing.py.
//
// C = TTC_road(y) * (y - y_FOE) is constant on a flat road (our speed over camera height,
// in image terms). Measured on the road band, it gives the static TTC at any contact row,
// and kappa = TTC_static / TTC_object = closing speed / our speed (~1 parked, < 1 going our
// way, well above 1 oncoming).
#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "fcw/types.hpp"

namespace fcw {

struct ClosingConfig {
  bool enabled = true;
  std::pair<double, double> band{0.25, 0.75};  // road rows, as fractions of the image below the FOE row
  int grid_step_x = 8;  // within 5 % of the 4 x 2 grid, 4x cheaper
  int grid_step_y = 4;
  int min_points = 30;
  double min_radius_px = 5.0;
  double min_outward_px = 0.05;
  double min_contact_px = 3.0;
  double oncoming_kappa = 1.8;
  int history = 8;
};

std::optional<double> roadConstant(const cv::Mat& flow, cv::Point2d foe, const cv::Matx33d& R, const cv::Matx33d& K,
                                   const std::vector<Box>& exclude, const ClosingConfig& cfg = {});
std::optional<double> staticTtc(double c, const Box& box, cv::Point2d foe, const ClosingConfig& cfg = {});
double median(std::vector<double> values);  // numpy semantics: mean of the middle two for even sizes

class ClosingTracker {
 public:
  explicit ClosingTracker(ClosingConfig cfg = {}) : cfg_(cfg) {}
  std::optional<double> updateC(std::optional<double> c);
  std::optional<double> update(int track_id, std::optional<double> kappa);
  void retain(const std::vector<int>& track_ids);

 private:
  ClosingConfig cfg_;
  std::deque<double> c_history_;
  std::map<int, std::deque<double>> samples_;
};

}  // namespace fcw
