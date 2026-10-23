// Course check and warning state machine. Port of collision_avoidance/fcw/collision.py.
#pragma once

#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "fcw/types.hpp"

namespace fcw {

struct CourseConfig {
  double ego_width_m = 1.8;
  double margin_m = 0.3;
  std::map<std::string, double> class_width_m{{"car", 1.8}, {"truck", 2.5}, {"bus", 2.5}, {"motorcycle", 0.8}, {"bicycle", 0.6}, {"person", 0.5}};
  std::map<std::string, double> class_height_m{{"car", 1.5}, {"truck", 3.0}, {"bus", 3.0}, {"motorcycle", 1.5}, {"bicycle", 1.7}, {"person", 1.7}};
  double history_s = 0.5;
  double turn_yaw_rate_rps = 3.0 * 3.14159265358979323846 / 180.0;
  double turn_widen = 1.5;
  double min_overlap = 0.5;  // of the narrower of our width and the object's
  int in_path_frames = 3;    // consecutive in-path frames before the lead may warn
  double max_entry_m = 4.0;  // predicted entry: only objects within this lateral distance
  int entry_frames = 6;      // consecutive predicted-entry frames before it may warn
  double min_entry_speed_mps = 0.3;  // lateral speed towards our path; parked-object jitter stays below
};

struct CourseResult {
  bool on_course = false;
  double r = 0;
  std::optional<double> r_contact;
  double threshold = 0;
  bool in_path = false;               // now: overlaps our width enough to be the lead vehicle
  std::optional<double> width_m;      // class width prior used
  std::optional<double> entry_speed_mps;  // outer edge's speed towards our path (see collision.py)
};

class CourseChecker {
 public:
  explicit CourseChecker(CourseConfig cfg = {}) : cfg_(std::move(cfg)) {}
  CourseResult update(int track_id, double t, const Box& box, const std::string& cls, double heading_x, double horizon_y,
                      std::optional<double> ttc_s, double yaw_rate_rps, cv::Size image = {},
                      std::optional<cv::Matx33d> rotation_h = std::nullopt, std::optional<double> ref_x = std::nullopt);
  void retain(const std::vector<int>& track_ids);
  // Creates these tracks' histories now, so update() from parallel workers never inserts.
  void prepare(const std::vector<int>& track_ids);
  const CourseConfig& config() const { return cfg_; }

 private:
  CourseConfig cfg_;
  struct Edge { double t, q, side; };
  std::optional<double> entrySpeed(int track_id, double t, const Box& box, double ref_x, const std::string& cls, cv::Size image,
                                   const std::optional<cv::Matx33d>& rotation_h);
  std::map<int, std::deque<std::pair<double, double>>> history_;
  std::map<int, std::deque<Edge>> edges_;
  std::map<int, double> rot_px_;  // accumulated rotation-only shift of the outer edge
};

enum class Level { None = 0, Warning = 1, Critical = 2 };

struct WarningConfig {
  double warn_ttc_s = 2.7;
  int warn_frames = 3;
  double warn_clear_ttc_s = 3.2;
  double critical_ttc_s = 1.5;
  double critical_clear_ttc_s = 2.0;
  int clear_frames = 10;
  int min_methods = 2;
};

class WarningFsm {
 public:
  explicit WarningFsm(WarningConfig cfg = {}) : cfg_(cfg) {}
  Level step(bool on_course, std::optional<double> ttc_s, std::optional<double> sigma_ttc_s, int recent_methods);
  Level level() const { return level_; }

 private:
  void enter(Level l) { level_ = l; warn_count_ = 0; clear_count_ = 0; }
  WarningConfig cfg_;
  Level level_ = Level::None;
  int warn_count_ = 0;
  int clear_count_ = 0;
};

}  // namespace fcw
