// Forward-collision-warning pipeline. Port of collision_avoidance/fcw/pipeline.py.
//
// frame -> resize/gray -> flow (injected: DIS on PC, Vitis LK on the board) + detections (injected)
//       -> ScaleTracker -> ego-rotation -> 4 TTC methods per track -> fusion -> course check -> FSM
#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include "fcw/closing.hpp"
#include "fcw/collision.hpp"
#include "fcw/divergence.hpp"
#include "fcw/ego_rotation.hpp"
#include "fcw/fusion.hpp"
#include "fcw/horn.hpp"
#include "fcw/scale_search.hpp"
#include "fcw/tracker.hpp"

namespace fcw {

using FlowSource = std::function<cv::Mat(const cv::Mat& prev_gray, const cv::Mat& curr_gray)>;  // CV_32FC2, px/frame
using DetectorFn = std::function<std::vector<Detection>(const cv::Mat& bgr_512x384)>;

struct FcwConfig {
  int frame_width = 512;
  int frame_height = 384;
  double fx = 500.0;
  std::optional<double> fy;  // none: square pixels
  double default_fps = 30.0;
  int max_ttc_tracks = 16;
  bool heading_from_foe = true;  // course reference: smoothed FOE (follows bends); false = principal point
  bool scale_enabled = true;     // scale-ratio search (needs up to 8 past frames)
  TrackerConfig tracker;
  EgoRotationConfig ego;
  ScaleSearchConfig scale;
  HornConfig horn;
  DivergenceConfig divergence;
  FusionConfig fusion;
  CourseConfig course;
  WarningConfig warning;
  ClosingConfig closing;
  cv::Matx33d K() const {
    return {fx, 0, frame_width / 2.0, 0, fy.value_or(fx), frame_height / 2.0, 0, 0, 1};
  }
};

struct ObjectResult {
  int track_id = 0;
  Box bbox;
  std::string cls;
  std::optional<TtcEstimate> estimate;
  std::optional<CourseResult> course;
  Level level = Level::None;
  std::vector<Measurement> measurements;  // everything produced, before gating
  std::optional<double> kappa;            // closing speed / our speed (median of recent frames)
};

struct FcwFrame {
  int index = 0;
  double time_s = 0;
  double dt_s = 0;
  cv::Mat frame;  // resized BGR
  cv::Mat flow;
  EgoMotion ego;
  cv::Point2d heading;  // FOE, or the principal point when there is none
  std::vector<ObjectResult> objects;
  std::map<std::string, double> timings_ms;
  Level level() const;
  const ObjectResult* threat() const;  // on-course object with the smallest TTC
};

// PC stand-in for the board's dense LK: OpenCV DIS optical flow (medium preset).
FlowSource disFlow();
// Processing frame for a source: target_width wide, the source aspect, height a multiple of 32 in [64, 512].
cv::Size processingSize(int width, int height, int target_width = 512);

class FcwPipeline {
 public:
  FcwPipeline(FcwConfig cfg, DetectorFn detector, FlowSource flow);
  FcwFrame process(const cv::Mat& frame_bgr, double t);
  const ScaleTracker& tracker() const { return tracker_; }
  const FcwConfig& config() const { return cfg_; }

 private:
  std::vector<const ScaleTrack*> select(const std::map<int, ScaleTrack>& tracks) const;
  std::optional<std::pair<double, cv::Mat>> frameAt(int frame_index) const;
  ObjectResult assess(const ScaleTrack& track, const cv::Mat& gray, const cv::Mat& prev_gray, const cv::Mat& flow,
                      const EgoMotion& ego, cv::Point2d heading, double yaw_rate, double t, double dt);

  FcwConfig cfg_;
  DetectorFn detector_;
  FlowSource flow_;
  cv::Matx33d K_;
  cv::Point2d pp_;
  ScaleTracker tracker_;
  EgoRotationEstimator ego_;
  TtcFusion fusion_;
  CourseChecker course_;
  std::map<int, WarningFsm> fsms_;
  std::deque<std::tuple<int, double, cv::Mat>> ring_;  // (frame index, time, gray), kHistory entries
  int index_ = -1;
  std::optional<double> prev_t_;
  std::set<int> on_course_;
  std::map<int, int> in_path_streak_;
  std::map<int, int> entry_streak_;
  std::map<int, std::pair<AffineFit, int>> fits_;  // track -> last affine flow fit and its frame
  cv::Mat flow_q_;                                 // this frame's flow in the PL engine's format
  std::map<std::string, double>* timings_ = nullptr;  // per-method accumulators during process()
  ClosingTracker closing_;
  void warn(std::vector<ObjectResult>& objects, const std::map<int, ScaleTrack>& tracks, const cv::Mat& flow,
            const EgoMotion& ego, double dt);
};

}  // namespace fcw
