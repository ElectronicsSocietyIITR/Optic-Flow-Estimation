// Camera rotation and heading from flow. Port of collision_avoidance/fcw/ego_rotation.py.
// R maps camera coordinates prev -> curr; only rotation is removed from the flow.
#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fcw/types.hpp"

namespace fcw {

struct EgoRotationConfig {
  std::string method = "linear";  // "linear": rotation + FOE least squares; "ransac": essential matrix / homography
  int grid_step_px = 16;
  double box_margin = 0.10;
  int hood_rows = 0;
  double ransac_threshold_px = 1.0;
  double ransac_confidence = 0.999;
  double rotation_only_tol = 0.01;
  double rotation_only_share = 0.9;
  double min_inlier_ratio = 0.5;
  double max_jump_rad = 1.0 * CV_PI / 180.0;
  int min_points = 30;
  double min_foe_tz = 0.5;
  int max_iters = 200;    // RANSAC cap: no accuracy loss on KITTI
  int pose_points = 64;   // evenly spread inliers given to recoverPose (0 = all)
  double foe_smoothing = 0.05;  // ~0.7 s: follows bends, not per-frame jitter
  int linear_rounds = 5;            // Gauss-Newton rounds of the linear method
  double huber_px = 1.0;            // robust (Cauchy) weight scale and inlier threshold (px)
  double stationary_px = 0.15;      // flow left after removing rotation below this (at the quantile): stopped
  double stationary_quantile = 0.75;  // not the median: distant background alone can be half the points
  double refit_px = 0.3;            // final rounds: plain least squares over points within this residual
  int refit_rounds = 2;
};

struct EgoMotion {
  cv::Matx33d R = cv::Matx33d::eye();
  cv::Vec3d rvec{0, 0, 0};             // radians over the frame pair
  std::optional<cv::Point2d> foe;      // none when stationary or not moving forward
  double inlier_ratio = 0;
  bool stationary = true;
  bool valid = false;                  // false: low confidence, R held from an earlier frame
  cv::Vec3d omegaRps(double dt) const { return rvec * (1.0 / dt); }
};

// Grid points (prev) and prev + flow (curr) as N x 2 CV_64F, skipping boxes and the hood.
std::pair<cv::Mat, cv::Mat> gridSamples(const cv::Mat& flow, const std::vector<Box>& exclude, const EgoRotationConfig& cfg);

std::optional<EgoMotion> estimate(const cv::Mat& prev, const cv::Mat& curr, const cv::Matx33d& K, const EgoRotationConfig& cfg);
// Rotation and FOE by robust least squares (see ego_rotation.py, estimate_linear).
std::optional<EgoMotion> estimateLinear(const cv::Mat& prev, const cv::Mat& curr, const cv::Matx33d& K, const EgoRotationConfig& cfg,
                                        std::optional<cv::Point2d> foe_init = std::nullopt);

class EgoRotationEstimator {
 public:
  explicit EgoRotationEstimator(const cv::Matx33d& K, EgoRotationConfig cfg = {}) : K_(K), cfg_(cfg) {}
  EgoMotion update(const cv::Mat& flow, const std::vector<Box>& exclude);
  EgoMotion updatePoints(const cv::Mat& prev, const cv::Mat& curr);
  const EgoMotion& last() const { return last_; }

 private:
  EgoMotion hold() const;
  cv::Matx33d K_;
  EgoRotationConfig cfg_;
  EgoMotion last_;
};

// Per-frame divergence the rotation alone adds at a pixel (exact, via H = K R K^-1).
double rotationDivergence(const cv::Matx33d& R, const cv::Matx33d& K, cv::Point2d center);

}  // namespace fcw
