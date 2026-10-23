// Parity: fcw::EgoRotationEstimator vs collision_avoidance/fcw/ego_rotation.py (golden file ego.yml.gz).
#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>

#include "fcw/ego_rotation.hpp"
#include "golden.hpp"

namespace {

double angleDeg(const cv::Matx33d& A, const cv::Matx33d& B) {
  cv::Mat r;
  cv::Rodrigues(cv::Mat(A * B.t()), r);
  return cv::norm(r) * 180.0 / CV_PI;
}

// Linear: deterministic, so tight. RANSAC: OpenCV's sampling differs from Python's, so loose.
void replay(const std::string& name, const std::string& method = "linear") {
  golden::File g("ego");
  cv::Matx33d K;
  g.mat("K").copyTo(cv::Mat(3, 3, CV_64F, K.val));
  fcw::EgoRotationConfig cfg;
  cfg.method = method;
  fcw::EgoRotationEstimator est(K, cfg);
  const bool tight = method == "linear";
  const int steps = static_cast<int>(g.num(name + "_steps"));
  for (int i = 0; i < steps; ++i) {
    const std::string k = name + "_s" + std::to_string(i);
    const fcw::EgoMotion ego = est.updatePoints(g.mat(k + "_prev"), g.mat(k + "_curr"));
    cv::Matx33d R;
    g.mat(k + "_R").copyTo(cv::Mat(3, 3, CV_64F, R.val));
    EXPECT_LT(angleDeg(ego.R, R), 0.05) << k;
    if (tight) {
      const cv::Mat rv = g.mat(k + "_rvec");
      for (int j = 0; j < 3; ++j) EXPECT_NEAR(ego.rvec[j], rv.at<double>(0, j), 1e-9) << k << " rvec " << j;
    }
    EXPECT_EQ(ego.valid, g.num(k + "_valid") != 0) << k;
    EXPECT_EQ(ego.stationary, g.num(k + "_stationary") != 0) << k;
    ASSERT_EQ(ego.foe.has_value(), g.num(k + "_has_foe") != 0) << k;
    if (ego.foe) {
      cv::Mat foe = g.mat(k + "_foe");
      EXPECT_NEAR(ego.foe->x, foe.at<double>(0, 0), tight ? 1e-6 : 0.5) << k;
      EXPECT_NEAR(ego.foe->y, foe.at<double>(0, 1), tight ? 1e-6 : 0.5) << k;
    }
  }
}

}  // namespace

TEST(EgoParity, ForwardNoYaw) { replay("yaw0"); }
TEST(EgoParity, ForwardYawRight) { replay("yaw04"); }
TEST(EgoParity, ForwardYawLeft) { replay("yawm08"); }
TEST(EgoParity, PureRotationIsStationary) { replay("pure"); }
TEST(EgoParity, ExcludedTruck) { replay("truck"); }
TEST(EgoParity, NoiseHoldsLastEstimate) { replay("hold"); }
TEST(EgoParity, LinearYawAndPitch) { replay("pitch"); }
TEST(EgoParity, LinearHoldsWhenBackgroundHidden) { replay("hidden"); }
TEST(EgoParity, RansacForwardYaw) { replay("ransac_yaw04", "ransac"); }
TEST(EgoParity, RansacPureRotation) { replay("ransac_pure", "ransac"); }

TEST(EgoParity, GridSamplingMatchesExactly) {
  golden::File g("ego");
  fcw::EgoRotationConfig cfg;
  cfg.grid_step_px = 8;
  cfg.hood_rows = 4;
  const auto [prev, curr] = fcw::gridSamples(g.mat("small_flow"), {{10, 5, 30, 20}}, cfg);
  cv::Mat gp = g.mat("small_prev"), gc = g.mat("small_curr");
  ASSERT_EQ(prev.rows, gp.rows);
  EXPECT_EQ(cv::norm(prev, gp, cv::NORM_INF), 0.0);
  EXPECT_LT(cv::norm(curr, gc, cv::NORM_INF), 1e-6);  // float32 flow added in double on both sides
}

TEST(EgoParity, RotationDivergence) {
  golden::File g("ego");
  cv::Matx33d K;
  g.mat("K").copyTo(cv::Mat(3, 3, CV_64F, K.val));
  for (int j = 0; j < 3; ++j) {
    cv::Matx33d R;
    g.mat("div" + std::to_string(j) + "_R").copyTo(cv::Mat(3, 3, CV_64F, R.val));
    cv::Mat c = g.mat("div" + std::to_string(j) + "_c");
    EXPECT_NEAR(fcw::rotationDivergence(R, K, {c.at<double>(0, 0), c.at<double>(0, 1)}), g.num("div" + std::to_string(j) + "_value"), 1e-9);
  }
}
