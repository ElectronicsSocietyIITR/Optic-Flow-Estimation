// Flow divergence per box from an affine fit, after removing camera rotation.
// Port of collision_avoidance/fcw/divergence.py.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "fcw/types.hpp"

namespace fcw {

// n, x, y, xx, xy, yy, u, v, xu, yu, xv, yv (x, y from the principal point).
using FlowMoments = std::array<double, 12>;

struct DivergenceConfig {
  double shrink = 0.10;
  double flow_noise_px = 0.3;
  double inlier_px = 1.0;
  int ransac_iterations = 64;
  int ransac_stride = 2;
  double min_pixels = 50;
  double min_inlier_share = 0.5;  // predicted-fit mask below this share of the box: fall back to RANSAC
  int max_fit_age = 1;            // frames: only last frame's fit predicts this frame's flow
};

// Fixed-point path, the PL engine's (see divergence.py): flow as int16 in 1/64 px, the affine
// fit in 1/64 px scaled by 2^16, moments as exact int64 sums.
constexpr int kFlowScale = 64;
constexpr int kFitShift = 16;
using FixedMoments = std::array<int64_t, 12>;
struct FitQ {
  int64_t a0 = 0, a1 = 0, a2 = 0, b0 = 0, b1 = 0, b2 = 0;
};
using AffineFit = std::pair<cv::Vec3d, cv::Vec3d>;

cv::Mat quantiseFlow(const cv::Mat& flow);  // CV_32FC2 px -> CV_16SC2 in 1/64 px (round half to even, clamped)
std::optional<cv::Rect> flowRect(const Box& box, double shrink, int width, int height);
std::optional<FixedMoments> flowMomentsFixed(const cv::Mat& flow_q, const Box& box, cv::Point2d principal, double shrink = 0.10,
                                             const cv::Mat& mask = {});
FlowMoments dequantiseMoments(const FixedMoments& m);
FitQ fitToFixed(const cv::Vec3d& a, const cv::Vec3d& b);
// Inliers of a predicted affine flow over rect (CV_8U, rect-sized): the engine's exact integer test.
cv::Mat predictedMaskFixed(const cv::Mat& flow_q, const cv::Rect& rect, cv::Point principal, const FitQ& fit, int inlier_q);
// Object pixels in the box (CV_8U over the frame) and whether last frame's fit was used (else RANSAC).
std::pair<cv::Mat, bool> divergenceMask(const cv::Mat& flow, const cv::Mat& flow_q, const Box& box, cv::Point2d principal,
                                        const std::optional<AffineFit>& prev_fit, int fit_age, const DivergenceConfig& cfg = {});

// flow: CV_32FC2; mask: optional CV_8U / bool-like, same size, nonzero = use the pixel.
std::optional<FlowMoments> flowMoments(const cv::Mat& flow, const Box& box, cv::Point2d principal, const cv::Mat& mask = {},
                                       double shrink = 0.10);
std::optional<std::pair<cv::Vec3d, cv::Vec3d>> affineFromMoments(const FlowMoments& m);
std::optional<Measurement> divergenceTtc(const std::optional<FlowMoments>& m, const Box& box, double rotation_div, int k,
                                         double dt, const DivergenceConfig& cfg = {});
// Pixels that follow the dominant affine flow in the box (RANSAC). Returns CV_8U, 1 = keep.
cv::Mat robustMask(const cv::Mat& flow, const Box& box, cv::Point2d principal, const DivergenceConfig& cfg = {});
// Index triples from the fixed 64-bit LCG shared with divergence.py (and the HLS engine).
std::vector<std::array<int, 3>> ransacSamples(int n, int iterations, uint64_t seed = 0);
// Test hook: RANSAC with given sample triples (indices into the stride-subsampled box pixels).
cv::Mat robustMask(const cv::Mat& flow, const Box& box, cv::Point2d principal, const DivergenceConfig& cfg,
                   const std::vector<std::array<int, 3>>& samples);

}  // namespace fcw
