// Port of collision_avoidance/fcw/divergence.py.
#include "fcw/divergence.hpp"

#include <algorithm>
#include <cmath>

namespace fcw {

std::optional<FlowMoments> flowMoments(const cv::Mat& flow, const Box& b, cv::Point2d pp, const cv::Mat& mask, double shrink) {
  const int h = flow.rows, w = flow.cols;
  const double mx = shrink / 2 * (b.x2 - b.x1), my = shrink / 2 * (b.y2 - b.y1);
  const int x1 = std::max(0, static_cast<int>(std::ceil(b.x1 + mx))), y1 = std::max(0, static_cast<int>(std::ceil(b.y1 + my)));
  const int x2 = std::min(w, static_cast<int>(std::floor(b.x2 - mx))), y2 = std::min(h, static_cast<int>(std::floor(b.y2 - my)));
  if (x2 <= x1 || y2 <= y1) return std::nullopt;
  FlowMoments m{};
  for (int yy = y1; yy < y2; ++yy) {
    for (int xx = x1; xx < x2; ++xx) {
      if (!mask.empty() && !mask.at<uint8_t>(yy, xx)) continue;
      const double x = xx - pp.x, y = yy - pp.y;
      const cv::Vec2f f = flow.at<cv::Vec2f>(yy, xx);
      const double u = f[0], v = f[1];
      m[0] += 1; m[1] += x; m[2] += y; m[3] += x * x; m[4] += x * y; m[5] += y * y;
      m[6] += u; m[7] += v; m[8] += x * u; m[9] += y * u; m[10] += x * v; m[11] += y * v;
    }
  }
  return m;
}

std::optional<std::pair<cv::Vec3d, cv::Vec3d>> affineFromMoments(const FlowMoments& m) {
  const cv::Matx33d A(m[0], m[1], m[2], m[1], m[3], m[4], m[2], m[4], m[5]);
  if (m[0] < 3 || std::abs(cv::determinant(A)) < 1e-9) return std::nullopt;
  cv::Vec3d a, b;
  cv::solve(A, cv::Vec3d(m[6], m[8], m[9]), a, cv::DECOMP_LU);
  cv::solve(A, cv::Vec3d(m[7], m[10], m[11]), b, cv::DECOMP_LU);
  return std::make_pair(a, b);
}

std::optional<Measurement> divergenceTtc(const std::optional<FlowMoments>& m, const Box& b, double rotation_div, int k,
                                         double dt, const DivergenceConfig& cfg) {
  if (!m || (*m)[0] < cfg.min_pixels) return std::nullopt;
  const auto fit = affineFromMoments(*m);
  if (!fit) return std::nullopt;
  const double div = fit->first[1] + fit->second[2] - rotation_div;
  const double w = b.x2 - b.x1, h = b.y2 - b.y1;
  const double var_div = cfg.flow_noise_px * cfg.flow_noise_px * 12.0 / (*m)[0] * (1.0 / (w * w) + 1.0 / (h * h));
  const double span = 2.0 * k * dt;
  return Measurement{"divergence", div / span, var_div / (span * span)};
}

namespace {

struct Region { int x1, y1, x2, y2; };

std::optional<Region> robustRegion(const cv::Mat& flow, const Box& b) {
  const Region r{std::max(0, static_cast<int>(b.x1)), std::max(0, static_cast<int>(b.y1)),
                 std::min(flow.cols, static_cast<int>(std::ceil(b.x2))), std::min(flow.rows, static_cast<int>(std::ceil(b.y2)))};
  if (r.x2 - r.x1 < 3 || r.y2 - r.y1 < 3) return std::nullopt;
  return r;
}

cv::Mat ransac(const cv::Mat& flow, const Region& r, cv::Point2d pp, const DivergenceConfig& cfg,
               const std::vector<std::array<int, 3>>& samples) {
  cv::Mat mask = cv::Mat::ones(flow.size(), CV_8U);
  std::vector<cv::Vec3d> X;
  std::vector<double> U, V;
  for (int yy = r.y1; yy < r.y2; yy += cfg.ransac_stride)
    for (int xx = r.x1; xx < r.x2; xx += cfg.ransac_stride) {
      const cv::Vec2f f = flow.at<cv::Vec2f>(yy, xx);
      X.emplace_back(1.0, xx - pp.x, yy - pp.y);
      U.push_back(f[0]);
      V.push_back(f[1]);
    }
  bool found = false;
  cv::Vec3d best_a, best_b;
  long best_count = -1;
  for (const auto& idx : samples) {
    const cv::Matx33d A(X[idx[0]][0], X[idx[0]][1], X[idx[0]][2], X[idx[1]][0], X[idx[1]][1], X[idx[1]][2],
                        X[idx[2]][0], X[idx[2]][1], X[idx[2]][2]);
    cv::Vec3d a, b;
    if (!cv::solve(A, cv::Vec3d(U[idx[0]], U[idx[1]], U[idx[2]]), a, cv::DECOMP_LU)) continue;
    if (!cv::solve(A, cv::Vec3d(V[idx[0]], V[idx[1]], V[idx[2]]), b, cv::DECOMP_LU)) continue;
    long count = 0;
    for (size_t i = 0; i < X.size(); ++i)
      if (std::hypot(X[i].dot(a) - U[i], X[i].dot(b) - V[i]) < cfg.inlier_px) ++count;
    if (count > best_count) { best_count = count; best_a = a; best_b = b; found = true; }
  }
  if (!found) return mask;
  for (int yy = r.y1; yy < r.y2; ++yy)
    for (int xx = r.x1; xx < r.x2; ++xx) {
      const double x = xx - pp.x, y = yy - pp.y;
      const cv::Vec2f f = flow.at<cv::Vec2f>(yy, xx);
      const double res = std::hypot(f[0] - (best_a[0] + best_a[1] * x + best_a[2] * y), f[1] - (best_b[0] + best_b[1] * x + best_b[2] * y));
      mask.at<uint8_t>(yy, xx) = res < cfg.inlier_px ? 1 : 0;
    }
  return mask;
}

}  // namespace

cv::Mat robustMask(const cv::Mat& flow, const Box& b, cv::Point2d pp, const DivergenceConfig& cfg,
                   const std::vector<std::array<int, 3>>& samples) {
  const auto r = robustRegion(flow, b);
  return r ? ransac(flow, *r, pp, cfg, samples) : cv::Mat::ones(flow.size(), CV_8U);
}

std::vector<std::array<int, 3>> ransacSamples(int n, int iterations, uint64_t seed) {
  uint64_t state = seed ^ 0x853C49E6748FEA9BULL;
  std::vector<std::array<int, 3>> out;
  for (int it = 0; it < iterations; ++it) {
    std::array<int, 3> triple{};
    int filled = 0;
    while (filled < 3) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;  // wraps mod 2^64
      const int value = static_cast<int>((state >> 33) % static_cast<uint64_t>(n));
      bool unique = true;
      for (int q = 0; q < filled; ++q) unique &= triple[q] != value;
      if (unique) triple[filled++] = value;
    }
    out.push_back(triple);
  }
  return out;
}

cv::Mat robustMask(const cv::Mat& flow, const Box& b, cv::Point2d pp, const DivergenceConfig& cfg) {
  const auto r = robustRegion(flow, b);
  if (!r) return cv::Mat::ones(flow.size(), CV_8U);
  const int nx = (r->x2 - r->x1 + cfg.ransac_stride - 1) / cfg.ransac_stride;
  const int ny = (r->y2 - r->y1 + cfg.ransac_stride - 1) / cfg.ransac_stride;
  return ransac(flow, *r, pp, cfg, ransacSamples(nx * ny, cfg.ransac_iterations));
}

cv::Mat quantiseFlow(const cv::Mat& flow) {
  // saturate_cast<short>: round half to even, then clip to int16 (np.round + np.clip); vectorised. The
  // scalar loop it replaces cost 2.5 ms per 512x288 frame on the laptop, ~40 ms on an A53.
  cv::Mat q;
  flow.convertTo(q, CV_16SC2, kFlowScale);
  return q;
}

std::optional<cv::Rect> flowRect(const Box& b, double shrink, int width, int height) {
  const double mx = shrink / 2 * (b.x2 - b.x1), my = shrink / 2 * (b.y2 - b.y1);
  const int x1 = std::max(0, static_cast<int>(std::ceil(b.x1 + mx))), y1 = std::max(0, static_cast<int>(std::ceil(b.y1 + my)));
  const int x2 = std::min(width, static_cast<int>(std::floor(b.x2 - mx))), y2 = std::min(height, static_cast<int>(std::floor(b.y2 - my)));
  if (x2 <= x1 || y2 <= y1) return std::nullopt;
  return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

std::optional<FixedMoments> flowMomentsFixed(const cv::Mat& flow_q, const Box& b, cv::Point2d pp, double shrink, const cv::Mat& mask) {
  const auto r = flowRect(b, shrink, flow_q.cols, flow_q.rows);
  if (!r) return std::nullopt;
  const int64_t px = static_cast<int64_t>(std::nearbyint(pp.x)), py = static_cast<int64_t>(std::nearbyint(pp.y));
  FixedMoments m{};
  for (int yy = r->y; yy < r->y + r->height; ++yy)
    for (int xx = r->x; xx < r->x + r->width; ++xx) {
      if (!mask.empty() && !mask.at<uchar>(yy, xx)) continue;
      const int64_t x = xx - px, y = yy - py;
      const cv::Vec2s f = flow_q.at<cv::Vec2s>(yy, xx);
      const int64_t u = f[0], v = f[1];
      const int64_t t[12] = {1, x, y, x * x, x * y, y * y, u, v, x * u, y * u, x * v, y * v};
      for (int k = 0; k < 12; ++k) m[k] += t[k];
    }
  return m;
}

FlowMoments dequantiseMoments(const FixedMoments& m) {
  FlowMoments out;
  for (int k = 0; k < 12; ++k) out[k] = static_cast<double>(m[k]) / (k >= 6 ? kFlowScale : 1);
  return out;
}

FitQ fitToFixed(const cv::Vec3d& a, const cv::Vec3d& b) {
  const double s = static_cast<double>(kFlowScale) * (int64_t{1} << kFitShift);
  auto q = [&](double c) { return static_cast<int64_t>(std::nearbyint(c * s)); };
  return {q(a[0]), q(a[1]), q(a[2]), q(b[0]), q(b[1]), q(b[2])};
}

cv::Mat predictedMaskFixed(const cv::Mat& flow_q, const cv::Rect& r, cv::Point pp, const FitQ& f, int inlier_q) {
  cv::Mat mask(r.size(), CV_8U);
  const int64_t lim = static_cast<int64_t>(inlier_q) * inlier_q;
  for (int yy = r.y; yy < r.y + r.height; ++yy)
    for (int xx = r.x; xx < r.x + r.width; ++xx) {
      const int64_t x = xx - pp.x, y = yy - pp.y;
      const cv::Vec2s q = flow_q.at<cv::Vec2s>(yy, xx);
      // >> on negative int64 is arithmetic (floor) in C++20 and in practice here, as numpy's right_shift.
      const int64_t du = std::clamp<int64_t>(((static_cast<int64_t>(q[0]) << kFitShift) - (f.a0 + f.a1 * x + f.a2 * y)) >> kFitShift, -32767, 32767);
      const int64_t dv = std::clamp<int64_t>(((static_cast<int64_t>(q[1]) << kFitShift) - (f.b0 + f.b1 * x + f.b2 * y)) >> kFitShift, -32767, 32767);
      mask.at<uchar>(yy - r.y, xx - r.x) = du * du + dv * dv < lim;
    }
  return mask;
}

std::pair<cv::Mat, bool> divergenceMask(const cv::Mat& flow, const cv::Mat& flow_q, const Box& box, cv::Point2d pp,
                                        const std::optional<AffineFit>& prev_fit, int fit_age, const DivergenceConfig& cfg) {
  const auto r = flowRect(box, cfg.shrink, flow.cols, flow.rows);
  if (prev_fit && fit_age <= cfg.max_fit_age && r) {
    const cv::Point ppi(static_cast<int>(std::nearbyint(pp.x)), static_cast<int>(std::nearbyint(pp.y)));
    const cv::Mat inside = predictedMaskFixed(flow_q, *r, ppi, fitToFixed(prev_fit->first, prev_fit->second),
                                              static_cast<int>(std::nearbyint(cfg.inlier_px * kFlowScale)));
    if (cv::countNonZero(inside) >= cfg.min_inlier_share * inside.total()) {
      cv::Mat mask = cv::Mat::zeros(flow.size(), CV_8U);
      inside.copyTo(mask(*r));
      return {mask, true};
    }
  }
  return {robustMask(flow, box, pp, cfg), false};
}

}  // namespace fcw
