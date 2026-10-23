// Port of collision_avoidance/fcw/horn.py.
#include "fcw/horn.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>

namespace fcw {

std::optional<cv::Rect> innerBox(const Box& b, double shrink, int width, int height) {
  const double mx = shrink / 2 * (b.x2 - b.x1), my = shrink / 2 * (b.y2 - b.y1);
  const int x1 = std::max(1, static_cast<int>(std::ceil(b.x1 + mx))), y1 = std::max(1, static_cast<int>(std::ceil(b.y1 + my)));
  const int x2 = std::min(width - 1, static_cast<int>(std::floor(b.x2 - mx)));
  const int y2 = std::min(height - 1, static_cast<int>(std::floor(b.y2 - my)));
  if (x2 <= x1 || y2 <= y1) return std::nullopt;
  return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

std::optional<HornSums> hornSums(const cv::Mat& prev, const cv::Mat& curr, const Box& box, cv::Point2d principal,
                                 const HornConfig& cfg) {
  CV_Assert(prev.type() == CV_8UC1 && curr.type() == CV_8UC1 && prev.size() == curr.size());
  const auto r = innerBox(box, cfg.shrink, curr.cols, curr.rows);
  if (!r) return std::nullopt;
  // Python's round() rounds half to even, like nearbyint in the default rounding mode.
  const int64_t px = static_cast<int64_t>(std::nearbyint(principal.x)), py = static_cast<int64_t>(std::nearbyint(principal.y));
  auto S = [&](int y, int x) { return static_cast<int64_t>(prev.at<uint8_t>(y, x)) + curr.at<uint8_t>(y, x); };
  HornSums out;
  for (int y = r->y; y < r->y + r->height; ++y) {
    for (int x = r->x; x < r->x + r->width; ++x) {
      const int64_t ex = (S(y - 1, x + 1) + 2 * S(y, x + 1) + S(y + 1, x + 1)) - (S(y - 1, x - 1) + 2 * S(y, x - 1) + S(y + 1, x - 1));
      const int64_t ey = (S(y + 1, x - 1) + 2 * S(y + 1, x) + S(y + 1, x + 1)) - (S(y - 1, x - 1) + 2 * S(y - 1, x) + S(y - 1, x + 1));
      if (std::llabs(ex) + std::llabs(ey) <= cfg.grad_threshold_l1) continue;
      const int64_t et = static_cast<int64_t>(curr.at<uint8_t>(y, x)) - prev.at<uint8_t>(y, x);
      const int64_t g = (x - px) * ex + (y - py) * ey;
      auto& s = out.s;
      s[0] += ex * ex; s[1] += ex * ey; s[2] += ex * g; s[3] += ey * ey; s[4] += ey * g;
      s[5] += g * g;   s[6] += ex * et; s[7] += ey * et; s[8] += g * et; s[9] += et * et;
      ++out.n;
    }
  }
  return out;
}

std::optional<Measurement> hornSolve(const HornSums& sums, double dt, const HornConfig& cfg) {
  if (sums.n < cfg.min_pixels) return std::nullopt;
  double s[10];
  for (int i = 0; i < 10; ++i) s[i] = static_cast<double>(sums.s[i]);
  const cv::Matx33d M(s[0], s[1], s[2], s[1], s[3], s[4], s[2], s[4], s[5]);
  const cv::Vec3d q(s[6], s[7], s[8]);
  const double d[3] = {std::sqrt(M(0, 0)), std::sqrt(M(1, 1)), std::sqrt(M(2, 2))};
  if (!(d[0] > 0 && d[1] > 0 && d[2] > 0)) return std::nullopt;
  cv::Matx33d Mn;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) Mn(a, b) = M(a, b) / (d[a] * d[b]);
  cv::Vec3d sv;
  cv::SVD::compute(Mn, sv);
  if (sv[2] <= 0 || sv[0] / sv[2] > cfg.max_condition) return std::nullopt;  // numpy.linalg.cond (2-norm)
  cv::Vec3d p;
  cv::solve(M, -q, p, cv::DECOMP_LU);  // (A', B', C')
  const double residual = s[9] + 2 * p.dot(q) + p.dot(M * p);
  if (s[9] <= 0 || residual / s[9] > cfg.max_rel_residual) return std::nullopt;
  const double sigma2 = std::max(residual, 0.0) / static_cast<double>(std::max<int64_t>(sums.n - 3, 1));
  const double var_c = sigma2 * M.inv(cv::DECOMP_LU)(2, 2);
  return Measurement{"horn", kSobelGain * p[2] / dt, kSobelGain * kSobelGain * var_c / (dt * dt)};
}

int chooseLevel(std::optional<double> eta, const Box& b, double speed, double dt, int levels) {
  const double half_diag = 0.5 * std::hypot(b.x2 - b.x1, b.y2 - b.y1);
  const double motion = (eta ? std::abs(*eta) : 0.0) * dt * half_diag + speed * dt;
  int level = 0;
  while (motion / std::pow(2.0, level) > 1.0 && level < levels - 1) ++level;
  return level;
}

cv::Mat downsample(const cv::Mat& gray, int level) {
  cv::Mat out = gray;
  for (int i = 0; i < level; ++i) cv::pyrDown(out, out, cv::Size(), cv::BORDER_REPLICATE);  // = the fabric's Vitis pyrDown
  return out;
}

std::optional<Measurement> hornAtLevel(const cv::Mat& prev, const cv::Mat& curr, const Box& b, cv::Point2d principal,
                                       int level, double dt, const HornConfig& cfg) {
  const double f = std::pow(2.0, -level);
  const auto sums = hornSums(downsample(prev, level), downsample(curr, level), {b.x1 * f, b.y1 * f, b.x2 * f, b.y2 * f},
                             {principal.x * f, principal.y * f}, cfg);
  return sums ? hornSolve(*sums, dt, cfg) : std::nullopt;
}

}  // namespace fcw
