// Port of collision_avoidance/fcw/ego_rotation.py.
#include "fcw/ego_rotation.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>

namespace fcw {

std::pair<cv::Mat, cv::Mat> gridSamples(const cv::Mat& flow, const std::vector<Box>& exclude, const EgoRotationConfig& cfg) {
  const int h = flow.rows, w = flow.cols, s = cfg.grid_step_px;
  std::vector<cv::Vec2d> prev, curr;
  for (int y = s / 2; y < h - cfg.hood_rows; y += s) {
    for (int x = s / 2; x < w; x += s) {
      bool keep = true;
      for (const Box& b : exclude) {
        const double mx = cfg.box_margin * (b.x2 - b.x1), my = cfg.box_margin * (b.y2 - b.y1);
        if (x >= b.x1 - mx && x <= b.x2 + mx && y >= b.y1 - my && y <= b.y2 + my) { keep = false; break; }
      }
      if (!keep) continue;
      const cv::Vec2f f = flow.at<cv::Vec2f>(y, x);
      prev.emplace_back(x, y);
      curr.emplace_back(x + static_cast<double>(f[0]), y + static_cast<double>(f[1]));
    }
  }
  return {cv::Mat(prev, true).reshape(1), cv::Mat(curr, true).reshape(1)};
}

static double angleOf(const cv::Matx33d& R) {
  cv::Vec3d r;
  cv::Rodrigues(R, r);
  return cv::norm(r);
}

static cv::Vec3d rvecOf(const cv::Matx33d& R) {
  cv::Vec3d r;
  cv::Rodrigues(R, r);
  return r;
}

// Nearest rotation to K^-1 H K and the largest |singular value - 1|.
static std::pair<cv::Matx33d, double> rotationFromHomography(const cv::Mat& H, const cv::Matx33d& K) {
  cv::Mat A = cv::Mat(K.inv()) * H * cv::Mat(K);
  A /= std::cbrt(cv::determinant(A));
  cv::Mat w, u, vt;
  cv::SVD::compute(A, w, u, vt);
  cv::Mat R = u * vt;
  if (cv::determinant(R) < 0) R = -R;
  double deviation = 0;
  for (int i = 0; i < 3; ++i) deviation = std::max(deviation, std::abs(w.at<double>(i) - 1.0));
  return {cv::Matx33d(R), deviation};
}

std::optional<EgoMotion> estimate(const cv::Mat& prev, const cv::Mat& curr, const cv::Matx33d& K, const EgoRotationConfig& cfg) {
  const int n = prev.rows;
  if (n < cfg.min_points) return std::nullopt;
  cv::Mat e_mask;
  cv::Mat E = cv::findEssentialMat(prev, curr, cv::Mat(K), cv::RANSAC, cfg.ransac_confidence, cfg.ransac_threshold_px,
                                   cfg.max_iters, e_mask);
  const double e_ratio = E.empty() ? 0.0 : cv::countNonZero(e_mask) / static_cast<double>(n);
  // Pure rotation (or a stopped car) is degenerate for E: prefer the rotation homography
  // when it explains about as many points as E does.
  cv::Mat h_mask;
  cv::Mat H = cv::findHomography(prev, curr, cv::RANSAC, cfg.ransac_threshold_px, h_mask, cfg.max_iters);
  if (!H.empty()) {
    const auto [R_h, deviation] = rotationFromHomography(H, K);
    const double h_ratio = cv::countNonZero(h_mask) / static_cast<double>(n);
    if (deviation < cfg.rotation_only_tol && h_ratio >= cfg.rotation_only_share * e_ratio) {
      EgoMotion m;
      m.R = R_h;
      m.rvec = rvecOf(R_h);
      m.inlier_ratio = h_ratio;
      m.stationary = true;
      m.valid = true;
      return m;
    }
  }
  if (E.empty()) return std::nullopt;
  E = E.rowRange(0, 3);  // findEssentialMat may stack several solutions
  cv::Mat R, t;
  // recoverPose's own mask also drops far background, so E's RANSAC share is the confidence.
  std::vector<int> inliers;
  for (int i = 0; i < n; ++i)
    if (e_mask.at<uint8_t>(i)) inliers.push_back(i);
  if (cfg.pose_points > 0 && static_cast<int>(inliers.size()) > cfg.pose_points) {
    // Triangulating every inlier was half the frame time; a spread subset picks the same pose.
    cv::Mat p(cfg.pose_points, 2, CV_64F), c(cfg.pose_points, 2, CV_64F);
    const double step = static_cast<double>(inliers.size() - 1) / (cfg.pose_points - 1);
    for (int j = 0; j < cfg.pose_points; ++j) {
      const int i = inliers[static_cast<int>(j * step)];  // numpy linspace(...).astype(int)
      prev.row(i).copyTo(p.row(j));
      curr.row(i).copyTo(c.row(j));
    }
    cv::recoverPose(E, p, c, cv::Mat(K), R, t);
  } else {
    cv::Mat pose_mask = e_mask.clone();
    cv::recoverPose(E, prev, curr, cv::Mat(K), R, t, pose_mask);
  }
  EgoMotion m;
  m.R = cv::Matx33d(R);
  m.rvec = rvecOf(m.R);
  m.inlier_ratio = e_ratio;
  m.stationary = false;
  m.valid = true;
  cv::Vec3d tv(t.at<double>(0), t.at<double>(1), t.at<double>(2));
  tv /= cv::norm(tv);
  if (std::abs(tv[2]) >= cfg.min_foe_tz) {
    const cv::Vec3d p = K * tv;
    m.foe = cv::Point2d(p[0] / p[2], p[1] / p[2]);
  }
  return m;
}

namespace {

double cauchy(double r, double k) { return 1.0 / (1.0 + (r / k) * (r / k)); }

// numpy.quantile with linear interpolation.
double quantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  const double pos = q * (v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos)), hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

void rotate(const cv::Matx33d& R, double x, double y, double& xr, double& yr) {
  const cv::Vec3d q = R * cv::Vec3d(x, y, 1.0);
  xr = q[0] / q[2];
  yr = q[1] / q[2];
}

cv::Matx33d expRot(const cv::Vec3d& w) {
  cv::Matx33d R;
  cv::Rodrigues(w, R);
  return R;
}

}  // namespace

std::optional<EgoMotion> estimateLinear(const cv::Mat& prev, const cv::Mat& curr, const cv::Matx33d& K, const EgoRotationConfig& cfg,
                                        std::optional<cv::Point2d> foe_init) {
  const int n = prev.rows;
  if (n < cfg.min_points) return std::nullopt;
  const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2), px = 0.5 * (fx + fy);
  std::vector<double> x(n), y(n), xc(n), yc(n);
  for (int i = 0; i < n; ++i) {
    x[i] = (prev.at<double>(i, 0) - cx) / fx;
    y[i] = (prev.at<double>(i, 1) - cy) / fy;
    xc[i] = (curr.at<double>(i, 0) - cx) / fx;
    yc[i] = (curr.at<double>(i, 1) - cy) / fy;
  }
  // d(flow)/d(omega), rows for u and for v.
  auto ju = [&](int i) { return cv::Vec3d(-x[i] * y[i], 1 + x[i] * x[i], -y[i]); };
  auto jv = [&](int i) { return cv::Vec3d(-(1 + y[i] * y[i]), x[i] * y[i], x[i]); };

  // Rotation only (also the stationary test).
  cv::Matx33d R = cv::Matx33d::eye();
  for (int round = 0; round < cfg.linear_rounds; ++round) {
    cv::Matx33d M = cv::Matx33d::eye() * 1e-12;
    cv::Vec3d b(0, 0, 0);
    for (int i = 0; i < n; ++i) {
      double xr, yr;
      rotate(R, x[i], y[i], xr, yr);
      const double ru = xc[i] - xr, rv = yc[i] - yr, w = cauchy(std::hypot(ru, rv) * px, cfg.huber_px);
      const cv::Vec3d a = ju(i), c = jv(i);
      M += w * (a * a.t() + c * c.t());
      b += w * (a * ru + c * rv);
    }
    cv::Vec3d d;
    cv::solve(M, b, d, cv::DECOMP_LU);
    R = expRot(d) * R;
  }
  std::vector<double> left(n);
  for (int i = 0; i < n; ++i) {
    double xr, yr;
    rotate(R, x[i], y[i], xr, yr);
    left[i] = std::hypot(xc[i] - xr, yc[i] - yr) * px;
  }
  if (quantile(left, cfg.stationary_quantile) < cfg.stationary_px) {
    EgoMotion m;
    m.R = R;
    m.rvec = rvecOf(R);
    m.inlier_ratio = std::count_if(left.begin(), left.end(), [&](double v) { return v < cfg.huber_px; }) / static_cast<double>(n);
    m.stationary = true;
    m.valid = true;
    return m;
  }

  // Driving: rotation update and FOE together (5 unknowns).
  R = cv::Matx33d::eye();
  double ex = foe_init ? (foe_init->x - cx) / fx : 0.0, ey = foe_init ? (foe_init->y - cy) / fy : 0.0;
  for (int round = 0; round < cfg.linear_rounds + cfg.refit_rounds; ++round) {
    cv::Matx<double, 5, 5> M = cv::Matx<double, 5, 5>::eye() * 1e-12;
    cv::Vec<double, 5> b = cv::Vec<double, 5>::all(0);
    double wsum = 0;
    for (int i = 0; i < n; ++i) {
      double xr, yr;
      rotate(R, x[i], y[i], xr, yr);
      const double ru = xc[i] - xr, rv = yc[i] - yr, dx = xr - ex, dy = yr - ey;
      const double dist = std::max(std::hypot(dx, dy), 0.05);
      const double res = (ru * dy - rv * dx) / dist;  // flow across the radial line (normalised)
      const cv::Vec3d g = (-ju(i) * dy + jv(i) * dx) * (1.0 / dist);
      const cv::Vec<double, 5> J(g[0], g[1], g[2], rv / dist, -ru / dist);
      const double w = round < cfg.linear_rounds ? cauchy(res * px, cfg.huber_px)
                                                 : (std::abs(res) * px < cfg.refit_px ? 1.0 : 0.0);
      wsum += w;
      M += w * (J * J.t());
      b -= w * res * J;
    }
    if (round >= cfg.linear_rounds && wsum < cfg.min_points) break;
    cv::Vec<double, 5> step;
    cv::solve(M, b, step, cv::DECOMP_LU);
    R = expRot(cv::Vec3d(step[0], step[1], step[2])) * R;
    ex += step[3];
    ey += step[4];
  }
  int inliers = 0;
  double outward = 0;
  for (int i = 0; i < n; ++i) {
    double xr, yr;
    rotate(R, x[i], y[i], xr, yr);
    const double ru = xc[i] - xr, rv = yc[i] - yr, dx = xr - ex, dy = yr - ey;
    if (std::abs(ru * dy - rv * dx) / std::max(std::hypot(dx, dy), 0.05) * px < cfg.huber_px) ++inliers;
    outward += ru * dx + rv * dy;
  }
  EgoMotion m;
  m.R = R;
  m.rvec = rvecOf(R);
  m.inlier_ratio = inliers / static_cast<double>(n);
  m.stationary = false;
  m.valid = true;
  const double max_e = std::sqrt(1.0 / (cfg.min_foe_tz * cfg.min_foe_tz) - 1.0);  // same limit as |t_z| >= min_foe_tz
  if (outward > 0 && std::hypot(ex, ey) <= max_e) m.foe = cv::Point2d(ex * fx + cx, ey * fy + cy);
  return m;
}

EgoMotion EgoRotationEstimator::hold() const {
  EgoMotion held = last_;
  held.inlier_ratio = 0;
  held.valid = false;
  return held;
}

EgoMotion EgoRotationEstimator::updatePoints(const cv::Mat& prev, const cv::Mat& curr) {
  std::optional<EgoMotion> est = cfg_.method == "linear" ? estimateLinear(prev, curr, K_, cfg_, last_.foe) : estimate(prev, curr, K_, cfg_);
  if (!est || est->inlier_ratio < cfg_.min_inlier_ratio) return hold();
  if (last_.valid && std::abs(angleOf(est->R) - angleOf(last_.R)) > cfg_.max_jump_rad) return hold();
  if (est->foe && last_.foe) {
    const double a = cfg_.foe_smoothing;
    est->foe = cv::Point2d(a * est->foe->x + (1 - a) * last_.foe->x, a * est->foe->y + (1 - a) * last_.foe->y);
  }
  last_ = *est;
  return *est;
}

EgoMotion EgoRotationEstimator::update(const cv::Mat& flow, const std::vector<Box>& exclude) {
  const auto [prev, curr] = gridSamples(flow, exclude, cfg_);
  return updatePoints(prev, curr);
}

double rotationDivergence(const cv::Matx33d& R, const cv::Matx33d& K, cv::Point2d c) {
  const cv::Matx33d H = K * R * K.inv();
  auto mapped = [&](double x, double y) {
    const cv::Vec3d p = H * cv::Vec3d(x, y, 1.0);
    return cv::Point2d(p[0] / p[2], p[1] / p[2]);
  };
  const double d = 1.0;
  const double dgx = (mapped(c.x + d, c.y).x - mapped(c.x - d, c.y).x) / (2 * d);
  const double dgy = (mapped(c.x, c.y + d).y - mapped(c.x, c.y - d).y) / (2 * d);
  return dgx + dgy - 2.0;
}

}  // namespace fcw
