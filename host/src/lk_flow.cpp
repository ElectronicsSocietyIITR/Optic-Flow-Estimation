#include "fcw/lk_flow.hpp"

#include <cmath>
#include <vector>

namespace fcw {

namespace {

// Gray image as float, plain 2x2-average pyramid.
std::vector<cv::Mat> pyramid(const cv::Mat& gray8, int levels) {
  std::vector<cv::Mat> pyr(1);
  gray8.convertTo(pyr[0], CV_32F);
  for (int l = 1; l < levels; ++l) {
    const cv::Mat& src = pyr[l - 1];
    cv::Mat dst(src.rows / 2, src.cols / 2, CV_32F);
    for (int y = 0; y < dst.rows; ++y)
      for (int x = 0; x < dst.cols; ++x)
        dst.at<float>(y, x) = 0.25f * (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) +
                                       src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1));
    pyr.push_back(dst);
  }
  return pyr;
}

float at(const cv::Mat& m, int y, int x) {
  x = std::min(std::max(x, 0), m.cols - 1);
  y = std::min(std::max(y, 0), m.rows - 1);
  return m.at<float>(y, x);
}

float bilinear(const cv::Mat& m, float y, float x) {
  const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
  const float ax = x - x0, ay = y - y0;
  return (1 - ay) * ((1 - ax) * at(m, y0, x0) + ax * at(m, y0, x0 + 1)) + ay * ((1 - ax) * at(m, y0 + 1, x0) + ax * at(m, y0 + 1, x0 + 1));
}

// Dense pyramidal Lucas-Kanade, every pixel, coarse to fine. Returns CV_32FC2 flow (prev -> curr).
}  // namespace

cv::Mat denseLk(const cv::Mat& prev8, const cv::Mat& curr8, const LkParams& p) {
  const auto P = pyramid(prev8, p.levels), C = pyramid(curr8, p.levels);
  const int r = p.win / 2;
  cv::Mat flow;
  for (int l = p.levels - 1; l >= 0; --l) {
    const cv::Mat &I = P[l], &J = C[l];
    cv::Mat f(I.size(), CV_32FC2, cv::Scalar(0, 0));
    if (!flow.empty())
      for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) f.at<cv::Vec2f>(y, x) = 2.0f * flow.at<cv::Vec2f>(std::min(y / 2, flow.rows - 1), std::min(x / 2, flow.cols - 1));
    cv::Mat ix(I.size(), CV_32F), iy(I.size(), CV_32F);
    for (int y = 0; y < I.rows; ++y)
      for (int x = 0; x < I.cols; ++x) {
        ix.at<float>(y, x) = 0.5f * (at(I, y, x + 1) - at(I, y, x - 1));
        iy.at<float>(y, x) = 0.5f * (at(I, y + 1, x) - at(I, y - 1, x));
      }
    for (int y = 0; y < I.rows; ++y)
      for (int x = 0; x < I.cols; ++x) {
        double gxx = 0, gxy = 0, gyy = 0;
        for (int v = -r; v <= r; ++v)
          for (int u = -r; u <= r; ++u) {
            const float gx = at(ix, y + v, x + u), gy = at(iy, y + v, x + u);
            gxx += gx * gx; gxy += gx * gy; gyy += gy * gy;
          }
        const double det = gxx * gyy - gxy * gxy;
        if (det < 1e-6) continue;
        cv::Vec2f d = f.at<cv::Vec2f>(y, x);
        for (int it = 0; it < p.iterations; ++it) {
          double bx = 0, by = 0;
          for (int v = -r; v <= r; ++v)
            for (int u = -r; u <= r; ++u) {
              const float e = at(I, y + v, x + u) - bilinear(J, y + v + d[1], x + u + d[0]);
              bx += e * at(ix, y + v, x + u);
              by += e * at(iy, y + v, x + u);
            }
          const float dx = static_cast<float>((gyy * bx - gxy * by) / det), dy = static_cast<float>((gxx * by - gxy * bx) / det);
          d[0] += dx; d[1] += dy;
          if (dx * dx + dy * dy < 1e-4f) break;
        }
        f.at<cv::Vec2f>(y, x) = d;
      }
    flow = f;
  }
  return flow;
}

FlowSource lkFlow(LkParams p) {
  return [p](const cv::Mat& prev, const cv::Mat& curr) { return denseLk(prev, curr, p); };
}

}  // namespace fcw
