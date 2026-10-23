// C-simulation of the preprocessing kernel against OpenCV.
//   tb_preprocess [dashcam_frame.png]
// 1. Scale 1 (source already 512x288): gray must equal cv::cvtColor and the tensor the reference
//    quantiser, bit for bit.
// 2. Pyramid: Vitis pyrDown vs cv::pyrDown on the same gray (issue 3: Horn's software pyramid).
// 3. A 1920x1080 frame: Vitis bilinear resize vs cv::resize(INTER_LINEAR), reported.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <vector>

#include "../src/preprocess.hpp"

namespace {

struct Out {
  cv::Mat tensor, gray0, gray1, gray2;
};

Out run(const cv::Mat& bgr, cv::Size dst, int shift) {
  std::vector<ap_uint<32>> in(bgr.total());
  for (int i = 0; i < bgr.rows; ++i)
    for (int j = 0; j < bgr.cols; ++j) {
      const cv::Vec3b p = bgr.at<cv::Vec3b>(i, j);
      in[i * bgr.cols + j] = ap_uint<32>(p[0]) | (ap_uint<32>(p[1]) << 8) | (ap_uint<32>(p[2]) << 16);
    }
  const int r1 = (dst.height + 1) / 2, c1 = (dst.width + 1) / 2, r2 = (r1 + 1) / 2, c2 = (c1 + 1) / 2;
  std::vector<ap_int<8>> tensor(dst.area() * 3);
  std::vector<ap_uint<8>> g0(dst.area()), g1(r1 * c1), g2(r2 * c2);
  preprocess(in.data(), tensor.data(), g0.data(), g1.data(), g2.data(), bgr.rows, bgr.cols, dst.height, dst.width, shift);
  Out o{cv::Mat(dst.height, dst.width, CV_8SC3), cv::Mat(dst, CV_8U), cv::Mat(r1, c1, CV_8U), cv::Mat(r2, c2, CV_8U)};
  for (int i = 0; i < dst.area() * 3; ++i) o.tensor.ptr<int8_t>()[i] = tensor[i].to_int();
  for (int i = 0; i < dst.area(); ++i) o.gray0.data[i] = g0[i].to_uint();
  for (int i = 0; i < r1 * c1; ++i) o.gray1.data[i] = g1[i].to_uint();
  for (int i = 0; i < r2 * c2; ++i) o.gray2.data[i] = g2[i].to_uint();
  return o;
}

cv::Mat quantise(const cv::Mat& bgr, int shift) {  // reference: RGB, round half up of v * 2^shift / 255, max 127
  cv::Mat q(bgr.size(), CV_8SC3);
  for (int i = 0; i < bgr.rows; ++i)
    for (int j = 0; j < bgr.cols; ++j) {
      const cv::Vec3b p = bgr.at<cv::Vec3b>(i, j);
      const int rgb[3] = {p[2], p[1], p[0]};
      for (int c = 0; c < 3; ++c) q.at<cv::Vec<int8_t, 3>>(i, j)[c] = static_cast<int8_t>(std::min(127, (rgb[c] * (1 << shift) * 2 + 255) / 510));
    }
  return q;
}

double maxAbs(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat d;
  cv::absdiff(a, b, d);
  double mx;
  cv::minMaxLoc(d.reshape(1), nullptr, &mx);
  return mx;
}

}  // namespace

int main(int argc, char** argv) {
  int failed = 0;
  cv::Mat frame;
  if (argc > 1) frame = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (frame.empty()) {
    frame = cv::Mat(1080, 1920, CV_8UC3);
    cv::randu(frame, 0, 255);
    cv::GaussianBlur(frame, frame, cv::Size(0, 0), 3.0);
  }
  cv::resize(frame, frame, cv::Size(1920, 1080), 0, 0, cv::INTER_AREA);

  // 1. Scale 1: gray and tensor exact.
  cv::Mat small;
  cv::resize(frame, small, cv::Size(512, 288), 0, 0, cv::INTER_AREA);
  const Out same = run(small, small.size(), 7);
  cv::Mat ref_gray(small.size(), CV_8U), cvt;
  for (int i = 0; i < small.rows; ++i)  // the pipeline's bgr_to_gray / fcw::bgrToGray
    for (int j = 0; j < small.cols; ++j) {
      const cv::Vec3b p = small.at<cv::Vec3b>(i, j);
      ref_gray.at<uchar>(i, j) = static_cast<uchar>((4899 * p[2] + 9617 * p[1] + 1868 * p[0] + 8192) >> 14);
    }
  cv::cvtColor(small, cvt, cv::COLOR_BGR2GRAY);
  const double dg = maxAbs(same.gray0, ref_gray), dt = maxAbs(same.tensor, quantise(small, 7));
  std::printf("scale 1   gray vs bgr_to_gray max |d| %.0f   tensor vs reference max |d| %.0f   %s   (cv::cvtColor: max |d| %.0f)\n", dg, dt,
              dg == 0 && dt == 0 ? "exact" : "FAIL", maxAbs(same.gray0, cvt));
  failed += !(dg == 0 && dt == 0);

  // 2. Pyramid: the kernel's levels vs cv::pyrDown of its own gray, replicated borders (the software's).
  cv::Mat ref1, ref2;
  cv::pyrDown(same.gray0, ref1, cv::Size(), cv::BORDER_REPLICATE);
  cv::pyrDown(same.gray1, ref2, cv::Size(), cv::BORDER_REPLICATE);
  const double d1 = maxAbs(same.gray1, ref1), d2 = maxAbs(same.gray2, ref2);
  std::printf("pyramid   levels 1, 2 vs cv::pyrDown(BORDER_REPLICATE) max |d| %.0f, %.0f   %s\n", d1, d2, d1 == 0 && d2 == 0 ? "exact" : "FAIL");
  failed += !(d1 == 0 && d2 == 0);
  cv::Mat def;
  cv::pyrDown(same.gray0, def);  // OpenCV's default border (REFLECT_101): differs on the outer rows and columns only
  std::printf("          (vs cv::pyrDown default border: max |d| %.0f)\n", maxAbs(same.gray1, def));

  // 3. 1080p -> 512x288: Vitis bilinear vs cv::resize(INTER_LINEAR), then gray.
  const Out big = run(frame, cv::Size(512, 288), 7);
  cv::Mat lin, lin_gray;
  cv::resize(frame, lin, cv::Size(512, 288), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(lin, lin_gray, cv::COLOR_BGR2GRAY);
  cv::Mat d;
  cv::absdiff(big.gray0, lin_gray, d);
  std::printf("resize    1920x1080 -> 512x288 gray vs cv::resize(INTER_LINEAR) max |d| %.0f   mean |d| %.3f\n", maxAbs(big.gray0, lin_gray),
              cv::mean(d)[0]);
  std::printf(failed ? "C-SIM FAILED\n" : "C-SIM PASSED\n");
  return failed;
}
