// Vitis Vision densePyrOpticalFlow in C-simulation: the board's exact flow.
//   lk_ref prev.png curr.png [flow.bin] [--compare]
// flow.bin: float32, H x W x 2, px. --compare (or no flow.bin) also runs fcw::denseLk and prints
// how far it is from the kernel's flow.
//
// Drives AMD's kernels (pyrDown + one LK pass per call) as their L1 testbench does: 5 levels x
// 5 iterations, flow ping-ponged between calls. Flow format (from AMD's source): 32 bits per
// pixel, u in the upper 16, v in the lower 16, each ap_fixed<16,10> = 1/64 px, the engine's format.
#include "../common/fcw_pyr_dense_optical_flow.hpp"  // the board's LK (FCW fork), not the library's
#include "xf_config_params.h"
#ifdef LK_LINES  // the remap line buffer, as hls/lk_wrapper builds it (-DLK_LINES)
#undef NUM_LINES_FINDIT
#define NUM_LINES_FINDIT LK_LINES
#endif
#include "xf_pyr_dense_optical_flow_accel.cpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "fcw/lk_flow.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: lk_ref prev.png curr.png [flow.bin] [--compare]\n");
    return 2;
  }
  const cv::Mat im0 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE), im1 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  if (im0.empty() || im1.empty() || im0.size() != im1.size() || im0.cols > WIDTH || im0.rows > HEIGHT) {
    std::fprintf(stderr, "bad images (max %dx%d)\n", WIDTH, HEIGHT);
    return 1;
  }

  static xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPPCX> pyr1[NUM_LEVELS], pyr2[NUM_LEVELS];
  static xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPPCX> flow, flow_iter;
  int pyr_h[NUM_LEVELS], pyr_w[NUM_LEVELS];
  pyr_h[0] = im0.rows;
  pyr_w[0] = im0.cols;
  for (int l = 1; l < NUM_LEVELS; ++l) {
    pyr_h[l] = (pyr_h[l - 1] + 1) >> 1;
    pyr_w[l] = (pyr_w[l - 1] + 1) >> 1;
  }
  for (int l = 0; l < NUM_LEVELS; ++l) {
    pyr1[l].init(HEIGHT, WIDTH);
    pyr2[l].init(HEIGHT, WIDTH);
    pyr1[l].rows = pyr2[l].rows = pyr_h[l];
    pyr1[l].cols = pyr2[l].cols = pyr_w[l];
    pyr1[l].size = pyr2[l].size = pyr_h[l] * pyr_w[l];
  }
  flow.init(HEIGHT, WIDTH);
  flow_iter.init(HEIGHT, WIDTH);
  const cv::Mat zeros = cv::Mat::zeros(HEIGHT, WIDTH, CV_32SC1);
  flow.copyTo((XF_PTSNAME(OUT_TYPE, NPPCX)*)zeros.data);
  flow_iter.copyTo((XF_PTSNAME(OUT_TYPE, NPPCX)*)zeros.data);
  for (int i = 0; i < pyr_h[0]; ++i)
    for (int j = 0; j < pyr_w[0]; ++j) {
      pyr1[0].write(i * pyr_w[0] + j, im0.at<uchar>(i, j));
      pyr2[0].write(i * pyr_w[0] + j, im1.at<uchar>(i, j));
    }
  for (int l = 0; l < NUM_LEVELS - 1; ++l) {
    pyr_dense_optical_flow_pyr_down_accel((ap_uint<INPUT_PTR_WIDTH>*)pyr1[l].data, (ap_uint<OUTPUT_PTR_WIDTH>*)pyr1[l + 1].data,
                                          pyr_h[l], pyr_w[l], pyr_h[l + 1], pyr_w[l + 1]);
    pyr_dense_optical_flow_pyr_down_accel((ap_uint<INPUT_PTR_WIDTH>*)pyr2[l].data, (ap_uint<OUTPUT_PTR_WIDTH>*)pyr2[l + 1].data,
                                          pyr_h[l], pyr_w[l], pyr_h[l + 1], pyr_w[l + 1]);
  }
  bool flowin = true;
  flow.rows = flow_iter.rows = pyr_h[NUM_LEVELS - 1];
  flow.cols = flow_iter.cols = pyr_w[NUM_LEVELS - 1];
  for (int l = NUM_LEVELS - 1; l >= 0; --l) {
    for (int it = 0; it < NUM_ITERATIONS; ++it) {
      const bool scale_up = it == 0 && l != NUM_LEVELS - 1;
      const int next_h = scale_up ? pyr_h[l + 1] : pyr_h[l];
      const float scale_in = (next_h - 1) * 1.0f / (pyr_h[l] - 1);
      const int init = (it == 0 && l == NUM_LEVELS - 1) ? 1 : 0;
      auto& in = flowin ? flow_iter : flow;
      auto& out = flowin ? flow : flow_iter;
      out.rows = pyr_h[l];
      out.cols = pyr_w[l];
      out.size = pyr_h[l] * pyr_w[l];
      pyr_dense_optical_flow_accel((ap_uint<INPUT_PTR_WIDTH>*)pyr1[l].data, (ap_uint<INPUT_PTR_WIDTH>*)pyr2[l].data,
                                   (ap_uint<OUTPUT_PTR_WIDTH>*)in.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out.data, l, scale_up, scale_in,
                                   init, pyr_h[l], pyr_w[l], pyr_h[l], pyr_w[l], in.rows, in.cols, out.rows, out.cols);
      flowin = !flowin;
    }
  }
  auto& result = flowin ? flow_iter : flow;
  cv::Mat vitis(im0.size(), CV_32FC2);
  for (int i = 0; i < pyr_h[0]; ++i)
    for (int j = 0; j < pyr_w[0]; ++j) {
      const unsigned int packed = result.read(i * pyr_w[0] + j);
      const short u = static_cast<short>(packed >> 16), v = static_cast<short>(packed & 0xFFFF);
      vitis.at<cv::Vec2f>(i, j) = cv::Vec2f(u / 64.0f, v / 64.0f);  // ap_fixed<16,10>: 6 fraction bits
    }
  const bool has_out = argc > 3 && std::string(argv[3]) != "--compare";
  if (has_out) std::ofstream(argv[3], std::ios::binary).write(reinterpret_cast<const char*>(vitis.data), vitis.total() * 8);

  if (!has_out || std::string(argv[argc - 1]) == "--compare") {
    auto pct = [](std::vector<float> v, double q) {
      std::nth_element(v.begin(), v.begin() + static_cast<long>(q * (v.size() - 1)), v.end());
      return v[static_cast<long>(q * (v.size() - 1))];
    };
    const cv::Mat ours = fcw::denseLk(im0, im1);
    std::vector<float> diff, mag;
    for (int i = 8; i < im0.rows - 8; ++i)
      for (int j = 8; j < im0.cols - 8; ++j) {
        diff.push_back(static_cast<float>(cv::norm(ours.at<cv::Vec2f>(i, j) - vitis.at<cv::Vec2f>(i, j))));
        mag.push_back(static_cast<float>(cv::norm(vitis.at<cv::Vec2f>(i, j))));
      }
    std::printf("%dx%d  vitis |flow| median %.3f px  |fcw::denseLk - vitis| median %.3f p75 %.3f p90 %.3f px\n", im0.cols, im0.rows,
                pct(mag, 0.5), pct(diff, 0.5), pct(diff, 0.75), pct(diff, 0.9));
  }
  return 0;
}
