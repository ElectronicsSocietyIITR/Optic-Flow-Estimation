// CPU-only baseline: the FCW system as plain native C++ on one core, with no FPGA logic and no DPU.
//
// Times the work the KV260 design moves to the fabric (preprocessing, dense pyramidal LK flow,
// Horn sums inside the "horn" step) next to the per-object stages that stay on the Arm cores in
// both designs. The LK here is a straightforward implementation (plain loops, no SIMD intrinsics,
// no threads) with the parameters of the Vitis Vision kernel; OpenCV's optimised sparse LK on
// every pixel is timed as a reference. YOLO on the CPU is timed by eval/bench_cpu_yolo.py.
//
//   fcw_cpu_baseline --source clip.mp4 --detections dets.yml.gz --fx 200 --frames 30
//   fcw_cpu_baseline --selftest        (LK accuracy on a known sub-pixel shift)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "fcw/lk_flow.hpp"
#include "fcw/pipeline.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// OpenCV's (SIMD, optimised) sparse pyramidal LK evaluated on every pixel: the best a CPU library does.
cv::Mat opencvDenseLk(const cv::Mat& prev8, const cv::Mat& curr8, const fcw::LkParams& p) {
  std::vector<cv::Point2f> pts;
  pts.reserve(prev8.total());
  for (int y = 0; y < prev8.rows; ++y)
    for (int x = 0; x < prev8.cols; ++x) pts.emplace_back(static_cast<float>(x), static_cast<float>(y));
  std::vector<cv::Point2f> next;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev8, curr8, pts, next, status, err, cv::Size(p.win, p.win), p.levels - 1,
                           cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, p.iterations, 0.01));
  cv::Mat flow(prev8.size(), CV_32FC2);
  for (size_t i = 0; i < pts.size(); ++i) flow.at<cv::Vec2f>(pts[i]) = next[i] - pts[i];
  return flow;
}

int selftest() {
  cv::Mat tex(384, 512, CV_8U);
  cv::randu(tex, 0, 255);
  cv::GaussianBlur(tex, tex, cv::Size(0, 0), 2.0);
  const cv::Matx23d shift(1, 0, 1.6, 0, 1, -0.7);
  cv::Mat moved;
  cv::warpAffine(tex, moved, shift, tex.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
  const cv::Mat flow = fcw::denseLk(tex, moved, {});
  std::vector<float> err;
  for (int y = 32; y < 352; ++y)
    for (int x = 32; x < 480; ++x) {
      const cv::Vec2f f = flow.at<cv::Vec2f>(y, x);
      err.push_back(std::hypot(f[0] - 1.6f, f[1] + 0.7f));
    }
  std::nth_element(err.begin(), err.begin() + err.size() / 2, err.end());
  const float median = err[err.size() / 2];
  std::printf("selftest: shift (1.6, -0.7) px, median endpoint error %.3f px -> %s\n", median, median < 0.05f ? "ok" : "FAIL");
  return median < 0.05f ? 0 : 1;
}

fcw::DetectorFn recorded(const std::string& path) {
  auto fs = std::make_shared<cv::FileStorage>(path, cv::FileStorage::READ);
  if (!fs->isOpened()) throw std::runtime_error("cannot open " + path);
  auto index = std::make_shared<int>(0);
  return [fs, index](const cv::Mat&) {
    const std::string k = "f" + std::to_string((*index)++);
    std::vector<fcw::Detection> out;
    cv::Mat boxes;
    (*fs)[k + "_dets"] >> boxes;
    std::stringstream names(static_cast<std::string>((*fs)[k + "_cls"]));
    for (int r = 0; r < boxes.rows; ++r) {
      std::string cls;
      std::getline(names, cls, ',');
      out.push_back({{boxes.at<double>(r, 0), boxes.at<double>(r, 1), boxes.at<double>(r, 2), boxes.at<double>(r, 3)}, cls, 1.0});
    }
    return out;
  };
}

}  // namespace

int main(int argc, char** argv) {
  std::string source, detections;
  double fx = 500;
  int frames = 30;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--selftest") return selftest();
    if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; return 2; }
    if (k == "--source") source = argv[++i];
    else if (k == "--detections") detections = argv[++i];
    else if (k == "--fx") fx = std::stod(argv[++i]);
    else if (k == "--frames") frames = std::stoi(argv[++i]);
    else { std::cerr << "unknown argument " << k << "\n"; return 2; }
  }
  if (source.empty() || detections.empty()) { std::cerr << "--source and --detections are required\n"; return 2; }
  cv::setNumThreads(1);  // one core, like one Arm core

  cv::VideoCapture cap(source);
  if (!cap.isOpened()) { std::cerr << "cannot open " << source << "\n"; return 1; }
  const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
  const fcw::LkParams lk;
  std::map<std::string, double> sum;
  cv::Mat flow_now, prev_gray;
  fcw::FcwConfig cfg;
  cfg.fx = fx;
  fcw::FcwPipeline pipe(cfg, recorded(detections), [&](const cv::Mat&, const cv::Mat&) { return flow_now; });
  int timed = 0;
  for (int i = 0; i < frames; ++i) {
    cv::Mat raw;
    if (!cap.read(raw)) break;
    auto t0 = Clock::now();
    cv::Mat bgr, gray, rgb, tensor;
    cv::resize(raw, bgr, cv::Size(512, 384), 0, 0, cv::INTER_AREA);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(tensor, CV_32F, 1.0 / 255);  // the detector's input tensor
    const double pre = msSince(t0);
    double naive = 0, ocv = 0;
    if (!prev_gray.empty()) {
      t0 = Clock::now();
      flow_now = fcw::denseLk(prev_gray, gray, lk);
      naive = msSince(t0);
      t0 = Clock::now();
      opencvDenseLk(prev_gray, gray, lk);
      ocv = msSince(t0);
    }
    const fcw::FcwFrame r = pipe.process(bgr, i / fps);
    prev_gray = gray;
    if (i == 0) continue;  // no flow on the first frame
    ++timed;
    sum["preprocess"] += pre;
    sum["lk_naive"] += naive;
    sum["lk_opencv"] += ocv;
    for (const auto& [k, v] : r.timings_ms) sum["pipeline_" + k] += v;
    std::fprintf(stderr, "frame %d: naive LK %.0f ms, OpenCV LK %.0f ms, pipeline %.1f ms\n", i, naive, ocv, r.timings_ms.at("total"));
  }
  std::printf("frames timed: %d (single thread)\n", timed);
  for (const auto& [k, v] : sum) std::printf("%-26s %10.2f ms/frame\n", k.c_str(), v / std::max(timed, 1));
  return 0;
}
