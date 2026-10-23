// fcw_app: run the C++ FCW pipeline on a video file or camera.
//
//   fcw_app --source clip.mp4 --fx 500 --detections dets.yml.gz [--output out.mp4] [--max-frames N] [--no-display]
//
// On the PC, detections come from a file recorded by the Python detector:
//   python tools/export_golden.py --record-detections clip.mp4 --record-out dets.yml.gz
// On the KV260 the VART DPU runner replaces it (plan 2, Task 10).
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "fcw/runtime.hpp"

namespace {

struct Args {
  std::string source, detections, output;
  double fx = 500.0;
  int max_frames = 0;
  bool display = true;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--source") a.source = next();
    else if (k == "--fx") a.fx = std::stod(next());
    else if (k == "--detections") a.detections = next();
    else if (k == "--output") a.output = next();
    else if (k == "--max-frames") a.max_frames = std::stoi(next());
    else if (k == "--no-display") a.display = false;
    else throw std::runtime_error("unknown argument " + k);
  }
  if (a.source.empty()) throw std::runtime_error("--source is required");
  return a;
}

// Per-frame detections recorded by tools/export_golden.py --record-detections.
fcw::DetectorFn recordedDetector(const std::string& path) {
  if (path.empty()) {
    std::cerr << "warning: no --detections file; running without detections\n";
    return [](const cv::Mat&) { return std::vector<fcw::Detection>{}; };
  }
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
  try {
    const Args args = parse(argc, argv);
    const bool live = !args.source.empty() && std::isdigit(static_cast<unsigned char>(args.source[0])) && args.source.size() <= 2;
    auto cap = std::make_shared<cv::VideoCapture>();
    if (live) cap->open(std::stoi(args.source)); else cap->open(args.source);
    if (!cap->isOpened()) throw std::runtime_error("cannot open source " + args.source);
    const double fps = cap->get(cv::CAP_PROP_FPS) > 0 ? cap->get(cv::CAP_PROP_FPS) : 30.0;
    const auto start = std::chrono::steady_clock::now();
    auto count = std::make_shared<int>(0);
    fcw::CaptureSource source = [=]() -> std::optional<fcw::CapturedFrame> {
      if (args.max_frames && *count >= args.max_frames) return std::nullopt;
      cv::Mat frame;
      if (!cap->read(frame)) return std::nullopt;
      const double t = live ? std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() : *count / fps;
      ++*count;
      return fcw::CapturedFrame{frame, t};
    };

    fcw::FcwConfig cfg;
    cfg.fx = args.fx;
    const int src_w = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_WIDTH)), src_h = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_HEIGHT));
    if (src_w > 0 && src_h > 0) {  // the source's aspect, square source pixels (see config_for_source in pipeline.py)
      const cv::Size size = fcw::processingSize(src_w, src_h);
      cfg.frame_width = size.width;
      cfg.frame_height = size.height;
      cfg.fy = args.fx * (static_cast<double>(size.height) / src_h) / (static_cast<double>(size.width) / src_w);
    }
    std::vector<fcw::Level> levels;
    std::vector<double> totals;
    std::map<std::string, double> stage_sum;
    cv::VideoWriter writer;
    fcw::RuntimeOptions opt;
    opt.live = live;
    fcw::Runtime runtime(cfg, opt, source, recordedDetector(args.detections), fcw::disFlow(), [&](const fcw::FcwFrame& f) {
      levels.push_back(f.level());
      totals.push_back(f.timings_ms.at("total"));
      if (levels.size() > 1)
        for (const auto& [k, v] : f.timings_ms) stage_sum[k] += v;  // first frame excluded, like the summary
      if (args.output.empty() && !args.display) return;
      const cv::Mat img = fcw::drawOverlay(f, cfg.fx);
      if (!args.output.empty()) {
        if (!writer.isOpened()) writer.open(args.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, img.size());
        writer.write(img);
      }
      if (args.display) {
        cv::imshow("Forward collision warning", img);
        cv::waitKey(1);
      }
    });
    const fcw::RuntimeStats stats = runtime.run();
    std::cout << fcw::summaryLine(levels, totals) << " | dropped " << stats.dropped << "\n";
    if (levels.size() > 1) {
      std::cout << "mean ms/frame by step:";
      for (const auto& [k, v] : stage_sum) std::printf(" %s=%.2f", k.c_str(), v / (levels.size() - 1));
      std::cout << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fcw_app: " << e.what() << "\n";
    return 2;
  }
}
