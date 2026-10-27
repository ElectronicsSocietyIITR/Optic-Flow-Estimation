// fcw_app: run the C++ FCW pipeline on a video file or camera.
//
//   fcw_app --source clip.mp4 --fx 500 --detections dets.yml.gz [--output out.mp4] [--max-frames N] [--no-display]
//
// On the PC, detections come from a file recorded by the Python detector:
//   python tools/export_golden.py --record-detections clip.mp4 --record-out dets.yml.gz
// On the KV260 (built with -DFCW_BOARD=ON): --board --xclbin fcw.xclbin --xmodel yolov9t.xmodel; capture by
// GStreamer, preprocessing / LK (and the Horn engine, when the overlay has it) in the fabric, YOLO on the DPU.
// Output: full-screen DRM display (the desktop switched off; --card picks the DRM card, default the one with a
// monitor), or --window for a window on the desktop; --output records the annotated video (plan 2, Task 10).
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
#ifdef FCW_BOARD
#include "fcw/board.hpp"
#include "fcw/overlay.hpp"
#endif

namespace {

struct Args {
  std::string source, detections, output, xclbin, xmodel, card;
  bool board = false, window = false;
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
    else if (k == "--board") a.board = true;
    else if (k == "--xclbin") a.xclbin = next();
    else if (k == "--xmodel") a.xmodel = next();
    else if (k == "--card") a.card = next();
    else if (k == "--window") a.window = true;
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

#ifdef FCW_BOARD
// Board: GStreamer -> preprocess kernel -> LK kernel || DPU -> frame stage -> object stage (PL engine) -> DRM.
int runBoard(const Args& args) {
  auto camera = fcw::board::gstCapture(args.source);
  auto first = camera();
  if (!first) throw std::runtime_error("no frames from " + args.source);
  const cv::Size src = first->bgr.size(), proc = fcw::processingSize(src.width, src.height);
  fcw::FcwConfig cfg;
  cfg.fx = args.fx;
  cfg.frame_width = proc.width;
  cfg.frame_height = proc.height;
  cfg.fy = args.fx * (static_cast<double>(proc.height) / src.height) / (static_cast<double>(proc.width) / src.width);
  fcw::board::XrtAccelerator accel(args.xclbin, args.xmodel, cfg, src);
  std::unique_ptr<fcw::board::DrmDisplay> display;
  if (args.display && !args.window) display = std::make_unique<fcw::board::DrmDisplay>(args.card);
  cv::VideoWriter writer;
  int count = 0;
  fcw::CaptureSource source = [&]() -> std::optional<fcw::CapturedFrame> {
    if (args.max_frames && count >= args.max_frames) return std::nullopt;
    ++count;
    if (first) return std::exchange(first, std::nullopt);
    return camera();
  };
  fcw::RuntimeOptions opt;
  opt.live = args.source.rfind("/dev/video", 0) == 0;
  std::vector<fcw::Level> levels;
  std::vector<double> totals;
  fcw::Runtime runtime(cfg, opt, source, [&](const fcw::CapturedFrame& f) { return accel(f); }, [&](const fcw::FcwFrame& f) {
    levels.push_back(f.level());
    totals.push_back(f.timings_ms.at("total"));
    if (display) display->show(f, proc);
    if (!args.window && args.output.empty()) return;
    const cv::Mat img = fcw::composeOverlay(f.frame, f, proc);  // costs Arm time: for testing, not the 30 FPS path
    if (!args.output.empty()) {
      // 30 FPS nominal: the recording holds every processed frame (a live camera drops some under load)
      if (!writer.isOpened()) writer.open(args.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, img.size());
      writer.write(img);
    }
    if (args.window) {
      cv::imshow("Forward collision warning", img);
      cv::waitKey(1);
    }
  });
  runtime.setMomentEngine(accel.momentEngine());
  const auto start = std::chrono::steady_clock::now();
  const fcw::RuntimeStats stats = runtime.run();
  const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << fcw::summaryLine(levels, totals) << " | dropped " << stats.dropped << "\n";
  std::printf("throughput %.1f FPS (%d frames in %.2f s wall)\n", stats.processed / wall_s, stats.processed, wall_s);
  return 0;
}
#endif

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    if (args.board) {
#ifdef FCW_BOARD
      return runBoard(args);
#else
      throw std::runtime_error("--board needs a build with -DFCW_BOARD=ON (on the KV260)");
#endif
    }
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
    const auto wall_start = std::chrono::steady_clock::now();
    const fcw::RuntimeStats stats = runtime.run();
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    std::cout << fcw::summaryLine(levels, totals) << " | dropped " << stats.dropped << "\n";
    if (levels.size() > 1) {
      std::cout << "mean ms/frame by step:";
      for (const auto& [k, v] : stage_sum) std::printf(" %s=%.2f", k.c_str(), v / (levels.size() - 1));
      std::cout << "\n";
    }
    // Stages overlap (capture, flow || detection, frame stage, object stage, render): throughput is set by
    // the slowest stage, not the sum of the steps above; "total" is a frame's time through perception.
    std::printf("throughput %.1f FPS (%d frames in %.2f s wall, %d object workers)\n", stats.processed / wall_s, stats.processed,
                wall_s, cfg.object_workers);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fcw_app: " << e.what() << "\n";
    return 2;
  }
}
