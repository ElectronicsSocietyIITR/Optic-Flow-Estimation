#include "fcw/runtime.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <future>
#include <numeric>
#include <thread>

namespace fcw {

Accelerator pcAccelerator(const FcwConfig& cfg, DetectorFn detector, FlowSource flow) {
  auto prev_gray = std::make_shared<cv::Mat>();
  const cv::Size size(cfg.frame_width, cfg.frame_height);
  return [=](const CapturedFrame& f) {
    AcceleratedFrame a;
    cv::resize(f.bgr, a.bgr, size);
    a.t = f.t;
    const cv::Mat gray = bgrToGray(a.bgr);
    // Flow and detection are independent (the fabric's LK and the DPU on the board): run them together.
    auto dets = std::async(std::launch::async, [&] { return detector(a.bgr); });
    if (!prev_gray->empty()) a.flow = flow(*prev_gray, gray);
    a.detections = dets.get();
    *prev_gray = gray;
    return a;  // gray stays empty: the frame stage makes it (same bytes)
  };
}

Runtime::Runtime(FcwConfig cfg, RuntimeOptions opt, CaptureSource source, DetectorFn detector, FlowSource flow, Sink sink)
    : cfg_(std::move(cfg)), opt_(opt), source_(std::move(source)), accel_(pcAccelerator(cfg_, std::move(detector), std::move(flow))),
      sink_(std::move(sink)) {}

Runtime::Runtime(FcwConfig cfg, RuntimeOptions opt, CaptureSource source, Accelerator accelerator, Sink sink)
    : cfg_(std::move(cfg)), opt_(opt), source_(std::move(source)), accel_(std::move(accelerator)), sink_(std::move(sink)) {}

RuntimeStats Runtime::run() {
  BoundedQueue<CapturedFrame> captured(opt_.capture_queue);
  BoundedQueue<AcceleratedFrame> work(opt_.work_queue);
  BoundedQueue<FrameState> perceived(opt_.perception_queue);
  BoundedQueue<FcwFrame> rendered(opt_.render_queue);
  RuntimeStats stats;

  std::thread t1([&] {  // capture
    while (auto f = source_()) {
      ++stats.captured;
      if (opt_.live) {
        if (captured.pushDropOldest(std::move(*f))) ++stats.dropped;
      } else {
        captured.pushBlocking(std::move(*f));
      }
    }
    captured.close();
  });

  std::thread t2([&] {  // accelerators: PC (resize, flow || detection) or board (preprocess, LK || DPU)
    while (auto f = captured.pop()) work.pushBlocking(accel_(*f));
    work.close();
  });

  // Perception in two stages on one pipeline: the frame stage of N+1 overlaps the object stage of N
  // (they touch disjoint state; see FcwPipeline::frameStage).
  const AcceleratedFrame* current = nullptr;
  FcwPipeline pipe(
      cfg_, [&](const cv::Mat&) { return current->detections; }, [&](const cv::Mat&, const cv::Mat&) { return current->flow; });
  if (engine_) pipe.setMomentEngine(std::move(engine_));
  std::thread t3a([&] {  // frame stage: tracker || ego-rotation
    while (auto w = work.pop()) {
      current = &*w;
      perceived.pushBlocking(w->gray.empty() ? pipe.frameStage(w->bgr, w->t) : pipe.frameStage(w->bgr, w->gray, w->t));
    }
    perceived.close();
  });
  std::thread t3b([&] {  // object stage: per-object TTC on the worker pool, lead selection, warnings
    while (auto s = perceived.pop()) {
      rendered.pushBlocking(pipe.objectStage(std::move(*s)));
      ++stats.processed;
    }
    rendered.close();
  });

  while (auto f = rendered.pop()) sink_(*f);  // T4: render / output on the calling thread
  t1.join();
  t2.join();
  t3a.join();
  t3b.join();
  stats.max_capture_queue = static_cast<int>(captured.maxSize());
  return stats;
}

int warningRaises(const std::vector<Level>& levels) {
  int raises = 0;
  for (size_t i = 0; i < levels.size(); ++i) {
    const bool now = levels[i] >= Level::Warning;
    const bool before = i > 0 && levels[i - 1] >= Level::Warning;
    raises += now && !before;
  }
  return raises;
}

std::string summaryLine(const std::vector<Level>& levels, const std::vector<double>& total_ms) {
  const double mean = total_ms.size() > 1
                          ? std::accumulate(total_ms.begin() + 1, total_ms.end(), 0.0) / static_cast<double>(total_ms.size() - 1)
                          : 0.0;
  int critical = 0;
  for (Level l : levels) critical += l == Level::Critical;
  char buf[160];
  std::snprintf(buf, sizeof(buf), "frames %zu | mean %.1f ms/frame | warnings raised %d | critical frames %d", levels.size(), mean,
                warningRaises(levels), critical);
  return buf;
}

cv::Mat drawOverlay(const FcwFrame& r, double fx) {
  static const cv::Scalar kNone(0, 200, 0), kWarn(0, 165, 255), kCrit(0, 0, 255), kOff(160, 160, 160), kCyan(255, 255, 0);
  constexpr double kCameraHeightM = 1.3, kHalfCorridorM = 1.05;  // drawing only
  cv::Mat img = r.frame.clone();
  const int h = img.rows, w = img.cols;
  const cv::Point heading(static_cast<int>(r.heading.x), static_cast<int>(r.heading.y));
  if (h - 1 > r.heading.y) {
    const double z_bottom = fx * kCameraHeightM / (h - 1 - r.heading.y);
    const double dx = fx * kHalfCorridorM / z_bottom;
    for (int side : {-1, 1}) cv::line(img, heading, {static_cast<int>(r.heading.x + side * dx), h - 1}, kCyan, 1, cv::LINE_AA);
  }
  cv::drawMarker(img, heading, kCyan, cv::MARKER_CROSS, 12, 2);
  for (const auto& o : r.objects) {
    const bool on = o.course && o.course->on_course;
    const cv::Scalar color = (on || o.level > Level::None) ? (o.level == Level::Critical ? kCrit : o.level == Level::Warning ? kWarn : kNone) : kOff;
    cv::rectangle(img, {static_cast<int>(o.bbox.x1), static_cast<int>(o.bbox.y1)}, {static_cast<int>(o.bbox.x2), static_cast<int>(o.bbox.y2)},
                  color, o.level > Level::None ? 3 : 1);
    std::string label = o.cls;
    if (o.estimate && o.estimate->ttc_s) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), " %.1fs", *o.estimate->ttc_s);
      label += buf;
    }
    cv::putText(img, label, {static_cast<int>(o.bbox.x1), std::max(12, static_cast<int>(o.bbox.y1) - 4)}, cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
  }
  if (r.level() > Level::None)
    cv::putText(img, r.level() == Level::Critical ? "BRAKE" : "COLLISION WARNING", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                r.level() == Level::Critical ? kCrit : kWarn, 2, cv::LINE_AA);
  const auto it = r.timings_ms.find("total");
  if (it != r.timings_ms.end())
    cv::putText(img, std::to_string(static_cast<int>(it->second)) + " ms", {w - 70, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 255}, 1, cv::LINE_AA);
  return img;
}

}  // namespace fcw
