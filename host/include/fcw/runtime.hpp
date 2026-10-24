// Runtime (spec section 5): T1 capture -> T2 accelerators (flow || detection) -> T3a frame stage
// -> T3b object stage (worker pool) -> T4 render, on bounded queues.
// Frames are dropped only at T1 (oldest first, live sources only): T2 computes flow between
// consecutive frames, so a drop after T2 would pair flow with the wrong previous frame.
#pragma once

#include <opencv2/core.hpp>

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "fcw/pipeline.hpp"

namespace fcw {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

  // Returns true when the oldest item had to be dropped to make room.
  bool pushDropOldest(T item) {
    std::lock_guard<std::mutex> lock(m_);
    bool dropped = false;
    if (items_.size() >= capacity_) { items_.pop_front(); dropped = true; }
    items_.push_back(std::move(item));
    max_size_ = std::max(max_size_, items_.size());
    cv_.notify_all();
    return dropped;
  }

  void pushBlocking(T item) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&] { return items_.size() < capacity_ || closed_; });
    if (closed_) return;
    items_.push_back(std::move(item));
    max_size_ = std::max(max_size_, items_.size());
    cv_.notify_all();
  }

  // Blocks until an item arrives; empty once closed and drained.
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&] { return !items_.empty() || closed_; });
    if (items_.empty()) return std::nullopt;
    T item = std::move(items_.front());
    items_.pop_front();
    cv_.notify_all();
    return item;
  }

  void close() {
    std::lock_guard<std::mutex> lock(m_);
    closed_ = true;
    cv_.notify_all();
  }

  size_t maxSize() const {
    std::lock_guard<std::mutex> lock(m_);
    return max_size_;
  }

 private:
  size_t capacity_;
  size_t max_size_ = 0;
  bool closed_ = false;
  std::deque<T> items_;
  mutable std::mutex m_;
  std::condition_variable cv_;
};

struct CapturedFrame {
  cv::Mat bgr;
  double t = 0;  // seconds
};

using CaptureSource = std::function<std::optional<CapturedFrame>()>;  // empty: end of stream

// What the accelerator stage (T2) makes of one frame: the frame to show, flow and detections, and the
// gray image when the accelerators already made it (the board's preprocess kernel; empty on the PC).
struct AcceleratedFrame {
  cv::Mat bgr;
  double t = 0;
  cv::Mat gray;
  cv::Mat flow;
  std::vector<Detection> detections;
};
using Accelerator = std::function<AcceleratedFrame(const CapturedFrame&)>;  // called in order, one frame at a time
// The PC's T2: resize to the processing size, then flow (from the previous frame) || detection.
Accelerator pcAccelerator(const FcwConfig& cfg, DetectorFn detector, FlowSource flow);
using Sink = std::function<void(const FcwFrame&)>;

struct RuntimeOptions {
  bool live = false;      // live camera: drop oldest at T1; file: never drop
  size_t capture_queue = 2;
  size_t work_queue = 2;
  size_t perception_queue = 2;  // frame stage -> object stage
  size_t render_queue = 4;
};

struct RuntimeStats {
  int captured = 0;
  int processed = 0;
  int dropped = 0;
  int max_capture_queue = 0;
};

class Runtime {
 public:
  Runtime(FcwConfig cfg, RuntimeOptions opt, CaptureSource source, DetectorFn detector, FlowSource flow, Sink sink);
  Runtime(FcwConfig cfg, RuntimeOptions opt, CaptureSource source, Accelerator accelerator, Sink sink);
  RuntimeStats run();  // returns when the source is exhausted and every frame has been rendered
  // The Horn / moment engine for the object stage (default: software; on the board the PL engine).
  void setMomentEngine(std::unique_ptr<MomentEngine> engine) { engine_ = std::move(engine); }

 private:
  FcwConfig cfg_;
  RuntimeOptions opt_;
  CaptureSource source_;
  Accelerator accel_;
  Sink sink_;
  std::unique_ptr<MomentEngine> engine_;
};

// Same summary as `python -m collision_avoidance.fcw` (first frame excluded from the mean).
std::string summaryLine(const std::vector<Level>& levels, const std::vector<double>& total_ms);
int warningRaises(const std::vector<Level>& levels);

// Overlay: boxes by warning level, TTC labels, heading and corridor (port of fcw/render.py).
cv::Mat drawOverlay(const FcwFrame& frame, double fx);

}  // namespace fcw
