// KV260 backends (spec sections 5 and 10). Capture and display (GStreamer, libdrm) build wherever
// those libraries exist, so they are compile-checked on the PC (FCW_BOARD_IO); the accelerators
// (XRT kernels, VART DPU runner) build only on the board image (FCW_BOARD). None of this has run
// on a board yet: plan 2, Task 10.
#pragma once

#include <memory>
#include <string>

#include "fcw/moment_engine.hpp"
#include "fcw/runtime.hpp"

namespace fcw::board {

// Frames from a file (decodebin: the K26's VCU hardware decoder when installed) or a V4L2 camera
// ("/dev/videoN", live: the appsink drops when perception falls behind). BGRx (CV_8UC4) frames, the
// preprocess kernel's input layout and the display's XRGB8888, so neither needs a copy-convert.
// Timestamps: the buffer's PTS (seconds), so dt follows the camera, not this thread. Frames are at most
// 1920x1080 (the preprocess kernel's input): larger sources are scaled down, aspect ratio kept.
CaptureSource gstCapture(const std::string& source);

// DRM/KMS display: the camera frame on the primary plane, boxes / TTC / banner on an ARGB8888 overlay
// plane the DisplayPort hardware alpha-blends (issue 15: no full-frame drawing on the Arm cores).
class DrmDisplay {
 public:
  explicit DrmDisplay(const std::string& card = "/dev/dri/card0");
  ~DrmDisplay();
  DrmDisplay(const DrmDisplay&) = delete;
  DrmDisplay& operator=(const DrmDisplay&) = delete;
  // frame.frame: the camera frame (CV_8UC4); objects in processing-frame pixels, scaled to it.
  void show(const FcwFrame& frame, cv::Size processing);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#ifdef FCW_BOARD
// YOLOv9t on the DPU (VART): INT8 tensor in, six-class detections out (decodeYolo).
class VartYolo {
 public:
  explicit VartYolo(const std::string& xmodel);
  ~VartYolo();
  int inputShift() const;  // the xmodel's input fix point: the preprocess kernel quantises with it
  std::vector<Detection> operator()(const int8_t* tensor_hwc, cv::Size input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// T2 on the board: preprocess kernel -> pyramid slot (FrameRing) -> LK wrapper || DPU, and the Horn
// engine for the object stage (Option A). Every buffer the Arm reads after the fabric wrote it is
// synced from the device first (issue 13).
class XrtAccelerator {
 public:
  XrtAccelerator(const std::string& xclbin, const std::string& xmodel, const FcwConfig& cfg, cv::Size source);
  ~XrtAccelerator();
  AcceleratedFrame operator()(const CapturedFrame& frame);
  // Shares the device and the pyramid / flow slots; null when the overlay has no engine (use the software one).
  std::unique_ptr<MomentEngine> momentEngine();
  struct Impl;  // opaque (xrt_accel.cpp)

 private:
  std::shared_ptr<Impl> impl_;
};
#endif

}  // namespace fcw::board
