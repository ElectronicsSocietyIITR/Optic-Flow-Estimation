// The board's accelerator stage and Horn engine through XRT (see fcw/board.hpp). Board only: needs
// XRT and the overlay (preprocess, lk_wrapper, ttc_engine_top + DPU). Not yet run on a board
// (plan 2, Task 10).
//
// Per frame (T2): BGRx in -> preprocess -> tensor + pyramid slot N; then LK(slot N-1, slot N) in the
// fabric while the DPU detects. Later (T3b, Option A): the engine reads slots N-1 and N and the flow of
// slot N, one start per pyramid level in use. Slots live in a FrameRing sized look-back 1 + frames in
// flight between T2 and the object stage (work queue + perception queue + the 2 stages in progress = 6 with
// the default queues); the Arm reads nothing the fabric wrote without a sync from the device first.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

#include "../../../hls/common/pyramid_layout.hpp"
#include "fcw/board.hpp"
#include "fcw/frame_ring.hpp"
// Note: level sub-buffers (xrt::bo(parent, size, offset)) need offsets the platform accepts; the slot
// layout aligns levels to 64 bytes. If the board's XRT wants more, raise align64 in pyramid_layout.hpp.

namespace fcw::board {

namespace {
constexpr int kBoxes = 16, kTerms = 23;
}

struct XrtAccelerator::Impl {
  FcwConfig cfg;
  cv::Size source, proc;
  xrt::device dev;
  xrt::uuid uuid;
  xrt::kernel pre, lk, engine;
  xrt::bo bgr, tensor, boxes, flow_boxes, fits, sums;
  struct Slot {
    xrt::bo pyr, flow_a, flow_b;  // the LK result lands in flow_a
  };
  std::vector<Slot> slots;
  FrameRing<int> ring{1, 6};  // tags only: slot of frame i = slots[i % ring.slots()]; T2 runs up to 7 frames ahead
  int64_t index = -1;
  std::unique_ptr<VartYolo> yolo;
  std::mutex m;  // T2 and T3b share the slot bookkeeping
  std::vector<int8_t> tensor_host;

  Slot& slot(int64_t i) { return slots[static_cast<size_t>(i % static_cast<int64_t>(slots.size()))]; }
};

XrtAccelerator::XrtAccelerator(const std::string& xclbin, const std::string& xmodel, const FcwConfig& cfg, cv::Size source)
    : impl_(std::make_shared<Impl>()) {
  Impl& d = *impl_;
  if (source.width > 1920 || source.height > 1080)  // the preprocess kernel's line buffers (gstCapture scales)
    throw std::runtime_error("source larger than the preprocess kernel's 1920x1080 input");
  d.cfg = cfg;
  d.source = source;
  d.proc = {cfg.frame_width, cfg.frame_height};
  d.dev = xrt::device(0);
  d.uuid = d.dev.load_xclbin(xclbin);
  d.pre = xrt::kernel(d.dev, d.uuid, "preprocess");
  d.lk = xrt::kernel(d.dev, d.uuid, "lk_wrapper");
  d.engine = xrt::kernel(d.dev, d.uuid, "ttc_engine_top");
  d.yolo = std::make_unique<VartYolo>(xmodel);
  d.bgr = xrt::bo(d.dev, static_cast<size_t>(source.area()) * 4, d.pre.group_id(0));
  d.tensor = xrt::bo(d.dev, static_cast<size_t>(d.proc.area()) * 3, d.pre.group_id(1));
  d.tensor_host.resize(static_cast<size_t>(d.proc.area()) * 3);
  d.boxes = xrt::bo(d.dev, kBoxes * 4 * 4, d.engine.group_id(3));
  d.flow_boxes = xrt::bo(d.dev, kBoxes * 4 * 4, d.engine.group_id(4));
  d.fits = xrt::bo(d.dev, kBoxes * 6 * 8, d.engine.group_id(5));
  d.sums = xrt::bo(d.dev, kBoxes * kTerms * 8, d.engine.group_id(14));
  const size_t pyr = static_cast<size_t>(pyrBytes(d.proc.height, d.proc.width)), flow = static_cast<size_t>(d.proc.area()) * 4;
  const std::vector<uint8_t> zeros(flow, 0);
  for (int i = 0; i < d.ring.slots(); ++i) {
    Impl::Slot s{xrt::bo(d.dev, pyr, d.pre.group_id(2)), xrt::bo(d.dev, flow, d.lk.group_id(2)), xrt::bo(d.dev, flow, d.lk.group_id(3))};
    for (xrt::bo* b : {&s.flow_a, &s.flow_b}) {  // the LK kernel's first pass reads them: zero once
      b->write(zeros.data());
      b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    d.slots.push_back(s);
  }
}

XrtAccelerator::~XrtAccelerator() = default;

AcceleratedFrame XrtAccelerator::operator()(const CapturedFrame& f) {
  Impl& d = *impl_;
  int64_t i;
  {
    std::lock_guard<std::mutex> lock(d.m);
    i = ++d.index;
    d.ring.acquire(i);  // throws if the engine still needs this slot (perception too far behind)
  }
  Impl::Slot& s = d.slot(i);
  // Preprocess: BGRx frame -> INT8 tensor + pyramid slot i.
  d.bgr.write(f.bgr.ptr());
  d.bgr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  d.pre(d.bgr, d.tensor, s.pyr, d.source.height, d.source.width, d.proc.height, d.proc.width, d.yolo->inputShift()).wait();
  AcceleratedFrame a;
  a.bgr = f.bgr;  // shown as is (the display draws in processing-frame coordinates, scaled)
  a.t = f.t;
  a.gray = cv::Mat(d.proc, CV_8U);
  s.pyr.sync(XCL_BO_SYNC_BO_FROM_DEVICE, a.gray.total(), 0);  // level 0 only
  s.pyr.read(a.gray.data, a.gray.total(), 0);
  d.tensor.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  d.tensor.read(d.tensor_host.data());
  // LK in the fabric || detection on the DPU.
  xrt::run lk_run;
  if (i > 0) lk_run = d.lk(d.slot(i - 1).pyr, s.pyr, s.flow_a, s.flow_b, d.proc.height, d.proc.width);
  a.detections = (*d.yolo)(d.tensor_host.data(), d.proc);
  if (i > 0) {
    lk_run.wait();
    s.flow_a.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint32_t* w = s.flow_a.map<const uint32_t*>();
    a.flow = cv::Mat(d.proc, CV_32FC2);
    for (int k = 0; k < d.proc.area(); ++k) {  // u: upper 16 bits, v: lower 16, 1/64 px
      a.flow.ptr<cv::Vec2f>()[k] = {static_cast<int16_t>(w[k] >> 16) / 64.0f, static_cast<int16_t>(w[k] & 0xFFFF) / 64.0f};
    }
  }
  return a;
}

namespace {

// The engine through XRT: same results as SoftwareMomentEngine (the kernel is bit-exact with it in C-simulation).
class XrtMomentEngine : public MomentEngine {
 public:
  explicit XrtMomentEngine(std::shared_ptr<void> keep, XrtAccelerator::Impl* d) : keep_(std::move(keep)), d_(d) {}

  std::vector<BoxSums> run(const cv::Mat&, const cv::Mat&, const cv::Mat&, const std::vector<EngineBox>& boxes, cv::Point2d pp,
                           int frame_index) override {
    XrtAccelerator::Impl& d = *d_;
    {  // this call needs frames frame_index - 1 and frame_index only; every call releases (also with no boxes)
      std::lock_guard<std::mutex> lock(d.m);
      d.ring.releaseBefore(frame_index - 1);
    }
    std::vector<BoxSums> out(boxes.size());
    if (frame_index < 1 || boxes.empty()) return out;
    const int n = std::min<int>(static_cast<int>(boxes.size()), kBoxes);
    std::set<int> levels{0};  // level 0 always: every box's flow moments
    for (int b = 0; b < n; ++b) levels.insert(boxes[b].level);
    auto& prev = d.slot(frame_index - 1);
    auto& curr = d.slot(frame_index);
    for (const int L : levels) {
      const int rows = pyrRows(d.proc.height, L), cols = pyrCols(d.proc.width, L), off = pyrOffset(d.proc.height, d.proc.width, L);
      const double f = std::pow(2.0, -L);
      std::vector<uint32_t> bw(kBoxes * 4, 0), fw(kBoxes * 4, 0);
      std::vector<int64_t> qw(kBoxes * 6, 0);
      uint32_t fit_valid = 0;
      for (int b = 0; b < n; ++b) {
        const Box& box = boxes[b].box;
        if (boxes[b].level == L)
          if (auto r = innerBox({box.x1 * f, box.y1 * f, box.x2 * f, box.y2 * f}, d.cfg.horn.shrink, cols, rows))
            std::copy_n(std::array<uint32_t, 4>{static_cast<uint32_t>(r->x), static_cast<uint32_t>(r->y), static_cast<uint32_t>(r->x + r->width),
                                                static_cast<uint32_t>(r->y + r->height)}.begin(), 4, bw.begin() + 4 * b);
        if (L == 0) {
          if (auto r = flowRect(box, d.cfg.divergence.shrink, cols, rows))
            std::copy_n(std::array<uint32_t, 4>{static_cast<uint32_t>(r->x), static_cast<uint32_t>(r->y), static_cast<uint32_t>(r->x + r->width),
                                                static_cast<uint32_t>(r->y + r->height)}.begin(), 4, fw.begin() + 4 * b);
          if (const auto& q = boxes[b].fit) {
            const int64_t c[6] = {q->a0, q->a1, q->a2, q->b0, q->b1, q->b2};
            std::copy_n(c, 6, qw.begin() + 6 * b);
            fit_valid |= 1u << b;
          }
        }
      }
      d.boxes.write(bw.data());
      d.flow_boxes.write(fw.data());
      d.fits.write(qw.data());
      for (xrt::bo* bo : {&d.boxes, &d.flow_boxes, &d.fits}) bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
      const size_t level_bytes = static_cast<size_t>(rows) * cols;
      xrt::bo prev_l(prev.pyr, level_bytes, off), curr_l(curr.pyr, level_bytes, off);
      d.engine(prev_l, curr_l, curr.flow_a, d.boxes, d.flow_boxes, d.fits, fit_valid,
               static_cast<int>(std::nearbyint(d.cfg.divergence.inlier_px * kFlowScale)), n, cols, rows,
               static_cast<int>(std::nearbyint(pp.x * f)), static_cast<int>(std::nearbyint(pp.y * f)), d.cfg.horn.grad_threshold_l1, d.sums)
          .wait();
      d.sums.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const int64_t* s = d.sums.map<const int64_t*>();
      for (int b = 0; b < n; ++b) {
        const int64_t* t = s + b * kTerms;
        if (boxes[b].level == L) {
          HornSums h;
          std::copy_n(t, 10, h.s.begin());
          h.n = t[10];
          out[b].horn = h;
        }
        if (L == 0) {
          FixedMoments m;
          std::copy_n(t + 11, 12, m.begin());
          out[b].flow = m;
        }
      }
    }
    return out;
  }

 private:
  std::shared_ptr<void> keep_;
  XrtAccelerator::Impl* d_;
};

}  // namespace

std::unique_ptr<MomentEngine> XrtAccelerator::momentEngine() { return std::make_unique<XrtMomentEngine>(impl_, impl_.get()); }

}  // namespace fcw::board
