// T1-T4 runtime: ordering, back-pressure and the CLI summary line.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "fcw/runtime.hpp"

namespace {

// Synthetic camera: a random texture that drifts one pixel per frame.
fcw::CaptureSource syntheticSource(int frames) {
  auto i = std::make_shared<int>(0);
  cv::Mat tex(384 + frames, 512 + frames, CV_8UC3);
  cv::randu(tex, 0, 255);
  return [=]() -> std::optional<fcw::CapturedFrame> {
    if (*i >= frames) return std::nullopt;
    const int k = (*i)++;
    return fcw::CapturedFrame{tex(cv::Rect(k, 0, 512, 384)).clone(), k / 30.0};
  };
}

fcw::DetectorFn oneCar(int delay_ms = 0) {
  return [delay_ms](const cv::Mat&) {
    if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    return std::vector<fcw::Detection>{{{200, 150, 300, 230}, "car", 0.9}};
  };
}

}  // namespace

TEST(BoundedQueue, DropsOldestWhenFull) {
  fcw::BoundedQueue<int> q(2);
  EXPECT_FALSE(q.pushDropOldest(1));
  EXPECT_FALSE(q.pushDropOldest(2));
  EXPECT_TRUE(q.pushDropOldest(3));  // 1 dropped
  EXPECT_EQ(*q.pop(), 2);
  EXPECT_EQ(*q.pop(), 3);
  q.close();
  EXPECT_FALSE(q.pop().has_value());
}

TEST(Runtime, ProcessesEveryFrameInOrderWhenNotLive) {
  std::vector<int> indices;
  std::vector<double> times;
  fcw::RuntimeOptions opt;
  opt.live = false;
  fcw::Runtime rt(fcw::FcwConfig{}, opt, syntheticSource(25), oneCar(), fcw::disFlow(),
                  [&](const fcw::FcwFrame& f) { indices.push_back(f.index); times.push_back(f.time_s); });
  const fcw::RuntimeStats s = rt.run();
  EXPECT_EQ(s.captured, 25);
  EXPECT_EQ(s.processed, 25);
  EXPECT_EQ(s.dropped, 0);
  ASSERT_EQ(indices.size(), 25u);
  for (int i = 0; i < 25; ++i) EXPECT_EQ(indices[i], i);
  for (size_t i = 1; i < times.size(); ++i) EXPECT_GT(times[i], times[i - 1]);
}

TEST(Runtime, DropsAtCaptureUnderLoadAndStaysMonotonic) {
  std::vector<double> times;
  fcw::RuntimeOptions opt;
  opt.live = true;
  opt.capture_queue = 2;
  fcw::Runtime rt(fcw::FcwConfig{}, opt, syntheticSource(60), oneCar(15), fcw::disFlow(),
                  [&](const fcw::FcwFrame& f) { times.push_back(f.time_s); });
  const fcw::RuntimeStats s = rt.run();
  EXPECT_EQ(s.captured, 60);
  EXPECT_GT(s.dropped, 0);                         // the slow detector cannot keep up
  EXPECT_EQ(s.processed + s.dropped, s.captured);  // nothing lost silently
  EXPECT_LE(s.max_capture_queue, 2);               // bounded
  EXPECT_EQ(static_cast<int>(times.size()), s.processed);
  for (size_t i = 1; i < times.size(); ++i) EXPECT_GT(times[i], times[i - 1]);
}

TEST(Runtime, SummaryLineMatchesThePythonCli) {
  const std::vector<fcw::Level> levels{fcw::Level::None, fcw::Level::Warning, fcw::Level::Warning, fcw::Level::None, fcw::Level::Critical};
  EXPECT_EQ(fcw::summaryLine(levels, {50.0, 10.0, 20.0, 30.0, 40.0}),
            "frames 5 | mean 25.0 ms/frame | warnings raised 2 | critical frames 1");
}

TEST(Pipeline, ReportsTimePerPerceptionStep) {
  // Per-method timings feed the on-board latency histograms (spec section 8).
  auto source = syntheticSource(3);
  fcw::FcwPipeline pipe(fcw::FcwConfig{}, oneCar(), fcw::disFlow());
  fcw::FcwFrame last;
  while (auto f = source()) last = pipe.process(f->bgr, f->t);
  for (const char* key : {"flow", "detection", "track", "ego", "looming", "scale", "horn", "divergence", "fusion_course", "warn", "total"})
    EXPECT_TRUE(last.timings_ms.count(key)) << key;
}

TEST(Runtime, MatchesTheSerialPipelineFrameByFrame) {
  // Four stages (capture, flow || detection, frame stage, object stage) must not change any result.
  // One video for both runs: syntheticSource draws from OpenCV's global RNG, so a second call differs.
  std::vector<fcw::CapturedFrame> video;
  for (auto src = syntheticSource(25); auto f = src();) video.push_back(*f);
  auto replay = [&video, i = std::make_shared<size_t>(0)]() -> std::optional<fcw::CapturedFrame> {
    return *i < video.size() ? std::optional<fcw::CapturedFrame>(video[(*i)++]) : std::nullopt;
  };
  std::vector<fcw::FcwFrame> threaded;
  fcw::RuntimeOptions opt;
  opt.live = false;
  fcw::Runtime rt(fcw::FcwConfig{}, opt, replay, oneCar(), fcw::disFlow(), [&](const fcw::FcwFrame& f) { threaded.push_back(f); });
  rt.run();
  fcw::FcwPipeline serial(fcw::FcwConfig{}, oneCar(), fcw::disFlow());
  ASSERT_EQ(threaded.size(), 25u);
  for (const fcw::FcwFrame& t : threaded) {
    const fcw::CapturedFrame* f = &video[t.index];
    const fcw::FcwFrame s = serial.process(f->bgr, f->t);
    EXPECT_EQ(s.ego.rvec, t.ego.rvec) << t.index;
    EXPECT_EQ(s.level(), t.level()) << t.index;
    ASSERT_EQ(s.objects.size(), t.objects.size()) << t.index;
    for (size_t o = 0; o < s.objects.size(); ++o) {
      ASSERT_EQ(s.objects[o].estimate.has_value(), t.objects[o].estimate.has_value()) << t.index;
      if (s.objects[o].estimate) EXPECT_EQ(s.objects[o].estimate->eta, t.objects[o].estimate->eta) << t.index;
    }
  }
}

TEST(Runtime, ACustomAcceleratorWithItsOwnGrayGivesTheSameResults) {
  // The board's accelerator stage (preprocess kernel, then LK || DPU) hands over a ready gray image;
  // with the same frames, gray and flow, results must not change.
  std::vector<fcw::CapturedFrame> video;
  for (auto src = syntheticSource(12); auto f = src();) video.push_back(*f);
  auto replay = [&video]() {
    return [&video, i = std::make_shared<size_t>(0)]() -> std::optional<fcw::CapturedFrame> {
      return *i < video.size() ? std::optional<fcw::CapturedFrame>(video[(*i)++]) : std::nullopt;
    };
  };
  std::vector<fcw::FcwFrame> pc, board;
  fcw::RuntimeOptions opt;
  fcw::Runtime(fcw::FcwConfig{}, opt, replay(), oneCar(), fcw::disFlow(), [&](const fcw::FcwFrame& f) { pc.push_back(f); }).run();
  auto dis = fcw::disFlow();
  auto prev = std::make_shared<cv::Mat>();
  fcw::Accelerator accel = [&, prev](const fcw::CapturedFrame& c) {
    fcw::AcceleratedFrame a;
    a.bgr = c.bgr;
    a.t = c.t;
    a.gray = fcw::bgrToGray(c.bgr);
    if (!prev->empty()) a.flow = dis(*prev, a.gray);
    a.detections = oneCar()(c.bgr);
    *prev = a.gray;
    return a;
  };
  fcw::Runtime(fcw::FcwConfig{}, opt, replay(), accel, [&](const fcw::FcwFrame& f) { board.push_back(f); }).run();
  ASSERT_EQ(pc.size(), board.size());
  for (size_t i = 0; i < pc.size(); ++i) {
    EXPECT_EQ(pc[i].ego.rvec, board[i].ego.rvec) << i;
    ASSERT_EQ(pc[i].objects.size(), board[i].objects.size()) << i;
    for (size_t o = 0; o < pc[i].objects.size(); ++o)
      if (pc[i].objects[o].estimate) EXPECT_EQ(pc[i].objects[o].estimate->eta, board[i].objects[o].estimate->eta) << i;
  }
}
