// The parallel object stage and the stage split must reproduce the serial pipeline exactly.
// Data in tests/golden_local (git-ignored): python tools/export_golden.py --only pipeline
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "fcw/divergence.hpp"
#include "fcw/pipeline.hpp"
#include "golden.hpp"

namespace {

cv::Mat readFlow16(const std::string& path, int rows, int cols) {
  std::ifstream in(path, std::ios::binary);
  std::vector<uint16_t> raw(static_cast<size_t>(rows) * cols * 2);
  in.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(uint16_t));
  cv::Mat half(rows, cols, CV_16FC2, raw.data()), out;
  half.convertTo(out, CV_32F);
  return out;
}

struct Scenario {
  std::vector<cv::Mat> frames;
  std::vector<double> times;
  std::vector<std::vector<fcw::Detection>> dets;
  std::vector<cv::Mat> flows;
};

Scenario load(const golden::File& g, const std::string& name, const std::string& local) {
  Scenario s;
  const std::string cls = g.str(name + "_class");
  for (int i = 0; i < static_cast<int>(g.num(name + "_frames")); ++i) {
    const std::string k = name + "_f" + std::to_string(i);
    cv::Mat bgr;
    cv::cvtColor(g.image(k + "_gray"), bgr, cv::COLOR_GRAY2BGR);
    s.frames.push_back(bgr);
    s.times.push_back(g.num(k + "_t"));
    const cv::Mat d = g.mat(k + "_dets");
    std::vector<fcw::Detection> dets;
    for (int r = 0; r < d.rows; ++r) dets.push_back({{d.at<double>(r, 0), d.at<double>(r, 1), d.at<double>(r, 2), d.at<double>(r, 3)}, cls, 0.9});
    s.dets.push_back(dets);
    s.flows.push_back(g.has(k + "_flow") ? readFlow16(local + "/" + g.str(k + "_flow"), bgr.rows, bgr.cols)
                                          : cv::Mat::zeros(bgr.size(), CV_32FC2));  // frame 0 has no flow
  }
  return s;
}

// Serial reference (workers 0) or parallel; interleaved = frame stage of N+1 before the object stage of N.
// give_flow_q: hand the object stage the flow already in 1/64 px, as the board's accelerator does.
std::vector<fcw::FcwFrame> run(const Scenario& s, int workers, bool interleaved, bool give_flow_q = false) {
  int current = 0;
  fcw::FcwConfig cfg;
  cfg.object_workers = workers;
  fcw::FcwPipeline pipe(cfg, [&](const cv::Mat&) { return s.dets[current]; },
                        [&](const cv::Mat&, const cv::Mat&) { return s.flows[current]; });
  std::vector<fcw::FcwFrame> out;
  if (!interleaved) {
    for (current = 0; current < static_cast<int>(s.frames.size()); ++current) out.push_back(pipe.process(s.frames[current], s.times[current]));
    return out;
  }
  std::optional<fcw::FrameState> pending;
  for (current = 0; current < static_cast<int>(s.frames.size()); ++current) {
    fcw::FrameState next = pipe.frameStage(s.frames[current], s.times[current]);
    if (give_flow_q) next.flow_q = fcw::quantiseFlow(next.out.flow);
    if (pending) out.push_back(pipe.objectStage(std::move(*pending)));
    pending = std::move(next);
  }
  if (pending) out.push_back(pipe.objectStage(std::move(*pending)));
  return out;
}

void expectSame(const std::vector<fcw::FcwFrame>& a, const std::vector<fcw::FcwFrame>& b, const std::string& what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  for (size_t f = 0; f < a.size(); ++f) {
    ASSERT_EQ(a[f].objects.size(), b[f].objects.size()) << what << " frame " << f;
    EXPECT_EQ(a[f].ego.rvec, b[f].ego.rvec) << what << " frame " << f;
    for (size_t o = 0; o < a[f].objects.size(); ++o) {
      const auto &x = a[f].objects[o], &y = b[f].objects[o];
      const std::string at = what + " frame " + std::to_string(f) + " obj " + std::to_string(o);
      EXPECT_EQ(x.track_id, y.track_id) << at;
      EXPECT_EQ(x.level, y.level) << at;
      ASSERT_EQ(x.measurements.size(), y.measurements.size()) << at;
      for (size_t m = 0; m < x.measurements.size(); ++m) {
        EXPECT_EQ(x.measurements[m].method, y.measurements[m].method) << at;
        EXPECT_EQ(x.measurements[m].eta, y.measurements[m].eta) << at;
      }
      ASSERT_EQ(x.estimate.has_value(), y.estimate.has_value()) << at;
      if (x.estimate) EXPECT_EQ(x.estimate->eta, y.estimate->eta) << at;
      EXPECT_EQ(x.kappa, y.kappa) << at;
      ASSERT_EQ(x.course.has_value(), y.course.has_value()) << at;
      if (x.course) EXPECT_EQ(x.course->r, y.course->r) << at;
    }
  }
}

}  // namespace

TEST(ParallelPipeline, SameResultsAsSerialOnEveryScenario) {
  const std::string local = FCW_GOLDEN_LOCAL_DIR;
  if (!std::filesystem::exists(local + "/pipeline.yml.gz")) GTEST_SKIP() << "run tools/export_golden.py --only pipeline";
  golden::File g("pipeline", local);
  std::stringstream names(g.str("scenarios"));
  std::string name;
  while (std::getline(names, name, ',')) {  // includes "twenty": 20 tracks, more than the 16 that get TTC
    const Scenario s = load(g, name, local);
    const auto serial = run(s, 0, false);
    for (int rep = 0; rep < 3; ++rep) expectSame(serial, run(s, 3, false), name + " workers=3 rep " + std::to_string(rep));
    expectSame(serial, run(s, 1, false), name + " workers=1");
    expectSame(serial, run(s, 3, true), name + " stages interleaved");
  }
}

TEST(ParallelPipeline, NoTracksIsFine) {
  fcw::FcwConfig cfg;
  cfg.object_workers = 3;
  fcw::FcwPipeline pipe(cfg, [](const cv::Mat&) { return std::vector<fcw::Detection>{}; },
                        [](const cv::Mat& a, const cv::Mat&) { return cv::Mat::zeros(a.size(), CV_32FC2); });
  const cv::Mat black = cv::Mat::zeros(384, 512, CV_8UC3);
  for (int i = 0; i < 3; ++i) EXPECT_TRUE(pipe.process(black, i / 30.0).objects.empty());
}

// The board's accelerator delivers the LK flow in 1/64 px: the object stage uses it as is (no float round
// trip, no requantising) and gives exactly the serial results.
TEST(ParallelPipeline, UsesTheAcceleratorsFixedPointFlowWhenGiven) {
  const std::string local = FCW_GOLDEN_LOCAL_DIR;
  if (!std::filesystem::exists(local + "/pipeline.yml.gz")) GTEST_SKIP() << "run tools/export_golden.py --only pipeline";
  golden::File g("pipeline", local);
  std::stringstream names(g.str("scenarios"));
  std::string name;
  while (std::getline(names, name, ',')) {
    const Scenario s = load(g, name, local);
    const auto given = run(s, 3, true, true);
    expectSame(run(s, 0, false), given, name + " flow_q given");
    for (size_t f = 0; f < given.size(); ++f) EXPECT_EQ(given[f].timings_ms.count("quantise"), 0u) << name << " frame " << f;
  }
}
