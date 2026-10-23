// Parity: fcw::FcwPipeline vs collision_avoidance/fcw/pipeline.py, frame by frame.
// Data lives in tests/golden_local (git-ignored): python tools/export_golden.py --only pipeline
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "fcw/pipeline.hpp"
#include "golden.hpp"

namespace {

cv::Mat readFlow16(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<uint16_t> raw(384 * 512 * 2);
  in.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(uint16_t));
  cv::Mat half(384, 512, CV_16FC2, raw.data()), out;
  half.convertTo(out, CV_32F);
  return out;
}

void replay(const std::string& name) {
  const std::string local = FCW_GOLDEN_LOCAL_DIR;
  if (!std::filesystem::exists(local + "/pipeline.yml.gz")) GTEST_SKIP() << "run tools/export_golden.py --only pipeline";
  golden::File g("pipeline", local);
  int current = 0;
  const std::string class_name = g.str(name + "_class");
  auto detector = [&](const cv::Mat&) {
    std::vector<fcw::Detection> out;
    const cv::Mat d = g.mat(name + "_f" + std::to_string(current) + "_dets");
    for (int r = 0; r < d.rows; ++r) out.push_back({{d.at<double>(r, 0), d.at<double>(r, 1), d.at<double>(r, 2), d.at<double>(r, 3)}, class_name, 0.9});
    return out;
  };
  auto flow = [&](const cv::Mat&, const cv::Mat&) { return readFlow16(local + "/" + g.str(name + "_f" + std::to_string(current) + "_flow")); };
  fcw::FcwPipeline pipe(fcw::FcwConfig{}, detector, flow);
  for (current = 0; current < static_cast<int>(g.num(name + "_frames")); ++current) {
    const std::string k = name + "_f" + std::to_string(current);
    cv::Mat bgr;
    cv::cvtColor(g.image(k + "_gray"), bgr, cv::COLOR_GRAY2BGR);
    const fcw::FcwFrame r = pipe.process(bgr, g.num(k + "_t"));
    EXPECT_NEAR(r.dt_s, g.num(k + "_dt"), 1e-12) << k;
    const cv::Mat heading = g.mat(k + "_heading");
    EXPECT_NEAR(r.heading.x, heading.at<double>(0, 0), 0.5) << k;
    EXPECT_NEAR(r.heading.y, heading.at<double>(0, 1), 0.5) << k;
    const cv::Mat objs = g.mat(k + "_objects");
    ASSERT_EQ(static_cast<int>(r.objects.size()), objs.rows) << k;
    for (int o = 0; o < objs.rows; ++o) {
      const fcw::ObjectResult& obj = r.objects[o];
      const std::string ok = k + " obj " + std::to_string(o);
      EXPECT_EQ(obj.track_id, static_cast<int>(objs.at<double>(o, 0))) << ok;   // same selection order
      EXPECT_NEAR(obj.bbox.x1, objs.at<double>(o, 1), 1e-6) << ok;
      EXPECT_NEAR(obj.bbox.y2, objs.at<double>(o, 4), 1e-6) << ok;
      EXPECT_EQ(static_cast<int>(obj.level), static_cast<int>(objs.at<double>(o, 5))) << ok;
      int bits = 0;
      for (const auto& m : obj.measurements) bits |= 1 << (m.method == "looming" ? 0 : m.method == "scale" ? 1 : m.method == "horn" ? 2 : 3);
      EXPECT_EQ(bits, static_cast<int>(objs.at<double>(o, 10))) << ok << " methods";
      ASSERT_EQ(obj.estimate.has_value(), objs.at<double>(o, 6) != 0) << ok;
      if (obj.estimate) {
        EXPECT_NEAR(obj.estimate->eta, objs.at<double>(o, 7), 1e-4 * (1 + std::abs(objs.at<double>(o, 7)))) << ok;
        const double ttc = objs.at<double>(o, 8);
        if (std::isnan(ttc)) EXPECT_FALSE(obj.estimate->ttc_s.has_value()) << ok;
        else { ASSERT_TRUE(obj.estimate->ttc_s.has_value()) << ok; EXPECT_NEAR(*obj.estimate->ttc_s, ttc, 1e-4 * ttc) << ok; }
      }
      EXPECT_EQ(obj.course && obj.course->on_course, objs.at<double>(o, 9) != 0) << ok;
      const double kappa = objs.at<double>(o, 11);
      if (std::isnan(kappa)) EXPECT_FALSE(obj.kappa.has_value()) << ok;
      else { ASSERT_TRUE(obj.kappa.has_value()) << ok; EXPECT_NEAR(*obj.kappa, kappa, 1e-3 * (1 + kappa)) << ok; }
    }
  }
}

}  // namespace

TEST(PipelineParity, HeadOnApproach) { replay("head_on"); }
TEST(PipelineParity, CarInTheNextLane) { replay("next_lane"); }
TEST(PipelineParity, StaticObject) { replay("static"); }
TEST(PipelineParity, DroppedFrames) { replay("dropped"); }
TEST(PipelineParity, IrregularFrames) { replay("irregular"); }
TEST(PipelineParity, OutOfOrderTimestamps) { replay("out_of_order"); }
TEST(PipelineParity, TwentyTracksSelectSixteen) { replay("twenty"); }
TEST(PipelineParity, OncomingOnTheRoadIsNotTheLead) { replay("ground_oncoming"); }
TEST(PipelineParity, ParkedOnTheRoadStillWarns) { replay("ground_parked"); }
TEST(PipelineParity, PedestrianCrossingIntoOurPath) { replay("ground_crossing"); }

TEST(ProcessingSize, KeepsTheSourceAspect) {
  const std::vector<std::pair<cv::Size, cv::Size>> cases{{{3840, 2160}, {512, 288}}, {{1920, 1200}, {512, 320}}, {{640, 480}, {512, 384}},
                                                         {{1242, 375}, {512, 160}}, {{480, 640}, {512, 512}}, {{100, 10}, {512, 64}}};
  for (const auto& [src, expected] : cases) EXPECT_EQ(fcw::processingSize(src.width, src.height), expected) << src;
}
