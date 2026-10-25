// MomentEngine: the per-box sums the PL engine returns (Option A: one call per frame, one pass per
// pyramid level). The software engine must give the HLS golden values.
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <fstream>
#include <vector>

#include "fcw/horn.hpp"
#include "fcw/moment_engine.hpp"
#include "golden.hpp"

namespace {

const std::string kHlsGolden = std::string(FCW_GOLDEN_DIR) + "/../../../hls/ttc_engine/golden/";

cv::Mat readRaw(const std::string& path, int rows, int cols, int type) {
  cv::Mat m(rows, cols, type);
  std::ifstream(path, std::ios::binary).read(reinterpret_cast<char*>(m.data), m.total() * m.elemSize());
  return m;
}

}  // namespace

TEST(MomentEngine, SoftwareMatchesTheHlsGolden) {
  // Case "masked": box 0 with last frame's fit (inliers only), box 1 without (all pixels).
  const std::string dir = kHlsGolden + "masked/";
  const cv::Mat prev = readRaw(dir + "prev.bin", 384, 512, CV_8U), curr = readRaw(dir + "curr.bin", 384, 512, CV_8U);
  const cv::Mat flow_q = readRaw(dir + "flow.bin", 384, 512, CV_16SC2);
  std::ifstream ff(dir + "fits.txt");
  int inlier_q, valid;
  fcw::FitQ fit;
  ff >> inlier_q >> valid >> fit.a0 >> fit.a1 >> fit.a2 >> fit.b0 >> fit.b1 >> fit.b2;
  ASSERT_EQ(valid, 1);
  const std::vector<fcw::EngineBox> boxes{{{180, 120, 330, 260}, 0, fit}, {{300, 100, 420, 220}, 0, std::nullopt}};
  fcw::SoftwareMomentEngine engine({}, {});
  const auto sums = engine.run(prev, curr, flow_q, boxes, {256, 192});
  ASSERT_EQ(sums.size(), 2u);
  std::ifstream ef(dir + "expected.txt");
  for (int b = 0; b < 2; ++b) {
    ASSERT_TRUE(sums[b].horn && sums[b].flow) << b;
    long long e;
    for (int k = 0; k < 10; ++k) { ef >> e; EXPECT_EQ(sums[b].horn->s[k], e) << "box " << b << " horn " << k; }
    ef >> e;
    EXPECT_EQ(sums[b].horn->n, e) << "box " << b << " horn n";
    for (int k = 0; k < 12; ++k) { ef >> e; EXPECT_EQ((*sums[b].flow)[k], e) << "box " << b << " moment " << k; }
  }
}

TEST(MomentEngine, HornRunsAtEachBoxsPyramidLevel) {
  // Level 1 sums equal the software Horn on the level-1 images (what hornAtLevel computes).
  cv::Mat prev(384, 512, CV_8U), curr;
  cv::randu(prev, 0, 255);
  cv::GaussianBlur(prev, prev, cv::Size(0, 0), 2.0);
  cv::warpAffine(prev, curr, cv::Matx23d(1.02, 0, -5, 0, 1.02, -4), prev.size());
  const cv::Mat flow_q(384, 512, CV_16SC2, cv::Scalar(0, 0));
  const fcw::Box box{150, 100, 350, 280};
  fcw::SoftwareMomentEngine engine({}, {});
  const auto sums = engine.run(prev, curr, flow_q, {{box, 0, std::nullopt}, {box, 1, std::nullopt}}, {256, 192});
  const auto l0 = fcw::hornSums(prev, curr, box, {256, 192});
  const auto l1 = fcw::hornSums(fcw::downsample(prev, 1), fcw::downsample(curr, 1), {75, 50, 175, 140}, {128, 96});
  ASSERT_TRUE(sums[0].horn && sums[1].horn && l0 && l1);
  EXPECT_EQ(sums[0].horn->s, l0->s);
  EXPECT_EQ(sums[1].horn->s, l1->s);
  ASSERT_TRUE(sums[0].flow && sums[1].flow);  // flow moments for every box (level-0 pass), whatever its Horn level
  EXPECT_EQ(*sums[1].flow, *sums[0].flow);
}

// A 4-lane engine takes at most 4 boxes per start: level 0 carries every box (flow moments, and Horn
// for its own level-0 boxes), each other level only its own boxes.
TEST(EnginePasses, BatchesBoxesIntoLaneSizedPassesPerLevel) {
  std::vector<fcw::EngineBox> boxes(6);
  const int levels[6] = {0, 0, 1, 2, 1, 0};
  for (int i = 0; i < 6; ++i) boxes[i].level = levels[i];
  const auto passes = fcw::planEnginePasses(boxes, 4);
  ASSERT_EQ(passes.size(), 4u);
  EXPECT_EQ(passes[0].level, 0);
  EXPECT_EQ(passes[0].boxes, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(passes[1].level, 0);
  EXPECT_EQ(passes[1].boxes, (std::vector<int>{4, 5}));
  EXPECT_EQ(passes[2].level, 1);
  EXPECT_EQ(passes[2].boxes, (std::vector<int>{2, 4}));
  EXPECT_EQ(passes[3].level, 2);
  EXPECT_EQ(passes[3].boxes, (std::vector<int>{3}));
}

TEST(EnginePasses, SixteenLanesNeedOnePassPerLevel) {
  std::vector<fcw::EngineBox> boxes(16);
  for (int i = 0; i < 16; ++i) boxes[i].level = i % 2;
  const auto passes = fcw::planEnginePasses(boxes, 16);
  ASSERT_EQ(passes.size(), 2u);
  EXPECT_EQ(passes[0].boxes.size(), 16u);
  EXPECT_EQ(passes[1].boxes.size(), 8u);
}

TEST(EnginePasses, NoBoxesNoPasses) { EXPECT_TRUE(fcw::planEnginePasses({}, 4).empty()); }
