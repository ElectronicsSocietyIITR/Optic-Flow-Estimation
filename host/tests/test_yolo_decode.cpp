// Parity: fcw::decodeYolo vs YOLOv9t's own decode (ultralytics) restricted to the six collision classes.
// Data in tests/golden_local (git-ignored): python tools/export_golden.py --only yolo
#include <gtest/gtest.h>

#include <filesystem>

#include "fcw/yolo_decode.hpp"
#include "golden.hpp"

namespace {

std::vector<cv::Mat> heads(const golden::File& g, const std::string& k) {
  std::vector<cv::Mat> out;
  for (int j = 0; j < 3; ++j) {
    const std::string h = k + "_head" + std::to_string(j);
    cv::Mat rows;
    g.mat(h).convertTo(rows, CV_32F);  // (H*W, 144)
    out.push_back(rows.reshape(rows.cols, static_cast<int>(g.num(h + "_h"))).clone());  // H x W, 144 channels
  }
  return out;
}

}  // namespace

TEST(YoloDecode, MatchesPythonDecode) {
  if (!std::filesystem::exists(std::string(FCW_GOLDEN_LOCAL_DIR) + "/yolo.yml.gz")) GTEST_SKIP() << "run tools/export_golden.py --only yolo";
  golden::File g("yolo", FCW_GOLDEN_LOCAL_DIR);
  const std::map<int, std::string> names{{0, "person"}, {1, "bicycle"}, {2, "car"}, {3, "motorcycle"}, {5, "bus"}, {7, "truck"}};
  for (int f = 0; f < static_cast<int>(g.num("frames")); ++f) {
    const std::string k = "f" + std::to_string(f);
    const auto dets = fcw::decodeYolo(heads(g, k), cv::Size(512, 384), {});
    const cv::Mat ref = g.mat(k + "_dets");
    ASSERT_EQ(static_cast<int>(dets.size()), ref.rows) << k;
    for (int r = 0; r < ref.rows; ++r) {
      EXPECT_NEAR(dets[r].box.x1, ref.at<double>(r, 0), 1e-3) << k << " " << r;
      EXPECT_NEAR(dets[r].box.y1, ref.at<double>(r, 1), 1e-3) << k << " " << r;
      EXPECT_NEAR(dets[r].box.x2, ref.at<double>(r, 2), 1e-3) << k << " " << r;
      EXPECT_NEAR(dets[r].box.y2, ref.at<double>(r, 3), 1e-3) << k << " " << r;
      EXPECT_NEAR(dets[r].conf, ref.at<double>(r, 4), 1e-4) << k << " " << r;
      EXPECT_EQ(dets[r].cls, names.at(static_cast<int>(ref.at<double>(r, 5)))) << k << " " << r;
    }
  }
}

TEST(YoloDecode, SkipsOtherClasses) {
  // One cell, strongly "traffic light" (class 9), weak everything else: nothing comes out.
  auto filled = [](int rows, int cols) {  // Scalar fills only 4 channels: fill the single-channel view
    cv::Mat m(rows, cols, CV_32FC(144));
    m.reshape(1).setTo(-20.0);
    return m;
  };
  std::vector<cv::Mat> h{filled(48, 64), filled(24, 32), filled(12, 16)};
  h[0].ptr<float>(10, 10)[64 + 9] = 8.0f;
  EXPECT_TRUE(fcw::decodeYolo(h, cv::Size(512, 384), {}).empty());
  h[0].ptr<float>(10, 10)[64 + 2] = 8.0f;  // now also "car"
  const auto dets = fcw::decodeYolo(h, cv::Size(512, 384), {});
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].cls, "car");
}
