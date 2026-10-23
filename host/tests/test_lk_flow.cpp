// Dense pyramidal LK (the board's flow algorithm, Vitis parameters) in software.
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>

#include "fcw/lk_flow.hpp"
#include "golden.hpp"

TEST(LkFlow, RecoversASubPixelShift) {
  cv::Mat tex(384, 512, CV_8U);
  cv::randu(tex, 0, 255);
  cv::GaussianBlur(tex, tex, cv::Size(0, 0), 2.0);
  cv::Mat moved;
  cv::warpAffine(tex, moved, cv::Matx23d(1, 0, 1.6, 0, 1, -0.7), tex.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
  const cv::Mat flow = fcw::lkFlow()(tex, moved);
  std::vector<float> err;
  for (int y = 32; y < 352; ++y)
    for (int x = 32; x < 480; ++x) {
      const cv::Vec2f f = flow.at<cv::Vec2f>(y, x);
      err.push_back(std::hypot(f[0] - 1.6f, f[1] + 0.7f));
    }
  std::nth_element(err.begin(), err.begin() + err.size() / 2, err.end());
  EXPECT_LT(err[err.size() / 2], 0.05f);
}

TEST(LkFlow, MatchesPythonOnARealFramePair) {
  if (!std::filesystem::exists(std::string(FCW_GOLDEN_LOCAL_DIR) + "/lk.yml.gz")) GTEST_SKIP() << "run tools/export_golden.py --only lk";
  golden::File g("lk", FCW_GOLDEN_LOCAL_DIR);
  const cv::Mat prev = g.image("prev"), curr = g.image("curr");
  cv::Mat ref;
  g.mat("flow").convertTo(ref, CV_32F);
  ref = ref.reshape(2, prev.rows);
  const cv::Mat mine = fcw::lkFlow()(prev, curr);
  std::vector<float> diff;
  for (int y = 0; y < prev.rows; ++y)
    for (int x = 0; x < prev.cols; ++x) diff.push_back(static_cast<float>(cv::norm(mine.at<cv::Vec2f>(y, x) - ref.at<cv::Vec2f>(y, x))));
  std::nth_element(diff.begin(), diff.begin() + diff.size() / 2, diff.end());
  EXPECT_LT(diff[diff.size() / 2], 0.05f);
}
