// The board's GStreamer capture, run on the PC (software decoders instead of the VCU).
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>

#include "fcw/board.hpp"

TEST(GstCapture, ReadsAFileAsBgrxWithStreamTimestamps) {
  const std::string clip = std::string(FCW_DATA_DIR) + "/normal_driving/241220_125301_002_FH.MP4";
  if (!std::filesystem::exists(clip)) GTEST_SKIP() << "dash-cam clip not present";
  auto source = fcw::board::gstCapture(clip);
  cv::VideoCapture reference(clip);
  double last_t = -1;
  for (int i = 0; i < 5; ++i) {
    const auto f = source();
    ASSERT_TRUE(f.has_value()) << i;
    EXPECT_EQ(f->bgr.type(), CV_8UC4);
    EXPECT_GT(f->t, last_t);
    if (i) EXPECT_NEAR(f->t - last_t, 1.0 / 30, 0.005) << i;
    last_t = f->t;
    cv::Mat ref, bgr;
    ASSERT_TRUE(reference.read(ref));
    cv::cvtColor(f->bgr, bgr, cv::COLOR_BGRA2BGR);
    ASSERT_EQ(bgr.size(), ref.size());
    cv::Mat d;
    cv::absdiff(bgr, ref, d);
    EXPECT_LT(cv::mean(d)[0], 2.0) << "frame " << i << " differs from OpenCV's decode";
  }
}
