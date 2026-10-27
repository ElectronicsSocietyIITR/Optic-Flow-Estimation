// The board's GStreamer capture, run on the PC (software decoders instead of the VCU).
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <fstream>

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
    cv::Mat decoded, ref, bgr;
    ASSERT_TRUE(reference.read(decoded));
    cv::cvtColor(f->bgr, bgr, cv::COLOR_BGRA2BGR);
    cv::resize(decoded, ref, bgr.size(), 0, 0, cv::INTER_AREA);  // capture scales sources above 1080p
    cv::Mat d;
    cv::absdiff(bgr, ref, d);
    EXPECT_LT(cv::mean(d)[0], 2.0) << "frame " << i << " differs from OpenCV's decode";
  }
}

// The preprocess kernel takes at most 1920x1080 (its line buffers): larger sources are scaled in
// capture, aspect ratio kept.
TEST(GstCapture, ScalesSourcesLargerThanTheKernelInputTo1080p) {
  const std::string clip = std::string(FCW_DATA_DIR) + "/normal_driving/241220_125301_002_FH.MP4";
  if (!std::filesystem::exists(clip)) GTEST_SKIP() << "dash-cam clip not present";
  cv::VideoCapture reference(clip);
  ASSERT_GT(reference.get(cv::CAP_PROP_FRAME_WIDTH), 1920) << "this test needs a source wider than 1920";
  auto source = fcw::board::gstCapture(clip);
  const auto f = source();
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(f->bgr.cols, 1920);
  EXPECT_EQ(f->bgr.rows, 1080);
}

// A source GStreamer cannot decode (a wrong file, a missing decoder plugin) must fail with GStreamer's
// message, not block on the first frame forever.
TEST(GstCapture, ReportsASourceItCannotDecode) {
  const auto path = std::filesystem::temp_directory_path() / "fcw_not_a_video.mp4";
  std::ofstream(path) << "this is not a video\n";
  try {
    auto source = fcw::board::gstCapture(path.string());
    source();
    FAIL() << "expected an error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("GStreamer"), std::string::npos) << e.what();
  }
}
