// OverlayPainter: the board display's annotation layer (ARGB overlay plane over the camera frame).
#include <gtest/gtest.h>

#include "fcw/overlay.hpp"

namespace {

fcw::FcwFrame frameWith(fcw::Box box, fcw::Level level, std::optional<double> ttc = std::nullopt) {
  fcw::FcwFrame f;
  fcw::ObjectResult o;
  o.bbox = box;
  o.level = level;
  if (ttc) {
    fcw::TtcEstimate e;
    e.ttc_s = ttc;
    o.estimate = e;
  }
  f.objects.push_back(o);
  return f;
}

uint8_t alpha(const cv::Mat& m, int x, int y) { return m.at<cv::Vec4b>(y, x)[3]; }

const cv::Size kProc(512, 288), kFrame(1920, 1080);  // boxes in processing pixels, drawn at 3.75x

}  // namespace

TEST(OverlayPainter, DrawsScaledBoxesOnATransparentOverlay) {
  cv::Mat overlay = cv::Mat::zeros(kFrame, CV_8UC4);
  fcw::OverlayPainter painter;
  painter.paint(overlay, frameWith({100, 100, 200, 200}, fcw::Level::Warning, 2.5), kProc, kFrame);
  EXPECT_EQ(alpha(overlay, 375, 500), 255) << "left edge of the box at x = 100 * 3.75";
  EXPECT_EQ(alpha(overlay, 560, 560), 0) << "inside the box stays transparent";
  EXPECT_EQ(alpha(overlay, 1800, 1000), 0) << "far away stays transparent";
}

// Each call clears only what the previous call on this buffer drew: no full-frame clear on the Arm cores.
TEST(OverlayPainter, ClearsOnlyWhatItDrewBefore) {
  cv::Mat overlay = cv::Mat::zeros(kFrame, CV_8UC4);
  fcw::OverlayPainter painter;
  painter.paint(overlay, frameWith({100, 100, 200, 200}, fcw::Level::Critical, 1.2), kProc, kFrame);
  ASSERT_EQ(alpha(overlay, 375, 500), 255);
  ASSERT_GT(alpha(overlay, 40, 40), 0) << "BRAKE banner drawn";
  overlay.at<cv::Vec4b>(1000, 1800) = {1, 2, 3, 4};  // a pixel nothing ever drew on
  painter.paint(overlay, frameWith({300, 50, 400, 150}, fcw::Level::None), kProc, kFrame);
  EXPECT_EQ(alpha(overlay, 375, 500), 0) << "the old box is gone";
  EXPECT_EQ(cv::countNonZero(overlay(cv::Rect(0, 0, 900, 100)).reshape(1)), 0) << "the old banner is gone";
  EXPECT_GT(alpha(overlay, 1125, 400), 0) << "the new box (x = 300 * 3.75) is drawn";
  EXPECT_EQ(overlay.at<cv::Vec4b>(1000, 1800), cv::Vec4b(1, 2, 3, 4)) << "untouched pixels are not rewritten";
}

// The board's window and recording: the camera frame with the annotation layer blended over it in software,
// the picture the display hardware shows.
TEST(ComposeOverlay, BlendsTheAnnotationLayerOverTheCameraFrame) {
  const cv::Mat camera(kFrame, CV_8UC4, cv::Scalar(10, 20, 30, 0));
  const fcw::FcwFrame f = frameWith({100, 100, 200, 200}, fcw::Level::Warning, 2.5);
  const cv::Mat out = fcw::composeOverlay(camera, f, kProc);
  ASSERT_EQ(out.type(), CV_8UC3);
  ASSERT_EQ(out.size(), kFrame);
  cv::Mat layer = cv::Mat::zeros(kFrame, CV_8UC4);
  fcw::OverlayPainter().paint(layer, f, kProc, kFrame);
  const cv::Vec4b edge = layer.at<cv::Vec4b>(500, 375);
  EXPECT_EQ(out.at<cv::Vec3b>(500, 375), cv::Vec3b(edge[0], edge[1], edge[2])) << "opaque box edge";
  EXPECT_EQ(out.at<cv::Vec3b>(560, 560), cv::Vec3b(10, 20, 30)) << "inside the box: the camera frame";
  EXPECT_EQ(out.at<cv::Vec3b>(1000, 1800), cv::Vec3b(10, 20, 30)) << "far away: the camera frame";
}
