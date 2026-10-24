#include "fcw/overlay.hpp"

#include <opencv2/imgproc.hpp>

namespace fcw {

namespace {

// Bounding box of text drawn at org (bottom-left), with the stroke and a pixel of margin.
cv::Rect textRect(const std::string& text, cv::Point org, double scale, int thickness) {
  int baseline = 0;
  const cv::Size s = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  const int m = thickness + 1;
  return {org.x - m, org.y - s.height - m, s.width + 2 * m, s.height + baseline + 2 * m};
}

}  // namespace

void OverlayPainter::paint(cv::Mat& overlay, const FcwFrame& f, cv::Size processing, cv::Size frame) {
  const cv::Rect all(0, 0, overlay.cols, overlay.rows);
  for (const cv::Rect& r : drawn_) overlay(r & all).setTo(cv::Scalar::all(0));
  drawn_.clear();
  auto mark = [&](const cv::Rect& r) {
    if (const cv::Rect c = r & all; c.area() > 0) drawn_.push_back(c);
  };

  const double sx = static_cast<double>(frame.width) / processing.width, sy = static_cast<double>(frame.height) / processing.height;
  for (const ObjectResult& o : f.objects) {
    const cv::Scalar color = o.level == Level::Critical ? cv::Scalar(0, 0, 255, 255)
                             : o.level == Level::Warning ? cv::Scalar(0, 165, 255, 255) : cv::Scalar(160, 160, 160, 200);
    const int thickness = o.level == Level::None ? 2 : 4;
    const cv::Rect r(cv::Point(static_cast<int>(o.bbox.x1 * sx), static_cast<int>(o.bbox.y1 * sy)),
                     cv::Point(static_cast<int>(o.bbox.x2 * sx), static_cast<int>(o.bbox.y2 * sy)));
    cv::rectangle(overlay, r, color, thickness);
    mark({r.x - thickness, r.y - thickness, r.width + 2 * thickness + 1, r.height + 2 * thickness + 1});
    if (o.estimate && o.estimate->ttc_s) {
      const std::string label = cv::format("%.1f s", *o.estimate->ttc_s);
      const cv::Point org = r.tl() + cv::Point(0, -6);
      cv::putText(overlay, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
      mark(textRect(label, org, 0.8, 2));
    }
  }
  if (f.level() != Level::None) {
    const bool critical = f.level() == Level::Critical;
    const std::string text = critical ? "BRAKE" : "COLLISION WARNING";
    const cv::Point org(24, 56);
    const cv::Rect box = textRect(text, org, 1.6, 4) + cv::Size(16, 16) - cv::Point(8, 8);
    overlay(box & all).setTo(cv::Scalar(0, 0, 0, 160));  // translucent background: readable over any scene
    cv::putText(overlay, text, org, cv::FONT_HERSHEY_SIMPLEX, 1.6, critical ? cv::Scalar(0, 0, 255, 255) : cv::Scalar(0, 165, 255, 255), 4);
    mark(box);
  }
}

}  // namespace fcw
