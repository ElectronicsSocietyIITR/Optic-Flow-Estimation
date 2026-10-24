// GStreamer capture (see fcw/board.hpp).
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <chrono>
#include <stdexcept>

#include "fcw/board.hpp"

namespace fcw::board {

namespace {

std::string pipelineFor(const std::string& source) {
  const std::string tail = " ! videoconvert ! video/x-raw,format=BGRx ! appsink name=sink max-buffers=2 sync=false";
  if (source.rfind("/dev/video", 0) == 0) return "v4l2src device=" + source + tail + " drop=true";
  return "filesrc location=\"" + source + "\" ! decodebin" + tail + " drop=false";
}

struct Pipeline {
  GstElement* pipe = nullptr;
  GstElement* sink = nullptr;
  ~Pipeline() {
    if (pipe) gst_element_set_state(pipe, GST_STATE_NULL);
    if (sink) gst_object_unref(sink);
    if (pipe) gst_object_unref(pipe);
  }
};

}  // namespace

CaptureSource gstCapture(const std::string& source) {
  gst_init(nullptr, nullptr);
  auto p = std::make_shared<Pipeline>();
  GError* err = nullptr;
  p->pipe = gst_parse_launch(pipelineFor(source).c_str(), &err);
  if (!p->pipe) {
    const std::string msg = err ? err->message : "unknown error";
    if (err) g_error_free(err);
    throw std::runtime_error("GStreamer: " + msg);
  }
  p->sink = gst_bin_get_by_name(GST_BIN(p->pipe), "sink");
  if (gst_element_set_state(p->pipe, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) throw std::runtime_error("GStreamer: cannot play " + source);
  const auto start = std::chrono::steady_clock::now();
  return [p, start]() -> std::optional<CapturedFrame> {
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(p->sink));
    if (!sample) return std::nullopt;  // end of stream
    GstStructure* st = gst_caps_get_structure(gst_sample_get_caps(sample), 0);
    int w = 0, h = 0;
    gst_structure_get_int(st, "width", &w);
    gst_structure_get_int(st, "height", &h);
    GstBuffer* buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_READ);
    const size_t stride = h > 0 ? map.size / h : 0;  // rows may be padded
    CapturedFrame f;
    f.bgr = cv::Mat(h, w, CV_8UC4, map.data, stride).clone();
    f.t = GST_BUFFER_PTS_IS_VALID(buf) ? GST_BUFFER_PTS(buf) / 1e9
                                        : std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);
    return f;
  };
}

}  // namespace fcw::board
