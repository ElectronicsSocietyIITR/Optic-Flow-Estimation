#include "fcw/yolo_decode.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <string>

namespace fcw {

namespace {

const std::map<int, std::string>& cocoNames() {
  static const std::map<int, std::string> names{{0, "person"}, {1, "bicycle"}, {2, "car"}, {3, "motorcycle"}, {5, "bus"}, {7, "truck"}};
  return names;
}

// Expected bin of one side's DFL distribution (softmax over reg_max logits).
double dflSide(const float* logits, int reg_max) {
  const float mx = *std::max_element(logits, logits + reg_max);
  double sum = 0, mean = 0;
  for (int i = 0; i < reg_max; ++i) {
    const double e = std::exp(static_cast<double>(logits[i] - mx));
    sum += e;
    mean += i * e;
  }
  return mean / sum;
}

double iou(const Box& a, const Box& b) {
  const double w = std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const double h = std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const double inter = w * h;
  return inter / (a.width() * a.height() + b.width() * b.height() - inter);
}

}  // namespace

std::vector<Detection> decodeYolo(const std::vector<cv::Mat>& raw_heads, cv::Size input, const YoloDecodeConfig& cfg) {
  const double logit_thr = std::log(cfg.conf / (1.0 - cfg.conf));
  const int box_ch = 4 * cfg.reg_max;
  struct Cand { Box box; double score; int cls; };
  std::vector<Cand> cands;
  for (const cv::Mat& head : raw_heads) {
    const double stride = static_cast<double>(input.width) / head.cols;
    for (int y = 0; y < head.rows; ++y)
      for (int x = 0; x < head.cols; ++x) {
        const float* p = head.ptr<float>(y, x);
        int best = -1;
        float best_logit = 0;
        for (int c : cfg.classes)
          if (best < 0 || p[box_ch + c] > best_logit) { best = c; best_logit = p[box_ch + c]; }
        if (!(best_logit > logit_thr)) continue;  // most cells end here: no softmax, no sigmoid
        const double l = dflSide(p, cfg.reg_max), t = dflSide(p + cfg.reg_max, cfg.reg_max);
        const double r = dflSide(p + 2 * cfg.reg_max, cfg.reg_max), b = dflSide(p + 3 * cfg.reg_max, cfg.reg_max);
        const double ax = x + 0.5, ay = y + 0.5;
        const double cx = (ax - l + ax + r) / 2 * stride, cy = (ay - t + ay + b) / 2 * stride;
        const double w = (l + r) * stride, h = (t + b) * stride;
        cands.push_back({{cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2}, 1.0 / (1.0 + std::exp(-best_logit)), best});
      }
  }
  std::vector<size_t> order(cands.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cands[a].score > cands[b].score; });
  std::vector<Detection> out;
  std::vector<const Cand*> kept;
  for (size_t i : order) {
    const Cand& c = cands[i];
    bool suppressed = false;
    for (const Cand* k : kept)
      if (k->cls == c.cls && iou(k->box, c.box) > cfg.iou) { suppressed = true; break; }  // class-aware, as batched_nms
    if (suppressed) continue;
    kept.push_back(&c);
    out.push_back({c.box, cocoNames().at(c.cls), c.score});
  }
  return out;
}

}  // namespace fcw
