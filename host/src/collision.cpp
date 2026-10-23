// Port of collision_avoidance/fcw/collision.py.
#include "fcw/collision.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace fcw {

namespace {

// find() first: parallel workers may only look up their own prepared track (std::map::find is
// safe concurrently, operator[] is not); insert only when the track is new (serial use).
template <class Map>
typename Map::mapped_type& entry(Map& m, int id) {
  auto it = m.find(id);
  if (it == m.end()) it = m.emplace(id, typename Map::mapped_type{}).first;
  return it->second;
}

// Least-squares d(value)/dt; none without 3 samples over a real time span (live cameras repeat timestamps).
template <class It, class T, class V>
std::optional<double> slope(It begin, It end, T time, V value) {
  const auto n = std::distance(begin, end);
  if (n < 3) return std::nullopt;
  if (time(*std::prev(end)) - time(*begin) <= 1e-3) return std::nullopt;
  double tm = 0, vm = 0;
  for (auto it = begin; it != end; ++it) { tm += time(*it); vm += value(*it); }
  tm /= n;
  vm /= n;
  double num = 0, den = 0;
  for (auto it = begin; it != end; ++it) { num += (time(*it) - tm) * (value(*it) - vm); den += (time(*it) - tm) * (time(*it) - tm); }
  return num / den;
}

}  // namespace

CourseResult CourseChecker::update(int track_id, double t, const Box& b, const std::string& cls, double heading_x,
                                   double horizon_y, std::optional<double> ttc_s, double yaw_rate_rps, cv::Size image,
                                   std::optional<cv::Matx33d> rotation_h, std::optional<double> ref_x) {
  CourseResult res;
  res.r = ((b.x1 + b.x2) / 2 - heading_x) / std::max(b.x2 - b.x1, 1.0);
  auto& hist = entry(history_, track_id);
  hist.emplace_back(t, res.r);
  while (!hist.empty() && t - hist.front().first > cfg_.history_s) hist.pop_front();
  res.entry_speed_mps = entrySpeed(track_id, t, b, ref_x.value_or(heading_x), cls, image, rotation_h);
  const auto width = cfg_.class_width_m.find(cls);
  if (width == cfg_.class_width_m.end()) return res;
  res.width_m = width->second;
  const double lateral = res.r * width->second;
  const double overlap = std::min(lateral + width->second / 2, cfg_.ego_width_m / 2) - std::max(lateral - width->second / 2, -cfg_.ego_width_m / 2);
  res.in_path = b.y2 >= horizon_y && overlap >= cfg_.min_overlap * std::min(width->second, cfg_.ego_width_m);
  res.threshold = 0.5 * (1.0 + (cfg_.ego_width_m + cfg_.margin_m) / width->second);
  if (std::abs(yaw_rate_rps) > cfg_.turn_yaw_rate_rps) res.threshold *= cfg_.turn_widen;
  if (b.y2 < horizon_y || !ttc_s) return res;
  const double r_dot = slope(hist.begin(), hist.end(), [](const auto& s) { return s.first; }, [](const auto& s) { return s.second; }).value_or(0.0);
  res.r_contact = res.r + r_dot * *ttc_s;
  res.on_course = std::abs(*res.r_contact) < res.threshold;
  return res;
}

std::optional<double> CourseChecker::entrySpeed(int track_id, double t, const Box& b, double ref_x, const std::string& cls,
                                                cv::Size image, const std::optional<cv::Matx33d>& rotation_h) {
  auto& edges = entry(edges_, track_id);
  const double side = (b.x1 + b.x2) / 2 >= ref_x ? 1.0 : -1.0;
  const double x_out = side > 0 ? b.x2 : b.x1;
  const bool clipped = !image.empty() && (b.x1 <= 1 || b.y1 <= 1 || b.x2 >= image.width - 1 || b.y2 >= image.height - 1);
  if (clipped || (!edges.empty() && edges.back().side != side)) edges.clear();  // no true edge; the outer edge switched
  double& rot = entry(rot_px_, track_id);
  if (edges.empty()) {
    rot = 0;
  } else if (rotation_h) {
    const cv::Vec3d p = *rotation_h * cv::Vec3d(x_out, (b.y1 + b.y2) / 2, 1.0);
    rot += p[0] / p[2] - x_out;
  }
  if (!clipped) edges.push_back({t, (x_out - rot - ref_x) / std::max(b.y2 - b.y1, 1.0), side});
  while (!edges.empty() && t - edges.front().t > cfg_.history_s) edges.pop_front();
  const auto height = cfg_.class_height_m.find(cls);
  const auto q_dot = slope(edges.begin(), edges.end(), [](const Edge& e) { return e.t; }, [](const Edge& e) { return e.q; });
  if (height == cfg_.class_height_m.end() || !q_dot) return std::nullopt;
  return -side * *q_dot * height->second;
}

void CourseChecker::prepare(const std::vector<int>& ids) {
  for (int id : ids) {
    history_[id];
    edges_[id];
    rot_px_[id];
  }
}

void CourseChecker::retain(const std::vector<int>& ids) {
  for (auto it = history_.begin(); it != history_.end();)
    it = std::find(ids.begin(), ids.end(), it->first) == ids.end() ? history_.erase(it) : std::next(it);
  for (auto it = edges_.begin(); it != edges_.end();)
    it = std::find(ids.begin(), ids.end(), it->first) == ids.end() ? edges_.erase(it) : std::next(it);
  for (auto it = rot_px_.begin(); it != rot_px_.end();)
    it = std::find(ids.begin(), ids.end(), it->first) == ids.end() ? rot_px_.erase(it) : std::next(it);
}

Level WarningFsm::step(bool on_course, std::optional<double> ttc_s, std::optional<double> sigma, int recent_methods) {
  const bool can_raise = on_course && ttc_s && recent_methods >= cfg_.min_methods;
  const bool critical = can_raise && *ttc_s + sigma.value_or(0.0) <= cfg_.critical_ttc_s;
  const bool warn = can_raise && *ttc_s <= cfg_.warn_ttc_s;
  if (level_ == Level::None) {
    warn_count_ = warn ? warn_count_ + 1 : 0;
    if (critical) enter(Level::Critical);
    else if (warn_count_ >= cfg_.warn_frames) enter(Level::Warning);
  } else if (level_ == Level::Warning) {
    if (critical) {
      enter(Level::Critical);
    } else {
      const bool clearing = !on_course || !ttc_s || *ttc_s > cfg_.warn_clear_ttc_s;
      clear_count_ = clearing ? clear_count_ + 1 : 0;
      if (clear_count_ >= cfg_.clear_frames) enter(Level::None);
    }
  } else {
    const bool clearing = !on_course || !ttc_s || *ttc_s > cfg_.critical_clear_ttc_s;
    clear_count_ = clearing ? clear_count_ + 1 : 0;
    if (clear_count_ >= cfg_.clear_frames) enter(Level::Warning);
  }
  return level_;
}

}  // namespace fcw
