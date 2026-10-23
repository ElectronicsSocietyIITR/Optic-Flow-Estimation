// Port of collision_avoidance/fcw/pipeline.py.
#include "fcw/pipeline.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

namespace fcw {

namespace {
double msSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace

Level FcwFrame::level() const {
  Level out = Level::None;
  for (const auto& o : objects) out = std::max(out, o.level);
  return out;
}

const ObjectResult* FcwFrame::threat() const {
  const ObjectResult* best = nullptr;
  for (const auto& o : objects) {
    if (!o.course || !o.course->on_course || !o.estimate || !o.estimate->ttc_s) continue;
    if (!best || *o.estimate->ttc_s < *best->estimate->ttc_s) best = &o;
  }
  return best;
}

FlowSource disFlow() {
  auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
  return [dis](const cv::Mat& prev, const cv::Mat& curr) {
    cv::Mat flow;
    dis->calc(prev, curr, flow);
    return flow;
  };
}

FcwPipeline::FcwPipeline(FcwConfig cfg, DetectorFn detector, FlowSource flow)
    : cfg_(std::move(cfg)),
      detector_(std::move(detector)),
      flow_(std::move(flow)),
      K_(cfg_.K()),
      pp_(K_(0, 2), K_(1, 2)),
      tracker_(cfg_.tracker),
      ego_(K_, cfg_.ego),
      fusion_(cfg_.fusion),
      course_(cfg_.course),
      closing_(cfg_.closing),
      engine_(std::make_unique<SoftwareMomentEngine>(cfg_.horn, cfg_.divergence)) {}

cv::Size processingSize(int width, int height, int target_width) {
  const int h = static_cast<int>(std::nearbyint(static_cast<double>(target_width) * height / width / 32.0)) * 32;  // half to even, as Python
  return {target_width, std::clamp(h, 64, 512)};
}

FcwFrame FcwPipeline::process(const cv::Mat& frame_bgr, double t) { return objectStage(frameStage(frame_bgr, t)); }

FrameState FcwPipeline::frameStage(const cv::Mat& frame_bgr, double t) {
  FrameState s;
  s.start = std::chrono::steady_clock::now();
  FcwFrame& out = s.out;
  ++index_;
  double dt = prev_t_ ? t - *prev_t_ : 0.0;
  if (dt <= 0) dt = 1.0 / cfg_.default_fps;  // first frame, or a repeated/out-of-order timestamp
  if (!prev_t_ || t > *prev_t_) prev_t_ = t;  // never move the clock backwards
  const cv::Size size(cfg_.frame_width, cfg_.frame_height);
  if (frame_bgr.size() == size) out.frame = frame_bgr;  // already prepared (the runtime's T2 resized it)
  else cv::resize(frame_bgr, out.frame, size);
  cv::cvtColor(out.frame, s.gray, cv::COLOR_BGR2GRAY);
  s.prev_gray = ring_.empty() ? cv::Mat() : std::get<2>(ring_.back());

  auto mark = std::chrono::steady_clock::now();
  out.flow = s.prev_gray.empty() ? cv::Mat::zeros(s.gray.size(), CV_32FC2) : flow_(s.prev_gray, s.gray);
  out.timings_ms["flow"] = msSince(mark);

  mark = std::chrono::steady_clock::now();
  std::vector<Detection> detections;
  for (const auto& d : detector_(out.frame))
    if (cfg_.course.class_width_m.count(d.cls)) detections.push_back(d);
  out.timings_ms["detection"] = msSince(mark);

  // Ego-rotation needs flow N and the boxes of N-1 only, so it runs alongside the tracker update.
  mark = std::chrono::steady_clock::now();
  std::vector<Box> prev_boxes;
  for (const auto& [id, tr] : tracker_.tracks()) prev_boxes.push_back(tr.bbox());
  auto ego = std::async(std::launch::async, [&] {
    const auto t0 = std::chrono::steady_clock::now();
    EgoMotion e = s.prev_gray.empty() ? EgoMotion{} : ego_.update(out.flow, prev_boxes);
    return std::make_pair(e, msSince(t0));
  });
  s.tracks = tracker_.update(detections, dt, index_);
  out.timings_ms["track"] = msSince(mark);
  const auto [ego_result, ego_ms] = ego.get();
  out.ego = ego_result;
  out.timings_ms["ego"] = ego_ms;
  out.heading = cfg_.heading_from_foe && out.ego.foe ? *out.ego.foe : pp_;
  s.yaw_rate = out.ego.rvec[1] / dt;
  ring_.emplace_back(index_, t, s.gray);
  if (static_cast<int>(ring_.size()) > kHistory) ring_.pop_front();
  s.ring = ring_;  // cv::Mat headers only: cheap, and the next frame stage may change ring_
  out.index = index_;
  out.time_s = t;
  out.dt_s = dt;
  return s;
}

FcwFrame FcwPipeline::objectStage(FrameState s) {
  FcwFrame& out = s.out;
  const auto mark = std::chrono::steady_clock::now();
  for (const char* k : {"looming", "scale", "horn", "divergence", "fusion_course"}) out.timings_ms[k] = 0.0;
  const std::vector<const ScaleTrack*> selected = select(s.tracks);
  std::vector<int> ids;
  for (const ScaleTrack* tr : selected) ids.push_back(tr->id);
  fusion_.prepare(ids);  // workers then only touch existing entries of their own track
  course_.prepare(ids);
  const cv::Mat flow_q = quantiseFlow(out.flow);  // divergence runs on the PL engine's flow format

  const int n = static_cast<int>(selected.size());
  // Option A: one engine call for every box, after the tracker, when each box's pyramid level and last
  // frame's fit are known; the workers then only solve.
  std::vector<BoxSums> sums;
  std::vector<bool> fit_used(n, false);
  if (!s.prev_gray.empty()) {
    const auto engine_mark = std::chrono::steady_clock::now();
    std::vector<EngineBox> boxes(n);
    for (int i = 0; i < n; ++i) {
      const ScaleTrack& tr = *selected[i];
      const EtaFilter* prior = fusion_.filter(tr.id);
      boxes[i].box = tr.bbox();
      boxes[i].level = chooseLevel(prior ? prior->eta : std::nullopt, boxes[i].box, std::hypot(tr.x[4], tr.x[5]), out.dt_s);
      const auto fit = fits_.find(tr.id);
      if (fit != fits_.end() && out.index - fit->second.second <= cfg_.divergence.max_fit_age) {
        boxes[i].fit = fitToFixed(fit->second.first.first, fit->second.first.second);
        fit_used[i] = true;
      }
    }
    sums = engine_->run(s.prev_gray, s.gray, flow_q, boxes, pp_);
    out.timings_ms["engine"] = msSince(engine_mark);
  }
  std::vector<ObjectResult> objects(n);
  std::vector<std::map<std::string, double>> timings(n);
  std::vector<std::optional<AffineFit>> fits(n);
  if (!pool_ && cfg_.object_workers > 0) pool_ = std::make_unique<ThreadPool>(cfg_.object_workers);
  auto one = [&](int i) { objects[i] = assess(*selected[i], s, flow_q, sums.empty() ? nullptr : &sums[i], fit_used[i], timings[i], fits[i]); };
  if (pool_) {
    pool_->parallelFor(n, one);
  } else {
    for (int i = 0; i < n; ++i) one(i);
  }
  for (int i = 0; i < n; ++i) {  // merge in selection order: the result never depends on thread timing
    for (const auto& [k, v] : timings[i]) out.timings_ms[k] += v;
    if (fits[i]) fits_[selected[i]->id] = {*fits[i], out.index};
  }
  out.objects = std::move(objects);

  const auto warn_mark = std::chrono::steady_clock::now();
  warn(out.objects, s.tracks, out.flow, out.ego, out.dt_s);
  out.timings_ms["warn"] = msSince(warn_mark);
  std::vector<int> live;
  for (const auto& [id, tr] : s.tracks) live.push_back(id);
  fusion_.retain(live);
  course_.retain(live);
  closing_.retain(live);
  for (auto it = fsms_.begin(); it != fsms_.end();) it = s.tracks.count(it->first) ? std::next(it) : fsms_.erase(it);
  for (auto it = in_path_streak_.begin(); it != in_path_streak_.end();) it = s.tracks.count(it->first) ? std::next(it) : in_path_streak_.erase(it);
  for (auto it = entry_streak_.begin(); it != entry_streak_.end();) it = s.tracks.count(it->first) ? std::next(it) : entry_streak_.erase(it);
  for (auto it = fits_.begin(); it != fits_.end();) it = s.tracks.count(it->first) ? std::next(it) : fits_.erase(it);
  on_course_.clear();
  for (const auto& o : out.objects)
    if (o.course && o.course->in_path) on_course_.insert(o.track_id);
  out.timings_ms["ttc"] = msSince(mark);
  out.timings_ms["total"] = msSince(s.start);
  return std::move(out);
}

std::vector<const ScaleTrack*> FcwPipeline::select(const std::map<int, ScaleTrack>& tracks) const {
  // Last frame's on-course tracks first, then the largest boxes (stable, like Python's sorted).
  std::vector<const ScaleTrack*> order;
  for (const auto& [id, tr] : tracks) order.push_back(&tr);
  std::stable_sort(order.begin(), order.end(), [&](const ScaleTrack* a, const ScaleTrack* b) {
    const bool ca = !on_course_.count(a->id), cb = !on_course_.count(b->id);
    if (ca != cb) return ca < cb;
    const Box ba = a->bbox(), bb = b->bbox();
    return -(ba.width() * ba.height()) < -(bb.width() * bb.height());
  });
  if (static_cast<int>(order.size()) > cfg_.max_ttc_tracks) order.resize(cfg_.max_ttc_tracks);
  return order;
}

namespace {
std::optional<std::pair<double, cv::Mat>> frameAt(const std::deque<std::tuple<int, double, cv::Mat>>& ring, int frame_index) {
  for (const auto& [index, time_s, gray] : ring)
    if (index == frame_index) return std::make_pair(time_s, gray);
  return std::nullopt;
}
}  // namespace

ObjectResult FcwPipeline::assess(const ScaleTrack& track, const FrameState& s, const cv::Mat& flow_q, const BoxSums* sums, bool fit_used,
                                 std::map<std::string, double>& timings, std::optional<AffineFit>& new_fit) {
  const FcwFrame& f = s.out;
  const double t = f.time_s, dt = f.dt_s;
  ObjectResult res;
  res.track_id = track.id;
  res.bbox = track.bbox();
  res.cls = track.cls;
  const Box& box = res.bbox;
  const EtaFilter* prior = fusion_.filter(track.id);
  const std::optional<double> eta_prior = prior ? prior->eta : std::nullopt;
  auto mark = std::chrono::steady_clock::now();
  auto lap = [&](const char* key) {  // add the time since the last lap to timings[key]
    const auto now = std::chrono::steady_clock::now();
    timings[key] += std::chrono::duration<double, std::milli>(now - mark).count();
    mark = now;
  };
  if (auto m = looming(track)) res.measurements.push_back(*m);
  lap("looming");
  if (!s.prev_gray.empty() && !track.lost) {
    const int k = cfg_.scale_enabled ? chooseGap(eta_prior, dt, static_cast<int>(s.ring.size()) - 1) : 0;
    const auto past = frameAt(s.ring, f.index - k);
    const auto box_tk = track.boxAt(f.index - k);
    if (k && past && box_tk && t > past->first) {
      // The real span, not k * dt: frames can arrive irregularly.
      if (auto m = scaleTtc(s.gray, past->second, box, *box_tk, k, (t - past->first) / k, cfg_.scale)) res.measurements.push_back(*m);
    }
    lap("scale");
    if (sums && sums->horn)
      if (auto m = hornSolve(*sums->horn, dt, cfg_.horn)) res.measurements.push_back(*m);
    lap("horn");
    // Engine moments over last frame's fit inliers, if enough of the box follows it (divergenceMask's
    // rule); otherwise RANSAC here, in software.
    std::optional<FixedMoments> fixed;
    const auto rect = flowRect(box, cfg_.divergence.shrink, flow_q.cols, flow_q.rows);
    if (fit_used && rect && sums && sums->flow && (*sums->flow)[0] >= cfg_.divergence.min_inlier_share * rect->area()) {
      fixed = sums->flow;
    } else {
      fixed = flowMomentsFixed(flow_q, box, pp_, cfg_.divergence.shrink, robustMask(f.flow, box, pp_, cfg_.divergence));
    }
    const std::optional<FlowMoments> moments = fixed ? std::optional<FlowMoments>(dequantiseMoments(*fixed)) : std::nullopt;
    const double rot = rotationDivergence(f.ego.R, K_, {box.cx(), box.cy()});
    if (auto m = divergenceTtc(moments, box, rot, 1, dt, cfg_.divergence)) {
      res.measurements.push_back(*m);
      if (auto fit = affineFromMoments(*moments)) new_fit = *fit;
    }
  }
  lap("divergence");
  res.estimate = fusion_.update(track.id, t, dt, res.measurements);
  const std::optional<double> ttc = res.estimate ? res.estimate->ttc_s : std::nullopt;
  res.course = course_.update(track.id, t, box, track.cls, f.heading.x, f.heading.y, ttc, s.yaw_rate, s.gray.size(),
                              cv::Matx33d(K_ * f.ego.R * K_.inv()), pp_.x);
  lap("fusion_course");
  return res;
}

// Only the lead vehicle may warn: the nearest object in our path, in path for a few frames.
// A lost track is a prediction, not an observation: it is never the lead.
// Clearly oncoming objects (closing much faster than we drive, see closing.hpp) are never the lead.
void FcwPipeline::warn(std::vector<ObjectResult>& objects, const std::map<int, ScaleTrack>& tracks, const cv::Mat& flow,
                       const EgoMotion& ego, double dt) {
  std::optional<double> c;
  if (cfg_.closing.enabled && ego.foe) {
    std::vector<Box> boxes;
    for (const auto& o : objects) boxes.push_back(o.bbox);
    c = closing_.updateC(roadConstant(flow, *ego.foe, ego.R, K_, boxes, cfg_.closing));
  }
  // Lead candidates: in our path now, or steadily entering it (predicted in path at contact, moving
  // towards it, close enough). Crossing pedestrians and cut-ins only overlap us shortly before contact.
  const CourseConfig& cc = cfg_.course;
  const ObjectResult* lead = nullptr;
  double lead_dist = 0;
  for (auto& o : objects) {
    const bool lost = tracks.at(o.track_id).lost > 0;
    const auto& co = o.course;
    bool in_path = co && co->in_path && !lost;
    const std::optional<double> ttc = o.estimate ? o.estimate->ttc_s : std::nullopt;
    bool entering = co && !lost && !co->in_path && co->on_course && co->r_contact && ttc && *ttc != 0 && co->width_m &&
                    std::abs(co->r) * *co->width_m <= cc.max_entry_m && co->r * (*co->r_contact - co->r) < 0 &&
                    co->entry_speed_mps && *co->entry_speed_mps >= cc.min_entry_speed_mps;
    if ((in_path || entering) && cfg_.closing.enabled) {
      std::optional<double> kappa;
      if (c && ego.foe && ttc)
        if (auto s = staticTtc(*c, o.bbox, *ego.foe, cfg_.closing)) kappa = *s * dt / *ttc;
      o.kappa = closing_.update(o.track_id, kappa);
      if (o.kappa && *o.kappa > cfg_.closing.oncoming_kappa) in_path = entering = false;
    }
    // Entry is a prediction, so it needs positive evidence the object is not oncoming: with kappa
    // unknown (e.g. we are stopped, so the road does not move) only objects already in our path warn.
    entering = entering && cfg_.closing.enabled && o.kappa.has_value();
    int& streak = in_path_streak_[o.track_id];
    streak = in_path ? streak + 1 : 0;
    int& entry = entry_streak_[o.track_id];
    entry = (in_path || entering) ? entry + 1 : 0;
    if (!entry) continue;
    const double dist = *o.course->width_m / std::max(o.bbox.x2 - o.bbox.x1, 1.0);  // distance ~ W_obj / w
    if (!lead || dist < lead_dist) { lead = &o; lead_dist = dist; }
  }
  for (auto& o : objects) {
    const bool ok = lead && o.track_id == lead->track_id && o.course->on_course &&
                    (in_path_streak_[o.track_id] >= cc.in_path_frames || entry_streak_[o.track_id] >= cc.entry_frames);
    WarningFsm& fsm = fsms_.try_emplace(o.track_id, cfg_.warning).first->second;
    o.level = fsm.step(ok, o.estimate ? o.estimate->ttc_s : std::nullopt, o.estimate ? o.estimate->sigma_ttc_s : std::nullopt,
                       o.estimate ? static_cast<int>(o.estimate->recent_methods.size()) : 0);
  }
}

}  // namespace fcw
