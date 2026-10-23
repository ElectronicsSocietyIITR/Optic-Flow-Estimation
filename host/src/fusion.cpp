// Port of collision_avoidance/fcw/fusion.py.
#include "fcw/fusion.hpp"

#include <algorithm>
#include <cmath>

namespace fcw {

namespace {
double lookup(const std::map<std::string, double>& m, const std::string& k, double fallback) {
  const auto it = m.find(k);
  return it == m.end() ? fallback : it->second;
}
}  // namespace

void EtaFilter::predict(double dt) {
  if (!eta) return;
  const double denom = std::max(1.0 - *eta * dt, 0.1);
  const double jac = 1.0 / (denom * denom);
  eta = *eta / denom;
  var = jac * jac * var + cfg_->process_var_per_s * dt;
}

Measurement EtaFilter::effective(const Measurement& m) const {
  const double floor = lookup(cfg_->rel_sigma_floor, m.method, 0.0) * std::abs(m.eta) + cfg_->abs_sigma_floor;
  return {m.method, m.eta, std::max(m.var * lookup(cfg_->var_scale, m.method, 1.0), floor * floor)};
}

std::vector<Measurement> EtaFilter::initialise(double t, const std::vector<Measurement>& ms) {
  if (ms.empty()) return {};
  double wsum = 0, weta = 0;
  for (const auto& m : ms) { wsum += 1.0 / m.var; weta += m.eta / m.var; }
  eta = weta / wsum;
  var = 1.0 / wsum;
  rejects_ = 0;
  for (const auto& m : ms) last_accept_[m.method] = t;
  return ms;
}

std::vector<Measurement> EtaFilter::update(double t, const std::vector<Measurement>& ms) {
  std::vector<Measurement> scaled;
  for (const auto& m : ms) scaled.push_back(effective(m));
  if (!eta || rejects_ >= cfg_->max_consecutive_rejects) return initialise(t, scaled);
  std::vector<Measurement> accepted;
  for (const auto& m : scaled) {
    const double s = var + m.var;
    const double innovation = m.eta - *eta;
    if (innovation * innovation / s > cfg_->gate_sigma2) continue;
    const double gain = var / s;
    eta = *eta + gain * innovation;
    var *= 1.0 - gain;
    last_accept_[m.method] = t;
    accepted.push_back(m);
  }
  rejects_ = (!accepted.empty() || scaled.empty()) ? 0 : rejects_ + 1;
  return accepted;
}

std::set<std::string> EtaFilter::recentMethods(double t) const {
  std::set<std::string> out;
  for (const auto& [m, ts] : last_accept_)
    if (t - ts <= cfg_->recent_window_s) out.insert(m);
  return out;
}

std::optional<TtcEstimate> TtcFusion::update(int track_id, double t, double dt, const std::vector<Measurement>& ms) {
  auto it = filters_.find(track_id);  // parallel workers: prepared entries are only looked up
  if (it == filters_.end()) it = filters_.try_emplace(track_id, &cfg_).first;
  EtaFilter& f = it->second;
  f.predict(dt);
  TtcEstimate est;
  est.accepted = f.update(t, ms);
  if (!f.eta) return std::nullopt;
  est.track_id = track_id;
  est.eta = *f.eta;
  est.eta_var = f.var;
  est.recent_methods = f.recentMethods(t);
  // Prediction alone drives eta up hyperbolically: without recent data there is no TTC.
  if (*f.eta > cfg_.min_eta && !est.recent_methods.empty()) {
    est.ttc_s = 1.0 / *f.eta;
    est.sigma_ttc_s = std::sqrt(f.var) / (*f.eta * *f.eta);
  }
  return est;
}

void TtcFusion::retain(const std::vector<int>& ids) {
  for (auto it = filters_.begin(); it != filters_.end();)
    it = std::find(ids.begin(), ids.end(), it->first) == ids.end() ? filters_.erase(it) : std::next(it);
}

void TtcFusion::prepare(const std::vector<int>& ids) {
  for (int id : ids) filters_.try_emplace(id, &cfg_);
}

const EtaFilter* TtcFusion::filter(int track_id) const {
  const auto it = filters_.find(track_id);
  return it == filters_.end() ? nullptr : &it->second;
}

}  // namespace fcw
