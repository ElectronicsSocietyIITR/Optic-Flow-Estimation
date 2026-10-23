// Per-track fusion of inverse-TTC measurements (one-state EKF on eta = 1/TTC).
// Port of collision_avoidance/fcw/fusion.py.
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "fcw/types.hpp"

namespace fcw {

struct FusionConfig {
  double process_var_per_s = 0.25;
  double gate_sigma2 = 9.0;
  int max_consecutive_rejects = 5;
  double min_eta = 0.05;
  double recent_window_s = 0.5;
  // Calibrated on EvTTC (plan 1 Task 13).
  std::map<std::string, double> var_scale{{"looming", 0.19}, {"scale", 6.61}, {"horn", 68.13}, {"divergence", 1012.81}};
  std::map<std::string, double> rel_sigma_floor{{"looming", 0.0}, {"scale", 0.05}, {"horn", 0.15}, {"divergence", 0.30}};
  double abs_sigma_floor = 0.02;
};

struct TtcEstimate {
  int track_id = 0;
  double eta = 0;
  double eta_var = 0;
  std::optional<double> ttc_s;
  std::optional<double> sigma_ttc_s;
  std::vector<Measurement> accepted;    // accepted this frame (effective variances)
  std::set<std::string> recent_methods; // accepted within recent_window_s
};

class EtaFilter {
 public:
  explicit EtaFilter(const FusionConfig* cfg) : cfg_(cfg) {}
  void predict(double dt);
  std::vector<Measurement> update(double t, const std::vector<Measurement>& ms);
  std::set<std::string> recentMethods(double t) const;
  std::optional<double> eta;
  double var = 0;

 private:
  Measurement effective(const Measurement& m) const;
  std::vector<Measurement> initialise(double t, const std::vector<Measurement>& ms);
  const FusionConfig* cfg_;
  int rejects_ = 0;
  std::map<std::string, double> last_accept_;
};

class TtcFusion {
 public:
  explicit TtcFusion(FusionConfig cfg = {}) : cfg_(std::move(cfg)) {}
  std::optional<TtcEstimate> update(int track_id, double t, double dt, const std::vector<Measurement>& ms);
  void retain(const std::vector<int>& track_ids);
  // Creates the filters of these tracks now, so update() from parallel workers never inserts.
  void prepare(const std::vector<int>& track_ids);
  const EtaFilter* filter(int track_id) const;

 private:
  FusionConfig cfg_;
  std::map<int, EtaFilter> filters_;
};

}  // namespace fcw
