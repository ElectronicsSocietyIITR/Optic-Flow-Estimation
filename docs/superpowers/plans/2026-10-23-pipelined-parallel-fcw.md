# Pipelined, parallel FCW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the FCW system pipelined and parallel enough for 30 FPS on the KV260, replace the two expensive algorithms, and fix the 16 issues of the low-level review.

**Architecture:** The Python package stays the reference; every change lands there first, then in C++ with golden parity, then (for per-pixel work) in HLS with bit-exact C-simulation. The C++ pipeline is split into a frame stage (ego-rotation ∥ tracker, road constant) and an object stage (one engine call per pyramid level, then a deterministic worker pool), run by a 4-stage threaded runtime.

**Tech Stack:** Python 3.11 (numpy, OpenCV 4.10, torch for the GPU LK), C++17 (OpenCV 4.6, GoogleTest, WSL Ubuntu 24.04), Vitis HLS 2024.2 headers for C-simulation, Vitis Vision library (C-simulation only until device data is installed).

**Spec:** `docs/superpowers/specs/2026-10-23-pipelined-parallel-fcw-design.md`

## Global Constraints

- Commits: identity `Shyam-24o7 <266705345+Shyam-24o7@users.noreply.github.com>`; **no Co-Authored-By or Claude attribution** (user's standing rule); push to `origin` and `org`.
- Gates (spec 6): KITTI yaw RMS < 1.0 °/s; EvTTC fused median relative error ≤ 20 %; 5/5 approaches warned before true TTC 2.0 s.
- Python tests: `COLLISION_WANDB=0 .venv/Scripts/python -m pytest tests/fcw -q`; C++: `wsl -d Ubuntu-24.04 -- bash -lc 'cmake --build ~/fcw-build -j8 && ~/fcw-build/fcw_tests'`; HLS: `make -C hls/ttc_engine csim` in WSL.
- Goldens: `tools/export_golden.py --only <name>`; committed goldens in `host/tests/golden`, pipeline replays in `host/tests/golden_local` (git-ignored).
- Engine capacity: 16 boxes, 23 int64 sums per box (11 Horn + 12 flow); flow in int16 at 1/64 px; 300 MHz target.
- The serial `FcwPipeline::process` remains the reference the parallel runtime must reproduce exactly.

## Review Focus

1. Ego-rotation when a large object hides most of the background (few static points): the estimator holds the last estimate instead of returning a wrong rotation. Test in Task 1.
2. A track that was lost for frames, or is new: its stored affine fit is stale or missing, so divergence falls back to RANSAC. Test in Task 3.
3. Live sources drop frames, so frame indices skip: `FrameRing` look-back is by frame index, not slot count. Test in Task 13.
4. Zero, one, or more than 16 tracks through the worker pool: same results as serial. Test in Task 14.
5. Portrait or very small inputs to the processing-size rule: height clamps to [64, 512]. Test in Task 7.

---

### Task 1: Linear ego-rotation (Python)

**Files:** Modify `collision_avoidance/fcw/ego_rotation.py`; Test `tests/fcw/test_ego_rotation.py`

**Interfaces:**
- Produces: `estimate_linear(prev_pts, curr_pts, K, cfg, foe_init=None) -> EgoMotion | None`; `EgoRotationConfig.method: str = "linear"` (`"ransac"` keeps `estimate`); new fields `linear_rounds: int = 5`, `huber_px: float = 1.0`, `stationary_px: float = 0.15` (median residual flow below which the car is stopped), `min_static_share: float = 0.5`.
- `EgoRotationEstimator.update` dispatches on `cfg.method`, passes the previous FOE as `foe_init`.

- [ ] Write failing tests: `test_linear_recovers_yaw_and_pitch` (synthetic scene, yaw 2°/s and pitch 0.5°/s at 30 FPS; |Δrvec| < 2e-4 rad), `test_linear_foe_matches_translation` (FOE within 3 px), `test_linear_stationary_rotation_only` (pure rotation → `stationary=True`, `foe is None`), `test_linear_holds_when_background_hidden` (90 % of points on one moving object → `valid=False` after `update`).
- [ ] Run; expect FAIL (no `estimate_linear`).
- [ ] Implement: normalised coordinates; small-rotation flow `u_r = xy·ωx − (1+x²)·ωy + y·ωz`, `v_r = (1+y²)·ωx − xy·ωy − x·ωz`; alternate the 3×3 solve for ω (FOE fixed) and the 2×2 solve for the FOE (ω fixed) from `(u−u_r)(y−y_F) − (v−v_r)(x−x_F) = 0`, Huber weights from the residual in px, `linear_rounds` rounds; `R = Rodrigues(ω)`; inlier ratio = share of points with residual < `huber_px`; stationary when the median residual flow magnitude < `stationary_px`.
- [ ] Run tests; PASS. Run `python eval/kitti_yaw.py data/kitti/2011_09_26/2011_09_26_drive_{0005,0014,0091}_sync`; expect overall RMS < 1.0 °/s; record it and the per-frame time vs `"ransac"`.
- [ ] Commit `Ego-rotation: linear rotation + FOE least squares`.

### Task 2: Linear ego-rotation (C++)

**Files:** Modify `host/include/fcw/ego_rotation.hpp`, `host/src/ego_rotation.cpp`, `tools/export_golden.py` (exporter `ego` gains linear cases), `host/tests/test_ego_rotation.cpp`

**Interfaces:** Consumes Task 1. Produces `estimateLinear(const cv::Mat& prev, const cv::Mat& curr, const cv::Matx33d& K, const EgoRotationConfig&, std::optional<cv::Point2d> foe_init) -> std::optional<EgoMotion>`; `EgoRotationConfig::method` (`"linear"` default).

- [ ] Regenerate `ego` goldens with linear cases; add `EgoParity.LinearMatchesPython` (rvec 1e-9, FOE 1e-6 px, flags equal).
- [ ] Run; FAIL. Implement the port. Run; PASS (all 38 + new). Commit.

### Task 3: Divergence with a predicted-fit inlier mask (Python)

**Files:** Modify `collision_avoidance/fcw/divergence.py`, `collision_avoidance/fcw/pipeline.py`; Test `tests/fcw/test_divergence.py`

**Interfaces:**
- Produces: `FitQ = tuple[int, int, int, int, int, int]` (a0, a1, a2, b0, b1, b2: a0/b0 in 1/64 px scaled by 2^16; a1, a2, b1, b2 in 1/64 px per px scaled by 2^16); `fit_to_fixed(a, b) -> FitQ`; `predicted_mask_fixed(flow_q, rect, principal_point_int, fit_q, inlier_q) -> np.ndarray` (bool, rect-shaped): inlier when `|(u<<16) − P_u| ≤ inlier_q<<16` and the same for v, with `P_u = a0 + a1·x + a2·y` in int64; `flow_moments_fixed(flow_q, box, principal_point, shrink, mask=None)` (mask added); `DivergenceConfig.min_inlier_share: float = 0.5`, `max_fit_age: int = 1`.
- Pipeline keeps per track `(FitQ, frame_index)`; uses the predicted mask when the fit is from the previous frame, else `robust_mask` (RANSAC). Divergence then runs on `quantise_flow(flow)` with fixed moments (the board path).

- [ ] Failing tests: `test_predicted_mask_matches_float_residual` (same inliers as the float residual test on a synthetic affine field + outliers), `test_stale_fit_falls_back_to_ransac` (fit 2 frames old → RANSAC path used; spy on `robust_mask`), `test_low_inlier_share_falls_back`, `test_fixed_moments_with_mask_match_float` (within 1/64 px quantisation).
- [ ] Run; FAIL. Implement. Run; PASS. All FCW tests pass.
- [ ] Gate: `python eval/evttc_fcw.py --data data/evttc` → fused error ≤ 20 %, 5/5, divergence error within 2 points of 9.5 %. Commit.

### Task 4: Predicted-fit divergence (C++)

**Files:** Modify `host/include/fcw/divergence.hpp`, `host/src/divergence.cpp`, `host/src/pipeline.cpp`, `tools/export_golden.py` (`divergence` exporter), `host/tests/test_divergence.cpp`

**Interfaces:** Consumes Task 3. Produces `struct FitQ { int64_t a0, a1, a2, b0, b1, b2; }`, `fitToFixed`, `predictedMaskFixed`, `flowMomentsFixed(const cv::Mat& flow_q, const Box&, cv::Point2i pp, double shrink, const cv::Mat* mask)` returning the 12 int64 sums.

- [ ] Goldens with masked cases; tests `DivergenceParity.PredictedMask`, `DivergenceParity.FixedMomentsMasked` (exact integer equality). Implement; PASS; pipeline goldens regenerated; all C++ tests pass. Commit.

### Task 5: Road-constant grid and the scale-search ablation

**Files:** Modify `collision_avoidance/fcw/closing.py`, `host/include/fcw/closing.hpp` (grid 8×4), `collision_avoidance/fcw/pipeline.py` + `host` (`FcwConfig.scale_enabled: bool`); report `docs/superpowers/reports/2026-10-p1-pc-evaluation.md`

- [ ] Change the default grid to (8, 4); run `tests/fcw/test_closing.py` and the synthetic κ probe; κ readings within 5 % of today's (0.95 / 2.37 / 0.47).
- [ ] Run EvTTC with `scale_enabled=False`. If fused error ≤ 13 % and 5/5: default `False`, ring look-back becomes 1. Else keep `True`. Record the numbers and the decision.
- [ ] C++ parity for both; goldens regenerated; tests pass. Commit.

### Task 6: Detector decode with six classes first (C++)

**Files:** Create `host/include/fcw/yolo_decode.hpp`, `host/src/yolo_decode.cpp`, `host/tests/test_yolo_decode.cpp`; Modify `tools/export_golden.py` (exporter `yolo` from `deploy/vitis_ai/yolo_dpu.py` raw head outputs on 3 frames)

**Interfaces:** Produces `std::vector<Detection> decodeYolo(const std::vector<cv::Mat>& raw_heads, cv::Size input, const YoloDecodeConfig&)`; `YoloDecodeConfig { double conf = 0.5; double iou = 0.45; std::vector<int> classes = {2, 7, 5, 3, 1, 0}; }` (COCO ids car, truck, bus, motorcycle, bicycle, person).

- [ ] Test `YoloDecode.MatchesPythonDecode` (same boxes ±1e-3 px, same classes, same order after NMS); `YoloDecode.SkipsOtherClasses`. Implement: logit threshold `log(conf/(1−conf))` on the six class channels before the DFL softmax. PASS. Commit.

### Task 7: Processing size from the source aspect

**Files:** Modify `collision_avoidance/fcw/pipeline.py`, `collision_avoidance/fcw/__main__.py`, `eval/*.py`, `host/include/fcw/pipeline.hpp`, `host/src/runtime/app_main.cpp`; Test `tests/fcw/test_pipeline.py`

**Interfaces:** Produces `processing_size(width: int, height: int, target_width: int = 512) -> tuple[int, int]`: height `round(target_width·h/w/32)·32` clamped to [64, 512]; C++ `processingSize`.

- [ ] Tests: 3840×2160 → (512, 288); 1920×1200 → (512, 320); 640×480 → (512, 384); 1242×375 → (512, 160); 480×640 → (512, 512); 100×10 → (512, 64).
- [ ] Wire into the CLI, eval scripts and `fcw_app` (intrinsics scale with the chosen size). PASS. Commit.

### Task 8: LK flow source and the Vitis reference

**Files:** Create `collision_avoidance/fcw/lk_flow.py`, `host/include/fcw/lk_flow.hpp`, `host/src/lk_flow.cpp`, `hls/lk_ref/` (C-simulation of Vitis `densePyrOpticalFlow` on a frame pair: `Makefile`, `lk_ref.cpp`, `README.md` recording the flow output format); Modify `host/tools/cpu_baseline.cpp` (use the library LK), `eval/bench_gpu.py` (import from `lk_flow.py`)

**Interfaces:** Produces Python `LkFlow(levels=5, iterations=5, win=11, device="cuda")` callable `(prev_gray, curr_gray) -> (H, W, 2) float32`; C++ `FlowSource lkFlow(LkParams = {})`; pipeline flow choice `flow="dis"|"lk"` in the CLI and eval scripts.

- [ ] Tests: Python and C++ self-test (shift (1.6, −0.7) px → median error < 0.05 px); `LkFlow` vs C++ `lkFlow` on one real frame pair: median |Δ| < 0.05 px.
- [ ] Sparse-clone `Xilinx/Vitis_Libraries` (vision/L1/include only) into `third_party/` (git-ignored); C-simulate `densePyrOpticalFlow` on 2 frame pairs; report the median endpoint difference to `lkFlow` and the flow bit format. Commit.

### Task 9: Re-validate with LK

**Files:** Modify `docs/superpowers/reports/2026-10-p1-pc-evaluation.md`

- [ ] Run KITTI yaw and EvTTC with `flow="lk"`; both gates must hold. Record DIS and LK side by side. If a gate fails, stop and report. Commit.

### Task 10: Engine: inlier mask, levels, flow format (HLS + host)

**Files:** Modify `hls/ttc_engine/src/ttc_engine.hpp/.cpp`, `hls/ttc_engine/tb/tb_ttc_engine.cpp`, `tools/export_golden.py` (`hls` exporter: masked cases, level-1 case); Create `host/include/fcw/moment_engine.hpp`, `host/src/moment_engine.cpp`, `host/tests/test_moment_engine.cpp`

**Interfaces:** Kernel gains `const FitQ fits[kMaxBoxes]`, `ap_uint<kMaxBoxes> fit_valid`, `ap_uint<16> inlier_q`; masked moments use an incremental row predictor (adders only). The input packer converts the Vitis LK flow format recorded in Task 8 to int16 at 1/64 px (issue 5). Host: `struct EngineBox { Box box; int level; std::optional<FitQ> fit; }`; `class MomentEngine { virtual std::vector<BoxSums> run(const cv::Mat& prev, const cv::Mat& curr, const cv::Mat& flow_q, const std::vector<EngineBox>&) = 0; }`; `SoftwareMomentEngine` (one call per level in use; flow moments only on level 0).

- [ ] Goldens and testbench cases: masked moments, level-1 Horn; C-simulation bit-exact (`make csim` prints all cases OK). `MomentEngine.SoftwareMatchesGolden`. PASS. Commit.

### Task 11: Shared pyramid and the preprocess kernel (HLS)

**Files:** Create `hls/preprocess/` (`src/preprocess.cpp`: Vitis `resize`, `cvtColor`, INT8 quantiser with the model's fixed-point scale, `pyrDown` levels; `tb/`, `Makefile`, `hls_config.cfg`)

- [ ] C-simulation: gray and pyramid vs `cv::resize`/`cv::cvtColor`/`cv::pyrDown`; report max |Δ|. If `pyrDown` is not bit-exact, switch the software Horn and LK pyramid to a fabric-matching `pyrDownFabric` (Python + C++, goldens regenerated). Commit.

### Task 12: LK wrapper kernel (HLS)

**Files:** Create `hls/lk_wrapper/` (`src/lk_wrapper.cpp` looping levels × iterations around `densePyrOpticalFlow` with DDR ping-pong flow buffers; `tb/`, `Makefile`, `hls_config.cfg`)

- [ ] C-simulation on a frame pair equals the per-call reference from Task 8 bit for bit. Commit.

### Task 13: FrameRing

**Files:** Create `host/include/fcw/frame_ring.hpp`, `host/tests/test_frame_ring.cpp`

**Interfaces:** `template <class Slot> class FrameRing { FrameRing(int look_back, int in_flight); int slots() const /* look_back + in_flight + 1 */; Slot& acquire(int64_t frame_index); const Slot* at(int64_t frame_index) const; void release_before(int64_t oldest_needed); }`; `acquire` throws if the slot still holds a frame ≥ `oldest_needed`.

- [ ] Tests: `slots()` = 12 for (8, 3) and 5 for (1, 3); look-back by index with skipped indices (live drops); `acquire` refuses to overwrite a frame still needed. Implement; PASS. Commit.

### Task 14: Frame stage, object stage, worker pool

**Files:** Modify `host/include/fcw/pipeline.hpp`, `host/src/pipeline.cpp`; Create `host/tests/test_parallel_pipeline.cpp`

**Interfaces:** Produces `FrameState frameStage(const cv::Mat& bgr_prepared, const cv::Mat& gray, const cv::Mat& flow, std::vector<Detection>, double t)` (tracker ∥ ego-rotation, road constant; returns a snapshot of the selected tracks) and `FcwFrame objectStage(FrameState)` (engine call via `MomentEngine`, then per-object work on `FcwConfig::object_workers` threads, merged in selection order, then `warn`). `process` = `objectStage(frameStage(...))`; it accepts an already-prepared frame (no second resize).

- [ ] Tests: every pipeline golden scenario gives identical `FcwFrame`s with `object_workers` = 0, 1, 3 and with random 0–2 ms delays injected per object; 0 tracks and 20 tracks included. Implement (pre-create per-track state before the parallel section so workers never insert into shared maps). PASS. Commit.

### Task 15: Four-stage runtime

**Files:** Modify `host/include/fcw/runtime.hpp`, `host/src/runtime/runtime.cpp`, `host/src/runtime/app_main.cpp`, `host/tests/test_runtime.cpp`

**Interfaces:** Stages T1 capture → T2 (flow ∥ detection) → T3a `frameStage` → T3b `objectStage` → T4 sink; queues bounded (2, 2, 2, 4); drop only at T1 for live sources.

- [ ] Test `Runtime.MatchesSerialPipeline` (same levels and TTCs per frame as `FcwPipeline::process` on a recorded clip) and `Runtime.LiveDropsOnlyAtCapture`. Implement. `fcw_app` prints per-stage times and FPS. Run `fcw_app` on 300 frames; record. Commit.

### Task 16: Board backends (compile-checked)

**Files:** Create `host/backends/board/` (`gst_capture.cpp`: VCU / v4l2 pipelines into NV12; `drm_overlay.cpp`: video plane + ARGB graphics plane; `xrt_kernels.cpp`: preprocess, LK wrapper and engine runs with `xrt::bo::sync(FROM_DEVICE)` before every Arm read; `vart_yolo.cpp`: VART async run + `decodeYolo`); Modify `host/CMakeLists.txt` (`FCW_BOARD`), `docs/superpowers/plans/2026-10-07-fcw-host-app.md` Task 10 (board test steps for each backend)

- [ ] Compile-check what WSL can (GStreamer, libdrm); XRT/VART sources are built on the board. Commit.

### Task 17: Fabric fit

**Files:** Create `docs/superpowers/reports/2026-10-fabric-budget.md`; `hls_config.cfg` for each kernel

- [ ] Blocked until the Zynq UltraScale+ device data for Vitis 2024.2 is installed (user action). Then `vitis-run --mode hls --csynth` per kernel; table LUT/FF/BRAM/URAM/DSP vs the K26; apply fallbacks (8-box engine, 48-bit accumulators, B3136) until it fits. Commit.

### Final: README and report

- [ ] Update the README (architecture, flow, numbers) and the report; push to `origin` and `org`.
