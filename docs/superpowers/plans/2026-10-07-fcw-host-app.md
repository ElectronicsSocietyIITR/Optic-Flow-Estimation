# FCW C++ Host App and Board Bring-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the FCW algorithm core to C++17 with numerical parity against the Python reference, wrap it in the four-thread on-board app (spec section 3), and bring it up on the KV260 (phase P0, then P1 on board).

**Architecture:** `host/` is a CMake project. `libfcw` holds one translation unit per Python module (`collision_avoidance/fcw/*.py`), all depending only on `host/include/fcw/types.hpp` and OpenCV. Parity is enforced by golden vectors: `tools/export_golden.py` runs the Python reference on synthetic scenes and writes OpenCV `FileStorage` files; GoogleTest suites in `host/tests/` read them back and compare. The runtime (`host/src/runtime/`) runs T1 capture → T2 accelerators → T3 perception → T4 render on bounded queues; accelerator backends sit behind interfaces with PC implementations (video file, OpenCV DIS flow, recorded or Python-produced detections) and board implementations (VART DPU runner, XRT LK kernel).

**Tech Stack:** C++17, CMake ≥ 3.22, OpenCV 4 (core, imgproc, calib3d, video, videoio, highgui), GoogleTest, pthreads; on the board additionally VART 3.5 and XRT. Development builds run in WSL Ubuntu 24.04, the same OS family as the KV260's Kria Ubuntu image.

**Spec:** `docs/superpowers/specs/2026-10-06-kv260-collision-warning-design.md` (sections 3, 5, 6, 7, 8, 9)

**Depends on:** plan 1 (`docs/superpowers/plans/2026-10-06-fcw-algorithm-core.md`), merged at `e2b25b3`, plus fixes `4cd7d34`.

**This is plan 2 of 3.** Plan 3 replaces `horn_sums`/`flow_moments` with the HLS engine behind the same interface.

## How this plan is written

Each C++ module is a line-for-line port of a Python module that already passes its tests. The Python module **is** the implementation reference, and each task names it. The plan therefore gives, per task, the exact C++ interface, the golden cases the exporter writes, and the GoogleTest that compares them; it does not repeat the algorithm as C++ text. Where Python and C++ must differ (integer types, OpenCV API names), the task says so.

## Global Constraints

- C++17, no exceptions across module boundaries for expected conditions: "no estimate" is `std::optional`, mirroring Python's `None`.
- Only OpenCV, the C++ standard library and GoogleTest on the PC; VART/XRT only inside `host/src/backends/board/`, compiled when `FCW_BOARD=ON`.
- Conventions identical to plan 1: camera x right, y down, z forward; R maps prev → curr; flow prev → curr in px/frame; boxes `(x1, y1, x2, y2)` in 512×384 pixels; η in s⁻¹; all thresholds read from the same defaults as the Python config dataclasses.
- Horn sums are `int64_t` and must match Python **exactly**. Every floating-point result matches Python within 1e-6 relative (or the tolerance stated in the task when OpenCV's RANSAC is involved).
- Do not modify `collision_avoidance/` except `tools/` additions; the Python package stays the reference.
- Commit messages carry no `Co-Authored-By` trailer and no Claude attribution (mandatory user instruction). Pushing uses the user's Shyam-24o7 account.

## Review Focus

1. **int64 overflow or truncation in Horn sums** (largest box, brightest texture): C++ must match Python's int64 values exactly. Golden case `horn_full_frame_saturated`.
2. **OpenCV RANSAC nondeterminism** between Python and C++ builds: parity for ego-rotation compares the recovered rotation within 0.05°, not bit-wise. Golden cases include a pure-rotation and a forward-motion frame.
3. **Queue back-pressure** when T3 runs slower than capture: frames must be dropped at T1 (oldest first), never queued without bound, and timestamps must stay monotonic. Test `runtime_drops_frames_under_load`.
4. **Repeated and out-of-order timestamps** reach the C++ pipeline from real cameras: same fallbacks as Python (`dt_s` reported). Golden case `pipeline_out_of_order`.
5. **More than 16 detections:** C++ selection order must equal Python's, or the HLS engine and the CPU twin would compute different boxes. Golden case `pipeline_twenty_tracks`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tools/export_golden.py` | runs the Python reference, writes `host/tests/golden/*.yml.gz` | 1 (grows per task) |
| `host/CMakeLists.txt`, `host/cmake/` | build: `fcw` library, `fcw_app`, `fcw_tests` | 1 |
| `host/include/fcw/types.hpp` | Box, Measurement, Detection, configs | 1 |
| `host/tests/golden.hpp` | FileStorage golden-vector reader | 1 |
| `host/src/synth.cpp` | not ported: tests read Python-rendered frames from the golden files | — |
| `host/src/tracker.cpp` + `include/fcw/tracker.hpp` | port of `tracker.py` | 2 |
| `host/src/ego_rotation.cpp` + header | port of `ego_rotation.py` | 3 |
| `host/src/scale_search.cpp` + header | port of `scale_search.py` | 4 |
| `host/src/horn.cpp` + header | port of `horn.py` (int64 golden model for HLS) | 5 |
| `host/src/divergence.cpp` + header | port of `divergence.py` | 6 |
| `host/src/fusion.cpp`, `collision.cpp` + headers | ports of `fusion.py`, `collision.py`, `looming.py` | 7 |
| `host/src/pipeline.cpp` + header | port of `pipeline.py` | 8 |
| `host/src/runtime/*.cpp` | T1–T4 threads, bounded queues, PC backends, `fcw_app` CLI | 9 |
| `host/src/backends/board/*.cpp` | VART YOLO runner, XRT LK runner | 10 (board) |
| `platform/` | overlay build scripts and README | 10 (board) |

---

### Task 1: Build skeleton, types and golden-vector harness

**Files:** create `host/CMakeLists.txt`, `host/include/fcw/types.hpp`, `host/tests/golden.hpp`, `host/tests/test_harness.cpp`, `tools/export_golden.py`.

**Interfaces produced:**
- `struct Box { double x1, y1, x2, y2; }`; `struct Detection { Box box; std::string cls; double conf; }`; `struct Measurement { std::string method; double eta; double var; }`.
- `golden::File(path)` with `mat(key)`, `num(key)`, `i64(key)` (int64 stored as decimal string), `str(key)`, `size(prefix)`.
- `tools/export_golden.py --out host/tests/golden` writes one `<module>.yml.gz` per module; Task 1 writes `harness.yml.gz` with one Mat, one double and one int64 above 2³².

- [ ] **Step 1:** Write `host/tests/test_harness.cpp`: reads `harness.yml.gz`, asserts the Mat shape and values, the double, and that `i64("big") == 6000000000`.
- [ ] **Step 2:** Configure and build in WSL: `cmake -S host -B build -G Ninja && cmake --build build`. Expected: test fails because `harness.yml.gz` does not exist.
- [ ] **Step 3:** Write `tools/export_golden.py` (harness section) and run it with the plan-1 venv: `.venv/Scripts/python tools/export_golden.py --out host/tests/golden`.
- [ ] **Step 4:** `ctest --test-dir build --output-on-failure`. Expected: `100% tests passed`.
- [ ] **Step 5:** Commit `host/ tools/export_golden.py` ("Add C++ host skeleton and golden-vector harness").

### Task 2: Tracker port (`tracker.py`)

**Interfaces:** `TrackerConfig` (same fields and defaults), `ScaleTrack { int id; cv::Vec<double,7> x; cv::Matx<double,7,7> P; std::string cls; int lost, age; std::deque<std::pair<int, Box>> history; Box bbox() const; std::optional<Box> boxAt(int) const; }`, `ScaleTracker::update(const std::vector<Detection>&, double dt, int frame) -> const std::map<int, ScaleTrack>&`, `std::optional<Measurement> looming(const ScaleTrack&)`.
**Golden cases:** the approaching scene of `test_scale_rate_converges_to_inverse_ttc` (state after every frame), the two-object case, the lost-frame history case. **Tolerance:** state 1e-9 absolute; history identical.
**Note:** Hungarian assignment via a small O(n³) implementation (scipy's `linear_sum_assignment` semantics: minimise −IoU); ties resolved like scipy (row order).

### Task 3: Ego-rotation port (`ego_rotation.py`)

**Interfaces:** `EgoRotationConfig`, `EgoMotion { cv::Matx33d R; cv::Vec3d rvec; std::optional<cv::Point2d> foe; double inlier_ratio; bool stationary, valid; }`, `EgoRotationEstimator(K, cfg).update(const cv::Mat& flow, const std::vector<Box>& exclude)`, `double rotationDivergence(R, K, cv::Point2d)`.
**Golden cases:** forward motion with yaw 0 / 0.4 / −0.8°, pure rotation, noise hold, excluded truck. Flow fields are exported as `CV_32FC2` Mats. **Tolerance:** angle(R_cpp · R_pyᵀ) < 0.05°, FOE within 0.5 px, flags identical; `rotationDivergence` 1e-9.

### Task 4: Scale-search port (`scale_search.py`)

**Interfaces:** `ScaleSearchConfig`, `std::optional<std::pair<double,double>> searchScale(gray_t, gray_tk, box_t, box_tk, cfg)`, `std::optional<Measurement> scaleTtc(..., int k, double dt, cfg)`, `int chooseGap(std::optional<double> eta, double dt, int available, double min_expansion = 0.02)`.
**Golden cases:** the k = 2/4/8 approach, static object, sideways object, tiny box, the gap table. **Tolerance:** s* 1e-6 (both sides use `cv::warpAffine` with identical flags).

### Task 5: Horn port (`horn.py`) — the HLS golden model

**Interfaces:** `HornConfig`, `struct HornSums { std::array<int64_t,10> s; int64_t n; }`, `gradients(prev, curr)` (CV_64S not available: use `std::vector<int32_t>` planes), `std::optional<HornSums> hornSums(prev, curr, box, pp, cfg)`, `std::optional<Measurement> hornSolve(const HornSums&, double dt, cfg)`, `int chooseLevel(...)`, `std::optional<Measurement> hornAtLevel(...)` (pyramid via `cv::pyrDown`).
**Golden cases:** the integer pixel-loop case (random 40×50 frames), the approach scene, the textureless box, and `horn_full_frame_saturated` (alternating 0/255 stripes over a 510×382 box, largest sums). **Tolerance:** sums and n **exact**; η 1e-9 relative.

### Task 6: Divergence port (`divergence.py`)

**Interfaces:** `DivergenceConfig`, `std::optional<std::array<double,12>> flowMoments(flow, box, pp, mask, shrink)`, `std::optional<Measurement> divergenceTtc(...)`, `cv::Mat robustMask(flow, box, pp, cfg)`.
**Note:** RANSAC hypotheses must use the same sample indices as Python for parity: the exporter writes the 64 index triples it drew (`numpy.random.default_rng(0)`), and the C++ test injects them through a test-only overload `robustMask(..., const std::vector<std::array<int,3>>& samples)`; production code draws its own with `std::mt19937(0)`.
**Golden cases:** expanding flow, rotation removal, background leak, empty box. **Tolerance:** moments 1e-9 relative, mask identical.

### Task 7: Fusion, looming, course check and warning FSM (`fusion.py`, `collision.py`)

**Interfaces:** `FusionConfig`, `TtcEstimate`, `TtcFusion::update(int id, double t, double dt, const std::vector<Measurement>&) -> std::optional<TtcEstimate>`, `retain`; `CourseConfig`, `CourseResult`, `CourseChecker::update(...)`; `enum class Level { None, Warning, Critical }`, `WarningConfig`, `WarningFsm::step(bool, std::optional<double>, std::optional<double>, int) -> Level`.
**Golden cases:** every input sequence of `test_fusion.py` and `test_collision.py`, replayed step by step. **Tolerance:** η 1e-12, levels identical.

### Task 8: Pipeline port (`pipeline.py`)

**Interfaces:** `FcwConfig` (aggregate of all configs, `K()`), `ObjectResult`, `FcwFrame { int index; double time_s, dt_s; EgoMotion ego; cv::Point2d heading; std::vector<ObjectResult> objects; Level level() const; }`, `using FlowSource = std::function<cv::Mat(const cv::Mat&, const cv::Mat&)>`, `using DetectorFn = std::function<std::vector<Detection>(const cv::Mat&)>`, `FcwPipeline(cfg, detector, flow).process(const cv::Mat& bgr, double t)`.
**Golden cases:** head-on approach, next-lane car, static object, dropped frames, irregular frames, out-of-order timestamps, twenty tracks. To isolate the port from optical-flow differences, the exporter stores the DIS flow Python computed for every frame and the C++ test injects it as the `FlowSource`. **Tolerance:** per frame identical object ids, selection order and warning levels; τ within 1e-6 relative.

### Task 9: Threaded runtime and PC app

**Interfaces:** `BoundedQueue<T>(capacity)` with `push_drop_oldest`, `pop(timeout)`; `Runtime(cfg, CaptureSource, DetectorFn, FlowSource, Sink)` running T1–T4; `fcw_app --source <video> --fx <px> [--detections <yml.gz>] [--output out.mp4] [--max-frames N]` printing the same summary line as the Python CLI.
**Tests:** `runtime_processes_every_frame_when_fast` (synthetic source, sink sees N results in order); `runtime_drops_frames_under_load` (artificially slow T3: no unbounded growth, monotonic timestamps, dropped count reported); `runtime_summary_line_matches_cli_format`.
**PC detector:** detections recorded by Python (`tools/export_golden.py --record-detections <video>` runs YOLOv9t through the existing `Detector` and stores per-frame boxes), so the PC app needs no DNN runtime.

### Task 10 (board): P0 bring-up on the KV260

Runs only with the board. Each step lists its command and its pass condition; results go into `docs/superpowers/reports/<date>-p0-board.md`.

1. **Image:** flash Kria Ubuntu 22.04 for KV260; `sudo snap install xlnx-config --classic`; confirm `xmutil platformstats` prints power rails.
2. **Overlay:** in `platform/`, build a Vitis 2024.2 platform with DPUCZDX8G B4096 (fallback B3136) and the Vitis Vision `densePyrOpticalFlow` kernel at 512×384; package as an accel app (`xmutil loadapp fcw`). Pass: `xbutil examine` lists both CUs; utilisation report saved.
3. **DPU detector:** compile upstream's `yolov9t_kv260.xmodel` (deploy/vitis_ai) if not already; implement `backends/board/vart_yolo.cpp` (VART runner, Detect-head decode + NMS on ARM, same class filter as Python). Pass: on the demo clip, detections match the PC YOLO within IoU ≥ 0.5 for ≥ 90% of boxes; latency logged.
4. **LK flow:** implement `backends/board/xrt_lk.cpp` (XRT buffers, zero-copy). Pass: flow on a synthetic shift pair within 0.3 px of truth; latency logged.
5. **Cross-build** `fcw_app` on the board (`cmake -DFCW_BOARD=ON`), run on the demo clip and a live USB camera. Pass: per-stage latency histograms; 30 FPS sustained or the bottleneck named; capture-to-warning latency < 100 ms.
6. **Power:** idle vs running from `xmutil platformstats`; recorded in the report.

### Board backends (added 2026-10-24, pipelined design)

The backends in `host/backends/board` (`fcw/board.hpp`) are written and compile-checked but have never run on the board. Test each before the full run:

7. **Capture** (`gstCapture`): file through `decodebin` (check that it picks the VCU decoder, `GST_DEBUG=decodebin:4`) and a USB3 camera; BGRx frames, PTS timestamps 1/30 s apart. (Already verified on the PC with software decoders: `fcw_board_io_tests`.)
8. **Display** (`DrmDisplay`): primary + ARGB overlay planes found on the DP CRTC; boxes and banner blend over the video without tearing.
9. **Preprocess kernel**: output pyramid slot and INT8 tensor equal the C-simulation (`hls/preprocess`) on a recorded frame.
10. **LK wrapper**: one start per frame pair; flow equals `hls/lk_ref` on the same pyramid (bit for bit); time per frame.
11. **Horn engine** (`XrtMomentEngine`): per-box sums equal `SoftwareMomentEngine` on recorded frames; check the level sub-buffer offsets are accepted (64-byte aligned slots).
12. **DPU** (`VartYolo`): `inputShift()` read from the xmodel; detections equal `decodeYolo` on the same raw outputs from the PC.
13. **Ring sizing**: no `FrameRing` refusals over a 10-minute live run (8 slots for the default queues).
14. **Full run**: `fcw_app --board --xclbin fcw.xclbin --xmodel yolov9t.xmodel --source <clip>`; per-stage times, throughput, power (`xmutil platformstats`).

### Exit (spec P1, board half)

30 FPS on dash-cam clips on the KV260 with every stage measured; warnings identical to the PC C++ app on the same inputs (parity carried from Task 8).
