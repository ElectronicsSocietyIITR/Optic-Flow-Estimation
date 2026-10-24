# Pipelined, parallel FCW on the KV260: design

- **Date:** 2026-10-23
- **Status:** approved in conversation on 2026-10-23 ("do all this and fix all the mentioned issues"). Items marked *(my call)* were decided by the builder, not the user.
- **Builds on:** `2026-10-06-kv260-collision-warning-design.md` (system design), plan 2 (C++ host), plan 3 (Horn engine).

## 1. Intent

The PC prototype is correct but laid out serially. Measured on a laptop core and scaled to the KV260's Cortex-A53 (PassMark ×15.9), the per-object stage (T3) takes 120–180 ms per frame, so the board would run at 6–8 FPS. Two components dominate: ego-rotation (5-point essential matrix + homography RANSAC, 78–128 ms) and the per-object divergence RANSAC (20–28 ms). A low-level review also found 16 issues, from an engine that computes different divergence sums than the software, to a gray-frame ring that is too small once frames are pipelined.

This design turns the system into a pipelined, parallel process that can reach 30 FPS on the board. It decides how each part is implemented in hardware, and fixes every issue found.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Every frame-level stage starts when its inputs are ready (dependency-driven), and the per-object work runs in parallel on the Arm cores.
2. Replace the two expensive algorithms with formulations that are about 10–100× cheaper, without losing accuracy (gates in section 6).
3. Fix all 16 issues of the low-level review (section 4 maps each to a change).
4. A decided, written implementation technology for every hardware block.
5. Every change verified against the existing golden-vector chain (Python → C++ → HLS) and the real-data evaluations.

**Non-goals**
- Moving branchy per-object logic (tracker, EKF, course check, lead selection, warning FSM) into the fabric.
- New sensors, braking control, night tuning.

**Anti-goals (a failure even if it runs)**
- A parallel build whose output differs from the serial reference on the golden frames.
- Faster algorithms that lose accuracy: KITTI yaw RMS ≥ 1 °/s, EvTTC fused error > 20 %, or fewer than 5/5 approaches warned by true TTC 2.0 s.
- Claiming board numbers before the board run.

## 3. Implementation technology *(my call)*

| Block | Technology | Why |
|---|---|---|
| YOLOv9t-Hardswish | AMD **DPUCZDX8G** IP (prebuilt RTL), model compiled with Vitis AI to an `.xmodel`, run with **VART** | the DPU is AMD's CNN engine; nothing custom to write |
| Preprocess (resize, gray, RGB INT8 tensor, Gaussian pyramid) | **Vitis HLS** kernel built from **Vitis Vision** primitives (`resize`, `cvtColor`, `pyrDown`) plus our INT8 quantiser | library primitives are already II = 1; HLS lets the C-simulation run against our golden vectors |
| Dense pyramidal LK | **Vitis Vision `densePyrOpticalFlow`** inside a thin **HLS** wrapper that loops levels and iterations on-chip (one kernel start per frame instead of 25) | proven kernel (12 FPS at 1080p in AMD's benchmark) |
| Horn / moment engine (`hls/ttc_engine`) | **Vitis HLS** (existing), extended | streaming multiply-accumulate is HLS's strongest case; it is already bit-exact in C-simulation |
| Flow-format adapter | folded into the engine's input packer (HLS) | a shift and a sign extension; not worth a separate kernel |
| Integration | **Vitis `v++` link** onto a Vivado platform (Zynq PS, clocks, SmartConnect, DPU), loaded as a device-tree overlay; controlled from Linux with **XRT** | standard KV260 accelerated-application flow |
| Hand-written RTL (SystemVerilog) | **only as a fallback**, for the engine's accumulator bank if HLS cannot meet 300 MHz or the LUT budget (section 9) | HLS output is verified against the same golden data; RTL would add verification cost for little gain |

HLS here means synthesisable C++ with `#pragma HLS` directives: `PIPELINE II=1` in pixel loops, `DATAFLOW` between stages connected by `hls::stream` FIFOs, AXI4-Stream between kernels, AXI4 master for DDR bursts, AXI4-Lite for control registers. Each kernel goes through C-simulation, C/RTL co-simulation, then synthesis and implementation.

## 4. The 16 issues and their fixes

| # | Issue | Fix | Section |
|---|---|---|---|
| 1 | engine's flow moments are summed over the whole box; the software sums only RANSAC inliers | engine masks each pixel with the track's predicted affine fit; software uses the same mask | 6.2 |
| 2 | engine has no per-box pyramid level | host runs the engine once per level in use; level 0 also carries the flow moments | 7.2 |
| 3 | Horn's pyramid is `cv::pyrDown`; the fabric's must match | one shared Gaussian pyramid feeds LK and Horn; bit-exactness of Vitis `pyrDown` vs `cv::pyrDown` checked in C-simulation, else the software reference follows the fabric | 7.3 |
| 4 | accuracy was measured with DIS flow; the board runs LK | an LK flow source with the Vitis parameters in the software path, checked against Vitis C-simulation; all evaluations re-run with it | 8 |
| 5 | Vitis LK flow format vs the engine's int16 at 1/64 px | read the format from the library source; the engine's input packer converts | 7.4 |
| 6 | LK is started 25 times per frame by the host | HLS wrapper loops levels × iterations on-chip | 3 |
| 7 | 9-slot gray ring too small once frames are pipelined | slots ≥ max look-back + frames in flight + 1; `FrameRing` with explicit slot ownership | 5.3 |
| 8 | 16:9 frames squeezed into 512×384 | processing size derived from the source aspect: width 512, height a multiple of 32 (16:9 → 512×288) | 9 |
| 9 | T2 runs flow, then detection | flow and detection run concurrently | 5.1 |
| 10 | T3 re-resizes the frame T2 prepared | removed: the pipeline accepts an already-prepared frame | 5.1 |
| 11 | ego-rotation waits for detection | ego-rotation starts as soon as flow is ready (it needs only flow N and the boxes of N−1) | 5.1 |
| 12 | per-object loop is serial | worker pool; results merged in track-ID order | 5.2 |
| 13 | caches vs fabric-written buffers | every PL-written buffer is `xrt::bo::sync(FROM_DEVICE)` before the Arm reads it; small ones are allocated non-cacheable | 5.3 |
| 14 | MJPEG decoding on the A53 | file playback through the K26's VCU (hardware H.264/H.265 decode, GStreamer); live input from MIPI or USB3 uncompressed | 10 |
| 15 | full-frame overlay drawing on the A53 | draw boxes and banner on the DisplayPort graphics layer (hardware alpha blend) | 10 |
| 16 | fabric fit: DPU B4096 + LK + engine may exceed the K26's LUTs | synthesis numbers decide; fallbacks in order: 8-box engine, 48-bit accumulators, B3136 DPU | 11 |

## 5. Execution model

### 5.1 Per-frame dependency graph

```
preprocess N ─┬─► LK N ──► ego-rotation N ──► road constant N ─────────────┐
              └─► DPU N ──► decode + NMS N ──► tracker N ─┐                │
                                                           ├─► engine N (Option A: box table + predicted fits;
                                ego-rotation N ───────────┘    one pass per pyramid level in use)
tracker N ──► looming (+ scale) per object  ─┐                                │
engine N  ──► Horn solve, divergence solve ──┴─► EKF, course per object ──► warn N ──► render N
```

- Ego-rotation of frame N uses flow N and the tracked boxes of frame N−1 (already known), so it runs while the DPU detects.
- The engine is started after the tracker (Option A): T3 writes the box table and each box's predicted affine fit; the engine reads gray N−1, gray N and flow N from DDR.
- While the engine runs, the workers do looming and scale search.
- `warn` merges results in track-ID order, so the output never depends on thread timing.
- Frame-level pipelining stays: three frames in flight (capture N+2, accelerators N+1, perception N).

### 5.2 Threads and cores

| Core | Work |
|---|---|
| 0 | T1 capture, T2 orchestration (blocking waits on the PL and DPU), T4 render |
| 1 | ego-rotation, road constant, decode + tracker |
| 2–3 | per-object worker pool (2–3 workers) |

On the PC, `std::thread`s stand in for cores; the same scheduler code runs on the board.

### 5.3 Buffers

All frame-sized buffers live in DDR; on-chip RAM holds only line buffers and FIFOs.

| Buffer | Slots × size | Producer | Consumers | Release |
|---|---|---|---|---|
| capture ring | 3 × input frame | T1 / VCU | preprocess | after preprocess |
| gray + pyramid ring | max look-back + 3 (12 with scale search, 4 without) × ≈ 256 KB | preprocess | LK (N−1, N), engine (N−1, N), scale search (N−k) | slot older than (oldest frame in perception) − look-back |
| DPU tensors | 2 in + 2 out | preprocess / DPU | DPU / decode | ping-pong |
| flow ring | 3 × 768 KB | LK | ego-rotation, road constant, engine, divergence fallback | perception done with that frame |
| box table + predicted fits | 1 per frame, < 1 KB | T3 | engine | engine done |
| sums | levels × 16 × 23 × 8 B | engine | workers | perception done |
| display buffers | 2–3 | T4 | DisplayPort | page flip |

`FrameRing` (C++) owns slot indices and enforces the release rule; on the PC it wraps `cv::Mat`, on the board `xrt::bo`.

### 5.4 State machines

- **Frame lifecycle:** CAPTURED → PREPROCESSED → {LK_BUSY ∥ DPU_BUSY} → FLOW_READY / DETS_READY → EGO_DONE ∧ TRACKED → ENGINE (per level: START → BUSY → DONE) → OBJECTS_DONE → WARNED → RENDERED.
- **Kernel handshake (every PL kernel and the DPU):** IDLE → buffers bound, `ap_start` → BUSY → `ap_done` (interrupt) → IDLE; host code waits with `xrt::run::wait` / VART `wait`.
- **Warning FSM:** unchanged (NONE / WARNING / CRITICAL).

## 6. Algorithm changes and their gates

### 6.1 Ego-rotation: linear rotation + FOE least squares
After rotation is removed, a static point's flow points away from the FOE: `(u − u_rot(ω))·(y − y_F) − (v − v_rot(ω))·(x − x_F) = 0`, with the small-rotation flow field `u_rot, v_rot` linear in ω (3 unknowns). For a fixed FOE this is linear in ω (3×3 normal equations); for a fixed ω it is linear in the FOE (2×2). Alternate the two with robust (Huber/IRLS) weights, starting from the previous FOE, about 5 rounds over the 768 grid points. Stationary: the residual flow after rotation is below a threshold for most points (then no FOE). Keeps the existing outputs (`R`, `rvec`, FOE, stationary, valid, inlier ratio) and the FOE smoothing.

**Gate:** KITTI yaw RMS < 1.0 °/s on the three drives (now 0.681); the synthetic yaw and stationary tests pass; at least 10× faster than today.

### 6.2 Divergence: predicted-fit inlier mask
Each track keeps last frame's affine flow fit. This frame, a pixel inside the box is an inlier when its flow is within the threshold of the fit's prediction (propagated by the track's expansion). The 12 flow moments are summed over inliers only; the fit is solved from them. A new track, or a fit with too few inliers, falls back to the existing RANSAC on the Arm. The same mask runs in Python, C++ and the HLS engine (where the prediction is evaluated incrementally along each row: adders only).

**Gate:** EvTTC fused error ≤ 20 % (now 11.5 %) and 5/5 warned; divergence's own median error within 2 points of today's 9.5 %.

### 6.3 Scale search: ablation
Run EvTTC with scale search disabled. If fused error stays ≤ 13 % and 5/5 warned, disable it by default (the gray ring drops to 4 slots); otherwise keep it.

### 6.4 Road constant: coarser grid
Sample the road band every 8 × 4 px instead of 4 × 2. **Gate:** the κ tests pass and the synthetic κ readings stay within 5 % of today's.

### 6.5 Detector decode: six classes first
Only the logits of the six collision classes are compared against the confidence threshold (in logit space, no sigmoid); box decoding (DFL softmax) runs only for cells that pass. Implemented in C++ for the board runner, checked against the Python decode in `deploy/vitis_ai/yolo_dpu.py`.

## 7. Engine (`hls/ttc_engine`) changes

1. **Inlier mask (issue 1):** per flow box, 6 affine coefficients (fixed point) and a threshold; incremental row evaluation; moments accumulate only inliers.
2. **Pyramid levels (issue 2):** the host calls the engine once per level in use; Horn boxes are grouped by level; flow boxes only in the level-0 pass.
3. **Pyramid (issue 3):** levels come from the shared Gaussian pyramid; bit-exactness of Vitis `pyrDown` vs `cv::pyrDown` is tested in C-simulation.
4. **Flow format (issue 5):** the input packer converts Vitis LK's flow format to int16 at 1/64 px.

Golden vectors gain masked-moment cases; C-simulation and C/RTL co-simulation must be bit-exact against Python and C++.

## 8. LK flow reference (issue 4)

An LK flow source with the Vitis kernel's parameters (5 levels, 5 iterations, 11 × 11 window) becomes selectable in the Python and C++ pipelines (`flow="lk"`): C++ from `host/tools/cpu_baseline.cpp`, Python on the GPU from `eval/bench_gpu.py`. It is checked against a Vitis `densePyrOpticalFlow` C-simulation on a few frame pairs (median endpoint difference reported). EvTTC and KITTI are re-run with it, and the report records both DIS and LK results.

## 9. Processing size (issue 8)

`FcwConfig` takes the processing size from the source: width 512, height `round(512 · h / w / 32) · 32` (16:9 → 288, 16:10 → 320, 4:3 → 384). Intrinsics scale accordingly (non-square pixels remain supported). Synthetic tests keep 512 × 384 (their native size). The DPU model is compiled for the board camera's size (Vitis AI, `deploy/vitis_ai`).

## 10. Board I/O (issues 14, 15)

- **Capture:** GStreamer pipelines: `filesrc ! qtdemux ! h264parse ! omxh264dec` (VCU) for files; `v4l2src` with an uncompressed format for USB3; the MIPI path through the KV260 ISP when available. Output NV12 into the capture ring.
- **Render:** DRM/KMS; video on the primary plane, overlay (boxes, TTC, banner) drawn on a small ARGB graphics plane composited by the DisplayPort hardware.
- These are written as board backends behind `FCW_BOARD` and tested on the board (plan 2, Task 10).

## 11. Fabric fit (issue 16)

PG338 lists B4096 at about 53k LUTs and 562 DSPs, with 257 BRAM or 68 URAM. The K26 has about 117k LUTs, 144 BRAM, 64 URAM, 1,248 DSPs (to confirm from the datasheet), so the DPU needs a mixed BRAM/URAM configuration. LK (about 31k LUTs in its 1080p configuration), the engine (about 21k), preprocessing and interconnect come on top. Synthesis decides; fallbacks in order: 8-box engine, 48-bit accumulators, B3136 DPU. Synthesis needs the Zynq UltraScale+ device data for Vitis 2024.2 (a user install). Revised 2026-10-24 (`docs/superpowers/reports/2026-10-fabric-budget.md`): the bottom-up count puts this design at about 140k LUTs, so the fallbacks become per-term accumulator widths (bit-exact; flat 48 bits would overflow), fewer engine lanes with more passes (keeps all 16 objects), B3136, and finally the Horn engine in software on the A53.

## 12. Verification

| Path | Check |
|---|---|
| algorithm changes | Python unit tests on synthetic scenes; gates in section 6 |
| Python ↔ C++ | golden-vector parity per module and for the whole pipeline (existing chain, regenerated) |
| parallel runtime | same results as the serial pipeline on every golden frame, also with random worker delays |
| engine | C-simulation and C/RTL co-simulation bit-exact against the golden vectors; AXI-Stream stall injection |
| LK | Vitis C-simulation vs our LK on frame pairs |
| board | plan 2 Task 10: measured FPS, latency, power; warnings identical to the PC C++ app on the same inputs |

## 13. Left to the builder

Thread pool details, queue depths beyond the minimums above, fixed-point widths of the affine coefficients (chosen so the mask decision is exact against the software), and the order of tasks within the plan.
