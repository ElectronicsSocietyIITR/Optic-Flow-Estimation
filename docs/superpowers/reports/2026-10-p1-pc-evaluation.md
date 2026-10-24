# P1 PC evaluation: FCW algorithm core

- **Plan:** `docs/superpowers/plans/2026-10-06-fcw-algorithm-core.md`, Task 13
- **Status:** all four PC criteria **pass** (as of commit 57442e0). Open gap: objects crossing into our path warn late (see the last section).
- **Environment:** Windows 11, Python 3.11 venv, torch 2.6.0+cu124 (RTX 4060 Laptop GPU), OpenCV 4.10 DIS flow as the PC stand-in for the board's Vitis LK.

## Exit criteria (PC half of P1)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| Yaw-rate RMS vs KITTI OXTS | < 1.0 °/s | **0.681 °/s** (805 frame pairs, 3 drives) | **pass** |
| Fused median relative TTC error (EvTTC) | ≤ 20% | **11.5%** with the stale-estimate fix (12.7% after calibration, 16.2% before) | **pass** |
| Approach events warned before true TTC 2.0 s (EvTTC) | ≥ 90% | **100%** (5 of 5) | **pass** |
| False alarms on normal driving | < 1 per 10 min | **0.67 per 10 min** (2 raises in 30.0 min); 17.0 with the lead-vehicle rule only, 117.2 before it | **pass** |
| Fusion `var_scale` calibration | from EvTTC | looming 0.19, scale 6.61, horn 68.13, divergence 1012.81 | applied |

## Ego-rotation detail

`python eval/kitti_yaw.py data/kitti/2011_09_26/2011_09_26_drive_{0005,0014,0091}_sync`

| Drive | Frame pairs | RMS (°/s) |
|---|---|---|
| 2011_09_26_drive_0005_sync | 153 | 0.589 |
| 2011_09_26_drive_0014_sync | 313 | 0.817 |
| 2011_09_26_drive_0091_sync | 339 | 0.572 |
| **overall** | **805** | **0.681** |

Every frame pair produced a valid estimate. The low error also confirms the rotation sign convention (R maps camera coordinates prev → curr; OXTS `wu` compares directly with `rvec[1] / dt`): a flipped sign would roughly double the error on turns.

## TTC detail (EvTTC, 5 car-rear approach sequences)

`python eval/evttc_fcw.py --data data/evttc`, YOLOv9t detector, DIS flow, 512×384 with EvTTC's intrinsics scaled (non-square pixels).

| Sequence | Fused error before | Fused error after calibration | Coverage |
|---|---|---|---|
| CCRs-1-high-100 | 0.126 | 0.089 | 0.37 |
| CCRs-1-low-100 | 0.188 | 0.154 | 0.92 |
| CCRs-1-medium-100 | 0.171 | 0.137 | 0.95 |
| CCRs-2-high-100 | 0.138 | 0.114 | 0.52 |
| CCRs-2-low-100 | 0.171 | 0.148 | 0.98 |
| **all** | **0.162** | **0.127** | 0.71 |

Per method, before calibration (median relative error, coverage): looming 0.170 / 0.53, scale search 0.268 / 0.47, Horn 0.256 / 0.45, LK divergence **0.095** / 0.56.

Findings:
- On real footage divergence is the most accurate method by median, the opposite of the synthetic scenes, but its stated variance is far too small on its bad frames (calibration factor about 1000). Its typical accuracy is good, its outliers are large.
- The calibrated factors replace the defaults of 1.0 in `FusionConfig.var_scale`; all 66 FCW tests still pass with them.
- Coverage is lowest on the "high" (fast-approach) sequences: the target is often too small or too close for a confident estimate at the start and end.

## False alarms (30 min of UK residential driving)

`python eval/false_alarms.py --videos data/normal_driving --fx 200`: 10 front-camera clips from the MIT-licensed `aap9002/UK-Road-DashCam` dataset (4K, 30 FPS, no calibration published; fx = 200 px at 512×384 assumed for a wide dash cam).

| Clip | Raises |
|---|---|
| 241220_125301_002 | 35 |
| 241220_125601_003 | 32 |
| 241220_125901_004 | 27 |
| 241220_130201_005 | 36 |
| 241220_130501_006 | 52 |
| 241220_130801_007 | 31 |
| 241220_131102_008 | 38 |
| 241220_131402_009 | 35 |
| 241220_131702_010 | 12 |
| 241220_132302_012 | 54 |
| **total** | **352 in 30.0 min = 117.2 per 10 min** |

Diagnosis on clip 003 (first 1,500 frames, 19 raises):
- Every raise is a car (18) or truck (1) and goes straight to CRITICAL. The TTC itself is plausible: these are oncoming cars and parked cars on narrow residential streets, closing fast in the image, and all methods agree.
- The **course check** is what fails. The heading (FOE x) jumps between 210 and 416 px on a 512 px frame, so the lateral-offset rate ṙ is noise and `r + ṙ·TTC` lands near 0 for cars that are currently 1.5–10 widths to the side (e.g. r = 4.30, r_contact = 0.52).
- Two real-footage bugs found on the way were fixed in Python and C++ (lost tracks holding warnings; stale predictions driving TTC to 0). They cut critical frames on the first clip from 88 to 67 but did not touch the course-check problem.

Course-check variants on the same 1,500 frames (raises / critical frames): FOE heading 10 / 347; FOE smoothed (EMA 0.05) 11 / 205; principal-point heading 14 / 224; on course now **and** at contact 7 / 151; no ṙ extrapolation 12 / 255. None comes near the target, so the heading jitter is not the main cause.

Calibrated control: the same pipeline on the three KITTI raw drives (calibrated camera, 1.39 min of normal driving) raises **79 warnings per 10 min**. The false alarms are algorithmic, not a calibration artefact.

What the remaining warnings are: parked cars along residential streets and oncoming cars, passed with roughly 0.5–1 m clearance. Their TTC to the object's plane really is about 1 s; the system cannot place them laterally precisely enough to tell "passing close" from "on our path", because the corridor test has only a 0.3–0.8 m margin and box/heading noise exceeds it. This needs a design change in how the path is defined (see the recommendations in the conversation of 2026-10-08), not a threshold tweak.

## Lead-vehicle change (2026-10-09)

Only the nearest object that overlaps at least half our width (the lead vehicle) may warn, after 3 consecutive in-path frames, with a heading smoothed over ~0.7 s.

| Data | Before | After |
|---|---|---|
| Dash-cam clips, 30 min | 352 raises, 117.2 per 10 min | **51 raises, 17.0 per 10 min** |
| — per clip | 35 / 32 / 27 / 36 / 52 / 31 / 38 / 35 / 12 / 54 | 5 / 8 / 3 / 6 / 5 / 12 / 5 / 4 / 3 / 0 |
| KITTI calibrated drives | 79 per 10 min | 7.2 per 10 min |
| EvTTC fused TTC error | 11.5% | 11.5% |
| EvTTC warned by true TTC 2.0 s | 5 / 5 | 5 / 5 |

The change removed 85% of false alarms without costing a single real warning. The remaining raises still need work (next candidates: estimating our own speed to separate oncoming from leading traffic, and per-clip review of what the 51 are).

## Perception cost (C++, laptop CPU in WSL, 300 frames of 4K dash cam)

| Step | Before | After speed-up |
|---|---|---|
| ego-rotation | 19.65 ms | **7.89 ms** |
| divergence | 1.04 | 1.06 |
| scale search | 0.39 | 0.49 |
| Horn | 0.32 | 0.26 |
| everything else | < 0.1 | < 0.1 |
| **total** | **22.0 ms** | **10.3 ms** |

Ego-rotation dominated: `recoverPose` triangulated every inlier to pick one of four poses. With 64 spread inliers and RANSAC capped at 200 iterations, KITTI yaw error is unchanged per drive. Correction (2026-10-10): PassMark puts a Cortex-A53 at 1.33 GHz about 15.9× below this laptop core per thread (196 vs 3124), not 3–6×, so this projects to roughly 120–180 ms per frame on one A53 core (about 6–8 FPS); the per-object work must use all four cores (or move ego-rotation to the PL) to reach 30 FPS. To be measured on the board. The README's "FPGA design vs CPU-only" section has the full comparison. Because Horn costs only ~0.3 ms here, the FPGA Horn engine (plan 3) would save little CPU time on this footage; its value is headroom for more objects and higher resolution.

## Oncoming-traffic exclusion (2026-10-09)

Most of the 51 remaining warnings were oncoming cars straight ahead on narrow streets: real short TTCs, but not the lead vehicle. The closing-speed ratio kappa = TTC_static / TTC_object (closing speed over our speed) is measured without calibration from the road's own flow: on a flat road TTC_road(y)·(y − y_FOE) is constant, measured on the road band where flow is large, and it gives the static TTC at any object's contact row. On synthetic road scenes kappa reads 0.95 / 2.37 / 0.47 for true 1.0 / 2.5 / 0.5. Objects with kappa > 1.8 are never the lead; an unmeasurable kappa never suppresses.

| Data | Lead-vehicle rule | + oncoming exclusion |
|---|---|---|
| Dash-cam clips, 30 min | 51 raises (17.0 / 10 min) | **2 raises (0.67 / 10 min)** |
| — per clip | 5 / 8 / 3 / 6 / 5 / 12 / 5 / 4 / 3 / 0 | 0 / 0 / 1 / 0 / 0 / 0 / 0 / 0 / 1 / 0 |
| KITTI calibrated drives | 7.2 / 10 min | 7.2 / 10 min (1 raise: parked car on a bend, heading lag) |
| EvTTC fused TTC error | 11.5% | 11.5% |
| EvTTC warned by true TTC 2.0 s | 5 / 5 | 5 / 5 |

## Known gap: late warnings for objects entering our path

Synthetic road scenes on true collision courses (object reaches our centre line at contact): a pedestrian crossing at 1.0 m/s warns at true TTC **0.80 s**, a car cutting in at 0.86 s, a cyclist at 0.37 s. Lead vehicles (parked, slower) warn at 2.17–2.60 s. The lead-vehicle rule only lets an object warn once it overlaps our width; crossing objects get there late. A pedestrian who clears our path correctly does not warn. Next: predicted path entry with a stability gate, measured against both the crossing tests and the false-alarm data.

## Pipelined, parallel design (2026-10-24)

Plan `docs/superpowers/plans/2026-10-23-pipelined-parallel-fcw.md` fixed the 16 issues from a low-level review of the KV260 execution flow. All numbers below are on the final code, re-run on 2026-10-24.

**Ego-rotation.** Linear rotation + FOE replaced essential versus homography model selection. It is a 5-unknown Gauss–Newton with Cauchy weights, a 2-round inlier refit at 0.3 px, and a stationary test at the 75th-percentile residual < 0.15 px.

| KITTI yaw RMS (805 pairs) | 0005 | 0014 | 0091 | all |
|---|---|---|---|---|
| essential / homography, DIS, full resolution (before) | 0.589 | 0.817 | 0.572 | 0.681 °/s |
| linear, DIS, full resolution | 0.357 | 0.358 | 0.236 | **0.312 °/s** |
| linear, DIS, 512×160 | 0.422 | 0.513 | 0.270 | 0.408 °/s |
| linear, AMD's LK kernel (C-sim), 512×160 | 1.144 | 0.602 (292 pairs) | 0.370 | **0.670 °/s** |

C++ ego-rotation time per frame: 8.03 → 1.16 ms when it was ported (Task 2), 6.9× against a gate of 10×, accepted for the 2.2× accuracy gain. It is 0.87 ms in the final `fcw_app` run.

**Divergence.** The inlier mask is now last frame's affine fit, in fixed point: pixels within 1 px, flow in 1/64 px, fit scaled by 2^16. RANSAC is the fallback when there is no fit, the fit is more than 1 frame old, or fewer than half the pixels agree. With the mask counted inside the engine, divergence needs no per-pixel pass on the Arm cores.

| EvTTC (5 approaches) | DIS flow | AMD's LK kernel (C-sim) |
|---|---|---|
| fused median relative error | **11.9 %** | **12.1 %** |
| divergence alone | 9.5 % | 8.6 % |
| warned before true TTC 2.0 s | 5 / 5 | 5 / 5 |
| per sequence, fused (CCRs-1-high / 1-low / 1-medium / 2-high / 2-low) | 0.062 / 0.117 / 0.141 / 0.114 / 0.129 | 0.061 / 0.122 / 0.141 / 0.121 / 0.132 |

The DIS figure was 11.48 % right after the divergence change (Task 3). The processing size, now 512 wide with the source's aspect (Task 7), and the integer gray formula with the replicate-border pyramid (Task 11) moved it to 11.9 %.

**Scale search ablation (Task 5).** Without scale search, EvTTC is 11.97 % against 11.48 % with it, both 5/5, which met the plan's rule for disabling it. But the far head-on synthetic case then warned only at true TTC 3.2 s, against a requirement of ≤ 3.0 s. Scale search stays on.

**κ grid.** The road constant is now sampled on an 8×4 px grid instead of 4×2: within 5 % on all three synthetic cases, at a quarter of the samples.

**Perception cost**, `fcw_app`, 300 frames of 4K dash cam, laptop in WSL, 3 object workers:

| | before | after |
|---|---|---|
| ego-rotation | 8.03 ms | 0.87 ms |
| TTC methods | 2.58 ms | 0.71 ms |
| Horn + flow sums for every box (software engine, Option A) | (in TTC methods) | 0.78 ms |
| flow to fixed point | — | 0.26 ms (2.52 ms before vectorising) |
| lead, κ, warnings | 0.53 ms | 0.11 ms |
| tracking | (in 0.06 ms) | 0.34 ms |
| a frame through perception | 11.2 ms | **4.5 ms** |

End to end, the laptop runs at 20–34 FPS between runs, set by T1/T2: software 4K decode and DIS flow.

`fcw_cpu_baseline` runs the same code serially on one core with plain dense LK flow, at 30 frames and 512×288:

| Step | ms per frame |
|---|---|
| preprocess | 46.6 |
| LK | 1,611 (1,505 with OpenCV) |
| tracking | 0.44 |
| ego-rotation | 1.07 |
| engine (software) | 0.44 |
| quantise | 0.36 |
| **divergence** | **1.86** |
| scale search | 0.21 |
| warnings | 0.10 |
| perception total | **5.4** (was 7.5) |

Divergence costs 4.6× more on LK flow than on DIS flow (0.40 ms). This is not yet explained. More RANSAC fallbacks would be the first suspect; profile it on the board.

On a Cortex-A53 (×15.9, PassMark) this projects to T3a ≈ 19–24 ms and T3b ≈ 22–44 ms with the PL engine. The low end of T3b is the DIS cost, the high end the LK-flow cost; Horn in software adds 7–12 ms. Each stage runs on its own core, so this is ≈ 23–30 FPS. Before, perception was one serial stage of 120–180 ms.

**Correctness of the parallel runtime.** The five-stage runtime with a 3-worker object stage equals the serial pipeline frame by frame on all 11 golden scenarios. ThreadSanitizer, run with `setarch -R`, reports only races inside OpenCV's TBB workers and GDAL.

**Kernels in C-simulation.**

| Kernel | Result |
|---|---|
| Horn engine | bit-exact on 7 stream cases and 2 top-level (DDR) cases, including predicted-fit masks and level 1 |
| LK wrapper | equals the host-driven reference: 0 of 147,456 flow words differ |
| Preprocess | gray, INT8 tensor and pyramid exact against the software definition; Vitis bilinear resize vs `cv::resize` on the gray image of a 1080p test frame: max 2, mean 0.37 levels |
| Software | now uses the kernel's gray formula `(4899 R + 9617 G + 1868 B + 8192) >> 14` and replicate borders in the pyramid |

**Fabric.** [2026-10-fabric-budget.md](2026-10-fabric-budget.md): the design as written needs about 140k of 117k LUTs. The fallbacks, in order, are per-term accumulator widths, fewer engine lanes, B3136, and Horn on the A53. Synthesis is waiting on the device data.

## To finish this report

1. Close the late-warning gap for crossing and cut-in targets without losing the false-alarm result.
