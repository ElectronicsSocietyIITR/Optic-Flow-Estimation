# Fabric budget on the KV260 (K26 SOM)

## 00. Linked overlay: built, timing met (2026-10-28)

`hls/system/build_overlay.sh`, Vitis/Vivado 2023.1, Kria platform built from `kv260_platform.tcl`. DPU B3136 (URAM build) + `lk_wrapper` + `preprocess`. **The Horn engine is left out** (fallback 5.4).

| Routed, in the linked design | LUT | BRAM36 | URAM | DSP | Clock | Setup slack |
|---|---|---|---|---|---|---|
| DPU B3136 | 45,817 | 78 | 40 | 566 | 275 / 550 MHz | +0.036 / +0.030 ns |
| `lk_wrapper` | 28,720 | 47 | 0 | 93 | 220 MHz | +0.012 ns (the 220 MHz clock) |
| `preprocess` | 9,549 | 9 | 0 | 45 | 220 MHz | |
| platform, interconnect | 5,601 | 0.5 | 0 | 0 | 100 MHz | +6.07 ns |
| **Device** | **89,681 (77 %)** | **134.5 of 144 (93 %)** | 40 of 64 | 704 (56 %) | | **WNS +0.012, WHS +0.008 ns** |

How the out-of-context plan in section 0 changed when linked:

1. **BRAM, not LUTs, ran out.** The first link needed 233 BRAM tiles:
   * The DPU's URAM build still uses 78 BRAM tiles. Section 0 assumed PG338's 2.
   * v++ put register slices and FIFO-mode upsizers on the HP-port interconnects, 56 tiles and ≈ 14k LUTs. `strip_interconnects.tcl`, a post-system-link hook, sets the interconnects to STRATEGY 0 and turns them off.

   With the engine the design still needed 148 tiles, so the engine is out (fallback 5.4; `ENGINE=1` links it back in).
2. **Clocks.** At 250 / 300 / 600 MHz the linked design missed timing by up to 0.22 ns on all three, and v++ added a second clock wizard for "600" (599 MHz). The platform now makes 100 / 220 / 275 / 550 MHz from one 1100 MHz VCO, so every clock is exact and the DPU's pair is phase-aligned.
3. **The last 0.07 ns** was LK's 3×3 median network: 7 logic levels in one cycle, at an HLS target of 250 MHz. LK is now scheduled for 320 MHz, which only adds pipeline registers, so the output is bit-identical. Implementation uses `Performance_ExplorePostRoutePhysOpt`.

**Cost in speed:**
* kernels 250 → 220 MHz: preprocess ≈ 9.8 ms, LK ≈ 6.6 ms;
* DPU 300 → 275 MHz: YOLOv9t ≈ 14–43 ms, ≈ 9 % slower;
* Horn in software: T3b ≈ 29–56 ms on the A53.

The projection becomes ≈ 18–30 FPS, from 21–30.

## 0. Result: routed on the xck26 (2026-10-25)

Synthesized with Vitis HLS 2026.1 and **placed and routed out of context in Vivado 2026.1** on `xck26-sfvc784-2LV-c` (BASIC licence). These numbers replace every estimate below. Sections 1–7 are kept as the record of how we got here.

| Kernel, chosen configuration | LUT | FF | BRAM36 | DSP | Clock achieved | Power, dynamic (Vivado) |
|---|---|---|---|---|---|---|
| `preprocess`: one `pyrDown` reused level by level | 9,598 | 11,419 | 16 | 45 | 3.838 ns (261 MHz) | 0.22 W |
| `lk_wrapper`: FCW LK fork, 3×3 median, 34 remap lines | 26,375 | 31,221 | 88 | 93 | 3.581 ns (279 MHz) | 0.72 W |
| `ttc_engine_top`: 4 lanes, per-term accumulators | 15,159 | 17,493 | 21 | 85 | 3.318 ns (301 MHz) | 0.51 W |
| **Our kernels** | **51,132** | 60,133 | 125 | 223 | | **1.45 W** |
| DPU B3136 (PG338, URAM build) | 47,270 | 85,782 | 2 + 64 URAM | 436 | 300 MHz | |
| **Fabric total** | **≈ 98.4k of 117k (84 %)** | 146k (62 %) | 127 of 144 (88 %) | 659 (53 %) | | |

**It fits.** Platform interconnect adds a few thousand LUTs (expect 88–92 % in the linked design). BRAM is the tightest resource; the LK's remap lines are the lever, at about 2 BRAM per line.

**Timing.** `preprocess` and `lk_wrapper` miss 300 MHz. Run our kernels on a **250 MHz** clock, with the DPU on its own 300 MHz clock. At 250 MHz, preprocess is ≈ 8.6 ms per 1080p frame and LK ≈ 5.8 ms at 512×288. That changes nothing at the system level: the DPU still sets T2.

**How we got here.** The first synthesis had our three kernels at 213k LUTs. The changes:

| Kernel | Change | Effect |
|---|---|---|
| preprocess | one `pyrDown` instance instead of four | 32.3k → 19.3k csynth |
| LK | the per-pass median filters are 29.7k of the core's 57k csynth LUTs; a 3×3 median keeps most of the 5×5's accuracy, while removing the median entirely doubles divergence error on the kernel's fixed-point flow; 34 remap lines | 75.0k → 49.3k csynth |
| engine | per-term accumulator widths, natural-width products, 4 lanes with host batching | 105.7k → 35.8k csynth |

csynth estimates were then 0.43–0.57× too high for LUTs and up to 0.22× for BRAM against routed results. That is why routed numbers, not csynth, decide the fit.

Accuracy of the LK choices, all bit-true (AMD's kernel in C-simulation). The chosen build keeps TTC accuracy and warning timing; yaw is about 27 % worse than AMD's default, still inside the 1.0 °/s target:

| LK build | KITTI yaw RMS | EvTTC fused | Divergence alone | Warned before 2 s |
|---|---|---|---|---|
| AMD's 5×5 median, 50 lines | 0.670 °/s | 12.1 % | 8.6 % | 5/5 |
| 3×3 median, 50 lines | 0.817 °/s | 12.2 % | 9.7 % | 5/5 |
| **3×3 median, 34 lines (chosen)** | **0.849 °/s** | **12.2 %** | **9.8 %** | **5/5** |
| no median, 50 lines | 0.688 °/s | 12.2 % | 16.3 % | 4/5 |
| no median, 34 lines | 0.673 °/s | 12.1 % | 16.3 % | 5/5 |
| no median, 26 lines | 0.737 °/s | 12.2 % | 16.2 % | 5/5 |

## 1. Bottom line (estimate, 2026-10-24, superseded by section 0)

- **The design as written does not fit.** DPU B4096, Vitis LK, a 16-lane Horn engine with 64-bit accumulators, preprocessing and interconnect add up to about 140k LUTs. The K26 has 117k.
- This matches outside evidence:
  - AMD's own KV260 smart-camera app ships a **B3136** DPU, not B4096.
  - A KV260 developer reports that one B4096 fits only when no other accelerator is instantiated.
- **Likely configuration:** B3136, LK, preprocessing, and the Horn engine either narrowed (4 lanes, accumulator widths sized per term) or kept in software on the Arm. That comes to about 82–93% LUTs. Section 5 gives the order to try, and synthesis decides.
- **Time is not the constraint; area is.** Every fabric kernel finishes in a few milliseconds of the 33 ms frame (section 6), so area can be traded for passes.

## 2. Device (DS987, K26 SOM)

| Resource | K26 |
|---|---|
| CLB LUTs | 117,120 |
| CLB flip-flops | 234,240 |
| Block RAM (36 Kb) | 144 (5.1 Mb) |
| UltraRAM (288 Kb) | 64 |
| DSP slices | 1,248 |

Closing timing at 300 MHz usually needs LUT use at or below about 80% (≈ 94k).

## 3. Per block

| Block | LUT | FF | BRAM36 | URAM | DSP | Source |
|---|---|---|---|---|---|---|
| DPU B4096, BRAM build | 53,540 | 105,008 | 257 | 0 | 562 | PG338 (ZCU102 sample) |
| DPU B4096, URAM build | 52,681 | 104,562 | 2 | 68 | 562 | PG338 (ZCU104 sample) |
| DPU B3136, URAM build | 47,270 | 85,782 | 2 | 64 | 436 | PG338 (ZCU104 sample) |
| LK (`densePyrOpticalFlow`, 1 px/clock, 11×11, 5 levels × 5 iterations) | 30.8–32.2k | 17–42k | 107 at FHD; *est.* 40–70 at 512 wide | 0 | 52–83 | Vitis Vision docs (FHD / 4K builds) |
| Horn / moment engine as written (16 lanes × 23 terms × 64 bit) | *est.* 35–45k | *est.* 35k | 3 | 0 | *est.* 60 | bottom-up, below |
| Horn engine, 4 lanes, per-term widths | *est.* 11–15k | *est.* 8k | 3 | 0 | *est.* 30 | bottom-up |
| Preprocess (bilinear resize, gray, INT8 quantiser, 4 × pyrDown) | *est.* 8–12k | *est.* 10k | *est.* 10–20 | 0 | *est.* 20 | Vitis resize / pyrDown sizes |
| SmartConnect and platform glue | *est.* 5–10k | | | | | depends on the platform |

**DPU memory.** B4096 does not fit in URAM alone: it needs 68 URAM-equivalents and the K26 has 64. So on the K26 it needs a mixed build of 64 URAM plus about 15–20 BRAM. B3136 fits in the 64 URAM by itself, which leaves almost all 144 BRAM to LK, preprocessing and the engine.

**Engine, bottom-up.** The core keeps `acc[16][23]` as `ap_int<64>`, completely partitioned. That is 23,552 accumulator bits: about 23.5k LUTs (one carry-chain LUT per bit) and as many flip-flops. On top of that, per lane, it has:

- the incremental fit predictor (`pred_u`/`pred_v`, eight 64-bit registers and adders);
- the 1 px inlier test (two 64-bit subtractions, clamps, two 17×17 squares);
- two box-membership tests.

That is roughly 10k LUTs for 16 lanes. The 22 products are computed once per pixel and shared by all lanes (about 25 DSPs). The five m_axi bundles add about 6k LUTs.

## 4. Totals (LUT)

| Configuration | LUT (est.) | % of 117k | Verdict |
|---|---|---|---|
| A. As written: B4096 + LK + 16-lane 64-bit engine + preprocess + glue | 134–148k | 115–126% | does not fit |
| B. B4096 + LK + 4-lane per-term engine + preprocess + glue | 108–122k | 92–104% | does not fit, or fails timing |
| C. B3136 + LK + 4-lane per-term engine + preprocess + glue | 102–116k | 87–99% | borderline; synthesis decides |
| D. B3136 + LK + preprocess + glue; Horn engine on the A53 | 91–101k | 78–86% | likely fits |

**DSP** is not a constraint: at most about 730 of 1,248 (configuration A).

**BRAM** fits in every configuration with B3136. With B4096 it fits only as the mixed URAM/BRAM build.

## 5. Fallbacks, in order

The spec lists three fallbacks: 8-box engine, then 48-bit accumulators, then B3136. The ruling below changes two of them and adds a fourth. The order is by what each costs in results.

### 5.1 Per-term accumulator widths

Costs nothing and stays bit-exact. This **replaces the flat 48-bit accumulators.** Flat 48 bits would overflow g², ex·g and ey·g for large boxes. The output would then differ from the int64 software reference, with no error raised.

Each sum is bounded by its term's worst case over the largest region (512 × 512 = 2^18 pixels) and by the declared operand ranges:

- |ex|, |ey| ≤ 2040 (Sobel of the frame *sum* I(t−1) + I(t), which runs 0–510; the kernel holds them in `ap_int<12>`);
- |rx|, |ry|, |fx|, |fy| ≤ 1024 (`ap_int<11>`);
- |g| ≤ 2^21 (`ap_int<22>`; |rx|·|ex| + |ry|·|ey| stays below that while the principal point is inside the frame);
- |et| ≤ 255;
- u, v are int16.

| Horn term | width | Flow term | width |
|---|---|---|---|
| ex², ex·ey, ey² | 41 | count | 20 |
| ex·g, ey·g | 51 | fx, fy | 29 |
| g² | 62 | fx², fx·fy, fy² | 40 |
| ex·et, ey·et | 38 | u, v | 34 |
| g·et | 48 | fx·u, fy·u, fx·v, fy·v | 45 |
| et² | 35 | | |
| count | 20 | | |

That is 912 bits per box against 1,472 at 64 bits, 38% fewer accumulator LUTs and flip-flops. (Corrected 2026-10-24: an earlier version of this table assumed |ex| ≤ 1020.)

- **Precondition:** width and height ≤ 512, which the line buffers already require and `processingSize` guarantees.
- **Verification:** the existing golden C-simulation cases, plus one worst-case case: a full-frame box over a 0/255 checkerboard with the principal point at the corner.

### 5.2 Fewer engine lanes, more passes

This keeps every object. It **replaces "8-box engine"**, which read literally drops objects 9–16.

The kernel is built with `kMaxBoxes = 4` (or 8). `XrtMomentEngine` then runs ceil(n / lanes) passes per pyramid level: every box at level 0 for its flow moments, and the Horn boxes at their own levels.

- **Cost:** about 0.5 ms per extra level-0 pass at 300 MHz (512 × 288 = 147k clocks). At 4 lanes, 16 boxes add 1.5 ms.
- **Saving:** the accumulator bank shrinks by 4×. Combined with 5.1, it drops from 23.6k to 3.6k bits (4 lanes × 912).

### 5.3 B3136 DPU

- **Saving:** 5.4k LUTs and 126 DSPs. It also fits in URAM alone, which frees the BRAM.
- **Cost:** about 1.31× longer DPU time, the ratio of peak operations per clock (4096/3136). YOLOv9t at 512×288 is about 2.8 GFLOP.

T2's time is preprocess plus max(LK, DPU), so the DPU sets the frame rate. Section 6 has the margin.

### 5.4 Horn engine on the A53

This fallback is new; the spec did not list it. **It is the one the built overlay uses** (section 00).

- Option A already isolates the engine behind `MomentEngine`. So the board can run `SoftwareMomentEngine` with no other code change, which frees all of the engine's LUTs.
- **Cost:** Horn sums and moments take 0.35 ms per frame on the PC. That projects to about 5–6 ms on one A53 core (PassMark ×15.9), on the object stage's own thread.

The engine's purpose was headroom (more objects, higher resolution), not a speed-up that the 30 FPS target depends on.

## 6. Time per frame in the fabric (300 MHz, 1 pixel per clock, 512×288)

| Kernel | Clocks | ms | Note |
|---|---|---|---|
| Preprocess | 2.07M (1080p input) | ≈ 6.9 | bound by reading the source frame; ≈ 0.6 ms for a 512-wide camera |
| LK, 5 levels × 5 iterations | 147k × 1.33 × 5 ≈ 0.98M | ≈ 3.3 | plus per-pass fill; runs in parallel with the DPU |
| Horn engine, 16 lanes | ≈ 0.2M | ≈ 0.7 | one pass per level in use |
| Horn engine, 4 lanes | ≤ 0.9M | ≤ 3 | 16 boxes |
| DPU, YOLOv9t | — | 10–30 (B4096); ×1.31 for B3136 | README estimate; to be measured |

- **DDR bandwidth** is small next to the 19.2 GB/s peak of the KV260's 64-bit DDR4-2400:
  - preprocess about 0.25 GB/s at 30 FPS (1080p BGRx in);
  - LK about 0.3 GB/s;
  - the engine at most about 0.2 GB/s;
  - the DPU about 1–2 GB/s.
- **T2 throughput:** preprocess and DPU run back to back inside T2 today. With B3136 and a slow DPU run, T2 can drop below 30 FPS before any other stage does. If the measurement shows that, the next fix is a second T2 thread so preprocess of frame N+1 overlaps the DPU of frame N. It costs no fabric.

## 7. Getting the real numbers

The user installs the device data: Vitis installer, **Add Design Tools or Devices**, **Zynq UltraScale+ MPSoC**. Then, per kernel:

```bash
cd hls/<kernel>
vitis-run --mode hls --csynth --config hls_config.cfg --work_dir build/hls      # LUT/FF/BRAM/DSP estimate, II, latency
v++ -c --mode hls --config hls_config.cfg --work_dir build/xo                     # packaged kernel (.xo) for linking
```

Then link with the DPU's XO from the DPU TRD and the KV260 Vitis platform, using `hls/system/link.cfg`. That config is a draft: check its memory names with `platforminfo`. Record the utilisation and timing reports here, and walk section 5 until the design fits with positive slack.

csynth numbers are pre-implementation and usually overstate LUTs by 10–30%. Only the post-route numbers from the link decide.

## Sources

- [PG338 DPUCZDX8G resource use](https://docs.amd.com/r/3.4-English/pg338-dpu/Resource-Use)
- [DS987 K26 SOM programmable logic](https://docs.amd.com/r/en-US/ds987-k26-som/Programmable-Logic)
- [Vitis Vision: dense pyramidal LK](https://docs.amd.com/r/2023.2-English/Vitis_Libraries/vision/api-reference.html_2_63)
- [Vitis Vision: LK dense pyramidal benchmark](https://docs.amd.com/r/2023.1-English/Vitis_Libraries/vision/denseLKpyrOF-bm.html_1)
- [KV260 smart-camera DPU B3136 report (element14)](https://community.element14.com/members-area/personalblogs/b/blog/posts/kria-reproducing-the-benchmark-results)
- [B4096 fits only without other accelerators (element14)](https://community.element14.com/technologies/fpga-group/b/blog/posts/kv260-vvas-sms-2021-1-part3)
