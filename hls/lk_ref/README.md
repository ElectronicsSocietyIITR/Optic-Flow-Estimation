# Vitis Vision dense LK in C-simulation

The board computes flow with AMD's `xf::cv::densePyrOpticalFlow` (Vitis Libraries 2024.2,
5 levels, 5 iterations, 11×11 window). `lk_ref` runs that exact kernel on the PC, so flow on the
PC can match the board bit for bit.

```bash
# once: AMD's headers (git-ignored)
git clone --depth 1 --branch 2024.2 --filter=blob:none --sparse https://github.com/Xilinx/Vitis_Libraries.git third_party/Vitis_Libraries
git -C third_party/Vitis_Libraries sparse-checkout set vision/L1/include vision/L1/examples/lkdensepyrof
make -C hls/lk_ref build/lk_ref                       # in WSL; XILINX_HLS_INCLUDE=<Vitis>/include
./hls/lk_ref/build/lk_ref prev.png curr.png flow.bin  # ~4 s per 512x288 pair
```

In the pipeline: `--flow vitis` (`collision_avoidance/fcw/vitis_flow.py`), which caches each frame
pair's flow and fills the cache in parallel (`VitisFlow.prefill`).

**Flow format** (from AMD's source): 32 bits per pixel, u in the upper 16 bits and v in the lower
16, each `ap_fixed<16,10>`, i.e. 1/64 px. That is exactly the Horn engine's int16 flow at 1/64 px,
so the engine's input packer only splits the word.

**Our software LK vs the kernel** (512×288 dash-cam pair, typical flow 0.91 px): median
difference 0.185 px, 75th percentile 0.59 px, 90th percentile 1.77 px. The kernel builds a
Gaussian pyramid, median-filters the flow (5×5) after every pass and works in fixed point.
Following the first two in software only reached 0.165 / 1.28 px, so results that must hold on
the board use the kernel itself (`--flow vitis`).
