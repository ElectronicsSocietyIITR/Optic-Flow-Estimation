# Vitis AI deployment (KV260 DPU)

Toolchain: Vitis AI 3.5 PyTorch docker (`xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306`,
`pytorch_nndct 3.5.0+torch1.13.1` - the same build that produced the 2025 quantization
files). Target `DPUCZDX8G_ISA1_B3136`, the DPU size that fits next to our kernels
([fabric report](../../docs/superpowers/reports/2026-10-fabric-budget.md)); the September results
below were for B4096. No KV260 board was available, so on-device latency is not measured.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Vitis AI image + Ultralytics/W&B, numpy and torch pinned to the builds `pytorch_nndct` was compiled against |
| `run_docker.sh` | runs a command in the image with the repo at `/workspace`, W&B credentials and the torch hub cache mounted |
| `quantize.py` | `inspect` / `calib` / `test` (evaluate vs float, export `.xmodel`) for `pwcnet` and `yolov9t` |
| `compile.sh` | `vai_c_xir` for the KV260 + subgraph report |
| `yolo_dpu.py` | YOLOv9t graph up to the Detect-head convolutions; CPU decode + NMS |
| `../../collision_avoidance/pwcnet.py` | PWC-DC-Net as a plain `nn.Module`; shared with the host pipeline |
| `../../collision_avoidance/yolo_pruning.py`, `../../scripts/prune_yolo.py` | Hardswish swap + channel pruning + fine-tuning (Colab notebook in `notebooks/`) |

## Run

```bash
docker build -t kria-vitis-ai:3.5 deploy/vitis_ai
CLIPS=$(ls PWC/ptlflow/*.mp4 | sed 's|^|../../|')      # dash-cam clips for calibration
for m in pwcnet yolov9t; do
  deploy/vitis_ai/run_docker.sh python quantize.py --model $m --mode inspect --videos $CLIPS
  deploy/vitis_ai/run_docker.sh python quantize.py --model $m --mode calib   --videos $CLIPS
  deploy/vitis_ai/run_docker.sh python quantize.py --model $m --mode test    --videos $CLIPS
done
deploy/vitis_ai/run_docker.sh ./compile.sh            # -> compiled/<model>/<model>_kv260.xmodel + subgraphs.txt
```

`compile.sh` targets `arch_b3136.json` by default. Once the overlay is built, compile against
the DPU actually in it: `ARCH=/opt/fcw/build/arch.json deploy/vitis_ai/run_docker.sh ./compile.sh yolov9t`.
`run_docker.sh` passes `ARCH`, `IN_DIR` and `OUT_DIR` into the container.

For YOLOv9t, `quantize.py` sets the quantizer option `nndct_convert_silu_to_hswish`, so the
original `yolov9t.pt` is quantised with Hardswish in place of SiLU; the test-mode score is
against the untouched SiLU float model. For the pruned detector pass
`--yolo-weights ../../outputs/models/yolov9t_pruned.pt` (produced by `scripts/prune_yolo.py`,
locally or with the Colab notebook).

## What the inspector found

**PWC-Net: LeakyReLU slope.** The DPU implements LeakyReLU only with slope
26/256 = 0.1015625. PWC-Net is trained with 0.1, and with 0.1 every one of its 134
conv + activation pairs was assigned to the CPU (the 2025 quantization used 0.1 too).
The deploy graph therefore uses 26/256 (`pwcnet.set_leaky_slope`). Float-model effect on
60 dash-cam frame pairs: mean end-point error 0.073 px (typical flow magnitude 2 px),
worst frame 0.93 px. What still runs on the CPU is the backward warp (`grid_sample`,
masks) and the correlation (`im2col`, channel mean) at each of the five pyramid levels.

**YOLOv9t: SiLU.** All 179 SiLU activations cannot be converted for the DPU (each also
forces layout permutes around it). Swapping SiLU for Hardswish leaves only the ELAN
`chunk` slices on the CPU, and after the pruning rewrite (which removes the chunks) the
inspector reports **"All the operators are assigned to the DPU."** The swap changes the
function, so it is fine-tuned together with pruning (`notebooks/prune_yolov9t_colab.ipynb`).

## Other fixes relative to the 2025 attempt

1. Wrong network: the pipeline runs PWC-**DC**-Net ("things" weights); the old copy was
   plain PWC-Net with "sintel" weights.
2. Double LeakyReLU after correlation; missing warp mask.
3. Channel order: calibration fed RGB where ptlflow feeds plain PWC-Net BGR.
4. Resolution: calibration at 256x256, export at 384x448. Now 384x512 throughout.

## Results (Sept 2026)

Calibration: 200 samples, evaluation: 48-50 disjoint samples, both from the eight Nexar
dash-cam clips at 384x512. All runs are in W&B (job types `quantize-calib` / `quantize-test`).

| Model | Quantized vs float | Compiled for KV260 | Top-level subgraphs |
|---|---|---|---|
| PWC-DC-Net (LeakyReLU 26/256) | EPE 0.95 px (float flow on the eval pairs averages 0.72 px) | yes, `pwcnet_kv260.xmodel` 13 MB | 11 DPU + 10 CPU (warp + correlation per pyramid level) |
| YOLOv9t, original (SiLU) | raw head MAE 0.34 | **no**: compiler aborts, `Op_type 18 is invalid for xcompiler` | - |
| YOLOv9t, pruned 30 % + Hardswish, fine-tuned | detection F1 vs float 0.76 (raw head MAE 0.45) | **yes**, `yolov9t_kv260.xmodel` 3.8 MB | **1 DPU (whole network)** + 3 CPU output transfers |
| **YOLOv9t, Hardswish only (recommended)** | detection F1 vs float 0.84 (raw head MAE 0.41) | **yes**, 4.8 MB | **1 DPU (whole network)** + 3 CPU output transfers |

The September F1 scores are against the float *Hardswish* model, so they measure INT8 alone.

## Results for B3136 (Oct 2026)

Calibration: 190 samples, evaluation: 50 disjoint samples from the ten `data/normal_driving` clips
at 288x512, the pipeline's DPU input.

| Model | Detection F1 vs float SiLU YOLOv9t | Compiled for B3136 | Subgraphs |
|---|---|---|---|
| YOLOv9t, original weights, SiLU kept | - | **no**: 358 layout transposes on the CPU, then `Op_type 18 is invalid for xcompiler` | - |
| Hardswish, float (no quantisation) | 0.59 (40 frames) | - | - |
| **Hardswish, INT8 (current `yolov9t_kv260.xmodel`)** | **0.64** | **yes**, 4.8 MB | **1 DPU (whole network)**, 0 CPU ops, 3 output transfers |

The scenes have about two collision-relevant detections per frame, so these F1 values are
noisy; what they show is that the SiLU → Hardswish swap, not INT8, costs most of the
accuracy (on COCO: 0.378 → 0.317 mAP50-95). Planned next: distil the Hardswish model from the
SiLU one on the dash-cam clips (the SiLU model's detections as labels) on a GPU.

PWC-Net with Vitis AI fast-finetune (AdaQuant) was started but stopped after ~11 h at 47/81
layers (projected ~15 h more on CPU), so only the plain post-training result above exists.

### Pruned YOLOv9t accuracy (COCO val2017, 640 px)

| Model | mAP50-95 | mAP50 | GFLOPs | Params |
|---|---|---|---|---|
| Original YOLOv9t (SiLU) | 0.378 | 0.524 | 8.40 | 2.13 M |
| Hardswish + 30 % channel pruning, before fine-tuning | 0.000 | 0.000 | 4.53 | 1.12 M |
| same, fine-tuned 20 epochs on 25 % of COCO train2017 | **0.158** | 0.242 | 4.53 | 1.12 M |
| **Hardswish only (no pruning), no fine-tuning** | **0.317** | 0.458 | 8.40 | 2.13 M |
| Hardswish only, fine-tuned (default Ultralytics warm-up) | 0.311 | 0.443 | 8.40 | 2.13 M |
| Hardswish only, fine-tuned gently (lr 0.001, no warm-up), 5 epochs | 0.317 | - | 8.40 | 2.13 M |

Fine-tuning lessons: Ultralytics' default warm-up (bias learning rate 0.1) is meant for
training from scratch; on the already-converged Hardswish model it collapsed mAP50-95 from
0.317 to 0.081 in the first epoch, and 19 more epochs only recovered 0.311. With lr 0.001 and
no warm-up the model holds 0.317 but does not improve, so on this data budget plain
fine-tuning cannot close the gap to 0.378; distillation from the SiLU model or longer
training on full COCO would be the next steps.

Pruned model, collision-relevant classes after fine-tuning (mAP50-95): person 0.36, bus 0.41, car 0.19,
motorcycle 0.19, truck 0.12, bicycle 0.08. The fine-tune was repeated independently on a
Colab T4 (0.159) and the local RTX 3050 Ti (0.158); both were still improving slowly at epoch
20. W&B runs `t6m4gvia` (local) and `bzx6jvbb` (Colab).

What this means:

* **YOLOv9t cannot be deployed as trained.** SiLU makes the compiler abort. Swapping in
  Hardswish (with the exact ELAN rewrite) maps the whole network onto one DPU subgraph at
  0.317 mAP50-95 (original 0.378) and INT8 keeps 84 % of its float detections - this is the
  recommended detector. Pruning 30 % of channels halves the compute but drops accuracy to
  0.158 after fine-tuning, so it is only worth it if the DPU turns out to be the bottleneck.
* **PWC-Net compiles, but post-training INT8 is not accurate enough for TTC.** An error of
  about 1 px swamps the sub-pixel motion between consecutive 30 FPS frames, and the TTC
  features are spatial derivatives of the flow. Quantization error accumulates through the
  five coarse-to-fine levels, each of which feeds a quantized flow estimate into a CPU warp.

## Host float benchmark (`scripts/benchmark_flow.py`, RTX 3050 Ti laptop, 384x512)

| Model | GPU mean | CPU mean |
|---|---|---|
| ptlflow PWC-DC-Net (reference) | 42.9 ms | 182 ms |
| DPU-graph PWC-DC-Net (pipeline default) | 29.8 ms | 251 ms |
| DPU-graph PWC-Net (no DC) | 25.1 ms | 209 ms |

The unfold-based correlation is slower than ptlflow's on the CPU; since correlation stays
on the KV260's Cortex-A53 cores, it is the likely on-board bottleneck. W&B run `uxtpuu6y`.

## Not done

* On-board latency (needs a KV260 and a VART runner), so the "72 % inference-time
  reduction" figure is still unverified.
* The Vitis AI Optimizer was not used for pruning; channel pruning is done with
  Torch-Pruning plus fine-tuning on a Colab GPU instead (the Vitis image here is CPU-only).
