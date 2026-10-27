# KV260 deployment bundle

The built overlay and the detector compiled for it, ready to copy to a KV260 running Kria
Ubuntu 22.04. Built on 2026-10-28 by `hls/system/build_overlay.sh` (Vitis/Vivado 2023.1); routed with
all timing met (worst setup slack +0.012 ns, hold +0.008 ns).

| File | What it is |
|---|---|
| `fcw/fcw.bit.bin` | PL bitstream: DPU B3136 (275 / 550 MHz), `lk_wrapper` and `preprocess` (220 MHz) |
| `fcw/fcw.dtbo` | device-tree overlay with the `zocl` node |
| `fcw/fcw.xclbin` | XRT container for the same design; the host opens the kernels from it |
| `fcw/shell.json` | Kria app descriptor (`XRT_FLAT`) |
| `yolov9t_kv260.xmodel` | YOLOv9t, Hardswish, distilled from the SiLU original, INT8, compiled for this DPU (fingerprint `0x101000056010406`) as one DPU subgraph |
| `SHA256SUMS` | checksums of the above |

The Horn engine is not in this overlay (BRAM); the host falls back to `SoftwareMomentEngine`.

## On the board

Not yet run on a board: the bring-up is plan 2, Task 10. The steps below are what the code expects.

The KV260's HDMI and DisplayPort outputs are both driven by the PS display controller (a splitter on
the carrier), so either shows the desktop and the app. Connect the monitor, keyboard, mouse and
Ethernet **before** powering on.

### Once: set up Kria Ubuntu 22.04 and build the app

Log in as `ubuntu` / `ubuntu` on the desktop (you set a new password), open a terminal:

```bash
sudo snap install xlnx-config --classic --channel=2.x
sudo xlnx-config.sysinit
sudo apt install -y git build-essential cmake libopencv-dev libgtest-dev libdrm-dev      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-libav v4l-utils
sudo usermod -aG video,render $USER     # camera, display and fabric without sudo; log out and in again
git clone https://github.com/Shyam-24o7/Optic-Flow-Estimation.git
```

The Vitis AI 3.5 runtime (VART, XIR) is also needed and is not in the Kria image. AMD's
[Kria-RoboticsAI](https://github.com/amd/Kria-RoboticsAI) install script
(`install_update_kr260_to_vitisai35.sh`, written for the KR260, same SOM and image) is the
reference. Then:

```bash
cd ~/Optic-Flow-Estimation/host
cmake -B build -DFCW_BOARD=ON && cmake --build build -j4     # add -DXRT_DIR=/opt/xilinx/xrt/share/cmake/XRT if XRT is not found
```

### Every run: load the overlay

```bash
cd ~/Optic-Flow-Estimation/deploy/kv260
sha256sum -c SHA256SUMS
sudo mkdir -p /lib/firmware/xilinx/fcw && sudo cp fcw/* /lib/firmware/xilinx/fcw/
sudo xmutil unloadapp
sudo xmutil loadapp fcw
v4l2-ctl --list-devices                  # the camera's /dev/videoN
```

### Run, option A: in a window on the desktop (easiest)

From the desktop terminal, without `sudo` (a root process cannot open a window on your desktop):

```bash
cd ~/Optic-Flow-Estimation
APP="./host/build/fcw_app --board --xclbin /lib/firmware/xilinx/fcw/fcw.xclbin --xmodel deploy/kv260/yolov9t_kv260.xmodel"
$APP --window --source /dev/video0 --fx 1370                       # live camera
$APP --window --source ~/drive.mp4 --fx 500 --output ~/fcw_out.mp4  # a file, also recorded with the overlay
```

The window shows the video with the boxes, TTC and warning banner. Drawing it costs Arm time, so
the FPS printed at the end is lower than the full-screen mode's.

### Run, option B: full screen, the deployment path

The app drives the monitor directly (video and annotations on two hardware planes, blended by the
display controller), so the desktop has to let go of it:

```bash
sudo xmutil desktop_disable      # the screen switches to a text login: log in there
cd ~/Optic-Flow-Estimation
./host/build/fcw_app --board --xclbin /lib/firmware/xilinx/fcw/fcw.xclbin      --xmodel deploy/kv260/yolov9t_kv260.xmodel --source /dev/video0 --fx 1370
# Ctrl+C to stop; the summary line and FPS are printed. "Permission denied" on /dev: run it with sudo
sudo xmutil desktop_enable       # back to the desktop
```

The app picks the DRM card with a connected monitor itself (`--card /dev/dri/cardN` overrides it).
`--no-display` runs headless; add `--output out.mp4` to record.

**Inputs.** `--source /dev/videoN` (USB or MIPI camera, frames dropped when processing falls behind) or
a video file (every frame processed). Sources above 1080p are scaled down. Still images are not an input:
the system measures motion between frames, so turn an image sequence into a video first
(`ffmpeg -framerate 30 -i %06d.png clip.mp4`).

**`--fx`** is the focal length in pixels at the source's resolution: `width / (2 tan(HFOV / 2))`, e.g.
about 1370 for a 1080p camera with a 70° field of view. TTC does not depend on it; the driving corridor does.
