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

```bash
sha256sum -c SHA256SUMS
sudo mkdir -p /lib/firmware/xilinx/fcw && sudo cp fcw/* /lib/firmware/xilinx/fcw/
sudo xmutil unloadapp && sudo xmutil loadapp fcw
./fcw_app --board --xclbin /lib/firmware/xilinx/fcw/fcw.xclbin --xmodel yolov9t_kv260.xmodel --source /dev/video0 --fx 200
```

Build the host with `cmake -DFCW_BOARD=ON` on the board (XRT, VART). Not yet run on a board:
the bring-up is plan 2, Task 10.
