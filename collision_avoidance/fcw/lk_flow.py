"""Dense pyramidal Lucas-Kanade with the Vitis Vision kernel's parameters: the board's flow, on the GPU.

Same algorithm as host/src/lk_flow.cpp (checked against it in host/tests/test_lk_flow.cpp):
per pixel, the whole 11x11 window is warped with that pixel's flow (exact LK, not
the smoothed-gradient shortcut); pixels whose gradient matrix is singular keep their
flow; a pixel stops iterating once its step is below 0.01 px. The 121 window
offsets are a batch, so each iteration is one grid_sample.
"""

import numpy as np


class LkFlow:
    """FlowSource: (prev_gray, curr_gray) uint8 -> (H, W, 2) float32 px."""

    def __init__(self, levels: int = 5, iterations: int = 5, win: int = 11, device: str | None = None):
        import torch

        self.levels, self.iterations, self.win = levels, iterations, win
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def __call__(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        import torch

        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device, torch.float32)[None, None]
        with torch.no_grad():
            return self._flow(t(prev), t(curr))[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    def _flow(self, prev, curr):
        import torch
        import torch.nn.functional as F

        P, C = [prev], [curr]
        for _ in range(self.levels - 1):
            P.append(F.avg_pool2d(P[-1], 2))
            C.append(F.avg_pool2d(C[-1], 2))
        r, n = self.win // 2, self.win * self.win
        dev = prev.device
        kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=dev)
        offs = torch.tensor([(k % self.win - r, k // self.win - r) for k in range(n)], device=dev, dtype=torch.float32)
        flow = None
        for lvl in range(self.levels - 1, -1, -1):
            I, J = P[lvl], C[lvl]
            h, w = I.shape[-2:]
            flow = torch.zeros(1, 2, h, w, device=dev) if flow is None else 2.0 * F.interpolate(flow, size=(h, w), mode="nearest")
            ix = F.conv2d(F.pad(I, (1, 1, 0, 0), mode="replicate"), kx)
            iy = F.conv2d(F.pad(I, (0, 0, 1, 1), mode="replicate"), kx.transpose(-1, -2))
            win = lambda a: F.unfold(F.pad(a, (r, r, r, r), mode="replicate"), self.win).view(n, h, w)
            Iw, ixw, iyw = win(I), win(ix), win(iy)
            gxx, gxy, gyy = (ixw * ixw).sum(0), (ixw * iyw).sum(0), (iyw * iyw).sum(0)
            det = gxx * gyy - gxy * gxy
            active = det >= 1e-6                      # singular: keep the flow, as the C++ loop skips the pixel
            det = torch.where(active, det, torch.ones_like(det))
            ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                    torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
            Jb = J.expand(n, 1, h, w)
            for _ in range(self.iterations):
                gx = (xs[None] + offs[:, 0, None, None] + flow[0, 0][None]) / (w - 1) * 2 - 1
                gy = (ys[None] + offs[:, 1, None, None] + flow[0, 1][None]) / (h - 1) * 2 - 1
                Jw = F.grid_sample(Jb, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
                e = Iw - Jw
                bx, by = (e * ixw).sum(0), (e * iyw).sum(0)
                dx, dy = (gyy * bx - gxy * by) / det, (gxx * by - gxy * bx) / det
                dx, dy = torch.where(active, dx, 0.0), torch.where(active, dy, 0.0)
                flow = flow + torch.stack([dx, dy])[None]
                active = active & (dx * dx + dy * dy >= 1e-4)   # converged pixels stop, as in the C++ loop
        return flow
