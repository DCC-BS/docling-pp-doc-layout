"""torchvision's CPU uint8 bicubic resize (antialias off), bit for bit, on the GPU.

The HF processor of PP-DocLayout-V3 resizes each page with ``tvF.resize(uint8, BICUBIC, antialias=False)``.
On the CPU that runs PyTorch's integer kernel (UpSampleKernelAVXAntialias.h, from Pillow-SIMD): per output
pixel four taps with float bicubic weights (a = -0.75), taps beyond the border clamped to the border pixel
and their weights summed, weights normalised and quantised to int16 with a shared precision, a horizontal
pass then a vertical pass, each ``clamp((sum + 2^(p-1)) >> p, 0, 255)``. This module does the same integer
math with int32 tensors on the GPU. :func:`exact_on_gpu` compares it once per size with the CPU kernel.
"""

from __future__ import annotations

import threading

import torch
from torchvision.transforms.v2 import functional as tvF  # noqa: N812

_A = -0.75  # torch's cubic coefficient without antialias
_SUPPORT = 2.0  # bicubic: two taps on each side
_MAX_PRECISION = 22  # torch's cap on the weights' fixed-point bits
_lock = threading.Lock()
_taps: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, int]] = {}
_checked: dict[tuple[int, int, int, int], bool] = {}


def _cubic(x: float) -> float:
    x = abs(x)
    if x < 1.0:
        return ((_A + 2) * x - (_A + 3)) * x * x + 1
    if x < _SUPPORT:
        return ((_A * x - 5 * _A) * x + 8 * _A) * x - 4 * _A
    return 0.0


def _weights(in_size: int, out_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    key = (in_size, out_size, str(device))
    with _lock:
        if key in _taps:
            return _taps[key]
    scale, support, interp = in_size / out_size, _SUPPORT, 4
    rows_idx, rows_w, wmax = [], [], 0.0
    for i in range(out_size):
        center = scale * (i + 0.5)
        x = center - support + 0.5
        xmin = int(x) if x >= 0 or float(int(x)) == x else int(x) - 1  # floor
        acc: dict[int, float] = {}
        for j in range(interp):
            k = min(max(xmin + j, 0), in_size - 1)
            acc[k] = acc.get(k, 0.0) + _cubic(j + xmin - center + 0.5)
        tot = sum(acc.values())
        keys = sorted(acc)
        w = [acc[k] / tot for k in keys] if tot != 0.0 else [acc[k] for k in keys]
        wmax = max(wmax, *w)
        rows_idx.append(keys + [keys[-1]] * (interp - len(keys)))
        rows_w.append(w + [0.0] * (interp - len(keys)))
    prec = 0
    while prec < _MAX_PRECISION and int(0.5 + wmax * (1 << (prec + 1))) < (1 << 15):
        prec += 1
    q = [
        [int(-0.5 + v * (1 << prec)) if v * (1 << prec) < 0 else int(0.5 + v * (1 << prec)) for v in row]
        for row in rows_w
    ]
    entry = (torch.tensor(rows_idx, device=device), torch.tensor(q, dtype=torch.int32, device=device), prec)
    with _lock:
        _taps[key] = entry
    return entry


def _pass(x: torch.Tensor, in_size: int, out_size: int, dim: int) -> torch.Tensor:
    idx, w, prec = _weights(in_size, out_size, x.device)
    out = x.index_select(dim, idx[:, 0]) * (w[:, 0] if dim == -1 else w[:, 0, None])
    for k in range(1, idx.shape[1]):
        wk = w[:, k] if dim == -1 else w[:, k, None]
        out.add_(x.index_select(dim, idx[:, k]) * wk)
    return out.add_(1 << (prec - 1)).bitwise_right_shift_(prec).clamp_(0, 255)


def resize(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """uint8 (..., C, H, W) -> uint8 (..., C, out_h, out_w), as tvF.resize(BICUBIC, antialias=False) on the CPU."""
    h, w = x.shape[-2:]
    y = x.to(torch.int32)
    if w != out_w:
        y = _pass(y, w, out_w, -1)
    if h != out_h:
        y = _pass(y, h, out_h, -2)
    return y.to(torch.uint8)


def exact_on_gpu(chw_cpu: torch.Tensor, out_h: int, out_w: int, device: torch.device | str) -> bool:
    """Once per input size: whether :func:`resize` on ``device`` matches the CPU kernel on this image."""
    key = (int(chw_cpu.shape[-2]), int(chw_cpu.shape[-1]), out_h, out_w)
    with _lock:
        if key in _checked:
            return _checked[key]
    cpu = tvF.resize(chw_cpu.unsqueeze(0), [out_h, out_w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=False)
    # random pixels as well: a page is mostly white and would hide differences
    noise = torch.randint(0, 256, chw_cpu.shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    cpu_n = tvF.resize(noise.unsqueeze(0), [out_h, out_w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=False)
    ok = torch.equal(resize(chw_cpu.unsqueeze(0).to(device), out_h, out_w).cpu(), cpu) and torch.equal(
        resize(noise.unsqueeze(0).to(device), out_h, out_w).cpu(), cpu_n
    )
    with _lock:
        _checked[key] = ok
    return ok
