"""ColorFit: per-pixel sRGB color-correction model.

Loads `structured_sandwich_v{1,2,3,4}` safetensors fits from the
pv_hina/upscale_calibration project (see FIT_FORMAT.md there for the spec)
and applies them on GPU as a per-pixel transform of the upscaled sfi_tensor.

Inserted in the upscale chain between the ESRGAN-style model loop and the
post-upscale EWA Lanczos3 resize (modules/upscaler_utils.py). Operates on
the float sRGB sfi_tensor, so the correction lands BEFORE the sRGB→linear
conversion and BEFORE any u8 quantization. No double-quantization, no extra
S3 round-trip.

Models live under `<models_path>/ColorFit/<name>.safetensors`. Names are
the file basename (with or without the `.safetensors` suffix). Loaded
models are cached by absolute path so repeated calls in the same process
reuse the parsed tensors and precomputed knot heights / Hermite slopes.

API surface:
  colorfit_model_dir()             -> directory path
  list_colorfit_models()           -> list of available basenames
  apply_colorfit_to_pil(img, name) -> PIL with sfi_tensor patched in place
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from modules import devices, shared


def colorfit_model_dir() -> str:
    return os.path.join(shared.models_path, "ColorFit")


def list_colorfit_models() -> list[str]:
    d = colorfit_model_dir()
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            out.append(fn[: -len(".safetensors")])
    return out


# ── Reconstruction helpers (PyTorch, autograd-free; run once per model load) ──

def _build_knot_y(theta: torch.Tensor,
                  start: Optional[torch.Tensor] = None,
                  total: Optional[torch.Tensor] = None) -> torch.Tensor:
    """theta: (C, K). Returns (C, K+1) monotone knot heights.

    Mirrors the spec's reconstruction:
      inc = softplus(theta); cum = cumsum(inc); ratios = [0, cum / cum[-1]];
      v1/v2 anchored: knot_y = ratios.
      v2/v3/v4 free-endpoint: knot_y = start + ratios * softplus(total).
    """
    inc = F.softplus(theta)
    cum = inc.cumsum(dim=1)
    ratios_inner = cum / cum[:, -1:].clamp(min=1e-9)
    zero = torch.zeros(theta.shape[0], 1, device=theta.device, dtype=theta.dtype)
    ratios = torch.cat([zero, ratios_inner], dim=1)
    if start is None:
        return ratios
    s = start.unsqueeze(1)
    t = F.softplus(total).unsqueeze(1)
    return s + ratios * t


def _fritsch_carlson_slopes(knot_y: torch.Tensor, K: int) -> torch.Tensor:
    """Monotone cubic Hermite slopes from knot heights. knot_y: (C, K+1).
    Uses uniform knot spacing dx = 1/K. Returns (C, K+1).
    """
    dx = 1.0 / K
    d = (knot_y[:, 1:] - knot_y[:, :-1]) / dx           # (C, K), secants
    d_pos = d.clamp(min=0.0)
    m_inner = 0.5 * (d_pos[:, :-1] + d_pos[:, 1:])      # (C, K-1)
    m = torch.cat([d_pos[:, :1], m_inner, d_pos[:, -1:]], dim=1)
    m_max_inner = 3.0 * torch.minimum(d_pos[:, :-1], d_pos[:, 1:])
    m_max = torch.cat([3.0 * d_pos[:, :1], m_max_inner, 3.0 * d_pos[:, -1:]], dim=1)
    return torch.minimum(m.clamp(min=0.0), m_max)


def _normalize_matrix(m_raw: torch.Tensor) -> torch.Tensor:
    rs = m_raw.sum(dim=1, keepdim=True).clamp(min=1e-3)
    return m_raw / rs


# ── Loaded model ─────────────────────────────────────────────────────────────

class ColorFitModel:
    """One loaded colorfit model. Reconstructs knot_y / slopes / normalized
    matrices once at load time; per-pixel apply is fast tensor ops on GPU.

    Versions handled:
      v1 — linear, all curves anchored.
      v2 — linear, f{N+1} free endpoints (always; via .start_theta/.total_theta).
      v3 — cubic Hermite, f{N+1} free endpoints optional (tensor presence).
      v4 — extensible: n_matrices and interpolation in metadata.config;
           free-endpoint extras optional (tensor presence).
    """

    SUPPORTED = {
        "structured_sandwich_v1",
        "structured_sandwich_v2",
        "structured_sandwich_v3",
        "structured_sandwich_v4",
    }

    def __init__(self, path: str, device: Optional[torch.device] = None):
        if device is None:
            device = devices.get_optimal_device()
        self.path = path
        self.device = device

        with safe_open(path, framework="pt", device=str(device)) as f:
            metadata = f.metadata() or {}
            model_id = metadata.get("model", "structured_sandwich_v1")
            if model_id not in self.SUPPORTED:
                raise ValueError(
                    f"Unsupported colorfit model '{model_id}' in {path}; "
                    f"supported: {sorted(self.SUPPORTED)}"
                )
            cfg = {}
            if "config" in metadata:
                try:
                    cfg = json.loads(metadata["config"])
                except Exception:
                    cfg = {}
            self.model_id = model_id
            self.metadata = dict(metadata)
            self.config = cfg

            # Stage-count axis. v1/v2/v3 are pinned at n_matrices=2 in the
            # spec; v4 puts it in metadata (default to 2 if missing for
            # legacy v4 files).
            if model_id == "structured_sandwich_v4":
                self.n_matrices = int(cfg.get("n_matrices", 2))
            else:
                self.n_matrices = 2

            # Interpolation. v3 is cubic; v4 reads from metadata; v1/v2 linear.
            if model_id == "structured_sandwich_v3":
                self.interpolation = "cubic"
            elif model_id == "structured_sandwich_v4":
                self.interpolation = cfg.get("interpolation", "linear")
                if self.interpolation not in ("linear", "cubic"):
                    raise ValueError(
                        f"Unknown interpolation '{self.interpolation}' in {path}"
                    )
            else:
                self.interpolation = "linear"

            tensors = {k: f.get_tensor(k) for k in f.keys()}

        n_curves = self.n_matrices + 1
        last_name = f"f{n_curves}"

        # K from f1.knot_x (asserted uniform).
        kx0 = tensors["f1.knot_x"].to(device)
        if kx0.dim() != 1:
            raise ValueError(f"{path}: f1.knot_x must be 1D")
        K = kx0.numel() - 1
        expected = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=kx0.dtype)
        if not torch.allclose(kx0, expected, atol=1e-5):
            raise ValueError(f"{path}: knot_x must be uniform on [0, 1]")
        for i in range(2, n_curves + 1):
            kxi = tensors[f"f{i}.knot_x"].to(device)
            if kxi.numel() - 1 != K:
                raise ValueError(f"{path}: K mismatch at f{i}.knot_x")
        self.K = K

        # f{N+1} free-endpoint extras.
        free_required = model_id == "structured_sandwich_v2"
        free_present = (f"{last_name}.start_theta" in tensors
                        and f"{last_name}.total_theta" in tensors)
        if free_required and not free_present:
            raise ValueError(
                f"{path}: {model_id} requires {last_name}.start_theta and "
                f"{last_name}.total_theta"
            )
        self.has_free_last = free_required or free_present

        # Precompute knot heights and (if cubic) Hermite slopes per curve.
        self.knot_ys: list[torch.Tensor] = []
        self.slopes: list[Optional[torch.Tensor]] = []
        for i in range(1, n_curves + 1):
            theta = tensors[f"f{i}.theta"].to(device, dtype=torch.float32)
            is_last = (i == n_curves)
            start = (tensors[f"{last_name}.start_theta"].to(device, dtype=torch.float32)
                     if is_last and self.has_free_last else None)
            total = (tensors[f"{last_name}.total_theta"].to(device, dtype=torch.float32)
                     if is_last and self.has_free_last else None)
            y = _build_knot_y(theta, start, total)
            self.knot_ys.append(y)
            self.slopes.append(_fritsch_carlson_slopes(y, K)
                               if self.interpolation == "cubic" else None)

        # Normalized row-stochastic matrices.
        self.matrices: list[torch.Tensor] = []
        for i in range(1, self.n_matrices + 1):
            m_raw = tensors[f"M{i}.M_raw"].to(device, dtype=torch.float32)
            self.matrices.append(_normalize_matrix(m_raw))

    # Curve apply, fully vectorized across channel & pixel. x: (N, 3).
    def _apply_curve(self, x: torch.Tensor, knot_y: torch.Tensor,
                     slopes: Optional[torch.Tensor]) -> torch.Tensor:
        K = self.K
        dtype = x.dtype
        ky = knot_y.to(dtype=dtype)                              # (3, K+1)
        # Per-channel segment index via uniform-knot lookup.
        seg_f = x * K                                            # (N, 3)
        seg = seg_f.floor().clamp(0, K - 1).long()
        t = seg_f - seg.to(dtype=dtype)                          # (N, 3)
        # Gather y0, y1 per channel.
        # ky[c, seg[n, c]]: build via per-channel advanced indexing.
        c_idx = torch.arange(3, device=x.device).expand_as(seg)  # (N, 3)
        y0 = ky[c_idx, seg]
        y1 = ky[c_idx, seg + 1]
        if slopes is None:
            return y0 + t * (y1 - y0)
        sl = slopes.to(dtype=dtype)
        m0 = sl[c_idx, seg]     / K
        m1 = sl[c_idx, seg + 1] / K
        t2 = t * t
        t3 = t2 * t
        h00 =  2.0 * t3 - 3.0 * t2 + 1.0
        h10 =        t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 =        t3 -       t2
        return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1

    def apply_bchw(self, rgb_bchw: torch.Tensor) -> torch.Tensor:
        """Apply transform to a (B, 3, H, W) sRGB float tensor on its current
        device. Returns same shape/dtype/device. Alpha (if present) should be
        handled by the caller — this method only takes 3-channel RGB."""
        if rgb_bchw.dim() != 4 or rgb_bchw.shape[1] != 3:
            raise ValueError(
                f"colorfit expected (B, 3, H, W); got {tuple(rgb_bchw.shape)}"
            )
        B, _, H, W = rgb_bchw.shape
        x = rgb_bchw.permute(0, 2, 3, 1).reshape(-1, 3).contiguous()  # (N, 3)
        x = self._apply_curve(x, self.knot_ys[0], self.slopes[0])
        for i in range(self.n_matrices):
            M = self.matrices[i].to(dtype=x.dtype)
            x = x @ M.T
            x = self._apply_curve(x, self.knot_ys[i + 1], self.slopes[i + 1])
        return x.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()


# ── Cache + resolution ───────────────────────────────────────────────────────

_CACHE: dict[str, ColorFitModel] = {}


def _resolve_path(name: str) -> str:
    """Resolve a colorfit model name to an absolute path. Accepts basename
    with or without the .safetensors suffix, or an absolute path."""
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    if not name.endswith(".safetensors"):
        name = name + ".safetensors"
    p = os.path.join(colorfit_model_dir(), name)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"ColorFit model not found: {p}. "
            f"Available: {list_colorfit_models()}"
        )
    return p


def get_colorfit_model(name: Optional[str]) -> Optional[ColorFitModel]:
    """Resolve and load a colorfit model. Returns None for falsy / 'None'."""
    if not name or name == "None":
        return None
    path = _resolve_path(name)
    if path in _CACHE:
        return _CACHE[path]
    model = ColorFitModel(path)
    _CACHE[path] = model
    return model


# ── PIL bridge ───────────────────────────────────────────────────────────────

def apply_colorfit_to_pil(img: Image.Image, name: Optional[str]) -> Image.Image:
    """Apply a colorfit model to a PIL image carrying an `sfi_tensor`.

    The sfi_tensor is the float sRGB output of the upscale chain. We apply
    the colorfit on GPU and re-attach the corrected tensor; the PIL's u8
    array is also regenerated so a downstream consumer that ignores
    sfi_tensor gets the corrected color too.

    When `name` is empty/'None' or no sfi_tensor is attached, this is a no-op
    (with a warning log for the missing-sfi case).
    """
    model = get_colorfit_model(name)
    if model is None:
        return img
    sfi = getattr(img, "sfi_tensor", None)
    if sfi is None:
        print(
            f"[colorfit] no sfi_tensor on input; skipping colorfit '{name}' "
            f"(can't apply without the float sRGB tensor)",
            flush=True,
        )
        return img
    if not isinstance(sfi, torch.Tensor):
        sfi = torch.as_tensor(sfi)
    if sfi.ndim != 3:
        raise ValueError(f"sfi_tensor must be HWC 3D, got shape {tuple(sfi.shape)}")

    h, w, channels = sfi.shape
    device = sfi.device if sfi.device.type != "cpu" else devices.get_optimal_device()
    src = sfi.detach().to(dtype=torch.float32, device=device)

    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    print(
        f"[colorfit] apply {os.path.basename(model.path)} on {w}x{h} "
        f"({device}) start",
        flush=True,
    )

    bchw = src.permute(2, 0, 1).unsqueeze(0).contiguous()
    rgb_out = model.apply_bchw(bchw[:, :3])
    if channels == 4:
        rgb_out = torch.cat([rgb_out, bchw[:, 3:4]], dim=1)
    new_sfi = rgb_out.squeeze(0).permute(1, 2, 0).contiguous().detach()

    if on_cuda:
        torch.cuda.synchronize(device)
    print(
        f"[colorfit] apply {os.path.basename(model.path)} done in "
        f"{(time.perf_counter() - t0) * 1000.0:.0f}ms",
        flush=True,
    )

    arr_u8 = (new_sfi.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    mode = "RGB" if channels == 3 else "RGBA"
    new_pil = Image.fromarray(arr_u8, mode)
    new_pil.sfi_tensor = new_sfi
    return new_pil
