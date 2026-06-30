"""ColorFit: per-pixel sRGB color-correction model.

Loads `structured_sandwich_v{1,2,3,4,5,6}` safetensors fits from the
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
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open

from modules import devices, shared


# Per-thread record of the most recently applied colorfit model. The extras
# API handler resets this before run_extras and reads it after — gives the
# caller (the bot) a positive confirmation that the chain actually ran the
# colorfit rather than silently dropping the request at some layer. The
# value is the basename (no `.safetensors`) so it matches what the request
# asked for. Thread-local because A1111 queue_locks the work but the field
# could outlive a request if a different thread handled it.
_apply_tracker = threading.local()


def _record_applied(basename: str) -> None:
    """Mark that a colorfit model ran on this thread. Called from
    ColorFitModel.apply_bchw — the closest point to the actual GPU work."""
    _apply_tracker.value = basename


def reset_applied_tracker() -> None:
    """Clear any prior apply record on this thread. Called by the API
    handler before run_extras so the response reflects only this request."""
    _apply_tracker.value = None


def get_applied() -> Optional[str]:
    """Read the most recently applied colorfit model name (basename) on
    this thread. Returns None if nothing applied. Called by the API
    handler after run_extras to populate the response."""
    return getattr(_apply_tracker, "value", None)


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


def _normalize_matrix(m_raw: torch.Tensor, gauge: str = "l1") -> torch.Tensor:
    """Per-row normalization. `gauge='l1'` (default) divides each row by its
    sum (row-stochastic — rows sum to 1, preserves the (1,1,1) eigenvector).
    `gauge='l2'` divides each row by its Euclidean norm (unit-norm rows,
    which combined with row orthogonality give proper rotations; inverse
    is the transpose). Both clamp to 1e-3 defensively against near-zero
    denominators."""
    if gauge == "l2":
        denom = m_raw.norm(dim=1, keepdim=True).clamp(min=1e-3)
    else:
        denom = m_raw.sum(dim=1, keepdim=True).clamp(min=1e-3)
    return m_raw / denom


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
        "structured_sandwich_v5",
        "structured_sandwich_v6",
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
            # spec; v4+ put it in metadata (default to 2 if missing).
            v45plus = ("structured_sandwich_v4", "structured_sandwich_v5",
                       "structured_sandwich_v6")
            if model_id in v45plus:
                self.n_matrices = int(cfg.get("n_matrices", 2))
            else:
                self.n_matrices = 2

            # Interpolation. v3 is cubic; v4+ read from metadata; v1/v2 linear.
            if model_id == "structured_sandwich_v3":
                self.interpolation = "cubic"
            elif model_id in v45plus:
                self.interpolation = cfg.get("interpolation", "linear")
                if self.interpolation not in ("linear", "cubic"):
                    raise ValueError(
                        f"Unknown interpolation '{self.interpolation}' in {path}"
                    )
            else:
                self.interpolation = "linear"

            tensors = {k: f.get_tensor(k) for k in f.keys()}

        if model_id == "structured_sandwich_v5":
            self._init_v5(path, device, tensors)
            return
        if model_id == "structured_sandwich_v6":
            self._init_v6(path, device, tensors)
            return

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

        # Sanity-check the computed buffers. NaN/inf at this stage means
        # something went wrong reconstructing the model (corrupt safetensors
        # file, extreme theta values, etc.) and the apply path would either
        # NaN-cascade into the output safetensors (cra dither collapses) or
        # produce out-of-bounds indices when seg = NaN.floor().long() (async
        # CUDA device-side assert). Fail fast at load time with a clear
        # message instead.
        def _finite(t, name):
            if not torch.isfinite(t).all():
                raise ValueError(
                    f"ColorFit model {path}: {name} contains NaN or inf "
                    f"after reconstruction. The file is corrupt or has "
                    f"extreme parameter values."
                )
        for i, ky in enumerate(self.knot_ys, start=1):
            _finite(ky, f"f{i} knot_y")
        for i, sl in enumerate(self.slopes, start=1):
            if sl is not None:
                _finite(sl, f"f{i} slopes")
        for i, m in enumerate(self.matrices, start=1):
            _finite(m, f"M{i} normalized")

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

    @property
    def basename(self) -> str:
        """Basename without the .safetensors suffix — what callers refer to
        the model by, and what the worker reports back to the bot."""
        return os.path.splitext(os.path.basename(self.path))[0]

    def apply_bchw(self, rgb_bchw: torch.Tensor) -> torch.Tensor:
        """Apply transform to a (B, 3, H, W) sRGB float tensor on its current
        device. Returns same shape/dtype/device, clamped to [0, 1]. Alpha
        (if present) should be handled by the caller — this method only
        takes 3-channel RGB.

        The output clamp guards against cubic Hermite extrapolation off the
        end segments amplifying tiny input overshoot (the EWA Lanczos pass
        upstream can produce values slightly outside [0, 1] from kernel
        lobes), and against free-endpoint variants (v2/v4 with
        free-output-endpoints) producing values outside [0, 1] structurally.
        Training data was sRGB in [0, 1]; anything outside is undefined per
        the calibration."""
        if rgb_bchw.dim() != 4 or rgb_bchw.shape[1] != 3:
            raise ValueError(
                f"colorfit expected (B, 3, H, W); got {tuple(rgb_bchw.shape)}"
            )
        B, _, H, W = rgb_bchw.shape
        x = rgb_bchw.permute(0, 2, 3, 1).reshape(-1, 3).contiguous()  # (N, 3)
        # Defensive: NaN/inf in the upstream sRGB float (e.g. from a
        # numerical edge case in the EWA Lanczos resize or
        # _linear_to_srgb) would propagate to .long() index conversion in
        # _apply_curve and produce out-of-bounds indices → asynchronous
        # CUDA device-side assert that surfaces in a later torch_gc().
        # AND finite values outside [0, 1] are equally lethal: composed
        # cubic Hermite extrapolation off the leftmost/rightmost segment
        # amplifies the overshoot exponentially across stages and overflows
        # float32 to NaN within a handful of curves. The calibration was
        # trained exclusively on sRGB inputs in [0, 1]; anything outside is
        # undefined and downstream by design.
        #
        # The EWA Lanczos pass upstream produces values slightly outside
        # [0, 1] because Lanczos has negative kernel lobes that, in linear
        # space, produce small negatives at high-contrast edges. The
        # subsequent _linear_to_srgb amplifies a -0.02 linear into a
        # -0.30 sRGB via the steep K≈12.92 slope of the linear segment.
        # The clamp here is mandatory, not optional — see FIT_FORMAT.md
        # "Input clamp" for the full rationale and observed numbers.
        x = x.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if self.model_id == "structured_sandwich_v5":
            x = self._apply_chain_v5(x)
        elif self.model_id == "structured_sandwich_v6":
            x = self._apply_chain_v6(x)
        else:
            x = self._apply_curve(x, self.knot_ys[0], self.slopes[0])
            for i in range(self.n_matrices):
                M = self.matrices[i].to(dtype=x.dtype)
                x = x @ M.T
                x = self._apply_curve(x, self.knot_ys[i + 1], self.slopes[i + 1])
        out = x.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous().clamp_(0.0, 1.0)
        _record_applied(self.basename)
        return out

    # ── v5: dynamic per-channel symmetric domains ─────────────────────────────
    #
    # v5 curves operate on [-S_c, +S_c] per channel. S is propagated forward
    # from the matrices (S_next[r] = sum_c |M[r,c]| * S_prev[c]). Curves are
    # parameterized internally in normalized [0, 1] space, then rescaled per
    # channel at apply time using the channel's S.

    def _init_v5(self, path: str, device: torch.device,
                 tensors: dict) -> None:
        n_matrices = self.n_matrices
        n_curves = n_matrices + 1
        last_name = f"f{n_curves}"
        # Gauge: l1 (row-stochastic) or l2 (unit-norm rows). Default l1 so
        # any v5 fit from before the gauge field existed still loads with
        # the original row-stochastic normalization.
        self.gauge = self.config.get("gauge", "l1")
        if self.gauge not in ("l1", "l2"):
            raise ValueError(
                f"{path}: unknown gauge '{self.gauge}' (expected 'l1' or 'l2')"
            )

        # K from f1.theta (no knot_x in v5).
        theta1 = tensors["f1.theta"]
        if theta1.dim() != 2 or theta1.shape[0] != 3:
            raise ValueError(f"{path}: f1.theta must be (3, K)")
        K = int(theta1.shape[1])
        for i in range(2, n_curves + 1):
            ti = tensors.get(f"f{i}.theta")
            if ti is None:
                raise ValueError(
                    f"{path}: structured_sandwich_v5 missing f{i}.theta"
                )
            if ti.shape != theta1.shape:
                raise ValueError(f"{path}: f{i}.theta shape mismatch")
        self.K = K

        # Normalized matrices (entries may be any sign). Gauge picks the
        # per-row normalization.
        self.matrices: list[torch.Tensor] = []
        for i in range(1, n_matrices + 1):
            m_raw = tensors[f"M{i}.M_raw"].to(device, dtype=torch.float32)
            self.matrices.append(_normalize_matrix(m_raw, gauge=self.gauge))

        # Forward S propagation. S_per_curve[i] is the bound for curve f_{i+1}
        # (i.e. S_per_curve[0] = f1 bound = (1, 1, 1); S_per_curve[1] = f2
        # bound = |M1| @ (1, 1, 1); etc.).
        S = torch.ones(3, device=device, dtype=torch.float32)
        self.S_per_curve: list[torch.Tensor] = [S]
        for M in self.matrices:
            S = M.abs() @ S
            self.S_per_curve.append(S)

        # Last curve free-endpoint extras (optional per tensor presence).
        free_present = (f"{last_name}.start_theta" in tensors
                        and f"{last_name}.total_theta" in tensors)
        self.has_free_last = free_present
        if free_present:
            self.last_start = tensors[f"{last_name}.start_theta"].to(
                device, dtype=torch.float32)
            self.last_total = F.softplus(
                tensors[f"{last_name}.total_theta"].to(device, dtype=torch.float32))
        else:
            self.last_start = None
            self.last_total = None

        # Per-curve normalized knot heights and (cubic) slopes. Heights are
        # ratios in [0, 1] regardless of S; the rescaling happens at apply.
        self.knot_ys_norm: list[torch.Tensor] = []
        self.slopes_norm: list[Optional[torch.Tensor]] = []
        for i in range(1, n_curves + 1):
            theta = tensors[f"f{i}.theta"].to(device, dtype=torch.float32)
            # start=None, total=None → returns ratios in [0, 1] anchored at 0/1.
            y_norm = _build_knot_y(theta, start=None, total=None)
            self.knot_ys_norm.append(y_norm)
            self.slopes_norm.append(
                _fritsch_carlson_slopes(y_norm, K)
                if self.interpolation == "cubic" else None)

        # Sanity-check the computed buffers.
        def _finite(t, name):
            if not torch.isfinite(t).all():
                raise ValueError(
                    f"ColorFit model {path}: {name} contains NaN or inf "
                    f"after reconstruction."
                )
        for i, ky in enumerate(self.knot_ys_norm, start=1):
            _finite(ky, f"f{i} knot_y_norm")
        for i, sl in enumerate(self.slopes_norm, start=1):
            if sl is not None:
                _finite(sl, f"f{i} slopes_norm")
        for i, m in enumerate(self.matrices, start=1):
            _finite(m, f"M{i} normalized")

    def _apply_curve_v5(self, x: torch.Tensor, S: torch.Tensor,
                        knot_y_norm: torch.Tensor,
                        slopes_norm: Optional[torch.Tensor],
                        is_last: bool) -> torch.Tensor:
        """v5 per-channel curve eval: x ∈ [-S_c, +S_c] → normalize to [0, 1],
        cubic/linear interpolate in normalized space, rescale to output.
        x: (N, 3). S: (3,). knot_y_norm: (3, K+1) ratios in [0, 1].
        """
        K = self.K
        dtype = x.dtype
        ky = knot_y_norm.to(dtype=dtype)
        S_row = S.to(dtype=dtype).unsqueeze(0)        # (1, 3)
        # Normalize input to [0, 1] for interior evaluation.
        u = (x + S_row) / (2.0 * S_row)               # (N, 3)
        seg_f = u * K
        seg = seg_f.floor().clamp(0, K - 1).long()
        t = seg_f - seg.to(dtype=dtype)
        c_idx = torch.arange(3, device=x.device).expand_as(seg)
        y0 = ky[c_idx, seg]
        y1 = ky[c_idx, seg + 1]
        if slopes_norm is None:
            y_norm = y0 + t * (y1 - y0)
        else:
            sl = slopes_norm.to(dtype=dtype)
            m0 = sl[c_idx, seg]     / K
            m1 = sl[c_idx, seg + 1] / K
            t2 = t * t
            t3 = t2 * t
            h00 =  2.0 * t3 - 3.0 * t2 + 1.0
            h10 =        t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 =        t3 -       t2
            y_norm = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
        # Rescale to actual coords.
        if is_last and self.has_free_last:
            start = self.last_start.to(dtype=dtype).unsqueeze(0)  # (1, 3)
            total = self.last_total.to(dtype=dtype).unsqueeze(0)  # (1, 3)
            return start + y_norm * total
        return (2.0 * y_norm - 1.0) * S_row

    def _apply_chain_v5(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n_matrices
        x = self._apply_curve_v5(x, self.S_per_curve[0], self.knot_ys_norm[0],
                                 self.slopes_norm[0], is_last=(n == 0))
        for i in range(n):
            M = self.matrices[i].to(dtype=x.dtype)
            x = x @ M.T
            is_last = (i == n - 1)
            x = self._apply_curve_v5(x, self.S_per_curve[i + 1],
                                     self.knot_ys_norm[i + 1],
                                     self.slopes_norm[i + 1], is_last=is_last)
        return x

    # ── v6: asymmetric per-channel domains + free endpoints on every curve ────
    #
    # v6 generalizes v5 in two coupled ways:
    # - Every curve has its own free start_theta / total_theta per channel
    #   (v5 had this only for the last curve). Output range per channel is
    #   [start_theta, start_theta + softplus(total_theta)].
    # - Bound propagation is per-channel asymmetric [L, H] instead of
    #   symmetric [-S, +S]. Through a matrix the bounds split into positive
    #   and negative parts: M_pos @ L + M_neg @ H gives the new lower,
    #   M_pos @ H + M_neg @ L gives the new upper.
    # - When `interior_normalized` is True in metadata, interior curves
    #   (all except the last) apply a scalar normalization at apply time:
    #   `scale = (|mean(start_eff)| + |mean(end_eff)|) / 2` then divide
    #   both start and end by scale. Fixes the uniform-scale gauge
    #   freedom; matches the training-time projection. The last curve
    #   stays free regardless (its start/total directly target sRGB).

    def _init_v6(self, path: str, device: torch.device,
                 tensors: dict) -> None:
        n_matrices = self.n_matrices
        n_curves = n_matrices + 1
        self.gauge = self.config.get("gauge", "l1")
        if self.gauge not in ("l1", "l2"):
            raise ValueError(
                f"{path}: unknown gauge '{self.gauge}' (expected 'l1' or 'l2')"
            )
        self.interior_normalized = bool(
            self.config.get("interior_normalized", False))

        theta1 = tensors["f1.theta"]
        if theta1.dim() != 2 or theta1.shape[0] != 3:
            raise ValueError(f"{path}: f1.theta must be (3, K)")
        K = int(theta1.shape[1])
        for i in range(2, n_curves + 1):
            ti = tensors.get(f"f{i}.theta")
            if ti is None:
                raise ValueError(
                    f"{path}: structured_sandwich_v6 missing f{i}.theta"
                )
            if ti.shape != theta1.shape:
                raise ValueError(f"{path}: f{i}.theta shape mismatch")
        self.K = K

        # v6 requires start_theta and total_theta on every curve.
        self.curve_start: list[torch.Tensor] = []
        self.curve_total: list[torch.Tensor] = []
        for i in range(1, n_curves + 1):
            s_key = f"f{i}.start_theta"
            t_key = f"f{i}.total_theta"
            if s_key not in tensors or t_key not in tensors:
                raise ValueError(
                    f"{path}: structured_sandwich_v6 requires {s_key} and "
                    f"{t_key} on every curve"
                )
            raw_start = tensors[s_key].to(device, dtype=torch.float32)
            raw_total = F.softplus(
                tensors[t_key].to(device, dtype=torch.float32))
            raw_end = raw_start + raw_total
            # Interior curves with interior_normalized: apply scalar
            # projection so (|mean(start)| + |mean(end)|) / 2 == 1.
            is_last = (i == n_curves)
            if self.interior_normalized and not is_last:
                scale = ((raw_start.mean().abs() + raw_end.mean().abs())
                         * 0.5).clamp(min=1e-6)
                eff_start = raw_start / scale
                eff_end = raw_end / scale
            else:
                eff_start = raw_start
                eff_end = raw_end
            self.curve_start.append(eff_start)
            self.curve_total.append(eff_end - eff_start)

        self.matrices: list[torch.Tensor] = []
        for i in range(1, n_matrices + 1):
            m_raw = tensors[f"M{i}.M_raw"].to(device, dtype=torch.float32)
            self.matrices.append(_normalize_matrix(m_raw, gauge=self.gauge))

        # Forward asymmetric bound propagation. bounds_per_curve[i] is
        # (L, H) input bounds for curve f_{i+1}. bounds_per_curve[0] is f1's
        # input (sRGB [0, 1]).
        L = torch.zeros(3, device=device, dtype=torch.float32)
        H = torch.ones(3, device=device, dtype=torch.float32)
        self.bounds_per_curve: list[tuple] = [(L, H)]
        for i, M in enumerate(self.matrices):
            # After this curve, output range = its effective [start, start+total].
            curve_L = self.curve_start[i]
            curve_H = curve_L + self.curve_total[i]
            # Through the matrix, asymmetric propagation.
            M_pos = M.clamp(min=0.0)
            M_neg = M.clamp(max=0.0)
            L = M_pos @ curve_L + M_neg @ curve_H
            H = M_pos @ curve_H + M_neg @ curve_L
            self.bounds_per_curve.append((L, H))

        # Per-curve normalized knot heights and (cubic) slopes.
        self.knot_ys_norm: list[torch.Tensor] = []
        self.slopes_norm: list[Optional[torch.Tensor]] = []
        for i in range(1, n_curves + 1):
            theta = tensors[f"f{i}.theta"].to(device, dtype=torch.float32)
            y_norm = _build_knot_y(theta, start=None, total=None)
            self.knot_ys_norm.append(y_norm)
            self.slopes_norm.append(
                _fritsch_carlson_slopes(y_norm, K)
                if self.interpolation == "cubic" else None)

        # Sanity-check.
        def _finite(t, name):
            if not torch.isfinite(t).all():
                raise ValueError(
                    f"ColorFit model {path}: {name} contains NaN or inf "
                    f"after reconstruction."
                )
        for i, ky in enumerate(self.knot_ys_norm, start=1):
            _finite(ky, f"f{i} knot_y_norm")
        for i, sl in enumerate(self.slopes_norm, start=1):
            if sl is not None:
                _finite(sl, f"f{i} slopes_norm")
        for i, m in enumerate(self.matrices, start=1):
            _finite(m, f"M{i} normalized")
        for i, (L_b, H_b) in enumerate(self.bounds_per_curve):
            _finite(L_b, f"bounds[{i}].L")
            _finite(H_b, f"bounds[{i}].H")
        for i, s in enumerate(self.curve_start, start=1):
            _finite(s, f"f{i} curve_start")
        for i, t in enumerate(self.curve_total, start=1):
            _finite(t, f"f{i} curve_total")

    def _apply_curve_v6(self, x: torch.Tensor,
                        L_in: torch.Tensor, H_in: torch.Tensor,
                        start_out: torch.Tensor, total_out: torch.Tensor,
                        knot_y_norm: torch.Tensor,
                        slopes_norm: Optional[torch.Tensor]) -> torch.Tensor:
        """v6 per-channel curve: x ∈ [L_in, H_in] → normalize to [0, 1],
        interpolate, rescale to [start_out, start_out + total_out].
        L_in, H_in, start_out, total_out: (3,). knot_y_norm: (3, K+1)."""
        K = self.K
        dtype = x.dtype
        L_row = L_in.to(dtype=dtype).unsqueeze(0)
        H_row = H_in.to(dtype=dtype).unsqueeze(0)
        span_in = (H_row - L_row).clamp(min=1e-6)
        u = (x - L_row) / span_in
        ky = knot_y_norm.to(dtype=dtype)
        seg_f = u * K
        seg = seg_f.floor().clamp(0, K - 1).long()
        t = seg_f - seg.to(dtype=dtype)
        c_idx = torch.arange(3, device=x.device).expand_as(seg)
        y0 = ky[c_idx, seg]
        y1 = ky[c_idx, seg + 1]
        if slopes_norm is None:
            y_norm = y0 + t * (y1 - y0)
        else:
            sl = slopes_norm.to(dtype=dtype)
            m0 = sl[c_idx, seg]     / K
            m1 = sl[c_idx, seg + 1] / K
            t2 = t * t
            t3 = t2 * t
            h00 =  2.0 * t3 - 3.0 * t2 + 1.0
            h10 =        t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 =        t3 -       t2
            y_norm = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
        start = start_out.to(dtype=dtype).unsqueeze(0)
        total = total_out.to(dtype=dtype).unsqueeze(0)
        return start + y_norm * total

    def _apply_chain_v6(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n_matrices
        L_in, H_in = self.bounds_per_curve[0]
        x = self._apply_curve_v6(x, L_in, H_in,
                                 self.curve_start[0], self.curve_total[0],
                                 self.knot_ys_norm[0], self.slopes_norm[0])
        for i in range(n):
            M = self.matrices[i].to(dtype=x.dtype)
            x = x @ M.T
            L_in, H_in = self.bounds_per_curve[i + 1]
            x = self._apply_curve_v6(x, L_in, H_in,
                                     self.curve_start[i + 1],
                                     self.curve_total[i + 1],
                                     self.knot_ys_norm[i + 1],
                                     self.slopes_norm[i + 1])
        return x


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
