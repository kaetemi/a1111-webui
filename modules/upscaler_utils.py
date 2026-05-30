import logging
import math
from typing import Callable

import numpy as np
import torch
import tqdm
from PIL import Image

from modules import devices, images, shared, torch_utils

logger = logging.getLogger(__name__)


# sRGB transfer function — high-precision constants from cra-web/COLORSPACES.md /
# port/src/colorspace_derived.rs. K and T are derived from γ=2.4 and a=0.055 by
# enforcing C¹ continuity; ICC profiles can't represent K exactly, so use the
# full-precision values for float math.
_SRGB_GAMMA = 2.4
_SRGB_OFFSET = 0.055
_SRGB_SCALE = 1.055
_SRGB_LINEAR_SLOPE = 12.923210180787855
_SRGB_THRESHOLD = 0.0030399346397784314          # linear-side junction
_SRGB_DECODE_THRESHOLD = 0.039285714285714285    # encoded-side junction (11/280)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    # sYCC extended-range, point-symmetric about 0
    ax = x.abs()
    linear = x / _SRGB_LINEAR_SLOPE
    power = torch.sign(x) * ((ax + _SRGB_OFFSET) / _SRGB_SCALE).pow(_SRGB_GAMMA)
    return torch.where(ax <= _SRGB_DECODE_THRESHOLD, linear, power)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    linear = x * _SRGB_LINEAR_SLOPE
    power = torch.sign(x) * (_SRGB_SCALE * ax.pow(1.0 / _SRGB_GAMMA) - _SRGB_OFFSET)
    return torch.where(ax <= _SRGB_THRESHOLD, linear, power)


def _lanczos3_weights_1d(src_len: int, dst_len: int, device: torch.device) -> torch.Tensor:
    """Dense (dst_len, src_len) Lanczos3 weight matrix matching cra-web's
    port/src/rescale/{kernels,separable}.rs: independent scaling, adaptive
    filter scale (max(src/dst, 1)), edge clipping with renormalization.
    Computed in float64 for exact normalization, returned in float32.
    """
    scale = src_len / dst_len
    filter_scale = max(scale, 1.0)

    dst_pos = torch.arange(dst_len, dtype=torch.float64) + 0.5
    src_pos = dst_pos * scale - 0.5                 # (dst_len,)
    si = torch.arange(src_len, dtype=torch.float64)  # (src_len,)
    d = (src_pos.unsqueeze(1) - si.unsqueeze(0)) / filter_scale  # (dst_len, src_len)

    abs_d = d.abs()
    pix = math.pi * d
    # safe sin(πx)/(πx) and sin(πx/3)/(πx/3); the abs<1e-12 path returns 1.0
    near_zero = abs_d < 1e-12
    safe_pix = torch.where(near_zero, torch.ones_like(pix), pix)
    sinc = torch.where(near_zero, torch.ones_like(pix), torch.sin(safe_pix) / safe_pix)
    safe_pix3 = safe_pix / 3.0
    win = torch.where(near_zero, torch.ones_like(pix), torch.sin(safe_pix3) / safe_pix3)
    w = sinc * win
    # outside the 3-lobe support
    w = torch.where(abs_d >= 3.0, torch.zeros_like(w), w)

    row_sum = w.sum(dim=1, keepdim=True)
    nonempty = row_sum.abs() > 1e-12
    w = torch.where(nonempty, w / row_sum.clamp(min=1e-30), torch.zeros_like(w))
    return w.to(dtype=torch.float32, device=device)


def _resize_2d_lanczos3_linear(
    bchw: torch.Tensor, dst_h: int, dst_w: int
) -> torch.Tensor:
    """Separable Lanczos3 resize for a (B, C, H, W) float tensor.

    Operates on whatever values are in the tensor (does not linearize). Callers
    are expected to convert to linear RGB beforehand.
    """
    _, _, h, w = bchw.shape
    wh = _lanczos3_weights_1d(w, dst_w, device=bchw.device)  # (dst_w, w)
    wv = _lanczos3_weights_1d(h, dst_h, device=bchw.device)  # (dst_h, h)
    # Horizontal: (B, C, H, W) @ (W, dst_w) -> (B, C, H, dst_w)
    out = bchw @ wh.t()
    # Vertical: pull H to last axis, multiply by Wv.T, swap back
    out = out.transpose(-2, -1) @ wv.t()  # (B, C, dst_w, dst_h)
    return out.transpose(-2, -1).contiguous()


def resize_preserving_float_gpu_linear(
    img: Image.Image, dest_w: int, dest_h: int
) -> Image.Image:
    """GPU-accelerated separable Lanczos3 resize performed in linear RGB.

    Matches cra-web's CLI `lanczos3` mode (port/src/rescale, separable path):
    sRGB → linear (high-precision transfer constants) → Lanczos3 with adaptive
    radius → linear → sRGB. The float result is reattached as `sfi_tensor` so
    downstream SFI saves and chained upscales keep float precision.

    Requires the input PIL to carry an `sfi_tensor` attribute (RGB HWC float in
    sRGB-encoded [0, 1] — i.e. ESRGAN-style model output). Without it we fall
    back to a plain PIL resize, which is sRGB-space and lossy but the only
    thing we can honestly do.
    """
    sfi = getattr(img, 'sfi_tensor', None)
    if sfi is None:
        pil_lanczos = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        return img.resize((dest_w, dest_h), resample=pil_lanczos)

    if not isinstance(sfi, torch.Tensor):
        sfi = torch.as_tensor(sfi)
    if sfi.ndim != 3:
        raise ValueError(f"sfi_tensor must be HWC 3D, got shape {tuple(sfi.shape)}")

    src_h, src_w, channels = sfi.shape
    if src_w == dest_w and src_h == dest_h:
        return img

    # If sfi_tensor is already on a usable accelerator (e.g. came from an
    # upstream GPU upscale), stay there. Otherwise move to the optimal device.
    if sfi.device.type == 'cpu':
        device = devices.get_optimal_device()
        src = sfi.detach().to(dtype=torch.float32, device=device)
    else:
        src = sfi.detach().to(dtype=torch.float32)

    bchw = src.permute(2, 0, 1).unsqueeze(0).contiguous()

    rgb_lin = _srgb_to_linear(bchw[:, :3])
    if channels == 4:
        rgb_lin = torch.cat([rgb_lin, bchw[:, 3:4]], dim=1)

    rescaled_lin = _resize_2d_lanczos3_linear(rgb_lin, dest_h, dest_w)

    rgb_out = _linear_to_srgb(rescaled_lin[:, :3])
    if channels == 4:
        rgb_out = torch.cat([rgb_out, rescaled_lin[:, 3:4]], dim=1)

    new_sfi = rgb_out.squeeze(0).permute(1, 2, 0).contiguous().detach()

    arr_u8 = (new_sfi.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    mode = "RGB" if channels == 3 else "RGBA"
    new_pil = Image.fromarray(arr_u8, mode)
    new_pil.sfi_tensor = new_sfi
    return new_pil


def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float CHW BGR tensor in [0, 1].

    If the PIL image carries an `sfi_tensor` attribute (RGB HWC or CHW float
    in [0, 1]), use it directly so the upscaler input chain stays in float —
    no PIL→uint8→float round-trip. The returned tensor inherits sfi_tensor's
    device, so a GPU-resident sfi_tensor from a prior upscale iteration stays
    on the GPU.
    """
    sfi = getattr(img, 'sfi_tensor', None)
    if sfi is not None:
        if not isinstance(sfi, torch.Tensor):
            sfi = torch.as_tensor(sfi)
        sfi = sfi.detach().to(dtype=torch.float32)
        if sfi.ndim != 3:
            raise ValueError(f"sfi_tensor must be 3D, got shape {tuple(sfi.shape)}")
        if sfi.shape[0] in (3, 4) and sfi.shape[-1] not in (3, 4):
            chw = sfi
        else:
            chw = sfi.permute(2, 0, 1).contiguous()
        rgb_chw = chw[:3]  # drop alpha if present; upscalers expect 3 channels
        bgr_chw = rgb_chw[[2, 1, 0]].contiguous()
        return bgr_chw

    arr = np.array(img.convert("RGB"))
    arr = arr[:, :, ::-1]  # flip RGB to BGR
    arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
    arr = np.ascontiguousarray(arr) / 255  # Rescale to [0, 1]
    return torch.from_numpy(arr)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a float CHW (or BCHW with batch=1) BGR tensor to a PIL Image.

    The PIL image is the standard uint8 view used by the rest of the codebase.
    A float copy of the same data (RGB HWC, clamped to [0, 1]) is attached as
    `sfi_tensor` so the upscaler chain can keep float precision end-to-end
    when a downstream consumer (SFI save, next upscale iteration, float-aware
    resize) wants it. The sfi_tensor stays on the source device — only the
    uint8 view is pulled to CPU.
    """
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"

    bgr_chw = tensor.float().detach().clamp(0, 1)

    arr = bgr_chw.cpu().numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    pil = Image.fromarray(arr, "RGB")
    pil.sfi_tensor = bgr_chw[[2, 1, 0]].permute(1, 2, 0).contiguous()
    return pil


def resize_preserving_float(img: Image.Image, dest_w: int, dest_h: int, resample) -> Image.Image:
    """PIL resize that also resamples the attached `sfi_tensor` losslessly.

    When the input PIL has an `sfi_tensor` attribute, each float channel is
    resampled independently through PIL's 'F' mode (which supports LANCZOS /
    BICUBIC / BILINEAR / NEAREST on float32 data). The resulting PIL view is
    built from the resampled float tensor, so no uint8 round-trip happens.

    If no sfi_tensor is attached, this falls back to a plain PIL resize, which
    is the only thing we can do — and the SFI save path will refuse it.
    """
    sfi = getattr(img, 'sfi_tensor', None)
    if sfi is None:
        return img.resize((dest_w, dest_h), resample=resample)

    if not isinstance(sfi, torch.Tensor):
        sfi = torch.as_tensor(sfi)
    arr = sfi.detach().cpu().to(torch.float32).numpy()  # HWC RGB float

    if arr.ndim != 3:
        raise ValueError(f"sfi_tensor must be HWC 3D, got shape {arr.shape}")

    channels = arr.shape[-1]
    out_channels = []
    for c in range(channels):
        chan = np.ascontiguousarray(arr[..., c], dtype=np.float32)
        pil_chan = Image.fromarray(chan, 'F')
        resized = pil_chan.resize((dest_w, dest_h), resample=resample)
        out_channels.append(np.array(resized, dtype=np.float32))
    out_arr = np.stack(out_channels, axis=-1)
    new_sfi = torch.from_numpy(np.ascontiguousarray(out_arr))

    arr_u8 = (new_sfi.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    mode = "RGB" if channels == 3 else "RGBA"
    new_pil = Image.fromarray(arr_u8, mode)
    new_pil.sfi_tensor = new_sfi
    return new_pil


def float_bgr_hwc_to_pil(tensor_bgr_hwc: torch.Tensor) -> Image.Image:
    """Wrap a float HWC BGR tensor into a PIL image with `sfi_tensor` attached.

    Used when the upscaler chain has built the full image in float HWC (via
    combine_grid_float) and we need to hand a PIL back to callers while
    preserving the float for SFI saves. The sfi_tensor stays on the source
    device; only the uint8 view is pulled to CPU.
    """
    bgr_hwc = tensor_bgr_hwc.float().detach().clamp(0, 1).contiguous()
    arr_u8 = (bgr_hwc.cpu().numpy() * 255.0).round().astype(np.uint8)
    arr_u8 = arr_u8[:, :, ::-1]  # BGR -> RGB
    pil = Image.fromarray(arr_u8, "RGB")
    pil.sfi_tensor = bgr_hwc[..., [2, 1, 0]].contiguous()  # BGR HWC -> RGB HWC
    return pil


def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """
    Upscale a given PIL image using the given model.
    """
    param = torch_utils.get_param(model)

    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)  # add batch dimension
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        with devices.without_autocast():
            return torch_bgr_to_pil_image(model(tensor))


def upscale_with_model(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
) -> Image.Image:
    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img)
        output = upscale_pil_patch(model, img)
        logger.debug("=> %s", output)
        return output

    # Float-tiled path: keep tiles as float tensors all the way through, so
    # per-tile uint8 quantization (the old combine_grid PIL paste path) is
    # eliminated and the final result can carry sfi_tensor for SFI saves.
    bgr_chw = pil_image_to_torch_bgr(img)             # CHW float BGR [0,1] on CPU
    bgr_hwc = bgr_chw.permute(1, 2, 0).contiguous()   # HWC for split_grid_float

    grid = images.split_grid_float(bgr_hwc, tile_size, tile_size, tile_overlap)

    param = torch_utils.get_param(model)
    scale_factor = None
    new_tiles = []

    with tqdm.tqdm(total=grid.tile_count, desc=desc, disable=not shared.opts.enable_upscale_progressbar) as p:
        for y, h, row in grid.tiles:
            new_row = []
            for x, w, tile_hwc in row:
                if shared.state.interrupted:
                    return img

                tile_bchw = tile_hwc.permute(2, 0, 1).unsqueeze(0).to(device=param.device, dtype=param.dtype)
                with torch.inference_mode():
                    with devices.without_autocast():
                        out_bchw = model(tile_bchw)

                out_hwc = out_bchw.squeeze(0).float().detach().clamp(0, 1).permute(1, 2, 0).contiguous()
                if scale_factor is None:
                    scale_factor = out_hwc.shape[1] // tile_hwc.shape[1]
                new_row.append([x * scale_factor, w * scale_factor, out_hwc])
                p.update(1)
            new_tiles.append([y * scale_factor, h * scale_factor, new_row])

    if scale_factor is None:
        return img

    new_grid = images.Grid(
        tiles=new_tiles,
        tile_w=grid.tile_w * scale_factor,
        tile_h=grid.tile_h * scale_factor,
        image_w=grid.image_w * scale_factor,
        image_h=grid.image_h * scale_factor,
        overlap=grid.overlap * scale_factor,
    )
    combined_bgr_hwc = images.combine_grid_float(new_grid)
    return float_bgr_hwc_to_pil(combined_bgr_hwc)


def tiled_upscale_2(
    img: torch.Tensor,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    device: torch.device,
    desc="Tiled upscale",
):
    # Alternative implementation of `upscale_with_model` originally used by
    # SwinIR and ScuNET.  It differs from `upscale_with_model` in that tiling and
    # weighting is done in PyTorch space, as opposed to `images.Grid` doing it in
    # Pillow space without weighting.

    b, c, h, w = img.size()
    tile_size = min(tile_size, h, w)

    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img.shape)
        return model(img)

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
    result = torch.zeros(
        b,
        c,
        h * scale,
        w * scale,
        device=device,
        dtype=img.dtype,
    )
    weights = torch.zeros_like(result)
    logger.debug("Upscaling %s to %s with tiles", img.shape, result.shape)
    with tqdm.tqdm(total=len(h_idx_list) * len(w_idx_list), desc=desc, disable=not shared.opts.enable_upscale_progressbar) as pbar:
        for h_idx in h_idx_list:
            if shared.state.interrupted or shared.state.skipped:
                break

            for w_idx in w_idx_list:
                if shared.state.interrupted or shared.state.skipped:
                    break

                # Only move this patch to the device if it's not already there.
                in_patch = img[
                    ...,
                    h_idx : h_idx + tile_size,
                    w_idx : w_idx + tile_size,
                ].to(device=device)

                out_patch = model(in_patch)

                result[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch)

                out_patch_mask = torch.ones_like(out_patch)

                weights[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch_mask)

                pbar.update(1)

    output = result.div_(weights)

    return output


def upscale_2(
    img: Image.Image,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    desc: str,
):
    """
    Convenience wrapper around `tiled_upscale_2` that handles PIL images.
    """
    param = torch_utils.get_param(model)
    tensor = pil_image_to_torch_bgr(img).to(dtype=param.dtype).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = tiled_upscale_2(
            tensor,
            model,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            scale=scale,
            desc=desc,
            device=param.device,
        )
    return torch_bgr_to_pil_image(output)
