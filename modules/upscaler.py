import math
import os
from abc import abstractmethod

import PIL
import torch
from PIL import Image

import modules.shared
from modules import devices, modelloader, shared

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)


class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers: list
    tile = True

    def __init__(self, create_dirs=False):
        self.mod_pad_h = None
        self.tile_size = modules.shared.opts.ESRGAN_tile
        self.tile_pad = modules.shared.opts.ESRGAN_tile_overlap
        self.device = modules.shared.device
        self.img = None
        self.output = None
        self.scale = 1
        self.half = not modules.shared.cmd_opts.no_half
        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = None

        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)

        try:
            import cv2  # noqa: F401
            self.can_tile = True
        except Exception:
            pass

    @abstractmethod
    def do_upscale(self, img: PIL.Image, selected_model: str):
        return img

    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        self.scale = scale
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        # A large upscale allocates several full-size float buffers on the GPU
        # (the model's output, combine/linear copies). If the Stable Diffusion
        # model is still GPU-resident -- which it now is for sub-threshold
        # --medvram-sdxl jobs that run fully resident, where the juggling that
        # used to keep it parked never engages -- the two together OOM. Park the
        # SD model to CPU first; under lowvram/medvram it pages back in on the
        # next sampling pass via the forward hooks.
        #
        # Gate on the *intermediate* peak, not the requested dest: the model
        # upscales by its native factor (e.g. 4x) and the result is only then
        # resized down to the target. A 4x of a ~3k source is a ~12k x 7k
        # (~84MP) buffer even when the final output is ~12MP, so the requested
        # dest badly understates the peak.
        from modules import lowvram, sd_models
        sd_model = sd_models.model_data.sd_model  # already-loaded model or None; reading the field does not trigger a load
        if sd_model is not None and lowvram.is_enabled(sd_model):
            native_scale = 4.0
            for s in getattr(self, 'scalers', []):
                if s.data_path == selected_model:
                    native_scale = getattr(s, 'scale', None) or native_scale
                    break

            # the loop runs ceil(log_native(scale)) native passes (>=1), so the
            # largest intermediate is native_scale**passes times the source
            if native_scale > 1.001:
                passes = max(1, math.ceil(math.log(max(scale, 1.0)) / math.log(native_scale)))
                peak_factor = native_scale ** passes
            else:
                peak_factor = max(scale, 1.0)

            peak_mp = (img.width * peak_factor) * (img.height * peak_factor) / 1_000_000
            if peak_mp > shared.cmd_opts.upscale_evict_threshold_mp:
                lowvram.park_all(sd_model)
                devices.torch_gc()

        for i in range(3):
            if img.width >= dest_w and img.height >= dest_h and (i > 0 or scale != 1):
                break

            if shared.state.interrupted:
                break

            shape = (img.width, img.height)

            img = self.do_upscale(img, selected_model)

            if shape == (img.width, img.height):
                break

        if img.width != dest_w or img.height != dest_h:
            from modules.upscaler_utils import resize_preserving_float_gpu_linear
            img = resize_preserving_float_gpu_linear(img, int(dest_w), int(dest_h))

        # The upscale chain may have left sfi_tensor on the model device so the
        # whole loop stays GPU-resident; pull it back to CPU once here so
        # downstream consumers (SFI save, processing.py, scripts) see the
        # familiar CPU-resident tensor.
        sfi = getattr(img, 'sfi_tensor', None)
        if sfi is not None and isinstance(sfi, torch.Tensor) and sfi.device.type != 'cpu':
            img.sfi_tensor = sfi.detach().to(device='cpu')
        del sfi  # drop the (possibly GPU-resident) reference before reclaiming

        # The GPU-resident upscale + linear resize allocate several full-size
        # float tensors on the accelerator (4× model output, combine buffer,
        # linear/rescaled copies). Now that the only survivor — sfi_tensor — is
        # back on CPU, return those transient blocks to the OS. Without this the
        # caching allocator keeps them reserved, fragmenting the arena for the
        # next model — e.g. an img2img VAE encode that needs a large contiguous
        # block then OOMs even though enough total memory is free.
        devices.torch_gc()

        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def find_models(self, ext_filter=None) -> list:
        return modelloader.load_models(model_path=self.model_path, model_url=self.model_url, command_path=self.user_path, ext_filter=ext_filter)

    def update_status(self, prompt):
        print(f"\nextras: {prompt}", file=shared.progress_print_out)


class UpscalerData:
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(self, name: str, path: str, upscaler: Upscaler = None, scale: int = 4, model=None, sha256: str = None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model
        self.sha256 = sha256

    def __repr__(self):
        return f"<UpscalerData name={self.name} path={self.data_path} scale={self.scale}>"


class UpscalerNone(Upscaler):
    name = "None"
    scalers = []

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img

    def __init__(self, dirname=None):
        super().__init__(False)
        self.scalers = [UpscalerData("None", None, self)]


class UpscalerLanczos(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        from modules.upscaler_utils import resize_preserving_float
        return resize_preserving_float(img, int(img.width * self.scale), int(img.height * self.scale), resample=LANCZOS)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


class UpscalerNearest(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        from modules.upscaler_utils import resize_preserving_float
        return resize_preserving_float(img, int(img.width * self.scale), int(img.height * self.scale), resample=NEAREST)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]
