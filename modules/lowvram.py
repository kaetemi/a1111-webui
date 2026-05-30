from collections import namedtuple

import torch
from modules import devices, shared

module_in_gpu = None
cpu = torch.device("cpu")

# Whether the medvram/lowvram module juggling is currently engaged. When False
# the hooks load each module to the GPU on demand and leave it resident, which
# is the full-speed, non-medvram behavior. --medvram-sdxl toggles this per job
# based on resolution (see should_juggle); plain --medvram/--lowvram keep it on.
juggle_active = True

ModuleWithParent = namedtuple('ModuleWithParent', ['module', 'parent'], defaults=['None'])

def send_everything_to_cpu():
    global module_in_gpu

    if module_in_gpu is not None:
        module_in_gpu.to(cpu)

    module_in_gpu = None


def park_all(sd_model):
    """Push every medvram-pinned big module to CPU and reset the tracker. Unlike
    send_everything_to_cpu (which only evicts the single tracked module), this
    also clears modules left resident by a preceding non-juggling (low-res) job,
    giving the swap a clean slate before a high-res job starts."""
    global module_in_gpu

    for module in getattr(sd_model, '_medvram_modules', []):
        module.to(cpu)

    module_in_gpu = None


def should_juggle(sd_model, width, height):
    """Whether module juggling should be engaged for a job at this resolution.
    Plain --lowvram/--medvram always juggle; --medvram-sdxl is gated on the
    target megapixels so the common low-res case runs fully resident for speed."""
    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        return True

    threshold = shared.cmd_opts.medvram_sdxl_threshold_mp * 1_000_000
    return (width * height) > threshold


def set_juggle(active):
    global juggle_active
    juggle_active = active


def is_needed(sd_model):
    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram or shared.cmd_opts.medvram_sdxl and hasattr(sd_model, 'conditioner')


def apply(sd_model):
    enable = is_needed(sd_model)
    shared.parallel_processing_allowed = not enable

    if enable:
        setup_for_low_vram(sd_model, not shared.cmd_opts.lowvram)
    else:
        sd_model.lowvram = False


def setup_for_low_vram(sd_model, use_medvram):
    if getattr(sd_model, 'lowvram', False):
        return

    sd_model.lowvram = True

    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if not juggle_active:
            # full-resident mode: page this module in and leave it there
            module.to(devices.device)
            return

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(devices.device)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods

    first_stage_model = sd_model.first_stage_model
    first_stage_model_encode = sd_model.first_stage_model.encode
    first_stage_model_decode = sd_model.first_stage_model.decode

    def first_stage_model_encode_wrap(x):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_encode(x)

    def first_stage_model_decode_wrap(z):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_decode(z)

    to_remain_in_cpu = [
        (sd_model, 'first_stage_model'),
        (sd_model, 'depth_model'),
        (sd_model, 'embedder'),
        (sd_model, 'model'),
    ]

    is_sdxl = hasattr(sd_model, 'conditioner')
    is_sd2 = not is_sdxl and hasattr(sd_model.cond_stage_model, 'model')

    if hasattr(sd_model, 'medvram_fields'):
        to_remain_in_cpu = sd_model.medvram_fields()
    elif is_sdxl:
        to_remain_in_cpu.append((sd_model, 'conditioner'))
    elif is_sd2:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'model'))
    else:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'transformer'))

    # remove several big modules: cond, first_stage, depth/embedder (if applicable), and unet from the model
    stored = []
    for obj, field in to_remain_in_cpu:
        module = getattr(obj, field, None)
        stored.append(module)
        setattr(obj, field, None)

    # send the model to GPU.
    sd_model.to(devices.device)

    # put modules back. the modules will be in CPU.
    for (obj, field), module in zip(to_remain_in_cpu, stored):
        setattr(obj, field, module)

    # remember the big modules so park_all can evict any left resident by a
    # non-juggling (low-res) job; their submodules ride along via .to(cpu)
    sd_model._medvram_modules = [module for module in stored if module is not None]

    # register hooks for those the first three models
    if hasattr(sd_model, "cond_stage_model") and hasattr(sd_model.cond_stage_model, "medvram_modules"):
        for module in sd_model.cond_stage_model.medvram_modules():
            if isinstance(module, ModuleWithParent):
                parent = module.parent
                module = module.module
            else:
                parent = None

            if module:
                module.register_forward_pre_hook(send_me_to_gpu)

                if parent:
                    parents[module] = parent

    elif is_sdxl:
        sd_model.conditioner.register_forward_pre_hook(send_me_to_gpu)
    elif is_sd2:
        sd_model.cond_stage_model.model.register_forward_pre_hook(send_me_to_gpu)
        sd_model.cond_stage_model.model.token_embedding.register_forward_pre_hook(send_me_to_gpu)
        parents[sd_model.cond_stage_model.model] = sd_model.cond_stage_model
        parents[sd_model.cond_stage_model.model.token_embedding] = sd_model.cond_stage_model
    else:
        sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
        parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = first_stage_model_encode_wrap
    sd_model.first_stage_model.decode = first_stage_model_decode_wrap
    if getattr(sd_model, 'depth_model', None) is not None:
        sd_model.depth_model.register_forward_pre_hook(send_me_to_gpu)
    if getattr(sd_model, 'embedder', None) is not None:
        sd_model.embedder.register_forward_pre_hook(send_me_to_gpu)

    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4 GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        sd_model.model.to(devices.device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)


def is_enabled(sd_model):
    return sd_model.lowvram
