#!/usr/bin/env python3
"""Dump reference tensors from `diffusers` for the Rust golden tests.

This is the ground truth the port is verified against. Run it once per model
component; the Rust tests then assert agreement within tolerance.

Why this exists
---------------
A wrong axis permute or an off-by-one in a norm does not crash — it produces a
*plausible but subtly wrong* image, with no stack trace. Without per-module
reference tensors you are reduced to guessing which of several hundred ops is
at fault. With them, a failing test names the module.

Usage
-----
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch diffusers safetensors accelerate
    python3 xtask/golden/dump_reference.py vae --output tests/golden

Then, in the repo root:

    cargo test -p sd-models --test golden_vae -- --nocapture
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

SEED = 0
LATENT_SHAPE = (1, 4, 32, 32)  # -> 256x256 image, small enough to commit


def _require(module: str):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(
            f"error: `{module}` is not installed.\n"
            "  pip install torch diffusers safetensors accelerate"
        )


def dump_flux_vae(output: pathlib.Path, model_id: str) -> None:
    """Flux's VAE: the same convolutional geometry as SD, 16 latent channels.

    Two things here that the SD dump does not exercise. The latent is 16
    channels wide rather than 4, which is what lets Flux hold detail SD's
    latent cannot represent. And the latent parameterisation has a *shift*
    as well as a scale — `(x - shift) * scale` — so this dumps the scaled
    tensors too. Getting the shift backwards leaves a recognisable image with
    wrong contrast, which survives eyeballing and only a reference catches.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import AutoencoderKL
    from safetensors.torch import save_file

    out = output / "flux_vae"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} (subfolder=vae)")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()

    latent_channels = vae.config.latent_channels
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    print(f"  latent_channels={latent_channels} scaling={scaling} shift={shift}")

    gen = torch.Generator().manual_seed(SEED)
    latent = torch.randn(1, latent_channels, 32, 32, generator=gen, dtype=torch.float32)
    image_in = torch.randn(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        # Flux sets use_quant_conv/use_post_quant_conv false, so diffusers
        # leaves both as None and the latent goes straight into the decoder.
        z = vae.post_quant_conv(latent) if vae.post_quant_conv is not None else latent
        image = vae.decoder(z)
        raw_moments = vae.encoder(image_in)
        moments = vae.quant_conv(raw_moments) if vae.quant_conv is not None else raw_moments
        mean = moments[:, :latent_channels]
        # What the pipeline actually feeds the transformer, and what it feeds
        # back to the decoder: the round trip through the parameterisation.
        scaled = (mean - shift) * scaling
        unscaled = latent / scaling + shift
        pq = vae.post_quant_conv(unscaled) if vae.post_quant_conv is not None else unscaled
        decoded_from_scaled = vae.decoder(pq)

    tensors = {
        "latent": latent.contiguous(),
        "image": image.detach().contiguous().clone(),
        "encoder_input": image_in.contiguous(),
        "encoder_moments": moments.detach().contiguous().clone(),
        "encoder_scaled_mean": scaled.detach().contiguous().clone(),
        "decoded_from_scaled": decoded_from_scaled.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))

    weights = {k: v.detach().contiguous().clone() for k, v in vae.state_dict().items()}
    save_file(weights, str(out / "vae.safetensors"))
    print(f"wrote {out}/reference.safetensors and vae.safetensors")



def dump_flow(output: pathlib.Path, model_id: str) -> None:
    """FlowMatchEulerDiscreteScheduler sigmas, timesteps, and a step.

    The sigma schedule is where every rectified-flow implementation goes
    wrong, because the warp is easy to write plausibly and slightly off. Two
    resolutions are dumped so that a hardcoded shift cannot pass, and the
    schedule is taken from the scheduler itself rather than reimplemented
    here — a reference that shares our reasoning would verify nothing.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import save_file

    out = output / "flow"
    out.mkdir(parents=True, exist_ok=True)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    print(f"loaded {type(sched).__name__} from {model_id}")

    tensors = {}
    for label, seq_len in [("1024tok", 1024), ("4096tok", 4096)]:
        s = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
        mu = calculate_shift_compat(s.config, seq_len)
        s.set_timesteps(num_inference_steps=20, mu=mu)
        tensors[f"sigmas_{label}"] = s.sigmas.detach().float().contiguous().clone()
        tensors[f"timesteps_{label}"] = s.timesteps.detach().float().contiguous().clone()
        tensors[f"mu_{label}"] = torch.tensor([mu], dtype=torch.float32)

    # One Euler step, so the update rule is checked and not just the schedule.
    s = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
    s.set_timesteps(num_inference_steps=20, mu=calculate_shift_compat(s.config, 4096))
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, 16, 8, 8, generator=gen)
    v = torch.randn(1, 16, 8, 8, generator=gen)
    stepped = s.step(v, s.timesteps[3], x, return_dict=False)[0]
    tensors["step_x"] = x.contiguous()
    tensors["step_velocity"] = v.contiguous()
    tensors["step_index"] = torch.tensor([3], dtype=torch.float32)
    tensors["step_out"] = stepped.detach().contiguous().clone()

    # img2img: the forward noising the model was trained on.
    noise = torch.randn(1, 16, 8, 8, generator=gen)
    scaled = s.scale_noise(x, s.timesteps[5:6], noise)
    # `.clone()`: this is the same tensor as `step_x`, and safetensors
    # refuses to serialize two entries sharing one storage.
    tensors["scale_noise_sample"] = x.contiguous().clone()
    tensors["scale_noise_noise"] = noise.contiguous()
    tensors["scale_noise_index"] = torch.tensor([5], dtype=torch.float32)
    tensors["scale_noise_out"] = scaled.detach().contiguous().clone()

    save_file(tensors, str(out / "reference.safetensors"))
    print(f"wrote {out}/reference.safetensors")


def calculate_shift_compat(config, image_seq_len):
    """diffusers moved this helper around between versions."""
    m = (config.max_shift - config.base_shift) / (
        config.max_image_seq_len - config.base_image_seq_len
    )
    b = config.base_shift - m * config.base_image_seq_len
    return image_seq_len * m + b


def dump_vae(output: pathlib.Path, model_id: str) -> None:
    torch = _require("torch")
    _require("diffusers")
    from diffusers import AutoencoderKL
    from safetensors.torch import save_file

    out = output / "vae_decoder"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} (subfolder=vae)")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()

    # Fixed seed: the latent is *saved*, not regenerated in Rust. Matching
    # PyTorch's RNG bit-for-bit is a separate problem and not one worth solving
    # to verify a decoder.
    gen = torch.Generator().manual_seed(SEED)
    latent = torch.randn(LATENT_SHAPE, generator=gen, dtype=torch.float32)

    captured: dict[str, "torch.Tensor"] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            # `.clone()` is load-bearing: the last hook (`conv_out`) captures
            # the very tensor the decoder returns as `image`, and `.detach()`,
            # `.contiguous()` and `.float()` are all no-ops on it, so the two
            # entries would alias one storage. safetensors refuses to serialize
            # tensors that share memory.
            captured[name] = t.detach().contiguous().float().clone()

        return hook

    handles = [
        vae.decoder.conv_in.register_forward_hook(capture("conv_in")),
        vae.decoder.mid_block.register_forward_hook(capture("mid_block")),
        vae.decoder.conv_out.register_forward_hook(capture("conv_out")),
    ]
    for i, blk in enumerate(vae.decoder.up_blocks):
        handles.append(blk.register_forward_hook(capture(f"up_block_{i}")))

    with torch.no_grad():
        # post_quant_conv + decoder, i.e. what `decode_raw` does in Rust.
        z = vae.post_quant_conv(latent)
        image = vae.decoder(z)

    for h in handles:
        h.remove()

    tensors = {
        "latent": latent.contiguous(),
        "post_quant_conv": z.detach().contiguous(),
        "image": image.detach().contiguous(),
        **captured,
    }

    save_file(tensors, str(out / "reference.safetensors"))

    # The weights, too. Reference activations alone cannot verify anything —
    # the Rust decoder has to be run with the *same* parameters that produced
    # them, and `state_dict()` is by construction in the diffusers naming the
    # loader expects. Writing them here is what makes this script sufficient on
    # its own; without it the numerical test finds the activations, cannot find
    # the weights, and skips.
    weights = {k: v.detach().contiguous().clone() for k, v in vae.state_dict().items()}
    save_file(weights, str(out / "vae.safetensors"))

    # Encoder references. The encoder is what img2img needs, and its
    # downsampler pads asymmetrically — one row at the bottom, one column at
    # the right — so a symmetric implementation produces a half-pixel shift
    # per level that only a numerical comparison catches.
    image = torch.randn(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        moments = vae.quant_conv(vae.encoder(image))
    tensors["encoder_input"] = image.contiguous()
    tensors["encoder_moments"] = moments.detach().contiguous().clone()
    save_file(tensors, str(out / "reference.safetensors"))

    # Also link the *raw* checkpoint, unmodified. `vae.safetensors` above went
    # through `state_dict()`, which silently renames the legacy attention keys
    # on load — so it cannot catch a loader that only understands the modern
    # names. The stock file can, and that is the file users actually have.
    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    legacy = out / "vae_legacy.safetensors"
    if legacy.is_symlink() or legacy.exists():
        legacy.unlink()
    legacy.symlink_to(
        hf_hub_download(repo_id=model_id, filename="vae/diffusion_pytorch_model.safetensors")
    )

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<18} {tuple(v.shape)}")
    print(f"\nwrote {out / 'vae.safetensors'} ({len(weights)} tensors)")
    print(f"linked {legacy} -> the unmodified checkpoint")
    print(
        "\nIntermediate tensors are included on purpose: when the final image "
        "\nmismatches, they tell you *which block* diverged first."
    )


# Fixed prompt set for the tokenizer reference. Each one is here for a reason:
# the empty string and the overlong string pin the padding and truncation
# rules, the uppercase one pins casing, and the punctuation/digits one pins
# that we are not doing our own preprocessing.
PROMPTS = [
    "",
    "a photo of an astronaut riding a horse on mars",
    "a rusty crab on a beach",
    "A PHOTO OF A CAT",
    "hello, world! 123",
    "a " * 200,  # overlong, must truncate to 77
]

MAX_LENGTH = 77


def dump_clip_tokenizer(output: pathlib.Path, model_id: str) -> None:
    _require("transformers")
    from transformers import CLIPTokenizer

    out = output / "clip_tokenizer"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    tok = CLIPTokenizer.from_pretrained(model_id)

    ids = [
        tok(
            p,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )["input_ids"]
        for p in PROMPTS
    ]
    for prompt, row in zip(PROMPTS, ids):
        if len(row) != MAX_LENGTH:
            sys.exit(f"error: {prompt!r} encoded to {len(row)} ids, expected {MAX_LENGTH}")

    reference = {
        "prompts": PROMPTS,
        "ids": ids,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "max_length": MAX_LENGTH,
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")

    # The Rust side loads a `tokenizer.json`, so put one next to the reference
    # rather than making the test hunt through a HuggingFace cache. Pull the
    # canonical file from the repo: `CLIPTokenizer` is the slow tokenizer and
    # `save_pretrained` writes vocab.json/merges.txt instead.
    #
    # Deliberately a different artifact from the one that produced `ids` above.
    # The Rust test then encodes with this file and compares against ids from
    # the Python tokenizer, so agreement is a real cross-check rather than a
    # restatement of the same code.
    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    shutil.copyfile(
        hf_hub_download(repo_id=model_id, filename="tokenizer.json"),
        out / "tokenizer.json",
    )

    print(f"\nwrote {out / 'reference.json'}")
    for prompt, row in zip(PROMPTS, ids):
        shown = prompt if len(prompt) <= 40 else prompt[:37] + "..."
        print(f"  {shown!r:<45} -> [{row[0]}, {row[1]}, ..., {row[-1]}]")
    print(f"wrote {out / 'tokenizer.json'}")


ENCODER_PROMPT = "a photo of an astronaut riding a horse on mars"


def dump_clip_encoder(output: pathlib.Path, model_id: str) -> None:
    torch = _require("torch")
    _require("transformers")
    from safetensors.torch import save_file
    from transformers import CLIPTextModel, CLIPTokenizer

    out = output / "clip_encoder"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    tok = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPTextModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    batch = tok(
        ENCODER_PROMPT,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    token_ids = batch["input_ids"]

    captured: dict[str, "torch.Tensor"] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            # `.clone()` for the same reason as in dump_vae: the last capture
            # can alias the returned tensor, and safetensors refuses to
            # serialize tensors that share storage.
            captured[name] = t.detach().contiguous().float().clone()

        return hook

    handles = [
        model.text_model.embeddings.register_forward_hook(capture("embeddings"))
    ]
    for i, layer in enumerate(model.text_model.encoder.layers):
        handles.append(layer.register_forward_hook(capture(f"layer_{i:02d}")))

    with torch.no_grad():
        outputs = model(input_ids=token_ids)

    for h in handles:
        h.remove()

    tensors = {
        "token_ids": token_ids.contiguous(),
        "last_hidden_state": outputs.last_hidden_state.detach().contiguous().clone(),
        **captured,
    }
    save_file(tensors, str(out / "reference.safetensors"))

    # Weights too, so the Rust test can run the same parameters that produced
    # these activations. See dump_vae for why this is not optional.
    weights = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(weights, str(out / "clip.safetensors"))

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<18} {tuple(v.shape)}")
    print(f"wrote {out / 'clip.safetensors'} ({len(weights)} tensors)")
    print(
        "\nPer-layer captures are the point: when last_hidden_state mismatches "
        "\nthey tell you which layer diverged first."
    )


def dump_unet_blocks(output: pathlib.Path, model_id: str) -> None:
    torch = _require("torch")
    _require("diffusers")
    from diffusers import UNet2DConditionModel
    from diffusers.models.embeddings import get_timestep_embedding
    from safetensors.torch import save_file

    out = output / "unet_blocks"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} (subfolder=unet)")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    )
    unet.eval()

    gen = torch.Generator().manual_seed(SEED)
    timesteps = torch.tensor([0.0, 1.0, 500.0, 999.0])

    # Part A — the raw sinusoid. flip_sin_to_cos=True means cos then sin.
    sin_emb = get_timestep_embedding(
        timesteps, 320, flip_sin_to_cos=True, downscale_freq_shift=0
    )

    # Part B — the MLP that widens 320 -> 1280.
    with torch.no_grad():
        temb = unet.time_embedding(sin_emb)

    # Part C — the first resnet of the first down block.
    blk = unet.down_blocks[0].resnets[0]
    x = torch.randn(2, 320, 16, 16, generator=gen)
    t2 = temb[:2]
    with torch.no_grad():
        resnet_output = blk(x, t2)

    tensors = {
        "timesteps": timesteps.contiguous(),
        "sin_emb": sin_emb.detach().contiguous().clone(),
        "temb": temb.detach().contiguous().clone(),
        "resnet_input": x.contiguous(),
        "resnet_temb": t2.detach().contiguous().clone(),
        "resnet_output": resnet_output.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))

    # The isolated blocks, so Rust runs the same parameters rather than a
    # random-weight approximation of them.
    save_file(
        {k: v.detach().contiguous().clone() for k, v in blk.state_dict().items()},
        str(out / "resnet.safetensors"),
    )
    save_file(
        {
            k: v.detach().contiguous().clone()
            for k, v in unet.time_embedding.state_dict().items()
        },
        str(out / "time_embedding.safetensors"),
    )

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<16} {tuple(v.shape)}")
    print(f"wrote {out / 'resnet.safetensors'}")
    print(f"wrote {out / 'time_embedding.safetensors'}")


def dump_unet_attention(output: pathlib.Path, model_id: str) -> None:
    torch = _require("torch")
    _require("diffusers")
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file

    out = output / "unet_attention"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} (subfolder=unet)")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    )
    unet.eval()

    attn = unet.down_blocks[0].attentions[0]
    block = attn.transformer_blocks[0]

    x = torch.randn(2, 320, 16, 16, generator=torch.Generator().manual_seed(0))
    context = torch.randn(2, 77, 768, generator=torch.Generator().manual_seed(1))

    captured: dict[str, "torch.Tensor"] = {}

    def capture(name: str):
        def hook(_module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            captured[name] = t.detach().contiguous().float().clone()
            if name == "block_output" and inputs:
                captured["block_input"] = inputs[0].detach().contiguous().float().clone()

        return hook

    # Sub-block captures matter more here than anywhere else: with four
    # independently checkable stages a failure localizes to one of them
    # instead of "the transformer is wrong".
    handles = [
        block.register_forward_hook(capture("block_output")),
        block.attn1.register_forward_hook(capture("attn1_output")),
        block.attn2.register_forward_hook(capture("attn2_output")),
        block.ff.register_forward_hook(capture("ff_output")),
    ]

    with torch.no_grad():
        attn_output = attn(x, encoder_hidden_states=context)
    if not isinstance(attn_output, torch.Tensor):
        attn_output = attn_output.sample

    for h in handles:
        h.remove()

    tensors = {
        "attn_input": x.contiguous(),
        "context": context.contiguous(),
        "attn_output": attn_output.detach().contiguous().clone(),
        **captured,
    }
    save_file(tensors, str(out / "reference.safetensors"))
    save_file(
        {k: v.detach().contiguous().clone() for k, v in attn.state_dict().items()},
        str(out / "attention.safetensors"),
    )

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<16} {tuple(v.shape)}")
    print(f"wrote {out / 'attention.safetensors'}")


def dump_unet_full(output: pathlib.Path, model_id: str) -> None:
    torch = _require("torch")
    _require("diffusers")
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file

    out = output / "unet_full"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} (subfolder=unet)")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    )
    unet.eval()

    gen = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, 32, 32, generator=gen)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 77, 768, generator=gen)

    # diffusers returns the skip stack from its own down pass, but only
    # internally, so reproduce the push order with hooks instead: conv_in
    # first, then each resnet(+attn) pair, then each downsampler.
    skips: list["torch.Tensor"] = []
    mid_out: dict[str, "torch.Tensor"] = {}

    def push(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        skips.append(t.detach().contiguous().float().clone())

    handles = [unet.conv_in.register_forward_hook(push)]
    for block in unet.down_blocks:
        for j, _ in enumerate(block.resnets):
            # Push after the attention when there is one, else after the
            # resnet — that is the pair the down pass records.
            target = (
                block.attentions[j]
                if getattr(block, "attentions", None) is not None
                and j < len(block.attentions)
                else block.resnets[j]
            )
            handles.append(target.register_forward_hook(push))
        if getattr(block, "downsamplers", None):
            handles.append(block.downsamplers[0].register_forward_hook(push))

    def capture_mid(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        mid_out["mid_output"] = t.detach().contiguous().float().clone()

    handles.append(unet.mid_block.register_forward_hook(capture_mid))

    with torch.no_grad():
        result = unet(sample, timestep, encoder_hidden_states=context).sample

    for h in handles:
        h.remove()

    if len(skips) != 12:
        sys.exit(f"error: expected 12 skip tensors, captured {len(skips)}")

    tensors = {
        "sample": sample.contiguous(),
        "timestep": timestep.contiguous(),
        "context": context.contiguous(),
        "output": result.detach().contiguous().clone(),
        **mid_out,
        **{f"down_{i:02d}": t for i, t in enumerate(skips)},
    }
    save_file(tensors, str(out / "reference.safetensors"))

    # Symlink the checkpoint's own weights rather than re-saving 3.4 GB. The
    # test then has one fixed path to open and no knowledge of the HF cache.
    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(
        repo_id=model_id, filename="unet/diffusion_pytorch_model.safetensors"
    )
    link = out / "unet.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<14} {tuple(v.shape)}")
    print(f"linked {link} -> {weights}")


SAMPLER_SIGMAS = [14.6146, 10.0, 6.0, 3.0, 1.5, 0.5, 0.0]


def dump_samplers(output: pathlib.Path, _model_id: str) -> None:
    """Reference steps for Euler ancestral and DPM++ 2M.

    Deliberately does not import k-diffusion. The formulas are written out in
    numpy right here so the Rust and Python sides are visibly the same
    equations rather than two libraries that agree for unknown reasons.
    """
    torch = _require("torch")
    np = _require("numpy")
    from safetensors.torch import save_file

    out = output / "samplers"
    out.mkdir(parents=True, exist_ok=True)

    shape = (1, 4, 8, 8)
    x0 = np.random.default_rng(0).standard_normal(shape).astype("float32")
    denoised = np.random.default_rng(1).standard_normal(shape).astype("float32")
    noise = np.random.default_rng(2).standard_normal(shape).astype("float32")

    tensors = {
        "x": torch.from_numpy(x0),
        "denoised": torch.from_numpy(denoised),
        "noise": torch.from_numpy(noise),
    }

    # -- Part A: the sampling sigma schedule -----------------------------
    #
    #   step = (len(train) - 1) / (n - 1)
    #   idx  = (n - 1 - i) * step        <- descending
    #   out  = lerp(train[floor(idx)], train[floor(idx)+1], frac)
    #   then a trailing 0.0
    betas = np.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=np.float64) ** 2
    alphas_cumprod = np.cumprod(1.0 - betas)
    train = np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)

    n = 20
    step = (len(train) - 1) / (n - 1)
    sigmas_20 = []
    for i in range(n):
        idx = (n - 1 - i) * step
        lo = int(np.floor(idx))
        hi = min(lo + 1, len(train) - 1)
        frac = idx - lo
        sigmas_20.append(train[lo] * (1.0 - frac) + train[hi] * frac)
    sigmas_20.append(0.0)
    tensors["sigmas_20"] = torch.tensor(sigmas_20, dtype=torch.float64)

    # -- Part B: Euler ancestral, one step per sigma pair ----------------
    #
    #   sigma_up   = min(s_next, sqrt(s_next^2 * (s^2 - s_next^2) / s^2))
    #   sigma_down = sqrt(max(0, s_next^2 - sigma_up^2))
    #   d = (x - denoised) / sigma
    #   x = x + d * (sigma_down - sigma)
    #   x = x + noise * sigma_up      (only when s_next > 0)
    for i in range(len(SAMPLER_SIGMAS) - 1):
        s, s_next = SAMPLER_SIGMAS[i], SAMPLER_SIGMAS[i + 1]
        sigma_up = min(s_next, np.sqrt(s_next**2 * (s**2 - s_next**2) / s**2))
        sigma_down = np.sqrt(max(0.0, s_next**2 - sigma_up**2))
        d = (x0 - denoised) / s
        stepped = x0 + d * (sigma_down - s)
        if s_next > 0:
            stepped = stepped + noise * sigma_up
        tensors[f"euler_step_{i}"] = torch.from_numpy(stepped.astype("float32"))

    # -- Part C: DPM++ 2M, sequential with carried state -----------------
    #
    #   t = -ln(sigma);  h = t_next - t
    #   first step or s_next == 0:  first-order
    #   else: r = (t - t_prev) / h
    #         d = (1 + 1/(2r)) * denoised - (1/(2r)) * prev_denoised
    #   x_next = (s_next / s) * x - (exp(-h) - 1) * d
    x_cur = x0.copy()
    prev_denoised = None
    prev_t = None
    for i in range(len(SAMPLER_SIGMAS) - 1):
        s, s_next = SAMPLER_SIGMAS[i], SAMPLER_SIGMAS[i + 1]
        if s_next == 0.0:
            x_cur = denoised.copy()
            prev_denoised, prev_t = denoised.copy(), -np.log(s)
        else:
            t = -np.log(s)
            t_next = -np.log(s_next)
            h = t_next - t
            if prev_denoised is None:
                d = denoised
            else:
                r = (t - prev_t) / h
                inv = 1.0 / (2.0 * r)
                d = (1.0 + inv) * denoised - inv * prev_denoised
            x_cur = (s_next / s) * x_cur - (np.exp(-h) - 1.0) * d
            prev_denoised, prev_t = denoised.copy(), t
        tensors[f"dpmpp_step_{i}"] = torch.from_numpy(
            np.ascontiguousarray(x_cur).astype("float32")
        )

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "reference.safetensors"))

    print(f"\nwrote {out / 'reference.safetensors'}")
    print(f"  sigmas_20[0]={sigmas_20[0]:.4f} .. [-1]={sigmas_20[-1]:.1f} ({len(sigmas_20)} values)")
    for k in sorted(tensors):
        if k.startswith(("euler", "dpmpp")):
            print(f"  {k:<16} {tuple(tensors[k].shape)}")


def dump_sdxl_text_encoder_2(output: pathlib.Path, model_id: str) -> None:
    torch = _require("torch")
    _require("transformers")
    from safetensors.torch import save_file
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    out = output / "sdxl_text_encoder_2"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} (subfolder=text_encoder_2)")
    tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    model = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch.float32
    )
    model.eval()

    batch = tok(
        ENCODER_PROMPT,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    token_ids = batch["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids=token_ids, output_hidden_states=True)

    # SDXL conditions on hidden_states[-2] — the penultimate layer, and raw
    # (no final_layer_norm) — plus the projected pooled embedding. Both are
    # saved because taking the wrong one still produces plausible images.
    tensors = {
        "token_ids": token_ids.contiguous(),
        "penultimate": outputs.hidden_states[-2].detach().contiguous().clone(),
        "last_hidden_state": outputs.last_hidden_state.detach().contiguous().clone(),
        "pooled": outputs.text_embeds.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))
    save_file(
        {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()},
        str(out / "text_encoder_2.safetensors"),
    )

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<20} {tuple(v.shape)}")
    print(f"wrote {out / 'text_encoder_2.safetensors'}")


def dump_sdxl_unet(output: pathlib.Path, model_id: str) -> None:
    torch = _require("torch")
    _require("diffusers")
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file

    out = output / "sdxl_unet"
    out.mkdir(parents=True, exist_ok=True)

    # fp16 variant: half the download, and both sides load the same file, so
    # the comparison stays apples-to-apples. Upcast to fp32 for the reference
    # because that is what the Rust side runs.
    print(f"loading {model_id} (subfolder=unet, variant=fp16)")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", variant="fp16", torch_dtype=torch.float32
    )
    unet.eval()

    gen = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, 32, 32, generator=gen)
    timestep = torch.tensor([500.0])
    # SDXL's cross-attention dim is 2048: the two encoders concatenated.
    context = torch.randn(1, 77, 2048, generator=gen)
    pooled = torch.randn(1, 1280, generator=gen)
    # original h/w, crop top/left, target h/w.
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]])

    mid_out = {}

    def capture_mid(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        mid_out["mid_output"] = t.detach().contiguous().float().clone()

    handle = unet.mid_block.register_forward_hook(capture_mid)
    with torch.no_grad():
        result = unet(
            sample,
            timestep,
            encoder_hidden_states=context,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample
    handle.remove()

    tensors = {
        "sample": sample.contiguous(),
        "timestep": timestep.contiguous(),
        "context": context.contiguous(),
        "pooled": pooled.contiguous(),
        "time_ids": time_ids.contiguous(),
        "output": result.detach().contiguous().clone(),
        **mid_out,
    }
    save_file(tensors, str(out / "reference.safetensors"))

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    link = out / "unet.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(
        hf_hub_download(
            repo_id=model_id, filename="unet/diffusion_pytorch_model.fp16.safetensors"
        )
    )

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<14} {tuple(v.shape)}")
    print(f"linked {link}")


# Small GGUF files produced by llama.cpp, for testing the header reader
# against something we did not write ourselves. Chosen for size — the largest
# is 67 MB — and for covering cases a synthetic file cannot: a real metadata
# block, a genuinely mixed quantisation spread, and a big-endian build.
GGUF_FIXTURES = [
    ("ggml-org/stories15M_MOE", "moe_shakespeare15M.gguf"),
    ("ggml-org/stories15M_MOE", "stories15M_MOE-Q8_0.gguf"),
    ("ggml-org/models", "bert-bge-small/ggml-model-f16-big-endian.gguf"),
    # A real Stable Diffusion checkpoint as stable-diffusion.cpp writes it:
    # CompVis/LDM names, no metadata whatsoever. 1.5 GB, and the only fixture
    # here that exercises the *naming* rather than the format.
    ("second-state/stable-diffusion-v1-5-GGUF",
     "stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf"),
]


def dump_gguf(output: pathlib.Path, _model_id: str) -> None:
    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    out = output / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    for repo, name in GGUF_FIXTURES:
        src = hf_hub_download(repo_id=repo, filename=name)
        # The SD checkpoint gets a short stable name; the rest keep theirs.
        short = "sd15-q4_0.gguf" if "stable-diffusion-v1-5" in name else pathlib.Path(name).name
        link = out / short
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        print(f"  {os.path.getsize(src) / 1e6:8.2f} MB  {link.name}")
    print(f"\nlinked into {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="component", required=True)

    vae = sub.add_parser("vae", help="dump VAE decoder references")
    vae.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HuggingFace model id",
    )
    vae.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    fluxvae = sub.add_parser("flux_vae", help="dump Flux VAE (16-channel) references")
    fluxvae.add_argument(
        "--model-id",
        default="Freepik/flux.1-lite-8B",
        help="ungated repo carrying the Flux VAE; black-forest-labs is gated",
    )
    fluxvae.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    flow = sub.add_parser("flow", help="dump rectified-flow scheduler references")
    flow.add_argument("--model-id", default="Freepik/flux.1-lite-8B")
    flow.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    clip = sub.add_parser("clip_tokenizer", help="dump CLIP tokenizer references")
    clip.add_argument(
        "--model-id",
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model id",
    )
    clip.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    encoder = sub.add_parser("clip_encoder", help="dump CLIP text encoder references")
    encoder.add_argument(
        "--model-id",
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model id",
    )
    encoder.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    blocks = sub.add_parser("unet_blocks", help="dump UNet resnet/embedding references")
    blocks.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HuggingFace model id",
    )
    blocks.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    unet_attn = sub.add_parser("unet_attention", help="dump UNet transformer references")
    unet_attn.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HuggingFace model id",
    )
    unet_attn.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    unet_full = sub.add_parser("unet_full", help="dump whole-UNet references")
    unet_full.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HuggingFace model id",
    )
    unet_full.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    samplers = sub.add_parser("samplers", help="dump sampler step references")
    samplers.add_argument("--model-id", default="", help="unused; samplers need no weights")
    samplers.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    sdxl2 = sub.add_parser("sdxl_text_encoder_2", help="dump SDXL text encoder 2 references")
    sdxl2.add_argument(
        "--model-id", default="stabilityai/stable-diffusion-xl-base-1.0", help="HuggingFace model id"
    )
    sdxl2.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    sdxlu = sub.add_parser("sdxl_unet", help="dump SDXL UNet references")
    sdxlu.add_argument(
        "--model-id", default="stabilityai/stable-diffusion-xl-base-1.0", help="HuggingFace model id"
    )
    sdxlu.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    gg = sub.add_parser("gguf", help="link small real GGUF files for the header tests")
    gg.add_argument("--model-id", default="", help="unused")
    gg.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    args = ap.parse_args()
    if args.component == "gguf":
        dump_gguf(args.output, args.model_id)
    elif args.component == "sdxl_unet":
        dump_sdxl_unet(args.output, args.model_id)
    elif args.component == "sdxl_text_encoder_2":
        dump_sdxl_text_encoder_2(args.output, args.model_id)
    elif args.component == "samplers":
        dump_samplers(args.output, args.model_id)
    elif args.component == "unet_full":
        dump_unet_full(args.output, args.model_id)
    elif args.component == "unet_attention":
        dump_unet_attention(args.output, args.model_id)
    elif args.component == "unet_blocks":
        dump_unet_blocks(args.output, args.model_id)
    elif args.component == "flux_vae":
        dump_flux_vae(args.output, args.model_id)
    elif args.component == "flow":
        dump_flow(args.output, args.model_id)
    elif args.component == "vae":
        dump_vae(args.output, args.model_id)
    elif args.component == "clip_tokenizer":
        dump_clip_tokenizer(args.output, args.model_id)
    elif args.component == "clip_encoder":
        dump_clip_encoder(args.output, args.model_id)


if __name__ == "__main__":
    main()
