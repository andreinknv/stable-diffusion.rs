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
import time

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

    out = output / ("sd35_vae" if "3.5" in model_id else "flux_vae")
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



def dump_t5(output: pathlib.Path, model_id: str) -> None:
    """T5 v1.1 encoder against `transformers`.

    Deliberately the *small* checkpoint. T5-XXL is 4.7B parameters and its
    reference tensors would be unusable as a fixture, while the architecture
    is identical — same RMSNorm, same unscaled attention, same relative
    position buckets, same gated GELU. Verifying the code here and then
    loading XXL weights into it separates "is the port right" from "is the
    name mapping right", which is the split that made the GGUF work
    tractable.

    Per-block hidden states are captured, not just the output, so a
    divergence localises to a block instead of being reported at the end.
    """
    torch = _require("torch")
    _require("transformers")
    from transformers import T5EncoderModel
    from safetensors.torch import save_file

    out = output / "t5"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    cfg = model.config
    print(
        f"  d_model={cfg.d_model} d_ff={cfg.d_ff} layers={cfg.num_layers} "
        f"heads={cfg.num_heads} d_kv={cfg.d_kv}"
    )

    gen = torch.Generator().manual_seed(SEED)
    # Ordinary ids well inside the vocabulary; the tokenizer is verified
    # separately and mixing the two would confuse a failure here.
    ids = torch.randint(0, 32000, (1, 24), generator=gen)

    with torch.no_grad():
        result = model(input_ids=ids, output_hidden_states=True)

    tensors = {
        "token_ids": ids.to(torch.int64).contiguous(),
        "last_hidden_state": result.last_hidden_state.detach().contiguous().clone(),
    }
    for i, h in enumerate(result.hidden_states):
        tensors[f"hidden_{i}"] = h.detach().contiguous().clone()

    # The position bias itself, which is the piece most likely to be wrong and
    # the hardest to infer from a whole-model mismatch.
    attn = model.encoder.block[0].layer[0].SelfAttention
    with torch.no_grad():
        bias = attn.compute_bias(ids.shape[1], ids.shape[1])
    tensors["position_bias"] = bias.detach().contiguous().clone()

    save_file(tensors, str(out / "reference.safetensors"))
    weights = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(weights, str(out / "t5.safetensors"))
    print(f"wrote {out}/reference.safetensors ({len(tensors)} tensors) and t5.safetensors")



def dump_llm(output: pathlib.Path, model_id: str) -> None:
    """A Qwen3-family decoder used as a *text encoder*.

    **Deliberately the smallest checkpoint of the family**, for the same
    reason `dump_t5` uses T5-small: the architecture is identical at every
    size — RMSNorm pre-norm, grouped-query attention with QK-norm, SwiGLU,
    rotary embeddings — so verifying the port at 0.6B and then loading 4B or
    7B weights into it separates "is the forward right" from "is the name
    mapping right".

    That split is what makes this tractable at all, because the checkpoints
    that actually matter here are the text encoders for Qwen-Image (Qwen2.5-VL,
    hidden 3584), Z-Image (Qwen3, hidden 2560) and FLUX.2 (Mistral, hidden
    5120), none of which is a reasonable fixture.

    **The hidden states are what a diffusion model consumes, not the logits.**
    These models are decoders being used as encoders: the sampling head is
    never run, and the conditioning is a hidden state from partway up the
    stack. Per-layer states are captured so a divergence localises.
    """
    torch = _require("torch")
    _require("transformers")
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_file

    out = output / "llm"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).eval()
    cfg = model.config
    print(
        f"  hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
        f"head_dim={getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)} "
        f"intermediate={cfg.intermediate_size} theta={cfg.rope_theta} "
        f"eps={cfg.rms_norm_eps}"
    )

    gen = torch.Generator().manual_seed(SEED)
    # Ids well inside the vocabulary. The tokenizer is a separate concern and
    # mixing the two would confuse a failure here.
    ids = torch.randint(0, 10000, (1, 16), generator=gen)

    with torch.no_grad():
        result = model(input_ids=ids, output_hidden_states=True)

    tensors = {
        "token_ids": ids.to(torch.int64).contiguous(),
        # `hidden_states[-1]` is post-final-norm; `hidden_states[0]` is the
        # embedding before any layer.
        "last_hidden_state": result.hidden_states[-1].detach().contiguous().clone(),
    }
    for i, h in enumerate(result.hidden_states):
        tensors[f"hidden_{i}"] = h.detach().contiguous().clone()

    save_file(tensors, str(out / "reference.safetensors"))
    weights = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(weights, str(out / "llm.safetensors"))

    import json

    (out / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": getattr(
                    cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
                ),
                "intermediate_size": cfg.intermediate_size,
                "rms_norm_eps": cfg.rms_norm_eps,
                "rope_theta": cfg.rope_theta,
                "vocab_size": cfg.vocab_size,
                "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", False),
            },
            indent=2,
        )
    )
    print(f"wrote {out}/reference.safetensors ({len(tensors)} tensors) and llm.safetensors")


def dump_flux_transformer(output: pathlib.Path, model_id: str) -> None:
    """Flux's MMDiT against diffusers.

    The checkpoint ships in black-forest-labs layout (`double_blocks.0.
    img_attn.qkv`) while diffusers uses its own renaming, so diffusers is
    loaded through `from_single_file`, which applies the conversion. Our Rust
    model reads the original names directly. That means this compares two
    independent readings of the *same published file* rather than a
    round trip through one library's conventions — the lesson the legacy VAE
    attention names taught earlier in this project.

    Kept deliberately small: 16x16 latent, 8 text tokens. The transformer is
    3.2B parameters and the reference has to fit next to it in memory.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import FluxTransformer2DModel
    from safetensors.torch import save_file

    out = output / "flux_transformer"
    out.mkdir(parents=True, exist_ok=True)

    src = output / "flux" / "flux-mini.safetensors"
    print(f"loading {src} with the {model_id} config")
    model = FluxTransformer2DModel.from_single_file(
        str(src), config=model_id, torch_dtype=torch.float32
    ).eval()

    gen = torch.Generator().manual_seed(SEED)
    lat_h, lat_w = 16, 16          # patch grid, so a 32x32 latent
    img_len = lat_h * lat_w
    txt_len = 8

    hidden = torch.randn(1, img_len, 64, generator=gen)
    encoder_hidden = torch.randn(1, txt_len, 4096, generator=gen)
    pooled = torch.randn(1, 768, generator=gen)
    timestep = torch.tensor([0.7])
    guidance = torch.tensor([3.5])

    # diffusers expects ids as [seq, 3] and builds the image grid itself.
    img_ids = torch.zeros(img_len, 3)
    img_ids[:, 1] = torch.arange(lat_h).repeat_interleave(lat_w).float()
    img_ids[:, 2] = torch.arange(lat_w).repeat(lat_h).float()
    txt_ids = torch.zeros(txt_len, 3)

    with torch.no_grad():
        result = model(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]

    tensors = {
        "hidden_states": hidden.contiguous(),
        "encoder_hidden_states": encoder_hidden.contiguous(),
        "pooled_projections": pooled.contiguous(),
        "timestep": timestep.contiguous(),
        "guidance": guidance.contiguous(),
        "latent_h": torch.tensor([lat_h], dtype=torch.float32),
        "latent_w": torch.tensor([lat_w], dtype=torch.float32),
        "output": result.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))
    print(f"wrote {out}/reference.safetensors, output {tuple(result.shape)}")



def dump_flux_sampling(output: pathlib.Path, model_id: str) -> None:
    """Twenty steps of diffusers' Flux loop, from fixed inputs.

    The per-component tests verify one forward pass. This verifies the *loop*
    — twenty steps of schedule, step rule and re-entry, where a small error
    compounds instead of showing up once.

    Conditioning and the initial noise are supplied rather than generated, so
    the only thing under comparison is the loop. Export them from Rust with
    `cargo run --release -p sd-cli --example flux_export_inputs`, which writes
    the same file this reads.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file, save_file

    out = output / "flux_sampling"
    src = out / "reference.safetensors"
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Export the inputs from Rust first:\n"
            "  cargo run --release -p sd-cli --example flux_export_inputs -- "
            f"{src}"
        )
    d = load_file(str(src))
    txt, pooled = d["txt"].float(), d["pooled"].float()
    xs = d["init_packed"].float().clone()

    model = FluxTransformer2DModel.from_single_file(
        str(output / "flux" / "flux-mini.safetensors"),
        config=model_id,
        torch_dtype=torch.float32,
    ).eval()

    steps, seq = 20, xs.shape[1]
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Freepik/flux.1-lite-8B", subfolder="scheduler"
    )
    mu = calculate_shift_compat(sched.config, seq)
    sched.set_timesteps(num_inference_steps=steps, mu=mu)

    ph = pw = int(seq ** 0.5)
    img_ids = torch.zeros(seq, 3)
    img_ids[:, 1] = torch.arange(ph).repeat_interleave(pw).float()
    img_ids[:, 2] = torch.arange(pw).repeat(ph).float()
    txt_ids = torch.zeros(txt.shape[1], 3)
    guidance = torch.tensor([3.5])

    with torch.no_grad():
        for t in sched.timesteps:
            v = model(
                hidden_states=xs, encoder_hidden_states=txt,
                pooled_projections=pooled, timestep=t.expand(1) / 1000,
                img_ids=img_ids, txt_ids=txt_ids, guidance=guidance,
                return_dict=False,
            )[0]
            xs = sched.step(v, t, xs, return_dict=False)[0]

    b, n, cc = xs.shape
    lat = (
        xs.view(b, ph, pw, cc // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, cc // 4, ph * 2, pw * 2)
    )
    d["reference_latent"] = lat.detach().contiguous().clone()
    save_file({k: v.contiguous() for k, v in d.items()}, str(src))
    print(f"wrote {src}")



def dump_sd3(output: pathlib.Path, model_id: str) -> None:
    """SD 3.5's MMDiT against diffusers.

    diffusers stores this model under its own renaming while the published
    checkpoint uses Stability's, so `from_single_file` does the conversion and
    our Rust side reads the original names — two independent readings of one
    published file rather than a round trip through either library's
    conventions.

    Small on purpose: a 32x32 latent and 16 text tokens. The point is
    numerical agreement, not throughput.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import SD3Transformer2DModel
    from safetensors.torch import save_file

    out = output / "sd3_transformer"
    out.mkdir(parents=True, exist_ok=True)

    # Load the *single-file* checkpoint, not the diffusers-converted copy.
    # The two are published from different sources and their weights differ by
    # up to 2e-3 — enough, through 24 blocks whose activations reach 97,000,
    # to swamp the thing being measured. Our Rust side reads this same file.
    single = output / "sd35" / "sd35-medium.safetensors"
    if single.exists():
        print(f"loading {single} (single-file, matching what Rust reads)")
        model = SD3Transformer2DModel.from_single_file(
            str(single), config=model_id, subfolder="transformer",
            torch_dtype=torch.float32,
        ).eval()
    else:
        print(f"loading {model_id} (converted copy; expect ~1e-2 disagreement)")
        model = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float32
        ).eval()
    cfg = model.config
    print(
        f"  hidden={cfg.num_attention_heads * cfg.attention_head_dim} "
        f"layers={cfg.num_layers} dual={getattr(cfg, 'dual_attention_layers', None)}"
    )

    gen = torch.Generator().manual_seed(SEED)
    latents = torch.randn(1, 16, 32, 32, generator=gen)
    context = torch.randn(1, 16, 4096, generator=gen)
    pooled = torch.randn(1, 2048, generator=gen)
    timestep = torch.tensor([700.0])

    with torch.no_grad():
        result = model(
            hidden_states=latents,
            encoder_hidden_states=context,
            pooled_projections=pooled,
            timestep=timestep,
            return_dict=False,
        )[0]

    save_file(
        {
            "latents": latents.contiguous(),
            "context": context.contiguous(),
            "pooled": pooled.contiguous(),
            "timestep": timestep.contiguous(),
            "output": result.detach().contiguous().clone(),
        },
        str(out / "reference.safetensors"),
    )
    print(f"wrote {out}/reference.safetensors, output {tuple(result.shape)}")


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

    # SD 2.x has its own directory: same code path, 1024-wide cross attention
    # and a different config, so the two sets of tensors are not comparable.
    out = output / "unet_full"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} (subfolder=unet)")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    )
    unet.eval()
    cross = unet.config.cross_attention_dim
    if cross != 768:
        out = output / f"unet_full_cross{cross}"
        out.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, 32, 32, generator=gen)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 77, cross, generator=gen)

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


def dump_controlnet(output: pathlib.Path, model_id: str) -> None:
    """ControlNet's corrections for one step, against a fixed hint.

    Thirteen tensors, not one: a ControlNet emits a correction per skip plus
    one for the mid block, and a single summary number could not say which of
    the thirteen is wrong. They are also the only outputs -- a ControlNet has
    no image of its own to look at -- so this is the whole of its behaviour.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import ControlNetModel
    from safetensors.torch import save_file

    out = output / "controlnet"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    net = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float32)
    net.eval()

    gen = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, 32, 32, generator=gen)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 77, 768, generator=gen)
    # The hint is at *pixel* resolution -- 8x the latent -- and in [0, 1],
    # unlike every other image in this project, which is [-1, 1]. Getting
    # either wrong still runs.
    hint = torch.rand(1, 3, 256, 256, generator=gen)

    with torch.no_grad():
        result = net(
            sample,
            timestep,
            encoder_hidden_states=context,
            controlnet_cond=hint,
            conditioning_scale=1.0,
            return_dict=True,
        )

    downs = list(result.down_block_res_samples)
    if len(downs) != 12:
        sys.exit(f"error: expected 12 corrections, got {len(downs)}")

    tensors = {
        "sample": sample.contiguous(),
        "timestep": timestep.contiguous(),
        "context": context.contiguous(),
        "hint": hint.contiguous(),
        "mid": result.mid_block_res_sample.detach().contiguous().clone(),
        **{f"down_{i:02d}": t.detach().contiguous().clone() for i, t in enumerate(downs)},
    }
    save_file(tensors, str(out / "reference.safetensors"))

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(
        repo_id=model_id, filename="diffusion_pytorch_model.safetensors"
    )
    link = out / "controlnet.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<12} {tuple(v.shape)}")
    print(f"linked {link} -> {weights}")


def dump_taesd(output: pathlib.Path, model_id: str) -> None:
    """TAESD, the tiny distilled decoder.

    Both the raw module stack and the wrapped `decode` are dumped. The wrapper
    is where TAESD's own latent convention lives -- `latent_magnitude = 3`,
    `latent_shift = 0.5`, and emphatically not the SD VAE's 0.18215 -- and
    mixing the two conventions gives a washed-out image rather than an error,
    so the test has to cover the scaling and not just the convolutions.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import AutoencoderTiny
    from safetensors.torch import save_file

    # Named for the checkpoint, so taesd and taesdxl can both be present.
    out = output / model_id.rsplit("/", 1)[-1]
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    tae = AutoencoderTiny.from_pretrained(model_id, torch_dtype=torch.float32)
    tae.eval()

    gen = torch.Generator().manual_seed(SEED)
    # Flux and SD 3 have 16-channel latents, SD 1.5/SDXL have 4. Read it from
    # the checkpoint rather than assuming: a wrong channel count fails to load
    # in Rust, which is the good direction, but it would fail here first.
    channels = tae.config.latent_channels
    latent = torch.randn(1, channels, LATENT_SHAPE[2], LATENT_SHAPE[3], generator=gen)
    image = torch.rand(1, 3, 256, 256, generator=gen) * 2 - 1

    with torch.no_grad():
        raw = tae.decoder(latent)
        decoded = tae.decode(latent).sample
        encoded_raw = tae.encoder(image)
        encoded = tae.encode(image).latents

    tensors = {
        "latent": latent.contiguous(),
        "decoder_raw": raw.detach().contiguous().clone(),
        "decoded": decoded.detach().contiguous().clone(),
        "image": image.contiguous(),
        "encoder_raw": encoded_raw.detach().contiguous().clone(),
        "encoded": encoded.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(
        repo_id=model_id, filename="diffusion_pytorch_model.safetensors"
    )
    link = out / "weights.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<13} {tuple(v.shape)}  range [{v.min():.3f}, {v.max():.3f}]")
    print(f"linked {link} -> {weights}")
    print(f"\nconfig: {dict(tae.config)}")


def dump_esrgan(output: pathlib.Path, model_id: str) -> None:
    """Real-ESRGAN x4 (RRDBNet), converted from .pth and run once.

    The checkpoint is a pickled torch state dict, not safetensors, so this
    converts it — that conversion is the only reason the Rust side needs a
    Python step at all. Weights are stored under `params_ema` in the official
    release; older mirrors use `params` or a bare dict, so all three are tried.
    """
    torch = _require("torch")
    from safetensors.torch import save_file

    out = output / "esrgan"
    out.mkdir(parents=True, exist_ok=True)

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    print(f"loading {model_id}")
    src = hf_hub_download(repo_id=model_id, filename="RealESRGAN_x4.pth")
    blob = torch.load(src, map_location="cpu", weights_only=True)
    state = blob.get("params_ema", blob.get("params", blob))
    state = {k: v.contiguous().float() for k, v in state.items()}
    save_file(state, str(out / "esrgan_x4.safetensors"))

    # A small input: 4x of 64 is 256, which is enough to see the upsampling
    # and small enough to commit alongside the other references.
    gen = torch.Generator().manual_seed(SEED)
    image = torch.rand(1, 3, 64, 64, generator=gen)

    # Rebuild the architecture here rather than importing basicsr, which drags
    # in a large dependency for forty lines of convolutions.
    import torch.nn as nn
    import torch.nn.functional as F

    class RDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, nf=64, nb=23, gc=32):
            super().__init__()
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    net = RRDBNet()
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        result = net(image)

    tensors = {
        "image": image.contiguous(),
        "output": result.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))
    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<8} {tuple(v.shape)}  range [{v.min():.3f}, {v.max():.3f}]")
    print(f"converted {len(state)} tensors -> {out / 'esrgan_x4.safetensors'}")


def dump_clip_vision(output: pathlib.Path, model_id: str) -> None:
    """CLIP's vision tower, as IP-Adapter ships it for SD 1.5."""
    torch = _require("torch")
    _require("transformers")
    from safetensors.torch import save_file
    from transformers import CLIPVisionModelWithProjection

    out = output / "clip_vision"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    net = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="models/image_encoder", torch_dtype=torch.float32
    )
    net.eval()
    cfg = net.config
    print(
        f"  hidden {cfg.hidden_size} layers {cfg.num_hidden_layers} "
        f"heads {cfg.num_attention_heads} patch {cfg.patch_size} image {cfg.image_size}"
    )

    gen = torch.Generator().manual_seed(SEED)
    # Already normalised: the Rust side is handed the same tensor, so the
    # preprocessing is compared separately rather than folded in here.
    pixels = torch.randn(1, 3, cfg.image_size, cfg.image_size, generator=gen)

    with torch.no_grad():
        outputs = net.vision_model(pixels, output_hidden_states=False)
        hidden = outputs.last_hidden_state
        pooled = outputs.pooler_output

    tensors = {
        "pixels": pixels.contiguous(),
        "hidden": hidden.detach().contiguous().clone(),
        "pooled": pooled.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(repo_id=model_id, filename="models/image_encoder/model.safetensors")
    link = out / "image_encoder.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<8} {tuple(v.shape)}")


def dump_ip_adapter(output: pathlib.Path, model_id: str) -> None:
    """A UNet forward with IP-Adapter attached, plus the projected tokens.

    The point of this reference is the *index mapping*. The checkpoint numbers
    its entries by diffusers' flat processor order, which is not the order a
    UNet builds its blocks, and a wrong mapping puts every correction on a
    differently-sized layer -- which mostly fails to load, but not between the
    two 1280-wide regions. Only an end-to-end comparison catches that.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import save_file

    out = output / "ip_adapter"
    out.mkdir(parents=True, exist_ok=True)

    print("loading SD 1.5 + ip-adapter_sd15")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.load_ip_adapter(model_id, subfolder="models", weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(1.0)
    unet = pipe.unet.eval()

    gen = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, 32, 32, generator=gen)
    timestep = torch.tensor([500.0])
    text = torch.randn(1, 77, 768, generator=gen)
    # The *raw* CLIP image embedding, 1024 wide. diffusers runs the adapter's
    # own projection on it, so this reference covers both the projection and
    # the attention wiring rather than assuming the first is right.
    # [batch, images, embed]: the projection is written for several reference
    # images per generation, so it wants the middle axis even for one.
    image_embeds = torch.randn(1, 1, 1024, generator=gen)

    with torch.no_grad():
        image_tokens = unet.encoder_hid_proj([image_embeds])[0]
        out_ip = unet(
            sample, timestep, encoder_hidden_states=text,
            added_cond_kwargs={"image_embeds": [image_embeds]},
        ).sample
        pipe.set_ip_adapter_scale(0.0)
        out_zero = unet(
            sample, timestep, encoder_hidden_states=text,
            added_cond_kwargs={"image_embeds": [image_embeds]},
        ).sample

    tensors = {
        "sample": sample.contiguous(),
        "timestep": timestep.contiguous(),
        "text": text.contiguous(),
        "image_embeds": image_embeds.contiguous(),
        "image_tokens": image_tokens.detach().contiguous().clone(),
        "output": out_ip.detach().contiguous().clone(),
        "output_scale0": out_zero.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(repo_id=model_id, filename="models/ip-adapter_sd15.safetensors")
    link = out / "ip-adapter_sd15.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<14} {tuple(v.shape)}")


def dump_motion(output: pathlib.Path, model_id: str) -> None:
    """One AnimateDiff motion module, in isolation.

    In isolation on purpose: the thing that goes wrong is the permute that
    makes attention temporal, and it produces correct shapes either way. A
    module-level comparison localises that, where a whole-UNet one would only
    say something is off.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import MotionAdapter
    from safetensors.torch import save_file

    out = output / "motion"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    adapter = MotionAdapter.from_pretrained(model_id, torch_dtype=torch.float32)
    adapter.eval()
    module = adapter.down_blocks[0].motion_modules[0]

    frames, channels, size = 4, 320, 8
    gen = torch.Generator().manual_seed(SEED)
    hidden = torch.randn(frames, channels, size, size, generator=gen)

    with torch.no_grad():
        result = module(hidden, num_frames=frames)
    if isinstance(result, tuple):
        result = result[0]

    # And the whole UNet with the adapter attached, which is what catches a
    # wrong *insertion* order — the module comparison above cannot.
    from diffusers import UNet2DConditionModel, UNetMotionModel

    base = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet",
        torch_dtype=torch.float32,
    )
    motion_unet = UNetMotionModel.from_unet2d(base, adapter).eval()

    gen2 = torch.Generator().manual_seed(SEED + 1)
    nframes = 2
    # UNetMotionModel takes [b, c, f, h, w]; this port carries frames on the
    # batch as [b*f, c, h, w]. Both views are dumped, derived from one tensor,
    # so the comparison cannot drift on the layout alone.
    sample5 = torch.randn(1, 4, nframes, 32, 32, generator=gen2)
    sample_flat = sample5.permute(0, 2, 1, 3, 4).reshape(nframes, 4, 32, 32).contiguous()
    timestep = torch.tensor([500.0])
    # One row per *frame*, not per batch entry. UNetMotionModel does not
    # repeat the conditioning itself — passing [1, 77, 768] here fails inside
    # the spatial cross-attention with a 2048-vs-1024 mismatch, because the
    # hidden states carry frames on the batch and the text does not.
    text_flat = torch.randn(nframes, 77, 768, generator=gen2)
    with torch.no_grad():
        out5 = motion_unet(sample5, timestep, encoder_hidden_states=text_flat).sample
    unet_out = out5.permute(0, 2, 1, 3, 4).reshape(nframes, 4, 32, 32).contiguous()

    tensors = {
        "hidden": hidden.contiguous(),
        "output": result.detach().contiguous().clone(),
        "unet_sample": sample_flat,
        "unet_timestep": timestep.contiguous(),
        "unet_text": text_flat,
        "unet_output": unet_out.detach().contiguous().clone(),
    }
    save_file(tensors, str(out / "reference.safetensors"))

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(repo_id=model_id, filename="diffusion_pytorch_model.safetensors")
    link = out / "motion_adapter.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)

    print(f"\\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<8} {tuple(v.shape)}")


def dump_gligen(output: pathlib.Path, model_id: str) -> None:
    """GLIGEN's grounding projection, and the checkpoint converted.

    The UNet ships as a pickled .bin, so this converts it -- the only reason
    the Rust side needs a Python step. The reference itself is `position_net`
    in isolation, because the axis order inside the Fourier embedding is the
    part that is easy to get wrong and produces a working shape either way.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers.models.embeddings import GLIGENTextBoundingboxProjection
    from safetensors.torch import save_file

    out = output / "gligen"
    out.mkdir(parents=True, exist_ok=True)

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(repo_id=model_id, filename="unet/diffusion_pytorch_model.bin")
    print(f"converting {src}")
    state = torch.load(src, map_location="cpu", weights_only=True)
    state = {k: v.contiguous().float() for k, v in state.items()}
    save_file(state, str(out / "gligen_unet.safetensors"))

    net = GLIGENTextBoundingboxProjection(positive_len=768, out_dim=768)
    net.load_state_dict(
        {k[len("position_net."):]: v for k, v in state.items() if k.startswith("position_net.")}
    )
    net.eval()

    gen = torch.Generator().manual_seed(SEED)
    boxes = torch.rand(1, 3, 4, generator=gen)
    # One slot masked off, so the learned nulls are exercised rather than
    # assumed -- a reference where every mask is 1 would not test them.
    masks = torch.tensor([[1.0, 1.0, 0.0]])
    phrases = torch.randn(1, 3, 768, generator=gen)

    with torch.no_grad():
        objs = net(boxes, masks, phrases)

    tensors = {
        "boxes": boxes.contiguous(), "masks": masks.contiguous(),
        "phrases": phrases.contiguous(), "objs": objs.detach().contiguous().clone(),
    }
    # And the whole UNet with grounding applied, which is what catches a fuser
    # in the wrong place — the projection comparison above cannot.
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    ).eval()
    gen2 = torch.Generator().manual_seed(SEED + 2)
    sample = torch.randn(1, 4, 32, 32, generator=gen2)
    timestep = torch.tensor([500.0])
    text = torch.randn(1, 77, 768, generator=gen2)
    with torch.no_grad():
        grounded = unet(
            sample, timestep, encoder_hidden_states=text,
            cross_attention_kwargs={"gligen": {"boxes": boxes, "masks": masks, "positive_embeddings": phrases}},
        ).sample
        plain = unet(sample, timestep, encoder_hidden_states=text).sample

    tensors["unet_sample"] = sample.contiguous()
    tensors["unet_timestep"] = timestep.contiguous()
    tensors["unet_text"] = text.contiguous()
    tensors["unet_grounded"] = grounded.detach().contiguous().clone()
    tensors["unet_plain"] = plain.detach().contiguous().clone()

    save_file(tensors, str(out / "reference.safetensors"))
    print(f"\\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<14} {tuple(v.shape)}")
    print(f"converted {len(state)} tensors")


# One entry per component of the unCLIP checkpoint: (subfolder, published
# name, the name the diffusers layout wants once converted).
#
# Every one ships as a pickled `.bin` -- this repository publishes nothing else
# -- so the whole checkpoint has to be converted before Rust can read any of it.
UNCLIP_COMPONENTS = [
    ("unet", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
    ("vae", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
    ("text_encoder", "pytorch_model.bin", "model.safetensors"),
    ("image_encoder", "pytorch_model.bin", "model.safetensors"),
    ("image_normalizer", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
]

# Fetched alongside the weights so the converted directory is a *complete*
# diffusers checkpoint: `from_pretrained` reads these, and so does the Rust
# side's prediction-type detection.
UNCLIP_CONFIGS = [
    ("unet", "config.json"),
    ("vae", "config.json"),
    ("text_encoder", "config.json"),
    ("image_encoder", "config.json"),
    ("image_normalizer", "config.json"),
    ("scheduler", "scheduler_config.json"),
    ("image_noising_scheduler", "scheduler_config.json"),
    ("feature_extractor", "preprocessor_config.json"),
]


def _convert_pickled(model_id: str, subfolder: str, source: str, dest: pathlib.Path) -> int:
    """Download one pickled `.bin` and rewrite it as f32 safetensors.

    The source is downloaded outside the shared HuggingFace cache and deleted
    **before the safetensors file is written**, not after. That is not
    tidiness: the whole checkpoint is 7.7 GB of `.bin` and 7.7 GB of
    safetensors, and the machine this was written on had 11 GB spare. Unlinking
    after `torch.load` -- by which point the weights are in memory -- means
    peak disk is one copy rather than two. The download is the thing at risk if
    `save_file` then fails, and a download is cheap to repeat.
    """
    torch = _require("torch")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    if dest.exists():
        print(f"  {subfolder}: already converted")
        return 0

    staging = dest.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    print(f"  {subfolder}: downloading {source}")
    # Retried with backoff because the Hub rate-limits (429) a session that has
    # pulled several multi-gigabyte files, and losing a 4 GB download to a
    # transient refusal is a bad trade for four lines.
    src = None
    for attempt in range(6):
        try:
            src = hf_hub_download(
                repo_id=model_id, filename=f"{subfolder}/{source}", local_dir=str(staging)
            )
            break
        except Exception as e:  # noqa: BLE001 - the hub raises several types
            if attempt == 5:
                raise
            delay = 30 * (attempt + 1)
            print(f"  {subfolder}: {type(e).__name__}, retrying in {delay}s")
            time.sleep(delay)
    state = torch.load(src, map_location="cpu", weights_only=True)
    state = {k: v.contiguous().float() for k, v in state.items()}
    pathlib.Path(src).unlink()
    shutil.rmtree(staging, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(dest))
    count = len(state)
    del state
    print(f"  {subfolder}: {count} tensors -> {dest.name}")
    return count


def dump_unclip(output: pathlib.Path, model_id: str) -> None:
    """unCLIP: conditioning on a CLIP *image* embedding.

    Three references, and they answer different questions:

    - `noised_*` is the augmentation in isolation -- normalise, add DDPM noise
      at a chosen level, un-normalise, then append the level's own sinusoid.
      Every step of that is arithmetic on a 1024-vector with no shape to check
      it, so it is dumped at two noise levels and compared directly.
    - `image_embeds` pins that this checkpoint's ViT-H loads into the existing
      vision tower.
    - `unet_out` is the whole UNet with `class_labels`. That is the one that
      matters: the projected embedding is *added into the timestep embedding*,
      so dropping it, projecting it wrong, or adding it in the wrong place all
      run and all return a tensor of exactly the right shape.

    `unet_out_zero` is the guidance batch's unconditional row -- zeros, not an
    absent argument -- and doubles as the control that says the class path
    changes anything at all.
    """
    torch = _require("torch")
    _require("diffusers")
    _require("transformers")
    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    out = output / "unclip"
    out.mkdir(parents=True, exist_ok=True)
    # The converted checkpoint is a model directory, not a bag of tensors: it
    # is what `sdrs unclip --model` is pointed at, so it is laid out that way
    # from the start rather than assembled by hand afterwards.
    models = pathlib.Path("models") / "unclip"

    total = 0
    for subfolder, source, dest_name in UNCLIP_COMPONENTS:
        total += _convert_pickled(model_id, subfolder, source, models / subfolder / dest_name)
    for subfolder, name in UNCLIP_CONFIGS:
        target = models / subfolder / name
        if target.exists():
            continue
        src = hf_hub_download(repo_id=model_id, filename=f"{subfolder}/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    # The repository ships the slow tokenizer (vocab.json + merges.txt) and the
    # Rust side reads `tokenizer.json`. Converting the checkpoint's own rather
    # than borrowing SD 1.5's keeps the model directory self-contained.
    from transformers import CLIPTokenizerFast

    tok_dir = models / "tokenizer"
    if not (tok_dir / "tokenizer.json").exists():
        tok_dir.mkdir(parents=True, exist_ok=True)
        CLIPTokenizerFast.from_pretrained(model_id, subfolder="tokenizer").save_pretrained(
            str(tok_dir)
        )

    print(f"converted {total} tensors into {models}")

    from diffusers import UNet2DConditionModel
    from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import (
        StableUnCLIPImg2ImgPipeline,
    )
    from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import (
        StableUnCLIPImageNormalizer,
    )
    from diffusers import DDPMScheduler
    from transformers import CLIPVisionModelWithProjection

    # Loaded from the converted directory, not from the hub: when a checkpoint
    # exists in two forms, the reference has to come from the one Rust reads.
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(models), subfolder="image_encoder", torch_dtype=torch.float32
    ).eval()
    normalizer = StableUnCLIPImageNormalizer.from_pretrained(
        str(models), subfolder="image_normalizer", torch_dtype=torch.float32
    ).eval()
    noising = DDPMScheduler.from_pretrained(str(models), subfolder="image_noising_scheduler")
    print(f"  noising schedule: {noising.config.beta_schedule}, {noising.config.num_train_timesteps}")

    gen = torch.Generator().manual_seed(SEED)
    # Already CLIP-normalised, like the vision-tower reference: preprocessing is
    # compared separately rather than folded in here.
    pixels = torch.randn(1, 3, 224, 224, generator=gen)
    with torch.no_grad():
        image_embeds = encoder(pixels).image_embeds

    # `noise_image_embeddings` is an instance method that touches exactly two
    # attributes, so it runs against a stand-in -- which means the published
    # implementation is what produced these numbers, not a paraphrase of it.
    stub = argparse.Namespace(image_normalizer=normalizer, image_noising_scheduler=noising)
    noise = torch.randn(1, 1024, generator=gen)

    tensors = {
        "pixels": pixels.contiguous(),
        "image_embeds": image_embeds.detach().contiguous().clone(),
        "noise": noise.contiguous(),
    }
    levels = (0, 250)
    for level in levels:
        with torch.no_grad():
            noised = StableUnCLIPImg2ImgPipeline.noise_image_embeddings(
                stub, image_embeds=image_embeds, noise_level=level, noise=noise
            )
        tensors[f"noised_{level}"] = noised.detach().contiguous().clone()

    unet = UNet2DConditionModel.from_pretrained(
        str(models), subfolder="unet", torch_dtype=torch.float32
    ).eval()
    print(
        f"  class_embed_type {unet.config.class_embed_type}, "
        f"input dim {unet.config.projection_class_embeddings_input_dim}"
    )

    gen2 = torch.Generator().manual_seed(SEED + 2)
    sample = torch.randn(*LATENT_SHAPE, generator=gen2)
    timestep = torch.tensor([500.0])
    text = torch.randn(1, 77, 1024, generator=gen2)
    class_labels = tensors["noised_250"]
    with torch.no_grad():
        conditioned = unet(
            sample, timestep, encoder_hidden_states=text, class_labels=class_labels
        ).sample
        unconditioned = unet(
            sample, timestep, encoder_hidden_states=text, class_labels=torch.zeros_like(class_labels)
        ).sample

    tensors["unet_sample"] = sample.contiguous()
    tensors["unet_timestep"] = timestep.contiguous()
    tensors["unet_text"] = text.contiguous()
    tensors["unet_out"] = conditioned.detach().contiguous().clone()
    tensors["unet_out_zero"] = unconditioned.detach().contiguous().clone()

    save_file(tensors, str(out / "reference.safetensors"))

    # One fixed path per weight file for the test, pointing at the model
    # directory rather than duplicating 7.7 GB.
    for subfolder, _, dest_name in UNCLIP_COMPONENTS:
        link = out / f"{subfolder}.safetensors"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((models / subfolder / dest_name).resolve())

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<16} {tuple(v.shape)}")
    spread = (tensors["unet_out"] - tensors["unet_out_zero"]).abs().max().item()
    print(f"\nclass conditioning moves the output by up to {spread:.4f}")


UNCLIP_PRIOR_COMPONENTS = [
    ("prior", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
    ("prior_text_encoder", "pytorch_model.bin", "model.safetensors"),
    # The image half again, because the text-to-image checkpoint's is *not*
    # the image-variation one -- see the note in `dump_unclip_prior`.
    ("unet", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
    ("vae", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
    ("text_encoder", "pytorch_model.bin", "model.safetensors"),
    ("image_normalizer", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
]

UNCLIP_PRIOR_CONFIGS = [
    ("prior", "config.json"),
    ("prior_text_encoder", "config.json"),
    ("prior_scheduler", "scheduler_config.json"),
    ("unet", "config.json"),
    ("vae", "config.json"),
    ("text_encoder", "config.json"),
    ("image_normalizer", "config.json"),
    ("scheduler", "scheduler_config.json"),
    ("image_noising_scheduler", "scheduler_config.json"),
]


def dump_unclip_prior(output: pathlib.Path, model_id: str) -> None:
    """unCLIP's prior: the model that invents an image embedding from text.

    # Why `-t2i-l` and not `-t2i-h`

    **`diffusers/stable-diffusion-2-1-unclip-t2i-h` is not usable here.** Its
    prior emits a 768-wide ViT-L embedding, while its image half is the ViT-H
    one: `image_normalizer` is 1024 wide and the UNet's
    `projection_class_embeddings_input_dim` is 2048, being twice that. The two
    halves cannot be connected, and the mismatch is in the published configs
    rather than anything this port does. `-t2i-l` is the consistent pairing --
    768 throughout, and a UNet whose class projection takes 1536.

    That is also why this writes `models/unclip-t2i` rather than adding to
    `models/unclip`. The image-variation checkpoint and this one share their
    *prior* (byte-identical between the two `t2i` mirrors) but **not** their
    UNet, VAE or text encoder, all three of which differ by sha256. One
    directory cannot hold both.

    Three references:

    - `prior_out` is the transformer itself, with a **partially masked** text
      sequence. The mask matters and is easy to skip: SD ignores CLIP's
      attention mask everywhere else in this project, and this is the one
      place that does not.
    - `text_embeds` / `text_hidden` pin the prior's own text encoder, which is
      SD 1.5's tower plus a projection head.
    - `stepped` is one DDPM step at `prediction_type="sample"`, which is a
      shape of sampler this project has none of: every other one here is
      sigma-based and predicts noise.
    """
    torch = _require("torch")
    _require("diffusers")
    _require("transformers")
    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    out = output / "unclip"
    out.mkdir(parents=True, exist_ok=True)
    models = pathlib.Path("models") / "unclip-t2i"

    total = 0
    for subfolder, source, dest_name in UNCLIP_PRIOR_COMPONENTS:
        total += _convert_pickled(model_id, subfolder, source, models / subfolder / dest_name)
    for subfolder, name in UNCLIP_PRIOR_CONFIGS:
        target = models / subfolder / name
        if target.exists():
            continue
        src = hf_hub_download(repo_id=model_id, filename=f"{subfolder}/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    from transformers import CLIPTokenizerFast

    for sub in ("prior_tokenizer", "tokenizer"):
        tok_dir = models / sub
        if not (tok_dir / "tokenizer.json").exists():
            tok_dir.mkdir(parents=True, exist_ok=True)
            CLIPTokenizerFast.from_pretrained(model_id, subfolder=sub).save_pretrained(
                str(tok_dir)
            )
    print(f"converted {total} tensors into {models}")

    from diffusers import DDPMScheduler, PriorTransformer
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    prior = PriorTransformer.from_pretrained(
        str(models), subfolder="prior", torch_dtype=torch.float32
    ).eval()
    cfg = prior.config
    print(
        f"  prior: {cfg.num_layers} layers, {cfg.num_attention_heads} heads x "
        f"{cfg.attention_head_dim}, embedding {cfg.embedding_dim}, "
        f"{cfg.num_embeddings} + {cfg.additional_embeddings} tokens"
    )

    encoder = CLIPTextModelWithProjection.from_pretrained(
        str(models), subfolder="prior_text_encoder", torch_dtype=torch.float32
    ).eval()
    tokenizer = CLIPTokenizer.from_pretrained(str(models), subfolder="prior_tokenizer")

    prompt = "a photograph of a crab on a beach"
    tokens = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        encoded = encoder(tokens.input_ids)
    text_embeds = encoded.text_embeds
    text_hidden = encoded.last_hidden_state
    mask = tokens.attention_mask
    print(f"  prompt occupies {int(mask.sum().item())} of {mask.shape[1]} positions")

    gen = torch.Generator().manual_seed(SEED)
    latents = torch.randn(1, cfg.embedding_dim, generator=gen)
    timestep = torch.tensor([500])
    with torch.no_grad():
        predicted = prior(
            latents,
            timestep=timestep,
            proj_embedding=text_embeds,
            encoder_hidden_states=text_hidden,
            attention_mask=mask.bool(),
        ).predicted_image_embedding
        # And again with every position unmasked, so a port that ignores the
        # mask disagrees on one of the two rather than passing both.
        unmasked = prior(
            latents,
            timestep=timestep,
            proj_embedding=text_embeds,
            encoder_hidden_states=text_hidden,
            attention_mask=torch.ones_like(mask).bool(),
        ).predicted_image_embedding

    scheduler = DDPMScheduler.from_pretrained(str(models), subfolder="prior_scheduler")
    print(
        f"  prior scheduler: {scheduler.config.beta_schedule}, "
        f"prediction {scheduler.config.prediction_type}, "
        f"variance {scheduler.config.variance_type}, "
        f"clip {scheduler.config.clip_sample} at {scheduler.config.clip_sample_range}"
    )
    scheduler.set_timesteps(25)
    # `DDPMScheduler.step` draws its own noise and takes no `variance_noise`,
    # so the draw is pinned by replacing the function it calls. The published
    # `step` then runs verbatim -- this is the same trick the augmentation
    # reference uses, and it beats transcribing the arithmetic into the dumper
    # and comparing the port against that.
    import diffusers.schedulers.scheduling_ddpm as ddpm_mod

    step_noise = torch.randn(1, cfg.embedding_dim, generator=gen)
    original_randn = ddpm_mod.randn_tensor
    ddpm_mod.randn_tensor = lambda shape, **kw: step_noise.clone()
    try:
        with torch.no_grad():
            # A step at a *listed* timestep, so the reference exercises the
            # same previous-timestep lookup a real run does, and one at the
            # final timestep, where no variance is added at all.
            t = int(scheduler.timesteps[3])
            stepped = scheduler.step(predicted, t, latents).prev_sample
            t_final = int(scheduler.timesteps[-1])
            stepped_final = scheduler.step(predicted, t_final, latents).prev_sample
    finally:
        ddpm_mod.randn_tensor = original_randn

    # The standard deviations the port has to reproduce, straight from
    # `_get_variance`. `fixed_small_log` returns the *deviation*, not the
    # variance, which `step` then multiplies by the noise with no further
    # square root -- so a port that squares or roots it once more is wrong by
    # exactly that and still produces a plausible embedding.
    probe_timesteps = [int(x) for x in scheduler.timesteps[:4]] + [int(scheduler.timesteps[-1])]
    stds = torch.tensor(
        [float(scheduler._get_variance(t)) for t in probe_timesteps[:-1]] + [0.0]
    )

    tensors = {
        "prior_tokens": tokens.input_ids.to(torch.int64).contiguous(),
        "prior_mask": mask.to(torch.int64).contiguous(),
        "text_embeds": text_embeds.detach().contiguous().clone(),
        "text_hidden": text_hidden.detach().contiguous().clone(),
        "prior_latents": latents.contiguous(),
        "prior_out": predicted.detach().contiguous().clone(),
        "prior_out_unmasked": unmasked.detach().contiguous().clone(),
        "prior_timesteps": scheduler.timesteps.to(torch.int64).contiguous(),
        "step_timestep": torch.tensor([t], dtype=torch.int64),
        "step_timestep_final": torch.tensor([t_final], dtype=torch.int64),
        "step_noise": step_noise.contiguous(),
        "stepped": stepped.detach().contiguous().clone(),
        "stepped_final": stepped_final.detach().contiguous().clone(),
        "probe_timesteps": torch.tensor(probe_timesteps, dtype=torch.int64),
        "probe_stds": stds.contiguous(),
        "clip_mean": prior.clip_mean.detach().contiguous().clone(),
        "clip_std": prior.clip_std.detach().contiguous().clone(),
    }
    # And the join: the prior's output, augmented, through this checkpoint's
    # own UNet. That is the only reference that says the two halves connect --
    # the widths are what `-t2i-h` gets wrong, and everything up to here would
    # pass on that broken pairing too.
    from diffusers import UNet2DConditionModel
    from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import (
        StableUnCLIPImg2ImgPipeline,
    )
    from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import (
        StableUnCLIPImageNormalizer,
    )

    normalizer = StableUnCLIPImageNormalizer.from_pretrained(
        str(models), subfolder="image_normalizer", torch_dtype=torch.float32
    ).eval()
    noising = DDPMScheduler.from_pretrained(str(models), subfolder="image_noising_scheduler")
    stub = argparse.Namespace(image_normalizer=normalizer, image_noising_scheduler=noising)
    image_embeds = prior.post_process_latents(predicted)
    aug_noise = torch.randn(1, cfg.embedding_dim, generator=gen)
    with torch.no_grad():
        class_labels = StableUnCLIPImg2ImgPipeline.noise_image_embeddings(
            stub, image_embeds=image_embeds, noise_level=0, noise=aug_noise
        )

    unet = UNet2DConditionModel.from_pretrained(
        str(models), subfolder="unet", torch_dtype=torch.float32
    ).eval()
    print(
        f"  t2i unet class dim {unet.config.projection_class_embeddings_input_dim} "
        f"against a {cfg.embedding_dim}-wide prior"
    )
    gen2 = torch.Generator().manual_seed(SEED + 2)
    sample = torch.randn(*LATENT_SHAPE, generator=gen2)
    unet_timestep = torch.tensor([500.0])
    text = torch.randn(1, 77, unet.config.cross_attention_dim, generator=gen2)
    with torch.no_grad():
        t2i_out = unet(
            sample, unet_timestep, encoder_hidden_states=text, class_labels=class_labels
        ).sample

    tensors["image_embeds"] = image_embeds.detach().contiguous().clone()
    tensors["aug_noise"] = aug_noise.contiguous()
    tensors["class_labels"] = class_labels.detach().contiguous().clone()
    tensors["t2i_unet_sample"] = sample.contiguous()
    tensors["t2i_unet_timestep"] = unet_timestep.contiguous()
    tensors["t2i_unet_text"] = text.contiguous()
    tensors["t2i_unet_out"] = t2i_out.detach().contiguous().clone()

    save_file(tensors, str(out / "prior_reference.safetensors"))

    for subfolder, _, dest_name in UNCLIP_PRIOR_COMPONENTS:
        link = out / f"t2i_{subfolder}.safetensors"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((models / subfolder / dest_name).resolve())

    print(f"\nwrote {out / 'prior_reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<18} {tuple(v.shape)}")
    moved = (tensors["prior_out"] - tensors["prior_out_unmasked"]).abs().max().item()
    print(f"\nthe attention mask moves the prediction by {moved:.4f}")


def dump_controlnet_sdxl(output: pathlib.Path, model_id: str) -> None:
    """An SDXL ControlNet, which is conditioned like an SDXL UNet.

    The reason this needs its own reference rather than reusing the SD 1.5
    one: an SDXL ControlNet is `addition_embed_type: "text_time"`, so its time
    embedding takes the pooled text embedding and the six time ids on top of
    the timestep. Getting that wrong does not fail — it produces thirteen
    corrections of exactly the right shapes, computed at a timestep embedding
    that means something else, and the image merely comes out wrong.

    Uses the **full** checkpoint, not the distilled `-small` one. The small
    variant is 640 MB and tempting, but it is a different architecture: its
    `down_block_types` are all `DownBlock2D`, `transformer_layers_per_block`
    is `[0, 0, 0]`, and it has no mid block — a purely convolutional
    ControlNet that `UNetConfig::sdxl()` does not describe. The full one
    matches that config field for field, which is the point.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import ControlNetModel
    from safetensors.torch import save_file

    out = output / "controlnet_sdxl"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    net = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    cfg = net.config
    print(
        f"  {cfg.addition_embed_type}, projection {cfg.projection_class_embeddings_input_dim}, "
        f"blocks {list(cfg.block_out_channels)}"
    )

    gen = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, 32, 32, generator=gen)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 77, cfg.cross_attention_dim, generator=gen)
    hint = torch.rand(1, 3, 256, 256, generator=gen)
    pooled = torch.randn(1, 1280, generator=gen)
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]])

    with torch.no_grad():
        res = net(
            sample,
            timestep,
            encoder_hidden_states=context,
            controlnet_cond=hint,
            conditioning_scale=1.0,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            return_dict=True,
        )

    tensors = {
        "sample": sample.contiguous(),
        "timestep": timestep.contiguous(),
        "context": context.contiguous(),
        "hint": hint.contiguous(),
        "pooled": pooled.contiguous(),
        "time_ids": time_ids.contiguous(),
        "mid": res.mid_block_res_sample.detach().contiguous().clone(),
    }
    for i, d in enumerate(res.down_block_res_samples):
        tensors[f"down_{i:02d}"] = d.detach().contiguous().clone()

    save_file(tensors, str(out / "reference.safetensors"))

    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(repo_id=model_id, filename="diffusion_pytorch_model.safetensors")
    link = out / "controlnet.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)

    print(f"\nwrote {out / 'reference.safetensors'}")
    print(f"  {len(res.down_block_res_samples)} down corrections plus mid")


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

    t5 = sub.add_parser("t5", help="dump T5 v1.1 encoder references")
    t5.add_argument("--model-id", default="google/t5-v1_1-small")
    t5.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    llm = sub.add_parser("llm", help="dump Qwen3-family text-encoder references")
    # The smallest of the family. See `dump_llm` for why the size does not
    # matter to what this verifies.
    llm.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    llm.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    fluxt = sub.add_parser("flux_transformer", help="dump Flux MMDiT references")
    fluxt.add_argument("--model-id", default="TencentARC/flux-mini")
    fluxt.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    fs = sub.add_parser("flux_sampling", help="dump a 20-step Flux loop reference")
    fs.add_argument("--model-id", default="TencentARC/flux-mini")
    fs.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    sd3 = sub.add_parser("sd3", help="dump SD 3.5 MMDiT references")
    sd3.add_argument("--model-id", default="adamo1139/stable-diffusion-3.5-medium-ungated")
    sd3.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

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

    gl = sub.add_parser("gligen", help="dump GLIGEN grounding references")
    gl.add_argument("--model-id", default="masterful/gligen-1-4-generation-text-box")
    gl.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    mo = sub.add_parser("motion", help="dump AnimateDiff motion module references")
    mo.add_argument("--model-id", default="guoyww/animatediff-motion-adapter-v1-5-2")
    mo.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    ipa = sub.add_parser("ip_adapter", help="dump IP-Adapter UNet references")
    ipa.add_argument("--model-id", default="h94/IP-Adapter")
    ipa.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    cv = sub.add_parser("clip_vision", help="dump CLIP vision tower references")
    cv.add_argument("--model-id", default="h94/IP-Adapter")
    cv.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    esr = sub.add_parser("esrgan", help="dump Real-ESRGAN x4 references")
    esr.add_argument("--model-id", default="ai-forever/Real-ESRGAN")
    esr.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    tae = sub.add_parser("taesd", help="dump TAESD tiny autoencoder references")
    tae.add_argument("--model-id", default="madebyollin/taesd")
    tae.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    cnet = sub.add_parser("controlnet", help="dump ControlNet correction references")
    cnet.add_argument("--model-id", default="lllyasviel/sd-controlnet-canny")
    cnet.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

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

    uc = sub.add_parser("unclip", help="dump unCLIP image-conditioning references")
    uc.add_argument("--model-id", default="diffusers/stable-diffusion-2-1-unclip-i2i-h")
    uc.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    ucp = sub.add_parser("unclip_prior", help="dump unCLIP text-to-image prior references")
    # `-t2i-l`, not `-t2i-h`: the latter pairs a 768-wide prior with a
    # 1024-wide image half and cannot run. See `dump_unclip_prior`.
    ucp.add_argument("--model-id", default="diffusers/stable-diffusion-2-1-unclip-t2i-l")
    ucp.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    cns = sub.add_parser("controlnet_sdxl", help="dump SDXL ControlNet references")
    cns.add_argument("--model-id", default="diffusers/controlnet-canny-sdxl-1.0")
    cns.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    gg = sub.add_parser("gguf", help="link small real GGUF files for the header tests")
    gg.add_argument("--model-id", default="", help="unused")
    gg.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tests/golden"))

    args = ap.parse_args()
    if args.component == "gligen":
        dump_gligen(args.output, args.model_id)
    elif args.component == "unclip":
        dump_unclip(args.output, args.model_id)
    elif args.component == "controlnet_sdxl":
        dump_controlnet_sdxl(args.output, args.model_id)
    elif args.component == "unclip_prior":
        dump_unclip_prior(args.output, args.model_id)
    elif args.component == "motion":
        dump_motion(args.output, args.model_id)
    elif args.component == "ip_adapter":
        dump_ip_adapter(args.output, args.model_id)
    elif args.component == "clip_vision":
        dump_clip_vision(args.output, args.model_id)
    elif args.component == "esrgan":
        dump_esrgan(args.output, args.model_id)
    elif args.component == "taesd":
        dump_taesd(args.output, args.model_id)
    elif args.component == "controlnet":
        dump_controlnet(args.output, args.model_id)
    elif args.component == "gguf":
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
    elif args.component == "t5":
        dump_t5(args.output, args.model_id)
    elif args.component == "llm":
        dump_llm(args.output, args.model_id)
    elif args.component == "flux_transformer":
        dump_flux_transformer(args.output, args.model_id)
    elif args.component == "flux_sampling":
        dump_flux_sampling(args.output, args.model_id)
    elif args.component == "sd3":
        dump_sd3(args.output, args.model_id)
    elif args.component == "vae":
        dump_vae(args.output, args.model_id)
    elif args.component == "clip_tokenizer":
        dump_clip_tokenizer(args.output, args.model_id)
    elif args.component == "clip_encoder":
        dump_clip_encoder(args.output, args.model_id)


if __name__ == "__main__":
    main()
