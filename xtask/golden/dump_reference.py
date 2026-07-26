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
import pathlib
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

    print(f"\nwrote {out / 'reference.safetensors'}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<18} {tuple(v.shape)}")
    print(f"\nwrote {out / 'vae.safetensors'} ({len(weights)} tensors)")
    print(
        "\nIntermediate tensors are included on purpose: when the final image "
        "\nmismatches, they tell you *which block* diverged first."
    )


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

    args = ap.parse_args()
    if args.component == "vae":
        dump_vae(args.output, args.model_id)


if __name__ == "__main__":
    main()
