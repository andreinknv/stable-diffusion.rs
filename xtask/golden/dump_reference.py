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

    args = ap.parse_args()
    if args.component == "vae":
        dump_vae(args.output, args.model_id)
    elif args.component == "clip_tokenizer":
        dump_clip_tokenizer(args.output, args.model_id)
    elif args.component == "clip_encoder":
        dump_clip_encoder(args.output, args.model_id)


if __name__ == "__main__":
    main()
