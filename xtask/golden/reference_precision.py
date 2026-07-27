#!/usr/bin/env python3
"""Measure the reference implementation's own f32-vs-f64 spread.

A golden test's tolerance answers "how far apart may two correct
implementations be?", and the only defensible way to set it is to ask the
reference that question about *itself*. Run the same diffusers module twice,
once in float32 and once in float64, on identical inputs. Neither run has a
bug; the difference between them is the floating-point noise floor of the
computation. A tolerance below that floor does not test correctness, it pins
one summation order — which is how `--features accelerate` came to fail two
UNet tests at 1.087e-4 against a 1.000e-4 bound while agreeing to 6.7e-6
*relative* on tensors that peak at 16.

Usage:

    python3 xtask/golden/reference_precision.py unet
    python3 xtask/golden/reference_precision.py vae

Prints, per captured tensor, the absolute and relative spread, so a tolerance
can be quoted with the measurement that justifies it rather than chosen to
make a failing test pass.
"""

import argparse
import sys

SEED = 0


def _require(name):
    try:
        return __import__(name)
    except ImportError:
        sys.exit(f"error: {name} is required; pip install torch diffusers")


def spread(a, b):
    """Absolute and relative divergence between an f32 and an f64 run.

    `b` is the f64 result and is treated as the truth. Relative error is
    against the tensor's own peak magnitude rather than per element: a
    per-element ratio explodes wherever the true value is near zero and
    reports noise as catastrophe.
    """
    import torch

    a64 = a.double()
    diff = (a64 - b).abs()
    peak = b.abs().max().item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "peak": peak,
        "max_rel": diff.max().item() / peak if peak > 0 else 0.0,
    }


def report(name, s):
    print(
        f"  {name:14} peak {s['peak']:10.3f}   max_abs {s['max_abs']:.3e}   "
        f"mean_abs {s['mean_abs']:.3e}   max_rel {s['max_rel']:.3e}"
    )


def unet(model_id):
    torch = _require("torch")
    _require("diffusers")
    from diffusers import UNet2DConditionModel

    # The same inputs `dump_unet_full` uses, so the numbers below describe the
    # tensors the golden test actually compares rather than a similar shape.
    gen = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, 32, 32, generator=gen)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 77, 768, generator=gen)

    results = {}
    for dtype in (torch.float32, torch.float64):
        print(f"running the reference UNet in {dtype}")
        net = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float32
        )
        net = net.to(dtype)
        net.eval()

        skips = []
        mid = {}

        def push(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            skips.append(t.detach().contiguous().clone())

        handles = [net.conv_in.register_forward_hook(push)]
        for block in net.down_blocks:
            for j, _ in enumerate(block.resnets):
                target = (
                    block.attentions[j]
                    if getattr(block, "attentions", None) is not None
                    and j < len(block.attentions)
                    else block.resnets[j]
                )
                handles.append(target.register_forward_hook(push))
            if getattr(block, "downsamplers", None):
                handles.append(block.downsamplers[0].register_forward_hook(push))

        def capture_mid(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            mid["mid_output"] = t.detach().contiguous().clone()

        handles.append(net.mid_block.register_forward_hook(capture_mid))

        with torch.no_grad():
            result = net(
                sample.to(dtype), timestep.to(dtype), encoder_hidden_states=context.to(dtype)
            ).sample
        for h in handles:
            h.remove()

        results[dtype] = {
            "output": result.detach().contiguous().clone(),
            **mid,
            **{f"down_{i:02d}": t for i, t in enumerate(skips)},
        }
        del net

    f32, f64 = results[torch.float32], results[torch.float64]
    print("\nreference f32 against its own f64, same weights and inputs:")
    worst_abs, worst_rel = 0.0, 0.0
    for name in f64:
        s = spread(f32[name], f64[name])
        report(name, s)
        worst_abs = max(worst_abs, s["max_abs"])
        worst_rel = max(worst_rel, s["max_rel"])
    print(
        f"\nworst across all captured tensors: max_abs {worst_abs:.3e}, "
        f"max_rel {worst_rel:.3e}"
    )
    print(
        "\nA tolerance below these numbers is measuring float32, not the port.\n"
        "Quote them where the tolerance is set."
    )


def vae(model_id):
    torch = _require("torch")
    _require("diffusers")
    from diffusers import AutoencoderKL

    # The same inputs `dump_vae` uses for the encoder half.
    image = torch.randn(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    latent = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(SEED))

    results = {}
    for dtype in (torch.float32, torch.float64):
        print(f"running the reference VAE in {dtype}")
        net = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(dtype)
        net.eval()
        with torch.no_grad():
            moments = net.quant_conv(net.encoder(image.to(dtype)))
            decoded = net.decoder(net.post_quant_conv(latent.to(dtype)))
        results[dtype] = {
            "encoder_moments": moments.detach().contiguous().clone(),
            "decoded_image": decoded.detach().contiguous().clone(),
        }
        del net

    f32, f64 = results[torch.float32], results[torch.float64]
    print("\nreference f32 against its own f64, same weights and inputs:")
    for name in f64:
        report(name, spread(f32[name], f64[name]))
    print(
        "\nA tolerance below these numbers is measuring float32, not the port.\n"
        "Quote them where the tolerance is set."
    )


def taesd(_model_id):
    """All four TAESD checkpoints at once: their floors differ by 80x.

    Worth running as a set rather than one at a time, because that spread is
    the finding. `taef1` is two orders of magnitude noisier than `taesd3` in
    f32 despite being the same architecture, so a single tolerance chosen from
    any one of them is wrong for the others.
    """
    torch = _require("torch")
    _require("diffusers")
    from diffusers import AutoencoderTiny

    gen = torch.Generator().manual_seed(SEED)
    latent16 = torch.randn(1, 16, 32, 32, generator=gen)
    image = torch.rand(1, 3, 256, 256, generator=gen) * 2 - 1
    latent4 = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(SEED))

    for name in ("taesd", "taesdxl", "taesd3", "taef1"):
        print(f"\n{name}:")
        results = {}
        for dtype in (torch.float32, torch.float64):
            net = AutoencoderTiny.from_pretrained(
                f"madebyollin/{name}", torch_dtype=torch.float32
            ).to(dtype)
            net.eval()
            latent = latent16 if net.config.latent_channels == 16 else latent4
            with torch.no_grad():
                results[dtype] = {
                    "encoded": net.encoder(image.to(dtype)).detach().clone(),
                    "decoded": net.decoder(latent.to(dtype)).detach().clone(),
                }
            del net
        f32, f64 = results[torch.float32], results[torch.float64]
        for key in f64:
            report(key, spread(f32[key], f64[key]))
    print(
        "\nA tolerance below these numbers is measuring float32, not the port.\n"
        "Quote them where the tolerance is set."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("component", choices=["unet", "vae", "taesd"])
    ap.add_argument(
        "--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    args = ap.parse_args()
    if args.component == "unet":
        unet(args.model_id)
    elif args.component == "vae":
        vae(args.model_id)
    elif args.component == "taesd":
        taesd(args.model_id)


if __name__ == "__main__":
    main()
