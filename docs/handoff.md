# Handoff

Written 2026-07-26. Say **"go"** to resume: read this file, pick the top
unstarted item under [Next](#next), and start it.

## Where things stand

Four architectures render, and **Metal now produces the right image for every
one that fits** — the Flux corruption is fixed (see the trap on storage
offsets below). Metal is 6-9x faster across the board, so it is now the
sensible default rather than a broken option.

| model | CPU | Metal | agreement |
|---|---|---|---|
| SD 1.5 512, 20 steps | 113 s | **17.5 s** | max 1/255, 98.8% of pixels exact |
| SDXL 1024, 20 steps | — | **86.5 s** | verified this session; `assets/sdxl-crab-1024-metal-f16.png` |
| Flux schnell (12B) 512, 4 steps | 159 s | **20.8 s** | mean 9.2/255, same image |
| SD 3.5 medium 512, 20 steps | 230 s (dense) | **24.5 s** † | now Q4_K_M by default |
| SD 3.5 medium 256, 20 steps | 71.8 s (dense) | **9.0 s** | mean 7.0/255, same image |
| Flux mini (3.2B) 512, 20 steps | 212 s | does not fit ‡ | — |

† **Now the quantised checkpoint, and reliable.** `sd3_paths_in` prefers
`sd35-medium-q4_k_m.gguf` (1.79 GB) over the dense `.safetensors` (10.2 GB at
f32) exactly as Flux's `paths_in` does. Loading drops from 14.7 s to ~4 s, and
512 renders in 24.5 s on a quiet machine — and, more to the point, it keeps
working under load, where the dense build died in denoise **step 1** leaving
1.1 GB free of 36 GB.

That step-1 failure was recorded here for several sessions as a *VAE decode*
failure, which was wrong: candle queues Metal work and reports a failed
command buffer only at the next synchronisation point, so the blame lands on
whatever waits first. See the trap below.

‡ Flux mini holds dense f32 weights: 12.8 GB for a 3.2B model, against
schnell's 6.8 GB for 12B held as Q4_K. Quantisation is what makes the *larger*
model the one that runs — worth remembering before reaching for a dense
checkpoint. Unlike SD 3.5, no quantised flux-mini has been published, so this
one has no equivalent fix available.

**Quantised is not free on CPU**, and the default now takes that trade
knowingly: SD 3.5 at 256 on CPU is 93 s quantised against 72 s dense, because
candle's CPU quantised matmul quantises the activation per call where a dense
f32 matmul just runs gemm. It is the right default anyway — Metal is 6-9x
faster than either and the quantised form is what fits there — but if you are
benchmarking CPU specifically, point `Sd3Paths::transformer` at the
`.safetensors` and expect the dense numbers.

**The VAE decode now sizes its own tile.** `decode_tiled` picks the largest
edge from 64 down whose projected peak fits the memory that is actually free,
so a decode that would not fit degrades to a seamed-but-correct image instead
of dying. `SD_VAE_TILE_LATENT` still overrides it outright. Where 64 already
fits, nothing changes — SD 1.5 on Metal is bit-identical before and after.

**Each pipeline stage can run on its own device.** `pipeline::Placement` is
caller-supplied policy — `Placement::on(&gpu).with_text_encoders_on(&Device::Cpu)`
— passed to `load_with_placement` on Flux and SD 3. `Placement::auto` picks
one from projected residency against free memory. The default is unchanged
(everything on one device), so nothing moves unless asked.

Measured on SD 3.5 at 512, encoders moved to the CPU: **4.4 GB freed on the
accelerator** (9.84 GB available after load against 14.24), for about 8 s of
one-time CPU text encoding. The images agree at mean 4.1/255, which is f32
reduction order in the encoders, not a defect.

**That trade is much better on a discrete GPU than it looks here**, and that
is the reason it exists: on unified memory the bytes come from one pool either
way, while on an 8-12 GB card 4.4 GB is often the difference between running
and not — and the weights never cross PCIe at all. The mechanism is the same;
this machine measures its weakest case. `stable-diffusion.cpp` reached the
same design: `backend`, `params_backend`, `max_vram` and `split_mode` are
fields of its public `sd_ctx_params_t`, not CLI flags.

**The diffusion model's blocks can stream.** `Placement::with_streamed_diffusion()`
keeps a quantised transformer's blocks in host memory and copies each to the
compute device as it is reached, releasing it after — `stable-diffusion.cpp`'s
`--offload-to-cpu`, and the answer for a GPU too small to hold the model at
all, where no static placement helps. Flux only, so far.

Output is **bit-identical** to the resident path — the copy moves quantised
block bytes verbatim, so nothing is rounded twice.

**How much it costs depends on how tightly you want the memory held**, and
that is a real dial rather than a tuning detail. Dropping a block frees
nothing on Metal by itself: candle pools its buffers and returns them only
inside `drop_unused_buffers`, which runs on synchronise. So the interval
between synchronises *is* the peak residency. On Flux schnell, 512, 4 steps,
against 20.9 s resident:

```text
  sync every  1    29.5 s    ~1 block resident   (191 MB)   default
  sync every  4    26.6 s
  sync every  8    25.2 s
  sync every 19    25.3 s    whole stack pools   (3.6 GB)
```

`SD_STREAM_SYNC_EVERY` sets it. The default is 1 because the reason to stream
is the memory.

**An earlier version of this note claimed 25.1 s and "2.4 GB freed" — that was
measured before the synchronise existed**, so the pool was growing and the run
was not holding the memory it claimed. Two lessons: leaving the synchronise
out is not merely untidy — with no release at all a 25 s run degraded past
60 s *per step* as the machine started swapping; and a memory claim measured
without checking when memory is actually returned is not a measurement.

`quantized::to_device` is the primitive; `FluxTransformer::resident_bytes`
reports 0 for streamed blocks, because saying otherwise would report the
opposite of what streaming achieves.

**One-shot runs release before decoding.** `FluxPipeline::run_releasing` and
`Sd3Pipeline::run_releasing` consume the pipeline, drop everything the decode
does not need, and then synchronise — on Metal a drop alone frees *nothing*,
because candle pools its buffers and only returns them inside
`drop_unused_buffers`, which runs on synchronise. Measured worth: about
0.7 GB for SD 3.5. Real, and far less than hoped; see the table note above for
what actually dominates.

Every component is verified against `diffusers`/`transformers` — the full
table is in [roadmap.md](roadmap.md). 332 tests, all gates green
(`cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
`scripts/check-seam.sh`, `scripts/check-native-deps.sh`).

**LCM sampling works**, `--sampler lcm`, in both SD 1.5 and SDXL. It is not
another ODE solver: a consistency model maps any point on the trajectory
straight to its origin, so the loop jumps to `x0` and re-noises out to the next
sigma rather than integrating. Four steps instead of twenty.

With `lcm-lora-sdv1-5` at 4 steps and `--cfg-scale 1.0`: a clean photographic
crab in **31 s**, against 57 s for 8 steps of `dpmpp2m` that came out blown
out. Two controls make it a demonstration rather than a hope — the LCM sampler
*without* the adapter gives a brown blob, and the adapter without the LCM
sampler blows out. Each piece is necessary.

Two things make it look broken if missed, both documented on the module:
guidance must be ~1, because the distillation folded one in already; and the
timesteps are a fixed subset of the distillation ladder
(`[999, 759, 519, 279]` at four steps), not an even spread.

**AnimateDiff motion modules are ported, wired and verified** — the module at
**2.662e-7**, and the whole UNet with all 21 inserted at **5.440e-6**.
`UNet2DConditionModel::new_with_motion` installs them through the same
construction-scoped thread-local `unet::ip` uses, one after each resnet, and
refuses unless all 21 are consumed.

The end-to-end comparison is the one that matters: *where* each module lands is
invisible to a per-module check, since every insertion order keeps every shape
valid. The order turned out right — down, then mid, then up, matching this
UNet's own construction order, unlike the IP-Adapter's index list.

A motion module is a small transformer whose attention runs across *frames*
rather than pixels, and our own `Attention` and `FeedForward` are reused
verbatim — the checkpoint's names match theirs exactly. **21 of them, one per
resnet**: down 2x4, mid 1, up 3x4. Note `down_blocks.3` has two despite having
no attentions, which is the clue that they attach to resnets rather than to
transformers.

Three things were wrong on the first attempt and **every one of them kept all
shapes valid**, which is why this was verified numerically rather than by
inspection:

- **The positional encoding goes on the normed states inside each attention
  path, and twice** — once before `attn1`, once before `attn2`. Adding it once
  to the residual stream is the natural reading and is wrong by ~3.
- **The GroupNorm spans frames.** `[b*f, c, h, w]` is regrouped to
  `[b, c, f, h, w]` first, so each group's statistics are taken over the whole
  clip. Normalising per frame is wrong by ~3.
- **The temporal permute happens once, in the module**, not per block: the
  blocks receive `[b*h*w, f, c]` already.

Frame count is ambient (`motion::with_frames`), for the same reason the
IP-Adapter's weights are: it must reach 21 modules and is uniform. The UNet has
no frame axis — frames ride on the batch.

**Two-pass generation works**, `--hires 1024`, and the failure it fixes is
visible rather than theoretical. SD 1.5 asked to compose at 1024 directly
produced **three knights** where one was asked for
(`assets/hires-off-1024-duplicates.png`); composing at 512 and refining at
1024 gives one, sharp (`assets/hires-on-512to1024.png`) — and is *faster*,
93.8 s against 115.1 s, because the first pass runs at the smaller size.

Three upscale modes between the passes, and the choice is not cosmetic:
latent-nearest (default), latent-bilinear, and pixel-lanczos, which is the
only one that pays for a VAE round trip. **Nearest introduces no colours that
were not already there** — every interpolating mode invents intermediate
values, which is right for photographic work and destructive for anything with
a fixed palette.

The second pass draws from `seed + 1`, so the two passes do not draw the same
noise for differently-sized latents.

**AnimateDiff renders**, `--motion-adapter <file> --frames 16`. Verified
through the UNet at the pipeline's own shapes (1.125e-5), and the output now
tracks the reference — see [Next](#next) for the beta-schedule trap that made
this look broken for a long time.

**GLIGEN's grounding projection is ported and verified** at **7.629e-6** — the
half that turns `(box, phrase)` pairs into tokens. The gated attention that
consumes them is not done; see [Next](#next).

Boxes are expanded into sinusoids at eight frequencies, the timestep-embedding
trick applied to space. **The axis order is the whole subtlety**: the axes are
`(coordinate, frequency, sin/cos)` and flatten as
`(frequency, sin/cos, coordinate)`. Every permutation produces 64 numbers and
loads against the same weights; only one lines up with what the MLP was
trained on, and the rest give grounding tokens that are wrong without being
malformed. That is what the golden comparison is for.

**A masked slot uses a *learned* null, not zeros** — both for the phrase and
the position. Padding with zeros reads as "a phrase whose embedding happens to
be zero", which is a different thing and one the model was never shown. The
reference masks one of three slots off so this is exercised rather than
assumed, and a second test pins that a masked slot does not depend on its
contents.

The checkpoint ships as a pickled `.bin`, so `dump_reference.py gligen`
converts the whole UNet to safetensors as a side effect — 966 tensors.

**Instruction editing works**, `sdrs instruct --prompt "make it winter with
snow" --init-image X`. Different from img2img: that takes a description of the
*result* plus a strength, this takes a description of the **change**, and the
source is held by the model's own conditioning rather than by stopping the
schedule early. `assets/instructpix2pix-crab-winter.png` — the same crab, in
the same pose, in a snowed-over scene.

**Three predictions per step, not two.** Ordinary guidance contrasts a prompt
against nothing; this contrasts three things so that instruction adherence and
image fidelity become independent axes:

```text
  pred = uncond + text_scale * (text - image) + image_scale * (image - uncond)
```

The rows are `[text+image, uncond+image, uncond+zeros]`, and the **zeroed
image latent in the third row** is what makes the middle term mean "what the
image contributes" rather than "what the prompt contributes".

`--image-guidance` is measurably a second axis — mean distance from the source
at 1.0 / 1.5 / 2.5 is **65.7 / 44.4 / 30.6**, monotone, and there is a test
that checks all three rather than one (one would pass with the parameter
ignored, two could pass on noise).

**Two traps, both silent.** The 8-channel `conv_in` is detected from its own
shape, since it is invisible in the cross attention that identifies SD 2.x.
And the source latent is **not scaled by 0.18215** — every other latent here
is, but InstructPix2Pix was trained on the raw encoder output, and scaling it
multiplies the conditioning by 5.5 and returns a plausible image that ignores
the source.

**Regional prompts work**, `sdrs area --region mask.png=prompt` (repeatable).
Each region contributes its own noise prediction, blended by its mask —
composed *before* sampling, so the regions see each other rather than being
generated separately and composited, which leaves visible joins.

**The control is real but soft, and that is inherent.** Two complementary
half-masks with irreconcilable prompts ("red brick wall" over "green grass")
put brick where the mask says and ground below
(`assets/area-conditioning-brick-over-ground.png`), but the model still
resolves the whole frame into one coherent scene. It has to: each step blends
predictions, then the *next* step's UNet sees a single latent and re-imposes
global structure. Anyone expecting a hard partition will be disappointed, and
that is a property of prediction-blending rather than of this implementation.

Two invariants are exact and tested, and they are what pin the normalisation:
an **empty mask is bit-identical to a plain run**, and a **full mask is
bit-identical to a plain run of that region's prompt**. Either would break if
the base leaked in at the wrong weight.

**The mask is mean-pooled to the latent grid, not max-pooled** — the opposite
of `latent_mask` for inpainting, and deliberately. An inpaint needs a cell
freed if *any* pixel under it is free; a region boundary should fade across
the cell it straddles. Reusing the inpaint pooling would give every region an
8-pixel-wide hard edge.

Cost is one UNet call per region per step on top of the base, which is
inherent to conditioning spatially.

**A clip is a batch of frames**, `--frames 8`. The pipeline no longer assumes
batch 1: the latent draw, the guidance concatenation, the timestep tensor and
the sampler's noise draws all follow the frame count, and `--output clip.png`
writes `clip-000.png`, `clip-001.png`, ...

**The frame count is read from the latent**, not the config, inside the loop.
That is the one thing every path already agrees on — and a caller supplying
its own latent through `run_with_latent` sets the count by doing so, which is
exactly what the coherence techniques in the animation issue need.

Three things this had to get right, each of which runs when wrong:

- **Conditioning is per frame.** The guidance batch is
  `[uncond x n, cond x n]`, not interleaved, and the reference UNet does not
  repeat it for you — one row where `n` are expected fails inside the *spatial*
  cross-attention. Interleaving instead runs and guides each frame by another
  frame's conditioning.
- **The guidance split is `narrow(0, 0, n)` / `narrow(0, n, n)`**, matching how
  the batch was concatenated.
- **The VAE decodes one frame at a time.** Frames are independent through it,
  so decoding `n` together only multiplies the largest single allocation — a
  three-frame 512 decode is 6.8 GiB in one call and trips the memory guard.
  Looping is byte-identical at one frame's peak, and there is a test that says
  so.

`frames: 1` is bit-identical to the previous behaviour, which is the property
that matters most here since every still-image path runs through this loop.

Motion modules pick the count up automatically — `denoise_inner` installs
`motion::with_frames` for the run.

**Checkpoints merge**, `sdrs merge --a X --b Y --alpha 0.3`. Loader-level
arithmetic — `(1-alpha)*a + alpha*b` per tensor, on the CPU regardless of the
active device, since moving gigabytes onto an accelerator to add them pays a
transfer for no gain.

What it refuses is the point. Merging is only meaningful within one
architecture, and the failure otherwise is quiet: an SD 1.5 and an SDXL share
enough tensor *names* to produce a file that loads and renders noise. So shape
mismatches and one-sided tensors are both refused — with a count and an
example — rather than skipped, since skipping silently takes one side's
weights and yields a third model nobody asked for. `--allow-unmatched` opts
back in.

Verified on real weights: merging the SD 1.5 VAE with itself at alpha 0.5 is
**bit-identical** across all 248 tensors, and merging it with the UNet is
refused naming `conv_in.bias`.

**Textual inversion works**, `--embedding <file>` (repeatable), triggered by
the file stem. Kilobytes against a checkpoint's gigabytes, which is the whole
point of it.

**A learned embedding has no token id**, so it is *spliced*, not looked up.
The trigger is tokenised like any other word — all it does is reserve
positions — and after the embedding lookup those rows are overwritten with the
learned vectors. Multi-vector embeddings expand the trigger to as many copies
as they have vectors, so a short trigger cannot silently drop the tail.

That needed `embed_tokens` and `forward_embeds` split apart on the text
encoder, with the splice landing *between* them: position embeddings are added
after, because a learned vector occupies a position like any other token.

Three tests, and each catches something the others do not: a prompt naming the
trigger must differ from the same prompt without the embedding loaded; a
prompt *not* naming it must be **bit-identical**, which is what fails if the
splice writes to arbitrary positions; and a wrong-width embedding is refused,
since SD 2.x's 1024 in an SD 1.5 prompt would otherwise be a shape error from
inside the transformer.

Three file layouts are accepted (`emb_params`, `string_to_param.*`, and a bare
single-tensor file) because a user downloading one has no reason to know which
tool wrote it.

**IP-Adapter works**, `sdrs txt2img --ip-adapter <ckpt> --image-encoder <dir>
--ip-image <img>`, verified against diffusers at **2.267e-6** through the whole
UNet and **1.016e-6** at scale 0.

The A/B that shows it: same prompt ("a photograph of a castle on a hill"),
same seed, a crab photograph as reference. `--ip-scale 0` gives a castle;
`--ip-scale 1` gives a crab on sand — the reference overrides the prompt
entirely at full strength, which is why published guidance uses 0.5-0.7. Both
in `assets/`.

**Decoupled cross-attention, not concatenation.** Each of the sixteen cross-
attention layers gains a second key/value pair and returns
`attn(text) + scale * attn(image)`, with `to_out` applied once to the sum.
Appending the image tokens to the text ones is a *different function* —
attention is not linear in K and V — and a plausible-looking one.

**The image tokens ride on the end of the context tensor** and the attention
layers split them off. That convention is why nothing between the pipeline and
the attention needed a new parameter: no block type, no transformer, no UNet
signature changed.

**The weights reach sixteen layers without sixteen parameters either.** The
source is installed thread-locally for the duration of
`UNet2DConditionModel::new_with_ip` and each cross-attention pulls the next
slot as it is built — construction-scoped, released by a guard, never read
outside that one call. It refuses if the count does not come out exactly at
sixteen, because consuming too few would leave the deepest layers
unconditioned and still render.

**The index order was the risk, and it is pinned.** The checkpoint numbers its
entries by diffusers' flat processor order — down blocks, up blocks, then
**mid** — while this UNet builds down, **mid**, up. Entries sit at *odd*
indices because that list alternates self- and cross-attention. So slot `i`
maps to key `2 * order[i] + 1` with
`order = [0..5, 15, 6..14]`. A wrong mapping mostly fails to load, but
*between the two 1280-wide regions it would not*, and the image would simply
be wrong — which is why the verification is end-to-end through the UNet rather
than per module.

The strength is thread-local rather than a process global, unlike
`conv::seamless`: two threads may want different strengths, and a shared one
made two tests race.

**CLIP's vision tower is ported and verified** — 1.270e-4 on the sequence,
3.485e-6 on the pooled vector, against `transformers`. This is IP-Adapter's
foundation, and it is the part that was missing; the adapter itself is next
(see [Next](#next)).

Structurally it is the text tower with a different front end, and
`ClipEncoderLayer` is reused literally. Three things differ and each is a
silent-wrongness bug if missed:

- **No causal mask.** An image has no order to respect. The layer takes a mask
  either way, so this passes zeros — the additive identity for attention
  logits. Reusing the text mask would let each patch see only those before it
  in raster order and still emit the right shape.
- **The class token is prepended**, because the pooled output reads position 0.
- **`pre_layrnorm`** is spelled that way in the checkpoint. A typo upstream,
  now load-bearing.

**IP-Adapter consumes the *projected* embedding, not the pooled one.** The
tower is 1280 wide and `visual_projection` narrows it to 1024, which is what
`image_proj` expects — `image_embeds` is a separate method from `pooled` for
that reason. The widths differ, so the mistake fails to load rather than
running wrong.

The adapter's own layout is confirmed: 4 image tokens from
`Linear(1024 -> 4*768)` plus a LayerNorm, and 16 `to_k_ip`/`to_v_ip` pairs at
**odd** indices 1..31 — odd because the flat processor list alternates self-
and cross-attention and only cross-attention gets them. Their order is
**down blocks, then up blocks, then mid**, read off the shapes
(320,320,640,640,1280,1280 | 1280x3, 640x3, 320x3 | 1280). That ordering is
worth having written down: it is not the order the UNet runs in, and guessing
it wrong would put every correction on the wrong layer while every shape stayed
valid.

**Cancellation, a per-step conditioning hook, and prompt-budget queries** —
the remaining asks from the integration issue.

`Cancel` is a token on the config rather than a callback return value, so the
ordinary `ProgressFn` stays a plain `FnMut` and callers who never cancel write
nothing. Checked at the top of each step, and the error names how far it got.

`run_conditioned` takes a slice of pre-encoded `Conditioning` and a
`(step, total) -> index` selector. That covers two asks at once: encode once
and reuse across a sequence, and *vary* the conditioning per step. The
motivating result is gating a negative prompt to a middle window of the
schedule — 65.1 % to 80.4 % on object removal (Ban et al., ECCV 2024) — which
is not expressible with one fixed conditioning. A single-entry slice with a
constant selector is bit-identical to the plain run, and there is a test that
says so; a second test gates a negative to a window and asserts the image
*changes*, which is what fails if the selector is ignored.

`ClipTokenizer::content_token_count`, `content_capacity` and `will_truncate`
answer the budgeting question. The counts are unintuitive and now pinned:
`"16-bit"` is **four** tokens and `"32x32"` is **five**, because CLIP's BPE
splits digits singly, and every comma costs one. Over the limit this
implementation **truncates, not chunks** — a term past the boundary is
discarded rather than encoded into a second window, which is a real difference
from `stable-diffusion.cpp` and is now tested rather than left to be
discovered.

**Explicit latent in and out**, `initial_latent` and `run_with_latent`, which
is what makes frame-to-frame coherence reachable by a caller: shared initial
latents across frames, correlated noise, interpolation between keyframes,
carrying a latent forward. None of that is expressible through a seed.

Two tests hold it together. `initial_latent` fed back to `run_with_latent`
must reproduce the seeded run **bit-identically** — so the initial draw still
happens even when a latent is supplied, keeping the sampler's own noise
sequence aligned — and a *different* latent must produce a different image,
which is the test that fails if the argument is quietly ignored.

**Determinism is now a tested guarantee**, not an assumption: same seed and
parameters give byte-identical output across runs *and across pipeline
instances*, the second of which is what would catch a load path that is not
bit-stable.

**Several ControlNets can be bound at once.** `ControlConfig::controls` is a
`Vec<Control>`, one map and strength per attached net, and the corrections are
summed before the UNet sees them — which is what diffusers does, and is
correct because each was trained against the same frozen base, so they are
independent additions rather than alternatives. Pose for a figure plus depth
for the scene is the motivating pairing. A count mismatch is refused rather
than zipped to the shorter list, which would otherwise hand the wrong net the
wrong hint at shapes that stay valid.

**4x upscaling works**, `sdrs upscale --model <esrgan_x4.safetensors>
--input <img>`, verified against the reference RRDBNet at **1.669e-6**. Real-
ESRGAN is a pure convolutional network — no attention, no normalisation, no
diffusion — so it runs after generation and knows nothing about it.
`assets/esrgan-crab-2048.png` is 512 -> 2048.

**It found a silent-corruption bug in candle's Metal convolution, and the
boundary is exactly `i32::MAX`.** A 3x3 convolution over `out_h * out_w`
positions at 64 channels builds an im2col matrix of `out_h * out_w * 64 * 9`
elements. Past `i32::MAX` the Metal kernel returns a dark, horizontally banded
image — **no error, no failed command buffer, nothing to catch**. CPU renders
the same input correctly, which is what identified it as candle's rather than
this port's.

The threshold was measured, not inferred, and it lands where the arithmetic
says it should:

```text
  output 1928 px   2,141,097,984 elements   under i32::MAX   correct
  output 1936 px   2,158,903,296 elements   over  i32::MAX   corrupt
  sqrt(i32::MAX / (64*9)) = 1930
```

`upscale_tiled` splits anything above it, 384 px tiles with 16 px of context,
and the seam is not visible: the column-to-column jump at the tile boundary is
0.24 where the largest jump anywhere in the image is 4.43. Against a one-pass
CPU render, 98.8 % of pixels agree within 16/255 — the rest is high-frequency
detail where a tile has less context to work with, which is the cost of tiling
and not a defect.

`upscale_in_tiles(image, tile, pad)` takes both explicitly so the tiling is
testable without a 2000 px image: give every tile a padding larger than the
image and the result must be **exactly** the one-pass result, which is what
pins the crop offsets and the stitching order.

**SD 2.x renders**, `sdrs txt2img --model <sd21-dir>`, verified against
diffusers at **3.696e-5** on the UNet output with all twelve skips and the mid
block inside the same bound SD 1.5 uses. `assets/sd21-crab-512.png`.

Almost all of it was already there. SD 2.x is SD 1.5's block geometry with a
1024-wide OpenCLIP ViT-H behind it: SDXL-style head counts (`[5, 10, 20, 20]`,
all 64 wide), `Linear` transformer projections rather than 1x1 convolutions,
and `ClipActivation::Gelu`. Two things are worth knowing:

**The text encoder has 23 layers, not 24.** SD 2.x conditions on the
penultimate hidden state, so the conversion to diffusers format drops the last
layer outright — the shipped checkpoint has 23 and the ordinary "last layer,
then `final_layer_norm`" path is then exactly right. Reaching for
`penultimate_hidden_state` here would silently condition on layer 22.

**It is v-prediction, and that is invisible in the weights.** The model outputs
`v`, not noise, so `x0 = x/(1+sigma^2) - v*sigma/sqrt(1+sigma^2)`. Sampled as
epsilon it loads, runs, and reports nothing wrong — it just returns saturated
colour noise. That is measured: forcing the detection to `Epsilon` for this
checkpoint and rendering the same seed gives no crab at all, where
v-prediction gives a sharp one.

Both the architecture and the prediction type are **detected, not flagged**.
The architecture comes from a tensor shape — the cross-attention key
projection is 768 wide for SD 1.5 and 1024 for SD 2.x — via
`sd_tensor::tensor_shape`, which reads a safetensors header without loading
data. The prediction type comes from a substring test on
`scheduler_config.json`, deliberately: the token is unambiguous and a JSON
parser for one boolean is not worth a dependency this workspace otherwise does
without.

**Stock SD 2.x is gated on HuggingFace** — every `stabilityai/stable-diffusion-2*`
repo 401s, and so do the community mirrors, while unrelated repos return 200.
The verification above uses `friedrichor/stable-diffusion-2-1-realistic`, an
open fine-tune with byte-identical architecture and the same v-prediction
scheduler. Anyone with an accepted licence can point the same dumper at the
stock checkpoint.

**Flux and SD 3.5 have CLI subcommands**: `sdrs flux --model <dir>` and
`sdrs sd3 --model <dir>`. `--model` is a *directory* and that was the whole
design question — Flux needs four checkpoints plus two tokenizers, SD 3 six
files, and naming each would be flags nobody remembers. `paths_in` and
`sd3_paths_in` already took a directory, so the answer was sitting in the
codebase. Both support `--taesd`, `--preview-every` and `--stream`; SD 3 also
takes `--encoders-on-cpu`.

**Step previews work for all four architectures** — SD 1.5 and SDXL via
`--preview-every N`, Flux and SD 3.5 via `SD_PREVIEW_EVERY` on their examples.
This is what the tiny decoder was for: a 20-step run no longer sits blank for
two minutes.

**Flux and SD 3.5 needed a different `x0`, and getting it wrong would have
looked plausible.** They are rectified flow: the model predicts a *velocity*,
not noise, so the estimate is `x - sigma*v` — the inverse of the forward
process `x = sigma*noise + (1-sigma)*x0` — and not the DDPM `x - sigma*eps`.
Flux also carries its latents 2x2-*packed* through the loop, so the estimate
has to be unpacked before anything decodes it; handing over the packed form
gives a tensor of an entirely reasonable shape that decodes to nonsense.

Both are confirmed by a property that falls out for free and is worth
keeping: **at the last step the x0 estimate equals the returned image
exactly.** `sigma_next` is 0 there, so the sampler lands on `x0`; a wrong
formula would not land. Measured against the final image, mean absolute
difference per 8-bit channel:

```text
  flux  step 1/4   18.1      sd3.5  step  5/20   20.6
        step 2/4   10.2             step 10/20   10.8
        step 3/4    0.2             step 15/20    5.5
        step 4/4    0.0             step 20/20    0.0
```

Flux schnell's estimate after a *single* step is already close to final
(`assets/flux-preview-step01-of-04.png`) — which is what four-step
distillation buys, seen directly.

**All four TAESD checkpoints are ported**: `taesd` (SD 1.5/2.x), `taesdxl`,
`taesd3` and `taef1`, the last two at 16 latent channels. Same architecture
throughout, so `TinyDecoder::new` takes the width; a 4-channel file in a
16-channel slot fails to load, which is the one mismatch in the family that
is loud.

**The preview is the `x0` estimate, not the sampler's latent**, and that is
the whole design. The latent at step 5 of 20 is `x0 + sigma*noise` with sigma
still near 4, so decoding it shows a field of coloured noise — the first
version did exactly that and looked broken. The model's `x0` prediction is
already computed every step; decoded, it is a blurry crab at step 5 that
sharpens as the run proceeds (`assets/preview-step05-of-20.png`). Every
diffusion UI shows this and now so does this one.

`ProgressFn` therefore carries a `Progress` struct rather than three
positional arguments. A fourth positional `&Tensor` would have been easy to
ignore, which is the opposite of the point.

**SDXL has it too**, with `taesdxl` — verified at 1.61e-5 decoding, 7.03e-5
encoding, and pinned as a *different* checkpoint from `taesd`: the two share an
architecture, so the wrong one loads without complaint and decodes in visibly
wrong colours. Nothing in the code can tell them apart, so the path is the
caller's to get right.

This is where it pays most. `decode_tiled` splits anything above a 64-latent
edge, so a 1024 VAE decode is four tiles with their seams; TAESD does it in one
pass. Measured at 1024, 4 steps, **both orderings**:

```text
  vae   161.7 s      taesd 117.4 s     (reversed)
  taesd 123.8 s      vae   164.9 s
```

38-48 s, direction consistent. The absolute numbers are inflated — the machine
was at load average 20-24 from unrelated work — which is also why they cannot
be compared with the 86.5 s in the table above. `assets/sdxl-1024-taesdxl-crab.png`.

**TAESD decodes**, `sdrs txt2img --taesd <ckpt>`. About 5 MB of 3x3
convolutions against the VAE's 330 — no attention, no GroupNorm, no sampling
head — verified against `diffusers.AutoencoderTiny` at **1.86e-5** decoding
and **1.24e-5** encoding.

Compared through the public `decode`/`encode` rather than the layer stack, on
purpose. TAESD's architecture is too simple to get wrong; what is easy to get
wrong is its *conventions* — its `scaling_factor` is 1.0, so it takes the
sampler's latent with **no `/ 0.18215`**, and it soft-clamps its input with
`tanh(x/3)*3` and returns `2x - 1`. Each of those is a plausible image and no
error if missed.

**The decode is 8-11 s cheaper at 512** on this machine, measured at one step,
where decode is a large fraction of the run rather than lost in it. At twenty
steps that is about 7 % of a 125 s run, and **below run-to-run variance** — the
first 20-step pair suggested a 25 s saving and the second suggested none. The
one-step figure reproduces in both directions and is the one to trust.

**The memory win is 7x, and it is the activations, not the weights.** An
earlier note here said the win was unrealised because `with_taesd` kept the
VAE resident. That was wrong twice over: the 189 MB of VAE decoder weights is
now dropped (`Decoder` is an enum, so attaching TAESD *replaces* rather than
adds, with a synchronise because a drop alone frees nothing on Metal) — and
that 189 MB was never the interesting number anyway. Decoding is:

```text
  latent edge   output      VAE            TAESD
  64            512 px      3.43 GB        0.49 GB     7.0x
  128           1024 px     3.22 GB *      1.71 GB
```

`* tiled.` `decode_tiled` splits anything above a 64-latent edge, so the VAE
holds ~3.4 GB at any size by **seaming the image** instead. TAESD does 1024 in
one pass. Reproduce with
`cargo run --release -p stable-diffusion-rs --example decode_peak -- taesd 64`.

Measuring this end to end is what hid it: `sdrs`'s peak RSS is dominated by
the UNet's 3.4 GB, so the two decoders looked 0.14 GB apart — which is only
the weights. The decode had to be isolated before the real number appeared.

Output agreement with the VAE is mean 18.9/255, max 214 — TAESD is genuinely
lossy, which is the trade, not a defect.

**ControlNet works**, `sdrs controlnet --controlnet <ckpt> --init-image X`, for
SD 1.5. A ControlNet is a copy of the UNet's down and mid stack that reads a
control map and emits one correction per skip connection; the UNet adds them
before the up pass consumes them and is otherwise untouched. So the module is
short — `DownBlock2D`, `MidBlock2DCrossAttn` and `TimestepEmbedding` are the
UNet's own, reused verbatim — and the only new parts are the hint encoder and
the zero convolutions.

Verified against `lllyasviel/sd-controlnet-canny` **correction by correction**,
all thirteen: worst excess **1.45e-5** against a 1e-3 bound. Comparing them
individually rather than as one tensor is the point — a ControlNet has no image
of its own, so those thirteen are its entire observable behaviour, and the
index of the first bad one localises the fault.

The A/B that makes it a demonstration rather than a hope: same prompt, same
seed, `--control-scale 0` gives a centred crab on a marble pedestal — exactly
what the prompt asks for. At scale 1 the crab takes the source photo's sprawled
pose and no pedestal appears at all, because the edge map says "sand". Both in
`assets/`, with the edge map beside them.

**Canny edge detection is built in** (`crate::canny`), so no second tool is
needed. Full four stages, and the last two are what matter: non-maximum
suppression, which makes edges one pixel wide instead of thick bands, and
hysteresis, which is a flood from the strong pixels rather than a raster sweep —
a sweep misses any chain running backwards.

**The thresholds matter more than they look.** The defaults (0.1 / 0.2) are
right for clean subjects and far too sensitive for a textured photograph: on a
crab on sand they turn the sand into a field of speckle, which the model then
faithfully renders as background hatching. 0.25 / 0.45 gives the clean result
in `assets/`. That the speckle *did* come through is itself evidence the
control is being followed.

**Inpainting works**, `sdrs inpaint --init-image X --mask M`, on any SD 1.5
checkpoint — no 9-channel inpaint UNet required. The mask follows the universal
convention: **white repaints**.

Two properties hold exactly, and both are tested:

- **The untouched region is bit-identical to the input.** Latent blending alone
  cannot give that, because it preserves the *encoded* original and a VAE round
  trip is lossy; the result is composited against the original in pixel space
  at the end.
- **A mask of all black returns the input unchanged**, max diff 0 over the
  whole image.

The latent mask is 8x8 **max**-pooled, not averaged. A latent cell is not a
pixel: if any pixel under it is free, the whole cell must be free, or the cells
straddling the mask edge end up nearly frozen and leave a hard seam exactly
where it shows most.

A visible seam remains on *large* holes — a half-image mask is the worst case,
and the one in `assets/` shows it. That is the honest limit of blending
without a dedicated inpaint checkpoint: the model never sees the mask, so it
has no way to compose across the boundary. Fixing it means the 9-channel UNet,
which is a checkpoint change, not a code change.

**A rounding bug in every image this project has saved was found by that
exactness check** and is fixed. `tensor_to_rgb8` used `v as u8`, which
truncates; `b/127.5 - 1` followed by `(x+1)*127.5` lands just below the
integer often enough that a load-and-save darkened most pixels by one level.
It now rounds, matching diffusers. This is why the inpaint invariant was worth
asserting as *exact* rather than *close*: at max-diff 1 it reads as noise, and
a tolerance would have hidden a real defect in the output path.

**LoRA adapters load and merge**, for SD 1.5's dense path:
`sdrs txt2img --lora <file> --lora-scale <f>`, or
`Txt2ImgPipeline::load_with_lora`. Verified against a published adapter
(`latent-consistency/lcm-lora-sdv1-5`, 278 layers): every entry finds a weight,
each is merged exactly once, `--lora-scale 0` is **bit-identical** to no LoRA
end to end, and scale 1 changes exactly those 278 tensors and nothing else.
A partial match is refused rather than rendered — see `PipelineError::LoraMismatch`.

Quantised bases are not supported: merging into them means dequantise and
requantise, which is lossy. That needs runtime application instead, which is
the same design question as the streamed-block cache.

**Attention now has four paths**, and `ops::attention_with_path` reports which
one ran: `Fused` (candle's Metal SDPA), `FlashCpu` (candle's CPU flash kernel,
new — taken only at or below 512 tokens, and never for T5, see
[roadmap.md](roadmap.md#cpu-flash-attention-a-short-sequence-win-not-a-general-one)),
`Chunked` and `Naive`. Compare paths with `--example attention_path`, which
prints both CPU timings side by side and flags a bad dispatch choice. Note its
rows are all unmasked, so they do not tell you what a *masked* caller gets —
that is pinned by tests instead.

## Next

In the order I would do them, with why.

### ~~1. Overlap the block copy with compute~~ — tried, and it is slower

Built and reverted. The prefetch itself worked exactly as intended: a
background thread copying block `i+1` while block `i` computes took the copy
wait from ~220 ms per step to ~15 ms. The run got *slower* by about the same
amount, and the whole generation went from **28.5 s to 36.0 s**.

**The cause is in candle, and it is device-independent.**
`MetalDevice::allocate_buffer` takes a write lock on a single shared buffer
pool (`Arc<RwLock<BufferMap>>`). The prefetch thread allocates roughly two
dozen buffers per block; the compute thread allocates activations constantly.
They serialise on that lock, so the two never overlap — they take turns with
extra contention on top.

That matters for whether to revisit this. The obvious reading of the numbers
is "unified memory shares bandwidth, so a discrete card would be different" —
but the lock is not a memory-architecture property. It would serialise the
same way over PCIe.

So this is not "prefetch does not pay on this machine", it is "prefetch cannot
pay until allocation stops being globally serialised". Two routes, neither
small: pre-allocate each block's buffers once and reuse them across steps
(sidestepping the allocator entirely), or fix the pool upstream. Measure the
lock before either — this was diagnosed by reading candle's source, not by
profiling it.

### 1. Finish GLIGEN: the gated self-attention

The grounding tokens are done and verified at 7.629e-6 (above). What consumes
them is a **`fuser` in each of the 16 transformer blocks**, between `attn1` and
`attn2`:

```text
  x = x + tanh(alpha_attn)  * attn(norm1([x ; objs]))[:, :n_visual]
  x = x + tanh(alpha_dense) * ff(norm2(x))
```

Each fuser carries `linear` (768 -> block width, projecting the tokens),
`attn`, `ff`, `norm1`, `norm2`, and two **scalar** gates `alpha_attn` and
`alpha_dense`. The `tanh` on the gates means an untrained fuser is an exact
identity, which is how these are trained against a frozen base — and it also
gives a free test: zero the gates and the UNet must be bit-identical to the
same UNet without grounding.

Install them the way `unet::ip` installs IP weights: a construction-scoped
thread-local pulled as each block is built. **One per cross-attention block,
16 of them**, and unlike the IP-Adapter's list they are named by path
(`down_blocks.0.attentions.0.transformer_blocks.0.fuser`) so no index mapping
is needed — which removes the trap that cost the most time there.

Then: phrase embeddings come from CLIP per phrase (pooled), and GLIGEN applies
grounding only for the first fraction of the schedule — "scheduled sampling",
typically 30 %. Verify end to end against `StableDiffusionGLIGENPipeline`,
since where a fuser lands is silent when wrong.

Note the checkpoint is a **whole SD 1.5 UNet** plus fusers, not an adapter, so
it loads as its own model rather than attaching to one.

### ~~2. TAESD and step previews~~ — both done

Ported, verified and wired up (see above). Two ends left loose, neither large:

- **Flux and SD 3 have no CLI subcommand**, so their previews are driven by
  `SD_PREVIEW_EVERY` / `SD_TAESD` on `examples/{flux,sd3}_txt2img.rs`. Folding
  them into `sdrs` is the obvious tidy-up and is a bigger question than it
  looks: they take paths to five and six files respectively, where txt2img
  takes one directory.

### 2. Extend streaming past Flux and SD 3.5, and measure it on a discrete GPU

`madebyollin/taesd` is a ~5 MB distilled replacement for the VAE decoder. Two
things it buys, and the second is the one that matters here:

- **Step previews.** Decoding is currently far too slow to show intermediate
  latents, so a 20-step run is a blank wait. TAESD decodes cheaply enough to
  preview every step.
- **A decode that always fits.** `decode_tiled` already degrades gracefully,
  but a decoder small enough to never need tiling removes the failure mode
  rather than managing it.

Architecturally it is simple — a stack of 3x3 convolutions and ReLU residual
blocks, no attention, no GroupNorm — so this is a short port with an easy
reference (`diffusers.AutoencoderTiny`). The trap to watch: TAESD's latent
convention is its own (`latent_magnitude = 3`, `latent_shift = 0.5`), *not*
the SD VAE's `0.18215`, and mixing them gives a washed-out image rather than
an error. Verify against `AutoencoderTiny.decode` end to end, not just the
module stack, so the scaling is covered by the test.

### 3. Extend streaming past Flux and SD 3.5, and measure it on a discrete GPU

`Residency::Streamed` works for **Flux and SD 3.5**, both quantised. Gaps, in
the order they matter:

- **Dense checkpoints cannot stream at all.** `quantized::to_device` moves
  quantised block bytes verbatim, which is what makes it cheap and bit-exact;
  a dense model would move 4x the bytes with no equivalent shortcut. Flux mini
  — the model that most needs this, at 12.8 GB dense — is therefore the one
  that cannot have it. Whether a dense path is worth it is a measurement
  nobody has made.
- **Nothing prefetches**, and per the profile above that is not the first
  thing to fix. `stable-diffusion.cpp` overlaps copy and compute
  (`stream_layers`); worth doing once the build cost is gone.

And the honest one: **the payoff has not been measured on the hardware it is
for.** On unified memory the host copy sits in the same pool, so what the
device gives up it gives up to the same allocator — the mechanism is
demonstrable there (one block resident instead of the stack, at the default
sync interval) but the *benefit* is not. On a discrete card it should take
VRAM from 6.66 GB to ~192 MB, by construction rather than by measurement. If a
CUDA machine ever appears, measure this before anything else here.

### ~~3. Deduplicate RMSNorm onto `candle_nn::ops::rms_norm`~~ — done, and the
answer was no

The three copies are now one, in `ops::rms_norm`, and `PlainLayerNorm`'s two
copies are one in `ops::plain_layer_norm`. But **candle's fused kernels are
not what they were deduped onto**, and that is the part worth keeping: its
`rms_norm` sums each row with a sequential `.sum::<f32>()` where
`mean_keepdim` reduces in blocks, so it is 4-11x less accurate, worst at long
rows. Measured against f64 at `[1, 154, 4096]`: 9.695e-7 for ours against
9.627e-6 for candle's. Swapping T5 onto it moved `golden_t5` to 3.891e-3,
past a 3e-3 bound that was itself measured. The speed it buys is 2.1x on that
shape and *negative* — 2.7x slower — at `[1, 77, 768]`.

**`candle_nn::ops::layer_norm` has now been measured too, and the answer is
also no — but it is the closest call of the three.** Unlike `rms_norm` it
breaks no golden bound: swapping `ops::plain_layer_norm` onto it leaves Flux
and SD 3.5 passing. What it costs is margin, and what it buys is 6 %:

```text
                       ours        candle fused
  norm alone, Metal    1.60 ms     0.14 ms        11.3x   (Flux shape)
  norm alone, CPU      1.44 ms     0.56 ms         2.6x
  SD 3.5 end to end    25.6 s      24.0 s          1.06x  (3 runs vs 2)
  SD 3.5 max_abs       5.484e-6    8.345e-6       +52 %
  Flux max_abs         1.190e-3    1.556e-3       +31 %
```

11x on the operation is 6 % on the run, because a norm is one cheap op in a
block dominated by attention and two large matmuls. Half the accuracy margin
for that is the wrong trade in a project whose product is the verified
agreement — and the bounds are not arbitrary, they were set from the
reference's own f32-against-f64 noise floor, so spending margin makes every
later change more likely to trip. On Metal the accuracy is not even
measurable: there is no f64 there to compare against.

Reversing this is a four-line change if priorities shift; the numbers are
here so it need not be re-measured. Note the pattern is *not* "fused is
always worse" — it is that fused trades accuracy for speed, and all three
times so far the trade has been bad.

`cargo run --release -p sd-tensor --features metal --example layer_norm_bench`
reproduces the first two rows.

### Also open

- **ControlNet for SDXL.** The mechanism is architecture-independent —
  `forward_controlled` takes plain tensors and `ControlNet::new` takes a
  `UNetConfig` — so this should be config and a checkpoint, not new code. Worth
  confirming that claim rather than assuming it.
- **Multiple ControlNets at once.** diffusers sums the corrections from several
  before applying them. The summing is trivial; the question is whether the
  API should take a list, and that is worth deciding once rather than twice.
- `candle_nn::rotary_emb::{rope, rope_i, rope_thd}` — fused RoPE. Flux's
  axis-wise 2x2 form may not map onto it; establish rather than assume.
- `candle_nn::ops::{pixel_shuffle, pixel_unshuffle}` — patchify/unpatchify by
  another name.
- Broaden fused attention to SD 1.5 by materialising `causal_mask` from
  `[1,1,s,s]` to `[b,h,s,s]`. A reshape, not a kernel.
- A **blocked CPU attention kernel**, tiling over query rows as well as keys.
  This is the one that would matter: attention at 4096 tokens is where CPU
  time actually goes, and it is exactly where candle's CPU flash kernel loses
  to gemm — see the measurements in
  [roadmap.md](roadmap.md#cpu-flash-attention-a-short-sequence-win-not-a-general-one).
  Real kernel work, not wiring.
- `--example attention_path` **flags two rows as "flash is faster here", by
  design** — they are known misses, not regressions:
  - *Flux 1024*, 1.15-1.25x. CPU flash has a second crossing at
    `head_dim = 128`: it sags to 0.85x around 1024 tokens then climbs back by
    4608. `DEFAULT_FLASH_CPU_MAX_SEQ` ignores it because capturing it needs
    two disjoint intervals fitted to one `head_dim` on one machine.
  - *SDXL cross-attention*, 1.07-1.12x, repeatable. But SD 1.5's
    cross-attention at the same sequence lengths and `head_dim = 40` is
    0.83x, so this one is not separable by sequence length at all — it would
    need `head_dim` in the rule. Worth ~0.8 ms per call, and SDXL runs on
    Metal anyway.
- Flux **dev** is gated on HuggingFace and needs the user's account.
- CUDA is untested — no device available here.
- ~~SDXL img2img is unverified end to end after the encoder-tiling fix.~~
  **Verified.** Encoder (tiled) through strength, sampler and decoder: the
  strength maps to steps correctly (`--steps 8 --strength 0.35` runs 3), the
  output holds the input's composition at low strength and leaves it at high
  (mean 24.5/255 from the input at 0.35 against 90.0 at 0.9), and forcing the
  encoder to tile changes the result by mean 6.6/255 — the "close to but not
  identical" that tiling is documented to be, not a defect.

## SDXL below its native resolution is garbage, and that is not a bug

Worth writing down because it looks exactly like one. SDXL at 256x256 produces
saturated colour noise with no subject at all — and it does so in **txt2img**,
which is what rules the pipeline out. An img2img at strength 0.9 looks equally
broken for the same reason: at that strength the output is mostly generated
rather than preserved, so it inherits the same failure.

At 512 it is coherent but heavily stylised; 1024 is where it belongs. If an
SDXL result looks like a bug, render the same prompt as txt2img at the same
size before investigating anything else — that single control separates "the
resolution" from "the code" in one run.

This is also why 1024 is a poor size to *test* at. Verifying the img2img path
means exercising encoder tiling, which only engages above
`TILE_LATENT_EDGE * 8`; lowering `SD_VAE_TILE_LATENT` makes a 256px image tile
into four, covering the same code at a sixteenth of the pixels. Reaching for
native resolution instead put a 1024 VAE *encode* on the GPU — wired memory,
the allocation class that has taken this machine down before — and did exactly
that again.

## Traps this codebase has already paid for

Each of these cost real time. They generalise.

**Verify against a published artifact, not one your own tooling
round-tripped.** A legacy VAE attention-naming bug survived a fully green
suite because the reference weights came from `vae.state_dict()`, which
diffusers renames on load.

**When a checkpoint exists in two published forms, verify against the one you
load.** SD 3.5's single fp16 file and its converted fp32 copy differ by up to
2e-3, which 24 blocks amplified into a 9.3e-3 output mismatch that looked
like a bug in our code. Regenerating the reference from the file Rust reads
took it to 5.5e-6.

**Absolute tolerances are wrong for anything that is not order-1 — including
the UNet, which nobody suspected.** CLIP peaks at 851, T5 at ~200,000, SD
3.5's blocks at ~97,000; the UNet's mid block peaks at a mere 16, and a 1e-4
*absolute* bound on it still turned out to be **below the reference's own
noise floor**. `xtask/golden/reference_precision.py` runs a diffusers module
against itself in f64: `mid_output` shows diffusers missing its own f64 by
1.108e-4. That test could only ever pass by accident of summation order, and
it duly broke the first time Apple's Accelerate reordered it — at 1.087e-4,
*closer to the reference than the reference's own f32 was*.

So: use `testing::allclose_excess(a, b, rtol)` with an absolute floor beside
it, and **measure before choosing either number.** The script exists now; run
it rather than picking a value that turns the light green. That is how the
Flux VAE encoder's 2e-3 bound and T5's 3e-3 were set, and both sat at or below
the reference's own noise floor.

**Test the smallest input that reaches the code, not the most realistic one.**
Verifying SDXL img2img meant exercising encoder tiling, which engages only
above 512px — so native 1024 looked like the honest choice and wedged the
machine on a wired Metal allocation. `SD_VAE_TILE_LATENT=16` tiles a 256px
image into four and covers the same path for a sixteenth of the memory. The
knob had been added earlier the same day and forgotten.

**Do not `git checkout` a file to undo a temporary edit if it also holds
uncommitted work.** Reverting a one-line test perturbation this way discarded
an hour of tolerance work in the same file. Copy the file aside and restore
from the copy, or commit first.

**F16 is not a safe way to halve memory.** T5's activations exceed 65,504 and
go NaN around block 10; Flux's transformer NaNs too. Hold weights quantised
instead (`weights::Source::Quantized`) — activations stay f32 and residency
drops further than F16 would have. bf16 would fix it but candle's CPU backend
has no bf16 matmul.

**On Metal, the op that reports the error is rarely the op that caused it.**
candle queues GPU work and only inspects the command buffer when something
synchronises, so a failure is attributed to the first thing that *waits*.
SD 3.5 at 512 "failed in the VAE decode" through several sessions and a
written handoff note; synchronising after each denoise step put it at step 1,
where it had been all along. Before believing where a Metal error happened,
put a `device.synchronize()` after each stage and watch it move.

**A tensor that does not own its buffer is a different tensor.** candle 0.11's
Metal quantised matmul ignores `start_offset`, so a view into the middle of a
buffer is read from the beginning of it — silently, with correct shapes. Flux
rendered a flat orange field for three sessions because every block projects
`attn.narrow(1, 512, 1024)`. **`contiguous()` does not fix this**: a narrow off
any axis but the last is *already* contiguous by candle's definition — the
elements are consecutive, they just start late — so it is a no-op.
`force_contiguous()` always copies. The workaround lives in
`QLinear::forward`, at the seam, because that is where every quantised matmul
in the workspace passes.

**Test tensors always own their buffers, which is why per-op checks missed
it.** `metal_check` built fresh inputs, so it exercised the op in exactly the
condition where it is correct. Attention agreed to 1.9e-7 at every sequence
length; QLinear at every row count and quant type; norms, RoPE, trig, cat,
narrow, weight loading, dequantisation to 1e-8 — all green, while the composed
model was 50% wrong. When every part passes and the whole fails, suspect
*provenance*: build the op's input the way the model builds it, not the way a
test does.

**Bisect against a full-precision reference, not against the other device.**
CPU-vs-Metal conflates two error sources. CPU's quantised matmul carries
0.3-1.9% activation-quantisation noise, so a per-layer CPU/Metal table read as
plausible drift everywhere and pointed at nothing. Dequantising the *same*
weights into a dense f32 model — only feasible at depth 1, which was enough —
turned it into one obvious line: Metal tracked truth better than CPU
(≤0.12%) through the whole block, then jumped to 36% at a single projection.
Right after that, recomputing that one op in numpy from *Metal's own dumped
input* proved the op wrong rather than its input.

**Check whether candle already does it.** The roadmap called a hand-written
fused Metal attention kernel the highest-value work available; candle 0.11
already shipped one and `attention_with_path` was already routing to it. Only
a stale doc comment said otherwise.

**…and then check whether it is actually faster, on your shapes.** The
follow-up to that lesson overcorrected: the survey listed CPU flash attention
as "potentially the largest single win available" on the strength of it
existing. It is 2-7x faster below 512 tokens and up to 2x slower above, and
everything above 512 is where CPU time actually goes — net, nothing
measurable end to end. Wiring it in unconditionally would have made SD 1.5
self-attention 2x slower while the enum said `FlashCpu` and every test stayed
green. Benchmark the new path against the existing one across the shapes you
really run, before the dispatcher prefers it, and keep both.

**On this machine, benchmark by interleaved minimum — never by mean, and
never from one A/B pair.** Two traps, both paid for on 2026-07-26:

- A mean-of-5 microbenchmark reported figures 10x apart on back-to-back runs
  of the same binary. Noise here is one-sided — preemption, page faults and
  thermal excursions only add time — so take the *minimum* of N runs after a
  warm-up, and *alternate* between the two things compared so drift lands on
  both. `--example attention_path` does both and says why.
- At whole-generation scale one A/B pair said CPU flash made SD 3.5 11.8%
  faster. Running the same pair with the order reversed said 0.7%, and two
  runs of the *identical* configuration differed by more than the two
  configurations did. **Always run the pair both ways round.** Had the first
  pair been believed, a fabricated 12% would now be in roadmap.md, and the
  arithmetic said it was impossible — the shapes involved are worth 0.4 s.
  When a measurement disagrees with the mechanism, suspect the measurement.

**Measure before diagnosing.** The Flux bottom-edge striping was assumed to
be our packing or positional encoding. Handing diffusers *our* conditioning
and *our* initial noise removed everything but the loop, and the loop agreed
to 6.5e-5 — the artifact is the checkpoint's.

## Machine

36 GB M4 Max. A previous session **crashed the whole machine** with an 81 GiB
wired Metal allocation. Metal allocations are wired and can take down the OS
rather than failing the process.

`sd_tensor::sysmem::check_headroom` guards loads against *actual* free memory
(`SD_MEMORY_HEADROOM` overrides). Check `memory_pressure` before large runs.
Note free space can read misleadingly low — macOS purgeable space once showed
5.3 GB where 142 GB was actually available.

## Running things

```bash
# fixtures live under tests/golden/, gitignored, mostly symlinks into ~/.cache/huggingface
cargo run --release -p sd-cli --example flux_txt2img -- "<prompt>" <steps> <size> out.png
cargo run --release -p sd-cli --example sd3_txt2img  -- "<prompt>" <steps> <size> out.png
cargo run --release -p sd-cli --example attention_path      # which path, how fast
cargo run --release -p sd-cli --example metal_check         # CPU vs Metal per op
cargo run --release -p sd-cli --example requantise -- in.gguf out.gguf Q4_K

# golden references (local only; CI skips the numerical tests)
python3 xtask/golden/dump_reference.py <component> --output tests/golden
#   components: vae flux_vae sd3 flux_transformer flux_sampling t5 flow clip_* unet_* samplers sdxl_*
```

`flux_txt2img` prefers `flux-schnell-q4_k_s.gguf` when present and falls back
to flux-mini; it reads block counts and guidance from the file rather than
assuming.

## Fixtures

Regenerable, so deleting them is cheap:

- `tests/golden/flux/` — flux-mini (6.4 GB), flux schnell Q4_K_S (6.8 GB),
  T5-XXL Q4_K_S (2.7 GB), Flux VAE, tokenizers
- `tests/golden/sd35/` — SD 3.5 single-file (5.1 GB), Q4_K_M gguf, VAE,
  CLIP-L, CLIP-G
- `tests/golden/gguf/` — SD 1.5 quant sweep. The k-quants were deleted to
  save space; regenerate with `--example requantise` from `sd15-f16.gguf`.

Ungated mirrors matter: `black-forest-labs/*` and `stabilityai/*` are gated.
Use `Freepik/flux.1-lite-8B` for the Flux VAE and T5 tokenizer, and
`adamo1139/stable-diffusion-3.5-medium-ungated` for SD 3.5.
