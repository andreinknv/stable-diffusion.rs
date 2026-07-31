//! img2img and inpainting on MLX.
//!
//! The models underneath are already gated against diffusers — the VAE encoder
//! at `mlx_golden_vae`, the UNet at `mlx_golden_unet`. What is left to check is
//! the part that has no reference tensor to compare against: whether `strength`
//! and the mask *mean* what they claim. Both are properties rather than
//! numbers, and both fail silently — a run that ignores its strength still
//! produces an image, and an inpaint that quietly repaints the whole canvas
//! still produces an image.
//!
//! ```bash
//! cargo test -p stable-diffusion-rs --features mlx --test mlx_img2img -- --nocapture
//! ```
//!
//! `Strength` itself is imported rather than reimplemented: `start_index` is
//! arithmetic on two integers and touches no tensor, so both backends call the
//! same function and cannot drift apart.
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::{clip, sample, unet_forward, vae, UNetConfig};
use sd_sample::{sigmas_for_steps, Schedule};
use sd_tensor::mlx::{concat, load_safetensors, Array, Stream};
use sd_tensor::rng::SeededRng;
use sd_tensor::{Device, Tensor};
use stable_diffusion_rs::pipeline::Strength;

const VAE_SCALE: f32 = 0.18215;
const STEPS: usize = 8;
const CFG_SCALE: f64 = 7.5;
const BOS: i32 = 49406;
const EOS: i32 = 49407;

fn golden(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(sub)
}

fn seeded_noise(rng: &mut SeededRng, shape: (usize, usize, usize, usize)) -> Vec<f32> {
    let t: Tensor = rng.randn(shape, &Device::Cpu).expect("randn");
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn nchw_to_nhwc(data: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ci] = data[ci * h * w + y * w + x];
            }
        }
    }
    out
}

/// Everything a run needs, loaded once.
struct Fixtures {
    unet: sd_models::mlx::Weights,
    vae: sd_models::mlx::Weights,
    context: Array,
    /// The source image, `[1, h, w, 3]` in `[-1, 1]`.
    image: Array,
    stream: Stream,
}

fn load() -> Option<Fixtures> {
    let needed = [
        golden("unet_full").join("unet.safetensors"),
        golden("vae_decoder").join("vae.safetensors"),
        golden("vae_decoder").join("reference.safetensors"),
        golden("clip_encoder").join("clip.safetensors"),
    ];
    if !needed.iter().all(|p| p.exists()) {
        return None;
    }
    let s = Stream::gpu();
    let unet = load_safetensors(&needed[0]).expect("unet");
    let vae_w = load_safetensors(&needed[1]).expect("vae");
    let refs = load_safetensors(&needed[2]).expect("vae reference");
    let clip_w = load_safetensors(&needed[3]).expect("clip");

    // A fixed prompt; this test is about strength and masks, not text.
    let mut ids = vec![EOS; clip::MAX_POSITION];
    ids[0] = BOS;
    let empty = Array::from_slice_i32(&ids, &[1, clip::MAX_POSITION]).unwrap();
    let e = clip::text_encoder(&empty, &clip_w, &s).expect("encode");
    let context = concat(&[&e, &e], 0, &s).expect("context");

    // The fixture's `image`, which is the decoder's own output — **not
    // `encoder_input`, which is `torch.randn` and is noise.** No autoencoder
    // round-trips white noise, so using it would make every distance below the
    // VAE's failure to represent it rather than anything img2img did.
    let image = refs
        .get("image")
        .expect("image")
        .transpose(&[0, 2, 3, 1], &s)
        .expect("NCHW -> NHWC");
    Some(Fixtures {
        unet,
        vae: vae_w,
        context,
        image,
        stream: s,
    })
}

/// One img2img run, optionally masked. Returns the decoded image, NHWC.
fn run(f: &Fixtures, strength: f64, mask: Option<&Array>, seed: u64) -> Array {
    let s = &f.stream;
    let cfg = UNetConfig::sd15();
    let [_, ih, iw, _] = f.image.shape()[..] else {
        panic!("image should be NHWC")
    };
    let (lh, lw) = (ih / 8, iw / 8);

    // The distribution mean, scaled — the sampler supplies all the randomness.
    let init = vae::encode(&f.image, &f.vae, s)
        .expect("encode")
        .mul(&Array::scalar_f32(VAE_SCALE).unwrap(), s)
        .unwrap();

    let schedule = Schedule::sd15();
    let sigmas = sigmas_for_steps(&schedule, STEPS);
    let start = Strength::new(strength).start_index(STEPS);

    let decode = |latent: &Array| -> Array {
        let scaled = latent
            .div(&Array::scalar_f32(VAE_SCALE).unwrap(), s)
            .unwrap();
        vae::decode(&scaled, &f.vae, s).expect("decode")
    };

    // Strength 0 leaves nothing to run, and the input is the answer.
    if start >= STEPS {
        return decode(&init);
    }

    let mut rng = SeededRng::new(seed);
    let draw = |rng: &mut SeededRng| -> Array {
        let n = seeded_noise(rng, (1, 4, lh, lw));
        Array::from_slice_f32(&nchw_to_nhwc(&n, 4, lh, lw), &[1, lh, lw, 4]).unwrap()
    };

    let mut latent =
        sample::noise_to_sigma(&init, &draw(&mut rng), sigmas[start], s).expect("noise to sigma");

    let train_sigmas = schedule.sigmas();
    for i in start..STEPS {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let latent_in = sample::scale_model_input(&latent, sigma, s).unwrap();
        let t = train_sigmas
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - sigma)
                    .abs()
                    .partial_cmp(&(*b - sigma).abs())
                    .expect("finite")
            })
            .map(|(i, _)| i as f32)
            .expect("non-empty schedule");
        let timestep = Array::from_slice_f32(&[t, t], &[2]).unwrap();

        let out = unet_forward(&latent_in, &timestep, &f.context, &cfg, &f.unet, s).expect("unet");
        let noise_pred = sample::guidance(&out, CFG_SCALE, s).expect("cfg");
        let denoised = sample::denoise_epsilon(&latent, &noise_pred, sigma, s).expect("denoise");
        latent =
            sample::euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &draw(&mut rng), s)
                .expect("step");

        if let Some(m) = mask {
            latent =
                sample::restore_outside_mask(&latent, &init, m, &draw(&mut rng), sigma_next, s)
                    .expect("restore");
        }
    }
    decode(&latent)
}

/// Mean absolute difference between two NHWC images.
fn distance(a: &Array, b: &Array, s: &Stream) -> f32 {
    let (a, b) = (a.to_vec_f32(s).unwrap(), b.to_vec_f32(s).unwrap());
    assert_eq!(a.len(), b.len());
    a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// **Strength has to mean something.** A run that ignored it would still
/// produce an image, so the only way to check it is the ordering: less strength
/// must land nearer the source.
#[test]
fn lower_strength_stays_nearer_the_source() {
    let Some(f) = load() else {
        sd_tensor::skip_missing_fixture!("SKIP: img2img needs the unet, vae and clip fixtures.");
        return;
    };
    let s = &f.stream;

    let untouched = run(&f, 0.0, None, 7);
    let gentle = run(&f, 0.25, None, 7);
    let heavy = run(&f, 0.95, None, 7);

    // Strength 0 runs nothing, so it is the VAE round trip and nothing else.
    let round_trip = distance(&untouched, &f.image, s);
    let near = distance(&gentle, &f.image, s);
    let far = distance(&heavy, &f.image, s);
    eprintln!("distance from source  strength 0 {round_trip:.4}  0.25 {near:.4}  0.95 {far:.4}");

    assert!(
        round_trip < near,
        "strength 0 ({round_trip:.4}) departed further than strength 0.25 ({near:.4}); it is \
         running steps it should have skipped"
    );
    assert!(
        near < far,
        "strength 0.25 ({near:.4}) departed as far as 0.95 ({far:.4}); the start index is not \
         being honoured"
    );
}

/// **Strength 0 is the VAE round trip, not a denoise.** It must reproduce the
/// source to within the autoencoder's own loss, which is the only error left.
#[test]
fn strength_zero_is_the_round_trip_alone() {
    let Some(f) = load() else {
        sd_tensor::skip_missing_fixture!("SKIP: img2img needs the unet, vae and clip fixtures.");
        return;
    };
    let got = run(&f, 0.0, None, 1);
    let d = distance(&got, &f.image, &f.stream);
    eprintln!("VAE round trip {d:.4}");
    // Measured 0.0373 on this fixture. The SD 1.5 VAE is lossy by design, so
    // the bound is not tighter than the autoencoder itself; what it rules out
    // is a denoise having happened, which moves the image by 0.68 at strength
    // 0.95 — an order of magnitude more, not a marginal difference.
    assert!(
        d < 0.1,
        "strength 0 moved the image by {d:.4}; steps ran that should not have"
    );
}

/// **The mask has to bound the edit.** An inpaint that quietly repaints the
/// whole canvas still produces an image, so the check is that the region
/// outside the mask moved far less than the region inside it.
#[test]
fn the_mask_bounds_where_the_edit_lands() {
    let Some(f) = load() else {
        sd_tensor::skip_missing_fixture!("SKIP: img2img needs the unet, vae and clip fixtures.");
        return;
    };
    let s = &f.stream;
    let [_, ih, iw, _] = f.image.shape()[..] else {
        panic!("NHWC")
    };

    // White (writeable) in the top-left quadrant only.
    let mut px = vec![0f32; ih * iw];
    for y in 0..ih / 2 {
        for x in 0..iw / 2 {
            px[y * iw + x] = 1.0;
        }
    }
    let mask_px = Array::from_slice_f32(&px, &[1, ih, iw, 1]).unwrap();
    let mask = sample::latent_mask(&mask_px, s).unwrap();
    assert_eq!(mask.shape(), vec![1, ih / 8, iw / 8, 1]);

    // Strength high enough that an unbounded run would visibly rewrite
    // everything — that is what makes the containment meaningful.
    let painted = run(&f, 0.95, Some(&mask), 3);

    let a = painted.to_vec_f32(s).unwrap();
    let b = f.image.to_vec_f32(s).unwrap();
    let (mut inside, mut outside, mut n_in, mut n_out) = (0.0f64, 0.0f64, 0usize, 0usize);
    for y in 0..ih {
        for x in 0..iw {
            for c in 0..3 {
                let i = (y * iw + x) * 3 + c;
                let d = (a[i] - b[i]).abs() as f64;
                if y < ih / 2 && x < iw / 2 {
                    inside += d;
                    n_in += 1;
                } else {
                    outside += d;
                    n_out += 1;
                }
            }
        }
    }
    let (inside, outside) = (inside / n_in as f64, outside / n_out as f64);
    eprintln!("masked edit  inside {inside:.4}  outside {outside:.4}");
    assert!(
        inside > outside * 3.0,
        "the edit moved the outside ({outside:.4}) nearly as much as the inside ({inside:.4}); \
         the mask is not bounding the run"
    );
}

/// **One white pixel frees its whole latent cell**, because `latent_mask`
/// reduces by max and not by mean.
///
/// This is the case that distinguishes them: a latent cell is not a pixel, and
/// averaging would give 1/64 — an almost-frozen cell, producing a hard seam
/// exactly at the mask edge, where it is most visible.
#[test]
fn a_single_masked_pixel_frees_its_whole_latent_cell() {
    let s = Stream::gpu();
    let mut px = vec![0f32; 16 * 16];
    px[0] = 1.0; // top-left pixel only
    let m = Array::from_slice_f32(&px, &[1, 16, 16, 1]).unwrap();

    let lm = sample::latent_mask(&m, &s).unwrap();
    assert_eq!(lm.shape(), vec![1, 2, 2, 1]);
    let v = lm.to_vec_f32(&s).unwrap();
    assert_eq!(
        v,
        vec![1.0, 0.0, 0.0, 0.0],
        "one white pixel must free its whole cell; mean would give {:.4}",
        1.0 / 64.0
    );
}

/// **A mask of all ones is an ordinary img2img.** The composite must be an
/// identity there, or every masked run carries a bias.
#[test]
fn an_open_mask_changes_nothing() {
    let s = Stream::gpu();
    let latent = Array::from_slice_f32(&[1.0, -2.0, 3.0, 0.5], &[1, 2, 2, 1]).unwrap();
    let init = Array::from_slice_f32(&[9.0, 9.0, 9.0, 9.0], &[1, 2, 2, 1]).unwrap();
    let noise = Array::from_slice_f32(&[5.0, 5.0, 5.0, 5.0], &[1, 2, 2, 1]).unwrap();
    let open = Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0], &[1, 2, 2, 1]).unwrap();

    let got = sample::restore_outside_mask(&latent, &init, &open, &noise, 0.7, &s).unwrap();
    assert_eq!(
        got.to_vec_f32(&s).unwrap(),
        latent.to_vec_f32(&s).unwrap(),
        "an open mask must leave the latent exactly as it was"
    );

    // And a closed mask is the other extreme: the original, noised.
    let closed = Array::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 2, 2, 1]).unwrap();
    let got = sample::restore_outside_mask(&latent, &init, &closed, &noise, 0.7, &s).unwrap();
    let want: Vec<f32> = vec![9.0 + 5.0 * 0.7; 4];
    for (g, w) in got.to_vec_f32(&s).unwrap().iter().zip(&want) {
        assert!((g - w).abs() < 1e-5, "closed mask: {g} vs {w}");
    }

    // At the last step there is no noise left to add.
    let got = sample::restore_outside_mask(&latent, &init, &closed, &noise, 0.0, &s).unwrap();
    assert_eq!(
        got.to_vec_f32(&s).unwrap(),
        vec![9.0; 4],
        "the final step must restore the original unnoised"
    );
}
