//! The two tokenizer forms must agree exactly.
//!
//! A stock SD 1.5 or SDXL download ships `vocab.json` + `merges.txt` — the
//! *slow* tokenizer — while `tokenizer.json` is what the fast loader wants.
//! `ClipTokenizer::from_dir` reads either, reconstructing the fast form from
//! the slow one when it has to.
//!
//! **That reconstruction is only safe because of this file.** A normalizer or
//! a split pattern that is nearly right produces nearly-right token ids, which
//! is a different picture with no error anywhere — the exact class of failure
//! this project keeps finding. So the two forms are compared id-for-id over
//! prompts chosen to exercise the parts that differ.
//!
//! ```bash
//! SD_TEST_MODEL_DIR=$(pwd)/models/sd15 \
//!   cargo test -p sd-models --test mlx_tokenizer_forms -- --nocapture
//! ```

use std::path::PathBuf;

use sd_models::clip::ClipTokenizer;

/// Prompts that hit the pieces the two forms could disagree on: case, runs of
/// whitespace, contractions (the split pattern names them individually),
/// digits (split one at a time), punctuation, and non-ASCII that NFC changes.
const PROMPTS: &[&str] = &[
    "a rusty crab on a beach",
    "A RUSTY CRAB",                   // lowercase normalizer
    "a   crab    on\ta\nbeach",       // whitespace collapse
    "it's a crab's beach, isn't it",  // 's / 't in the split pattern
    "12345 crabs and 007 beaches",    // digits split individually
    "crab!!! (beach) — 100%, right?", // punctuation runs
    "un café à la plage, naïve",      // NFC + accents
    "emoji 🦀 on a beach",            // multi-byte, byte-level fallback
    "",                               // empty: BOS then EOS padding
    // overlong: past 77 tokens, so truncation has to keep EOS last
    "a rusty crab on a beach at sunset with waves and gulls and driftwood and \
     salt spray and a lighthouse and a pier and boats and nets and rope and \
     buoys and shells and kelp and sand and footprints and a distant storm",
];

fn model_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SD_TEST_MODEL_DIR").ok()?);
    p.is_dir().then_some(p)
}

/// Write the slow form beside a fast one.
///
/// Asks the tokenizer to save its own model rather than taking the JSON apart:
/// `BPE::save` writes exactly the `vocab.json` + `merges.txt` a stock
/// repository ships. Derived rather than checked in, so if the two disagree it
/// is the *reconstruction* that is wrong and not a stale fixture.
fn slow_form_from(fast: &std::path::Path, out: &std::path::Path) -> std::io::Result<()> {
    use tokenizers::Model;
    std::fs::create_dir_all(out)?;
    let tok = tokenizers::Tokenizer::from_file(fast).expect("tokenizer.json");
    tok.get_model()
        .save(out, None)
        .expect("saving vocab.json and merges.txt");
    Ok(())
}

#[test]
fn the_two_forms_agree_exactly() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    let fast_path = dir.join("tokenizer/tokenizer.json");
    if !fast_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no tokenizer.json to compare against.");
        return;
    }

    let tmp = std::env::temp_dir().join("sdrs-slow-tokenizer");
    slow_form_from(&fast_path, &tmp).expect("deriving the slow form");

    let fast = ClipTokenizer::from_file(&fast_path).expect("fast");
    let slow = ClipTokenizer::from_vocab_and_merges(tmp.join("vocab.json"), tmp.join("merges.txt"))
        .expect("slow");

    for prompt in PROMPTS {
        let a = fast.encode(prompt).expect("fast encode");
        let b = slow.encode(prompt).expect("slow encode");
        assert_eq!(
            a.len(),
            77,
            "CLIP pads to 77 whatever the prompt: {prompt:?}"
        );
        assert_eq!(
            a,
            b,
            "the two forms disagree on {prompt:?}\n  fast {:?}\n  slow {:?}",
            &a[..12.min(a.len())],
            &b[..12.min(b.len())]
        );
    }
    eprintln!(
        "{} prompts agree id-for-id between the two forms",
        PROMPTS.len()
    );
}

/// **`from_dir` prefers the fast form and falls back to the slow one**, so a
/// stock download needs no manual step.
#[test]
fn from_dir_reads_whichever_form_is_present() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    let fast_path = dir.join("tokenizer/tokenizer.json");
    if !fast_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no tokenizer fixture.");
        return;
    }

    // The real directory has tokenizer.json.
    let from_fast = ClipTokenizer::from_dir(dir.join("tokenizer")).expect("fast dir");

    // A directory with only the slow form — what a stock download looks like.
    let tmp = std::env::temp_dir().join("sdrs-slow-only");
    let _ = std::fs::remove_dir_all(&tmp);
    slow_form_from(&fast_path, &tmp).expect("deriving");
    let from_slow = ClipTokenizer::from_dir(&tmp).expect("a stock download must just work");

    let prompt = "a rusty crab on a beach";
    assert_eq!(
        from_fast.encode(prompt).unwrap(),
        from_slow.encode(prompt).unwrap(),
        "from_dir must give the same tokenizer whichever form it found"
    );

    // `from_dir` is the strict spelling and still refuses, naming what it
    // looked for. Only `open` falls back to the vendored copy.
    let empty = std::env::temp_dir().join("sdrs-no-tokenizer");
    std::fs::create_dir_all(&empty).expect("mkdir");
    let err = ClipTokenizer::from_dir(&empty).expect_err("neither form present");
    assert!(
        format!("{err}").contains("tokenizer.json"),
        "the error should name what it looked for, got {err}"
    );
}

/// **The vendored vocabulary must be the same tokenizer.**
///
/// This is the one that makes `open`'s last resort safe. The embedded copy is
/// built through a different code path from the on-disk one — a parsed JSON
/// object and a split merges file, rather than `BPE::from_file` — so "it is the
/// same bytes" is an argument, not a check. Compared id-for-id against a real
/// `tokenizer.json` over the same prompt set.
#[test]
fn the_embedded_vocabulary_is_the_same_tokenizer() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    let fast_path = dir.join("tokenizer/tokenizer.json");
    if !fast_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no tokenizer.json to compare against.");
        return;
    }

    let fast = ClipTokenizer::from_file(&fast_path).expect("fast");
    let embedded = ClipTokenizer::embedded().expect("the vendored vocabulary must load");

    for prompt in PROMPTS {
        assert_eq!(
            fast.encode(prompt).expect("fast"),
            embedded.encode(prompt).expect("embedded"),
            "the embedded vocabulary disagrees on {prompt:?}"
        );
    }

    // The padding token is the one thing that differs between towers, and it
    // is applied on top rather than baked in.
    let bang = embedded
        .with_pad_token("!")
        .expect("SDXL's second tokenizer");
    assert_eq!(bang.pad_token_id(), 0);
    eprintln!(
        "{} prompts agree with the vendored vocabulary",
        PROMPTS.len()
    );
}

/// **`open` never fails for want of a file.**
///
/// A directory with no tokenizer at all — a single-file checkpoint, which is
/// the community norm — still yields a working tokenizer.
#[test]
fn open_falls_back_to_the_vendored_vocabulary() {
    let empty = std::env::temp_dir().join("sdrs-no-tokenizer-at-all");
    let _ = std::fs::remove_dir_all(&empty);
    std::fs::create_dir_all(&empty).expect("mkdir");

    assert!(
        !ClipTokenizer::present(empty.join("tokenizer.json")),
        "present() reports what the checkpoint carries, which is nothing"
    );
    let tok = ClipTokenizer::open(empty.join("tokenizer.json"))
        .expect("open must not fail for want of a file");
    let ids = tok.encode("a rusty crab on a beach").expect("encode");
    assert_eq!(ids.len(), 77);
    assert_eq!(ids[0], tok.bos_token_id());
    assert_eq!(*ids.last().expect("77 ids"), tok.eos_token_id());
}

/// **Neither stock repository ships `tokenizer.json`.**
///
/// Not a hypothetical. `stable-diffusion-v1-5/stable-diffusion-v1-5` and
/// `stabilityai/stable-diffusion-xl-base-1.0` both publish `vocab.json` +
/// `merges.txt` and nothing else — a local `diffusers` cache records the
/// absence under `.no_exist/`. So `open` has to succeed on a path naming a
/// `tokenizer.json` that was never going to be there, and this pins that
/// rather than leaving it to the pipelines to rediscover.
#[test]
fn open_succeeds_on_a_named_fast_form_that_does_not_exist() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    let fast_path = dir.join("tokenizer/tokenizer.json");
    if !fast_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no tokenizer fixture.");
        return;
    }

    // A directory laid out exactly as a stock download is: slow form only.
    let stock = std::env::temp_dir().join("sdrs-stock-layout");
    let _ = std::fs::remove_dir_all(&stock);
    slow_form_from(&fast_path, &stock).expect("deriving");
    assert!(!stock.join("tokenizer.json").exists(), "slow form only");

    // What a pipeline holds is the fast form's path, because that is what the
    // layout names. It must still load.
    let named = stock.join("tokenizer.json");
    assert!(
        ClipTokenizer::present(&named),
        "present() must see the slow form beside the name it was given"
    );
    let tok = ClipTokenizer::open(&named).expect("a stock download must just work");
    assert_eq!(
        tok.encode("a rusty crab on a beach").unwrap(),
        ClipTokenizer::from_file(&fast_path)
            .unwrap()
            .encode("a rusty crab on a beach")
            .unwrap()
    );

    // The slow form beside it is what was used, not the vendored copy — which
    // is the point of preferring the checkpoint's own files. Checked by
    // removing them and seeing `present` change its answer.
    std::fs::remove_file(stock.join("vocab.json")).expect("rm");
    assert!(!ClipTokenizer::present(&named));
}
