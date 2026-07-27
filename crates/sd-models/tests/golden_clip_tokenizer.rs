//! Golden verification for the CLIP tokenizer.
//!
//! Two kinds of test, matching `golden_vae.rs`:
//!
//! * **Structural** — the padding contract (exactly 77 ids, EOS padding, EOS
//!   last after truncation). These need the vocabulary but no reference, and
//!   they are where the classic mistakes live.
//! * **Reference** — id-for-id agreement with HuggingFace. Skips when
//!   `tests/golden/clip_tokenizer/` is absent, so CI stays green without
//!   committing the vocabulary.
//!
//! The reference ids come from Python's `CLIPTokenizer` while the Rust side
//! loads `tokenizer.json`, so agreement is a genuine cross-check between two
//! implementations rather than one restating the other.

use std::path::PathBuf;

use sd_models::clip::ClipTokenizer;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/clip_tokenizer")
}

/// The vocabulary, or `None` when the reference data was never generated.
fn tokenizer() -> Option<ClipTokenizer> {
    let path = golden_dir().join("tokenizer.json");
    if !path.exists() {
        eprintln!(
            "SKIP: no tokenizer at {}.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py clip_tokenizer --output tests/golden\n",
            path.display()
        );
        return None;
    }
    Some(ClipTokenizer::from_file(&path).expect("loading tokenizer.json"))
}

#[test]
fn encodes_to_exactly_77_ids() {
    let Some(tok) = tokenizer() else { return };
    for prompt in [
        "",
        "a rusty crab on a beach",
        "a photo of an astronaut riding a horse on mars",
        &"a ".repeat(200),
    ] {
        let ids = tok.encode(prompt).expect("encode");
        assert_eq!(
            ids.len(),
            77,
            "prompt {prompt:?} encoded to {} ids",
            ids.len()
        );
        assert_eq!(ids[0], tok.bos_token_id(), "must start with BOS");
        assert_eq!(ids[76], tok.eos_token_id(), "must end with EOS");
    }
}

#[test]
fn empty_prompt_is_bos_then_all_eos() {
    let Some(tok) = tokenizer() else { return };
    let ids = tok.encode("").expect("encode");

    let mut expected = vec![tok.eos_token_id(); 77];
    expected[0] = tok.bos_token_id();
    assert_eq!(ids, expected);
}

#[test]
fn padding_uses_eos_and_never_zero() {
    // The single most common mistake in this task: padding with 0 produces a
    // plausible-looking vector that yields subtly wrong embeddings later.
    let Some(tok) = tokenizer() else { return };
    let ids = tok.encode("a rusty crab on a beach").expect("encode");

    let eos = tok.eos_token_id();
    let first_eos = ids.iter().position(|&id| id == eos).expect("EOS present");
    assert!(
        ids[first_eos..].iter().all(|&id| id == eos),
        "everything after the prompt must be EOS, got {:?}",
        &ids[first_eos..]
    );
    assert!(!ids.contains(&0), "0 is a real token id, not padding");
}

#[test]
fn overlong_prompt_truncates_with_eos_last() {
    let Some(tok) = tokenizer() else { return };
    // 200 words cannot fit in 77 slots, so this exercises truncation rather
    // than padding.
    let ids = tok.encode(&"a ".repeat(200)).expect("encode");

    assert_eq!(ids.len(), 77);
    assert_eq!(ids[0], tok.bos_token_id());
    assert_eq!(
        ids[76],
        tok.eos_token_id(),
        "truncation must still leave EOS last; a naive ids[..77] does not"
    );
    // The interior must be real prompt tokens — if truncation collapsed to
    // padding, this would be all EOS and the assertion above would still pass.
    assert!(
        ids[1..76].iter().all(|&id| id != tok.eos_token_id()),
        "an overlong prompt should fill every slot before the final EOS"
    );
}

#[test]
fn matches_huggingface_reference() {
    let dir = golden_dir();
    let reference = dir.join("reference.json");
    if !reference.exists() {
        eprintln!(
            "SKIP matches_huggingface_reference: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py clip_tokenizer --output tests/golden\n\
             \nSee xtask/golden/README.md."
        );
        return;
    }
    let Some(tok) = tokenizer() else { return };

    let raw = std::fs::read_to_string(&reference).expect("reading reference.json");
    let prompts = json_string_array(&raw, "prompts");
    let expected = json_id_rows(&raw, "ids");
    assert_eq!(
        prompts.len(),
        expected.len(),
        "reference.json is malformed: {} prompts, {} id rows",
        prompts.len(),
        expected.len()
    );
    assert!(!prompts.is_empty(), "reference.json has no prompts");

    for (prompt, want) in prompts.iter().zip(&expected) {
        let got = tok.encode(prompt).expect("encode");
        // Report the first divergence rather than dumping 77 ids twice.
        if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
            let shown: String = prompt.chars().take(40).collect();
            panic!(
                "prompt {shown:?} diverges at index {i}: expected {}, got {}\n  expected: {:?}\n  got:      {:?}",
                want[i],
                got[i],
                &want[..want.len().min(12)],
                &got[..got.len().min(12)],
            );
        }
        assert_eq!(got.len(), want.len(), "length differs for {prompt:?}");
    }
    eprintln!("clip tokenizer: {} prompts match id for id", prompts.len());
}

// -- minimal JSON readers -------------------------------------------------
//
// `serde_json` is not a dependency of this workspace and the task forbids
// adding one, so these pull the two fields the test needs out of the file
// written by dump_reference.py. They are deliberately narrow: they understand
// that file's shape and nothing else, and panic loudly rather than guessing.

/// Extract `"key": ["a", "b"]`.
fn json_string_array(raw: &str, key: &str) -> Vec<String> {
    let body = json_bracketed_value(raw, key);
    let mut out = Vec::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut s = String::new();
        // The prompts contain no escapes beyond `\"`; handle that and nothing
        // else, so an unexpected escape shows up as a mismatch rather than
        // being silently mangled.
        while let Some(c) = chars.next() {
            match c {
                '\\' => s.push(chars.next().expect("escape at end of string")),
                '"' => break,
                _ => s.push(c),
            }
        }
        out.push(s);
    }
    out
}

/// Extract `"key": [[1, 2], [3, 4]]`.
fn json_id_rows(raw: &str, key: &str) -> Vec<Vec<u32>> {
    let body = json_bracketed_value(raw, key);
    let mut rows = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find('[') {
        let end = rest[start..].find(']').expect("unterminated id row") + start;
        rows.push(
            rest[start + 1..end]
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<u32>().expect("id is not a u32"))
                .collect(),
        );
        rest = &rest[end + 1..];
    }
    rows
}

/// The `[...]` following `"key":`, with brackets balanced.
fn json_bracketed_value<'a>(raw: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\"");
    let at = raw
        .find(&needle)
        .unwrap_or_else(|| panic!("reference.json has no {needle} field"));
    let open = raw[at..].find('[').expect("field is not an array") + at;

    let mut depth = 0usize;
    for (i, c) in raw[open..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return &raw[open + 1..open + i];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated array for {needle}");
}

#[test]
fn token_counts_are_reported_and_are_unintuitive() {
    // The counts callers cannot guess. Every one of these is a real budgeting
    // surprise: BPE splits digits singly, and punctuation is not free.
    let Some(tok) = tokenizer() else { return };

    assert_eq!(tok.content_token_count("cat").unwrap(), 1);
    // Digits one at a time: 1, 6, -, bit.
    assert_eq!(tok.content_token_count("16-bit").unwrap(), 4);
    // 3, 2, x, 3, 2.
    assert_eq!(tok.content_token_count("32x32").unwrap(), 5);
    // Commas count.
    assert_eq!(
        tok.content_token_count("a cat, a dog").unwrap(),
        tok.content_token_count("a cat a dog").unwrap() + 1
    );
}

#[test]
fn over_limit_prompts_truncate_rather_than_chunk() {
    // The behaviour a caller has to know to budget: past the limit, tokens are
    // *discarded*, not encoded into a second window. Getting this wrong the
    // other way — assuming chunking — means believing a trailing qualifier
    // still applies when it does not.
    let Some(tok) = tokenizer() else { return };

    assert_eq!(tok.content_capacity(), 75, "77 minus BOS and EOS");
    let short = "a cat";
    assert!(!tok.will_truncate(short).unwrap());

    let long = "cat ".repeat(100);
    assert!(tok.will_truncate(&long).unwrap());
    let ids = tok.encode(&long).unwrap();
    assert_eq!(ids.len(), 77, "always exactly the context length");
    assert_eq!(
        *ids.last().unwrap(),
        tok.eos_token_id(),
        "truncation still ends in EOS"
    );
}
