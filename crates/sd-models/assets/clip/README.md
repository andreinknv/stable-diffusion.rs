# CLIP's BPE vocabulary

`vocab.json` and `merges.txt`, verbatim from
[`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14),
which OpenAI publishes under the MIT licence.

**One copy serves every model in this project.** Checked rather than assumed:
`stabilityai/stable-diffusion-xl-base-1.0`'s `tokenizer_2` — the OpenCLIP bigG
tower, a different text encoder trained by a different organisation — ships
these two files byte for byte identical, all 49,408 entries and all 524,619
bytes of merges. SD 1.x, SD 2.x, SDXL's two towers, SD 3.x and Flux's CLIP
tower therefore share one vocabulary.

What differs between them is the *padding token*, not the vocabulary, and that
is `ClipTokenizer::with_pad_token`.

These are vendored so that a checkpoint which ships no tokenizer still runs.
A tokenizer found in the model directory always wins; see
`clip::tokenizer::ClipTokenizer::open`.
