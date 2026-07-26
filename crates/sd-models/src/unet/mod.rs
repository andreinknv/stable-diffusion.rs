//! UNet: the denoiser.
//!
//! Built bottom-up and verified block by block. A whole-model comparison on a
//! net this size tells you only that something is wrong, never what.

mod attention;
mod embeddings;
mod resnet;

pub use attention::{Attention, BasicTransformerBlock, FeedForward, Transformer2DModel};
pub use embeddings::{timestep_embedding, TimestepEmbedding};
pub use resnet::ResnetBlock2D;
