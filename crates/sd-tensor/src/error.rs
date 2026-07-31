//! The crate's error type.
//!
//! Native rather than re-exported. It was `candle_core::Error` for as long as
//! candle was the backend, and every caller matched on `Error::Msg` — so this
//! keeps that variant and nothing else changes at a call site.
//!
//! **Deliberately small.** A tensor library's errors are almost all "this
//! shape does not work with that shape", which is a sentence, not a taxonomy.
//! The variants below exist because a caller does something different with
//! them: an IO failure names a file, and a memory refusal is the guard
//! declining rather than something being wrong.

/// What went wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A description. The overwhelmingly common case.
    #[error("{0}")]
    Msg(String),

    /// Reading or writing a file.
    #[error("{context}: {source}")]
    Io {
        context: String,
        source: std::io::Error,
    },

    /// The memory guard declining a request — **not** a failure.
    ///
    /// Separate because a caller can respond: tile the decode, drop a stage to
    /// the CPU, or ask for a smaller image. Folding it into `Msg` would make
    /// "too big for this machine" indistinguishable from "wrong shape".
    #[error("{0}")]
    Refused(String),
}

impl Error {
    /// Wrap an IO error with what was being read or written.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Self::Io {
            context: "io".into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal must stay distinguishable from a fault, because a caller
    /// responds differently to each.
    #[test]
    fn a_refusal_is_not_a_message() {
        let refused = Error::Refused("2.4 GB exceeds the budget".into());
        let faulty = Error::Msg("2.4 GB exceeds the budget".into());
        assert!(matches!(refused, Error::Refused(_)));
        assert!(matches!(faulty, Error::Msg(_)));
        // They render the same, which is deliberate: the difference is for the
        // program, and the user sees one sentence either way.
        assert_eq!(refused.to_string(), faulty.to_string());
    }

    /// An IO error says what it was doing, not just what the OS said.
    #[test]
    fn an_io_error_names_its_context() {
        let e = Error::io(
            "opening models/sd15/unet.safetensors",
            std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        );
        let text = e.to_string();
        assert!(text.contains("unet.safetensors"), "{text}");
        assert!(text.contains("no such file"), "{text}");
    }
}
