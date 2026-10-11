//! Error type shared by every module.

/// Errors returned by `speech-prep`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The input audio or samples are malformed or empty.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The input is valid audio, but its format or codec is not supported.
    #[error("unsupported audio: {0}")]
    Unsupported(String),

    /// A configuration value is out of range.
    #[error("invalid configuration: {0}")]
    Configuration(String),

    /// An internal processing step failed.
    #[error("processing failed: {0}")]
    Processing(String),

    /// Reading input failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// `Result` alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
