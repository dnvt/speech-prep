//! Error types for speech-prep.

/// Errors that can occur during audio preprocessing.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Processing(String),

    #[error("{0}")]
    InvalidInput(String),

    #[error("{0}")]
    Configuration(String),

    #[error("{0}")]
    TemporalOperation(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(any(test, feature = "fixtures"))]
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn processing(msg: impl Into<String>) -> Self {
        Self::Processing(msg.into())
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    pub fn configuration(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }

    pub fn temporal_operation(msg: impl Into<String>) -> Self {
        Self::TemporalOperation(msg.into())
    }

    pub fn invalid_format(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    pub fn empty_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
