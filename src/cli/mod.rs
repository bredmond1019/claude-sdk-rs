//! Interactive CLI for managing multiple Claude sessions and agents in parallel.
//! ## Usage
//!
//! ```bash
//! # List available commands
//! claude-interactive list
//!
//! # Create a new session
//! claude-interactive session create "my-project"
//!
//! # Run a command
//! claude-interactive run my-command --session my-project
//!
//! # View costs
//! claude-interactive cost --session my-project
//!
//! # Search history
//! claude-interactive history --search "rust error"
//! ```

pub mod analytics;
pub mod cli;
pub mod commands;
pub mod config;
pub mod cost;
pub mod error;
pub mod execution;
pub mod history;
pub mod output;
pub mod profiling;
pub mod session;

// pub mod testing; // Temporarily disabled due to analytics dependency

#[cfg(test)]
mod error_test;

#[cfg(test)]
mod cli_simple_test;

// Re-export commonly used types
pub use error::{InteractiveError, Result, UserFriendlyError};

/// Version information for the interactive CLI
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default timeout for operations in seconds
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Default directory for storing session data
pub fn default_data_dir() -> std::path::PathBuf {
    directories::ProjectDirs::from("", "", "claude-sdk-rs")
        .map(|dirs| dirs.data_dir().to_path_buf())
        .unwrap_or_else(|| {
            // Fallback to home directory if project dirs not available
            dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".claude-sdk-rs")
        })
}

/// Initialize the data directory if it doesn't exist
pub fn ensure_data_dir() -> Result<std::path::PathBuf> {
    let data_dir = default_data_dir();
    if !data_dir.exists() {
        std::fs::create_dir_all(&data_dir)?;
    }
    Ok(data_dir)
}
