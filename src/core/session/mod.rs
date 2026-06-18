/// Session management module
#[allow(clippy::module_inception)]
pub mod session;

/// SQLite-backed persistent session storage (requires the `sqlite` feature).
#[cfg(feature = "sqlite")]
pub mod sqlite_storage;

pub use session::*;
