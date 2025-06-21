//! Runtime execution and process management for the Claude AI SDK.

/// Backpressure monitoring and management for streaming operations.
pub mod backpressure;
/// Main client implementation for Claude AI SDK.
pub mod client;
pub mod error_handling;
/// Process execution and management for Claude CLI.
pub mod process;
pub mod recovery;
/// Stream processing and message parsing utilities.
pub mod stream;
/// Configuration for streaming operations and buffering.
pub mod stream_config;
pub mod telemetry;

pub use backpressure::{BackpressureMonitor, BackpressureSender};
pub use client::{Client, QueryBuilder};
pub use error_handling::{
    log_error_with_context, retry_with_backoff, ErrorContext, ProcessErrorDetails, RetryConfig,
};
pub use recovery::{
    CircuitBreaker, CircuitState, PartialResultRecovery, StreamReconnectionManager,
    TokenBucketRateLimiter,
};
pub use stream::MessageStream;
pub use stream_config::{get_stream_config, set_stream_config, StreamConfig, StreamConfigBuilder};
pub use telemetry::{
    init_telemetry, record_error, record_recovery, telemetry, ErrorTelemetry, TelemetryConfig,
};
