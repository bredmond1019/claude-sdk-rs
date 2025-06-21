//! Command execution engine
//!
//! This module handles:
//! - Executing Claude commands with session context
//! - Parallel execution of multiple agents
//! - Output management and formatting
//! - Real-time streaming support

pub mod parallel;
pub mod runner;

// #[cfg(test)]
// pub mod execution_test; // Temporarily disabled

// Re-export key types for convenience
pub use parallel::{
    ExecutionHandle, ParallelConfig, ParallelExecutor, ParallelOutput, ParallelResult,
    ParallelStats,
};
pub use runner::RunnerConfig;

use crate::cli::session::SessionId;

/// Command execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub session_id: Option<SessionId>,
    pub command_name: String,
    pub args: Vec<String>,
    pub parallel: bool,
    pub agent_count: usize,
}

/// Result of command execution
#[derive(Debug)]
pub struct ExecutionResult {
    pub output: String,
    pub cost: Option<f64>,
    pub duration: std::time::Duration,
    pub success: bool,
}

// CommandRunner is now implemented in the runner module
