//! Output formatting and display utilities
//!
//! This module handles:
//! - Formatting command outputs
//! - Color coding for different agents
//! - Table formatting for data display
//! - Progress indicators

pub mod formatter;

// Re-export key types for convenience
pub use formatter::{FormatterConfig, OutputStyle, ProgressIndicator};

use colored::Colorize;

/// Agent identifier for colored output
#[derive(Debug, Clone, Copy)]
pub struct AgentId(pub usize);

impl AgentId {
    /// Get a color for this agent
    pub fn color(&self) -> colored::Color {
        match self.0 % 6 {
            0 => colored::Color::Red,
            1 => colored::Color::Green,
            2 => colored::Color::Yellow,
            3 => colored::Color::Blue,
            4 => colored::Color::Magenta,
            5 => colored::Color::Cyan,
            _ => colored::Color::White,
        }
    }

    /// Format a message with this agent's color
    pub fn format_message(&self, message: &str) -> String {
        format!("[Agent {}] {}", self.0, message)
            .color(self.color())
            .to_string()
    }
}

// OutputFormatter is now implemented in the formatter module
