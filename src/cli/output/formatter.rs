//! Output formatting and display utilities
//!
//! This module provides comprehensive output formatting for Claude command results,
//! supporting colored output, agent identification, table formatting, and progress indicators.

use crate::cli::execution::ExecutionResult;
use crate::cli::output::AgentId;
use crate::cli::session::Session;
use crate::{cli::error::InteractiveError, cli::error::Result};
use chrono::Utc;
use colored::{Color, Colorize};
use prettytable::{format, Cell, Row, Table};
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Duration;

/// Configuration for output formatting
#[derive(Debug, Clone)]
pub struct FormatterConfig {
    /// Whether to use colors in output
    pub use_colors: bool,
    /// Whether to show timestamps
    pub show_timestamps: bool,
    /// Whether to show agent IDs
    pub show_agent_ids: bool,
    /// Maximum width for output lines
    pub max_width: Option<usize>,
    /// Indentation level
    pub indent_level: usize,
    /// Whether to show cost information
    pub show_costs: bool,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            use_colors: true,
            show_timestamps: true,
            show_agent_ids: true,
            max_width: Some(120),
            indent_level: 0,
            show_costs: true,
        }
    }
}

/// Output formatting styles
#[derive(Debug, Clone, Copy)]
pub enum OutputStyle {
    Plain,
    Success,
    Error,
    Warning,
    Info,
    Debug,
    Header,
    Highlight,
}

impl OutputStyle {
    /// Get the color associated with this style
    pub fn color(&self) -> Color {
        match self {
            Self::Plain => Color::White,
            Self::Success => Color::Green,
            Self::Error => Color::Red,
            Self::Warning => Color::Yellow,
            Self::Info => Color::Blue,
            Self::Debug => Color::Magenta,
            Self::Header => Color::Cyan,
            Self::Highlight => Color::BrightWhite,
        }
    }

    /// Get the prefix symbol for this style
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Plain => "",
            Self::Success => "✓",
            Self::Error => "✗",
            Self::Warning => "⚠",
            Self::Info => "ℹ",
            Self::Debug => "🐛",
            Self::Header => "📋",
            Self::Highlight => "→",
        }
    }
}

/// Progress indicator for long-running operations
#[derive(Debug)]
pub struct ProgressIndicator {
    current: usize,
    total: usize,
    label: String,
    width: usize,
}

impl ProgressIndicator {
    /// Create a new progress indicator
    pub fn new(total: usize, label: String) -> Self {
        Self {
            current: 0,
            total,
            label,
            width: 50,
        }
    }

    /// Update progress
    pub fn update(&mut self, current: usize) {
        self.current = current;
    }

    /// Increment progress by 1
    pub fn increment(&mut self) {
        if self.current < self.total {
            self.current += 1;
        }
    }

    /// Format the progress bar
    pub fn format(&self) -> String {
        let percentage = if self.total > 0 {
            (self.current as f64 / self.total as f64 * 100.0) as usize
        } else {
            100
        };

        let filled = (self.current as f64 / self.total as f64 * self.width as f64) as usize;
        let empty = self.width - filled;

        let bar = format!(
            "[{}{}]",
            "█".repeat(filled).green(),
            "░".repeat(empty).bright_black()
        );

        format!(
            "{} {} {}/{} ({}%)",
            self.label.cyan(),
            bar,
            self.current,
            self.total,
            percentage
        )
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }

    /// Get current progress value
    pub fn current(&self) -> usize {
        self.current
    }

    /// Get total progress value
    pub fn total(&self) -> usize {
        self.total
    }
}

/// Comprehensive output formatter
#[derive(Clone)]
pub struct OutputFormatter {
    config: FormatterConfig,
    agent_colors: HashMap<usize, Color>,
}

impl OutputFormatter {
    /// Create a new output formatter with default configuration
    pub fn new() -> Self {
        Self {
            config: FormatterConfig::default(),
            agent_colors: HashMap::new(),
        }
    }

    /// Create a new output formatter with custom configuration
    pub fn with_config(config: FormatterConfig) -> Self {
        Self {
            config,
            agent_colors: HashMap::new(),
        }
    }

    /// Format a message with the given style
    pub fn format_message(&self, message: &str, style: OutputStyle) -> String {
        let mut formatted = String::new();

        // Add indent
        if self.config.indent_level > 0 {
            formatted.push_str(&" ".repeat(self.config.indent_level * 2));
        }

        // Add timestamp if enabled
        if self.config.show_timestamps {
            let timestamp = Utc::now().format("%H:%M:%S%.3f").to_string();
            if self.config.use_colors {
                formatted.push_str(&format!("[{}] ", timestamp.bright_black()));
            } else {
                formatted.push_str(&format!("[{}] ", timestamp));
            }
        }

        // Add style symbol and format message
        let symbol = style.symbol();
        if !symbol.is_empty() {
            if self.config.use_colors {
                formatted.push_str(&format!("{} ", symbol.color(style.color())));
            } else {
                formatted.push_str(&format!("{} ", symbol));
            }
        }

        // Apply color and word wrapping
        let content = if let Some(max_width) = self.config.max_width {
            self.wrap_text(message, max_width - formatted.len())
        } else {
            message.to_string()
        };

        if self.config.use_colors {
            formatted.push_str(&content.color(style.color()).to_string());
        } else {
            formatted.push_str(&content);
        }

        formatted
    }

    /// Format output with agent identification
    pub fn format_agent_output(&mut self, output: &str, agent_id: AgentId) -> String {
        // Get or assign color for this agent
        let color = *self
            .agent_colors
            .entry(agent_id.0)
            .or_insert_with(|| agent_id.color());

        let mut lines = Vec::new();

        for line in output.lines() {
            let mut formatted = String::new();

            // Add indent
            if self.config.indent_level > 0 {
                formatted.push_str(&" ".repeat(self.config.indent_level * 2));
            }

            // Add timestamp if enabled
            if self.config.show_timestamps {
                let timestamp = Utc::now().format("%H:%M:%S%.3f").to_string();
                if self.config.use_colors {
                    formatted.push_str(&format!("[{}] ", timestamp.bright_black()));
                } else {
                    formatted.push_str(&format!("[{}] ", timestamp));
                }
            }

            // Add agent ID if enabled
            if self.config.show_agent_ids {
                if self.config.use_colors {
                    formatted
                        .push_str(&format!("[Agent {}] ", agent_id.0.to_string().color(color)));
                } else {
                    formatted.push_str(&format!("[Agent {}] ", agent_id.0));
                }
            }

            // Add the actual content
            if self.config.use_colors {
                formatted.push_str(&line.color(color).to_string());
            } else {
                formatted.push_str(line);
            }

            lines.push(formatted);
        }

        lines.join("\n")
    }

    /// Format execution result for display
    pub fn format_execution_result(
        &mut self,
        result: &ExecutionResult,
        agent_id: Option<AgentId>,
    ) -> String {
        let mut output = String::new();

        // Format success/failure header
        let status_style = if result.success {
            OutputStyle::Success
        } else {
            OutputStyle::Error
        };
        let status_text = if result.success { "SUCCESS" } else { "FAILED" };

        output.push_str(&self.format_message(
            &format!("Command {} in {:?}", status_text, result.duration),
            status_style,
        ));
        output.push('\n');

        // Format cost information if available and enabled
        if self.config.show_costs {
            if let Some(cost) = result.cost {
                output.push_str(
                    &self.format_message(&format!("Cost: ${:.6}", cost), OutputStyle::Info),
                );
                output.push('\n');
            }
        }

        // Format the actual output
        if !result.output.trim().is_empty() {
            output.push_str(&self.format_message("Output:", OutputStyle::Header));
            output.push('\n');

            if let Some(agent) = agent_id {
                output.push_str(&self.format_agent_output(&result.output, agent));
            } else {
                output.push_str(&self.format_message(&result.output, OutputStyle::Plain));
            }
            output.push('\n');
        }

        output
    }

    /// Format session information as a table
    pub fn format_session_table(&self, sessions: &[Session]) -> Result<String> {
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);

        // Set headers
        table.set_titles(Row::new(vec![
            Cell::new("ID").style_spec(if self.config.use_colors { "Fc" } else { "" }),
            Cell::new("Name").style_spec(if self.config.use_colors { "Fc" } else { "" }),
            Cell::new("Description").style_spec(if self.config.use_colors { "Fc" } else { "" }),
            Cell::new("Created").style_spec(if self.config.use_colors { "Fc" } else { "" }),
            Cell::new("Last Active").style_spec(if self.config.use_colors { "Fc" } else { "" }),
            Cell::new("Commands").style_spec(if self.config.use_colors { "Fc" } else { "" }),
            Cell::new("Cost").style_spec(if self.config.use_colors { "Fc" } else { "" }),
        ]));

        // Add rows
        for session in sessions {
            let id_short = session.id.to_string().chars().take(8).collect::<String>();
            let description = session.description.as_deref().unwrap_or("-").to_string();
            let created = session.created_at.format("%Y-%m-%d %H:%M").to_string();
            let last_active = session.last_accessed.format("%Y-%m-%d %H:%M").to_string();
            let commands = session
                .metadata
                .get("total_commands")
                .unwrap_or(&"0".to_string())
                .to_string();
            let cost = format!(
                "${:.6}",
                session
                    .metadata
                    .get("total_cost")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0)
            );

            table.add_row(Row::new(vec![
                Cell::new(&id_short),
                Cell::new(&session.name),
                Cell::new(&description),
                Cell::new(&created),
                Cell::new(&last_active),
                Cell::new(&commands),
                Cell::new(&cost),
            ]));
        }

        Ok(table.to_string())
    }

    /// Format parallel execution summary
    pub fn format_parallel_summary(
        &self,
        total: usize,
        successful: usize,
        failed: usize,
        duration: Duration,
    ) -> String {
        let mut output = String::new();

        output.push_str(&self.format_message("Parallel Execution Summary", OutputStyle::Header));
        output.push('\n');

        output.push_str(
            &self.format_message(&format!("Total commands: {}", total), OutputStyle::Info),
        );
        output.push('\n');

        output.push_str(
            &self.format_message(&format!("Successful: {}", successful), OutputStyle::Success),
        );
        output.push('\n');

        if failed > 0 {
            output
                .push_str(&self.format_message(&format!("Failed: {}", failed), OutputStyle::Error));
            output.push('\n');
        }

        output.push_str(&self.format_message(
            &format!("Total duration: {:?}", duration),
            OutputStyle::Info,
        ));
        output.push('\n');

        let success_rate = if total > 0 {
            (successful as f64 / total as f64 * 100.0) as usize
        } else {
            100
        };

        let style = if success_rate >= 90 {
            OutputStyle::Success
        } else if success_rate >= 70 {
            OutputStyle::Warning
        } else {
            OutputStyle::Error
        };

        output.push_str(&self.format_message(&format!("Success rate: {}%", success_rate), style));

        output
    }

    /// Print formatted output to stdout
    pub fn print(&self, text: &str) -> Result<()> {
        print!("{}", text);
        io::stdout().flush().map_err(|e| InteractiveError::Io(e))?;
        Ok(())
    }

    /// Display a progress bar (overwrites current line)
    pub fn print_progress(&self, progress: &ProgressIndicator) -> Result<()> {
        print!("\r{}", progress.format());
        io::stdout().flush().map_err(|e| InteractiveError::Io(e))?;
        Ok(())
    }

    /// Clear the current line and finish progress display
    pub fn finish_progress(&self, final_message: Option<&str>) -> Result<()> {
        print!("\r{}", " ".repeat(80)); // Clear line
        if let Some(msg) = final_message {
            println!("\r{}", msg);
        } else {
            println!();
        }
        io::stdout().flush().map_err(|e| InteractiveError::Io(e))?;
        Ok(())
    }

    /// Create a progress indicator for operations
    pub fn create_progress(&self, total: usize, label: &str) -> ProgressIndicator {
        ProgressIndicator::new(total, label.to_string())
    }

    /// Print formatted output with newline to stdout
    pub fn println(&self, text: &str) -> Result<()> {
        println!("{}", text);
        Ok(())
    }

    /// Print error message to stderr
    pub fn print_error(&self, message: &str) -> Result<()> {
        let formatted = self.format_message(message, OutputStyle::Error);
        eprintln!("{}", formatted);
        Ok(())
    }

    /// Clear the current line (useful for progress indicators)
    pub fn clear_line(&self) -> Result<()> {
        print!("\r\x1b[K");
        io::stdout().flush().map_err(|e| InteractiveError::Io(e))?;
        Ok(())
    }

    /// Word wrap text to fit within specified width
    fn wrap_text(&self, text: &str, width: usize) -> String {
        if width == 0 {
            return text.to_string();
        }

        let mut lines = Vec::new();
        for line in text.lines() {
            if line.len() <= width {
                lines.push(line.to_string());
            } else {
                let mut current_line = String::new();
                for word in line.split_whitespace() {
                    if current_line.len() + word.len() + 1 > width && !current_line.is_empty() {
                        lines.push(current_line);
                        current_line = String::new();
                    }

                    if !current_line.is_empty() {
                        current_line.push(' ');
                    }
                    current_line.push_str(word);
                }

                if !current_line.is_empty() {
                    lines.push(current_line);
                }
            }
        }

        lines.join("\n")
    }

    /// Get current configuration
    pub fn config(&self) -> &FormatterConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: FormatterConfig) {
        self.config = config;
    }

    /// Reset agent colors
    pub fn reset_agent_colors(&mut self) {
        self.agent_colors.clear();
    }
}

impl Default for OutputFormatter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::session::SessionMetadata;
    use chrono::Utc;
    use std::time::Duration;
    use uuid::Uuid;

    #[test]
    fn test_formatter_creation() {
        let formatter = OutputFormatter::new();
        assert!(formatter.config.use_colors);
        assert!(formatter.config.show_timestamps);
    }

    #[test]
    fn test_message_formatting() {
        let formatter = OutputFormatter::new();
        let message = formatter.format_message("Test message", OutputStyle::Success);

        // Should contain the success symbol
        assert!(message.contains("✓"));
    }

    #[test]
    fn test_agent_output_formatting() {
        let mut formatter = OutputFormatter::new();
        let agent_id = AgentId(1);
        let output = formatter.format_agent_output("Test output", agent_id);

        // Should contain agent identifier
        assert!(output.contains("Agent 1"));
    }

    #[test]
    fn test_execution_result_formatting() {
        let mut formatter = OutputFormatter::new();
        let result = ExecutionResult {
            output: "Command output".to_string(),
            cost: Some(0.001),
            duration: Duration::from_millis(500),
            success: true,
        };

        let formatted = formatter.format_execution_result(&result, None);
        assert!(formatted.contains("SUCCESS"));
        assert!(formatted.contains("Command output"));
        assert!(formatted.contains("Cost:"));
    }

    #[test]
    fn test_session_table_formatting() {
        let formatter = OutputFormatter::new();
        let session = Session {
            id: Uuid::new_v4(),
            name: "Test Session".to_string(),
            description: Some("Test description".to_string()),
            created_at: Utc::now(),
            last_active: Utc::now(),
            metadata: SessionMetadata::default(),
        };

        let table = formatter.format_session_table(&[session]).unwrap();
        assert!(table.contains("Test Session"));
        assert!(table.contains("Test description"));
    }

    #[test]
    fn test_progress_indicator() {
        let mut progress = ProgressIndicator::new(10, "Testing".to_string());

        assert_eq!(progress.current, 0);
        assert!(!progress.is_complete());

        progress.update(5);
        assert_eq!(progress.current, 5);

        progress.update(10);
        assert!(progress.is_complete());

        let formatted = progress.format();
        assert!(formatted.contains("Testing"));
        assert!(formatted.contains("10/10"));
    }

    #[test]
    fn test_parallel_summary_formatting() {
        let formatter = OutputFormatter::new();
        let summary = formatter.format_parallel_summary(10, 8, 2, Duration::from_secs(30));

        assert!(summary.contains("Total commands: 10"));
        assert!(summary.contains("Successful: 8"));
        assert!(summary.contains("Failed: 2"));
        assert!(summary.contains("Success rate: 80%"));
    }

    #[test]
    fn test_text_wrapping() {
        let formatter = OutputFormatter::new();
        let long_text = "This is a very long line that should be wrapped to fit within the specified width limit";
        let wrapped = formatter.wrap_text(long_text, 20);

        for line in wrapped.lines() {
            assert!(line.len() <= 20);
        }
    }

    #[test]
    fn test_output_styles() {
        assert_eq!(OutputStyle::Success.symbol(), "✓");
        assert_eq!(OutputStyle::Error.symbol(), "✗");
        assert_eq!(OutputStyle::Warning.symbol(), "⚠");
        assert_eq!(OutputStyle::Success.color(), Color::Green);
        assert_eq!(OutputStyle::Error.color(), Color::Red);
    }
}
