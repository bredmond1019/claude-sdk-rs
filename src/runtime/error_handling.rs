//! Advanced error handling utilities for Claude AI SDK
//!
//! This module provides enhanced error handling capabilities including:
//! - Contextual error information for debugging
//! - Retry mechanisms for recoverable errors
//! - Error recovery strategies
//! - Comprehensive error logging

use crate::core::{Error, Result};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, warn};

/// Enhanced error context for debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The operation being performed when the error occurred
    pub operation: String,
    /// Additional debugging information
    pub debug_info: Vec<(String, String)>,
    /// The timestamp when the error occurred
    pub timestamp: std::time::SystemTime,
    /// The error chain leading to this error
    pub error_chain: Vec<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            debug_info: Vec::new(),
            timestamp: std::time::SystemTime::now(),
            error_chain: Vec::new(),
        }
    }

    /// Add debugging information
    pub fn with_debug_info(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.debug_info.push((key.into(), value.into()));
        self
    }

    /// Add an error to the chain
    pub fn with_error_chain(mut self, error: impl Into<String>) -> Self {
        self.error_chain.push(error.into());
        self
    }

    /// Convert to a formatted debug string
    pub fn to_debug_string(&self) -> String {
        let mut debug_str = format!("Operation: {}\n", self.operation);
        debug_str.push_str(&format!("Timestamp: {:?}\n", self.timestamp));

        if !self.debug_info.is_empty() {
            debug_str.push_str("Debug Info:\n");
            for (key, value) in &self.debug_info {
                debug_str.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        if !self.error_chain.is_empty() {
            debug_str.push_str("Error Chain:\n");
            for (i, error) in self.error_chain.iter().enumerate() {
                debug_str.push_str(&format!("  {}: {}\n", i + 1, error));
            }
        }

        debug_str
    }
}

/// Enhanced ProcessError with additional context
#[derive(Debug)]
pub struct ProcessErrorDetails {
    /// The original error message
    pub message: String,
    /// The command that was executed
    pub command: String,
    /// The arguments passed to the command
    pub args: Vec<String>,
    /// The exit code if available
    pub exit_code: Option<i32>,
    /// Standard error output
    pub stderr: String,
    /// Standard output (first 500 chars for debugging)
    pub stdout_preview: String,
    /// Working directory when command was executed
    pub working_dir: Option<String>,
    /// Environment variables that might be relevant
    pub relevant_env: Vec<(String, String)>,
    /// System information for debugging
    pub system_info: SystemInfo,
    /// Claude CLI version if available
    pub claude_version: Option<String>,
    /// Timestamp when the error occurred
    pub timestamp: std::time::SystemTime,
    /// Network connectivity status
    pub network_status: Option<String>,
}

/// System information for debugging process errors
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Operating system name and version
    pub os: String,
    /// System architecture
    pub arch: String,
    /// Available memory in MB
    pub available_memory_mb: Option<u64>,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Current system load
    pub system_load: Option<(f64, f64, f64)>,
}

impl SystemInfo {
    /// Collect current system information
    pub fn current() -> Self {
        Self {
            os: Self::get_os_info(),
            arch: std::env::consts::ARCH.to_string(),
            available_memory_mb: Self::get_available_memory(),
            cpu_cores: num_cpus::get(),
            system_load: Self::get_system_load(),
        }
    }

    #[cfg(target_os = "linux")]
    fn get_available_memory() -> Option<u64> {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("MemAvailable:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|kb| kb.parse::<u64>().ok())
                    .map(|kb| kb / 1024) // Convert KB to MB
            })
    }

    #[cfg(not(target_os = "linux"))]
    fn get_available_memory() -> Option<u64> {
        // Platform-specific memory detection could be added here
        None
    }

    #[cfg(unix)]
    fn get_system_load() -> Option<(f64, f64, f64)> {
        let mut loadavg = [0.0; 3];
        unsafe {
            if libc::getloadavg(loadavg.as_mut_ptr(), 3) != -1 {
                Some((loadavg[0], loadavg[1], loadavg[2]))
            } else {
                None
            }
        }
    }

    #[cfg(not(unix))]
    fn get_system_load() -> Option<(f64, f64, f64)> {
        None
    }

    #[cfg(target_os = "macos")]
    fn get_macos_version() -> String {
        let Ok(output) = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
        else {
            return "macOS".to_string();
        };
        
        if !output.status.success() {
            return "macOS".to_string();
        }
        
        if let Ok(version) = String::from_utf8(output.stdout) {
            format!("macOS {}", version.trim())
        } else {
            "macOS".to_string()
        }
    }

    /// Get OS information string
    fn get_os_info() -> String {
        #[cfg(target_os = "macos")]
        {
            Self::get_macos_version()
        }
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
                for line in content.lines() {
                    if line.starts_with("PRETTY_NAME=") {
                        let name = line.strip_prefix("PRETTY_NAME=").unwrap_or("");
                        return name.trim_matches('"').to_string();
                    }
                }
            }
            "Linux".to_string()
        }
        #[cfg(target_os = "windows")]
        {
            format!("Windows {}", std::env::consts::ARCH)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
        }
    }
}

impl ProcessErrorDetails {
    /// Create enhanced ProcessError
    pub fn new(message: impl Into<String>, command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            message: message.into(),
            command: command.into(),
            args,
            exit_code: None,
            stderr: String::new(),
            stdout_preview: String::new(),
            working_dir: std::env::current_dir()
                .ok()
                .map(|p| p.to_string_lossy().to_string()),
            relevant_env: Self::collect_relevant_env(),
            system_info: SystemInfo::current(),
            claude_version: Self::get_claude_version(),
            timestamp: std::time::SystemTime::now(),
            network_status: Self::check_network_status(),
        }
    }

    /// Set exit code
    pub fn with_exit_code(mut self, code: i32) -> Self {
        self.exit_code = Some(code);
        self
    }

    /// Set stderr output
    pub fn with_stderr(mut self, stderr: impl Into<String>) -> Self {
        self.stderr = stderr.into();
        self
    }

    /// Set stdout preview
    pub fn with_stdout_preview(mut self, stdout: impl Into<String>) -> Self {
        let stdout = stdout.into();
        self.stdout_preview = if stdout.len() > 500 {
            format!("{}...[truncated]", &stdout[..500])
        } else {
            stdout
        };
        self
    }

    /// Convert to ProcessError with enhanced debugging information
    pub fn to_error(self) -> Error {
        let mut debug_sections = vec![];

        // Command execution details
        debug_sections.push(format!(
            "Command Execution:\n  Command: {} {}\n  Working Dir: {}\n  Exit Code: {:?}\n  Timestamp: {:?}",
            self.command,
            self.args.join(" "),
            self.working_dir.as_deref().unwrap_or("<unknown>"),
            self.exit_code,
            self.timestamp
        ));

        // System information
        debug_sections.push(format!(
            "System Information:\n  OS: {} ({})\n  CPU Cores: {}\n  Available Memory: {}MB\n  System Load: {:?}",
            self.system_info.os,
            self.system_info.arch,
            self.system_info.cpu_cores,
            self.system_info.available_memory_mb.map_or("Unknown".to_string(), |m| m.to_string()),
            self.system_info.system_load.map_or("Unknown".to_string(), |(l1, l5, l15)| 
                format!("{:.2}, {:.2}, {:.2}", l1, l5, l15))
        ));

        // Claude CLI version
        if let Some(version) = &self.claude_version {
            debug_sections.push(format!("Claude CLI Version: {}", version));
        } else {
            debug_sections.push("Claude CLI Version: Unable to determine".to_string());
        }

        // Network status
        if let Some(status) = &self.network_status {
            debug_sections.push(format!("Network Status: {}", status));
        }

        // Output details
        if !self.stderr.is_empty() || !self.stdout_preview.is_empty() {
            debug_sections.push(format!(
                "Process Output:\n  Stderr: {}\n  Stdout Preview: {}",
                if self.stderr.is_empty() {
                    "<empty>"
                } else {
                    &self.stderr
                },
                if self.stdout_preview.is_empty() {
                    "<empty>"
                } else {
                    &self.stdout_preview
                }
            ));
        }

        // Environment variables
        if !self.relevant_env.is_empty() {
            debug_sections.push(format!(
                "Environment Variables:\n{}",
                self.relevant_env
                    .iter()
                    .map(|(k, v)| format!("  {}={}", k, v))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }

        // Debugging suggestions based on error type
        let suggestions = self.get_debugging_suggestions();
        if !suggestions.is_empty() {
            debug_sections.push(format!("Debugging Suggestions:\n{}", suggestions));
        }

        Error::ProcessError(format!(
            "{}\n\nDebugging Information:\n{}",
            self.message,
            debug_sections.join("\n\n")
        ))
    }

    /// Get debugging suggestions based on the error context
    fn get_debugging_suggestions(&self) -> String {
        let mut suggestions = vec![];

        // Check exit codes
        match self.exit_code {
            Some(1) => suggestions.push(
                "• Exit code 1 typically indicates a general error. Check stderr for details.",
            ),
            Some(2) => suggestions
                .push("• Exit code 2 often indicates invalid command usage. Verify arguments."),
            Some(126) => suggestions
                .push("• Exit code 126: Command found but not executable. Check file permissions."),
            Some(127) => suggestions
                .push("• Exit code 127: Command not found. Verify Claude CLI is in PATH."),
            Some(130) => {
                suggestions.push("• Exit code 130: Process terminated by Ctrl+C (SIGINT).")
            }
            Some(137) => {
                suggestions.push("• Exit code 137: Process killed (SIGKILL), possibly due to OOM.")
            }
            Some(139) => suggestions
                .push("• Exit code 139: Segmentation fault. This may indicate a Claude CLI bug."),
            _ => {}
        }

        // Check for common error patterns in stderr
        if self.stderr.contains("authentication") || self.stderr.contains("unauthorized") {
            suggestions.push("• Authentication issue detected. Run 'claude auth' to authenticate.");
        }

        if self.stderr.contains("rate limit") {
            suggestions.push(
                "• Rate limit exceeded. Wait before retrying or implement exponential backoff.",
            );
        }

        if self.stderr.contains("timeout") {
            suggestions.push("• Timeout detected. Consider increasing timeout_secs in Config.");
        }

        if self.stderr.contains("network") || self.stderr.contains("connection") {
            suggestions.push(
                "• Network issue detected. Check internet connectivity and firewall settings.",
            );
        }

        // System resource checks
        if let Some(mem_mb) = self.system_info.available_memory_mb {
            if mem_mb < 100 {
                suggestions
                    .push("• Low memory detected (<100MB). This may cause process failures.");
            }
        }

        if let Some((l1, _, _)) = self.system_info.system_load {
            if l1 > self.system_info.cpu_cores as f64 * 0.8 {
                suggestions.push("• High system load detected. This may affect performance.");
            }
        }

        // Add general debugging tips
        if suggestions.is_empty() {
            suggestions.push("• Enable verbose logging with config.verbose = true");
            suggestions.push("• Check Claude CLI logs for additional details");
            suggestions.push("• Try running the command manually to reproduce the issue");
        }

        suggestions.join("\n")
    }

    /// Get a masked environment variable value
    fn get_masked_env_var(var: &str) -> Option<(String, String)> {
        let value = std::env::var(var).ok()?;
        
        let masked_value = Self::mask_sensitive_value(var, &value);
        Some((var.to_string(), masked_value))
    }
    
    /// Mask sensitive values in environment variables
    fn mask_sensitive_value(var: &str, value: &str) -> String {
        if var.contains("KEY") || var.contains("TOKEN") || var.contains("PASSWORD") {
            if value.len() > 8 {
                format!("{}***", &value[..4])
            } else {
                "***".to_string()
            }
        } else if var == "PATH" && value.len() > 200 {
            // Truncate very long PATH variables
            format!("{}...[truncated]", &value[..200])
        } else {
            value.to_string()
        }
    }

    /// Collect environment variables that might be relevant for debugging
    fn collect_relevant_env() -> Vec<(String, String)> {
        let relevant_vars = [
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "TERM",
            "CLAUDE_API_KEY",
            "CLAUDE_CONFIG",
            "ANTHROPIC_API_KEY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "RUST_LOG",
            "RUST_BACKTRACE",
        ];

        relevant_vars
            .iter()
            .filter_map(|var| Self::get_masked_env_var(var))
            .collect()
    }

    /// Get Claude CLI version by running claude --version
    fn get_claude_version() -> Option<String> {
        std::process::Command::new("claude")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }

    /// Check basic network connectivity
    fn check_network_status() -> Option<String> {
        // Simple check - try to resolve a well-known domain
        match std::net::ToSocketAddrs::to_socket_addrs(&("anthropic.com", 443)) {
            Ok(_) => Some("Connected (DNS resolution successful)".to_string()),
            Err(e) => Some(format!("Network issue: {}", e)),
        }
    }
}

/// Retry configuration for recoverable operations
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays
    pub add_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            add_jitter: true,
        }
    }
}

impl RetryConfig {
    /// Calculate delay for the given attempt number (0-based)
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        let base_delay_ms = self.base_delay.as_millis() as f64;
        let delay_ms = base_delay_ms * self.backoff_multiplier.powi(attempt as i32);
        let delay_ms = delay_ms.min(self.max_delay.as_millis() as f64);

        let final_delay = if self.add_jitter {
            // Add up to 25% jitter
            let jitter = (rand::random::<f64>() - 0.5) * 0.5 * delay_ms;
            (delay_ms + jitter).max(0.0)
        } else {
            delay_ms
        };

        Duration::from_millis(final_delay as u64)
    }
}

/// Retry a potentially failing operation with exponential backoff
pub async fn retry_with_backoff<T, F, Fut>(
    mut operation: F,
    config: RetryConfig,
    operation_name: &str,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..config.max_attempts {
        match operation().await {
            Ok(result) => {
                if attempt > 0 {
                    debug!(
                        "Operation '{}' succeeded on attempt {} after {} previous failures",
                        operation_name,
                        attempt + 1,
                        attempt
                    );
                }
                return Ok(result);
            }
            Err(error) => {
                if error.is_recoverable() {
                    last_error = Some(error.clone());

                    if attempt < config.max_attempts - 1 {
                        let delay = config.delay_for_attempt(attempt);
                        warn!(
                            "Operation '{}' failed on attempt {}, retrying in {:?}: {}",
                            operation_name,
                            attempt + 1,
                            delay,
                            error
                        );
                        sleep(delay).await;
                    } else {
                        error!(
                            "Operation '{}' failed after {} attempts: {}",
                            operation_name, config.max_attempts, error
                        );
                    }
                } else {
                    // Non-recoverable error, don't retry
                    error!(
                        "Operation '{}' failed with non-recoverable error: {}",
                        operation_name, error
                    );
                    return Err(error);
                }
            }
        }
    }

    // All retries exhausted
    Err(last_error.unwrap_or(Error::ProcessError(format!(
        "Operation '{}' failed after {} attempts",
        operation_name, config.max_attempts
    ))))
}

/// Error recovery strategies
pub struct ErrorRecovery;

impl ErrorRecovery {
    /// Attempt to recover from a BinaryNotFound error
    pub async fn recover_binary_not_found() -> Result<()> {
        debug!("Attempting to recover from BinaryNotFound error");

        // Check if Claude CLI is in common installation locations
        let common_paths = [
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            "~/.local/bin/claude",
            "./claude",
        ];

        for path in &common_paths {
            if std::path::Path::new(path).exists() {
                warn!("Found Claude CLI at {} but it's not in PATH", path);
                return Err(Error::ProcessError(format!(
                    "Claude CLI found at {} but not in PATH. Please add it to your PATH environment variable",
                    path
                )));
            }
        }

        // Suggest installation steps
        let install_suggestion = r#"
Claude CLI not found. Please install it using one of these methods:

1. Using npm (recommended):
   npm install -g @anthropic-ai/claude-cli

2. Using pip:
   pip install claude-cli

3. Download from GitHub releases:
   https://github.com/anthropics/claude-cli/releases

After installation, ensure the binary is in your PATH."#;

        Err(Error::ProcessError(install_suggestion.to_string()))
    }

    /// Attempt to recover from authentication errors
    pub async fn recover_not_authenticated() -> Result<()> {
        debug!("Attempting to recover from NotAuthenticated error");

        // Check for common authentication issues
        if std::env::var("ANTHROPIC_API_KEY").is_err() && std::env::var("CLAUDE_API_KEY").is_err() {
            let _auth_help = r#"
Authentication required. Please set up authentication using one of these methods:

1. Run the auth command:
   claude auth

2. Set environment variable:
   export ANTHROPIC_API_KEY=your_api_key_here

3. Create a config file:
   echo "api_key=your_api_key_here" > ~/.claude/config

For more information, visit: https://docs.anthropic.com/claude/docs/cli"#;

            return Err(Error::NotAuthenticated);
        }

        // If API key is set but still failing, might be invalid
        Err(Error::ProcessError(
            "API key found but authentication failed. Please check if your API key is valid and has not expired.".to_string()
        ))
    }

    /// Attempt to recover from timeout errors
    pub async fn recover_timeout(timeout_secs: u64) -> Result<()> {
        debug!(
            "Attempting to recover from timeout error after {}s",
            timeout_secs
        );

        let recovery_suggestion = format!(
            r#"Operation timed out after {}s. Consider:

1. Increasing the timeout (current: {}s):
   config.timeout_secs = Some({}); // Increase timeout

2. Simplifying your query if it's complex

3. Checking your network connection

4. Verifying Claude API service status"#,
            timeout_secs,
            timeout_secs,
            timeout_secs * 2
        );

        Err(Error::ProcessError(recovery_suggestion))
    }

    /// Attempt to recover from rate limit errors
    pub async fn recover_rate_limit_exceeded() -> Result<()> {
        debug!("Attempting to recover from rate limit error");

        // Check if we have rate limit information from headers
        let retry_after = Self::get_retry_after_duration();

        let recovery_suggestion = if let Some(duration) = retry_after {
            format!(
                r#"Rate limit exceeded. The API suggests retrying after {} seconds.

Recovery options:

1. Wait and retry:
   tokio::time::sleep(Duration::from_secs({})).await;

2. Implement exponential backoff:
   Use retry_with_backoff() with appropriate configuration

3. Consider using a rate limiter:
   - Token bucket algorithm
   - Sliding window rate limiting

4. Batch requests to reduce API calls"#,
                duration.as_secs(),
                duration.as_secs()
            )
        } else {
            r#"Rate limit exceeded. 

Recovery options:

1. Wait before retrying (suggested: 60 seconds)
2. Implement request queuing with rate limiting
3. Use exponential backoff for retries
4. Consider upgrading your API plan for higher limits

Best practice: Track your API usage to avoid hitting limits"#
                .to_string()
        };

        Err(Error::ProcessError(recovery_suggestion))
    }

    /// Attempt to recover from MCP server errors
    pub async fn recover_mcp_error(error_message: &str) -> Result<()> {
        debug!("Attempting to recover from MCP error: {}", error_message);

        // Analyze error message for specific issues
        let recovery_suggestion = if error_message.contains("connection refused") {
            r#"MCP server connection refused. 

Recovery steps:

1. Verify the MCP server is running:
   Check server process and logs

2. Confirm server address and port:
   Review your MCP configuration

3. Check firewall settings:
   Ensure the port is accessible

4. Try alternative connection:
   Use a different transport (HTTP/WebSocket)"#
                .to_string()
        } else if error_message.contains("timeout") {
            r#"MCP server timeout. 

Recovery steps:

1. Increase MCP timeout in configuration
2. Check server health and responsiveness
3. Verify network connectivity to server
4. Consider implementing connection pooling"#
                .to_string()
        } else if error_message.contains("protocol") {
            r#"MCP protocol error. 

Recovery steps:

1. Verify MCP protocol version compatibility
2. Check message format and encoding
3. Review server implementation
4. Enable debug logging for protocol details"#
                .to_string()
        } else {
            format!(
                r#"MCP server error: {}

General recovery steps:

1. Check MCP server logs for details
2. Verify server configuration
3. Test with a simple MCP client
4. Consider using fallback behavior

For persistent issues, consult MCP documentation."#,
                error_message
            )
        };

        Err(Error::McpError(recovery_suggestion))
    }

    /// Attempt to recover from stream closed errors
    pub async fn recover_stream_closed() -> Result<()> {
        debug!("Attempting to recover from stream closed error");

        let recovery_suggestion = r#"Stream closed unexpectedly.

Recovery options:

1. Automatic reconnection:
   - Implement reconnection with exponential backoff
   - Track reconnection attempts

2. Partial result recovery:
   - Save received data before closure
   - Resume from last known position

3. Stream health monitoring:
   - Implement heartbeat/keepalive
   - Detect early disconnection signs

4. Alternative streaming approach:
   - Use polling instead of streaming
   - Implement chunked requests

Example reconnection logic:
```rust
let mut attempts = 0;
loop {
    match create_stream().await {
        Ok(stream) => break,
        Err(_) if attempts < 3 => {
            attempts += 1;
            sleep(Duration::from_secs(attempts * 2)).await;
        }
        Err(e) => return Err(e),
    }
}
```"#;

        Err(Error::ProcessError(recovery_suggestion.to_string()))
    }

    /// Get retry-after duration from rate limit response
    fn get_retry_after_duration() -> Option<Duration> {
        // In a real implementation, this would parse headers from the response
        // For now, return a reasonable default
        Some(Duration::from_secs(60))
    }
}

/// Enhanced logging for errors with structured information
pub fn log_error_with_context(error: &Error, context: &ErrorContext) {
    match error {
        Error::BinaryNotFound => {
            error!(
                operation = %context.operation,
                error_code = "C001",
                "Claude CLI binary not found. {}",
                context.to_debug_string()
            );
        }
        Error::NotAuthenticated => {
            error!(
                operation = %context.operation,
                error_code = "C012",
                "Claude CLI authentication required. {}",
                context.to_debug_string()
            );
        }
        Error::Timeout(seconds) => {
            error!(
                operation = %context.operation,
                error_code = "C007",
                timeout_seconds = seconds,
                "Operation timed out. {}",
                context.to_debug_string()
            );
        }
        Error::ProcessError(msg) => {
            error!(
                operation = %context.operation,
                error_code = "C010",
                process_error = %msg,
                "Process execution failed. {}",
                context.to_debug_string()
            );
        }
        _ => {
            error!(
                operation = %context.operation,
                error_code = %error.code(),
                "Operation failed: {}. {}",
                error,
                context.to_debug_string()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("test_operation")
            .with_debug_info("key1", "value1")
            .with_debug_info("key2", "value2")
            .with_error_chain("first error")
            .with_error_chain("second error");

        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.debug_info.len(), 2);
        assert_eq!(context.error_chain.len(), 2);

        let debug_string = context.to_debug_string();
        assert!(debug_string.contains("test_operation"));
        assert!(debug_string.contains("key1: value1"));
        assert!(debug_string.contains("first error"));
    }

    #[test]
    fn test_process_error_details() {
        let details =
            ProcessErrorDetails::new("Test error", "claude", vec!["--version".to_string()])
                .with_exit_code(1)
                .with_stderr("Error message")
                .with_stdout_preview("Output preview");

        let error = details.to_error();
        let error_string = error.to_string();

        // Check basic content
        assert!(error_string.contains("Test error"));
        assert!(error_string.contains("claude --version"));
        assert!(error_string.contains("Exit Code: Some(1)"));

        // Check new enhanced content
        assert!(error_string.contains("System Information:"));
        assert!(error_string.contains("CPU Cores:"));
        assert!(error_string.contains("Debugging Suggestions:"));
        assert!(error_string.contains("Exit code 1 typically indicates"));
    }

    #[test]
    fn test_process_error_debugging_suggestions() {
        // Test authentication error
        let auth_error = ProcessErrorDetails::new("Auth failed", "claude", vec![])
            .with_stderr("authentication failed");

        let error_string = auth_error.to_error().to_string();
        assert!(error_string.contains("Run 'claude auth' to authenticate"));

        // Test rate limit error
        let rate_error = ProcessErrorDetails::new("Rate limited", "claude", vec![])
            .with_stderr("rate limit exceeded");

        let error_string = rate_error.to_error().to_string();
        assert!(error_string.contains("implement exponential backoff"));

        // Test exit code 127
        let not_found_error =
            ProcessErrorDetails::new("Command failed", "claude", vec![]).with_exit_code(127);

        let error_string = not_found_error.to_error().to_string();
        assert!(error_string.contains("Command not found"));
    }

    #[test]
    fn test_retry_config_delay_calculation() {
        let config = RetryConfig {
            base_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(5),
            add_jitter: false,
            ..Default::default()
        };

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));

        // Should cap at max_delay
        let large_attempt_delay = config.delay_for_attempt(10);
        assert!(large_attempt_delay <= config.max_delay);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_success() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        let operation = move || {
            let count = Arc::clone(&count_clone);
            async move {
                let current_count = count.fetch_add(1, Ordering::SeqCst) + 1;
                if current_count < 3 {
                    Err(Error::Timeout(30)) // Recoverable error
                } else {
                    Ok("success".to_string())
                }
            }
        };

        let config = RetryConfig {
            max_attempts: 5,
            base_delay: Duration::from_millis(1), // Fast for testing
            add_jitter: false,
            ..Default::default()
        };

        let result = retry_with_backoff(operation, config, "test").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_non_recoverable() {
        let operation = || async { Err(Error::BinaryNotFound) }; // Non-recoverable

        let config = RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            add_jitter: false,
            ..Default::default()
        };

        let result: Result<String> = retry_with_backoff(operation, config, "test").await;
        assert!(result.is_err());
        // Should fail immediately without retries for non-recoverable errors
        assert!(matches!(result.unwrap_err(), Error::BinaryNotFound));
    }
}
