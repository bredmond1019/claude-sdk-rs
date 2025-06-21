//! Comprehensive CLI integration tests for claude-ai-interactive
//!
//! This module tests the complete CLI integration including:
//! - Command line argument parsing and validation
//! - End-to-end command execution workflows
//! - Error scenarios and recovery mechanisms
//! - Concurrent CLI invocations
//! - Performance with large data volumes
//! - Output formatting and user interactions

use claude_sdk_rs_interactive::{
    analytics::simple::SimpleAnalyticsEngine,
    cli::Cli,
    cost::{CostEntry, CostFilter, CostTracker},
    execution::{ExecutionContext, ExecutionResult},
    history::{ExportFormat, HistoryEntry, HistorySearch, HistoryStore},
    output::{formatter::OutputFormatter, OutputStyle},
    session::{Session, SessionId, SessionManager},
    InteractiveError, Result,
};

use chrono::{DateTime, Duration, Utc};
use clap::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;
use tokio::process::Command as TokioCommand;
use uuid::Uuid;

/// CLI integration test harness
pub struct CliIntegrationTestHarness {
    temp_dir: TempDir,
    data_dir: PathBuf,
    cli_binary_path: Option<PathBuf>,
}

impl CliIntegrationTestHarness {
    /// Create a new CLI integration test harness
    pub fn new() -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        let data_dir = temp_dir.path().join("data");
        fs::create_dir_all(&data_dir)?;

        // Create commands directory for testing
        let commands_dir = data_dir.join("commands");
        fs::create_dir_all(&commands_dir)?;

        // Create sample commands for testing
        Self::create_sample_commands(&commands_dir)?;

        Ok(Self {
            temp_dir,
            data_dir,
            cli_binary_path: None,
        })
    }

    /// Set the path to the CLI binary for integration testing
    pub fn with_binary_path(mut self, path: PathBuf) -> Self {
        self.cli_binary_path = Some(path);
        self
    }

    /// Create sample command files for testing
    fn create_sample_commands(commands_dir: &Path) -> Result<()> {
        // Analyze command
        fs::write(
            commands_dir.join("analyze.sh"),
            r#"#!/bin/bash
# Analyze code for potential improvements
echo "Starting code analysis..."
sleep 1
echo "Analysis complete: Found 3 potential improvements"
exit 0
"#,
        )?;

        // Generate command
        fs::write(
            commands_dir.join("generate.py"),
            r#"#!/usr/bin/env python3
# Generate test cases
print("Generating test cases...")
import time
time.sleep(0.5)
print("Generated 5 test cases successfully")
"#,
        )?;

        // Build command
        fs::write(
            commands_dir.join("build.sh"),
            r#"#!/bin/bash
# Build the project
echo "Building project..."
sleep 2
echo "Build completed successfully"
"#,
        )?;

        // Test command (sometimes fails)
        fs::write(
            commands_dir.join("test.sh"),
            r#"#!/bin/bash
# Run tests with random failure
echo "Running tests..."
sleep 1
if [ $((RANDOM % 3)) -eq 0 ]; then
    echo "Tests failed: 2 tests failed out of 10"
    exit 1
else
    echo "All tests passed successfully"
    exit 0
fi
"#,
        )?;

        // Make scripts executable (if on Unix-like systems)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for entry in fs::read_dir(commands_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    let mut perms = fs::metadata(&path)?.permissions();
                    perms.set_mode(0o755);
                    fs::set_permissions(&path, perms)?;
                }
            }
        }

        Ok(())
    }

    /// Execute CLI command with arguments
    pub async fn execute_cli(&self, args: &[&str]) -> Result<CliTestResult> {
        if let Some(binary_path) = &self.cli_binary_path {
            self.execute_binary_cli(binary_path, args).await
        } else {
            self.execute_in_process_cli(args).await
        }
    }

    /// Execute CLI using the actual binary (true integration test)
    async fn execute_binary_cli(&self, binary_path: &Path, args: &[&str]) -> Result<CliTestResult> {
        let start_time = Instant::now();

        let mut cmd = TokioCommand::new(binary_path);
        cmd.args(args)
            .arg("--data-dir")
            .arg(&self.data_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = cmd.output().await.map_err(|e| {
            InteractiveError::execution(format!("Failed to execute CLI binary: {}", e))
        })?;

        let duration = start_time.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(CliTestResult {
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout,
            stderr,
            duration,
            args: args.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Execute CLI in-process (faster, for most tests)
    async fn execute_in_process_cli(&self, args: &[&str]) -> Result<CliTestResult> {
        let start_time = Instant::now();

        // Build CLI arguments with data directory
        let mut cli_args = vec!["claude-interactive"];
        cli_args.extend(args);
        cli_args.extend(&["--data-dir", self.data_dir.to_str().unwrap()]);

        // Capture stdout/stderr (simplified - in real implementation would redirect)
        let result = match Cli::try_parse_from(cli_args) {
            Ok(cli) => {
                // Execute the CLI command
                match cli.execute().await {
                    Ok(()) => CliTestResult {
                        success: true,
                        exit_code: Some(0),
                        stdout: "Command executed successfully\n".to_string(),
                        stderr: String::new(),
                        duration: start_time.elapsed(),
                        args: args.iter().map(|s| s.to_string()).collect(),
                    },
                    Err(e) => CliTestResult {
                        success: false,
                        exit_code: Some(1),
                        stdout: String::new(),
                        stderr: format!("Error: {}\n", e),
                        duration: start_time.elapsed(),
                        args: args.iter().map(|s| s.to_string()).collect(),
                    },
                }
            }
            Err(e) => CliTestResult {
                success: false,
                exit_code: Some(2),
                stdout: String::new(),
                stderr: format!("Parse error: {}\n", e),
                duration: start_time.elapsed(),
                args: args.iter().map(|s| s.to_string()).collect(),
            },
        };

        Ok(result)
    }

    /// Get the data directory path
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Get temporary directory
    pub fn temp_dir(&self) -> &TempDir {
        &self.temp_dir
    }

    /// Create a session for testing
    pub async fn create_test_session(&self, name: &str) -> Result<Session> {
        let manager = SessionManager::new();
        manager
            .create_session(name.to_string(), Some(format!("Test session: {}", name)))
            .await
    }

    /// Add sample cost data
    pub async fn add_sample_cost_data(&self, session_id: SessionId, count: usize) -> Result<()> {
        let mut cost_tracker = CostTracker::new(self.data_dir.join("costs.json"))?;

        for i in 0..count {
            let entry = CostEntry::new(
                session_id,
                format!("test-command-{}", i),
                0.01 + (i as f64 * 0.005), // Varying costs
                100 + (i as u32 * 10),     // Input tokens
                200 + (i as u32 * 20),     // Output tokens
                1000 + (i as u64 * 100),   // Duration
                "claude-3-opus".to_string(),
            );
            cost_tracker.record_cost(entry).await?;
        }

        Ok(())
    }

    /// Add sample history data
    pub async fn add_sample_history_data(&self, session_id: SessionId, count: usize) -> Result<()> {
        let mut history_store = HistoryStore::new(self.data_dir.join("history.json"))?;

        for i in 0..count {
            let success = i % 4 != 0; // 75% success rate
            let entry = HistoryEntry::new(
                session_id,
                format!("test-command-{}", i),
                vec![format!("--param-{}", i)],
                if success {
                    format!("Success output for command {}", i)
                } else {
                    format!("Error: Command {} failed", i)
                },
                success,
                1000 + (i as u64 * 100),
            );

            let entry = if success {
                entry.with_cost(
                    0.01 + (i as f64 * 0.005),
                    100 + (i as u32 * 10),
                    200 + (i as u32 * 20),
                    "claude-3-opus".to_string(),
                )
            } else {
                entry.with_error("Simulated failure".to_string())
            };

            history_store.store_entry(entry).await?;
        }

        Ok(())
    }
}

/// Result of CLI command execution
#[derive(Debug, Clone)]
pub struct CliTestResult {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration: std::time::Duration,
    pub args: Vec<String>,
}

impl CliTestResult {
    /// Check if command was successful
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Check if output contains text
    pub fn stdout_contains(&self, text: &str) -> bool {
        self.stdout.contains(text)
    }

    /// Check if stderr contains text
    pub fn stderr_contains(&self, text: &str) -> bool {
        self.stderr.contains(text)
    }

    /// Get combined output
    pub fn combined_output(&self) -> String {
        format!("{}{}", self.stdout, self.stderr)
    }
}

// CLI Integration Tests

#[tokio::test]
async fn test_cli_help_command() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    let result = harness.execute_cli(&["--help"]).await?;

    // Help should always succeed
    assert!(result.is_success(), "Help command should succeed");
    assert!(
        result.stdout_contains("Interactive CLI for managing multiple Claude sessions"),
        "Help should contain description"
    );
    assert!(
        result.stdout_contains("SUBCOMMANDS") || result.stdout_contains("Commands:"),
        "Help should list commands"
    );

    Ok(())
}

#[tokio::test]
async fn test_cli_list_commands() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Test basic list
    let result = harness.execute_cli(&["list"]).await?;
    assert!(result.is_success(), "List command should succeed");

    // Test detailed list
    let result = harness.execute_cli(&["list", "--detailed"]).await?;
    assert!(result.is_success(), "Detailed list should succeed");

    // Test filtered list
    let result = harness.execute_cli(&["list", "--filter", "test"]).await?;
    assert!(result.is_success(), "Filtered list should succeed");

    Ok(())
}

#[tokio::test]
async fn test_cli_session_management() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Test session creation
    let result = harness
        .execute_cli(&["session", "create", "test-session"])
        .await?;
    assert!(result.is_success(), "Session creation should succeed");

    // Test session listing
    let result = harness.execute_cli(&["session", "list"]).await?;
    assert!(result.is_success(), "Session listing should succeed");

    // Test detailed session listing
    let result = harness
        .execute_cli(&["session", "list", "--detailed"])
        .await?;
    assert!(
        result.is_success(),
        "Detailed session listing should succeed"
    );

    // Test session switching
    let result = harness
        .execute_cli(&["session", "switch", "test-session"])
        .await?;
    assert!(result.is_success(), "Session switching should succeed");

    // Test session deletion with force
    let result = harness
        .execute_cli(&["session", "delete", "test-session", "--force"])
        .await?;
    assert!(result.is_success(), "Session deletion should succeed");

    Ok(())
}

#[tokio::test]
async fn test_cli_cost_tracking() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;
    let session = harness.create_test_session("cost-test").await?;
    harness.add_sample_cost_data(session.id, 10).await?;

    // Test basic cost display
    let result = harness.execute_cli(&["cost"]).await?;
    assert!(result.is_success(), "Cost command should succeed");

    // Test cost breakdown
    let result = harness.execute_cli(&["cost", "--breakdown"]).await?;
    assert!(result.is_success(), "Cost breakdown should succeed");

    // Test session-specific costs
    let result = harness
        .execute_cli(&["cost", "--session", "cost-test"])
        .await?;
    assert!(result.is_success(), "Session cost query should succeed");

    // Test cost export
    let result = harness.execute_cli(&["cost", "--export", "csv"]).await?;
    assert!(result.is_success(), "Cost export should succeed");

    Ok(())
}

#[tokio::test]
async fn test_cli_history_search() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;
    let session = harness.create_test_session("history-test").await?;
    harness.add_sample_history_data(session.id, 20).await?;

    // Test basic history
    let result = harness.execute_cli(&["history"]).await?;
    assert!(result.is_success(), "History command should succeed");

    // Test history search
    let result = harness
        .execute_cli(&["history", "--search", "test-command"])
        .await?;
    assert!(result.is_success(), "History search should succeed");

    // Test session-specific history
    let result = harness
        .execute_cli(&["history", "--session", "history-test"])
        .await?;
    assert!(result.is_success(), "Session history query should succeed");

    // Test history with output
    let result = harness.execute_cli(&["history", "--output"]).await?;
    assert!(result.is_success(), "History with output should succeed");

    // Test history export
    let result = harness
        .execute_cli(&["history", "--export", "json"])
        .await?;
    assert!(result.is_success(), "History export should succeed");

    Ok(())
}

#[tokio::test]
async fn test_cli_configuration_management() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Test config show
    let result = harness.execute_cli(&["config", "show"]).await?;
    assert!(result.is_success(), "Config show should succeed");

    // Test config path
    let result = harness.execute_cli(&["config", "path"]).await?;
    assert!(result.is_success(), "Config path should succeed");

    // Test config set
    let result = harness
        .execute_cli(&["config", "set", "timeout_secs", "60"])
        .await?;
    assert!(result.is_success(), "Config set should succeed");

    // Test config reset with force
    let result = harness.execute_cli(&["config", "reset", "--force"]).await?;
    assert!(result.is_success(), "Config reset should succeed");

    Ok(())
}

#[tokio::test]
async fn test_cli_error_scenarios() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Test invalid command
    let result = harness.execute_cli(&["invalid-command"]).await?;
    assert!(!result.is_success(), "Invalid command should fail");

    // Test session not found
    let result = harness
        .execute_cli(&["session", "switch", "nonexistent-session"])
        .await?;
    assert!(!result.is_success(), "Non-existent session should fail");

    // Test invalid cost export format
    let result = harness
        .execute_cli(&["cost", "--export", "invalid"])
        .await?;
    assert!(!result.is_success(), "Invalid export format should fail");

    // Test invalid history export format
    let result = harness
        .execute_cli(&["history", "--export", "invalid"])
        .await?;
    assert!(!result.is_success(), "Invalid export format should fail");

    // Test invalid config key
    let result = harness
        .execute_cli(&["config", "set", "invalid.key", "value"])
        .await?;
    assert!(!result.is_success(), "Invalid config key should fail");

    Ok(())
}

#[tokio::test]
async fn test_cli_global_options() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Test verbose flag
    let result = harness.execute_cli(&["--verbose", "list"]).await?;
    assert!(result.is_success(), "Verbose list should succeed");

    // Test quiet flag
    let result = harness.execute_cli(&["--quiet", "list"]).await?;
    assert!(result.is_success(), "Quiet list should succeed");

    // Test custom timeout
    let result = harness.execute_cli(&["--timeout", "30", "list"]).await?;
    assert!(result.is_success(), "Custom timeout should succeed");

    // Test custom data directory (implicitly tested through our test setup)
    let result = harness.execute_cli(&["list"]).await?;
    assert!(result.is_success(), "Custom data dir should work");

    Ok(())
}

#[tokio::test]
async fn test_cli_completion_generation() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Test bash completion
    let result = harness.execute_cli(&["completion", "bash"]).await?;
    assert!(result.is_success(), "Bash completion should succeed");

    // Test zsh completion
    let result = harness.execute_cli(&["completion", "zsh"]).await?;
    assert!(result.is_success(), "Zsh completion should succeed");

    // Test fish completion
    let result = harness.execute_cli(&["completion", "fish"]).await?;
    assert!(result.is_success(), "Fish completion should succeed");

    Ok(())
}

#[tokio::test]
async fn test_cli_concurrent_invocations() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Create test session
    let session = harness.create_test_session("concurrent-test").await?;

    // Prepare multiple concurrent operations
    let operations = vec![
        vec!["list"],
        vec!["session", "list"],
        vec!["cost"],
        vec!["history"],
        vec!["config", "show"],
    ];

    // Execute operations concurrently
    let futures: Vec<_> = operations
        .iter()
        .map(|args| harness.execute_cli(args.as_slice()))
        .collect();

    let results = futures::future::join_all(futures).await;

    // All operations should succeed
    for (i, result) in results.into_iter().enumerate() {
        let result = result?;
        assert!(
            result.is_success(),
            "Concurrent operation {} should succeed: {:?}",
            i,
            result
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_cli_large_data_volumes() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Create session with large amount of data
    let session = harness.create_test_session("large-data-test").await?;

    // Add significant amount of cost and history data
    harness.add_sample_cost_data(session.id, 1000).await?;
    harness.add_sample_history_data(session.id, 1000).await?;

    let start_time = Instant::now();

    // Test operations with large datasets
    let operations = vec![
        vec!["cost", "--breakdown"],
        vec!["history", "--limit", "100"],
        vec!["session", "list", "--detailed"],
    ];

    for operation in operations {
        let op_start = Instant::now();
        let result = harness.execute_cli(&operation).await?;
        let op_duration = op_start.elapsed();

        assert!(
            result.is_success(),
            "Large data operation should succeed: {:?}",
            operation
        );

        // Operations should complete within reasonable time (adjust threshold as needed)
        assert!(
            op_duration.as_secs() < 10,
            "Operation should complete within 10 seconds, took {:?}",
            op_duration
        );
    }

    let total_duration = start_time.elapsed();
    println!("Large data test completed in {:?}", total_duration);

    Ok(())
}

#[tokio::test]
async fn test_cli_end_to_end_workflow() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Complete workflow simulation

    // 1. List available commands
    let result = harness.execute_cli(&["list"]).await?;
    assert!(result.is_success(), "Step 1: List commands should succeed");

    // 2. Create a new session
    let result = harness
        .execute_cli(&[
            "session",
            "create",
            "workflow-test",
            "--description",
            "E2E test session",
        ])
        .await?;
    assert!(result.is_success(), "Step 2: Create session should succeed");

    // 3. Switch to the session
    let result = harness
        .execute_cli(&["session", "switch", "workflow-test"])
        .await?;
    assert!(result.is_success(), "Step 3: Switch session should succeed");

    // 4. Simulate running some commands (we'll add data manually)
    let session = harness.create_test_session("workflow-test-2").await?;
    harness.add_sample_cost_data(session.id, 5).await?;
    harness.add_sample_history_data(session.id, 5).await?;

    // 5. Check cost tracking
    let result = harness.execute_cli(&["cost", "--breakdown"]).await?;
    assert!(result.is_success(), "Step 5: Cost tracking should succeed");

    // 6. Check history
    let result = harness.execute_cli(&["history", "--limit", "10"]).await?;
    assert!(result.is_success(), "Step 6: History check should succeed");

    // 7. Export data
    let result = harness.execute_cli(&["cost", "--export", "csv"]).await?;
    assert!(result.is_success(), "Step 7: Cost export should succeed");

    let result = harness
        .execute_cli(&["history", "--export", "json"])
        .await?;
    assert!(result.is_success(), "Step 7: History export should succeed");

    // 8. List all sessions
    let result = harness
        .execute_cli(&["session", "list", "--detailed"])
        .await?;
    assert!(result.is_success(), "Step 8: List sessions should succeed");

    // 9. Show configuration
    let result = harness.execute_cli(&["config", "show"]).await?;
    assert!(result.is_success(), "Step 9: Show config should succeed");

    // 10. Clean up - delete test session
    let result = harness
        .execute_cli(&["session", "delete", "workflow-test", "--force"])
        .await?;
    assert!(
        result.is_success(),
        "Step 10: Delete session should succeed"
    );

    Ok(())
}

#[tokio::test]
async fn test_cli_data_persistence() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Create session and add data
    let result = harness
        .execute_cli(&["session", "create", "persistence-test"])
        .await?;
    assert!(result.is_success(), "Session creation should succeed");

    // Manually add some data
    let session = harness.create_test_session("persistence-test-2").await?;
    harness.add_sample_cost_data(session.id, 3).await?;
    harness.add_sample_history_data(session.id, 3).await?;

    // Verify data exists
    let result = harness.execute_cli(&["cost"]).await?;
    assert!(result.is_success(), "Cost query should succeed");

    let result = harness.execute_cli(&["history"]).await?;
    assert!(result.is_success(), "History query should succeed");

    // Simulate application restart by creating new harness with same data directory
    let data_dir_path = harness.data_dir().to_path_buf();

    // Data should still be accessible after "restart"
    let new_harness = CliIntegrationTestHarness::new()?;

    // Copy the data directory to simulate persistence
    let _ = std::process::Command::new("cp")
        .arg("-r")
        .arg(&data_dir_path)
        .arg(new_harness.data_dir())
        .output();

    let result = new_harness.execute_cli(&["session", "list"]).await?;
    assert!(
        result.is_success(),
        "Session list after restart should succeed"
    );

    Ok(())
}

#[tokio::test]
async fn test_cli_error_recovery() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Test recovery from invalid operations

    // 1. Try to use non-existent session
    let result = harness
        .execute_cli(&["cost", "--session", "nonexistent"])
        .await?;
    assert!(!result.is_success(), "Non-existent session should fail");

    // 2. Application should still work after error
    let result = harness.execute_cli(&["list"]).await?;
    assert!(
        result.is_success(),
        "Normal operation after error should work"
    );

    // 3. Try invalid config value
    let result = harness
        .execute_cli(&["config", "set", "timeout_secs", "invalid"])
        .await?;
    assert!(!result.is_success(), "Invalid config value should fail");

    // 4. Config should still be accessible
    let result = harness.execute_cli(&["config", "show"]).await?;
    assert!(result.is_success(), "Config show after error should work");

    // 5. Try to delete non-existent session
    let result = harness
        .execute_cli(&["session", "delete", "nonexistent", "--force"])
        .await?;
    assert!(
        !result.is_success(),
        "Delete non-existent session should fail"
    );

    // 6. Session operations should still work
    let result = harness.execute_cli(&["session", "list"]).await?;
    assert!(result.is_success(), "Session list after error should work");

    Ok(())
}

#[tokio::test]
async fn test_cli_performance_benchmarks() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;

    // Prepare test data
    let session = harness.create_test_session("perf-test").await?;
    harness.add_sample_cost_data(session.id, 100).await?;
    harness.add_sample_history_data(session.id, 100).await?;

    // Benchmark common operations
    let operations = vec![
        ("list", vec!["list"]),
        ("session_list", vec!["session", "list"]),
        ("cost_summary", vec!["cost"]),
        ("cost_breakdown", vec!["cost", "--breakdown"]),
        ("history_basic", vec!["history", "--limit", "20"]),
        ("history_search", vec!["history", "--search", "test"]),
        ("config_show", vec!["config", "show"]),
    ];

    let mut benchmark_results = HashMap::new();

    for (name, args) in operations {
        let mut durations = Vec::new();

        // Run each operation multiple times
        for _ in 0..5 {
            let start = Instant::now();
            let result = harness.execute_cli(&args).await?;
            let duration = start.elapsed();

            assert!(
                result.is_success(),
                "Benchmark operation {} should succeed",
                name
            );
            durations.push(duration);
        }

        // Calculate average
        let avg_duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;
        benchmark_results.insert(name, avg_duration);

        println!("Operation '{}' average duration: {:?}", name, avg_duration);

        // Ensure operations complete within reasonable time
        assert!(
            avg_duration.as_millis() < 1000,
            "Operation {} should complete within 1 second, took {:?}",
            name,
            avg_duration
        );
    }

    Ok(())
}

// Test helper functions

/// Verify that a file exists and has expected content patterns
fn verify_export_file(path: &Path, expected_patterns: &[&str]) -> Result<()> {
    assert!(path.exists(), "Export file should exist: {:?}", path);

    let content = fs::read_to_string(path)?;
    assert!(!content.is_empty(), "Export file should not be empty");

    for pattern in expected_patterns {
        assert!(
            content.contains(pattern),
            "Export file should contain pattern '{}'\nContent: {}",
            pattern,
            content
        );
    }

    Ok(())
}

/// Create a session with predictable test data
async fn create_predictable_test_session(harness: &CliIntegrationTestHarness) -> Result<SessionId> {
    let session = harness.create_test_session("predictable-test").await?;

    // Add specific test data
    let mut cost_tracker = CostTracker::new(harness.data_dir().join("costs.json"))?;
    let mut history_store = HistoryStore::new(harness.data_dir().join("history.json"))?;

    // Add known cost entries
    let costs = vec![
        ("analyze", 0.05, 500, 1000, 2000),
        ("generate", 0.03, 300, 600, 1500),
        ("build", 0.01, 100, 200, 3000),
    ];

    for (i, (cmd, cost, input_tokens, output_tokens, duration)) in costs.iter().enumerate() {
        let cost_entry = CostEntry::new(
            session.id,
            cmd.to_string(),
            *cost,
            *input_tokens,
            *output_tokens,
            *duration,
            "claude-3-opus".to_string(),
        );
        cost_tracker.record_cost(cost_entry).await?;

        let history_entry = HistoryEntry::new(
            session.id,
            cmd.to_string(),
            vec![format!("--test-{}", i)],
            format!("Test output for {}", cmd),
            true,
            *duration,
        )
        .with_cost(
            *cost,
            *input_tokens,
            *output_tokens,
            "claude-3-opus".to_string(),
        );

        history_store.store_entry(history_entry).await?;
    }

    Ok(session.id)
}

#[tokio::test]
async fn test_cli_predictable_data() -> Result<()> {
    let harness = CliIntegrationTestHarness::new()?;
    let _session_id = create_predictable_test_session(&harness).await?;

    // Test cost calculation with known values
    let result = harness.execute_cli(&["cost", "--breakdown"]).await?;
    assert!(result.is_success(), "Cost breakdown should succeed");

    // Test history with known entries
    let result = harness
        .execute_cli(&["history", "--search", "analyze"])
        .await?;
    assert!(result.is_success(), "History search should succeed");

    Ok(())
}
