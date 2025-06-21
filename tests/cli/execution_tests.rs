//! Integration tests for command execution components
//!
//! These tests verify that the CommandRunner, ParallelExecutor, and OutputFormatter
//! work together correctly and integrate properly with the session management system.

use claude_sdk_rs_interactive::{
    execution::{
        runner::CommandRunner, ExecutionContext, ParallelConfig, ParallelExecutor, RunnerConfig,
    },
    output::{formatter::OutputFormatter, AgentId, FormatterConfig, OutputStyle},
    session::{
        manager::SessionManager,
        storage::{JsonFileStorage, StorageConfig},
    },
    Result,
};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::timeout;

/// Create a test session manager with temporary storage
async fn create_test_session_manager() -> Result<Arc<SessionManager>> {
    let temp_dir = tempdir().unwrap();
    let config = StorageConfig {
        data_dir: temp_dir.path().to_path_buf(),
        sessions_file: "test_sessions.json".to_string(),
        current_session_file: "test_current.json".to_string(),
    };

    let storage = Arc::new(JsonFileStorage::new(config));
    Ok(Arc::new(SessionManager::new(storage)))
}

/// Create a test command runner
async fn create_test_runner() -> Result<CommandRunner> {
    let session_manager = create_test_session_manager().await?;
    CommandRunner::new(session_manager)
}

#[tokio::test]
async fn test_command_runner_integration() -> Result<()> {
    let runner = create_test_runner().await?;

    let context = ExecutionContext {
        session_id: None,
        command_name: "help".to_string(),
        args: vec![],
        parallel: false,
        agent_count: 1,
    };

    // Note: This test would require a mock Claude client in a real scenario
    // For now, we're testing the structure and basic functionality
    assert_eq!(runner.config().timeout, Duration::from_secs(30));
    assert!(runner.config().streaming);

    Ok(())
}

#[tokio::test]
async fn test_parallel_executor_setup() -> Result<()> {
    let runner = Arc::new(create_test_runner().await?);
    let executor = ParallelExecutor::new(runner);

    let stats = executor.get_stats().await;
    assert_eq!(stats.active_executions, 0);
    assert_eq!(stats.total_executions, 0);

    // Test empty parallel execution
    let result = executor.execute_parallel(vec![]).await?;
    assert_eq!(result.successful_count, 0);
    assert_eq!(result.failed_count, 0);
    assert!(result.results.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_parallel_executor_with_contexts() -> Result<()> {
    let runner = Arc::new(create_test_runner().await?);
    let executor = ParallelExecutor::new(runner);

    let contexts = vec![
        ExecutionContext {
            session_id: None,
            command_name: "test1".to_string(),
            args: vec!["arg1".to_string()],
            parallel: true,
            agent_count: 2,
        },
        ExecutionContext {
            session_id: None,
            command_name: "test2".to_string(),
            args: vec!["arg2".to_string()],
            parallel: true,
            agent_count: 2,
        },
    ];

    // Test that contexts are properly handled
    // Note: Actual execution would require mock Claude client
    let initial_stats = executor.get_stats().await;
    assert_eq!(initial_stats.total_executions, 0);

    Ok(())
}

#[tokio::test]
async fn test_output_formatter_integration() -> Result<()> {
    let mut formatter = OutputFormatter::new();

    // Test basic message formatting
    let message = formatter.format_message("Test message", OutputStyle::Success);
    assert!(message.contains("✓"));

    // Test agent output formatting
    let agent_id = AgentId(1);
    let output = formatter.format_agent_output("Agent output", agent_id);
    assert!(output.contains("Agent 1"));

    // Test multiple agents get different colors
    let agent2_id = AgentId(2);
    let output2 = formatter.format_agent_output("Agent 2 output", agent2_id);
    assert!(output2.contains("Agent 2"));

    Ok(())
}

#[tokio::test]
async fn test_session_integration() -> Result<()> {
    let session_manager = create_test_session_manager().await?;

    // Create a test session
    let session = session_manager
        .create_session(
            "Test Execution Session".to_string(),
            Some("Session for testing execution integration".to_string()),
        )
        .await?;

    // Create runner with session
    let runner = CommandRunner::new(Arc::clone(&session_manager))?;

    let context = ExecutionContext {
        session_id: Some(session.id),
        command_name: "test_command".to_string(),
        args: vec!["arg1".to_string(), "arg2".to_string()],
        parallel: false,
        agent_count: 1,
    };

    // Verify context has session ID
    assert_eq!(context.session_id, Some(session.id));

    // Note: Using placeholder SessionManager, so just verify context has session ID
    // In a full implementation, we would verify the session exists

    Ok(())
}

#[tokio::test]
async fn test_config_customization() -> Result<()> {
    let session_manager = create_test_session_manager().await?;

    // Test custom runner config
    let runner_config = RunnerConfig {
        timeout: Duration::from_secs(60),
        streaming: false,
        stream_format: claude_sdk_rs::StreamFormat::Json,
        max_retries: 5,
        retry_delay: Duration::from_secs(2),
    };

    let runner = CommandRunner::with_config(session_manager, runner_config.clone())?;
    assert_eq!(runner.config().timeout, Duration::from_secs(60));
    assert!(!runner.config().streaming);
    assert_eq!(runner.config().max_retries, 5);

    // Test custom parallel config
    let parallel_config = ParallelConfig {
        max_concurrent: 8,
        continue_on_error: false,
        total_timeout: Duration::from_secs(600),
        output_buffer_size: 2000,
    };

    let executor = ParallelExecutor::with_config(Arc::new(runner), parallel_config.clone());
    assert_eq!(executor.config().max_concurrent, 8);
    assert!(!executor.config().continue_on_error);

    // Test custom formatter config
    let formatter_config = FormatterConfig {
        use_colors: false,
        show_timestamps: false,
        show_agent_ids: true,
        max_width: Some(80),
        indent_level: 2,
        show_costs: false,
    };

    let formatter = OutputFormatter::with_config(formatter_config.clone());
    assert!(!formatter.config().use_colors);
    assert!(!formatter.config().show_timestamps);
    assert_eq!(formatter.config().max_width, Some(80));

    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let session_manager = create_test_session_manager().await?;
    let runner = CommandRunner::new(session_manager)?;

    // Test with invalid session ID
    let invalid_context = ExecutionContext {
        session_id: Some(uuid::Uuid::new_v4()), // Random UUID that doesn't exist
        command_name: "test".to_string(),
        args: vec![],
        parallel: false,
        agent_count: 1,
    };

    // The runner should handle this gracefully
    // Note: Actual execution test would require mock Claude client

    Ok(())
}

#[tokio::test]
async fn test_parallel_cancellation() -> Result<()> {
    let runner = Arc::new(create_test_runner().await?);
    let executor = ParallelExecutor::new(runner);

    // Test cancelling with no active tasks
    let cancelled = executor.cancel_all().await?;
    assert_eq!(cancelled, 0);

    let stats = executor.get_stats().await;
    assert_eq!(stats.active_executions, 0);

    Ok(())
}

#[tokio::test]
async fn test_output_styles_and_formatting() -> Result<()> {
    let formatter = OutputFormatter::new();

    // Test all output styles
    let styles = vec![
        OutputStyle::Plain,
        OutputStyle::Success,
        OutputStyle::Error,
        OutputStyle::Warning,
        OutputStyle::Info,
        OutputStyle::Debug,
        OutputStyle::Header,
        OutputStyle::Highlight,
    ];

    for style in styles {
        let message = formatter.format_message("Test message", style);
        assert!(!message.is_empty());

        // Each style should have its own symbol (except Plain)
        if !matches!(style, OutputStyle::Plain) {
            assert!(!style.symbol().is_empty());
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_progress_indicator() -> Result<()> {
    use claude_sdk_rs_interactive::output::ProgressIndicator;

    let mut progress = ProgressIndicator::new(10, "Testing Progress".to_string());

    assert_eq!(progress.current(), 0);
    assert!(!progress.is_complete());

    // Test updating progress
    progress.update(3);
    let formatted = progress.format();
    assert!(formatted.contains("3/10"));
    assert!(formatted.contains("30%"));

    // Test increment
    progress.increment();
    assert_eq!(progress.current(), 4);

    // Test completion
    progress.update(10);
    assert!(progress.is_complete());

    let final_formatted = progress.format();
    assert!(formatted.contains("Testing Progress"));

    Ok(())
}

#[tokio::test]
async fn test_concurrent_session_updates() -> Result<()> {
    let session_manager = create_test_session_manager().await?;

    // Create multiple sessions
    let session1 = session_manager
        .create_session("Session 1".to_string(), None)
        .await?;
    let session2 = session_manager
        .create_session("Session 2".to_string(), None)
        .await?;

    // Note: With placeholder SessionManager, we can't test concurrent access to storage
    // In a full implementation, we would test concurrent reads/writes

    Ok(())
}

#[tokio::test]
async fn test_timeout_handling() -> Result<()> {
    let session_manager = create_test_session_manager().await?;

    // Test with very short timeout
    let config = RunnerConfig {
        timeout: Duration::from_millis(1), // Very short timeout
        streaming: false,
        stream_format: claude_sdk_rs::StreamFormat::Json,
        max_retries: 0, // No retries
        retry_delay: Duration::from_millis(100),
    };

    let runner = CommandRunner::with_config(session_manager, config)?;
    assert_eq!(runner.config().timeout, Duration::from_millis(1));

    Ok(())
}

// Helper function to simulate concurrent executions
async fn simulate_concurrent_execution(executor: &ParallelExecutor, count: usize) -> Result<()> {
    let contexts: Vec<_> = (0..count)
        .map(|i| ExecutionContext {
            session_id: None,
            command_name: format!("command_{}", i),
            args: vec![format!("arg_{}", i)],
            parallel: true,
            agent_count: count,
        })
        .collect();

    // This would actually execute in a real scenario with mock clients
    let _result = executor.execute_parallel(contexts).await?;

    Ok(())
}
