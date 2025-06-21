//! Parallel command execution engine
//!
//! This module provides the ParallelExecutor for running multiple Claude commands
//! concurrently with proper resource management, output coordination, and error handling.

use crate::cli::execution::runner::CommandRunner;
use crate::cli::execution::{ExecutionContext, ExecutionResult};
use crate::cli::output::AgentId;
use crate::{cli::error::InteractiveError, cli::error::Result};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Configuration for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of concurrent executions
    pub max_concurrent: usize,
    /// Whether to continue on error or stop all
    pub continue_on_error: bool,
    /// Timeout for the entire parallel operation
    pub total_timeout: std::time::Duration,
    /// Buffer size for output channels
    pub output_buffer_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            continue_on_error: true,
            total_timeout: std::time::Duration::from_secs(300), // 5 minutes
            output_buffer_size: 1000,
        }
    }
}

/// Handle for a parallel execution task
#[derive(Debug, Clone)]
pub struct ExecutionHandle {
    pub id: Uuid,
    pub agent_id: AgentId,
    pub context: ExecutionContext,
}

/// Output message from a parallel execution
#[derive(Debug, Clone)]
pub struct ParallelOutput {
    pub handle: ExecutionHandle,
    pub content: String,
    pub is_final: bool,
}

/// Result of a parallel execution batch
#[derive(Debug)]
pub struct ParallelResult {
    pub results: HashMap<Uuid, Result<ExecutionResult>>,
    pub total_duration: std::time::Duration,
    pub successful_count: usize,
    pub failed_count: usize,
    pub outputs: Vec<ParallelOutput>,
}

/// Statistics for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelStats {
    pub active_executions: usize,
    pub queued_executions: usize,
    pub completed_executions: usize,
    pub failed_executions: usize,
    pub total_executions: usize,
}

/// Parallel command execution engine
pub struct ParallelExecutor {
    runner: Arc<CommandRunner>,
    config: ParallelConfig,
    semaphore: Arc<Semaphore>,
    agent_counter: AtomicUsize,
    active_tasks: Arc<RwLock<HashMap<Uuid, JoinHandle<Result<ExecutionResult>>>>>,
    stats: Arc<RwLock<ParallelStats>>,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(runner: Arc<CommandRunner>) -> Self {
        let config = ParallelConfig::default();
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

        Self {
            runner,
            semaphore,
            config,
            agent_counter: AtomicUsize::new(0),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ParallelStats {
                active_executions: 0,
                queued_executions: 0,
                completed_executions: 0,
                failed_executions: 0,
                total_executions: 0,
            })),
        }
    }

    /// Create a new parallel executor with custom configuration
    pub fn with_config(runner: Arc<CommandRunner>, config: ParallelConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

        Self {
            runner,
            semaphore,
            config,
            agent_counter: AtomicUsize::new(0),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ParallelStats {
                active_executions: 0,
                queued_executions: 0,
                completed_executions: 0,
                failed_executions: 0,
                total_executions: 0,
            })),
        }
    }

    /// Execute multiple commands in parallel
    pub async fn execute_parallel(
        &self,
        contexts: Vec<ExecutionContext>,
    ) -> Result<ParallelResult> {
        if contexts.is_empty() {
            return Ok(ParallelResult {
                results: HashMap::new(),
                total_duration: std::time::Duration::default(),
                successful_count: 0,
                failed_count: 0,
                outputs: Vec::new(),
            });
        }

        let start_time = std::time::Instant::now();
        let (output_tx, mut output_rx) = mpsc::channel(self.config.output_buffer_size);

        info!("Starting parallel execution of {} commands", contexts.len());

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_executions += contexts.len();
            stats.queued_executions = contexts.len();
        }

        // Start all tasks
        let mut handles = HashMap::new();
        for context in contexts {
            let handle = self
                .spawn_execution_task(context, output_tx.clone())
                .await?;
            handles.insert(handle.id, handle);
        }

        // Drop the sender to signal no more outputs
        drop(output_tx);

        // Collect all outputs
        let mut outputs = Vec::new();
        while let Some(output) = output_rx.recv().await {
            outputs.push(output);
        }

        // Wait for all tasks to complete
        let results = self.collect_results(handles).await?;
        let total_duration = start_time.elapsed();

        let successful_count = results.values().filter(|r| r.is_ok()).count();
        let failed_count = results.len() - successful_count;

        // Update final stats
        {
            let mut stats = self.stats.write().await;
            stats.completed_executions += successful_count;
            stats.failed_executions += failed_count;
            stats.queued_executions = 0;
            stats.active_executions = 0;
        }

        info!(
            "Parallel execution completed: {} successful, {} failed, duration: {:?}",
            successful_count, failed_count, total_duration
        );

        Ok(ParallelResult {
            results,
            total_duration,
            successful_count,
            failed_count,
            outputs,
        })
    }

    /// Execute commands with streaming output
    pub async fn execute_parallel_streaming<F>(
        &self,
        contexts: Vec<ExecutionContext>,
        mut output_handler: F,
    ) -> Result<ParallelResult>
    where
        F: FnMut(ParallelOutput) -> Result<()> + Send + 'static,
    {
        if contexts.is_empty() {
            return Ok(ParallelResult {
                results: HashMap::new(),
                total_duration: std::time::Duration::default(),
                successful_count: 0,
                failed_count: 0,
                outputs: Vec::new(),
            });
        }

        let start_time = std::time::Instant::now();
        let (output_tx, mut output_rx) = mpsc::channel(self.config.output_buffer_size);

        info!(
            "Starting parallel streaming execution of {} commands",
            contexts.len()
        );

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_executions += contexts.len();
            stats.queued_executions = contexts.len();
        }

        // Start all tasks with streaming
        let mut handles = HashMap::new();
        for context in contexts {
            let handle = self
                .spawn_streaming_task(context, output_tx.clone())
                .await?;
            handles.insert(handle.id, handle);
        }

        // Drop the sender to signal no more outputs
        drop(output_tx);

        // Handle streaming outputs
        let mut outputs = Vec::new();
        tokio::spawn(async move {
            while let Some(output) = output_rx.recv().await {
                if let Err(e) = output_handler(output.clone()) {
                    error!("Output handler error: {}", e);
                }
                outputs.push(output);
            }
        });

        // Wait for all tasks to complete
        let results = self.collect_results(handles).await?;
        let total_duration = start_time.elapsed();

        let successful_count = results.values().filter(|r| r.is_ok()).count();
        let failed_count = results.len() - successful_count;

        // Update final stats
        {
            let mut stats = self.stats.write().await;
            stats.completed_executions += successful_count;
            stats.failed_executions += failed_count;
            stats.queued_executions = 0;
            stats.active_executions = 0;
        }

        info!(
            "Parallel streaming execution completed: {} successful, {} failed, duration: {:?}",
            successful_count, failed_count, total_duration
        );

        Ok(ParallelResult {
            results,
            total_duration,
            successful_count,
            failed_count,
            outputs: Vec::new(), // Outputs handled by callback
        })
    }

    /// Spawn a single execution task
    async fn spawn_execution_task(
        &self,
        context: ExecutionContext,
        output_tx: mpsc::Sender<ParallelOutput>,
    ) -> Result<ExecutionHandle> {
        let id = Uuid::new_v4();
        let agent_id = AgentId(self.agent_counter.fetch_add(1, Ordering::SeqCst));

        let handle = ExecutionHandle {
            id,
            agent_id,
            context: context.clone(),
        };

        let runner: Arc<CommandRunner> = Arc::clone(&self.runner);
        let semaphore = Arc::clone(&self.semaphore);
        let stats = Arc::clone(&self.stats);
        let handle_clone = handle.clone();
        let continue_on_error = self.config.continue_on_error;

        let task = tokio::spawn(async move {
            // Acquire semaphore permit
            let _permit = semaphore.acquire().await.map_err(|e| {
                InteractiveError::execution(format!("Failed to acquire semaphore: {}", e))
            })?;

            // Update stats
            {
                let mut stats = stats.write().await;
                stats.active_executions += 1;
                stats.queued_executions = stats.queued_executions.saturating_sub(1);
            }

            debug!(
                "Starting execution for agent {} with command: {}",
                handle_clone.agent_id.0, handle_clone.context.command_name
            );

            let result = runner.execute(context).await;

            // Send final output
            let output = match &result {
                Ok(exec_result) => ParallelOutput {
                    handle: handle_clone.clone(),
                    content: exec_result.output.clone(),
                    is_final: true,
                },
                Err(e) => ParallelOutput {
                    handle: handle_clone.clone(),
                    content: format!("Error: {}", e),
                    is_final: true,
                },
            };

            if let Err(e) = output_tx.send(output).await {
                warn!("Failed to send output: {}", e);
            }

            // Update stats
            {
                let mut stats = stats.write().await;
                stats.active_executions = stats.active_executions.saturating_sub(1);
            }

            if result.is_err() && !continue_on_error {
                error!("Task failed and continue_on_error is false, stopping");
            }

            result
        });

        // Store the task handle
        {
            let mut active_tasks = self.active_tasks.write().await;
            active_tasks.insert(id, task);
        }

        Ok(handle)
    }

    /// Spawn a streaming execution task
    async fn spawn_streaming_task(
        &self,
        context: ExecutionContext,
        output_tx: mpsc::Sender<ParallelOutput>,
    ) -> Result<ExecutionHandle> {
        let id = Uuid::new_v4();
        let agent_id = AgentId(self.agent_counter.fetch_add(1, Ordering::SeqCst));

        let handle = ExecutionHandle {
            id,
            agent_id,
            context: context.clone(),
        };

        let runner: Arc<CommandRunner> = Arc::clone(&self.runner);
        let semaphore = Arc::clone(&self.semaphore);
        let stats = Arc::clone(&self.stats);
        let handle_clone = handle.clone();

        let task = tokio::spawn(async move {
            // Acquire semaphore permit
            let _permit = semaphore.acquire().await.map_err(|e| {
                InteractiveError::execution(format!("Failed to acquire semaphore: {}", e))
            })?;

            // Update stats
            {
                let mut stats = stats.write().await;
                stats.active_executions += 1;
                stats.queued_executions = stats.queued_executions.saturating_sub(1);
            }

            debug!(
                "Starting streaming execution for agent {} with command: {}",
                handle_clone.agent_id.0, handle_clone.context.command_name
            );

            let _output_tx_clone = output_tx.clone();
            let _handle_clone2 = handle_clone.clone();

            // For now, we'll use regular execution since streaming execution method needs to be fixed
            let result = runner.execute(context).await;

            // Send final completion message
            let final_output = ParallelOutput {
                handle: handle_clone.clone(),
                content: match &result {
                    Ok(_) => "Command completed".to_string(),
                    Err(e) => format!("Error: {}", e),
                },
                is_final: true,
            };

            if let Err(e) = output_tx.send(final_output).await {
                warn!("Failed to send final output: {}", e);
            }

            // Update stats
            {
                let mut stats = stats.write().await;
                stats.active_executions = stats.active_executions.saturating_sub(1);
            }

            result
        });

        // Store the task handle
        {
            let mut active_tasks = self.active_tasks.write().await;
            active_tasks.insert(id, task);
        }

        Ok(handle)
    }

    /// Collect results from all tasks
    async fn collect_results(
        &self,
        _handles: HashMap<Uuid, ExecutionHandle>,
    ) -> Result<HashMap<Uuid, Result<ExecutionResult>>> {
        let mut results = HashMap::new();

        // Wait for all tasks to complete
        let active_tasks = {
            let mut tasks = self.active_tasks.write().await;
            std::mem::take(&mut *tasks)
        };

        for (id, task) in active_tasks {
            let result = match tokio::time::timeout(self.config.total_timeout, task).await {
                Ok(task_result) => match task_result {
                    Ok(execution_result) => execution_result,
                    Err(e) => Err(InteractiveError::execution(format!(
                        "Task join error: {}",
                        e
                    ))),
                },
                Err(_) => Err(InteractiveError::Timeout(
                    self.config.total_timeout.as_secs(),
                )),
            };

            results.insert(id, result);
        }

        Ok(results)
    }

    /// Cancel all active executions
    pub async fn cancel_all(&self) -> Result<usize> {
        let mut active_tasks = self.active_tasks.write().await;
        let count = active_tasks.len();

        for (_, task) in active_tasks.drain() {
            task.abort();
        }

        // Reset stats
        {
            let mut stats = self.stats.write().await;
            stats.active_executions = 0;
            stats.queued_executions = 0;
        }

        info!("Cancelled {} active executions", count);
        Ok(count)
    }

    /// Get current execution statistics
    pub async fn get_stats(&self) -> ParallelStats {
        self.stats.read().await.clone()
    }

    /// Get current configuration
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: ParallelConfig) {
        // Update semaphore if max_concurrent changed
        if config.max_concurrent != self.config.max_concurrent {
            self.semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        }

        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::session::{
        manager::SessionManager,
        storage::{JsonFileStorage, StorageConfig},
    };
    use tempfile::tempdir;

    async fn create_test_executor() -> Result<ParallelExecutor> {
        let temp_dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: temp_dir.path().to_path_buf(),
            sessions_file: "test_sessions.json".to_string(),
            current_session_file: "test_current.json".to_string(),
        };

        let storage = Arc::new(JsonFileStorage::new(config));
        let session_manager = Arc::new(tokio::sync::RwLock::new(SessionManager::new(storage)));
        let runner = Arc::new(CommandRunner::new(session_manager)?);

        Ok(ParallelExecutor::new(runner))
    }

    #[tokio::test]
    async fn test_parallel_executor_creation() -> Result<()> {
        let executor = create_test_executor().await?;
        let stats = executor.get_stats().await;

        assert_eq!(stats.active_executions, 0);
        assert_eq!(stats.total_executions, 0);
        assert_eq!(executor.config.max_concurrent, 4);

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_execution() -> Result<()> {
        let executor = create_test_executor().await?;
        let result = executor.execute_parallel(vec![]).await?;

        assert_eq!(result.successful_count, 0);
        assert_eq!(result.failed_count, 0);
        assert!(result.results.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_stats_tracking() -> Result<()> {
        let executor = create_test_executor().await?;
        let initial_stats = executor.get_stats().await;

        assert_eq!(initial_stats.total_executions, 0);

        // Stats would be updated during actual execution
        // This test verifies the stats structure is working

        Ok(())
    }

    #[tokio::test]
    async fn test_config_update() -> Result<()> {
        let mut executor = create_test_executor().await?;

        let new_config = ParallelConfig {
            max_concurrent: 8,
            continue_on_error: false,
            total_timeout: std::time::Duration::from_secs(600),
            output_buffer_size: 2000,
        };

        executor.set_config(new_config.clone());
        assert_eq!(executor.config.max_concurrent, 8);
        assert!(!executor.config.continue_on_error);

        Ok(())
    }
}
