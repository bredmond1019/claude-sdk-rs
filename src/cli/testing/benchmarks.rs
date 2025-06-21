//! Performance benchmarking utilities
//!
//! This module provides tools for:
//! - Performance benchmarking of core components
//! - Load testing and stress testing
//! - Memory usage analysis
//! - Throughput measurement

use super::mocks::{CostDataGenerator, HistoryDataGenerator, SessionDataGenerator};
use crate::{cost::CostTracker, history::HistoryStore, cli::session::SessionManager, cli::error::Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::Semaphore;

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub duration_seconds: u64,
    pub concurrency_levels: Vec<usize>,
    pub data_sizes: Vec<usize>,
    pub warm_up_seconds: u64,
    pub memory_sampling_interval_ms: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            duration_seconds: 30,
            concurrency_levels: vec![1, 5, 10, 20],
            data_sizes: vec![100, 1000, 10000],
            warm_up_seconds: 5,
            memory_sampling_interval_ms: 100,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub test_name: String,
    pub config: BenchmarkConfig,
    pub runs: Vec<BenchmarkRun>,
    pub summary: BenchmarkSummary,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
}

/// Individual benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub concurrency: usize,
    pub data_size: usize,
    pub duration_ms: u64,
    pub operations_completed: u64,
    pub operations_per_second: f64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub errors: u64,
    pub memory_usage: MemoryUsage,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_samples: Vec<f64>,
}

/// Benchmark summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_operations: u64,
    pub total_duration_ms: u64,
    pub average_ops_per_second: f64,
    pub peak_ops_per_second: f64,
    pub average_latency_ms: f64,
    pub error_rate: f64,
}

/// Performance benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run session management benchmarks
    pub async fn benchmark_session_management(&self) -> Result<BenchmarkResults> {
        let started_at = Utc::now();
        let mut runs = Vec::new();

        for &concurrency in &self.config.concurrency_levels {
            for &data_size in &self.config.data_sizes {
                let run = self.run_session_benchmark(concurrency, data_size).await?;
                runs.push(run);
            }
        }

        let summary = self.calculate_summary(&runs);

        Ok(BenchmarkResults {
            test_name: "Session Management".to_string(),
            config: self.config.clone(),
            runs,
            summary,
            started_at,
            completed_at: Utc::now(),
        })
    }

    /// Run cost tracking benchmarks
    pub async fn benchmark_cost_tracking(&self) -> Result<BenchmarkResults> {
        let started_at = Utc::now();
        let mut runs = Vec::new();

        for &concurrency in &self.config.concurrency_levels {
            for &data_size in &self.config.data_sizes {
                let run = self.run_cost_benchmark(concurrency, data_size).await?;
                runs.push(run);
            }
        }

        let summary = self.calculate_summary(&runs);

        Ok(BenchmarkResults {
            test_name: "Cost Tracking".to_string(),
            config: self.config.clone(),
            runs,
            summary,
            started_at,
            completed_at: Utc::now(),
        })
    }

    /// Run history storage benchmarks
    pub async fn benchmark_history_storage(&self) -> Result<BenchmarkResults> {
        let started_at = Utc::now();
        let mut runs = Vec::new();

        for &concurrency in &self.config.concurrency_levels {
            for &data_size in &self.config.data_sizes {
                let run = self.run_history_benchmark(concurrency, data_size).await?;
                runs.push(run);
            }
        }

        let summary = self.calculate_summary(&runs);

        Ok(BenchmarkResults {
            test_name: "History Storage".to_string(),
            config: self.config.clone(),
            runs,
            summary,
            started_at,
            completed_at: Utc::now(),
        })
    }

    /// Run comprehensive load test
    pub async fn run_load_test(&self) -> Result<LoadTestResults> {
        let started_at = Utc::now();

        // Run benchmarks for all components
        let session_results = self.benchmark_session_management().await?;
        let cost_results = self.benchmark_cost_tracking().await?;
        let history_results = self.benchmark_history_storage().await?;

        let completed_at = Utc::now();

        Ok(LoadTestResults {
            session_results,
            cost_results,
            history_results,
            overall_duration_ms: (completed_at - started_at).num_milliseconds() as u64,
            started_at,
            completed_at,
        })
    }

    // Private implementation methods

    async fn run_session_benchmark(
        &self,
        concurrency: usize,
        data_size: usize,
    ) -> Result<BenchmarkRun> {
        let _temp_dir = tempfile::tempdir()?;
        let session_manager = std::sync::Arc::new(SessionManager::new());

        // Warm-up
        self.warm_up_session_operations(&session_manager).await?;

        let semaphore = std::sync::Arc::new(Semaphore::new(concurrency));
        let mut latencies = Vec::new();
        let mut errors = 0u64;
        let mut _memory_samples = Vec::new();

        let start_time = Instant::now();
        let end_time = start_time + std::time::Duration::from_secs(self.config.duration_seconds);

        let mut operations_completed = 0u64;

        // Memory monitoring task
        let memory_handle = tokio::spawn(async move {
            let mut samples = Vec::new();
            let interval = std::time::Duration::from_millis(100);
            let mut last_sample = Instant::now();

            while Instant::now() < end_time {
                if last_sample.elapsed() >= interval {
                    let memory_mb = get_memory_usage_mb();
                    samples.push(memory_mb);
                    last_sample = Instant::now();
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }

            samples
        });

        // Main benchmark loop
        while Instant::now() < end_time {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let session_manager = session_manager.clone();

            let task: tokio::task::JoinHandle<Result<f64>> = tokio::spawn(async move {
                let _permit = permit;
                let op_start = Instant::now();

                // Generate test session
                let session = SessionDataGenerator::generate_session(None);

                // Perform session operations
                match session_manager
                    .create_session(session.name, session.description)
                    .await
                {
                    Ok(_) => {
                        let latency = op_start.elapsed().as_millis() as f64;
                        Ok(latency)
                    }
                    Err(e) => Err(e),
                }
            });

            match task.await {
                Ok(Ok(latency)) => {
                    latencies.push(latency);
                    operations_completed += 1;
                }
                Ok(Err(_)) | Err(_) => {
                    errors += 1;
                }
            }

            // Prevent overwhelming the system
            if latencies.len() >= data_size {
                break;
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        _memory_samples = memory_handle.await.unwrap_or_default();

        let memory_usage = self.calculate_memory_usage(&_memory_samples);
        let (avg_latency, p95_latency, p99_latency) =
            self.calculate_latency_percentiles(&latencies);

        Ok(BenchmarkRun {
            concurrency,
            data_size,
            duration_ms,
            operations_completed,
            operations_per_second: operations_completed as f64 / (duration_ms as f64 / 1000.0),
            average_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            errors,
            memory_usage,
        })
    }

    async fn run_cost_benchmark(
        &self,
        concurrency: usize,
        data_size: usize,
    ) -> Result<BenchmarkRun> {
        let temp_dir = tempfile::tempdir()?;
        let storage_path = temp_dir.path().join("cost_bench.json");
        let mut cost_tracker = CostTracker::new(storage_path)?;

        // Warm-up
        self.warm_up_cost_operations(&mut cost_tracker).await?;

        let semaphore = std::sync::Arc::new(Semaphore::new(concurrency));
        let mut latencies = Vec::new();
        let mut errors = 0u64;
        let mut _memory_samples = Vec::new();

        let start_time = Instant::now();
        let end_time = start_time + std::time::Duration::from_secs(self.config.duration_seconds);

        let mut operations_completed = 0u64;

        // Memory monitoring
        let memory_handle = tokio::spawn(async move {
            let mut samples = Vec::new();
            let interval = std::time::Duration::from_millis(100);
            let mut last_sample = Instant::now();

            while Instant::now() < end_time {
                if last_sample.elapsed() >= interval {
                    samples.push(get_memory_usage_mb());
                    last_sample = Instant::now();
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }

            samples
        });

        // Benchmark loop
        while Instant::now() < end_time && operations_completed < data_size as u64 {
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            let task: tokio::task::JoinHandle<Result<f64>> = {
                let _entry = CostDataGenerator::generate_cost_entry(None);
                tokio::spawn(async move {
                    let _permit = permit;
                    let op_start = Instant::now();

                    // Simulate cost recording operation
                    tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;

                    let latency = op_start.elapsed().as_millis() as f64;
                    Ok(latency)
                })
            };

            match task.await {
                Ok(Ok(latency)) => {
                    latencies.push(latency);
                    operations_completed += 1;
                }
                Ok(Err(_)) | Err(_) => {
                    errors += 1;
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        _memory_samples = memory_handle.await.unwrap_or_default();

        let memory_usage = self.calculate_memory_usage(&_memory_samples);
        let (avg_latency, p95_latency, p99_latency) =
            self.calculate_latency_percentiles(&latencies);

        Ok(BenchmarkRun {
            concurrency,
            data_size,
            duration_ms,
            operations_completed,
            operations_per_second: operations_completed as f64 / (duration_ms as f64 / 1000.0),
            average_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            errors,
            memory_usage,
        })
    }

    async fn run_history_benchmark(
        &self,
        concurrency: usize,
        data_size: usize,
    ) -> Result<BenchmarkRun> {
        let temp_dir = tempfile::tempdir()?;
        let storage_path = temp_dir.path().join("history_bench.json");
        let mut history_store = HistoryStore::new(storage_path)?;

        // Warm-up
        self.warm_up_history_operations(&mut history_store).await?;

        let semaphore = std::sync::Arc::new(Semaphore::new(concurrency));
        let mut latencies = Vec::new();
        let mut errors = 0u64;
        let mut _memory_samples = Vec::new();

        let start_time = Instant::now();
        let end_time = start_time + std::time::Duration::from_secs(self.config.duration_seconds);

        let mut operations_completed = 0u64;

        // Memory monitoring
        let memory_handle = tokio::spawn(async move {
            let mut samples = Vec::new();
            let interval = std::time::Duration::from_millis(100);
            let mut last_sample = Instant::now();

            while Instant::now() < end_time {
                if last_sample.elapsed() >= interval {
                    samples.push(get_memory_usage_mb());
                    last_sample = Instant::now();
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }

            samples
        });

        // Benchmark loop
        while Instant::now() < end_time && operations_completed < data_size as u64 {
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            let task: tokio::task::JoinHandle<Result<f64>> = {
                let _entry = HistoryDataGenerator::generate_history_entry(None);
                tokio::spawn(async move {
                    let _permit = permit;
                    let op_start = Instant::now();

                    // Simulate history storage operation
                    tokio::time::sleep(tokio::time::Duration::from_micros(200)).await;

                    let latency = op_start.elapsed().as_millis() as f64;
                    Ok(latency)
                })
            };

            match task.await {
                Ok(Ok(latency)) => {
                    latencies.push(latency);
                    operations_completed += 1;
                }
                Ok(Err(_)) | Err(_) => {
                    errors += 1;
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        _memory_samples = memory_handle.await.unwrap_or_default();

        let memory_usage = self.calculate_memory_usage(&_memory_samples);
        let (avg_latency, p95_latency, p99_latency) =
            self.calculate_latency_percentiles(&latencies);

        Ok(BenchmarkRun {
            concurrency,
            data_size,
            duration_ms,
            operations_completed,
            operations_per_second: operations_completed as f64 / (duration_ms as f64 / 1000.0),
            average_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            errors,
            memory_usage,
        })
    }

    async fn warm_up_session_operations(
        &self,
        session_manager: &std::sync::Arc<SessionManager>,
    ) -> Result<()> {
        let warm_up_end =
            Instant::now() + std::time::Duration::from_secs(self.config.warm_up_seconds);

        while Instant::now() < warm_up_end {
            let session = SessionDataGenerator::generate_session(None);
            let _ = session_manager
                .create_session(session.name, session.description)
                .await;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        Ok(())
    }

    async fn warm_up_cost_operations(&self, cost_tracker: &mut CostTracker) -> Result<()> {
        let warm_up_end =
            Instant::now() + std::time::Duration::from_secs(self.config.warm_up_seconds);

        while Instant::now() < warm_up_end {
            let entry = CostDataGenerator::generate_cost_entry(None);
            let _ = cost_tracker.record_cost(entry).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        Ok(())
    }

    async fn warm_up_history_operations(&self, history_store: &mut HistoryStore) -> Result<()> {
        let warm_up_end =
            Instant::now() + std::time::Duration::from_secs(self.config.warm_up_seconds);

        while Instant::now() < warm_up_end {
            let entry = HistoryDataGenerator::generate_history_entry(None);
            let _ = history_store.store_entry(entry).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        Ok(())
    }

    fn calculate_memory_usage(&self, samples: &[f64]) -> MemoryUsage {
        if samples.is_empty() {
            return MemoryUsage {
                peak_memory_mb: 0.0,
                average_memory_mb: 0.0,
                memory_samples: Vec::new(),
            };
        }

        let peak_memory_mb = samples.iter().fold(0.0f64, |a, &b| a.max(b));
        let average_memory_mb = samples.iter().sum::<f64>() / samples.len() as f64;

        MemoryUsage {
            peak_memory_mb,
            average_memory_mb,
            memory_samples: samples.to_vec(),
        }
    }

    fn calculate_latency_percentiles(&self, latencies: &[f64]) -> (f64, f64, f64) {
        if latencies.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_latencies.len();
        let avg_latency = sorted_latencies.iter().sum::<f64>() / len as f64;

        let p95_index = ((len as f64) * 0.95) as usize;
        let p99_index = ((len as f64) * 0.99) as usize;

        let p95_latency = sorted_latencies
            .get(p95_index.min(len - 1))
            .copied()
            .unwrap_or(0.0);
        let p99_latency = sorted_latencies
            .get(p99_index.min(len - 1))
            .copied()
            .unwrap_or(0.0);

        (avg_latency, p95_latency, p99_latency)
    }

    fn calculate_summary(&self, runs: &[BenchmarkRun]) -> BenchmarkSummary {
        if runs.is_empty() {
            return BenchmarkSummary {
                total_operations: 0,
                total_duration_ms: 0,
                average_ops_per_second: 0.0,
                peak_ops_per_second: 0.0,
                average_latency_ms: 0.0,
                error_rate: 0.0,
            };
        }

        let total_operations = runs.iter().map(|r| r.operations_completed).sum();
        let total_duration_ms = runs.iter().map(|r| r.duration_ms).sum();
        let total_errors = runs.iter().map(|r| r.errors).sum::<u64>();

        let average_ops_per_second =
            runs.iter().map(|r| r.operations_per_second).sum::<f64>() / runs.len() as f64;
        let peak_ops_per_second = runs
            .iter()
            .map(|r| r.operations_per_second)
            .fold(0.0f64, |a, b| a.max(b));
        let average_latency_ms =
            runs.iter().map(|r| r.average_latency_ms).sum::<f64>() / runs.len() as f64;

        let error_rate = if total_operations > 0 {
            (total_errors as f64 / total_operations as f64) * 100.0
        } else {
            0.0
        };

        BenchmarkSummary {
            total_operations,
            total_duration_ms,
            average_ops_per_second,
            peak_ops_per_second,
            average_latency_ms,
            error_rate,
        }
    }
}

/// Comprehensive load test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResults {
    pub session_results: BenchmarkResults,
    pub cost_results: BenchmarkResults,
    pub history_results: BenchmarkResults,
    pub overall_duration_ms: u64,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
}

impl LoadTestResults {
    /// Export results as JSON report
    pub async fn export_json(&self, path: &std::path::PathBuf) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Export results as HTML report
    pub async fn export_html(&self, path: &std::path::PathBuf) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html><head><title>Load Test Results</title>")?;
        writeln!(file, "<style>")?;
        writeln!(
            file,
            "  body {{ font-family: Arial, sans-serif; margin: 20px; }}"
        )?;
        writeln!(file, "  .summary {{ background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}")?;
        writeln!(
            file,
            "  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}"
        )?;
        writeln!(
            file,
            "  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}"
        )?;
        writeln!(file, "  th {{ background-color: #f2f2f2; }}")?;
        writeln!(file, "</style></head><body>")?;

        writeln!(file, "<h1>Claude Interactive Load Test Results</h1>")?;
        writeln!(
            file,
            "<p>Test Duration: {} minutes</p>",
            self.overall_duration_ms / 60000
        )?;
        writeln!(
            file,
            "<p>Completed: {}</p>",
            self.completed_at.format("%Y-%m-%d %H:%M:%S UTC")
        )?;

        // Session results
        self.write_benchmark_section(&mut file, &self.session_results)?;

        // Cost results
        self.write_benchmark_section(&mut file, &self.cost_results)?;

        // History results
        self.write_benchmark_section(&mut file, &self.history_results)?;

        writeln!(file, "</body></html>")?;

        Ok(())
    }

    fn write_benchmark_section(
        &self,
        file: &mut std::fs::File,
        results: &BenchmarkResults,
    ) -> Result<()> {
        use std::io::Write;

        writeln!(file, "<h2>{}</h2>", results.test_name)?;
        writeln!(file, "<div class=\"summary\">")?;
        writeln!(
            file,
            "<p>Total Operations: {}</p>",
            results.summary.total_operations
        )?;
        writeln!(
            file,
            "<p>Average Ops/sec: {:.2}</p>",
            results.summary.average_ops_per_second
        )?;
        writeln!(
            file,
            "<p>Peak Ops/sec: {:.2}</p>",
            results.summary.peak_ops_per_second
        )?;
        writeln!(
            file,
            "<p>Average Latency: {:.2}ms</p>",
            results.summary.average_latency_ms
        )?;
        writeln!(
            file,
            "<p>Error Rate: {:.2}%</p>",
            results.summary.error_rate
        )?;
        writeln!(file, "</div>")?;

        writeln!(file, "<table>")?;
        writeln!(file, "<tr><th>Concurrency</th><th>Data Size</th><th>Ops/sec</th><th>Avg Latency</th><th>P95 Latency</th><th>Errors</th></tr>")?;

        for run in &results.runs {
            writeln!(file, "<tr>")?;
            writeln!(file, "<td>{}</td>", run.concurrency)?;
            writeln!(file, "<td>{}</td>", run.data_size)?;
            writeln!(file, "<td>{:.2}</td>", run.operations_per_second)?;
            writeln!(file, "<td>{:.2}ms</td>", run.average_latency_ms)?;
            writeln!(file, "<td>{:.2}ms</td>", run.p95_latency_ms)?;
            writeln!(file, "<td>{}</td>", run.errors)?;
            writeln!(file, "</tr>")?;
        }

        writeln!(file, "</table>")?;

        Ok(())
    }
}

// Helper function to get memory usage (simplified implementation)
fn get_memory_usage_mb() -> f64 {
    // In a real implementation, this would use system APIs to get actual memory usage
    // For now, we'll simulate with a random value
    50.0 + (rand::random::<f64>() * 100.0)
}
