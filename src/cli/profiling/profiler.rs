//! Application profiler for performance analysis

use super::*;
use crate::cli::error::Result;
use crate::cli::history::store::{EnhancedHistoryStore, FullTextSearch, SearchField};
use crate::cli::history::{HistoryEntry, HistorySearch, HistoryStore};
use crate::cli::session::SessionManager;
use std::sync::Arc;
use tempfile::TempDir;
use uuid::Uuid;

/// Main profiler for the application
pub struct ApplicationProfiler {
    temp_dir: TempDir,
    results: Vec<ProfilingResult>,
}

impl ApplicationProfiler {
    /// Create a new application profiler
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        Ok(Self {
            temp_dir,
            results: Vec::new(),
        })
    }

    /// Run a comprehensive profiling suite with the given configuration
    pub async fn run_profiling_suite(
        &mut self,
        config: ProfilingConfig,
    ) -> Result<Vec<ProfilingResult>> {
        println!(
            "Starting profiling suite with {} operations on {} entries",
            config.operation_count, config.dataset_size
        );

        let mut suite_results = Vec::new();

        for operation_type in &config.operation_types {
            let result = match operation_type {
                OperationType::HistoryStore => self.profile_history_storage(&config).await?,
                OperationType::HistorySearch => self.profile_history_search(&config).await?,
                OperationType::FullTextSearch => self.profile_full_text_search(&config).await?,
                OperationType::IndexOptimization => {
                    self.profile_index_optimization(&config).await?
                }
                OperationType::SessionManagement => {
                    self.profile_session_management(&config).await?
                }
                OperationType::BatchOperations => self.profile_batch_operations(&config).await?,
            };

            suite_results.push(result);
        }

        self.results.extend(suite_results.clone());
        Ok(suite_results)
    }

    /// Profile history storage operations
    async fn profile_history_storage(&self, config: &ProfilingConfig) -> Result<ProfilingResult> {
        let storage_path = self.temp_dir.path().join("history_profile.json");
        let mut store = HistoryStore::new(storage_path)?;

        let mut metrics = PerformanceMetrics::new();
        let initial_memory = get_memory_usage_mb();
        metrics.memory_usage.initial_mb = initial_memory;

        // Generate test data
        let entries = self.generate_test_entries(config.dataset_size);

        // Warmup if configured
        if config.warmup {
            for entry in entries.iter().take(config.warmup_iterations) {
                store.store_entry(entry.clone()).await?;
            }
        }

        let timer = Timer::start("history_storage");

        // Perform storage operations
        for entry in entries.iter().take(config.operation_count) {
            let storage_timer = Timer::start("single_store");
            store.store_entry(entry.clone()).await?;
            let storage_duration = storage_timer.stop();

            metrics.storage_metrics.writes += 1;
            metrics.storage_metrics.write_time += storage_duration;
        }

        metrics.elapsed_time = timer.stop();
        metrics.operation_count = config.operation_count;
        metrics.memory_usage.final_mb = get_memory_usage_mb();
        metrics.peak_memory_mb = metrics.memory_usage.final_mb.max(initial_memory);

        metrics.calculate_ops_per_second();
        metrics.calculate_storage_averages();

        Ok(ProfilingResult {
            scenario_name: "History Storage".to_string(),
            dataset_size: config.dataset_size,
            metrics,
            metadata: HashMap::new(),
            config: config.clone(),
        })
    }

    /// Profile history search operations
    async fn profile_history_search(&self, config: &ProfilingConfig) -> Result<ProfilingResult> {
        let storage_path = self.temp_dir.path().join("search_profile.json");
        let mut store = HistoryStore::new(storage_path)?;

        // Populate with test data
        let entries = self.generate_test_entries(config.dataset_size);
        for entry in &entries {
            store.store_entry(entry.clone()).await?;
        }

        let mut metrics = PerformanceMetrics::new();
        let initial_memory = get_memory_usage_mb();
        metrics.memory_usage.initial_mb = initial_memory;

        // Generate search criteria
        let search_patterns = self.generate_search_patterns(&entries);

        let timer = Timer::start("history_search");

        // Perform search operations
        for i in 0..config.operation_count {
            let pattern = &search_patterns[i % search_patterns.len()];
            let search_timer = Timer::start("single_search");

            let _results = store.search(pattern).await?;

            let search_duration = search_timer.stop();
            metrics.search_metrics.search_count += 1;
            metrics.search_metrics.total_search_time += search_duration;
        }

        metrics.elapsed_time = timer.stop();
        metrics.operation_count = config.operation_count;
        metrics.memory_usage.final_mb = get_memory_usage_mb();
        metrics.peak_memory_mb = metrics.memory_usage.final_mb.max(initial_memory);

        metrics.calculate_ops_per_second();
        metrics.calculate_search_averages();

        Ok(ProfilingResult {
            scenario_name: "History Search".to_string(),
            dataset_size: config.dataset_size,
            metrics,
            metadata: HashMap::new(),
            config: config.clone(),
        })
    }

    /// Profile full-text search operations
    async fn profile_full_text_search(&self, config: &ProfilingConfig) -> Result<ProfilingResult> {
        let storage_path = self.temp_dir.path().join("fulltext_profile");
        let store = EnhancedHistoryStore::new(storage_path)?;

        // Populate with test data
        let entries = self.generate_test_entries(config.dataset_size);
        // Note: For now, we simulate this. In a real implementation,
        // we would populate the enhanced store with test data

        let mut metrics = PerformanceMetrics::new();
        let initial_memory = get_memory_usage_mb();
        metrics.memory_usage.initial_mb = initial_memory;

        // Generate full-text search queries
        let search_queries = self.generate_fulltext_queries(&entries);

        let timer = Timer::start("fulltext_search");

        // Perform full-text search operations
        for i in 0..config.operation_count {
            let query = &search_queries[i % search_queries.len()];
            let search_timer = Timer::start("single_fulltext_search");

            let _results = store.full_text_search(query).await?;

            let search_duration = search_timer.stop();
            metrics.search_metrics.search_count += 1;
            metrics.search_metrics.total_search_time += search_duration;
        }

        metrics.elapsed_time = timer.stop();
        metrics.operation_count = config.operation_count;
        metrics.memory_usage.final_mb = get_memory_usage_mb();
        metrics.peak_memory_mb = metrics.memory_usage.final_mb.max(initial_memory);

        metrics.calculate_ops_per_second();
        metrics.calculate_search_averages();

        Ok(ProfilingResult {
            scenario_name: "Full-Text Search".to_string(),
            dataset_size: config.dataset_size,
            metrics,
            metadata: HashMap::new(),
            config: config.clone(),
        })
    }

    /// Profile index optimization operations
    async fn profile_index_optimization(
        &self,
        config: &ProfilingConfig,
    ) -> Result<ProfilingResult> {
        let storage_path = self.temp_dir.path().join("index_profile");
        let mut store = EnhancedHistoryStore::new(storage_path)?;

        // Populate with test data first
        let _entries = self.generate_test_entries(config.dataset_size);
        // Simulate populating the store

        let mut metrics = PerformanceMetrics::new();
        let initial_memory = get_memory_usage_mb();
        metrics.memory_usage.initial_mb = initial_memory;

        let timer = Timer::start("index_optimization");

        // Perform index optimization
        let optimization_result = store.optimize_index().await?;

        metrics.elapsed_time = timer.stop();
        metrics.operation_count = 1; // Single optimization operation
        metrics.memory_usage.final_mb = get_memory_usage_mb();
        metrics.peak_memory_mb = metrics.memory_usage.final_mb.max(initial_memory);

        metrics.calculate_ops_per_second();

        let mut metadata = HashMap::new();
        metadata.insert(
            "optimization_duration".to_string(),
            format!("{:?}", optimization_result.optimization_duration),
        );
        metadata.insert(
            "words_removed".to_string(),
            optimization_result.removed_words.to_string(),
        );
        metadata.insert(
            "memory_saved".to_string(),
            optimization_result.memory_saved.to_string(),
        );

        Ok(ProfilingResult {
            scenario_name: "Index Optimization".to_string(),
            dataset_size: config.dataset_size,
            metrics,
            metadata,
            config: config.clone(),
        })
    }

    /// Profile session management operations
    async fn profile_session_management(
        &self,
        config: &ProfilingConfig,
    ) -> Result<ProfilingResult> {
        let _storage_path = self.temp_dir.path().join("session_profile");
        let session_manager = Arc::new(std::sync::Mutex::new(SessionManager::new(
            self.temp_dir.path().to_path_buf(),
        )?));

        let mut metrics = PerformanceMetrics::new();
        let initial_memory = get_memory_usage_mb();
        metrics.memory_usage.initial_mb = initial_memory;

        let timer = Timer::start("session_management");

        // Perform session operations
        for i in 0..config.operation_count {
            let session_name = format!("test_session_{}", i);
            let session_timer = Timer::start("session_operation");

            // Create session
            let _session =
                session_manager
                    .lock()
                    .unwrap()
                    .create_session(session_name, None, None)?;

            let session_duration = session_timer.stop();
            metrics.storage_metrics.writes += 1;
            metrics.storage_metrics.write_time += session_duration;
        }

        metrics.elapsed_time = timer.stop();
        metrics.operation_count = config.operation_count;
        metrics.memory_usage.final_mb = get_memory_usage_mb();
        metrics.peak_memory_mb = metrics.memory_usage.final_mb.max(initial_memory);

        metrics.calculate_ops_per_second();
        metrics.calculate_storage_averages();

        Ok(ProfilingResult {
            scenario_name: "Session Management".to_string(),
            dataset_size: config.dataset_size,
            metrics,
            metadata: HashMap::new(),
            config: config.clone(),
        })
    }

    /// Profile batch operations
    async fn profile_batch_operations(&self, config: &ProfilingConfig) -> Result<ProfilingResult> {
        let storage_path = self.temp_dir.path().join("batch_profile.json");
        let mut store = HistoryStore::new(storage_path)?;

        let mut metrics = PerformanceMetrics::new();
        let initial_memory = get_memory_usage_mb();
        metrics.memory_usage.initial_mb = initial_memory;

        // Generate test data
        let entries = self.generate_test_entries(config.dataset_size);

        let timer = Timer::start("batch_operations");

        // Perform batch storage operations
        let batch_size = 100;
        let mut batch_count = 0;

        for chunk in entries.chunks(batch_size) {
            let batch_timer = Timer::start("batch_store");

            for entry in chunk {
                store.store_entry(entry.clone()).await?;
            }

            let batch_duration = batch_timer.stop();
            metrics.storage_metrics.writes += chunk.len();
            metrics.storage_metrics.write_time += batch_duration;
            batch_count += 1;

            if batch_count * batch_size >= config.operation_count {
                break;
            }
        }

        metrics.elapsed_time = timer.stop();
        metrics.operation_count = config.operation_count.min(entries.len());
        metrics.memory_usage.final_mb = get_memory_usage_mb();
        metrics.peak_memory_mb = metrics.memory_usage.final_mb.max(initial_memory);

        metrics.calculate_ops_per_second();
        metrics.calculate_storage_averages();

        Ok(ProfilingResult {
            scenario_name: "Batch Operations".to_string(),
            dataset_size: config.dataset_size,
            metrics,
            metadata: HashMap::new(),
            config: config.clone(),
        })
    }

    /// Generate test history entries for profiling
    fn generate_test_entries(&self, count: usize) -> Vec<HistoryEntry> {
        let mut entries = Vec::with_capacity(count);
        let session_id = Uuid::new_v4();

        for i in 0..count {
            let entry = HistoryEntry::new(
                session_id,
                format!("test_command_{}", i % 50), // 50 different command types
                vec![format!("arg_{}", i), format!("value_{}", i % 10)],
                format!(
                    "Output for operation {}: {}",
                    i,
                    "sample output text ".repeat(i % 20)
                ),
                i % 10 != 0,             // 90% success rate
                100 + (i % 1000) as u64, // Variable duration
            )
            .with_tags(vec![
                format!("tag_{}", i % 5),
                if i % 3 == 0 {
                    "important".to_string()
                } else {
                    "normal".to_string()
                },
            ]);

            entries.push(entry);
        }

        entries
    }

    /// Generate search patterns for profiling
    fn generate_search_patterns(&self, _entries: &[HistoryEntry]) -> Vec<HistorySearch> {
        let mut patterns = Vec::new();

        // Command pattern searches
        for i in 0..10 {
            patterns.push(HistorySearch {
                command_pattern: Some(format!("test_command_{}", i)),
                limit: 50,
                ..Default::default()
            });
        }

        // Output pattern searches
        patterns.push(HistorySearch {
            output_pattern: Some("sample output".to_string()),
            limit: 100,
            ..Default::default()
        });

        // Success/failure searches
        patterns.push(HistorySearch {
            success_only: true,
            limit: 200,
            ..Default::default()
        });

        patterns.push(HistorySearch {
            failures_only: true,
            limit: 50,
            ..Default::default()
        });

        // Tag-based searches
        patterns.push(HistorySearch {
            tags: vec!["important".to_string()],
            limit: 100,
            ..Default::default()
        });

        patterns
    }

    /// Generate full-text search queries for profiling
    fn generate_fulltext_queries(&self, _entries: &[HistoryEntry]) -> Vec<FullTextSearch> {
        vec![
            FullTextSearch {
                query: "test command output".to_string(),
                fields: vec![SearchField::All],
                fuzzy: false,
                highlight: false,
            },
            FullTextSearch {
                query: "sample".to_string(),
                fields: vec![SearchField::Output],
                fuzzy: true,
                highlight: true,
            },
            FullTextSearch {
                query: "arg_".to_string(),
                fields: vec![SearchField::Args],
                fuzzy: false,
                highlight: false,
            },
            FullTextSearch {
                query: "important operation".to_string(),
                fields: vec![SearchField::Tags, SearchField::Output],
                fuzzy: true,
                highlight: true,
            },
        ]
    }

    /// Get all profiling results
    pub fn get_results(&self) -> &[ProfilingResult] {
        &self.results
    }

    /// Save all results to a directory
    pub async fn save_results(&self, output_dir: &std::path::Path) -> Result<()> {
        tokio::fs::create_dir_all(output_dir).await?;

        for (i, result) in self.results.iter().enumerate() {
            let filename = format!("profile_result_{}.json", i);
            let filepath = output_dir.join(filename);
            result.save_to_file(&filepath).await?;
        }

        // Save summary report
        let summary_path = output_dir.join("profiling_summary.txt");
        let summary = self.generate_summary_report();
        tokio::fs::write(&summary_path, summary).await?;

        Ok(())
    }

    /// Generate a summary report of all profiling results
    fn generate_summary_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Profiling Summary Report\n");
        report.push_str("=======================\n\n");

        for result in &self.results {
            report.push_str(&result.generate_summary());
            report.push_str("\n\n");
        }

        // Add performance comparison
        if self.results.len() > 1 {
            report.push_str("Performance Comparison:\n");
            report.push_str("-----------------------\n");

            let mut ops_per_second: Vec<_> = self
                .results
                .iter()
                .map(|r| (&r.scenario_name, r.metrics.ops_per_second))
                .collect();
            ops_per_second
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (name, ops) in ops_per_second {
                report.push_str(&format!("{}: {:.2} ops/sec\n", name, ops));
            }
        }

        report
    }
}

impl Default for ApplicationProfiler {
    fn default() -> Self {
        Self::new().expect("Failed to create temporary directory for profiling")
    }
}
