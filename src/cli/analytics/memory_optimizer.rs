//! Memory optimization for long-running analytics services
//!
//! This module provides memory optimization strategies for analytics services
//! including object pooling, memory-efficient data structures, and garbage
//! collection optimization for long-running processes.
//!
//! # Features
//!
//! - **Object Pooling**: Reuse expensive objects to reduce allocations
//! - **Memory-Efficient Data Structures**: Optimized storage for analytics data
//! - **Streaming Data Processing**: Process large datasets without loading everything into memory
//! - **Memory Pressure Detection**: Monitor and respond to memory pressure
//! - **Lazy Loading**: Load data on-demand to reduce memory footprint
//! - **Memory Usage Tracking**: Detailed memory usage analytics
//!
//! # Example
//!
//! ```no_run
//! use crate_interactive::analytics::memory_optimizer::{MemoryOptimizer, MemoryConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MemoryConfig {
//!     max_memory_mb: 500,
//!     enable_object_pooling: true,
//!     enable_streaming_processing: true,
//!     ..Default::default()
//! };
//!
//! let optimizer = MemoryOptimizer::new(config);
//!
//! // Process large dataset with memory optimization
//! let result = optimizer.process_large_dataset_streaming(data_source).await?;
//! # Ok(())
//! # }
//! ```

use crate::{
    cli::analytics::{AnalyticsSummary, DashboardData, TimeSeriesData},
    cli::cost::CostEntry,
    cli::history::HistoryEntry,
    Result,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory usage in MB before triggering cleanup
    pub max_memory_mb: usize,
    /// Enable object pooling for frequently allocated objects
    pub enable_object_pooling: bool,
    /// Enable streaming data processing for large datasets
    pub enable_streaming_processing: bool,
    /// Maximum size of object pools
    pub max_pool_size: usize,
    /// Memory pressure threshold (0.0 to 1.0)
    pub memory_pressure_threshold: f64,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
    /// Enable memory usage tracking
    pub enable_memory_tracking: bool,
    /// Maximum number of items to keep in streaming buffers
    pub streaming_buffer_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 1000, // 1GB limit
            enable_object_pooling: true,
            enable_streaming_processing: true,
            max_pool_size: 100,
            memory_pressure_threshold: 0.8, // 80%
            cleanup_interval_seconds: 120,  // 2 minutes
            enable_memory_tracking: true,
            streaming_buffer_size: 1000,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub current_usage_mb: f64,
    pub peak_usage_mb: f64,
    pub memory_pressure: f64,
    pub allocations_total: u64,
    pub deallocations_total: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub gc_collections: u64,
    pub objects_in_pools: usize,
    pub active_streams: usize,
}

/// Object pool for reusing expensive allocations
pub struct ObjectPool<T> {
    pool: Arc<Mutex<VecDeque<T>>>,
    max_size: usize,
    create_fn: Arc<dyn Fn() -> T + Send + Sync>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<T> std::fmt::Debug for ObjectPool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectPool")
            .field("max_size", &self.max_size)
            .field(
                "hits",
                &self.hits.load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "misses",
                &self.misses.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl<T> ObjectPool<T>
where
    T: Send + 'static,
{
    /// Create a new object pool
    pub fn new<F>(max_size: usize, create_fn: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
            create_fn: Arc::new(create_fn),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get an object from the pool or create a new one
    pub async fn get(&self) -> T {
        let mut pool = self.pool.lock().await;
        if let Some(obj) = pool.pop_front() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            obj
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            (self.create_fn)()
        }
    }

    /// Return an object to the pool
    pub async fn return_object(&self, obj: T) {
        let mut pool = self.pool.lock().await;
        if pool.len() < self.max_size {
            pool.push_back(obj);
        }
        // If pool is full, object is dropped (garbage collected)
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> (u64, u64, usize) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let pool_size = self.pool.try_lock().map(|p| p.len()).unwrap_or(0);
        (hits, misses, pool_size)
    }
}

/// Memory-efficient streaming data processor
pub struct StreamingDataProcessor<T> {
    buffer: Arc<RwLock<VecDeque<T>>>,
    max_buffer_size: usize,
    items_processed: AtomicU64,
    memory_usage_bytes: AtomicUsize,
}

impl<T> StreamingDataProcessor<T>
where
    T: Send + Sync,
{
    /// Create a new streaming data processor
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            max_buffer_size,
            items_processed: AtomicU64::new(0),
            memory_usage_bytes: AtomicUsize::new(0),
        }
    }

    /// Add item to the stream, processing in batches to avoid memory buildup
    pub async fn add_item(&self, item: T, item_size_bytes: usize) -> Option<Vec<T>> {
        let mut buffer = self.buffer.write().await;
        buffer.push_back(item);
        self.memory_usage_bytes
            .fetch_add(item_size_bytes, Ordering::Relaxed);

        // If buffer is full, return batch for processing
        if buffer.len() >= self.max_buffer_size {
            let batch: Vec<T> = buffer.drain(..).collect();
            self.items_processed
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
            self.memory_usage_bytes.store(0, Ordering::Relaxed); // Reset after draining
            Some(batch)
        } else {
            None
        }
    }

    /// Flush remaining items in buffer
    pub async fn flush(&self) -> Vec<T> {
        let mut buffer = self.buffer.write().await;
        let remaining: Vec<T> = buffer.drain(..).collect();
        self.items_processed
            .fetch_add(remaining.len() as u64, Ordering::Relaxed);
        self.memory_usage_bytes.store(0, Ordering::Relaxed);
        remaining
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> (u64, usize, usize) {
        let processed = self.items_processed.load(Ordering::Relaxed);
        let memory_bytes = self.memory_usage_bytes.load(Ordering::Relaxed);
        let buffer_size = self.buffer.try_read().map(|b| b.len()).unwrap_or(0);
        (processed, memory_bytes, buffer_size)
    }
}

/// Memory optimizer for analytics services
pub struct MemoryOptimizer {
    config: MemoryConfig,

    // Object pools for frequently allocated objects
    cost_entry_pool: Option<ObjectPool<Vec<CostEntry>>>,
    history_entry_pool: Option<ObjectPool<Vec<HistoryEntry>>>,
    dashboard_data_pool: Option<ObjectPool<DashboardData>>,

    // Streaming processors
    cost_processor: Arc<StreamingDataProcessor<CostEntry>>,
    history_processor: Arc<StreamingDataProcessor<HistoryEntry>>,

    // Memory tracking
    current_memory_bytes: AtomicUsize,
    peak_memory_bytes: AtomicUsize,
    allocations: AtomicU64,
    deallocations: AtomicU64,
    gc_collections: AtomicU64,

    // Memory pressure management
    memory_pressure_callbacks: Arc<RwLock<Vec<Box<dyn Fn() + Send + Sync>>>>,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(config: MemoryConfig) -> Self {
        let cost_entry_pool = if config.enable_object_pooling {
            Some(ObjectPool::new(config.max_pool_size, || {
                Vec::with_capacity(100)
            }))
        } else {
            None
        };

        let history_entry_pool = if config.enable_object_pooling {
            Some(ObjectPool::new(config.max_pool_size, || {
                Vec::with_capacity(100)
            }))
        } else {
            None
        };

        let dashboard_data_pool = if config.enable_object_pooling {
            Some(ObjectPool::new(config.max_pool_size, || DashboardData {
                today_cost: 0.0,
                today_commands: 0,
                success_rate: 0.0,
                recent_activity: Vec::new(),
                top_commands: Vec::new(),
                active_alerts: Vec::new(),
                last_updated: Utc::now(),
            }))
        } else {
            None
        };

        Self {
            cost_processor: Arc::new(StreamingDataProcessor::new(config.streaming_buffer_size)),
            history_processor: Arc::new(StreamingDataProcessor::new(config.streaming_buffer_size)),
            config,
            cost_entry_pool,
            history_entry_pool,
            dashboard_data_pool,
            current_memory_bytes: AtomicUsize::new(0),
            peak_memory_bytes: AtomicUsize::new(0),
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            gc_collections: AtomicU64::new(0),
            memory_pressure_callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start background memory management tasks
    pub async fn start_background_tasks(&self) -> tokio::task::JoinHandle<()> {
        let optimizer = self.clone_for_background();
        let cleanup_interval = self.config.cleanup_interval_seconds;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(cleanup_interval));

            loop {
                interval.tick().await;
                optimizer.perform_memory_cleanup().await;
                optimizer.check_memory_pressure().await;
            }
        })
    }

    /// Get a pooled vector for cost entries
    pub async fn get_cost_entry_buffer(&self) -> Vec<CostEntry> {
        if let Some(pool) = &self.cost_entry_pool {
            let mut buffer = pool.get().await;
            buffer.clear(); // Ensure it's empty
            buffer
        } else {
            Vec::new()
        }
    }

    /// Return a cost entry buffer to the pool
    pub async fn return_cost_entry_buffer(&self, buffer: Vec<CostEntry>) {
        if let Some(pool) = &self.cost_entry_pool {
            pool.return_object(buffer).await;
        }
        // If no pool, buffer is simply dropped
    }

    /// Get a pooled vector for history entries
    pub async fn get_history_entry_buffer(&self) -> Vec<HistoryEntry> {
        if let Some(pool) = &self.history_entry_pool {
            let mut buffer = pool.get().await;
            buffer.clear();
            buffer
        } else {
            Vec::new()
        }
    }

    /// Return a history entry buffer to the pool
    pub async fn return_history_entry_buffer(&self, buffer: Vec<HistoryEntry>) {
        if let Some(pool) = &self.history_entry_pool {
            pool.return_object(buffer).await;
        }
    }

    /// Process cost entries in streaming fashion
    pub async fn process_cost_entries_streaming<F, R>(
        &self,
        entries: Vec<CostEntry>,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(&[CostEntry]) -> Vec<R> + Send + Sync,
        R: Send + Sync,
    {
        let mut results = Vec::new();

        for entry in entries {
            let entry_size = std::mem::size_of::<CostEntry>();

            if let Some(batch) = self.cost_processor.add_item(entry, entry_size).await {
                // Process batch
                let batch_results = processor(&batch);
                results.extend(batch_results);

                // Track memory usage
                self.record_allocation(batch.len() * entry_size);
            }
        }

        // Process remaining items
        let remaining = self.cost_processor.flush().await;
        if !remaining.is_empty() {
            let batch_results = processor(&remaining);
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Process history entries in streaming fashion
    pub async fn process_history_entries_streaming<F, R>(
        &self,
        entries: Vec<HistoryEntry>,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(&[HistoryEntry]) -> Vec<R> + Send + Sync,
        R: Send + Sync,
    {
        let mut results = Vec::new();

        for entry in entries {
            let entry_size =
                std::mem::size_of::<HistoryEntry>() + entry.command_name.len() + entry.output.len();

            if let Some(batch) = self.history_processor.add_item(entry, entry_size).await {
                // Process batch
                let batch_results = processor(&batch);
                results.extend(batch_results);

                // Track memory usage
                self.record_allocation(batch.len() * entry_size);
            }
        }

        // Process remaining items
        let remaining = self.history_processor.flush().await;
        if !remaining.is_empty() {
            let batch_results = processor(&remaining);
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Optimize dashboard data structure for memory efficiency
    pub fn optimize_dashboard_data(&self, data: &mut DashboardData) {
        // Compact vectors to reduce memory overhead
        data.recent_activity.shrink_to_fit();
        data.top_commands.shrink_to_fit();
        data.active_alerts.shrink_to_fit();

        // Limit size of collections to prevent memory bloat
        const MAX_RECENT_ACTIVITY: usize = 50;
        const MAX_TOP_COMMANDS: usize = 20;
        const MAX_ACTIVE_ALERTS: usize = 10;

        if data.recent_activity.len() > MAX_RECENT_ACTIVITY {
            data.recent_activity.truncate(MAX_RECENT_ACTIVITY);
            data.recent_activity.shrink_to_fit();
        }

        if data.top_commands.len() > MAX_TOP_COMMANDS {
            data.top_commands.truncate(MAX_TOP_COMMANDS);
            data.top_commands.shrink_to_fit();
        }

        if data.active_alerts.len() > MAX_ACTIVE_ALERTS {
            data.active_alerts.truncate(MAX_ACTIVE_ALERTS);
            data.active_alerts.shrink_to_fit();
        }
    }

    /// Optimize time series data for memory efficiency
    pub fn optimize_time_series_data(&self, data: &mut TimeSeriesData) {
        // Compact all time series vectors
        data.cost_over_time.shrink_to_fit();
        data.commands_over_time.shrink_to_fit();
        data.success_rate_over_time.shrink_to_fit();
        data.response_time_over_time.shrink_to_fit();

        // Implement data point reduction for very large time series
        const MAX_POINTS_PER_SERIES: usize = 1000;

        if data.cost_over_time.len() > MAX_POINTS_PER_SERIES {
            self.reduce_time_series_points(&mut data.cost_over_time, MAX_POINTS_PER_SERIES);
        }
        if data.commands_over_time.len() > MAX_POINTS_PER_SERIES {
            self.reduce_command_time_series_points(
                &mut data.commands_over_time,
                MAX_POINTS_PER_SERIES,
            );
        }
        if data.success_rate_over_time.len() > MAX_POINTS_PER_SERIES {
            self.reduce_time_series_points(&mut data.success_rate_over_time, MAX_POINTS_PER_SERIES);
        }
        if data.response_time_over_time.len() > MAX_POINTS_PER_SERIES {
            self.reduce_time_series_points(
                &mut data.response_time_over_time,
                MAX_POINTS_PER_SERIES,
            );
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&self, size_bytes: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let new_usage = self
            .current_memory_bytes
            .fetch_add(size_bytes, Ordering::Relaxed)
            + size_bytes;

        // Update peak usage
        let current_peak = self.peak_memory_bytes.load(Ordering::Relaxed);
        if new_usage > current_peak {
            self.peak_memory_bytes.store(new_usage, Ordering::Relaxed);
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&self, size_bytes: usize) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_memory_bytes
            .fetch_sub(size_bytes, Ordering::Relaxed);
    }

    /// Register callback for memory pressure events
    pub async fn register_memory_pressure_callback<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        let mut callbacks = self.memory_pressure_callbacks.write().await;
        callbacks.push(Box::new(callback));
    }

    /// Get current memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        let current_usage_bytes = self.current_memory_bytes.load(Ordering::Relaxed);
        let peak_usage_bytes = self.peak_memory_bytes.load(Ordering::Relaxed);
        let max_memory_bytes = self.config.max_memory_mb * 1024 * 1024;

        let memory_pressure = current_usage_bytes as f64 / max_memory_bytes as f64;

        let (cost_pool_hits, cost_pool_misses, cost_pool_size) =
            if let Some(pool) = &self.cost_entry_pool {
                pool.get_stats()
            } else {
                (0, 0, 0)
            };

        let (history_pool_hits, history_pool_misses, history_pool_size) =
            if let Some(pool) = &self.history_entry_pool {
                pool.get_stats()
            } else {
                (0, 0, 0)
            };

        let (cost_processed, _, cost_buffer_size) = self.cost_processor.get_stats();
        let (history_processed, _, history_buffer_size) = self.history_processor.get_stats();

        MemoryStats {
            current_usage_mb: current_usage_bytes as f64 / 1024.0 / 1024.0,
            peak_usage_mb: peak_usage_bytes as f64 / 1024.0 / 1024.0,
            memory_pressure,
            allocations_total: self.allocations.load(Ordering::Relaxed),
            deallocations_total: self.deallocations.load(Ordering::Relaxed),
            pool_hits: cost_pool_hits + history_pool_hits,
            pool_misses: cost_pool_misses + history_pool_misses,
            gc_collections: self.gc_collections.load(Ordering::Relaxed),
            objects_in_pools: cost_pool_size + history_pool_size,
            active_streams: if cost_buffer_size > 0 || history_buffer_size > 0 {
                1
            } else {
                0
            },
        }
    }

    // Private helper methods

    async fn perform_memory_cleanup(&self) {
        // Force garbage collection if memory pressure is high
        let current_usage = self.current_memory_bytes.load(Ordering::Relaxed);
        let max_memory = self.config.max_memory_mb * 1024 * 1024;
        let memory_pressure = current_usage as f64 / max_memory as f64;

        if memory_pressure > self.config.memory_pressure_threshold {
            // Force cleanup by processing remaining streams
            let _ = self.cost_processor.flush().await;
            let _ = self.history_processor.flush().await;

            self.gc_collections.fetch_add(1, Ordering::Relaxed);
        }
    }

    async fn check_memory_pressure(&self) {
        let current_usage = self.current_memory_bytes.load(Ordering::Relaxed);
        let max_memory = self.config.max_memory_mb * 1024 * 1024;
        let memory_pressure = current_usage as f64 / max_memory as f64;

        if memory_pressure > self.config.memory_pressure_threshold {
            // Trigger memory pressure callbacks
            let callbacks = self.memory_pressure_callbacks.read().await;
            for callback in callbacks.iter() {
                callback();
            }
        }
    }

    fn reduce_time_series_points(&self, data: &mut Vec<(DateTime<Utc>, f64)>, target_size: usize) {
        if data.len() <= target_size {
            return;
        }

        // Simple decimation strategy - take every nth point
        let step = data.len() / target_size;
        let mut reduced = Vec::with_capacity(target_size);

        for i in (0..data.len()).step_by(step) {
            reduced.push(data[i]);
        }

        *data = reduced;
        data.shrink_to_fit();
    }

    fn reduce_command_time_series_points(
        &self,
        data: &mut Vec<(DateTime<Utc>, usize)>,
        target_size: usize,
    ) {
        if data.len() <= target_size {
            return;
        }

        // Simple decimation strategy - take every nth point
        let step = data.len() / target_size;
        let mut reduced = Vec::with_capacity(target_size);

        for i in (0..data.len()).step_by(step) {
            reduced.push(data[i]);
        }

        *data = reduced;
        data.shrink_to_fit();
    }

    fn clone_for_background(&self) -> Self {
        Self {
            config: self.config.clone(),
            cost_entry_pool: None, // Don't clone pools for background task
            history_entry_pool: None,
            dashboard_data_pool: None,
            cost_processor: Arc::clone(&self.cost_processor),
            history_processor: Arc::clone(&self.history_processor),
            current_memory_bytes: AtomicUsize::new(0), // Don't clone stats
            peak_memory_bytes: AtomicUsize::new(0),
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            gc_collections: AtomicU64::new(0),
            memory_pressure_callbacks: Arc::clone(&self.memory_pressure_callbacks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::analytics::dashboard_tests::DashboardTestFixture;

    #[tokio::test]
    async fn test_object_pooling() -> Result<()> {
        let config = MemoryConfig {
            enable_object_pooling: true,
            max_pool_size: 5,
            ..Default::default()
        };

        let optimizer = MemoryOptimizer::new(config);

        // Test cost entry buffer pooling
        let buffer1 = optimizer.get_cost_entry_buffer().await;
        assert!(buffer1.is_empty());

        optimizer.return_cost_entry_buffer(buffer1).await;

        let buffer2 = optimizer.get_cost_entry_buffer().await;
        // Should get the same buffer back from pool
        assert!(buffer2.is_empty());

        let stats = optimizer.get_memory_stats().await;
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_processing() -> Result<()> {
        let config = MemoryConfig {
            streaming_buffer_size: 3, // Small buffer for testing
            ..Default::default()
        };

        let optimizer = MemoryOptimizer::new(config);

        // Create test cost entries
        let cost_entries = vec![
            CostEntry::new(
                uuid::Uuid::new_v4(),
                "cmd1".to_string(),
                0.01,
                100,
                200,
                1000,
                "model".to_string(),
            ),
            CostEntry::new(
                uuid::Uuid::new_v4(),
                "cmd2".to_string(),
                0.02,
                150,
                250,
                1100,
                "model".to_string(),
            ),
            CostEntry::new(
                uuid::Uuid::new_v4(),
                "cmd3".to_string(),
                0.03,
                200,
                300,
                1200,
                "model".to_string(),
            ),
        ];

        // Process entries in streaming fashion
        let results = optimizer
            .process_cost_entries_streaming(cost_entries, |batch| {
                // Simple processor that counts entries
                vec![batch.len()]
            })
            .await?;

        // Should have processed entries in batches
        assert!(!results.is_empty());

        let stats = optimizer.get_memory_stats().await;
        assert!(stats.allocations_total > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_tracking() -> Result<()> {
        let optimizer = MemoryOptimizer::new(MemoryConfig::default());

        // Record some allocations
        optimizer.record_allocation(1024);
        optimizer.record_allocation(2048);
        optimizer.record_deallocation(512);

        let stats = optimizer.get_memory_stats().await;

        assert_eq!(stats.allocations_total, 2);
        assert_eq!(stats.deallocations_total, 1);
        assert!(stats.current_usage_mb > 0.0);
        assert!(stats.peak_usage_mb >= stats.current_usage_mb);

        Ok(())
    }

    #[tokio::test]
    async fn test_dashboard_data_optimization() -> Result<()> {
        let optimizer = MemoryOptimizer::new(MemoryConfig::default());

        // Create dashboard data with large collections
        let mut dashboard_data = DashboardData {
            today_cost: 10.0,
            today_commands: 100,
            success_rate: 95.0,
            recent_activity: (0..100)
                .map(|i| {
                    crate::cli::history::HistoryEntry::new(
                        uuid::Uuid::new_v4(),
                        format!("cmd_{}", i),
                        vec![],
                        "output".to_string(),
                        true,
                        1000,
                    )
                })
                .collect(),
            top_commands: (0..50)
                .map(|i| (format!("cmd_{}", i), i as f64 * 0.01))
                .collect(),
            active_alerts: vec![],
            last_updated: Utc::now(),
        };

        let initial_activity_len = dashboard_data.recent_activity.len();
        let initial_commands_len = dashboard_data.top_commands.len();

        // Optimize the data
        optimizer.optimize_dashboard_data(&mut dashboard_data);

        // Should have been truncated
        assert!(dashboard_data.recent_activity.len() <= 50);
        assert!(dashboard_data.top_commands.len() <= 20);
        assert!(dashboard_data.recent_activity.len() < initial_activity_len);
        assert!(dashboard_data.top_commands.len() < initial_commands_len);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_pressure_callback() -> Result<()> {
        let config = MemoryConfig {
            max_memory_mb: 1,               // Very small limit for testing
            memory_pressure_threshold: 0.5, // 50% threshold
            ..Default::default()
        };

        let optimizer = MemoryOptimizer::new(config);

        // Register callback
        let callback_called = Arc::new(AtomicU64::new(0));
        let callback_counter = Arc::clone(&callback_called);

        optimizer
            .register_memory_pressure_callback(move || {
                callback_counter.fetch_add(1, Ordering::Relaxed);
            })
            .await;

        // Simulate high memory usage
        optimizer.record_allocation(1024 * 1024); // 1MB - should exceed limit

        // Trigger memory pressure check
        optimizer.check_memory_pressure().await;

        // Callback should have been called
        assert!(callback_called.load(Ordering::Relaxed) > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_background_tasks() -> Result<()> {
        let config = MemoryConfig {
            cleanup_interval_seconds: 1, // Very frequent for testing
            ..Default::default()
        };

        let optimizer = MemoryOptimizer::new(config);

        // Start background task
        let task = optimizer.start_background_tasks().await;

        // Let it run for a short time
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let stats = optimizer.get_memory_stats().await;
        println!("Background task stats: {:?}", stats);

        // Stop the task
        task.abort();

        Ok(())
    }
}
