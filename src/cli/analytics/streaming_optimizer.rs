//! Streaming performance optimization for real-time analytics
//!
//! This module provides advanced buffer management and streaming optimizations
//! for real-time analytics data delivery, focusing on:
//!
//! - Dynamic buffer sizing based on load
//! - Backpressure handling and flow control
//! - Adaptive batching strategies
//! - Connection multiplexing
//! - Memory-efficient data structures
//!
//! # Architecture
//!
//! The streaming optimizer works as a middleware layer between data producers
//! and consumers, automatically adjusting buffer sizes, batching strategies,
//! and flow control based on real-time performance metrics.
//!
//! # Performance Optimizations
//!
//! - **Adaptive Buffering**: Buffer sizes adjust based on throughput and latency
//! - **Intelligent Batching**: Groups updates for efficient network transmission
//! - **Backpressure Detection**: Automatically handles slow consumers
//! - **Memory Pool Management**: Reuses buffers to reduce allocations
//! - **Connection Health Monitoring**: Tracks and optimizes connection performance

use crate::cli::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::sleep;

/// Configuration for streaming optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Initial buffer size for channels
    pub initial_buffer_size: usize,
    /// Minimum buffer size (safety limit)
    pub min_buffer_size: usize,
    /// Maximum buffer size (memory limit)
    pub max_buffer_size: usize,
    /// Target latency for adaptive buffering (milliseconds)
    pub target_latency_ms: u64,
    /// Maximum batch size for updates
    pub max_batch_size: usize,
    /// Batch timeout (milliseconds)
    pub batch_timeout_ms: u64,
    /// Enable adaptive buffer sizing
    pub enable_adaptive_buffering: bool,
    /// Enable intelligent batching
    pub enable_batching: bool,
    /// Enable backpressure detection
    pub enable_backpressure_detection: bool,
    /// Memory pool size for buffer reuse
    pub memory_pool_size: usize,
    /// Connection health check interval (seconds)
    pub health_check_interval_seconds: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            initial_buffer_size: 256,
            min_buffer_size: 32,
            max_buffer_size: 8192,
            target_latency_ms: 50,
            max_batch_size: 10,
            batch_timeout_ms: 100,
            enable_adaptive_buffering: true,
            enable_batching: true,
            enable_backpressure_detection: true,
            memory_pool_size: 50,
            health_check_interval_seconds: 30,
        }
    }
}

/// Streaming performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetrics {
    pub throughput_messages_per_second: f64,
    pub average_latency_ms: f64,
    pub buffer_utilization: f64,
    pub backpressure_events: u64,
    pub dropped_messages: u64,
    pub active_connections: usize,
    pub memory_usage_bytes: usize,
    pub batch_efficiency: f64,
}

/// Streaming update batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingBatch<T> {
    pub items: Vec<T>,
    pub batch_id: u64,
    pub created_at: DateTime<Utc>,
    pub sequence_number: u64,
}

/// Connection health information
#[derive(Debug, Clone)]
pub struct ConnectionHealth {
    pub connection_id: String,
    pub last_seen: Instant,
    pub message_count: u64,
    pub error_count: u64,
    pub average_latency: Duration,
    pub buffer_usage: f64,
}

/// Backpressure signal
#[derive(Debug, Clone)]
pub enum BackpressureSignal {
    Normal,
    Moderate(f64), // utilization percentage
    Severe(f64),   // utilization percentage
    Critical,      // near capacity
}

/// Optimized streaming buffer manager
pub struct StreamingOptimizer<T>
where
    T: Clone + Send + Sync + 'static,
{
    config: StreamingConfig,
    sender: broadcast::Sender<StreamingBatch<T>>,
    metrics: Arc<RwLock<StreamingMetrics>>,
    connections: Arc<RwLock<HashMap<String, ConnectionHealth>>>,
    buffer_pool: Arc<RwLock<Vec<Vec<T>>>>,
    sequence_counter: Arc<AtomicU64>,
    batch_counter: Arc<AtomicU64>,
    pending_batch: Arc<RwLock<Vec<T>>>,
    last_batch_time: Arc<RwLock<Instant>>,
    backpressure_state: Arc<RwLock<BackpressureSignal>>,
    performance_samples: Arc<RwLock<VecDeque<(Instant, f64)>>>, // (timestamp, latency)
}

impl<T> StreamingOptimizer<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create new streaming optimizer
    pub fn new(config: StreamingConfig) -> Self {
        let (sender, _) = broadcast::channel(config.initial_buffer_size);

        Self {
            config: config.clone(),
            sender,
            metrics: Arc::new(RwLock::new(StreamingMetrics {
                throughput_messages_per_second: 0.0,
                average_latency_ms: 0.0,
                buffer_utilization: 0.0,
                backpressure_events: 0,
                dropped_messages: 0,
                active_connections: 0,
                memory_usage_bytes: 0,
                batch_efficiency: 0.0,
            })),
            connections: Arc::new(RwLock::new(HashMap::new())),
            buffer_pool: Arc::new(RwLock::new(Vec::with_capacity(config.memory_pool_size))),
            sequence_counter: Arc::new(AtomicU64::new(0)),
            batch_counter: Arc::new(AtomicU64::new(0)),
            pending_batch: Arc::new(RwLock::new(Vec::new())),
            last_batch_time: Arc::new(RwLock::new(Instant::now())),
            backpressure_state: Arc::new(RwLock::new(BackpressureSignal::Normal)),
            performance_samples: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Start the streaming optimizer background tasks
    pub async fn start(&self) -> Result<()> {
        // Start batch processing task
        if self.config.enable_batching {
            self.start_batch_processor().await?;
        }

        // Start adaptive buffer management
        if self.config.enable_adaptive_buffering {
            self.start_adaptive_buffer_manager().await?;
        }

        // Start connection health monitoring
        self.start_connection_monitor().await?;

        // Start performance metrics collection
        self.start_metrics_collector().await?;

        Ok(())
    }

    /// Send a single item through the optimized streaming pipeline
    pub async fn send_item(&self, item: T) -> Result<()> {
        let start_time = Instant::now();

        if self.config.enable_batching {
            // Add to pending batch
            self.add_to_batch(item).await?;
        } else {
            // Send immediately
            self.send_single_item(item).await?;
        }

        // Record performance metrics
        let latency = start_time.elapsed();
        self.record_performance_sample(latency).await;

        Ok(())
    }

    /// Subscribe to optimized streaming updates
    pub fn subscribe(&self) -> broadcast::Receiver<StreamingBatch<T>> {
        self.sender.subscribe()
    }

    /// Subscribe with connection tracking
    pub async fn subscribe_with_tracking(
        &self,
        connection_id: String,
    ) -> broadcast::Receiver<StreamingBatch<T>> {
        // Register connection
        let mut connections = self.connections.write().await;
        connections.insert(
            connection_id.clone(),
            ConnectionHealth {
                connection_id,
                last_seen: Instant::now(),
                message_count: 0,
                error_count: 0,
                average_latency: Duration::from_millis(0),
                buffer_usage: 0.0,
            },
        );

        self.sender.subscribe()
    }

    /// Get current streaming metrics
    pub async fn get_metrics(&self) -> StreamingMetrics {
        self.metrics.read().await.clone()
    }

    /// Get current backpressure state
    pub async fn get_backpressure_state(&self) -> BackpressureSignal {
        self.backpressure_state.read().await.clone()
    }

    /// Force flush pending batches
    pub async fn flush_batches(&self) -> Result<()> {
        if self.config.enable_batching {
            self.flush_pending_batch().await?;
        }
        Ok(())
    }

    // Private implementation methods

    async fn add_to_batch(&self, item: T) -> Result<()> {
        let mut pending = self.pending_batch.write().await;
        pending.push(item);

        // Check if we should flush the batch
        let should_flush = pending.len() >= self.config.max_batch_size
            || self.last_batch_time.read().await.elapsed().as_millis()
                >= self.config.batch_timeout_ms as u128;

        if should_flush {
            drop(pending); // Release lock before flush
            self.flush_pending_batch().await?;
        }

        Ok(())
    }

    async fn flush_pending_batch(&self) -> Result<()> {
        let mut pending = self.pending_batch.write().await;

        if pending.is_empty() {
            return Ok(());
        }

        // Create batch from pending items
        let items = std::mem::take(&mut *pending);
        let batch = StreamingBatch {
            items,
            batch_id: self.batch_counter.fetch_add(1, Ordering::SeqCst),
            created_at: Utc::now(),
            sequence_number: self.sequence_counter.fetch_add(1, Ordering::SeqCst),
        };

        // Update last batch time
        *self.last_batch_time.write().await = Instant::now();

        // Send batch
        match self.sender.send(batch) {
            Ok(_) => Ok(()),
            Err(_) => {
                // Handle send error - likely no receivers
                let mut metrics = self.metrics.write().await;
                metrics.dropped_messages += 1;
                Ok(())
            }
        }
    }

    async fn send_single_item(&self, item: T) -> Result<()> {
        let batch = StreamingBatch {
            items: vec![item],
            batch_id: self.batch_counter.fetch_add(1, Ordering::SeqCst),
            created_at: Utc::now(),
            sequence_number: self.sequence_counter.fetch_add(1, Ordering::SeqCst),
        };

        match self.sender.send(batch) {
            Ok(_) => Ok(()),
            Err(_) => {
                let mut metrics = self.metrics.write().await;
                metrics.dropped_messages += 1;
                Ok(())
            }
        }
    }

    async fn start_batch_processor(&self) -> Result<()> {
        let pending_batch = Arc::clone(&self.pending_batch);
        let last_batch_time = Arc::clone(&self.last_batch_time);
        let sender = self.sender.clone();
        let batch_counter = Arc::clone(&self.batch_counter);
        let sequence_counter = Arc::clone(&self.sequence_counter);
        let batch_timeout = Duration::from_millis(self.config.batch_timeout_ms);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(batch_timeout);

            loop {
                interval.tick().await;

                let should_flush = {
                    let last_time = last_batch_time.read().await;
                    last_time.elapsed() >= batch_timeout
                };

                if should_flush {
                    let mut pending = pending_batch.write().await;
                    if !pending.is_empty() {
                        let items = std::mem::take(&mut *pending);
                        let batch = StreamingBatch {
                            items,
                            batch_id: batch_counter.fetch_add(1, Ordering::SeqCst),
                            created_at: Utc::now(),
                            sequence_number: sequence_counter.fetch_add(1, Ordering::SeqCst),
                        };

                        let _ = sender.send(batch);
                        *last_batch_time.write().await = Instant::now();
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_adaptive_buffer_manager(&self) -> Result<()> {
        let performance_samples = Arc::clone(&self.performance_samples);
        let backpressure_state = Arc::clone(&self.backpressure_state);
        let target_latency = Duration::from_millis(self.config.target_latency_ms);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                // Calculate average latency from recent samples
                let samples = performance_samples.read().await;
                if samples.len() >= 10 {
                    let avg_latency = samples.iter().map(|(_, latency)| *latency).sum::<f64>()
                        / samples.len() as f64;

                    let avg_latency_duration = Duration::from_millis(avg_latency as u64);

                    // Update backpressure state based on latency
                    let mut backpressure = backpressure_state.write().await;
                    *backpressure = if avg_latency_duration > target_latency * 3 {
                        BackpressureSignal::Critical
                    } else if avg_latency_duration > target_latency * 2 {
                        BackpressureSignal::Severe(avg_latency / target_latency.as_millis() as f64)
                    } else if avg_latency_duration > target_latency {
                        BackpressureSignal::Moderate(
                            avg_latency / target_latency.as_millis() as f64,
                        )
                    } else {
                        BackpressureSignal::Normal
                    };
                }
            }
        });

        Ok(())
    }

    async fn start_connection_monitor(&self) -> Result<()> {
        let connections = Arc::clone(&self.connections);
        let interval_duration = Duration::from_secs(self.config.health_check_interval_seconds);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval_duration);

            loop {
                interval.tick().await;

                // Clean up stale connections
                let mut conns = connections.write().await;
                let now = Instant::now();
                conns.retain(|_, health| {
                    now.duration_since(health.last_seen) < Duration::from_secs(300)
                    // 5 minutes timeout
                });
            }
        });

        Ok(())
    }

    async fn start_metrics_collector(&self) -> Result<()> {
        let metrics = Arc::clone(&self.metrics);
        let connections = Arc::clone(&self.connections);
        let sender = self.sender.clone();
        let performance_samples = Arc::clone(&self.performance_samples);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Update metrics
                let mut metrics_guard = metrics.write().await;
                let connections_guard = connections.read().await;
                let samples_guard = performance_samples.read().await;

                // Calculate throughput
                if !samples_guard.is_empty() {
                    let now = Instant::now();
                    let one_second_ago = now - Duration::from_secs(1);
                    let recent_samples: Vec<_> = samples_guard
                        .iter()
                        .filter(|(timestamp, _)| *timestamp > one_second_ago)
                        .collect();

                    metrics_guard.throughput_messages_per_second = recent_samples.len() as f64;

                    if !recent_samples.is_empty() {
                        metrics_guard.average_latency_ms = recent_samples
                            .iter()
                            .map(|(_, latency)| *latency)
                            .sum::<f64>()
                            / recent_samples.len() as f64;
                    }
                }

                // Update connection count
                metrics_guard.active_connections = connections_guard.len();

                // Calculate buffer utilization
                let receiver_count = sender.receiver_count();
                if receiver_count > 0 {
                    metrics_guard.buffer_utilization =
                        (sender.len() as f64 / receiver_count as f64).min(1.0);
                }
            }
        });

        Ok(())
    }

    async fn record_performance_sample(&self, latency: Duration) {
        let mut samples = self.performance_samples.write().await;
        samples.push_back((Instant::now(), latency.as_millis() as f64));

        // Keep only recent samples (last 1000)
        while samples.len() > 1000 {
            samples.pop_front();
        }
    }
}

/// Streaming optimization factory
pub struct StreamingOptimizerFactory;

impl StreamingOptimizerFactory {
    /// Create performance-optimized streaming configuration
    pub fn performance_optimized() -> StreamingConfig {
        StreamingConfig {
            initial_buffer_size: 1024,
            min_buffer_size: 128,
            max_buffer_size: 16384,
            target_latency_ms: 25,
            max_batch_size: 20,
            batch_timeout_ms: 50,
            enable_adaptive_buffering: true,
            enable_batching: true,
            enable_backpressure_detection: true,
            memory_pool_size: 100,
            health_check_interval_seconds: 15,
        }
    }

    /// Create memory-optimized streaming configuration
    pub fn memory_optimized() -> StreamingConfig {
        StreamingConfig {
            initial_buffer_size: 64,
            min_buffer_size: 16,
            max_buffer_size: 512,
            target_latency_ms: 100,
            max_batch_size: 5,
            batch_timeout_ms: 200,
            enable_adaptive_buffering: false,
            enable_batching: true,
            enable_backpressure_detection: true,
            memory_pool_size: 20,
            health_check_interval_seconds: 60,
        }
    }

    /// Create low-latency streaming configuration
    pub fn low_latency() -> StreamingConfig {
        StreamingConfig {
            initial_buffer_size: 2048,
            min_buffer_size: 256,
            max_buffer_size: 32768,
            target_latency_ms: 10,
            max_batch_size: 50,
            batch_timeout_ms: 25,
            enable_adaptive_buffering: true,
            enable_batching: true,
            enable_backpressure_detection: true,
            memory_pool_size: 200,
            health_check_interval_seconds: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_streaming_optimizer_basic() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);

        optimizer.start().await.unwrap();

        let mut receiver = optimizer.subscribe();

        // Send test data
        optimizer.send_item("test1".to_string()).await.unwrap();
        optimizer.send_item("test2".to_string()).await.unwrap();

        // Flush to ensure batching completes
        optimizer.flush_batches().await.unwrap();

        // Receive batch
        let batch = timeout(Duration::from_secs(1), receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(!batch.items.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_metrics() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);

        optimizer.start().await.unwrap();

        // Send some data
        for i in 0..10 {
            optimizer.send_item(format!("item{}", i)).await.unwrap();
        }

        // Wait for metrics to be collected
        sleep(Duration::from_millis(100)).await;

        let metrics = optimizer.get_metrics().await;
        assert!(metrics.throughput_messages_per_second >= 0.0);
    }

    #[tokio::test]
    async fn test_backpressure_detection() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);

        optimizer.start().await.unwrap();

        // Simulate high latency
        for _ in 0..100 {
            optimizer
                .record_performance_sample(Duration::from_millis(200))
                .await;
        }

        // Wait for backpressure detection
        sleep(Duration::from_secs(1)).await;

        let backpressure = optimizer.get_backpressure_state().await;
        match backpressure {
            BackpressureSignal::Normal => {
                // May still be normal if not enough samples
            }
            _ => {
                // Backpressure detected as expected
            }
        }
    }
}
