//! Performance fixes and optimizations for analytics processing
//!
//! This module provides performance-critical implementations and fixes
//! for handling large-scale analytics data processing.

use super::*;
use crate::cli::cost::CostEntry;
use crate::cli::error::Result;
use crate::cli::history::HistoryEntry;
use chrono::{DateTime, Duration, Utc};
use futures::future::BoxFuture;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};

/// Simple batch processor for handling analytics updates efficiently
pub struct SimpleBatchProcessor {
    batch_size: usize,
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
}

impl SimpleBatchProcessor {
    pub fn new(batch_size: usize, max_concurrent: usize) -> Self {
        Self {
            batch_size,
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Process updates in batches with concurrency control
    pub async fn process_updates<F, Fut>(
        &self,
        updates: Vec<super::realtime_performance_tests::AnalyticsUpdate>,
        processor: F,
    ) -> Result<()>
    where
        F: Fn(Vec<super::realtime_performance_tests::AnalyticsUpdate>) -> Fut,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let chunks: Vec<_> = updates
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut handles = Vec::new();

        for batch in chunks {
            let permit = self.semaphore.clone().acquire_owned().await.map_err(|_| {
                crate::error::InteractiveError::Execution("Failed to acquire semaphore".to_string())
            })?;
            let batch_future = processor(batch);

            let handle = tokio::spawn(async move {
                let result = batch_future.await;
                drop(permit);
                result
            });

            handles.push(handle);
        }

        // Wait for all batches to complete
        for handle in handles {
            handle.await??;
        }

        Ok(())
    }
}

/// Generate large dataset with optimized parallel processing
pub async fn generate_large_dataset_optimized(size: usize) -> (Vec<CostEntry>, Vec<HistoryEntry>) {
    let chunk_size = 1000;
    let num_chunks = (size + chunk_size - 1) / chunk_size;

    let mut all_cost_entries = Vec::with_capacity(size);
    let mut all_history_entries = Vec::with_capacity(size);

    // Generate chunks in parallel
    let mut handles = Vec::new();

    for chunk_idx in 0..num_chunks {
        let start_idx = chunk_idx * chunk_size;
        let end_idx = ((chunk_idx + 1) * chunk_size).min(size);
        let chunk_len = end_idx - start_idx;

        let handle = tokio::spawn(async move {
            let mut cost_entries = Vec::with_capacity(chunk_len);
            let mut history_entries = Vec::with_capacity(chunk_len);

            for i in start_idx..end_idx {
                let session_id = uuid::Uuid::from_u128(i as u128);
                let timestamp = Utc::now() - Duration::days(30) + Duration::minutes(i as i64);

                // Create cost entry
                let mut cost_entry = CostEntry::new(
                    session_id,
                    format!("command_{}", i),
                    0.01 + (i as f64 * 0.0001),
                    100 + (i % 1000) as u32,
                    200 + (i % 2000) as u32,
                    1000 + (i % 5000) as u64,
                    match i % 3 {
                        0 => "claude-3-opus",
                        1 => "claude-3-sonnet",
                        _ => "claude-3-haiku",
                    }
                    .to_string(),
                );
                cost_entry.timestamp = timestamp;
                cost_entries.push(cost_entry);

                // Create history entry
                let mut history_entry = HistoryEntry::new(
                    session_id,
                    format!("command_{}", i),
                    vec![format!("--arg{}", i % 5)],
                    format!("Output for command {}", i),
                    i % 20 != 0, // 95% success rate
                    1000 + (i % 5000) as u64,
                );
                history_entry.timestamp = timestamp;
                history_entries.push(history_entry);
            }

            (cost_entries, history_entries)
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        let (cost_chunk, history_chunk) = handle.await.unwrap();
        all_cost_entries.extend(cost_chunk);
        all_history_entries.extend(history_chunk);
    }

    (all_cost_entries, all_history_entries)
}

/// Memory-efficient data aggregator
pub struct DataAggregator {
    window_size: usize,
    buffer: Arc<Mutex<Vec<f64>>>,
}

impl DataAggregator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: Arc::new(Mutex::new(Vec::with_capacity(window_size))),
        }
    }

    pub async fn add_value(&self, value: f64) {
        let mut buffer = self.buffer.lock().await;
        if buffer.len() >= self.window_size {
            buffer.remove(0);
        }
        buffer.push(value);
    }

    pub async fn get_average(&self) -> f64 {
        let buffer = self.buffer.lock().await;
        if buffer.is_empty() {
            0.0
        } else {
            buffer.iter().sum::<f64>() / buffer.len() as f64
        }
    }

    pub async fn get_max(&self) -> f64 {
        let buffer = self.buffer.lock().await;
        buffer.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub async fn get_min(&self) -> f64 {
        let buffer = self.buffer.lock().await;
        buffer.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}

/// Optimized time-based indexing for faster queries
pub struct TimeIndex {
    index: Arc<Mutex<std::collections::BTreeMap<DateTime<Utc>, Vec<usize>>>>,
}

impl TimeIndex {
    pub fn new() -> Self {
        Self {
            index: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        }
    }

    pub async fn add_entry(&self, timestamp: DateTime<Utc>, index: usize) {
        let mut idx = self.index.lock().await;
        idx.entry(
            timestamp
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc(),
        )
        .or_insert_with(Vec::new)
        .push(index);
    }

    pub async fn get_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<usize> {
        let idx = self.index.lock().await;
        let mut indices = Vec::new();

        for (date, date_indices) in idx.range(start..=end) {
            indices.extend_from_slice(date_indices);
        }

        indices
    }
}

/// Fast incremental statistics calculator
pub struct IncrementalStats {
    count: u64,
    sum: f64,
    sum_of_squares: f64,
    min: f64,
    max: f64,
}

impl IncrementalStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_of_squares: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn add_value(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_of_squares += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_of_squares / self.count as f64) - (mean * mean)
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_batch_processor() {
        let processor = SimpleBatchProcessor::new(10, 4);
        let updates = vec![
            super::super::realtime_performance_tests::AnalyticsUpdate {
                timestamp: Utc::now(),
                update_type: super::super::realtime_performance_tests::UpdateType::CommandCompleted,
                session_id: uuid::Uuid::new_v4(),
                metric_deltas: super::super::realtime_performance_tests::MetricDeltas::default(),
            };
            25
        ];

        let processed_count = Arc::new(Mutex::new(0));
        let count_clone = Arc::clone(&processed_count);

        processor
            .process_updates(updates, move |batch| {
                let count = Arc::clone(&count_clone);
                Box::pin(async move {
                    let mut counter = count.lock().await;
                    *counter += batch.len();
                    Ok(())
                })
            })
            .await
            .unwrap();

        let final_count = *processed_count.lock().await;
        assert_eq!(final_count, 25);
    }

    #[tokio::test]
    async fn test_incremental_stats() {
        let mut stats = IncrementalStats::new();

        for i in 1..=100 {
            stats.add_value(i as f64);
        }

        assert_eq!(stats.count, 100);
        assert_eq!(stats.mean(), 50.5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 100.0);
    }
}
