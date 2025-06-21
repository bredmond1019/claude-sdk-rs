//! Advanced performance optimizations for analytics processing
//!
//! This module provides high-performance data structures and algorithms
//! for efficient analytics data processing at scale.

use super::*;
use crate::cli::analytics::AnalyticsSummary;
use crate::cli::cost::CostEntry;
use crate::cli::error::Result;
use crate::cli::history::HistoryEntry;
use chrono::{DateTime, Duration, Utc};
use futures::{Stream, StreamExt};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time::Instant;

/// Bounded cache with size and memory limits
pub struct BoundedCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    cache: Arc<RwLock<HashMap<K, (V, Instant)>>>,
    max_entries: usize,
    max_memory_mb: usize,
    size_estimator: Box<dyn Fn(&V) -> usize + Send + Sync>,
    current_size: Arc<RwLock<usize>>,
}

impl<K, V> BoundedCache<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new<F>(max_entries: usize, max_memory_mb: usize, size_estimator: F) -> Self
    where
        F: Fn(&V) -> usize + Send + Sync + 'static,
    {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
            max_memory_mb,
            size_estimator: Box::new(size_estimator),
            current_size: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn get(&self, key: &K) -> Option<V> {
        let cache = self.cache.read().await;
        cache.get(key).map(|(v, _)| v.clone())
    }

    pub async fn put(&self, key: K, value: V) -> Result<()> {
        let value_size = (self.size_estimator)(&value);
        let mut cache = self.cache.write().await;
        let mut current_size = self.current_size.write().await;

        // Check if we need to evict entries
        while cache.len() >= self.max_entries
            || *current_size + value_size > self.max_memory_mb * 1024 * 1024
        {
            // Evict oldest entry (simple LRU)
            if let Some((oldest_key, _)) = cache
                .iter()
                .min_by_key(|(_, (_, time))| time)
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                if let Some((evicted_value, _)) = cache.remove(&oldest_key) {
                    *current_size -= (self.size_estimator)(&evicted_value);
                }
            } else {
                break;
            }
        }

        cache.insert(key, (value, Instant::now()));
        *current_size += value_size;

        Ok(())
    }

    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let mut current_size = self.current_size.write().await;
        cache.clear();
        *current_size = 0;
    }
}

/// Batch processor with time and size-based batching
pub struct BatchProcessor<T> {
    batch_size: usize,
    batch_timeout: StdDuration,
    max_concurrent: usize,
    processor: Box<dyn Fn(Vec<T>) -> futures::future::BoxFuture<'static, Result<()>> + Send + Sync>,
    buffer: Arc<Mutex<Vec<T>>>,
    semaphore: Arc<Semaphore>,
}

impl<T> BatchProcessor<T>
where
    T: Send + 'static,
{
    pub fn new<F, Fut>(
        batch_size: usize,
        batch_timeout: StdDuration,
        max_concurrent: usize,
        processor: F,
    ) -> Self
    where
        F: Fn(Vec<T>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        Self {
            batch_size,
            batch_timeout,
            max_concurrent,
            processor: Box::new(move |batch| Box::pin(processor(batch))),
            buffer: Arc::new(Mutex::new(Vec::new())),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    pub async fn add(&self, item: T) -> Result<()> {
        let mut buffer = self.buffer.lock().await;
        buffer.push(item);

        if buffer.len() >= self.batch_size {
            let batch: Vec<T> = buffer.drain(..).collect();
            drop(buffer);
            self.process_batch(batch).await?;
        }

        Ok(())
    }

    pub async fn flush(&self) -> Result<()> {
        let mut buffer = self.buffer.lock().await;
        if !buffer.is_empty() {
            let batch: Vec<T> = buffer.drain(..).collect();
            drop(buffer);
            self.process_batch(batch).await?;
        }
        Ok(())
    }

    async fn process_batch(&self, batch: Vec<T>) -> Result<()> {
        let permit = self.semaphore.clone().acquire_owned().await.map_err(|_| {
            crate::error::InteractiveError::Execution("Failed to acquire semaphore".to_string())
        })?;
        let processor = self.processor.as_ref();
        let future = processor(batch);

        tokio::spawn(async move {
            let _ = future.await;
            drop(permit);
        });

        Ok(())
    }
}

/// Optimized data generator with streaming support
pub struct OptimizedDataGenerator {
    batch_size: usize,
}

impl OptimizedDataGenerator {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    pub fn generate_stream(&self, total_items: usize) -> impl Stream<Item = Vec<CostEntry>> {
        let batch_size = self.batch_size;
        let num_batches = (total_items + batch_size - 1) / batch_size;

        futures::stream::iter(0..num_batches).map(move |batch_idx| {
            let start_idx = batch_idx * batch_size;
            let end_idx = ((batch_idx + 1) * batch_size).min(total_items);

            (start_idx..end_idx)
                .map(|i| {
                    let mut entry = CostEntry::new(
                        uuid::Uuid::from_u128(i as u128),
                        format!("cmd_{}", i),
                        0.01 + (i as f64 * 0.0001),
                        100 + (i % 1000) as u32,
                        200 + (i % 2000) as u32,
                        1000 + (i % 5000) as u64,
                        "claude-3-opus".to_string(),
                    );
                    entry.timestamp = Utc::now() - Duration::days(30) + Duration::minutes(i as i64);
                    entry
                })
                .collect()
        })
    }
}

/// High-throughput processor for streaming data
pub struct HighThroughputProcessor {
    worker_count: usize,
    buffer_size: usize,
    processed: Arc<RwLock<u64>>,
    errors: Arc<RwLock<u64>>,
    start_time: Instant,
}

impl HighThroughputProcessor {
    pub fn new(worker_count: usize, buffer_size: usize) -> Self {
        Self {
            worker_count,
            buffer_size,
            processed: Arc::new(RwLock::new(0)),
            errors: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
        }
    }

    pub async fn process_stream<S, T, F, Fut>(&self, mut stream: S, processor: F) -> Result<()>
    where
        S: Stream<Item = T> + Unpin + Send,
        T: Send + 'static,
        F: Fn(Vec<T>) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        let (tx, rx) = tokio::sync::mpsc::channel::<Vec<T>>(self.buffer_size);
        let rx = Arc::new(Mutex::new(rx));

        // Spawn workers
        let mut handles = Vec::new();
        for _ in 0..self.worker_count {
            let processor = processor.clone();
            let processed = Arc::clone(&self.processed);
            let errors = Arc::clone(&self.errors);
            let worker_rx = Arc::clone(&rx);

            let handle = tokio::spawn(async move {
                loop {
                    let batch = {
                        let mut rx = worker_rx.lock().await;
                        rx.recv().await
                    };

                    if let Some(batch) = batch {
                        let batch_size = batch.len();
                        match processor(batch).await {
                            Ok(_) => {
                                let mut p = processed.write().await;
                                *p += batch_size as u64;
                            }
                            Err(_) => {
                                let mut e = errors.write().await;
                                *e += batch_size as u64;
                            }
                        }
                    } else {
                        break;
                    }
                }
            });
            handles.push(handle);
        }

        // Feed stream to workers
        let mut batch = Vec::with_capacity(self.buffer_size);
        while let Some(item) = stream.next().await {
            batch.push(item);
            if batch.len() >= self.buffer_size {
                let _ = tx
                    .send(std::mem::replace(
                        &mut batch,
                        Vec::with_capacity(self.buffer_size),
                    ))
                    .await;
            }
        }

        // Send remaining batch
        if !batch.is_empty() {
            let _ = tx.send(batch).await;
        }

        // Close channel and wait for workers
        drop(tx);
        for handle in handles {
            handle.await?;
        }

        Ok(())
    }

    pub fn get_metrics(&self) -> (u64, u64, f64) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let processed = self.processed.blocking_read();
        let errors = self.errors.blocking_read();
        let throughput = *processed as f64 / elapsed;

        (*processed, *errors, throughput)
    }
}

/// Object pool for reducing allocations
pub struct ObjectPool<T> {
    pool: Arc<Mutex<Vec<T>>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
}

impl<T> ObjectPool<T>
where
    T: Send + 'static,
{
    pub fn new<F>(max_size: usize, factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Arc::new(Mutex::new(Vec::new())),
            factory: Box::new(factory),
            max_size,
        }
    }

    pub async fn acquire(&self) -> PooledObject<T> {
        let mut pool = self.pool.lock().await;
        let object = pool.pop().unwrap_or_else(|| (self.factory)());

        PooledObject {
            object: Some(object),
            pool: Arc::clone(&self.pool),
            max_size: self.max_size,
        }
    }
}

pub struct PooledObject<T>
where
    T: Send + 'static,
{
    object: Option<T>,
    pool: Arc<Mutex<Vec<T>>>,
    max_size: usize,
}

impl<T> std::ops::Deref for PooledObject<T>
where
    T: Send + 'static,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.object.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for PooledObject<T>
where
    T: Send + 'static,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.object.as_mut().unwrap()
    }
}

impl<T> Drop for PooledObject<T>
where
    T: Send + 'static,
{
    fn drop(&mut self) {
        if let Some(object) = self.object.take() {
            let pool = self.pool.clone();
            let max_size = self.max_size;

            tokio::spawn(async move {
                let mut pool = pool.lock().await;
                if pool.len() < max_size {
                    pool.push(object);
                }
            });
        }
    }
}

/// Memory-efficient aggregator with streaming support
pub struct MemoryEfficientAggregator {
    max_memory_mb: usize,
    aggregated_data: Arc<RwLock<AggregatedData>>,
}

#[derive(Default)]
struct AggregatedData {
    total_cost: f64,
    total_commands: u64,
    command_costs: HashMap<String, f64>,
}

impl MemoryEfficientAggregator {
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            max_memory_mb,
            aggregated_data: Arc::new(RwLock::new(AggregatedData::default())),
        }
    }

    pub async fn process_entry(&self, entry: &CostEntry) -> Result<()> {
        let mut data = self.aggregated_data.write().await;

        data.total_cost += entry.cost_usd;
        data.total_commands += 1;

        *data
            .command_costs
            .entry(entry.command_name.clone())
            .or_insert(0.0) += entry.cost_usd;

        // Simple memory check - trim if too many unique commands
        if data.command_costs.len() > 10000 {
            // Keep only top 1000 commands by cost
            let mut costs: Vec<_> = data
                .command_costs
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            costs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            costs.truncate(1000);
            data.command_costs = costs.into_iter().collect();
        }

        Ok(())
    }

    pub async fn get_summary(&self) -> (f64, u64, HashMap<String, f64>) {
        let data = self.aggregated_data.read().await;
        (
            data.total_cost,
            data.total_commands,
            data.command_costs.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bounded_cache() {
        let cache = BoundedCache::new(2, 1, |s: &String| s.len());

        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        cache
            .put("key2".to_string(), "value2".to_string())
            .await
            .unwrap();

        assert!(cache.get(&"key1".to_string()).await.is_some());
        assert!(cache.get(&"key2".to_string()).await.is_some());

        // This should evict the oldest entry
        cache
            .put("key3".to_string(), "value3".to_string())
            .await
            .unwrap();

        assert!(cache.get(&"key3".to_string()).await.is_some());
    }

    #[tokio::test]
    async fn test_object_pool() {
        let pool: ObjectPool<Vec<u8>> = ObjectPool::new(5, || vec![0u8; 1024]);

        let mut objects = Vec::new();
        for _ in 0..3 {
            objects.push(pool.acquire().await);
        }

        // Objects should be unique
        for obj in &mut objects {
            obj[0] = 1;
        }

        drop(objects);

        // Wait a bit for objects to return to pool
        tokio::time::sleep(StdDuration::from_millis(100)).await;

        // Acquire again - should get recycled objects
        let recycled = pool.acquire().await;
        assert_eq!(recycled[0], 1); // Should have our modification
    }
}
