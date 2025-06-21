//! Dashboard caching with TTL-based invalidation
//!
//! This module provides a sophisticated caching layer for dashboard data,
//! analytics results, and time series data with intelligent invalidation
//! strategies and memory management.
//!
//! # Features
//!
//! - **TTL-based Expiration**: Configurable time-to-live for cached data
//! - **Smart Invalidation**: Invalidate related caches when underlying data changes
//! - **Memory Management**: LRU eviction and memory usage limits
//! - **Cache Warming**: Proactive cache population for frequently accessed data
//! - **Performance Metrics**: Detailed cache hit/miss statistics
//! - **Multi-level Caching**: Different cache strategies for different data types
//!
//! # Example
//!
//! ```no_run
//! use crate_interactive::analytics::dashboard_cache::{DashboardCache, CacheConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = CacheConfig {
//!     default_ttl_seconds: 300,
//!     max_memory_mb: 100,
//!     enable_smart_invalidation: true,
//!     ..Default::default()
//! };
//!
//! let cache = DashboardCache::new(config);
//!
//! // Cache dashboard data
//! cache.set_dashboard_data("key", dashboard_data, None).await;
//!
//! // Retrieve cached data
//! if let Some(data) = cache.get_dashboard_data("key").await {
//!     println!("Cache hit!");
//! }
//! # Ok(())
//! # }
//! ```

use super::{
    AnalyticsSummary, DashboardData, LiveDashboardData, OptimizedTimeSeriesData,
    PerformanceMetrics, TimeSeriesData,
};
use crate::{cli::error::Result, cli::session::SessionId};
use chrono::{DateTime, Duration, Utc};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Cache configuration options
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Default TTL for cached items in seconds
    pub default_ttl_seconds: u64,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Maximum number of cached items
    pub max_items: usize,
    /// Enable smart invalidation based on data dependencies
    pub enable_smart_invalidation: bool,
    /// Enable cache warming for frequently accessed data
    pub enable_cache_warming: bool,
    /// Background cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
    /// Cache hit ratio threshold for warnings
    pub min_hit_ratio_threshold: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            default_ttl_seconds: 300, // 5 minutes
            max_memory_mb: 200,       // 200 MB
            max_items: 1000,
            enable_smart_invalidation: true,
            enable_cache_warming: true,
            cleanup_interval_seconds: 60, // 1 minute
            min_hit_ratio_threshold: 0.8, // 80%
        }
    }
}

/// Different types of cacheable data
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheDataType {
    DashboardData,
    LiveDashboardData,
    AnalyticsSummary,
    TimeSeriesData,
    PerformanceMetrics,
    SessionReport,
}

/// Cache key with metadata
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    pub data_type: CacheDataType,
    pub identifier: String,
    pub parameters: HashMap<String, String>,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data_type.hash(state);
        self.identifier.hash(state);

        // Hash parameters in a deterministic order
        let mut params: Vec<_> = self.parameters.iter().collect();
        params.sort_by_key(|(k, _)| *k);
        for (k, v) in params {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(data_type: CacheDataType, identifier: String) -> Self {
        Self {
            data_type,
            identifier,
            parameters: HashMap::new(),
        }
    }

    /// Add a parameter to the cache key
    pub fn with_param(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Create key for dashboard data
    pub fn dashboard_data(session_id: Option<SessionId>) -> Self {
        let identifier = match session_id {
            Some(id) => format!("session:{}", id),
            None => "global".to_string(),
        };
        Self::new(CacheDataType::DashboardData, identifier)
    }

    /// Create key for live dashboard data
    pub fn live_dashboard_data() -> Self {
        Self::new(CacheDataType::LiveDashboardData, "current".to_string())
    }

    /// Create key for analytics summary
    pub fn analytics_summary(period_days: u32) -> Self {
        Self::new(CacheDataType::AnalyticsSummary, "summary".to_string())
            .with_param("period_days".to_string(), period_days.to_string())
    }

    /// Create key for time series data
    pub fn time_series_data(start: DateTime<Utc>, end: DateTime<Utc>, metric_type: &str) -> Self {
        Self::new(CacheDataType::TimeSeriesData, metric_type.to_string())
            .with_param("start".to_string(), start.timestamp().to_string())
            .with_param("end".to_string(), end.timestamp().to_string())
    }
}

/// Cached item with metadata
#[derive(Debug)]
pub struct CachedItem<T> {
    pub data: T,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub access_count: AtomicU64,
    pub last_accessed: AtomicU64, // Unix timestamp
    pub estimated_size_bytes: usize,
}

impl<T: Clone> Clone for CachedItem<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            created_at: self.created_at,
            expires_at: self.expires_at,
            access_count: AtomicU64::new(
                self.access_count.load(std::sync::atomic::Ordering::Relaxed),
            ),
            last_accessed: AtomicU64::new(
                self.last_accessed
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            estimated_size_bytes: self.estimated_size_bytes,
        }
    }
}

impl<T> CachedItem<T> {
    /// Create a new cached item
    pub fn new(data: T, ttl_seconds: u64, estimated_size: usize) -> Self {
        let now = Utc::now();
        Self {
            data,
            created_at: now,
            expires_at: now + Duration::seconds(ttl_seconds as i64),
            access_count: AtomicU64::new(0),
            last_accessed: AtomicU64::new(now.timestamp() as u64),
            estimated_size_bytes: estimated_size,
        }
    }

    /// Check if the item has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Mark the item as accessed
    pub fn mark_accessed(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_accessed
            .store(Utc::now().timestamp() as u64, Ordering::Relaxed);
    }

    /// Get access count
    pub fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Get last accessed timestamp
    pub fn get_last_accessed(&self) -> DateTime<Utc> {
        let timestamp = self.last_accessed.load(Ordering::Relaxed);
        DateTime::from_timestamp(timestamp as i64, 0).unwrap_or(self.created_at)
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_ratio: f64,
    pub total_items: usize,
    pub memory_usage_mb: f64,
    pub memory_usage_percentage: f64,
    pub evictions: u64,
    pub expired_items_cleaned: u64,
    pub average_item_size_bytes: f64,
    pub most_accessed_keys: Vec<(String, u64)>,
}

/// Dashboard cache with TTL-based invalidation
pub struct DashboardCache {
    config: CacheConfig,

    // Cache storage
    dashboard_data_cache: Arc<RwLock<LruCache<CacheKey, CachedItem<DashboardData>>>>,
    live_dashboard_cache: Arc<RwLock<LruCache<CacheKey, CachedItem<LiveDashboardData>>>>,
    analytics_summary_cache: Arc<RwLock<LruCache<CacheKey, CachedItem<AnalyticsSummary>>>>,
    time_series_cache: Arc<RwLock<LruCache<CacheKey, CachedItem<OptimizedTimeSeriesData>>>>,

    // Cache statistics
    total_requests: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    evictions: AtomicU64,
    expired_cleaned: AtomicU64,
    current_memory_bytes: AtomicUsize,

    // Invalidation tracking
    data_dependencies: Arc<RwLock<HashMap<String, Vec<CacheKey>>>>,
}

impl DashboardCache {
    /// Create a new dashboard cache
    pub fn new(config: CacheConfig) -> Self {
        let max_items =
            NonZeroUsize::new(config.max_items).unwrap_or(NonZeroUsize::new(1000).unwrap());

        Self {
            config,
            dashboard_data_cache: Arc::new(RwLock::new(LruCache::new(max_items))),
            live_dashboard_cache: Arc::new(RwLock::new(LruCache::new(max_items))),
            analytics_summary_cache: Arc::new(RwLock::new(LruCache::new(max_items))),
            time_series_cache: Arc::new(RwLock::new(LruCache::new(max_items))),
            total_requests: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expired_cleaned: AtomicU64::new(0),
            current_memory_bytes: AtomicUsize::new(0),
            data_dependencies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start background cleanup task
    pub async fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let cache_clone = self.clone_for_cleanup();
        let interval_secs = self.config.cleanup_interval_seconds;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;
                cache_clone.cleanup_expired_items().await;
            }
        })
    }

    /// Get dashboard data from cache
    pub async fn get_dashboard_data(&self, key: &CacheKey) -> Option<DashboardData> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut cache = self.dashboard_data_cache.write().await;
        if let Some(item) = cache.get(key) {
            if !item.is_expired() {
                item.mark_accessed();
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(item.data.clone());
            } else {
                // Remove expired item
                cache.pop(key);
                self.expired_cleaned.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Set dashboard data in cache
    pub async fn set_dashboard_data(
        &self,
        key: CacheKey,
        data: DashboardData,
        ttl_seconds: Option<u64>,
    ) {
        let ttl = ttl_seconds.unwrap_or(self.config.default_ttl_seconds);
        let estimated_size = self.estimate_dashboard_data_size(&data);

        let item = CachedItem::new(data, ttl, estimated_size);

        let mut cache = self.dashboard_data_cache.write().await;

        // Check memory limits before adding
        if self.would_exceed_memory_limit(estimated_size).await {
            self.evict_lru_items(estimated_size).await;
        }

        if let Some(old_item) = cache.put(key.clone(), item) {
            self.current_memory_bytes
                .fetch_sub(old_item.estimated_size_bytes, Ordering::Relaxed);
        }

        self.current_memory_bytes
            .fetch_add(estimated_size, Ordering::Relaxed);

        // Register data dependencies for smart invalidation
        if self.config.enable_smart_invalidation {
            self.register_data_dependency(&key).await;
        }
    }

    /// Get live dashboard data from cache
    pub async fn get_live_dashboard_data(&self, key: &CacheKey) -> Option<LiveDashboardData> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut cache = self.live_dashboard_cache.write().await;
        if let Some(item) = cache.get(key) {
            if !item.is_expired() {
                item.mark_accessed();
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(item.data.clone());
            } else {
                cache.pop(key);
                self.expired_cleaned.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Set live dashboard data in cache
    pub async fn set_live_dashboard_data(
        &self,
        key: CacheKey,
        data: LiveDashboardData,
        ttl_seconds: Option<u64>,
    ) {
        let ttl = ttl_seconds.unwrap_or(30); // Shorter TTL for live data
        let estimated_size = self.estimate_live_dashboard_data_size(&data);

        let item = CachedItem::new(data, ttl, estimated_size);

        let mut cache = self.live_dashboard_cache.write().await;

        if self.would_exceed_memory_limit(estimated_size).await {
            self.evict_lru_items(estimated_size).await;
        }

        if let Some(old_item) = cache.put(key, item) {
            self.current_memory_bytes
                .fetch_sub(old_item.estimated_size_bytes, Ordering::Relaxed);
        }

        self.current_memory_bytes
            .fetch_add(estimated_size, Ordering::Relaxed);
    }

    /// Get analytics summary from cache
    pub async fn get_analytics_summary(&self, key: &CacheKey) -> Option<AnalyticsSummary> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut cache = self.analytics_summary_cache.write().await;
        if let Some(item) = cache.get(key) {
            if !item.is_expired() {
                item.mark_accessed();
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(item.data.clone());
            } else {
                cache.pop(key);
                self.expired_cleaned.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Set analytics summary in cache
    pub async fn set_analytics_summary(
        &self,
        key: CacheKey,
        data: AnalyticsSummary,
        ttl_seconds: Option<u64>,
    ) {
        let ttl = ttl_seconds.unwrap_or(600); // Longer TTL for summary data
        let estimated_size = self.estimate_analytics_summary_size(&data);

        let item = CachedItem::new(data, ttl, estimated_size);

        let mut cache = self.analytics_summary_cache.write().await;

        if self.would_exceed_memory_limit(estimated_size).await {
            self.evict_lru_items(estimated_size).await;
        }

        if let Some(old_item) = cache.put(key, item) {
            self.current_memory_bytes
                .fetch_sub(old_item.estimated_size_bytes, Ordering::Relaxed);
        }

        self.current_memory_bytes
            .fetch_add(estimated_size, Ordering::Relaxed);
    }

    /// Get time series data from cache
    pub async fn get_time_series_data(&self, key: &CacheKey) -> Option<OptimizedTimeSeriesData> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut cache = self.time_series_cache.write().await;
        if let Some(item) = cache.get(key) {
            if !item.is_expired() {
                item.mark_accessed();
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(item.data.clone());
            } else {
                cache.pop(key);
                self.expired_cleaned.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Set time series data in cache
    pub async fn set_time_series_data(
        &self,
        key: CacheKey,
        data: OptimizedTimeSeriesData,
        ttl_seconds: Option<u64>,
    ) {
        let ttl = ttl_seconds.unwrap_or(300); // 5 minutes for time series
        let estimated_size = self.estimate_time_series_data_size(&data);

        let item = CachedItem::new(data, ttl, estimated_size);

        let mut cache = self.time_series_cache.write().await;

        if self.would_exceed_memory_limit(estimated_size).await {
            self.evict_lru_items(estimated_size).await;
        }

        if let Some(old_item) = cache.put(key, item) {
            self.current_memory_bytes
                .fetch_sub(old_item.estimated_size_bytes, Ordering::Relaxed);
        }

        self.current_memory_bytes
            .fetch_add(estimated_size, Ordering::Relaxed);
    }

    /// Invalidate caches based on data changes
    pub async fn invalidate_related_caches(&self, data_source: &str) {
        if !self.config.enable_smart_invalidation {
            return;
        }

        let dependencies = self.data_dependencies.read().await;
        if let Some(keys_to_invalidate) = dependencies.get(data_source) {
            for key in keys_to_invalidate {
                match key.data_type {
                    CacheDataType::DashboardData => {
                        self.dashboard_data_cache.write().await.pop(key);
                    }
                    CacheDataType::LiveDashboardData => {
                        self.live_dashboard_cache.write().await.pop(key);
                    }
                    CacheDataType::AnalyticsSummary => {
                        self.analytics_summary_cache.write().await.pop(key);
                    }
                    CacheDataType::TimeSeriesData => {
                        self.time_series_cache.write().await.pop(key);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Clear all caches
    pub async fn clear_all(&self) {
        self.dashboard_data_cache.write().await.clear();
        self.live_dashboard_cache.write().await.clear();
        self.analytics_summary_cache.write().await.clear();
        self.time_series_cache.write().await.clear();
        self.current_memory_bytes.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);

        let hit_ratio = if total_requests > 0 {
            cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let memory_bytes = self.current_memory_bytes.load(Ordering::Relaxed);
        let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0;
        let memory_percentage = memory_mb / self.config.max_memory_mb as f64 * 100.0;

        let total_items = self.get_total_items().await;
        let average_item_size = if total_items > 0 {
            memory_bytes as f64 / total_items as f64
        } else {
            0.0
        };

        let most_accessed_keys = self.get_most_accessed_keys().await;

        CacheStats {
            total_requests,
            cache_hits,
            cache_misses,
            hit_ratio,
            total_items,
            memory_usage_mb: memory_mb,
            memory_usage_percentage: memory_percentage,
            evictions: self.evictions.load(Ordering::Relaxed),
            expired_items_cleaned: self.expired_cleaned.load(Ordering::Relaxed),
            average_item_size_bytes: average_item_size,
            most_accessed_keys,
        }
    }

    // Private helper methods

    async fn cleanup_expired_items(&self) {
        let mut cleaned_count = 0;
        let mut memory_freed = 0;

        // Clean dashboard data cache
        {
            let mut cache = self.dashboard_data_cache.write().await;
            let keys_to_remove: Vec<_> = cache
                .iter()
                .filter(|(_, item)| item.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                if let Some(item) = cache.pop(&key) {
                    memory_freed += item.estimated_size_bytes;
                    cleaned_count += 1;
                }
            }
        }

        // Clean other caches similarly...
        {
            let mut cache = self.live_dashboard_cache.write().await;
            let keys_to_remove: Vec<_> = cache
                .iter()
                .filter(|(_, item)| item.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                if let Some(item) = cache.pop(&key) {
                    memory_freed += item.estimated_size_bytes;
                    cleaned_count += 1;
                }
            }
        }

        {
            let mut cache = self.analytics_summary_cache.write().await;
            let keys_to_remove: Vec<_> = cache
                .iter()
                .filter(|(_, item)| item.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                if let Some(item) = cache.pop(&key) {
                    memory_freed += item.estimated_size_bytes;
                    cleaned_count += 1;
                }
            }
        }

        {
            let mut cache = self.time_series_cache.write().await;
            let keys_to_remove: Vec<_> = cache
                .iter()
                .filter(|(_, item)| item.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                if let Some(item) = cache.pop(&key) {
                    memory_freed += item.estimated_size_bytes;
                    cleaned_count += 1;
                }
            }
        }

        if memory_freed > 0 {
            self.current_memory_bytes
                .fetch_sub(memory_freed, Ordering::Relaxed);
        }

        if cleaned_count > 0 {
            self.expired_cleaned
                .fetch_add(cleaned_count, Ordering::Relaxed);
        }
    }

    async fn would_exceed_memory_limit(&self, additional_bytes: usize) -> bool {
        let current_bytes = self.current_memory_bytes.load(Ordering::Relaxed);
        let max_bytes = self.config.max_memory_mb * 1024 * 1024;
        current_bytes + additional_bytes > max_bytes
    }

    async fn evict_lru_items(&self, needed_bytes: usize) {
        // This is a simplified LRU eviction - in practice, you'd implement
        // more sophisticated strategies based on access patterns

        let mut evicted = 0;
        let mut memory_freed = 0;

        // Evict from dashboard data cache first (typically largest items)
        {
            let mut cache = self.dashboard_data_cache.write().await;
            while memory_freed < needed_bytes && !cache.is_empty() {
                if let Some((_, item)) = cache.pop_lru() {
                    memory_freed += item.estimated_size_bytes;
                    evicted += 1;
                }
            }
        }

        // Continue with other caches if needed
        if memory_freed < needed_bytes {
            let mut cache = self.time_series_cache.write().await;
            while memory_freed < needed_bytes && !cache.is_empty() {
                if let Some((_, item)) = cache.pop_lru() {
                    memory_freed += item.estimated_size_bytes;
                    evicted += 1;
                }
            }
        }

        if memory_freed > 0 {
            self.current_memory_bytes
                .fetch_sub(memory_freed, Ordering::Relaxed);
        }

        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
        }
    }

    async fn register_data_dependency(&self, cache_key: &CacheKey) {
        let mut dependencies = self.data_dependencies.write().await;

        // Determine what data sources this cache depends on
        let data_sources = match cache_key.data_type {
            CacheDataType::DashboardData => {
                vec!["cost_data".to_string(), "history_data".to_string()]
            }
            CacheDataType::LiveDashboardData => vec![
                "cost_data".to_string(),
                "history_data".to_string(),
                "system_metrics".to_string(),
            ],
            CacheDataType::AnalyticsSummary => {
                vec!["cost_data".to_string(), "history_data".to_string()]
            }
            CacheDataType::TimeSeriesData => {
                vec!["cost_data".to_string(), "history_data".to_string()]
            }
            _ => vec![],
        };

        for source in data_sources {
            dependencies
                .entry(source)
                .or_insert_with(Vec::new)
                .push(cache_key.clone());
        }
    }

    async fn get_total_items(&self) -> usize {
        let dashboard_count = self.dashboard_data_cache.read().await.len();
        let live_count = self.live_dashboard_cache.read().await.len();
        let analytics_count = self.analytics_summary_cache.read().await.len();
        let time_series_count = self.time_series_cache.read().await.len();

        dashboard_count + live_count + analytics_count + time_series_count
    }

    async fn get_most_accessed_keys(&self) -> Vec<(String, u64)> {
        let mut access_counts = Vec::new();

        // Collect access counts from all caches
        {
            let cache = self.dashboard_data_cache.read().await;
            for (key, item) in cache.iter() {
                let key_str = format!("{:?}:{}", key.data_type, key.identifier);
                access_counts.push((key_str, item.get_access_count()));
            }
        }

        // Sort by access count and take top 10
        access_counts.sort_by(|a, b| b.1.cmp(&a.1));
        access_counts.truncate(10);
        access_counts
    }

    fn clone_for_cleanup(&self) -> Self {
        Self {
            config: self.config.clone(),
            dashboard_data_cache: Arc::clone(&self.dashboard_data_cache),
            live_dashboard_cache: Arc::clone(&self.live_dashboard_cache),
            analytics_summary_cache: Arc::clone(&self.analytics_summary_cache),
            time_series_cache: Arc::clone(&self.time_series_cache),
            total_requests: AtomicU64::new(0), // Don't clone stats
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expired_cleaned: AtomicU64::new(0),
            current_memory_bytes: AtomicUsize::new(0),
            data_dependencies: Arc::clone(&self.data_dependencies),
        }
    }

    // Size estimation methods (simplified)
    fn estimate_dashboard_data_size(&self, _data: &DashboardData) -> usize {
        // Rough estimation - in practice, you'd implement more accurate sizing
        1024 // 1KB base estimate
    }

    fn estimate_live_dashboard_data_size(&self, _data: &LiveDashboardData) -> usize {
        2048 // 2KB estimate
    }

    fn estimate_analytics_summary_size(&self, data: &AnalyticsSummary) -> usize {
        // Base size + insights + alerts
        1024 + (data.insights.len() * 100) + (data.alerts.len() * 200)
    }

    fn estimate_time_series_data_size(&self, data: &OptimizedTimeSeriesData) -> usize {
        // Estimate based on number of data points
        let total_points = data.cost_over_time.len()
            + data.commands_over_time.len()
            + data.success_rate_over_time.len()
            + data.response_time_over_time.len();
        total_points * 64 // 64 bytes per data point estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::analytics::dashboard_tests::DashboardTestFixture;

    #[tokio::test]
    async fn test_dashboard_cache_basic_operations() -> Result<()> {
        let config = CacheConfig {
            default_ttl_seconds: 60,
            max_memory_mb: 10,
            max_items: 100,
            ..Default::default()
        };

        let cache = DashboardCache::new(config);

        // Test dashboard data caching
        let key = CacheKey::dashboard_data(None);
        let dashboard_data = DashboardData {
            today_cost: 10.50,
            today_commands: 25,
            success_rate: 95.0,
            recent_activity: vec![],
            top_commands: vec![],
            active_alerts: vec![],
            last_updated: Utc::now(),
        };

        // Cache miss initially
        assert!(cache.get_dashboard_data(&key).await.is_none());

        // Set data
        cache
            .set_dashboard_data(key.clone(), dashboard_data.clone(), None)
            .await;

        // Cache hit
        let cached_data = cache.get_dashboard_data(&key).await;
        assert!(cached_data.is_some());
        assert_eq!(cached_data.unwrap().today_cost, 10.50);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_expiration() -> Result<()> {
        let config = CacheConfig {
            default_ttl_seconds: 1, // Very short TTL for testing
            ..Default::default()
        };

        let cache = DashboardCache::new(config);

        let key = CacheKey::live_dashboard_data();
        let fixture = DashboardTestFixture::new().await?;
        let live_data = fixture.dashboard_manager.generate_live_data().await?;

        // Set data with short TTL
        cache
            .set_live_dashboard_data(key.clone(), live_data, Some(1))
            .await;

        // Should be cached initially
        assert!(cache.get_live_dashboard_data(&key).await.is_some());

        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Should be expired now
        assert!(cache.get_live_dashboard_data(&key).await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_statistics() -> Result<()> {
        let cache = DashboardCache::new(CacheConfig::default());

        let key1 = CacheKey::dashboard_data(None);
        let key2 = CacheKey::analytics_summary(7);

        let dashboard_data = DashboardData {
            today_cost: 5.25,
            today_commands: 15,
            success_rate: 90.0,
            recent_activity: vec![],
            top_commands: vec![],
            active_alerts: vec![],
            last_updated: Utc::now(),
        };

        // Generate some cache activity
        cache.get_dashboard_data(&key1).await; // Miss
        cache
            .set_dashboard_data(key1.clone(), dashboard_data, None)
            .await;
        cache.get_dashboard_data(&key1).await; // Hit
        cache.get_analytics_summary(&key2).await; // Miss

        let stats = cache.get_stats().await;

        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 2);
        assert!((stats.hit_ratio - 0.333).abs() < 0.01); // Approximately 1/3

        println!("Cache Stats: {:?}", stats);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_invalidation() -> Result<()> {
        let config = CacheConfig {
            enable_smart_invalidation: true,
            ..Default::default()
        };

        let cache = DashboardCache::new(config);

        let key = CacheKey::dashboard_data(None);
        let dashboard_data = DashboardData {
            today_cost: 15.75,
            today_commands: 30,
            success_rate: 88.0,
            recent_activity: vec![],
            top_commands: vec![],
            active_alerts: vec![],
            last_updated: Utc::now(),
        };

        // Set data
        cache
            .set_dashboard_data(key.clone(), dashboard_data, None)
            .await;
        assert!(cache.get_dashboard_data(&key).await.is_some());

        // Invalidate related caches
        cache.invalidate_related_caches("cost_data").await;

        // Should be invalidated now
        assert!(cache.get_dashboard_data(&key).await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_cleanup_task() -> Result<()> {
        let config = CacheConfig {
            cleanup_interval_seconds: 1, // Very frequent for testing
            ..Default::default()
        };

        let cache = DashboardCache::new(config);

        // Start cleanup task
        let cleanup_task = cache.start_cleanup_task().await;

        let key = CacheKey::live_dashboard_data();
        let fixture = DashboardTestFixture::new().await?;
        let live_data = fixture.dashboard_manager.generate_live_data().await?;

        // Set data with very short TTL
        cache
            .set_live_dashboard_data(key.clone(), live_data, Some(1))
            .await;

        // Wait for cleanup to run
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        let stats = cache.get_stats().await;
        assert!(stats.expired_items_cleaned > 0);

        cleanup_task.abort();

        Ok(())
    }
}
