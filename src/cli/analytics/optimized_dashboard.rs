//! Optimized dashboard manager with advanced streaming capabilities
//!
//! This module provides an enhanced dashboard manager that integrates with the
//! streaming optimizer for better performance, buffer management, and real-time
//! analytics delivery.
//!
//! # Key Improvements
//!
//! - **Adaptive Streaming**: Dynamic buffer sizing based on consumer speed
//! - **Intelligent Batching**: Groups updates for efficient delivery
//! - **Backpressure Handling**: Automatically manages slow consumers
//! - **Connection Health Monitoring**: Tracks client performance
//! - **Advanced Caching**: Multi-layer caching with intelligent invalidation
//! - **Performance Metrics**: Real-time streaming performance monitoring

use super::dashboard::{
    DashboardConfig, DashboardLayout, HealthStatus, LiveDashboardData, SystemStatus,
    TimeSeriesData, WidgetConfig,
};
use super::{
    dashboard_cache::{CacheConfig, DashboardCache},
    streaming_optimizer::{StreamingBatch, StreamingConfig, StreamingMetrics, StreamingOptimizer},
    AnalyticsEngine, DashboardData,
};
use crate::cli::error::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, sleep};

/// Enhanced dashboard configuration with streaming optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedDashboardConfig {
    /// Base dashboard configuration
    pub base_config: DashboardConfig,
    /// Streaming optimization configuration
    pub streaming_config: StreamingConfig,
    /// Enable differential updates (only send changes)
    pub enable_differential_updates: bool,
    /// Compression threshold for large updates (bytes)
    pub compression_threshold: usize,
    /// Maximum update frequency per client (updates/second)
    pub max_client_update_rate: f64,
    /// Enable priority-based updates
    pub enable_priority_updates: bool,
    /// Client-specific buffer sizes
    pub enable_per_client_buffering: bool,
}

impl Default for OptimizedDashboardConfig {
    fn default() -> Self {
        Self {
            base_config: DashboardConfig::default(),
            streaming_config: StreamingConfig::default(),
            enable_differential_updates: true,
            compression_threshold: 1024,
            max_client_update_rate: 10.0,
            enable_priority_updates: true,
            enable_per_client_buffering: true,
        }
    }
}

/// Dashboard update with priority and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardUpdate {
    pub data: LiveDashboardData,
    pub update_type: UpdateType,
    pub priority: UpdatePriority,
    pub client_id: Option<String>,
    pub sequence_number: u64,
    pub differential_data: Option<DifferentialUpdate>,
}

/// Type of dashboard update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    Full,
    Incremental,
    Differential,
    HeartBeat,
    Alert,
}

/// Update priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Differential update containing only changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialUpdate {
    pub changed_fields: Vec<String>,
    pub field_values: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

/// Client connection information
#[derive(Debug, Clone)]
pub struct ClientConnection {
    pub client_id: String,
    pub connected_at: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub update_count: u64,
    pub preferred_update_rate: f64,
    pub buffer_size: usize,
    pub supports_compression: bool,
    pub supports_differential: bool,
}

/// Optimized dashboard manager with advanced streaming
pub struct OptimizedDashboardManager {
    analytics_engine: Arc<AnalyticsEngine>,
    config: OptimizedDashboardConfig,
    layouts: Arc<RwLock<HashMap<String, DashboardLayout>>>,
    streaming_optimizer: Arc<StreamingOptimizer<DashboardUpdate>>,
    cache: Option<Arc<DashboardCache>>,
    clients: Arc<RwLock<HashMap<String, ClientConnection>>>,
    last_full_data: Arc<RwLock<Option<LiveDashboardData>>>,
    update_counter: Arc<std::sync::atomic::AtomicU64>,
}

impl OptimizedDashboardManager {
    /// Create new optimized dashboard manager
    pub async fn new(
        analytics_engine: Arc<AnalyticsEngine>,
        config: OptimizedDashboardConfig,
    ) -> Result<Self> {
        let streaming_optimizer =
            Arc::new(StreamingOptimizer::new(config.streaming_config.clone()));
        streaming_optimizer.start().await?;

        Ok(Self {
            analytics_engine,
            config,
            layouts: Arc::new(RwLock::new(HashMap::new())),
            streaming_optimizer,
            cache: None,
            clients: Arc::new(RwLock::new(HashMap::new())),
            last_full_data: Arc::new(RwLock::new(None)),
            update_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Create optimized dashboard manager with caching
    pub async fn with_cache(
        analytics_engine: Arc<AnalyticsEngine>,
        config: OptimizedDashboardConfig,
        cache_config: CacheConfig,
    ) -> Result<Self> {
        let mut manager = Self::new(analytics_engine, config).await?;
        manager.cache = Some(Arc::new(DashboardCache::new(cache_config)));
        Ok(manager)
    }

    /// Start the optimized dashboard update system
    pub async fn start(&self) -> Result<()> {
        // Start main update loop
        self.start_update_loop().await?;

        // Start client health monitoring
        self.start_client_monitor().await?;

        // Start performance optimization
        self.start_performance_optimizer().await?;

        Ok(())
    }

    /// Subscribe to optimized updates with client tracking
    pub async fn subscribe_client(
        &self,
        client_id: String,
    ) -> broadcast::Receiver<StreamingBatch<DashboardUpdate>> {
        // Register client
        let mut clients = self.clients.write().await;
        clients.insert(
            client_id.clone(),
            ClientConnection {
                client_id: client_id.clone(),
                connected_at: Utc::now(),
                last_update: Utc::now(),
                update_count: 0,
                preferred_update_rate: self.config.max_client_update_rate,
                buffer_size: self.config.streaming_config.initial_buffer_size,
                supports_compression: true,
                supports_differential: self.config.enable_differential_updates,
            },
        );

        // Subscribe with tracking
        self.streaming_optimizer
            .subscribe_with_tracking(client_id)
            .await
    }

    /// Subscribe without client tracking (for backward compatibility)
    pub fn subscribe(&self) -> broadcast::Receiver<StreamingBatch<DashboardUpdate>> {
        self.streaming_optimizer.subscribe()
    }

    /// Get streaming performance metrics
    pub async fn get_streaming_metrics(&self) -> StreamingMetrics {
        self.streaming_optimizer.get_metrics().await
    }

    /// Get client statistics
    pub async fn get_client_stats(&self) -> HashMap<String, ClientConnection> {
        self.clients.read().await.clone()
    }

    /// Set client preferences
    pub async fn set_client_preferences(
        &self,
        client_id: &str,
        update_rate: Option<f64>,
        buffer_size: Option<usize>,
        supports_compression: Option<bool>,
        supports_differential: Option<bool>,
    ) -> Result<()> {
        let mut clients = self.clients.write().await;

        if let Some(client) = clients.get_mut(client_id) {
            if let Some(rate) = update_rate {
                client.preferred_update_rate = rate.min(self.config.max_client_update_rate);
            }
            if let Some(size) = buffer_size {
                client.buffer_size = size;
            }
            if let Some(compression) = supports_compression {
                client.supports_compression = compression;
            }
            if let Some(differential) = supports_differential {
                client.supports_differential = differential;
            }
        }

        Ok(())
    }

    /// Force send update to all clients
    pub async fn broadcast_update(&self, priority: UpdatePriority) -> Result<()> {
        let update = self.generate_dashboard_update(None, priority).await?;
        self.streaming_optimizer.send_item(update).await?;
        Ok(())
    }

    /// Send targeted update to specific client
    pub async fn send_client_update(
        &self,
        client_id: &str,
        priority: UpdatePriority,
    ) -> Result<()> {
        let update = self
            .generate_dashboard_update(Some(client_id.to_string()), priority)
            .await?;
        self.streaming_optimizer.send_item(update).await?;
        Ok(())
    }

    // Private implementation methods

    async fn start_update_loop(&self) -> Result<()> {
        let streaming_optimizer = Arc::clone(&self.streaming_optimizer);
        let analytics_engine = Arc::clone(&self.analytics_engine);
        let cache = self.cache.clone();
        let config = self.config.clone();
        let last_full_data = Arc::clone(&self.last_full_data);
        let update_counter = Arc::clone(&self.update_counter);

        tokio::spawn(async move {
            let mut interval = interval(std::time::Duration::from_secs(
                config.base_config.refresh_interval_seconds,
            ));

            loop {
                interval.tick().await;

                // Generate update
                if let Ok(update) = Self::generate_update_internal(
                    &analytics_engine,
                    &cache,
                    &config,
                    &last_full_data,
                    &update_counter,
                    None,
                    UpdatePriority::Normal,
                )
                .await
                {
                    // Send through streaming optimizer
                    let _ = streaming_optimizer.send_item(update).await;
                }
            }
        });

        Ok(())
    }

    async fn start_client_monitor(&self) -> Result<()> {
        let clients = Arc::clone(&self.clients);

        tokio::spawn(async move {
            let mut interval = interval(std::time::Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Clean up disconnected clients
                let now = Utc::now();
                let mut clients_guard = clients.write().await;

                clients_guard.retain(|_, client| {
                    // Keep clients that have been active in the last 5 minutes
                    now.signed_duration_since(client.last_update).num_minutes() < 5
                });
            }
        });

        Ok(())
    }

    async fn start_performance_optimizer(&self) -> Result<()> {
        let streaming_optimizer = Arc::clone(&self.streaming_optimizer);
        let clients = Arc::clone(&self.clients);

        tokio::spawn(async move {
            let mut interval = interval(std::time::Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Get streaming metrics
                let metrics = streaming_optimizer.get_metrics().await;

                // Optimize based on metrics
                if metrics.buffer_utilization > 0.8 {
                    // High buffer utilization - consider reducing update frequency
                    // or increasing buffer sizes for clients
                    let mut clients_guard = clients.write().await;
                    for client in clients_guard.values_mut() {
                        if client.preferred_update_rate > 1.0 {
                            client.preferred_update_rate *= 0.9; // Reduce by 10%
                        }
                    }
                }

                if metrics.average_latency_ms > 100.0 {
                    // High latency - optimize client connections
                    // This could trigger buffer size adjustments or compression
                }
            }
        });

        Ok(())
    }

    async fn generate_dashboard_update(
        &self,
        client_id: Option<String>,
        priority: UpdatePriority,
    ) -> Result<DashboardUpdate> {
        Self::generate_update_internal(
            &self.analytics_engine,
            &self.cache,
            &self.config,
            &self.last_full_data,
            &self.update_counter,
            client_id,
            priority,
        )
        .await
    }

    async fn generate_update_internal(
        analytics_engine: &Arc<AnalyticsEngine>,
        cache: &Option<Arc<DashboardCache>>,
        config: &OptimizedDashboardConfig,
        last_full_data: &Arc<RwLock<Option<LiveDashboardData>>>,
        update_counter: &Arc<std::sync::atomic::AtomicU64>,
        client_id: Option<String>,
        priority: UpdatePriority,
    ) -> Result<DashboardUpdate> {
        // Generate fresh dashboard data
        let current_data = Self::generate_live_data_internal(analytics_engine, cache).await?;

        // Determine update type
        let (update_type, differential_data) = if config.enable_differential_updates {
            let last_data_guard = last_full_data.read().await;

            if let Some(ref last_data) = *last_data_guard {
                // Generate differential update
                let diff = Self::generate_differential_update(last_data, &current_data);
                (UpdateType::Differential, Some(diff))
            } else {
                (UpdateType::Full, None)
            }
        } else {
            (UpdateType::Full, None)
        };

        // Update last full data
        *last_full_data.write().await = Some(current_data.clone());

        // Create update
        let update = DashboardUpdate {
            data: current_data,
            update_type,
            priority,
            client_id,
            sequence_number: update_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            differential_data,
        };

        Ok(update)
    }

    async fn generate_live_data_internal(
        analytics_engine: &Arc<AnalyticsEngine>,
        cache: &Option<Arc<DashboardCache>>,
    ) -> Result<LiveDashboardData> {
        // Check cache first if available
        if let Some(cache) = cache {
            use super::dashboard_cache::CacheKey;
            let cache_key = CacheKey::live_dashboard_data();
            if let Some(cached_data) = cache.get_live_dashboard_data(&cache_key).await {
                return Ok(cached_data);
            }
        }

        // Generate fresh data (reuse existing logic from original dashboard)
        let current_metrics = analytics_engine.get_dashboard_data().await?;
        let time_series = Self::generate_time_series_internal(analytics_engine).await?;
        let system_status = Self::get_system_status_internal(analytics_engine).await?;

        let live_data = LiveDashboardData {
            timestamp: Utc::now(),
            current_metrics,
            time_series,
            system_status,
        };

        // Cache the result if caching is enabled
        if let Some(cache) = cache {
            use super::dashboard_cache::CacheKey;
            let cache_key = CacheKey::live_dashboard_data();
            cache
                .set_live_dashboard_data(cache_key, live_data.clone(), Some(30))
                .await;
        }

        Ok(live_data)
    }

    async fn generate_time_series_internal(
        analytics_engine: &Arc<AnalyticsEngine>,
    ) -> Result<TimeSeriesData> {
        // Use optimized time series generation
        use super::time_series_optimizer::{TimeSeriesOptimizer, TimeSeriesType};

        let hours = 24u32; // Default to 24 hours
        let start_time = Utc::now() - Duration::hours(hours as i64);
        let end_time = Utc::now();

        let optimizer = TimeSeriesOptimizer::new((**analytics_engine).clone());
        let types = vec![
            TimeSeriesType::Cost,
            TimeSeriesType::Commands,
            TimeSeriesType::SuccessRate,
            TimeSeriesType::ResponseTime,
        ];

        let optimized_data = optimizer
            .generate_optimized_time_series(start_time, end_time, types)
            .await?;
        Ok(TimeSeriesOptimizer::to_legacy_format(&optimized_data))
    }

    async fn get_system_status_internal(
        analytics_engine: &Arc<AnalyticsEngine>,
    ) -> Result<SystemStatus> {
        // Basic system status implementation
        let health = HealthStatus::Healthy;
        let uptime_hours = 24.0; // Placeholder
        let active_sessions = 1; // Placeholder

        Ok(SystemStatus {
            health,
            uptime_hours,
            active_sessions,
            memory_usage_mb: 0.0,
            disk_usage_percent: 0.0,
            last_error: None,
        })
    }

    fn generate_differential_update(
        last_data: &LiveDashboardData,
        current_data: &LiveDashboardData,
    ) -> DifferentialUpdate {
        let mut changed_fields = Vec::new();
        let mut field_values = HashMap::new();

        // Compare relevant fields and track changes
        if last_data.current_metrics.today_cost != current_data.current_metrics.today_cost {
            changed_fields.push("today_cost".to_string());
            field_values.insert(
                "today_cost".to_string(),
                serde_json::json!(current_data.current_metrics.today_cost),
            );
        }

        if last_data.current_metrics.today_commands != current_data.current_metrics.today_commands {
            changed_fields.push("today_commands".to_string());
            field_values.insert(
                "today_commands".to_string(),
                serde_json::json!(current_data.current_metrics.today_commands),
            );
        }

        // Add more field comparisons as needed...

        DifferentialUpdate {
            changed_fields,
            field_values,
            timestamp: Utc::now(),
        }
    }
}

/// Factory for creating optimized dashboard configurations
pub struct OptimizedDashboardFactory;

impl OptimizedDashboardFactory {
    /// Create high-performance dashboard configuration
    pub fn high_performance() -> OptimizedDashboardConfig {
        OptimizedDashboardConfig {
            base_config: DashboardConfig {
                refresh_interval_seconds: 5,
                max_recent_entries: 100,
                enable_live_updates: true,
                chart_time_range_hours: 24,
                enable_real_system_monitoring: true,
            },
            streaming_config:
                super::streaming_optimizer::StreamingOptimizerFactory::performance_optimized(),
            enable_differential_updates: true,
            compression_threshold: 512,
            max_client_update_rate: 20.0,
            enable_priority_updates: true,
            enable_per_client_buffering: true,
        }
    }

    /// Create memory-efficient dashboard configuration
    pub fn memory_efficient() -> OptimizedDashboardConfig {
        OptimizedDashboardConfig {
            base_config: DashboardConfig {
                refresh_interval_seconds: 15,
                max_recent_entries: 25,
                enable_live_updates: true,
                chart_time_range_hours: 12,
                enable_real_system_monitoring: false,
            },
            streaming_config:
                super::streaming_optimizer::StreamingOptimizerFactory::memory_optimized(),
            enable_differential_updates: true,
            compression_threshold: 2048,
            max_client_update_rate: 2.0,
            enable_priority_updates: false,
            enable_per_client_buffering: false,
        }
    }

    /// Create low-latency dashboard configuration
    pub fn low_latency() -> OptimizedDashboardConfig {
        OptimizedDashboardConfig {
            base_config: DashboardConfig {
                refresh_interval_seconds: 1,
                max_recent_entries: 200,
                enable_live_updates: true,
                chart_time_range_hours: 48,
                enable_real_system_monitoring: true,
            },
            streaming_config: super::streaming_optimizer::StreamingOptimizerFactory::low_latency(),
            enable_differential_updates: true,
            compression_threshold: 256,
            max_client_update_rate: 50.0,
            enable_priority_updates: true,
            enable_per_client_buffering: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn test_optimized_dashboard_creation() {
        let config = OptimizedDashboardConfig::default();
        let analytics_engine = Arc::new(AnalyticsEngine::new_mock());

        let dashboard = OptimizedDashboardManager::new(analytics_engine, config).await;
        assert!(dashboard.is_ok());
    }

    #[tokio::test]
    async fn test_client_subscription() {
        let config = OptimizedDashboardConfig::default();
        let analytics_engine = Arc::new(AnalyticsEngine::new_mock());
        let dashboard = OptimizedDashboardManager::new(analytics_engine, config)
            .await
            .unwrap();

        dashboard.start().await.unwrap();

        let mut receiver = dashboard.subscribe_client("test_client".to_string()).await;

        // Send a test update
        dashboard
            .broadcast_update(UpdatePriority::Normal)
            .await
            .unwrap();

        // Should receive update within timeout
        let result = timeout(Duration::from_secs(2), receiver.recv()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_streaming_metrics() {
        let config = OptimizedDashboardConfig::default();
        let analytics_engine = Arc::new(AnalyticsEngine::new_mock());
        let dashboard = OptimizedDashboardManager::new(analytics_engine, config)
            .await
            .unwrap();

        dashboard.start().await.unwrap();

        // Generate some activity
        dashboard
            .broadcast_update(UpdatePriority::Normal)
            .await
            .unwrap();

        let metrics = dashboard.get_streaming_metrics().await;
        assert!(metrics.throughput_messages_per_second >= 0.0);
    }
}
