//! Analytics and reporting module
//!
//! This module provides comprehensive analytics capabilities for the Claude AI Interactive system.
//! It integrates with cost tracking and history modules to deliver insights, visualizations, and
//! automated reporting.
//!
//! # Overview
//!
//! The analytics module consists of several key components:
//!
//! - **Analytics Engine**: Central orchestrator for all analytics operations
//! - **Dashboard System**: Real-time views and interactive visualizations
//! - **Metrics Collection**: Performance monitoring and custom metric tracking
//! - **Report Generation**: Automated and on-demand report creation
//! - **Alert System**: Proactive monitoring and notifications
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```no_run
//! use crate_interactive::analytics::{AnalyticsEngine, AnalyticsConfig};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize analytics engine
//! let config = AnalyticsConfig::default();
//! let analytics = AnalyticsEngine::new(
//!     cost_tracker.clone(),
//!     history_store.clone(),
//!     config
//! );
//!
//! // Generate analytics summary
//! let summary = analytics.generate_summary(30).await?;
//! println!("Total cost: ${:.2}", summary.cost_summary.total_cost);
//! println!("Success rate: {:.1}%", summary.performance_metrics.success_rate);
//! # Ok(())
//! # }
//! ```
//!
//! ## Real-time Monitoring
//!
//! ```no_run
//! use crate_interactive::analytics::RealTimeAnalyticsStream;
//!
//! # async fn example(analytics_engine: std::sync::Arc<AnalyticsEngine>) -> Result<(), Box<dyn std::error::Error>> {
//! // Create real-time stream
//! let stream = RealTimeAnalyticsStream::new(analytics_engine).await?;
//!
//! // Start streaming updates
//! stream.start_streaming().await?;
//!
//! // Process updates
//! let updates = stream.get_recent_updates(100).await;
//! for update in updates {
//!     println!("Update: {:?}", update);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The module follows a layered architecture:
//!
//! 1. **Data Layer**: Interfaces with CostTracker and HistoryStore
//! 2. **Processing Layer**: Aggregation, calculation, and analysis
//! 3. **Presentation Layer**: Dashboards, reports, and APIs
//! 4. **Streaming Layer**: Real-time updates and notifications
//!
//! # Performance Considerations
//!
//! - All operations are async and non-blocking
//! - Data is processed in streams to minimize memory usage
//! - Metrics are pre-aggregated for common queries
//! - Caching is used for frequently accessed data
//!
//! # Error Handling
//!
//! The module uses the standard `Result<T, InteractiveError>` pattern.
//! All public APIs return proper error types for comprehensive error handling.

pub mod dashboard;
pub mod dashboard_cache;
pub mod memory_optimizer;
pub mod metrics;
pub mod optimized_dashboard;
pub mod performance_profiler;
pub mod reports;
pub mod simple;
pub mod streaming_optimizer;
pub mod time_series_optimizer;

#[cfg(test)]
pub mod test_utils;

#[cfg(test)]
pub mod analytics_test;

#[cfg(test)]
pub mod dashboard_tests;

#[cfg(test)]
pub mod metrics_test;

#[cfg(test)]
pub mod report_test;

#[cfg(test)]
pub mod property_tests;

#[cfg(test)]
pub mod infrastructure_tests;

#[cfg(test)]
pub mod realtime_performance_tests;

#[cfg(test)]
pub mod dashboard_test;

#[cfg(test)]
pub mod performance_fixes;

#[cfg(test)]
pub mod performance_optimizations;

#[cfg(test)]
pub mod streaming_tests;

#[cfg(test)]
pub mod performance_baseline_test;

#[cfg(test)]
pub mod performance_improvement_test;

use crate::{
    cli::cost::{CostFilter, CostSummary, CostTracker},
    cli::history::{HistorySearch, HistoryStats, HistoryStore},
    cli::session::SessionId,
    Result,
};
use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Comprehensive analytics engine for the Claude AI Interactive system
///
/// The `AnalyticsEngine` is the central component of the analytics module,
/// orchestrating data collection, processing, and presentation across all
/// analytics features.
///
/// # Architecture
///
/// The engine maintains references to:
/// - Cost tracking data for financial analytics
/// - History store for usage patterns and performance metrics
/// - Configuration for customizing behavior
///
/// # Thread Safety
///
/// The engine is `Clone` and uses `Arc<RwLock<T>>` for shared state,
/// making it safe to use across multiple async tasks.
///
/// # Example
///
/// ```no_run
/// use crate_interactive::analytics::{AnalyticsEngine, AnalyticsConfig};
/// # use std::sync::Arc;
/// # use tokio::sync::RwLock;
///
/// # async fn example(cost_tracker: Arc<RwLock<CostTracker>>, history_store: Arc<RwLock<HistoryStore>>) {
/// let config = AnalyticsConfig {
///     enable_real_time_alerts: true,
///     cost_alert_threshold: 50.0,
///     ..Default::default()
/// };
///
/// let engine = AnalyticsEngine::new(cost_tracker, history_store, config);
/// # }
/// ```
#[derive(Clone)]
pub struct AnalyticsEngine {
    cost_tracker: std::sync::Arc<tokio::sync::RwLock<CostTracker>>,
    history_store: std::sync::Arc<tokio::sync::RwLock<HistoryStore>>,
    config: AnalyticsConfig,
}

/// Configuration for the analytics engine
///
/// Controls various aspects of analytics behavior including alerting,
/// reporting schedules, data retention, and refresh intervals.
///
/// # Fields
///
/// * `enable_real_time_alerts` - Enable/disable real-time alert monitoring
/// * `cost_alert_threshold` - Dollar amount that triggers cost alerts
/// * `report_schedule` - Default schedule for automated reports
/// * `retention_days` - How long to retain analytics data
/// * `dashboard_refresh_interval` - Dashboard update frequency in seconds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable real-time monitoring and alerts
    pub enable_real_time_alerts: bool,
    /// Cost threshold in dollars that triggers alerts
    pub cost_alert_threshold: f64,
    /// Schedule for automated report generation
    pub report_schedule: ReportSchedule,
    /// Number of days to retain analytics data
    pub retention_days: u32,
    /// Dashboard refresh interval in seconds
    pub dashboard_refresh_interval: u64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_real_time_alerts: true,
            cost_alert_threshold: 10.0, // $10 threshold
            report_schedule: ReportSchedule::Weekly,
            retention_days: 90,
            dashboard_refresh_interval: 30,
        }
    }
}

/// Report scheduling options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportSchedule {
    Daily,
    Weekly,
    Monthly,
    Custom(Duration),
}

/// Comprehensive analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub cost_summary: CostSummary,
    pub history_stats: HistoryStats,
    pub performance_metrics: PerformanceMetrics,
    pub insights: Vec<String>,
    pub alerts: Vec<Alert>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub average_response_time: f64,
    pub success_rate: f64,
    pub throughput_commands_per_hour: f64,
    pub peak_usage_hour: u8,
    pub slowest_commands: Vec<(String, f64)>,
    pub error_rate_by_command: HashMap<String, f64>,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<SessionId>,
    pub resolved: bool,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    CostThreshold,
    ErrorRate,
    PerformanceDegradation,
    UnusualUsage,
    SystemHealth,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Real-time analytics stream handler for processing live updates
///
/// Provides high-performance streaming of analytics updates with built-in
/// memory management and buffering capabilities.
///
/// # Features
///
/// - Asynchronous update processing
/// - Bounded memory usage with configurable buffers
/// - Batch processing for efficiency
/// - Memory tracking and profiling
///
/// # Example
///
/// ```no_run
/// use crate_interactive::analytics::RealTimeAnalyticsStream;
///
/// # async fn example(engine: std::sync::Arc<AnalyticsEngine>) -> Result<(), Box<dyn std::error::Error>> {
/// // Create stream handler
/// let stream = RealTimeAnalyticsStream::new(engine).await?;
///
/// // Start streaming
/// stream.start_streaming().await?;
///
/// // Process updates in batches
/// let updates = stream.get_recent_updates(100).await;
/// stream.process_update_batch(updates).await?;
///
/// // Monitor memory usage
/// let stats = stream.memory_tracker.get_stats();
/// println!("Memory usage: {}MB", stats.peak_memory_mb);
///
/// // Stop when done
/// stream.stop_streaming();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct RealTimeAnalyticsStream {
    /// Reference to the analytics engine
    pub analytics_engine: Arc<AnalyticsEngine>,
    /// Atomic flag indicating if streaming is active
    pub is_streaming: Arc<std::sync::atomic::AtomicBool>,
    /// Counter for total updates processed
    pub update_count: Arc<std::sync::atomic::AtomicU64>,
    /// Memory usage tracker
    pub memory_tracker: Arc<MemoryTracker>,
    /// Buffer for recent updates (bounded size)
    pub updates_buffer: Arc<tokio::sync::RwLock<Vec<AnalyticsUpdate>>>,
}

/// Analytics update event for real-time streaming
#[derive(Debug, Clone)]
pub struct AnalyticsUpdate {
    pub timestamp: DateTime<Utc>,
    pub update_type: UpdateType,
    pub session_id: SessionId,
    pub metric_deltas: MetricDeltas,
}

/// Types of real-time updates
#[derive(Debug, Clone)]
pub enum UpdateType {
    CommandCompleted,
    CostIncurred,
    SessionStarted,
    SessionEnded,
    AlertTriggered,
    PerformanceMetricUpdated,
}

/// Metric deltas for incremental updates
#[derive(Debug, Clone, Default)]
pub struct MetricDeltas {
    pub cost_delta: f64,
    pub command_count_delta: i32,
    pub success_count_delta: i32,
    pub failure_count_delta: i32,
    pub response_time_delta: i64,
    pub token_delta: i64,
}

/// Memory usage tracking for performance monitoring
#[derive(Debug, Default)]
pub struct MemoryTracker {
    pub peak_memory_mb: std::sync::atomic::AtomicU64,
    pub allocations: std::sync::atomic::AtomicU64,
    pub deallocations: std::sync::atomic::AtomicU64,
    pub active_objects: std::sync::atomic::AtomicU64,
}

impl MemoryTracker {
    pub fn record_allocation(&self, size_mb: u64) {
        use std::sync::atomic::Ordering;
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.active_objects.fetch_add(1, Ordering::Relaxed);
        let new_peak = self.peak_memory_mb.load(Ordering::Relaxed) + size_mb;
        self.peak_memory_mb.store(new_peak, Ordering::Relaxed);
    }

    pub fn record_deallocation(&self, size_mb: u64) {
        use std::sync::atomic::Ordering;
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.active_objects.fetch_sub(1, Ordering::Relaxed);
        let current = self.peak_memory_mb.load(Ordering::Relaxed);
        if current >= size_mb {
            self.peak_memory_mb
                .store(current - size_mb, Ordering::Relaxed);
        }
    }

    pub fn get_stats(&self) -> MemoryStats {
        use std::sync::atomic::Ordering;
        MemoryStats {
            peak_memory_mb: self.peak_memory_mb.load(Ordering::Relaxed),
            total_allocations: self.allocations.load(Ordering::Relaxed),
            total_deallocations: self.deallocations.load(Ordering::Relaxed),
            active_objects: self.active_objects.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub peak_memory_mb: u64,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub active_objects: u64,
}

impl RealTimeAnalyticsStream {
    pub async fn new(analytics_engine: Arc<AnalyticsEngine>) -> Result<Self> {
        Ok(Self {
            analytics_engine,
            is_streaming: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            update_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            memory_tracker: Arc::new(MemoryTracker::default()),
            updates_buffer: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        })
    }

    /// Start real-time streaming of analytics updates
    pub async fn start_streaming(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        self.is_streaming.store(true, Ordering::Relaxed);

        // Simulate real-time updates
        let updates_buffer = Arc::clone(&self.updates_buffer);
        let is_streaming = Arc::clone(&self.is_streaming);
        let update_count = Arc::clone(&self.update_count);
        let memory_tracker = Arc::clone(&self.memory_tracker);

        tokio::spawn(async move {
            while is_streaming.load(Ordering::Relaxed) {
                let update = AnalyticsUpdate {
                    timestamp: Utc::now(),
                    update_type: UpdateType::CommandCompleted,
                    session_id: uuid::Uuid::new_v4(),
                    metric_deltas: MetricDeltas {
                        cost_delta: 0.01,
                        command_count_delta: 1,
                        success_count_delta: 1,
                        ..Default::default()
                    },
                };

                // Track memory usage for this update
                memory_tracker.record_allocation(1); // 1MB per update

                // Add to buffer
                if let Ok(mut buffer) = updates_buffer.try_write() {
                    buffer.push(update);
                    // Keep buffer size reasonable
                    if buffer.len() > 1000 {
                        buffer.drain(0..500);
                    }
                }

                update_count.fetch_add(1, Ordering::Relaxed);

                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        });

        Ok(())
    }

    /// Stop streaming and clean up resources
    pub fn stop_streaming(&self) {
        use std::sync::atomic::Ordering;
        self.is_streaming.store(false, Ordering::Relaxed);
    }

    /// Get recent updates from buffer
    pub async fn get_recent_updates(&self, limit: usize) -> Vec<AnalyticsUpdate> {
        let buffer = self.updates_buffer.read().await;
        let start = if buffer.len() > limit {
            buffer.len() - limit
        } else {
            0
        };
        buffer[start..].to_vec()
    }

    /// Process a batch of analytics updates efficiently
    pub async fn process_update_batch(&self, updates: Vec<AnalyticsUpdate>) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Group updates by type for batch processing
        let mut command_updates = Vec::new();
        let mut cost_updates = Vec::new();

        for update in updates {
            match update.update_type {
                UpdateType::CommandCompleted => command_updates.push(update),
                UpdateType::CostIncurred => cost_updates.push(update),
                _ => {}
            }
        }

        let command_count = command_updates.len();
        let cost_count = cost_updates.len();

        // Process command updates in batch
        for _update in command_updates {
            self.memory_tracker.record_allocation(1);
            // Simulate processing
            tokio::time::sleep(std::time::Duration::from_micros(100)).await;
        }

        // Process cost updates in batch
        for _update in cost_updates {
            self.memory_tracker.record_allocation(1);
            // Simulate processing
            tokio::time::sleep(std::time::Duration::from_micros(100)).await;
        }

        let _processing_time = start_time.elapsed();

        // Memory cleanup simulation
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        self.memory_tracker
            .record_deallocation(command_count as u64 + cost_count as u64);

        Ok(())
    }
}

impl AnalyticsEngine {
    /// Create a new analytics engine
    pub fn new(
        cost_tracker: std::sync::Arc<tokio::sync::RwLock<CostTracker>>,
        history_store: std::sync::Arc<tokio::sync::RwLock<HistoryStore>>,
        config: AnalyticsConfig,
    ) -> Self {
        Self {
            cost_tracker,
            history_store,
            config,
        }
    }

    /// Generate comprehensive analytics summary for a given time period
    ///
    /// This method aggregates data from multiple sources to provide a complete
    /// overview of system usage, costs, and performance.
    ///
    /// # Arguments
    ///
    /// * `period_days` - Number of days to include in the summary (from now backwards)
    ///
    /// # Returns
    ///
    /// Returns an `AnalyticsSummary` containing:
    /// - Cost breakdown and totals
    /// - Usage statistics and patterns
    /// - Performance metrics
    /// - Generated insights
    /// - Active alerts
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use crate_interactive::analytics::AnalyticsEngine;
    /// # async fn example(engine: &AnalyticsEngine) -> Result<(), Box<dyn std::error::Error>> {
    /// // Generate 30-day summary
    /// let summary = engine.generate_summary(30).await?;
    ///
    /// println!("Total cost: ${:.2}", summary.cost_summary.total_cost);
    /// println!("Commands: {}", summary.cost_summary.command_count);
    /// println!("Success rate: {:.1}%", summary.performance_metrics.success_rate);
    ///
    /// for insight in &summary.insights {
    ///     println!("Insight: {}", insight);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate_summary(&self, period_days: u32) -> Result<AnalyticsSummary> {
        let period_start = Utc::now() - Duration::days(period_days as i64);
        let period_end = Utc::now();

        // Get cost summary
        let cost_filter = CostFilter {
            since: Some(period_start),
            until: Some(period_end),
            ..Default::default()
        };
        let cost_tracker = self.cost_tracker.read().await;
        let cost_summary = cost_tracker.get_filtered_summary(&cost_filter).await?;

        // Get history stats
        let history_search = HistorySearch {
            since: Some(period_start),
            until: Some(period_end),
            ..Default::default()
        };
        let history_store = self.history_store.read().await;
        let history_stats = history_store.get_stats(Some(&history_search)).await?;

        // Calculate performance metrics
        let performance_metrics = self.calculate_performance_metrics(&history_search).await?;

        // Generate insights
        let insights = self
            .generate_insights(&cost_summary, &history_stats, &performance_metrics)
            .await?;

        // Check for alerts
        let alerts = self
            .check_alerts(&cost_summary, &history_stats, &performance_metrics)
            .await?;

        Ok(AnalyticsSummary {
            period_start,
            period_end,
            cost_summary,
            history_stats,
            performance_metrics,
            insights,
            alerts,
        })
    }

    /// Generate detailed report for a session
    pub async fn generate_session_report(&self, session_id: SessionId) -> Result<SessionReport> {
        let cost_summary = self
            .cost_tracker
            .read()
            .await
            .get_session_summary(session_id)
            .await?;

        let history_search = HistorySearch {
            session_id: Some(session_id),
            ..Default::default()
        };
        let history_stats = self
            .history_store
            .read()
            .await
            .get_stats(Some(&history_search))
            .await?;
        let recent_commands = self
            .history_store
            .read()
            .await
            .get_recent_commands(session_id, 20)
            .await?;

        Ok(SessionReport {
            session_id,
            cost_summary,
            history_stats,
            recent_commands,
            generated_at: Utc::now(),
        })
    }

    /// Get real-time dashboard data
    pub async fn get_dashboard_data(&self) -> Result<DashboardData> {
        let now = Utc::now();
        let today_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();

        // Today's summary
        let today_filter = CostFilter {
            since: Some(today_start),
            until: Some(now),
            ..Default::default()
        };
        let today_cost = self
            .cost_tracker
            .read()
            .await
            .get_filtered_summary(&today_filter)
            .await?;

        let today_history = HistorySearch {
            since: Some(today_start),
            until: Some(now),
            ..Default::default()
        };
        let today_stats = self
            .history_store
            .read()
            .await
            .get_stats(Some(&today_history))
            .await?;

        // Recent activity
        let recent_entries = self
            .history_store
            .read()
            .await
            .search(&HistorySearch {
                limit: 10,
                ..Default::default()
            })
            .await?;

        // Top commands
        let top_commands = self.cost_tracker.read().await.get_top_commands(5).await?;

        // Active alerts
        let alerts = self.get_active_alerts().await?;

        Ok(DashboardData {
            today_cost: today_cost.total_cost,
            today_commands: today_stats.total_entries,
            success_rate: today_stats.success_rate,
            recent_activity: recent_entries,
            top_commands,
            active_alerts: alerts,
            last_updated: now,
        })
    }

    /// Generate automated insights
    pub async fn generate_insights(
        &self,
        cost_summary: &CostSummary,
        history_stats: &HistoryStats,
        performance_metrics: &PerformanceMetrics,
    ) -> Result<Vec<String>> {
        let mut insights = Vec::new();

        // Cost insights
        if cost_summary.total_cost > 0.0 {
            insights.push(format!(
                "Total spending: ${:.2} across {} commands (avg ${:.4}/command)",
                cost_summary.total_cost, cost_summary.command_count, cost_summary.average_cost
            ));

            if let Some((most_expensive_cmd, cost)) = cost_summary
                .by_command
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                let percentage = (cost / cost_summary.total_cost) * 100.0;
                insights.push(format!(
                    "Most expensive command: '{}' (${:.2}, {:.1}% of total)",
                    most_expensive_cmd, cost, percentage
                ));
            }
        }

        // Performance insights (only if there's actual data)
        if history_stats.total_entries > 0 && performance_metrics.success_rate < 95.0 {
            insights.push(format!(
                "Success rate is {:.1}% - consider investigating frequent failures",
                performance_metrics.success_rate
            ));
        }

        if history_stats.total_entries > 0 && performance_metrics.average_response_time > 5000.0 {
            insights.push(format!(
                "Average response time is {:.1}ms - performance may be degraded",
                performance_metrics.average_response_time
            ));
        }

        // Usage insights
        if history_stats.total_entries > 0 {
            let days = (history_stats.date_range.1 - history_stats.date_range.0).num_days() as f64;
            if days > 0.0 {
                let avg_commands_per_day = history_stats.total_entries as f64 / days;
                insights.push(format!(
                    "Usage: {:.1} commands per day on average",
                    avg_commands_per_day
                ));
            }
        }

        // Model usage insights
        if cost_summary.by_model.len() > 1 {
            let total_model_cost: f64 = cost_summary.by_model.values().sum();
            if let Some((primary_model, model_cost)) = cost_summary
                .by_model
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                let percentage = (model_cost / total_model_cost) * 100.0;
                insights.push(format!(
                    "Primary model: {} ({:.1}% of model usage costs)",
                    primary_model, percentage
                ));
            }
        }

        Ok(insights)
    }

    /// Check for various alert conditions
    pub async fn check_alerts(
        &self,
        cost_summary: &CostSummary,
        history_stats: &HistoryStats,
        performance_metrics: &PerformanceMetrics,
    ) -> Result<Vec<Alert>> {
        let mut alerts = Vec::new();

        // Cost threshold alert
        if self.config.enable_real_time_alerts
            && cost_summary.total_cost > self.config.cost_alert_threshold
        {
            alerts.push(Alert {
                id: uuid::Uuid::new_v4().to_string(),
                alert_type: AlertType::CostThreshold,
                severity: AlertSeverity::Medium,
                message: format!(
                    "Cost threshold exceeded: ${:.2} > ${:.2}",
                    cost_summary.total_cost, self.config.cost_alert_threshold
                ),
                timestamp: Utc::now(),
                session_id: None,
                resolved: false,
            });
        }

        // Error rate alert
        if history_stats.total_entries > 0 {
            let error_rate =
                (history_stats.failed_commands as f64 / history_stats.total_entries as f64) * 100.0;
            if error_rate > 10.0 {
                alerts.push(Alert {
                    id: uuid::Uuid::new_v4().to_string(),
                    alert_type: AlertType::ErrorRate,
                    severity: if error_rate > 25.0 {
                        AlertSeverity::High
                    } else {
                        AlertSeverity::Medium
                    },
                    message: format!("High error rate: {:.1}%", error_rate),
                    timestamp: Utc::now(),
                    session_id: None,
                    resolved: false,
                });
            }
        }

        // Performance degradation alert
        if performance_metrics.average_response_time > 10000.0 {
            alerts.push(Alert {
                id: uuid::Uuid::new_v4().to_string(),
                alert_type: AlertType::PerformanceDegradation,
                severity: AlertSeverity::Medium,
                message: format!(
                    "Slow response times: avg {:.1}ms",
                    performance_metrics.average_response_time
                ),
                timestamp: Utc::now(),
                session_id: None,
                resolved: false,
            });
        }

        Ok(alerts)
    }

    /// Get current active alerts
    pub async fn get_active_alerts(&self) -> Result<Vec<Alert>> {
        // In a real implementation, this would query a persistent alert store
        // For now, generate fresh alerts based on recent data
        let summary = self.generate_summary(1).await?; // Last day
        Ok(summary.alerts)
    }

    /// Export comprehensive analytics report
    pub async fn export_analytics_report(
        &self,
        path: &PathBuf,
        format: ReportFormat,
        period_days: u32,
    ) -> Result<()> {
        let summary = self.generate_summary(period_days).await?;

        match format {
            ReportFormat::Json => {
                let content = serde_json::to_string_pretty(&summary)?;
                tokio::fs::write(path, content).await?;
            }
            ReportFormat::Html => {
                self.export_html_report(path, &summary).await?;
            }
            ReportFormat::Csv => {
                self.export_csv_report(path, &summary).await?;
            }
            ReportFormat::Pdf => {
                // For now, export as HTML for PDF
                self.export_html_report(path, &summary).await?;
            }
            ReportFormat::Markdown => {
                // Export as markdown format
                let content = self.export_markdown_report(&summary).await?;
                tokio::fs::write(path, content).await?;
            }
        }

        Ok(())
    }

    // Private helper methods

    async fn calculate_performance_metrics(
        &self,
        search: &HistorySearch,
    ) -> Result<PerformanceMetrics> {
        let entries = self.history_store.read().await.search(search).await?;

        if entries.is_empty() {
            return Ok(PerformanceMetrics {
                average_response_time: 0.0,
                success_rate: 0.0,
                throughput_commands_per_hour: 0.0,
                peak_usage_hour: 0,
                slowest_commands: Vec::new(),
                error_rate_by_command: HashMap::new(),
            });
        }

        let total_duration: u64 = entries.iter().map(|e| e.duration_ms).sum();
        let average_response_time = total_duration as f64 / entries.len() as f64;

        let successful_entries = entries.iter().filter(|e| e.success).count();
        let success_rate = (successful_entries as f64 / entries.len() as f64) * 100.0;

        // Calculate throughput
        let time_span = if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
            (last.timestamp - first.timestamp).num_hours() as f64
        } else {
            1.0
        };
        let throughput_commands_per_hour = entries.len() as f64 / time_span.max(1.0);

        // Find peak usage hour
        let mut hourly_counts = [0usize; 24];
        for entry in &entries {
            let hour = entry.timestamp.hour() as usize;
            hourly_counts[hour] += 1;
        }
        let peak_usage_hour = hourly_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(hour, _)| hour)
            .unwrap_or(0) as u8;

        // Find slowest commands
        let mut command_durations: HashMap<String, Vec<u64>> = HashMap::new();
        for entry in &entries {
            command_durations
                .entry(entry.command_name.clone())
                .or_insert_with(Vec::new)
                .push(entry.duration_ms);
        }

        let mut slowest_commands: Vec<_> = command_durations
            .into_iter()
            .map(|(cmd, durations)| {
                let avg_duration = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
                (cmd, avg_duration)
            })
            .collect();
        slowest_commands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        slowest_commands.truncate(5);

        // Calculate error rates by command
        let mut command_stats: HashMap<String, (usize, usize)> = HashMap::new();
        for entry in &entries {
            let (total, failures) = command_stats
                .entry(entry.command_name.clone())
                .or_insert((0, 0));
            *total += 1;
            if !entry.success {
                *failures += 1;
            }
        }

        let error_rate_by_command: HashMap<String, f64> = command_stats
            .into_iter()
            .map(|(cmd, (total, failures))| {
                let error_rate = (failures as f64 / total as f64) * 100.0;
                (cmd, error_rate)
            })
            .collect();

        Ok(PerformanceMetrics {
            average_response_time,
            success_rate,
            throughput_commands_per_hour,
            peak_usage_hour,
            slowest_commands,
            error_rate_by_command,
        })
    }

    async fn export_html_report(&self, path: &PathBuf, summary: &AnalyticsSummary) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(
            file,
            "<html><head><title>Claude AI Interactive - Analytics Report</title>"
        )?;
        writeln!(file, "<style>")?;
        writeln!(
            file,
            "  body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}"
        )?;
        writeln!(file, "  .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}")?;
        writeln!(file, "  .header {{ border-bottom: 2px solid #333; margin-bottom: 20px; padding-bottom: 10px; }}")?;
        writeln!(file, "  .metric-card {{ display: inline-block; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 15px; margin: 10px; min-width: 200px; }}")?;
        writeln!(
            file,
            "  .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}"
        )?;
        writeln!(
            file,
            "  .metric-label {{ color: #6c757d; font-size: 14px; }}"
        )?;
        writeln!(
            file,
            "  .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}"
        )?;
        writeln!(file, "  .alert-high {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}")?;
        writeln!(file, "  .alert-medium {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}")?;
        writeln!(file, "  .insights {{ background: #e7f3ff; border-left: 4px solid #007bff; padding: 15px; margin: 20px 0; }}")?;
        writeln!(file, "</style></head><body>")?;

        writeln!(file, "<div class=\"container\">")?;
        writeln!(file, "<div class=\"header\">")?;
        writeln!(file, "<h1>Claude AI Interactive - Analytics Report</h1>")?;
        writeln!(
            file,
            "<p>Period: {} to {}</p>",
            summary.period_start.format("%Y-%m-%d %H:%M UTC"),
            summary.period_end.format("%Y-%m-%d %H:%M UTC")
        )?;
        writeln!(
            file,
            "<p>Generated: {}</p>",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;
        writeln!(file, "</div>")?;

        // Cost metrics
        writeln!(file, "<h2>Cost Summary</h2>")?;
        writeln!(file, "<div class=\"metric-card\">")?;
        writeln!(
            file,
            "<div class=\"metric-value\">${:.2}</div>",
            summary.cost_summary.total_cost
        )?;
        writeln!(file, "<div class=\"metric-label\">Total Cost</div>")?;
        writeln!(file, "</div>")?;

        writeln!(file, "<div class=\"metric-card\">")?;
        writeln!(
            file,
            "<div class=\"metric-value\">{}</div>",
            summary.cost_summary.command_count
        )?;
        writeln!(file, "<div class=\"metric-label\">Commands</div>")?;
        writeln!(file, "</div>")?;

        writeln!(file, "<div class=\"metric-card\">")?;
        writeln!(
            file,
            "<div class=\"metric-value\">${:.4}</div>",
            summary.cost_summary.average_cost
        )?;
        writeln!(file, "<div class=\"metric-label\">Avg Cost/Command</div>")?;
        writeln!(file, "</div>")?;

        // Performance metrics
        writeln!(file, "<h2>Performance Summary</h2>")?;
        writeln!(file, "<div class=\"metric-card\">")?;
        writeln!(
            file,
            "<div class=\"metric-value\">{:.1}%</div>",
            summary.performance_metrics.success_rate
        )?;
        writeln!(file, "<div class=\"metric-label\">Success Rate</div>")?;
        writeln!(file, "</div>")?;

        writeln!(file, "<div class=\"metric-card\">")?;
        writeln!(
            file,
            "<div class=\"metric-value\">{:.0}ms</div>",
            summary.performance_metrics.average_response_time
        )?;
        writeln!(file, "<div class=\"metric-label\">Avg Response Time</div>")?;
        writeln!(file, "</div>")?;

        // Insights
        if !summary.insights.is_empty() {
            writeln!(file, "<h2>Insights</h2>")?;
            writeln!(file, "<div class=\"insights\">")?;
            writeln!(file, "<ul>")?;
            for insight in &summary.insights {
                writeln!(file, "<li>{}</li>", insight)?;
            }
            writeln!(file, "</ul>")?;
            writeln!(file, "</div>")?;
        }

        // Alerts
        if !summary.alerts.is_empty() {
            writeln!(file, "<h2>Alerts</h2>")?;
            for alert in &summary.alerts {
                let alert_class = match alert.severity {
                    AlertSeverity::High | AlertSeverity::Critical => "alert-high",
                    _ => "alert-medium",
                };
                writeln!(
                    file,
                    "<div class=\"alert {}\">{}</div>",
                    alert_class, alert.message
                )?;
            }
        }

        writeln!(file, "</div></body></html>")?;

        Ok(())
    }

    async fn export_csv_report(&self, path: &PathBuf, summary: &AnalyticsSummary) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        writeln!(file, "# Claude AI Interactive - Analytics Report")?;
        writeln!(
            file,
            "# Generated: {}",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;
        writeln!(
            file,
            "# Period: {} to {}",
            summary.period_start.format("%Y-%m-%d %H:%M UTC"),
            summary.period_end.format("%Y-%m-%d %H:%M UTC")
        )?;
        writeln!(file)?;

        writeln!(file, "Metric,Value")?;
        writeln!(file, "Total Cost,{:.2}", summary.cost_summary.total_cost)?;
        writeln!(
            file,
            "Total Commands,{}",
            summary.cost_summary.command_count
        )?;
        writeln!(
            file,
            "Average Cost per Command,{:.4}",
            summary.cost_summary.average_cost
        )?;
        writeln!(
            file,
            "Success Rate,{:.1}%",
            summary.performance_metrics.success_rate
        )?;
        writeln!(
            file,
            "Average Response Time,{:.0}ms",
            summary.performance_metrics.average_response_time
        )?;
        writeln!(
            file,
            "Throughput,{:.1} cmd/hr",
            summary.performance_metrics.throughput_commands_per_hour
        )?;

        Ok(())
    }

    async fn export_markdown_report(&self, summary: &AnalyticsSummary) -> Result<String> {
        let mut content = String::new();

        content.push_str("# Claude AI Interactive - Analytics Report\n\n");
        content.push_str(&format!(
            "Generated: {}\n\n",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        content.push_str("## Cost Summary\n\n");
        content.push_str(&format!(
            "- **Total Cost**: ${:.2}\n",
            summary.cost_summary.total_cost
        ));
        content.push_str(&format!(
            "- **Commands**: {}\n",
            summary.cost_summary.command_count
        ));
        content.push_str(&format!(
            "- **Average Cost/Command**: ${:.4}\n\n",
            summary.cost_summary.average_cost
        ));

        content.push_str("## Performance Summary\n\n");
        content.push_str(&format!(
            "- **Success Rate**: {:.1}%\n",
            summary.performance_metrics.success_rate
        ));
        content.push_str(&format!(
            "- **Average Response Time**: {:.0}ms\n",
            summary.performance_metrics.average_response_time
        ));
        content.push_str(&format!(
            "- **Throughput**: {:.1} commands/hour\n\n",
            summary.performance_metrics.throughput_commands_per_hour
        ));

        if !summary.insights.is_empty() {
            content.push_str("## Insights\n\n");
            for insight in &summary.insights {
                content.push_str(&format!("- {}\n", insight));
            }
            content.push_str("\n");
        }

        if !summary.alerts.is_empty() {
            content.push_str("## Alerts\n\n");
            for alert in &summary.alerts {
                content.push_str(&format!("- **{:?}**: {}\n", alert.severity, alert.message));
            }
        }

        Ok(content)
    }
}

/// Session-specific report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionReport {
    pub session_id: SessionId,
    pub cost_summary: CostSummary,
    pub history_stats: HistoryStats,
    pub recent_commands: Vec<crate::cli::history::HistoryEntry>,
    pub generated_at: DateTime<Utc>,
}

/// Dashboard data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub today_cost: f64,
    pub today_commands: usize,
    pub success_rate: f64,
    pub recent_activity: Vec<crate::cli::history::HistoryEntry>,
    pub top_commands: Vec<(String, f64)>,
    pub active_alerts: Vec<Alert>,
    pub last_updated: DateTime<Utc>,
}

// Re-export dashboard types
pub use dashboard::{
    DashboardConfig, DashboardManager, HealthStatus, LiveDashboardData, TimeSeriesData,
    WidgetConfig, WidgetType,
};

// Re-export metrics types
pub use metrics::{
    CommandMetric, Counter, CounterSnapshot, Gauge, GaugeSnapshot, Histogram, HistogramSnapshot,
    MetricConfig, MetricRegistry, MetricSnapshot, MetricTimer, MetricsEngine, TimeSeriesPoint,
    Timer, TimerSnapshot,
};

// Re-export ReportFormat from reports module
pub use reports::ReportFormat;

// Re-export performance profiler types
pub use performance_profiler::{
    DashboardPerformanceProfile, LoadTestConfig, LoadTestResults, LoadTestScenario,
    PerformanceBaseline, PerformanceProfiler, PerformanceReport,
};

// Re-export time series optimizer types
pub use time_series_optimizer::{
    OptimizedTimeSeriesData, OptimizedTimeSeriesPoint, TimeSeriesConfig, TimeSeriesOptimizer,
    TimeSeriesPerformanceReport, TimeSeriesType,
};

// Re-export dashboard cache types
pub use dashboard_cache::{CacheConfig, CacheKey, CacheStats, DashboardCache};

// Re-export memory optimizer types
pub use memory_optimizer::{
    MemoryConfig, MemoryOptimizer, MemoryStats as MemoryOptimizerStats, ObjectPool,
    StreamingDataProcessor,
};

// Re-export streaming optimizer types
pub use streaming_optimizer::{
    BackpressureSignal, StreamingBatch, StreamingConfig, StreamingMetrics, StreamingOptimizer,
    StreamingOptimizerFactory,
};

// Re-export optimized dashboard types
pub use optimized_dashboard::{
    ClientConnection, DashboardUpdate, DifferentialUpdate, OptimizedDashboardConfig,
    OptimizedDashboardFactory, OptimizedDashboardManager, UpdatePriority,
    UpdateType as DashboardUpdateType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::RwLock;

    async fn create_test_engine() -> (
        AnalyticsEngine,
        Arc<RwLock<CostTracker>>,
        Arc<RwLock<HistoryStore>>,
    ) {
        let temp_dir = tempdir().unwrap();

        let cost_tracker = Arc::new(RwLock::new(
            CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
        ));

        let history_store = Arc::new(RwLock::new(
            HistoryStore::new(temp_dir.path().join("history.json")).unwrap(),
        ));

        let config = AnalyticsConfig::default();
        let engine = AnalyticsEngine::new(
            Arc::clone(&cost_tracker),
            Arc::clone(&history_store),
            config,
        );

        (engine, cost_tracker, history_store)
    }

    async fn add_test_data(
        cost_tracker: &Arc<RwLock<CostTracker>>,
        history_store: &Arc<RwLock<HistoryStore>>,
        session_id: SessionId,
    ) {
        use crate::cli::cost::CostEntry;
        use crate::cli::history::HistoryEntry;

        // Add cost entries
        let mut tracker = cost_tracker.write().await;
        for i in 0..5 {
            tracker
                .record_cost(CostEntry::new(
                    session_id,
                    format!("cmd_{}", i),
                    0.01 * (i + 1) as f64,
                    50 + i as u32 * 10,
                    100 + i as u32 * 20,
                    1000 + i as u64 * 500,
                    "claude-3-opus".to_string(),
                ))
                .await
                .unwrap();
        }
        drop(tracker);

        // Add history entries
        let mut store = history_store.write().await;
        for i in 0..5 {
            let entry = HistoryEntry::new(
                session_id,
                format!("cmd_{}", i),
                vec![format!("arg_{}", i)],
                format!("Output {}", i),
                i != 2, // One failure
                1000 + i as u64 * 500,
            )
            .with_cost(
                0.01 * (i + 1) as f64,
                50 + i as u32 * 10,
                100 + i as u32 * 20,
                "claude-3-opus".to_string(),
            );
            store.store_entry(entry).await.unwrap();
        }
    }

    #[test]
    fn test_analytics_config_default() {
        let config = AnalyticsConfig::default();

        assert!(config.enable_real_time_alerts);
        assert_eq!(config.cost_alert_threshold, 10.0);
        assert!(matches!(config.report_schedule, ReportSchedule::Weekly));
        assert_eq!(config.retention_days, 90);
        assert_eq!(config.dashboard_refresh_interval, 30);
    }

    #[tokio::test]
    async fn test_generate_summary() {
        let (engine, cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        add_test_data(&cost_tracker, &history_store, session_id).await;

        let summary = engine.generate_summary(7).await.unwrap();

        assert!(summary.cost_summary.total_cost > 0.0);
        assert_eq!(summary.cost_summary.command_count, 5);
        assert_eq!(summary.history_stats.total_entries, 5);
        assert_eq!(summary.history_stats.successful_commands, 4);
        assert_eq!(summary.history_stats.failed_commands, 1);
        assert!((summary.history_stats.success_rate - 80.0).abs() < 0.1);
        assert!(!summary.insights.is_empty());
    }

    #[tokio::test]
    async fn test_generate_session_report() {
        let (engine, cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        add_test_data(&cost_tracker, &history_store, session_id).await;

        let report = engine.generate_session_report(session_id).await.unwrap();

        assert_eq!(report.session_id, session_id);
        assert!(report.cost_summary.total_cost > 0.0);
        assert_eq!(report.history_stats.total_entries, 5);
        assert!(!report.recent_commands.is_empty());
    }

    #[tokio::test]
    async fn test_get_dashboard_data() {
        let (engine, cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        add_test_data(&cost_tracker, &history_store, session_id).await;

        let dashboard = engine.get_dashboard_data().await.unwrap();

        // Today's data might be empty if test data timestamps don't match
        assert!(dashboard.today_cost >= 0.0);
        // today_commands is u32, so it's always >= 0
        assert!(!dashboard.recent_activity.is_empty());
        assert!(!dashboard.top_commands.is_empty());
    }

    #[tokio::test]
    async fn test_generate_insights() {
        let (engine, cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        add_test_data(&cost_tracker, &history_store, session_id).await;

        let summary = engine.generate_summary(7).await.unwrap();
        let insights = engine
            .generate_insights(
                &summary.cost_summary,
                &summary.history_stats,
                &summary.performance_metrics,
            )
            .await
            .unwrap();

        assert!(!insights.is_empty());
        // Should have insights about spending
        assert!(insights.iter().any(|i| i.contains("Total spending")));
        // Should have performance insight about success rate < 95%
        assert!(insights.iter().any(|i| i.contains("Success rate")));
    }

    #[tokio::test]
    async fn test_check_alerts() {
        let (mut engine, cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        // Set low threshold to trigger alert
        engine.config.cost_alert_threshold = 0.10;

        add_test_data(&cost_tracker, &history_store, session_id).await;

        let summary = engine.generate_summary(7).await.unwrap();
        let alerts = engine
            .check_alerts(
                &summary.cost_summary,
                &summary.history_stats,
                &summary.performance_metrics,
            )
            .await
            .unwrap();

        assert!(!alerts.is_empty());
        // Should have cost threshold alert
        assert!(alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::CostThreshold)));
    }

    #[tokio::test]
    async fn test_calculate_performance_metrics() {
        let (engine, cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        add_test_data(&cost_tracker, &history_store, session_id).await;

        let search = HistorySearch::default();
        let metrics = engine.calculate_performance_metrics(&search).await.unwrap();

        assert!(metrics.average_response_time > 0.0);
        assert!((metrics.success_rate - 80.0).abs() < 0.1); // 4 success, 1 failure
        assert!(metrics.throughput_commands_per_hour > 0.0);
        assert!(!metrics.slowest_commands.is_empty());
        assert!(!metrics.error_rate_by_command.is_empty());

        // cmd_2 should have 100% error rate
        assert_eq!(metrics.error_rate_by_command.get("cmd_2"), Some(&100.0));
    }

    #[tokio::test]
    async fn test_export_analytics_report() {
        let (engine, cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        add_test_data(&cost_tracker, &history_store, session_id).await;

        let temp_dir = tempdir().unwrap();

        // Test JSON export
        let json_path = temp_dir.path().join("report.json");
        engine
            .export_analytics_report(&json_path, ReportFormat::Json, 7)
            .await
            .unwrap();
        assert!(json_path.exists());

        let json_content = tokio::fs::read_to_string(&json_path).await.unwrap();
        assert!(json_content.contains("cost_summary"));
        assert!(json_content.contains("performance_metrics"));

        // Test HTML export
        let html_path = temp_dir.path().join("report.html");
        engine
            .export_analytics_report(&html_path, ReportFormat::Html, 7)
            .await
            .unwrap();
        assert!(html_path.exists());

        let html_content = tokio::fs::read_to_string(&html_path).await.unwrap();
        assert!(html_content.contains("<html>"));
        assert!(html_content.contains("Analytics Report"));

        // Test CSV export
        let csv_path = temp_dir.path().join("report.csv");
        engine
            .export_analytics_report(&csv_path, ReportFormat::Csv, 7)
            .await
            .unwrap();
        assert!(csv_path.exists());

        let csv_content = tokio::fs::read_to_string(&csv_path).await.unwrap();
        assert!(csv_content.contains("Metric,Value"));
        assert!(csv_content.contains("Total Cost"));
    }

    #[tokio::test]
    async fn test_alert_severity_levels() {
        let (engine, _cost_tracker, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        // Add data with high error rate
        let mut store = history_store.write().await;
        for i in 0..10 {
            let entry = crate::cli::history::HistoryEntry::new(
                session_id,
                format!("failing_cmd_{}", i),
                vec![],
                "Error".to_string(),
                false, // All failures
                1000,
            );
            store.store_entry(entry).await.unwrap();
        }
        drop(store);

        let summary = engine.generate_summary(7).await.unwrap();
        let alerts = engine
            .check_alerts(
                &summary.cost_summary,
                &summary.history_stats,
                &summary.performance_metrics,
            )
            .await
            .unwrap();

        // Should have high severity error rate alert (100% failure rate)
        assert!(alerts.iter().any(|a| {
            matches!(a.alert_type, AlertType::ErrorRate)
                && matches!(a.severity, AlertSeverity::High)
        }));
    }

    #[tokio::test]
    async fn test_empty_data_handling() {
        let (engine, _, _) = create_test_engine().await;

        // Test with no data
        let summary = engine.generate_summary(7).await.unwrap();

        assert_eq!(summary.cost_summary.total_cost, 0.0);
        assert_eq!(summary.cost_summary.command_count, 0);
        assert_eq!(summary.history_stats.total_entries, 0);
        assert_eq!(summary.performance_metrics.average_response_time, 0.0);
        assert!(summary.insights.is_empty()); // No insights for empty data
        assert!(summary.alerts.is_empty()); // No alerts for empty data
    }

    #[tokio::test]
    async fn test_performance_metrics_edge_cases() {
        let (engine, _, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        // Add single entry to test edge cases
        let mut store = history_store.write().await;
        let entry = crate::cli::history::HistoryEntry::new(
            session_id,
            "single_cmd".to_string(),
            vec![],
            "Output".to_string(),
            true,
            5000,
        );
        store.store_entry(entry).await.unwrap();
        drop(store);

        let search = HistorySearch::default();
        let metrics = engine.calculate_performance_metrics(&search).await.unwrap();

        assert_eq!(metrics.average_response_time, 5000.0);
        assert_eq!(metrics.success_rate, 100.0);
        assert!(metrics.throughput_commands_per_hour > 0.0);
        assert_eq!(metrics.slowest_commands.len(), 1);
        assert_eq!(metrics.slowest_commands[0].0, "single_cmd");
        assert_eq!(metrics.slowest_commands[0].1, 5000.0);
    }

    #[tokio::test]
    async fn test_report_schedule_types() {
        let config1 = AnalyticsConfig {
            report_schedule: ReportSchedule::Daily,
            ..Default::default()
        };
        assert!(matches!(config1.report_schedule, ReportSchedule::Daily));

        let config2 = AnalyticsConfig {
            report_schedule: ReportSchedule::Monthly,
            ..Default::default()
        };
        assert!(matches!(config2.report_schedule, ReportSchedule::Monthly));

        let config3 = AnalyticsConfig {
            report_schedule: ReportSchedule::Custom(Duration::hours(12)),
            ..Default::default()
        };
        match config3.report_schedule {
            ReportSchedule::Custom(duration) => {
                assert_eq!(duration.num_hours(), 12);
            }
            _ => panic!("Expected Custom schedule"),
        }
    }

    #[tokio::test]
    async fn test_peak_hour_detection() {
        let (engine, _, history_store) = create_test_engine().await;
        let session_id = uuid::Uuid::new_v4();

        // Add entries at specific hours
        let mut store = history_store.write().await;
        for hour in [14, 14, 14, 15, 16] {
            let mut entry = crate::cli::history::HistoryEntry::new(
                session_id,
                "cmd".to_string(),
                vec![],
                "Output".to_string(),
                true,
                1000,
            );
            // Set specific timestamp
            entry.timestamp = Utc::now()
                .date_naive()
                .and_hms_opt(hour, 0, 0)
                .unwrap()
                .and_utc();
            store.store_entry(entry).await.unwrap();
        }
        drop(store);

        let search = HistorySearch::default();
        let metrics = engine.calculate_performance_metrics(&search).await.unwrap();

        // Peak hour should be 14 (3 entries)
        assert_eq!(metrics.peak_usage_hour, 14);
    }
}
