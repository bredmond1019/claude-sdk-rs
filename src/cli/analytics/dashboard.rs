//! Real-time dashboard functionality for analytics visualization
//!
//! This module provides comprehensive dashboard capabilities for monitoring
//! Claude AI Interactive usage, costs, and performance in real-time.
//!
//! # Features
//!
//! - **Live Updates**: Real-time data streaming via async channels
//! - **Customizable Widgets**: Modular widget system for flexible layouts
//! - **Time Series Visualization**: Historical data tracking and charting
//! - **System Health Monitoring**: Track system status and resource usage
//! - **Interactive Controls**: Dynamic filtering and time range selection
//!
//! # Architecture
//!
//! The dashboard system follows a pub-sub pattern where:
//! 1. Data sources publish updates to the dashboard manager
//! 2. Dashboard manager aggregates and transforms data
//! 3. Subscribers receive formatted updates for display
//!
//! # Example
//!
//! ```no_run
//! use crate_interactive::analytics::{DashboardManager, DashboardConfig};
//!
//! # async fn example(analytics_engine: std::sync::Arc<AnalyticsEngine>) -> Result<(), Box<dyn std::error::Error>> {
//! // Create dashboard with custom config
//! let config = DashboardConfig {
//!     refresh_interval_seconds: 5,
//!     max_recent_entries: 50,
//!     enable_live_updates: true,
//!     chart_time_range_hours: 24,
//! };
//!
//! let dashboard = DashboardManager::new(analytics_engine, config);
//!
//! // Start live updates
//! dashboard.start_live_updates().await?;
//!
//! // Get current dashboard data
//! let data = dashboard.get_live_data().await?;
//! println!("Today's cost: ${:.2}", data.current_metrics.today_cost);
//! # Ok(())
//! # }
//! ```

use super::{AnalyticsEngine, DashboardData};
use crate::cli::error::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use sysinfo::{Disks, System};
use tokio::sync::broadcast;

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub refresh_interval_seconds: u64,
    pub max_recent_entries: usize,
    pub enable_live_updates: bool,
    pub chart_time_range_hours: u32,
    pub enable_real_system_monitoring: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            refresh_interval_seconds: 30,
            max_recent_entries: 20,
            enable_live_updates: true,
            chart_time_range_hours: 24,
            enable_real_system_monitoring: true,
        }
    }
}

/// Real-time dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveDashboardData {
    pub timestamp: DateTime<Utc>,
    pub current_metrics: DashboardData,
    pub time_series: TimeSeriesData,
    pub system_status: SystemStatus,
}

/// Time series data for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub cost_over_time: Vec<(DateTime<Utc>, f64)>,
    pub commands_over_time: Vec<(DateTime<Utc>, usize)>,
    pub success_rate_over_time: Vec<(DateTime<Utc>, f64)>,
    pub response_time_over_time: Vec<(DateTime<Utc>, f64)>,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub health: HealthStatus,
    pub uptime_hours: f64,
    pub active_sessions: usize,
    pub memory_usage_mb: f64,
    pub disk_usage_percent: f64,
    pub last_error: Option<DateTime<Utc>>,
}

/// Health status levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Dashboard widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub widget_type: WidgetType,
    pub title: String,
    pub position: (u32, u32), // (row, column)
    pub size: (u32, u32),     // (width, height)
    pub refresh_interval: Option<u64>,
    pub settings: HashMap<String, serde_json::Value>,
}

/// Available widget types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    CostSummary,
    CommandActivity,
    SuccessRate,
    ResponseTime,
    TopCommands,
    RecentAlerts,
    SessionList,
    ErrorLog,
    CustomChart,
}

/// Dashboard layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    pub name: String,
    pub widgets: Vec<WidgetConfig>,
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
}

/// Live dashboard manager
pub struct DashboardManager {
    analytics_engine: AnalyticsEngine,
    config: DashboardConfig,
    layouts: HashMap<String, DashboardLayout>,
    update_sender: broadcast::Sender<LiveDashboardData>,
    _update_receiver: broadcast::Receiver<LiveDashboardData>,
    cache: Option<Arc<super::dashboard_cache::DashboardCache>>,
}

impl DashboardManager {
    /// Create a new dashboard manager
    pub fn new(analytics_engine: AnalyticsEngine, config: DashboardConfig) -> Self {
        let (update_sender, update_receiver) = broadcast::channel(100);

        Self {
            analytics_engine,
            config,
            layouts: HashMap::new(),
            update_sender,
            _update_receiver: update_receiver,
            cache: None,
        }
    }

    /// Create a new dashboard manager with caching enabled
    pub fn with_cache(
        analytics_engine: AnalyticsEngine,
        config: DashboardConfig,
        cache_config: super::dashboard_cache::CacheConfig,
    ) -> Self {
        let (update_sender, update_receiver) = broadcast::channel(100);
        let cache = Arc::new(super::dashboard_cache::DashboardCache::new(cache_config));

        Self {
            analytics_engine,
            config,
            layouts: HashMap::new(),
            update_sender,
            _update_receiver: update_receiver,
            cache: Some(cache),
        }
    }

    /// Start the dashboard update loop
    pub async fn start_update_loop(&self) -> Result<()> {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
            self.config.refresh_interval_seconds,
        ));

        loop {
            interval.tick().await;

            if let Ok(dashboard_data) = self.generate_live_data().await {
                // Send update to subscribers
                let _ = self.update_sender.send(dashboard_data);
            }
        }
    }

    /// Generate live dashboard data
    pub async fn generate_live_data(&self) -> Result<LiveDashboardData> {
        use super::dashboard_cache::CacheKey;

        // Check cache first if caching is enabled
        if let Some(cache) = &self.cache {
            let cache_key = CacheKey::live_dashboard_data();
            if let Some(cached_data) = cache.get_live_dashboard_data(&cache_key).await {
                return Ok(cached_data);
            }
        }

        // Generate fresh data
        let current_metrics = self.analytics_engine.get_dashboard_data().await?;
        let time_series = self.generate_time_series().await?;
        let system_status = self.get_system_status().await?;

        let live_data = LiveDashboardData {
            timestamp: Utc::now(),
            current_metrics,
            time_series,
            system_status,
        };

        // Cache the result if caching is enabled
        if let Some(cache) = &self.cache {
            let cache_key = CacheKey::live_dashboard_data();
            cache
                .set_live_dashboard_data(cache_key, live_data.clone(), Some(30))
                .await; // 30 second TTL
        }

        Ok(live_data)
    }

    /// Subscribe to live dashboard updates
    pub fn subscribe_to_updates(&self) -> broadcast::Receiver<LiveDashboardData> {
        self.update_sender.subscribe()
    }

    /// Create a custom dashboard layout
    pub fn create_layout(&mut self, name: String, widgets: Vec<WidgetConfig>) -> Result<()> {
        let layout = DashboardLayout {
            name: name.clone(),
            widgets,
            created_at: Utc::now(),
            last_modified: Utc::now(),
        };

        self.layouts.insert(name, layout);
        Ok(())
    }

    /// Get available dashboard layouts
    pub fn get_layouts(&self) -> Vec<&DashboardLayout> {
        self.layouts.values().collect()
    }

    /// Generate default dashboard layout
    pub fn create_default_layout(&mut self) -> Result<()> {
        let widgets = vec![
            WidgetConfig {
                widget_type: WidgetType::CostSummary,
                title: "Cost Overview".to_string(),
                position: (0, 0),
                size: (2, 1),
                refresh_interval: None,
                settings: HashMap::new(),
            },
            WidgetConfig {
                widget_type: WidgetType::CommandActivity,
                title: "Command Activity".to_string(),
                position: (0, 2),
                size: (2, 1),
                refresh_interval: None,
                settings: HashMap::new(),
            },
            WidgetConfig {
                widget_type: WidgetType::SuccessRate,
                title: "Success Rate".to_string(),
                position: (1, 0),
                size: (1, 1),
                refresh_interval: None,
                settings: HashMap::new(),
            },
            WidgetConfig {
                widget_type: WidgetType::ResponseTime,
                title: "Response Time".to_string(),
                position: (1, 1),
                size: (1, 1),
                refresh_interval: None,
                settings: HashMap::new(),
            },
            WidgetConfig {
                widget_type: WidgetType::TopCommands,
                title: "Top Commands".to_string(),
                position: (2, 0),
                size: (2, 1),
                refresh_interval: None,
                settings: HashMap::new(),
            },
            WidgetConfig {
                widget_type: WidgetType::RecentAlerts,
                title: "Recent Alerts".to_string(),
                position: (2, 2),
                size: (2, 1),
                refresh_interval: None,
                settings: HashMap::new(),
            },
        ];

        self.create_layout("default".to_string(), widgets)
    }

    /// Export dashboard data for external visualization
    pub async fn export_dashboard_data(&self, format: DashboardExportFormat) -> Result<String> {
        let data = self.generate_live_data().await?;

        match format {
            DashboardExportFormat::Json => Ok(serde_json::to_string_pretty(&data)?),
            DashboardExportFormat::Prometheus => self.export_prometheus_metrics(&data).await,
            DashboardExportFormat::InfluxDB => self.export_influxdb_line_protocol(&data).await,
        }
    }

    /// Generate chart data for specific metrics
    pub async fn generate_chart_data(
        &self,
        metric: ChartMetric,
        hours: u32,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        let start_time = Utc::now() - Duration::hours(hours as i64);
        let end_time = Utc::now();

        match metric {
            ChartMetric::CostOverTime => {
                self.generate_cost_time_series(start_time, end_time, hours)
                    .await
            }
            ChartMetric::CommandsOverTime => {
                self.generate_commands_time_series(start_time, end_time, hours)
                    .await
            }
            ChartMetric::SuccessRate => {
                self.generate_success_rate_time_series(start_time, end_time, hours)
                    .await
            }
            ChartMetric::ResponseTime => {
                self.generate_response_time_time_series(start_time, end_time, hours)
                    .await
            }
        }
    }

    // Private helper methods

    async fn generate_time_series(&self) -> Result<TimeSeriesData> {
        let hours = self.config.chart_time_range_hours;
        let start_time = Utc::now() - Duration::hours(hours as i64);
        let end_time = Utc::now();

        // Use optimized time series generation
        use super::time_series_optimizer::{TimeSeriesOptimizer, TimeSeriesType};

        let optimizer = TimeSeriesOptimizer::new(self.analytics_engine.clone());
        let types = vec![
            TimeSeriesType::Cost,
            TimeSeriesType::Commands,
            TimeSeriesType::SuccessRate,
            TimeSeriesType::ResponseTime,
        ];

        let optimized_data = optimizer
            .generate_optimized_time_series(start_time, end_time, types)
            .await?;

        // Convert to legacy format for compatibility
        Ok(TimeSeriesOptimizer::to_legacy_format(&optimized_data))
    }

    async fn get_system_status(&self) -> Result<SystemStatus> {
        // Get actual system information where possible
        let mut health = HealthStatus::Healthy;
        let mut last_error = None;

        // Check for recent errors in the last hour
        let one_hour_ago = Utc::now() - Duration::hours(1);
        let recent_search = crate::cli::history::HistorySearch {
            since: Some(one_hour_ago),
            until: Some(Utc::now()),
            ..Default::default()
        };

        let recent_entries = self
            .analytics_engine
            .history_store
            .read()
            .await
            .search(&recent_search)
            .await?;

        let error_count = recent_entries.iter().filter(|e| !e.success).count();
        let total_count = recent_entries.len();

        // Determine health based on error rate
        if total_count > 0 {
            let error_rate = (error_count as f64 / total_count as f64) * 100.0;
            health = match error_rate {
                r if r > 50.0 => HealthStatus::Critical,
                r if r > 25.0 => HealthStatus::Warning,
                _ => HealthStatus::Healthy,
            };

            // Set last error if there were any
            if error_count > 0 {
                last_error = recent_entries
                    .iter()
                    .filter(|e| !e.success)
                    .map(|e| e.timestamp)
                    .max();
            }
        }

        // Get real system metrics if enabled, otherwise use "no data available" state
        let (memory_usage_mb, uptime_hours, disk_usage_percent) =
            if self.config.enable_real_system_monitoring {
                self.get_real_system_metrics().await
            } else {
                // Return clear "no data available" indicators
                (f64::NAN, f64::NAN, f64::NAN)
            };

        // Count active sessions (unique sessions in recent data)
        let active_sessions = recent_entries
            .iter()
            .map(|e| e.session_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(SystemStatus {
            health,
            uptime_hours,
            active_sessions,
            memory_usage_mb,
            disk_usage_percent,
            last_error,
        })
    }

    /// Get real system metrics using sysinfo crate
    async fn get_real_system_metrics(&self) -> (f64, f64, f64) {
        // Use tokio::task::spawn_blocking to handle potentially blocking sysinfo calls
        tokio::task::spawn_blocking(|| {
            let mut system = System::new_all();
            system.refresh_all();

            // Get memory usage in MB
            let used_memory = system.used_memory() as f64 / 1024.0 / 1024.0; // Convert bytes to MB

            // Get system uptime in hours
            let uptime_seconds = System::uptime();
            let uptime_hours = uptime_seconds as f64 / 3600.0;

            // Get disk usage percentage for the root disk
            let disks = Disks::new_with_refreshed_list();
            let disk_usage_percent = disks
                .first()
                .map(|disk| {
                    let total = disk.total_space();
                    let available = disk.available_space();
                    if total > 0 {
                        ((total - available) as f64 / total as f64) * 100.0
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);

            (used_memory, uptime_hours, disk_usage_percent)
        })
        .await
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN)) // Fallback on error
    }

    async fn export_prometheus_metrics(&self, data: &LiveDashboardData) -> Result<String> {
        let mut metrics = String::new();

        metrics.push_str(&format!(
            "claude_interactive_cost_total {:.2}\n",
            data.current_metrics.today_cost
        ));

        metrics.push_str(&format!(
            "claude_interactive_commands_total {}\n",
            data.current_metrics.today_commands
        ));

        metrics.push_str(&format!(
            "claude_interactive_success_rate {:.2}\n",
            data.current_metrics.success_rate
        ));

        metrics.push_str(&format!(
            "claude_interactive_active_sessions {}\n",
            data.system_status.active_sessions
        ));

        metrics.push_str(&format!(
            "claude_interactive_memory_usage_mb {:.2}\n",
            data.system_status.memory_usage_mb
        ));

        Ok(metrics)
    }

    async fn export_influxdb_line_protocol(&self, data: &LiveDashboardData) -> Result<String> {
        let timestamp = data.timestamp.timestamp_nanos_opt().unwrap_or(0);
        let mut lines = String::new();

        lines.push_str(&format!(
            "claude_interactive,metric=cost value={:.2} {}\n",
            data.current_metrics.today_cost, timestamp
        ));

        lines.push_str(&format!(
            "claude_interactive,metric=commands value={} {}\n",
            data.current_metrics.today_commands, timestamp
        ));

        lines.push_str(&format!(
            "claude_interactive,metric=success_rate value={:.2} {}\n",
            data.current_metrics.success_rate, timestamp
        ));

        Ok(lines)
    }

    /// Generate cost time series data
    async fn generate_cost_time_series(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        _hours: u32,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        use crate::cli::cost::CostFilter;

        let mut data = Vec::new();
        let interval = Duration::hours(1);
        let mut current_time = start_time;

        while current_time < end_time {
            let next_time = current_time + interval;

            // Query cost data for this hour
            let filter = CostFilter {
                since: Some(current_time),
                until: Some(next_time),
                ..Default::default()
            };

            let cost_summary = self
                .analytics_engine
                .cost_tracker
                .read()
                .await
                .get_filtered_summary(&filter)
                .await?;

            data.push((current_time, cost_summary.total_cost));
            current_time = next_time;
        }

        Ok(data)
    }

    /// Generate commands count time series data
    async fn generate_commands_time_series(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        _hours: u32,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        use crate::cli::history::HistorySearch;

        let mut data = Vec::new();
        let interval = Duration::hours(1);
        let mut current_time = start_time;

        while current_time < end_time {
            let next_time = current_time + interval;

            // Query history data for this hour
            let search = HistorySearch {
                since: Some(current_time),
                until: Some(next_time),
                ..Default::default()
            };

            let entries = self
                .analytics_engine
                .history_store
                .read()
                .await
                .search(&search)
                .await?;

            data.push((current_time, entries.len() as f64));
            current_time = next_time;
        }

        Ok(data)
    }

    /// Generate success rate time series data
    async fn generate_success_rate_time_series(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        _hours: u32,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        use crate::cli::history::HistorySearch;

        let mut data = Vec::new();
        let interval = Duration::hours(1);
        let mut current_time = start_time;

        while current_time < end_time {
            let next_time = current_time + interval;

            // Query history data for this hour
            let search = HistorySearch {
                since: Some(current_time),
                until: Some(next_time),
                ..Default::default()
            };

            let entries = self
                .analytics_engine
                .history_store
                .read()
                .await
                .search(&search)
                .await?;

            let success_rate = if entries.is_empty() {
                f64::NAN // No data available - return NaN instead of misleading 100%
            } else {
                let successful = entries.iter().filter(|e| e.success).count();
                (successful as f64 / entries.len() as f64) * 100.0
            };

            data.push((current_time, success_rate));
            current_time = next_time;
        }

        Ok(data)
    }

    /// Generate response time time series data
    async fn generate_response_time_time_series(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        _hours: u32,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        use crate::cli::history::HistorySearch;

        let mut data = Vec::new();
        let interval = Duration::hours(1);
        let mut current_time = start_time;

        while current_time < end_time {
            let next_time = current_time + interval;

            // Query history data for this hour
            let search = HistorySearch {
                since: Some(current_time),
                until: Some(next_time),
                ..Default::default()
            };

            let entries = self
                .analytics_engine
                .history_store
                .read()
                .await
                .search(&search)
                .await?;

            let avg_response_time = if entries.is_empty() {
                f64::NAN // No data available - return NaN instead of misleading default
            } else {
                let total_duration: u64 = entries.iter().map(|e| e.duration_ms).sum();
                total_duration as f64 / entries.len() as f64
            };

            data.push((current_time, avg_response_time));
            current_time = next_time;
        }

        Ok(data)
    }
}

/// Chart metric types
#[derive(Debug, Clone)]
pub enum ChartMetric {
    CostOverTime,
    CommandsOverTime,
    SuccessRate,
    ResponseTime,
}

/// Dashboard export formats
#[derive(Debug, Clone)]
pub enum DashboardExportFormat {
    Json,
    Prometheus,
    InfluxDB,
}
