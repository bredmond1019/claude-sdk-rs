//! # MCP Connection Health Monitoring
//!
//! This module provides comprehensive health monitoring for MCP connections,
//! including connection validation, keep-alive mechanisms, and health checks.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::{interval, timeout};
// sleep unused for now
use serde::{Deserialize, Serialize};
// use uuid::Uuid;

use crate::mcp::clients::MCPClient;
use crate::mcp::core::error::WorkflowError;
// use crate::mcp::protocol::{MCPRequest, MCPResponse};

/// Health status of an MCP connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Connection is healthy and responsive
    Healthy,
    /// Connection is degraded but functional
    Degraded,
    /// Connection is unhealthy and may be failing
    Unhealthy,
    /// Connection is disconnected
    Disconnected,
}

/// Configuration for connection health monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Interval between health checks
    pub check_interval: Duration,
    /// Timeout for health check requests
    pub check_timeout: Duration,
    /// Number of consecutive failures before marking as unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes to mark as healthy again
    pub recovery_threshold: u32,
    /// Enable keep-alive ping messages
    pub enable_keep_alive: bool,
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    /// Maximum response time for healthy status
    pub healthy_response_time: Duration,
    /// Maximum response time for degraded status
    pub degraded_response_time: Duration,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
            enable_keep_alive: true,
            keep_alive_interval: Duration::from_secs(60),
            healthy_response_time: Duration::from_millis(500),
            degraded_response_time: Duration::from_secs(2),
        }
    }
}

/// Health metrics for a connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub connection_id: String,
    pub server_id: String,
    pub status: HealthStatus,
    #[serde(skip)]
    pub last_check: Option<Instant>,
    #[serde(skip)]
    pub last_success: Option<Instant>,
    #[serde(skip)]
    pub last_failure: Option<Instant>,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub total_checks: u64,
    pub total_failures: u64,
    pub total_successes: u64,
    pub average_response_time: Duration,
    pub last_response_time: Option<Duration>,
    pub uptime_percentage: f64,
}

impl HealthMetrics {
    pub fn new(connection_id: String, server_id: String) -> Self {
        Self {
            connection_id,
            server_id,
            status: HealthStatus::Disconnected,
            last_check: None,
            last_success: None,
            last_failure: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
            total_checks: 0,
            total_failures: 0,
            total_successes: 0,
            average_response_time: Duration::from_millis(0),
            last_response_time: None,
            uptime_percentage: 0.0,
        }
    }

    pub fn update_success(&mut self, response_time: Duration) {
        let now = Instant::now();
        self.last_check = Some(now);
        self.last_success = Some(now);
        self.last_response_time = Some(response_time);
        self.consecutive_failures = 0;
        self.consecutive_successes += 1;
        self.total_checks += 1;
        self.total_successes += 1;

        // Update average response time
        let total_time = self.average_response_time.as_millis() as u64 * (self.total_successes - 1);
        self.average_response_time = Duration::from_millis(
            (total_time + response_time.as_millis() as u64) / self.total_successes,
        );

        // Update uptime percentage
        self.uptime_percentage = (self.total_successes as f64 / self.total_checks as f64) * 100.0;
    }

    pub fn update_failure(&mut self) {
        let now = Instant::now();
        self.last_check = Some(now);
        self.last_failure = Some(now);
        self.consecutive_successes = 0;
        self.consecutive_failures += 1;
        self.total_checks += 1;
        self.total_failures += 1;

        // Update uptime percentage
        self.uptime_percentage = (self.total_successes as f64 / self.total_checks as f64) * 100.0;
    }
}

/// Connection health monitor
pub struct ConnectionHealthMonitor {
    config: HealthConfig,
    metrics: Arc<RwLock<HashMap<String, HealthMetrics>>>,
    running: Arc<RwLock<bool>>,
}

impl ConnectionHealthMonitor {
    pub fn new(config: HealthConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start monitoring a connection
    pub async fn start_monitoring(&self, connection_id: String, server_id: String) {
        let mut metrics = self.metrics.write().await;
        metrics.insert(
            connection_id.clone(),
            HealthMetrics::new(connection_id, server_id),
        );
    }

    /// Stop monitoring a connection
    pub async fn stop_monitoring(&self, connection_id: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.remove(connection_id);
    }

    /// Perform a health check on a specific connection
    pub async fn check_connection_health(
        &self,
        connection_id: &str,
        client: &mut Box<dyn MCPClient>,
    ) -> Result<HealthStatus, WorkflowError> {
        let start_time = Instant::now();

        // Perform health check (ping or list tools)
        let health_check_result =
            timeout(self.config.check_timeout, self.perform_health_check(client)).await;

        let response_time = start_time.elapsed();
        let mut metrics = self.metrics.write().await;

        if let Some(metric) = metrics.get_mut(connection_id) {
            match health_check_result {
                Ok(Ok(_)) => {
                    metric.update_success(response_time);

                    // Determine status based on response time
                    let status = if response_time <= self.config.healthy_response_time {
                        HealthStatus::Healthy
                    } else if response_time <= self.config.degraded_response_time {
                        HealthStatus::Degraded
                    } else {
                        HealthStatus::Degraded
                    };

                    metric.status = status;

                    // Check if we've recovered after failures
                    if metric.consecutive_successes >= self.config.recovery_threshold {
                        metric.status = if response_time <= self.config.healthy_response_time {
                            HealthStatus::Healthy
                        } else {
                            HealthStatus::Degraded
                        };
                    }

                    Ok(metric.status)
                }
                Ok(Err(_)) | Err(_) => {
                    metric.update_failure();

                    // Determine if connection is unhealthy
                    if metric.consecutive_failures >= self.config.failure_threshold {
                        metric.status = HealthStatus::Unhealthy;
                    }

                    Ok(metric.status)
                }
            }
        } else {
            Err(WorkflowError::MCPError {
                message: format!("Connection {} not being monitored", connection_id),
            })
        }
    }

    /// Perform the actual health check operation
    async fn perform_health_check(
        &self,
        client: &mut Box<dyn MCPClient>,
    ) -> Result<(), WorkflowError> {
        if !client.is_connected() {
            return Err(WorkflowError::MCPConnectionError {
                message: "Client not connected".to_string(),
            });
        }

        // Try to list tools as a health check
        client.list_tools().await.map(|_| ())
    }

    /// Send keep-alive ping to a connection
    pub async fn send_keep_alive(
        &self,
        client: &mut Box<dyn MCPClient>,
    ) -> Result<(), WorkflowError> {
        if !self.config.enable_keep_alive {
            return Ok(());
        }

        // For keep-alive, we'll use a simple tool list call
        // In a real implementation, you might want a dedicated ping message
        client.list_tools().await.map(|_| ())
    }

    /// Get health metrics for a specific connection
    pub async fn get_connection_metrics(&self, connection_id: &str) -> Option<HealthMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(connection_id).cloned()
    }

    /// Get health metrics for all connections
    pub async fn get_all_metrics(&self) -> HashMap<String, HealthMetrics> {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Get overall health summary
    pub async fn get_health_summary(&self) -> HealthSummary {
        let metrics = self.metrics.read().await;
        let total_connections = metrics.len();
        let healthy_count = metrics
            .values()
            .filter(|m| m.status == HealthStatus::Healthy)
            .count();
        let degraded_count = metrics
            .values()
            .filter(|m| m.status == HealthStatus::Degraded)
            .count();
        let unhealthy_count = metrics
            .values()
            .filter(|m| m.status == HealthStatus::Unhealthy)
            .count();
        let disconnected_count = metrics
            .values()
            .filter(|m| m.status == HealthStatus::Disconnected)
            .count();

        let overall_uptime = if total_connections > 0 {
            metrics.values().map(|m| m.uptime_percentage).sum::<f64>() / total_connections as f64
        } else {
            0.0
        };

        HealthSummary {
            total_connections,
            healthy_connections: healthy_count,
            degraded_connections: degraded_count,
            unhealthy_connections: unhealthy_count,
            disconnected_connections: disconnected_count,
            overall_uptime_percentage: overall_uptime,
        }
    }

    /// Mark a connection as disconnected
    pub async fn mark_disconnected(&self, connection_id: &str) {
        let mut metrics = self.metrics.write().await;
        if let Some(metric) = metrics.get_mut(connection_id) {
            metric.status = HealthStatus::Disconnected;
        }
    }

    /// Start the health monitoring background task
    pub async fn start_background_monitoring(&self) -> tokio::task::JoinHandle<()> {
        let _metrics = Arc::clone(&self.metrics);
        let running = Arc::clone(&self.running);
        let config = self.config.clone();

        {
            let mut running_guard = running.write().await;
            *running_guard = true;
        }

        tokio::spawn(async move {
            let mut interval = interval(config.check_interval);

            while *running.read().await {
                interval.tick().await;

                // This would need access to the connection pool to perform actual checks
                // For now, we'll just log that monitoring is running
                tracing::debug!("Background health monitoring tick");

                // In a full implementation, this would:
                // 1. Get all active connections from the pool
                // 2. Perform health checks on each
                // 3. Update metrics accordingly
                // 4. Trigger circuit breaker actions if needed
            }
        })
    }

    /// Stop the background monitoring
    pub async fn stop_background_monitoring(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }
}

/// Overall health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub degraded_connections: usize,
    pub unhealthy_connections: usize,
    pub disconnected_connections: usize,
    pub overall_uptime_percentage: f64,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub connection_id: String,
    pub server_id: String,
    pub status: HealthStatus,
    pub response_time: Duration,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub error_message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_health_config_default() {
        let config = HealthConfig::default();
        assert_eq!(config.check_interval, Duration::from_secs(30));
        assert_eq!(config.failure_threshold, 3);
        assert!(config.enable_keep_alive);
    }

    #[tokio::test]
    async fn test_health_metrics_creation() {
        let metrics = HealthMetrics::new("conn1".to_string(), "server1".to_string());
        assert_eq!(metrics.connection_id, "conn1");
        assert_eq!(metrics.server_id, "server1");
        assert_eq!(metrics.status, HealthStatus::Disconnected);
        assert_eq!(metrics.total_checks, 0);
    }

    #[tokio::test]
    async fn test_health_metrics_success_update() {
        let mut metrics = HealthMetrics::new("conn1".to_string(), "server1".to_string());
        let response_time = Duration::from_millis(100);

        metrics.update_success(response_time);

        assert_eq!(metrics.consecutive_successes, 1);
        assert_eq!(metrics.consecutive_failures, 0);
        assert_eq!(metrics.total_successes, 1);
        assert_eq!(metrics.total_checks, 1);
        assert_eq!(metrics.uptime_percentage, 100.0);
        assert!(metrics.last_success.is_some());
    }

    #[tokio::test]
    async fn test_health_metrics_failure_update() {
        let mut metrics = HealthMetrics::new("conn1".to_string(), "server1".to_string());

        metrics.update_failure();

        assert_eq!(metrics.consecutive_failures, 1);
        assert_eq!(metrics.consecutive_successes, 0);
        assert_eq!(metrics.total_failures, 1);
        assert_eq!(metrics.total_checks, 1);
        assert_eq!(metrics.uptime_percentage, 0.0);
        assert!(metrics.last_failure.is_some());
    }

    #[tokio::test]
    async fn test_connection_health_monitor_creation() {
        let config = HealthConfig::default();
        let monitor = ConnectionHealthMonitor::new(config);

        let summary = monitor.get_health_summary().await;
        assert_eq!(summary.total_connections, 0);
    }

    #[tokio::test]
    async fn test_start_stop_monitoring() {
        let monitor = ConnectionHealthMonitor::new(HealthConfig::default());

        monitor
            .start_monitoring("conn1".to_string(), "server1".to_string())
            .await;

        let metrics = monitor.get_connection_metrics("conn1").await;
        assert!(metrics.is_some());

        monitor.stop_monitoring("conn1").await;

        let metrics = monitor.get_connection_metrics("conn1").await;
        assert!(metrics.is_none());
    }

    #[tokio::test]
    async fn test_health_summary() {
        let monitor = ConnectionHealthMonitor::new(HealthConfig::default());

        monitor
            .start_monitoring("conn1".to_string(), "server1".to_string())
            .await;
        monitor
            .start_monitoring("conn2".to_string(), "server2".to_string())
            .await;

        let summary = monitor.get_health_summary().await;
        assert_eq!(summary.total_connections, 2);
        assert_eq!(summary.disconnected_connections, 2); // Default status
    }
}
