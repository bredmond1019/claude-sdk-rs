//! # MCP Metrics Integration
//!
//! This module provides integration between MCP connection pooling and the
//! existing Prometheus metrics system for comprehensive monitoring.

#[cfg(feature = "mcp")]
use crate::mcp::connection_pool::{DetailedHealthInfo, MCPConnectionPool};
#[cfg(feature = "mcp")]
use prometheus::{Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};
#[cfg(feature = "mcp")]
use std::collections::HashMap;
#[cfg(feature = "mcp")]
use std::sync::Arc;
#[cfg(feature = "mcp")]
use tokio::sync::RwLock;
#[cfg(feature = "mcp")]
use tokio::time::{interval, Duration};

#[cfg(not(feature = "mcp"))]
use crate::mcp::connection_pool::MCPConnectionPool;
use crate::mcp::core::error::circuit_breaker::CircuitState;
use crate::mcp::core::error::WorkflowError;

#[cfg(feature = "mcp")]
/// MCP-specific metrics collector
pub struct MCPMetrics {
    // Connection metrics
    pub active_connections: IntGauge,
    pub total_connections: IntCounter,
    pub connection_errors: IntCounter,
    pub connection_duration: Histogram,

    // Request metrics
    pub requests_total: IntCounter,
    pub requests_failed: IntCounter,
    pub request_duration: Histogram,

    // Circuit breaker metrics
    pub circuit_breaker_state: IntGauge,
    pub circuit_breaker_trips: IntCounter,

    // Health check metrics
    pub health_check_duration: Histogram,
    pub health_check_failures: IntCounter,

    // Pool metrics
    pub pool_size: IntGauge,
    pub pool_available: IntGauge,
    pub pool_waiters: IntGauge,

    // Server-specific metrics
    pub server_metrics: Arc<RwLock<HashMap<String, ServerMetrics>>>,
}

#[cfg(feature = "mcp")]
struct ServerMetrics {
    requests: IntCounter,
    errors: IntCounter,
    latency: Histogram,
    health_score: Gauge,
}

#[cfg(feature = "mcp")]
impl MCPMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let active_connections = IntGauge::with_opts(Opts::new(
            "mcp_active_connections",
            "Number of active MCP connections",
        ))?;
        registry.register(Box::new(active_connections.clone()))?;

        let total_connections = IntCounter::with_opts(Opts::new(
            "mcp_total_connections",
            "Total number of MCP connections created",
        ))?;
        registry.register(Box::new(total_connections.clone()))?;

        let connection_errors = IntCounter::with_opts(Opts::new(
            "mcp_connection_errors",
            "Number of MCP connection errors",
        ))?;
        registry.register(Box::new(connection_errors.clone()))?;

        let connection_duration = Histogram::with_opts(HistogramOpts::new(
            "mcp_connection_duration_seconds",
            "Time to establish MCP connection",
        ))?;
        registry.register(Box::new(connection_duration.clone()))?;

        let requests_total = IntCounter::with_opts(Opts::new(
            "mcp_requests_total",
            "Total number of MCP requests",
        ))?;
        registry.register(Box::new(requests_total.clone()))?;

        let requests_failed = IntCounter::with_opts(Opts::new(
            "mcp_requests_failed",
            "Number of failed MCP requests",
        ))?;
        registry.register(Box::new(requests_failed.clone()))?;

        let request_duration = Histogram::with_opts(HistogramOpts::new(
            "mcp_request_duration_seconds",
            "MCP request duration",
        ))?;
        registry.register(Box::new(request_duration.clone()))?;

        let circuit_breaker_state = IntGauge::with_opts(Opts::new(
            "mcp_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        ))?;
        registry.register(Box::new(circuit_breaker_state.clone()))?;

        let circuit_breaker_trips = IntCounter::with_opts(Opts::new(
            "mcp_circuit_breaker_trips",
            "Number of circuit breaker trips",
        ))?;
        registry.register(Box::new(circuit_breaker_trips.clone()))?;

        let health_check_duration = Histogram::with_opts(HistogramOpts::new(
            "mcp_health_check_duration_seconds",
            "Health check duration",
        ))?;
        registry.register(Box::new(health_check_duration.clone()))?;

        let health_check_failures = IntCounter::with_opts(Opts::new(
            "mcp_health_check_failures",
            "Number of health check failures",
        ))?;
        registry.register(Box::new(health_check_failures.clone()))?;

        let pool_size =
            IntGauge::with_opts(Opts::new("mcp_pool_size", "Total size of connection pool"))?;
        registry.register(Box::new(pool_size.clone()))?;

        let pool_available = IntGauge::with_opts(Opts::new(
            "mcp_pool_available",
            "Available connections in pool",
        ))?;
        registry.register(Box::new(pool_available.clone()))?;

        let pool_waiters = IntGauge::with_opts(Opts::new(
            "mcp_pool_waiters",
            "Number of waiters for connections",
        ))?;
        registry.register(Box::new(pool_waiters.clone()))?;

        Ok(Self {
            active_connections,
            total_connections,
            connection_errors,
            connection_duration,
            requests_total,
            requests_failed,
            request_duration,
            circuit_breaker_state,
            circuit_breaker_trips,
            health_check_duration,
            health_check_failures,
            pool_size,
            pool_available,
            pool_waiters,
            server_metrics: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Update metrics from connection pool status
    pub async fn update_from_pool(&self, pool: &MCPConnectionPool) -> Result<(), WorkflowError> {
        let stats = pool.get_pool_stats().await;

        let total_connections: i64 = stats.values().map(|s| s.total_connections as i64).sum();
        let healthy_connections: i64 = stats.values().map(|s| s.healthy_connections as i64).sum();
        let connected_connections: i64 =
            stats.values().map(|s| s.connected_connections as i64).sum();

        self.pool_size.set(total_connections);
        self.pool_available.set(connected_connections);
        self.active_connections.set(healthy_connections);

        // Update server-specific metrics
        let health_info = pool.get_detailed_health().await;
        self.update_server_metrics(&health_info).await;

        Ok(())
    }

    async fn update_server_metrics(&self, health_info: &DetailedHealthInfo) {
        let mut server_metrics = self.server_metrics.write().await;

        for (server_name, server_health) in &health_info.server_health {
            // Update health score based on circuit breaker state and pool health
            let score = match server_health.circuit_state {
                CircuitState::Closed => 1.0,
                CircuitState::HalfOpen => 0.5,
                CircuitState::Open => 0.0,
            };

            // If we don't have metrics for this server yet, create them
            if !server_metrics.contains_key(server_name) {
                // For now, we'll skip creating new server metrics dynamically
                // In a full implementation, you'd create new metrics here
                continue;
            }

            if let Some(metrics) = server_metrics.get_mut(server_name) {
                metrics.health_score.set(score);
            }
        }
    }

    /// Record a request
    pub fn record_request(&self, _server: &str, duration: Duration, success: bool) {
        self.requests_total.inc();
        self.request_duration.observe(duration.as_secs_f64());

        if !success {
            self.requests_failed.inc();
        }
    }

    /// Record circuit breaker state change
    pub fn record_circuit_state(&self, state: CircuitState) {
        let state_value = match state {
            CircuitState::Closed => 0,
            CircuitState::Open => 1,
            CircuitState::HalfOpen => 2,
        };
        self.circuit_breaker_state.set(state_value);

        if state == CircuitState::Open {
            self.circuit_breaker_trips.inc();
        }
    }

    /// Start periodic metrics collection from pool
    pub fn start_collector(
        self: Arc<Self>,
        pool: Arc<MCPConnectionPool>,
        update_interval: Duration,
    ) {
        tokio::spawn(async move {
            let mut interval = interval(update_interval);

            loop {
                interval.tick().await;

                if let Err(e) = self.update_from_pool(&pool).await {
                    tracing::warn!("Failed to update MCP metrics: {}", e);
                }
            }
        });
    }
}

#[cfg(not(feature = "mcp"))]
/// Stub implementation when metrics feature is disabled
pub struct MCPMetrics;

#[cfg(not(feature = "mcp"))]
impl MCPMetrics {
    pub fn new(_registry: &()) -> Result<Self, WorkflowError> {
        Ok(Self)
    }

    pub async fn update_from_pool(&self, _pool: &MCPConnectionPool) -> Result<(), WorkflowError> {
        Ok(())
    }

    pub fn record_request(&self, _server: &str, _duration: Duration, _success: bool) {}

    pub fn record_circuit_state(&self, _state: CircuitState) {}

    pub fn start_collector(
        self: Arc<Self>,
        _pool: Arc<MCPConnectionPool>,
        _update_interval: Duration,
    ) {
    }
}
