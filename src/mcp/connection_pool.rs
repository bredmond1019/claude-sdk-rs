use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::{interval, sleep, timeout};
use uuid::Uuid;

use crate::mcp::clients::{MCPClient, StdioMCPClient, WebSocketMCPClient};
use crate::mcp::core::error::{
    circuit_breaker::{CircuitBreakerConfig, CircuitBreakerRegistry},
    WorkflowError,
};
use crate::mcp::health::{ConnectionHealthMonitor, HealthConfig, HealthStatus};
use crate::mcp::load_balancer::{ConnectionInfo, MCPLoadBalancer};
use crate::mcp::transport::TransportType;

/// A borrowed connection that automatically returns to the pool when dropped
pub struct BorrowedConnection {
    client: Arc<RwLock<Box<dyn MCPClient>>>,
    connection_id: String,
    pool: std::sync::Weak<MCPConnectionPool>,
    server_id: String,
}

impl BorrowedConnection {
    pub async fn list_tools(
        &self,
    ) -> Result<Vec<crate::mcp::protocol::ToolDefinition>, WorkflowError> {
        let mut client = self.client.write().await;
        client.list_tools().await
    }

    pub async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<crate::mcp::protocol::MCPResponse, WorkflowError> {
        let mut client = self.client.write().await;

        // Convert args to expected format
        let args_map = if args.is_null() {
            None
        } else if let serde_json::Value::Object(map) = args {
            Some(map.into_iter().collect())
        } else {
            Some(HashMap::from([("value".to_string(), args)]))
        };

        let result = client.call_tool(name, args_map).await?;

        // Convert CallToolResult to MCPResponse
        Ok(crate::mcp::protocol::MCPResponse::Result {
            id: uuid::Uuid::new_v4().to_string(),
            result: crate::mcp::protocol::ResponseResult::CallTool(result),
        })
    }

    pub async fn is_connected(&self) -> bool {
        let client = self.client.read().await;
        client.is_connected()
    }

    pub fn connection_id(&self) -> &str {
        &self.connection_id
    }
}

impl Drop for BorrowedConnection {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            let server_id = self.server_id.clone();
            let connection_id = self.connection_id.clone();

            // Return connection to pool asynchronously
            tokio::spawn(async move {
                let _ = pool
                    .return_connection_internal(&server_id, &connection_id)
                    .await;
            });
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub max_connections_per_server: usize,
    #[serde(with = "serde_duration")]
    pub connection_timeout: Duration,
    #[serde(with = "serde_duration")]
    pub idle_timeout: Duration,
    pub retry_attempts: usize,
    #[serde(with = "serde_duration")]
    pub retry_delay: Duration,
    #[serde(with = "serde_duration")]
    pub health_check_interval: Duration,
    /// Enable load balancing across multiple connections
    pub enable_load_balancing: bool,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Health monitoring configuration
    pub health_monitoring: HealthConfig,
    /// Enable automatic reconnection
    pub enable_auto_reconnect: bool,
    /// Exponential backoff configuration
    pub backoff_config: BackoffConfig,
}

// Helper module for serializing Duration as seconds
mod serde_duration {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Load balancing strategies for connection selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin selection
    RoundRobin,
    /// Random selection
    Random,
    /// Least connections first
    LeastConnections,
    /// Health-based selection (prefer healthy connections)
    HealthBased,
}

/// Exponential backoff configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffConfig {
    /// Initial delay
    pub initial_delay: Duration,
    /// Maximum delay
    pub max_delay: Duration,
    /// Multiplier for each retry
    pub multiplier: f64,
    /// Add jitter to prevent thundering herd
    pub jitter: bool,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: true,
        }
    }
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_connections_per_server: 5,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            retry_attempts: 3,
            retry_delay: Duration::from_millis(1000),
            health_check_interval: Duration::from_secs(60),
            enable_load_balancing: true,
            load_balancing_strategy: LoadBalancingStrategy::HealthBased,
            circuit_breaker: CircuitBreakerConfig::default(),
            health_monitoring: HealthConfig::default(),
            enable_auto_reconnect: true,
            backoff_config: BackoffConfig::default(),
        }
    }
}

#[derive(Debug)]
struct PooledConnection {
    client: Arc<RwLock<Box<dyn MCPClient>>>,
    last_used: Arc<RwLock<Instant>>,
    is_healthy: Arc<RwLock<bool>>,
    connection_id: String,
    created_at: Instant,
    use_count: Arc<RwLock<u64>>,
    transport_type: TransportType,
    in_use: Arc<RwLock<bool>>,
}

impl PooledConnection {
    fn new(
        client: Box<dyn MCPClient>,
        connection_id: String,
        transport_type: TransportType,
    ) -> Self {
        let now = Instant::now();
        Self {
            client: Arc::new(RwLock::new(client)),
            last_used: Arc::new(RwLock::new(now)),
            is_healthy: Arc::new(RwLock::new(true)),
            connection_id,
            created_at: now,
            use_count: Arc::new(RwLock::new(0)),
            transport_type,
            in_use: Arc::new(RwLock::new(false)),
        }
    }

    async fn is_expired(&self, idle_timeout: Duration) -> bool {
        let last_used = *self.last_used.read().await;
        last_used.elapsed() > idle_timeout
    }

    async fn touch(&self) {
        let mut last_used = self.last_used.write().await;
        *last_used = Instant::now();
        let mut use_count = self.use_count.write().await;
        *use_count += 1;
    }

    fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    async fn is_busy(&self) -> bool {
        let in_use = *self.in_use.read().await;
        let last_used = *self.last_used.read().await;
        // Connection is busy if in use OR used very recently
        in_use || last_used.elapsed() < Duration::from_millis(100)
    }

    async fn is_healthy(&self) -> bool {
        *self.is_healthy.read().await
    }

    async fn set_healthy(&self, healthy: bool) {
        let mut is_healthy = self.is_healthy.write().await;
        *is_healthy = healthy;
    }

    async fn set_in_use(&self, in_use: bool) {
        let mut in_use_guard = self.in_use.write().await;
        *in_use_guard = in_use;
    }

    async fn is_available(&self) -> bool {
        let is_healthy = self.is_healthy().await;
        let is_busy = self.is_busy().await;
        let client_guard = self.client.read().await;
        let is_connected = client_guard.is_connected();

        is_healthy && !is_busy && is_connected
    }

    async fn get_use_count(&self) -> u64 {
        *self.use_count.read().await
    }
}

pub struct MCPConnectionPool {
    connections: Arc<RwLock<HashMap<String, Vec<PooledConnection>>>>,
    config: ConnectionConfig,
    server_configs: Arc<RwLock<HashMap<String, (TransportType, String, String)>>>, // server_id -> (transport, client_name, client_version)
    circuit_breakers: Arc<CircuitBreakerRegistry>,
    health_monitor: Arc<ConnectionHealthMonitor>,
    load_balancer: Arc<RwLock<MCPLoadBalancer>>,
    // metrics_collector: Option<Arc<MCPMetricsCollector>>,
    background_tasks: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

impl MCPConnectionPool {
    pub fn new(config: ConnectionConfig) -> Arc<Self> {
        let circuit_breaker_registry = Arc::new(CircuitBreakerRegistry::new());
        let health_monitor = Arc::new(ConnectionHealthMonitor::new(
            config.health_monitoring.clone(),
        ));
        let load_balancer = Arc::new(RwLock::new(MCPLoadBalancer::new(
            config.load_balancing_strategy,
        )));

        Arc::new(Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            config,
            server_configs: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: circuit_breaker_registry,
            health_monitor,
            load_balancer,
            // metrics_collector: None,
            background_tasks: Arc::new(RwLock::new(Vec::new())),
        })
    }

    // Set metrics collector for monitoring
    // pub fn set_metrics_collector(&mut self, collector: Arc<MCPMetricsCollector>) {
    //     self.metrics_collector = Some(collector);
    // }

    /// Record connection request metrics
    fn record_connection_request(&self, _success: bool, _latency: Duration) {
        // if let Some(collector) = &self.metrics_collector {
        //     collector.record_connection_request(success, latency);
        // }
    }

    /// Record health check metrics
    fn record_health_check(&self, _success: bool) {
        // if let Some(collector) = &self.metrics_collector {
        //     collector.record_health_check(success);
        // }
    }

    /// Start background tasks for health monitoring and cleanup
    pub async fn start_background_tasks(self: &Arc<Self>) {
        let mut tasks = self.background_tasks.write().await;

        // Health monitoring task
        let health_task = self.health_monitor.start_background_monitoring().await;
        tasks.push(health_task);

        // Connection cleanup task
        let cleanup_task = self.start_cleanup_task().await;
        tasks.push(cleanup_task);

        // Reconnection task
        if self.config.enable_auto_reconnect {
            let reconnect_task = self.start_reconnection_task().await;
            tasks.push(reconnect_task);
        }

        // Health check task
        let health_check_task = self.start_health_check_task().await;
        tasks.push(health_check_task);
    }

    /// Stop all background tasks
    pub async fn stop_background_tasks(&self) {
        let mut tasks = self.background_tasks.write().await;
        for task in tasks.drain(..) {
            task.abort();
        }
        self.health_monitor.stop_background_monitoring().await;
    }

    pub async fn register_server(
        &self,
        server_id: String,
        transport: TransportType,
        client_name: String,
        client_version: String,
    ) {
        let mut configs = self.server_configs.write().await;
        configs.insert(server_id, (transport, client_name, client_version));
    }

    pub async fn get_connection(
        self: &Arc<Self>,
        server_id: &str,
    ) -> Result<BorrowedConnection, WorkflowError> {
        // Try to get an existing connection first
        if let Some(borrowed_conn) = self.try_get_existing_connection(server_id).await? {
            return Ok(borrowed_conn);
        }

        // Create a new connection with retry logic
        self.create_connection_with_retry(server_id).await
    }

    async fn try_get_existing_connection(
        self: &Arc<Self>,
        server_id: &str,
    ) -> Result<Option<BorrowedConnection>, WorkflowError> {
        let connections = self.connections.read().await;

        if let Some(pool) = connections.get(server_id) {
            // Collect connection info for load balancer
            let mut connection_infos = Vec::new();

            for conn in pool.iter() {
                if !conn.is_expired(self.config.idle_timeout).await {
                    let health_status = match conn.is_healthy().await {
                        true => HealthStatus::Healthy,
                        false => HealthStatus::Unhealthy,
                    };

                    connection_infos.push(ConnectionInfo {
                        connection_id: conn.connection_id.clone(),
                        server_id: server_id.to_string(),
                        health_status,
                        response_time: None, // Would need to track this
                        use_count: conn.get_use_count().await,
                        is_available: conn.is_available().await,
                    });
                }
            }

            // Use load balancer to select the best connection
            let load_balancer = self.load_balancer.read().await;
            let Some(selected_id) = load_balancer
                .select_connection(server_id, &connection_infos)
                .await?
            else {
                return Ok(None);
            };

            // Find the selected connection and mark it as in use
            for conn in pool.iter() {
                if conn.connection_id != selected_id || !conn.is_available().await {
                    continue;
                }

                conn.set_in_use(true).await;
                conn.touch().await;

                return Ok(Some(BorrowedConnection {
                    client: Arc::clone(&conn.client),
                    connection_id: conn.connection_id.clone(),
                    pool: Arc::downgrade(self),
                    server_id: server_id.to_string(),
                }));
            }
        }

        Ok(None)
    }

    async fn create_connection_with_retry(
        self: &Arc<Self>,
        server_id: &str,
    ) -> Result<BorrowedConnection, WorkflowError> {
        let configs = self.server_configs.read().await;
        let (transport, client_name, client_version) = configs
            .get(server_id)
            .ok_or_else(|| WorkflowError::MCPError {
                message: format!("Server {} not registered", server_id),
            })?
            .clone();
        drop(configs);

        let mut last_error = None;

        for attempt in 0..self.config.retry_attempts {
            let start_time = Instant::now();

            match self
                .create_single_connection(&transport, &client_name, &client_version)
                .await
            {
                Ok(client) => {
                    let latency = start_time.elapsed();
                    self.record_connection_request(true, latency);

                    // Create a new connection for the pool
                    let pool_client = self
                        .create_single_connection(&transport, &client_name, &client_version)
                        .await?;
                    let connection_id =
                        self.add_to_pool(server_id, pool_client, &transport).await?;

                    // Start monitoring the new connection
                    self.health_monitor
                        .start_monitoring(connection_id.clone(), server_id.to_string())
                        .await;

                    // Return a borrowed connection for the new client
                    return self.create_borrowed_connection(server_id, client).await;
                }
                Err(error) => {
                    let latency = start_time.elapsed();
                    self.record_connection_request(false, latency);

                    last_error = Some(error);
                    if attempt < self.config.retry_attempts - 1 {
                        tracing::warn!(
                            "Connection attempt {} failed for server {}, retrying in {:?}",
                            attempt + 1,
                            server_id,
                            self.config.retry_delay
                        );
                        sleep(self.config.retry_delay).await;
                    }
                }
            }
        }

        Err(
            last_error.unwrap_or_else(|| WorkflowError::MCPConnectionError {
                message: "Failed to create connection after retries".to_string(),
            }),
        )
    }

    async fn create_single_connection(
        &self,
        transport: &TransportType,
        client_name: &str,
        client_version: &str,
    ) -> Result<Box<dyn MCPClient>, WorkflowError> {
        let mut client: Box<dyn MCPClient> = match transport {
            TransportType::Stdio { command, args, .. } => {
                Box::new(StdioMCPClient::new(command.clone(), args.clone()))
            }
            TransportType::WebSocket { url, .. } => Box::new(WebSocketMCPClient::new(url.clone())),
            TransportType::Http { .. } => {
                return Err(WorkflowError::MCPError {
                    message: "HTTP transport not yet supported for connection pooling".to_string(),
                });
            }
        };

        // Connect with timeout
        timeout(self.config.connection_timeout, client.connect())
            .await
            .map_err(|_| WorkflowError::MCPConnectionError {
                message: "Connection timeout".to_string(),
            })??;

        // Initialize with timeout
        timeout(
            self.config.connection_timeout,
            client.initialize(client_name, client_version),
        )
        .await
        .map_err(|_| WorkflowError::MCPConnectionError {
            message: "Initialization timeout".to_string(),
        })??;

        Ok(client)
    }

    async fn add_to_pool(
        &self,
        server_id: &str,
        client: Box<dyn MCPClient>,
        transport_type: &TransportType,
    ) -> Result<String, WorkflowError> {
        let mut connections = self.connections.write().await;
        let pool = connections
            .entry(server_id.to_string())
            .or_insert_with(Vec::new);

        // Check if we're at the connection limit
        if pool.len() >= self.config.max_connections_per_server {
            // Remove the oldest connection (LRU)
            let mut oldest_index = None;
            let mut oldest_time = Instant::now();

            for (index, conn) in pool.iter().enumerate() {
                let last_used = *conn.last_used.read().await;
                if last_used < oldest_time {
                    oldest_time = last_used;
                    oldest_index = Some(index);
                }
            }

            if let Some(index) = oldest_index {
                let old_conn = pool.remove(index);
                let _ = old_conn.client.write().await.disconnect().await;
                self.health_monitor
                    .stop_monitoring(&old_conn.connection_id)
                    .await;
            }
        }

        let connection_id = format!("{}_{}", server_id, Uuid::new_v4());
        let pooled_conn =
            PooledConnection::new(client, connection_id.clone(), transport_type.clone());
        pool.push(pooled_conn);

        tracing::debug!(
            "Added connection {} to pool for server {}",
            connection_id,
            server_id
        );
        Ok(connection_id)
    }

    /// Internal method to return a connection to the pool
    pub async fn return_connection_internal(
        &self,
        server_id: &str,
        connection_id: &str,
    ) -> Result<(), WorkflowError> {
        let connections = self.connections.read().await;
        if let Some(pool) = connections.get(server_id) {
            for conn in pool.iter() {
                if conn.connection_id == connection_id {
                    conn.set_in_use(false).await;
                    tracing::debug!(
                        "Returned connection {} to pool for server {}",
                        connection_id,
                        server_id
                    );
                    break;
                }
            }
        }
        Ok(())
    }

    /// Create a borrowed connection wrapper
    async fn create_borrowed_connection(
        self: &Arc<Self>,
        server_id: &str,
        _client: Box<dyn MCPClient>,
    ) -> Result<BorrowedConnection, WorkflowError> {
        // Create a new connection for the pool
        let pool_client = self.create_single_connection_from_server(server_id).await?;
        let connection_id = self
            .add_to_pool(
                server_id,
                pool_client,
                &self.get_transport_for_server(server_id).await?,
            )
            .await?;

        // Create the borrowed connection
        let connections = self.connections.read().await;
        if let Some(pool) = connections.get(server_id) {
            for conn in pool.iter() {
                if conn.connection_id == connection_id {
                    conn.set_in_use(true).await;
                    return Ok(BorrowedConnection {
                        client: Arc::clone(&conn.client),
                        connection_id: conn.connection_id.clone(),
                        pool: Arc::downgrade(self),
                        server_id: server_id.to_string(),
                    });
                }
            }
        }

        Err(WorkflowError::MCPError {
            message: "Failed to create borrowed connection".to_string(),
        })
    }

    /// Get transport type for a server
    async fn get_transport_for_server(
        &self,
        server_id: &str,
    ) -> Result<TransportType, WorkflowError> {
        let configs = self.server_configs.read().await;
        let (transport, _, _) = configs
            .get(server_id)
            .ok_or_else(|| WorkflowError::MCPError {
                message: format!("Server {} not registered", server_id),
            })?
            .clone();
        Ok(transport)
    }

    /// Create a single connection for a registered server
    async fn create_single_connection_from_server(
        &self,
        server_id: &str,
    ) -> Result<Box<dyn MCPClient>, WorkflowError> {
        let configs = self.server_configs.read().await;
        let (transport, client_name, client_version) = configs
            .get(server_id)
            .ok_or_else(|| WorkflowError::MCPError {
                message: format!("Server {} not registered", server_id),
            })?
            .clone();
        drop(configs);

        self.create_single_connection(&transport, &client_name, &client_version)
            .await
    }

    pub async fn health_check(&self) -> Result<HashMap<String, bool>, WorkflowError> {
        let mut results = HashMap::new();
        let connections = self.connections.read().await;

        for (server_id, pool) in connections.iter() {
            // Check both pool health and circuit breaker state
            let mut healthy_count = 0;
            for conn in pool.iter() {
                if conn.is_healthy().await {
                    healthy_count += 1;
                }
            }

            let circuit_breaker = self.circuit_breakers.get(server_id).await;
            let circuit_state = if let Some(breaker) = circuit_breaker {
                breaker.read().await.state()
            } else {
                crate::mcp::core::error::circuit_breaker::CircuitState::Closed
            };

            let is_healthy = healthy_count > 0
                && circuit_state != crate::mcp::core::error::circuit_breaker::CircuitState::Open;

            results.insert(server_id.clone(), is_healthy);
        }

        Ok(results)
    }

    pub async fn cleanup_expired_connections(&self) -> Result<usize, WorkflowError> {
        let mut connections = self.connections.write().await;
        let mut cleaned_count = 0;

        for pool in connections.values_mut() {
            let original_len = pool.len();
            let mut to_remove = Vec::new();

            for (index, conn) in pool.iter().enumerate() {
                if conn.is_expired(self.config.idle_timeout).await {
                    to_remove.push(index);
                }
            }

            // Remove connections in reverse order to maintain indices
            for &index in to_remove.iter().rev() {
                pool.remove(index);
            }

            cleaned_count += original_len - pool.len();
        }

        Ok(cleaned_count)
    }

    pub async fn disconnect_all(&self) -> Result<(), WorkflowError> {
        let mut connections = self.connections.write().await;

        for pool in connections.values_mut() {
            for conn in pool.drain(..) {
                let mut client = conn.client.write().await;
                let _ = client.disconnect().await;
            }
        }

        Ok(())
    }

    pub async fn get_pool_stats(&self) -> HashMap<String, PoolStats> {
        let connections = self.connections.read().await;
        let mut stats = HashMap::new();

        for (server_id, pool) in connections.iter() {
            let mut healthy_count = 0;
            let mut connected_count = 0;
            let mut busy_count = 0;
            let mut total_use_count = 0;

            for conn in pool.iter() {
                if conn.is_healthy().await {
                    healthy_count += 1;
                }

                let client = conn.client.read().await;
                if client.is_connected() {
                    connected_count += 1;
                }

                if conn.is_busy().await {
                    busy_count += 1;
                }

                total_use_count += conn.get_use_count().await;
            }

            stats.insert(
                server_id.clone(),
                PoolStats {
                    total_connections: pool.len(),
                    healthy_connections: healthy_count,
                    connected_connections: connected_count,
                    busy_connections: busy_count,
                    average_age_seconds: if !pool.is_empty() {
                        pool.iter().map(|conn| conn.age().as_secs()).sum::<u64>()
                            / pool.len() as u64
                    } else {
                        0
                    },
                    total_use_count,
                },
            );
        }

        stats
    }

    /// Get detailed health information including circuit breaker metrics
    pub async fn get_detailed_health(&self) -> DetailedHealthInfo {
        let connections = self.connections.read().await;
        let mut server_health = HashMap::new();

        for (server_id, pool) in connections.iter() {
            let circuit_breaker = self.circuit_breakers.get(server_id).await;
            // let circuit_metrics = circuit_breaker.metrics();
            let circuit_state = if let Some(breaker) = circuit_breaker {
                breaker.read().await.state()
            } else {
                crate::mcp::core::error::circuit_breaker::CircuitState::Closed
            };

            let mut healthy_count = 0;
            let mut connected_count = 0;
            let mut busy_count = 0;
            let mut total_use_count = 0u64;

            for conn in pool.iter() {
                if *conn.is_healthy.read().await {
                    healthy_count += 1;
                }
                if conn.client.read().await.is_connected() {
                    connected_count += 1;
                }
                if conn.is_busy().await {
                    busy_count += 1;
                }
                total_use_count += *conn.use_count.read().await;
            }

            let pool_stats = PoolStats {
                total_connections: pool.len(),
                healthy_connections: healthy_count,
                connected_connections: connected_count,
                busy_connections: busy_count,
                average_age_seconds: if !pool.is_empty() {
                    pool.iter().map(|conn| conn.age().as_secs()).sum::<u64>() / pool.len() as u64
                } else {
                    0
                },
                total_use_count,
            };

            server_health.insert(
                server_id.clone(),
                ServerHealthInfo {
                    pool_stats,
                    circuit_state,
                    // circuit_metrics,
                },
            );
        }

        let health_summary = self.health_monitor.get_health_summary().await;

        DetailedHealthInfo {
            server_health,
            overall_summary: health_summary,
        }
    }

    /// Start the cleanup task for removing expired connections
    #[allow(clippy::excessive_nesting)]
    async fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let connections = Arc::clone(&self.connections);
        let health_monitor = Arc::clone(&self.health_monitor);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval);

            loop {
                interval.tick().await;

                let mut connections_guard = connections.write().await;
                for (server_id, pool) in connections_guard.iter_mut() {
                    let original_len = pool.len();
                    let mut removed_connections = Vec::new();

                    // We need to handle async calls, so we'll collect expired connections first
                    let mut expired_indices = Vec::new();
                    for (index, conn) in pool.iter().enumerate() {
                        if conn.is_expired(config.idle_timeout).await {
                            expired_indices.push(index);
                            removed_connections.push(conn.connection_id.clone());
                        }
                    }

                    // Remove expired connections in reverse order
                    for &index in expired_indices.iter().rev() {
                        pool.remove(index);
                    }

                    let removed_count = original_len - pool.len();
                    if removed_count > 0 {
                        tracing::info!(
                            "Cleaned up {} expired connections for server {}",
                            removed_count,
                            server_id
                        );

                        // Stop monitoring removed connections
                        for connection_id in removed_connections {
                            health_monitor.stop_monitoring(&connection_id).await;
                        }
                    }
                }
            }
        })
    }

    /// Start the reconnection task for automatically reconnecting failed connections
    async fn start_reconnection_task(&self) -> tokio::task::JoinHandle<()> {
        let connections = Arc::clone(&self.connections);
        let server_configs = Arc::clone(&self.server_configs);
        let circuit_breakers = Arc::clone(&self.circuit_breakers);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval * 2); // Less frequent than cleanup

            loop {
                interval.tick().await;

                let server_configs_guard = server_configs.read().await;
                for server_id in server_configs_guard.keys() {
                    // Check circuit breaker state
                    let circuit_breaker = circuit_breakers.get(server_id).await;
                    let circuit_state = if let Some(breaker) = circuit_breaker {
                        breaker.read().await.state()
                    } else {
                        crate::mcp::core::error::circuit_breaker::CircuitState::Closed
                    };

                    // Only attempt reconnection if circuit is not open
                    if circuit_state != crate::mcp::core::error::circuit_breaker::CircuitState::Open
                    {
                        let connections_guard = connections.read().await;
                        if let Some(pool) = connections_guard.get(server_id) {
                            let mut healthy_count = 0;
                            for conn in pool.iter() {
                                if *conn.is_healthy.read().await
                                    && conn.client.read().await.is_connected()
                                {
                                    healthy_count += 1;
                                }
                            }

                            // If we have fewer healthy connections than desired, try to add more
                            if healthy_count < config.max_connections_per_server / 2 {
                                tracing::debug!(
                                    "Server {} has only {} healthy connections, attempting to add more",
                                    server_id,
                                    healthy_count
                                );
                                // Note: In a real implementation, we'd need access to the pool methods
                                // This would require restructuring to avoid circular dependencies
                            }
                        }
                    }
                }
                drop(server_configs_guard);
            }
        })
    }

    /// Force reconnection for a specific server
    pub async fn force_reconnect(&self, server_id: &str) -> Result<(), WorkflowError> {
        // Reset circuit breaker
        let circuit_breaker = self.circuit_breakers.get(server_id).await;
        if let Some(breaker) = circuit_breaker {
            breaker.write().await.reset();
        }

        // Clear existing connections
        let mut connections = self.connections.write().await;
        if let Some(pool) = connections.get_mut(server_id) {
            let mut removed_connections = Vec::new();
            for conn in pool.drain(..) {
                removed_connections.push(conn.connection_id.clone());
                let _ = conn.client.write().await.disconnect().await;
            }

            // Stop monitoring removed connections
            for connection_id in removed_connections {
                self.health_monitor.stop_monitoring(&connection_id).await;
            }
        }

        tracing::info!("Forced reconnection for server {}", server_id);
        Ok(())
    }

    // /// Get circuit breaker metrics for all servers
    // pub async fn get_circuit_breaker_metrics(&self) -> HashMap<String, crate::core::error::circuit_breaker::CircuitBreakerMetrics> {
    //     let mut metrics = HashMap::new();
    //
    //     let breakers = self.circuit_breakers.all().await;
    //     for (service, breaker) in breakers {
    //         metrics.insert(service, breaker.metrics());
    //     }
    //
    //     metrics
    // }

    /// Perform health checks on all connections and update their health status
    pub async fn perform_health_checks(&self) -> Result<(), WorkflowError> {
        let connections = self.connections.read().await;

        for (server_id, pool) in connections.iter() {
            for conn in pool.iter() {
                // Skip connections that are currently in use
                if conn.is_busy().await {
                    continue;
                }

                // Perform health check
                let mut client_guard = conn.client.write().await;
                match self
                    .health_monitor
                    .check_connection_health(&conn.connection_id, &mut client_guard)
                    .await
                {
                    Ok(health_status) => {
                        let is_healthy = matches!(
                            health_status,
                            HealthStatus::Healthy | HealthStatus::Degraded
                        );
                        conn.set_healthy(is_healthy).await;
                        self.record_health_check(is_healthy);

                        if !is_healthy {
                            tracing::warn!(
                                "Connection {} for server {} is unhealthy: {:?}",
                                conn.connection_id,
                                server_id,
                                health_status
                            );
                        }
                    }
                    Err(e) => {
                        conn.set_healthy(false).await;
                        self.record_health_check(false);
                        tracing::error!(
                            "Health check failed for connection {} on server {}: {}",
                            conn.connection_id,
                            server_id,
                            e
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Attempt to recover failed connections by creating new ones
    pub async fn recover_failed_connections(self: &Arc<Self>) -> Result<(), WorkflowError> {
        let servers_to_recover = {
            let connections = self.connections.read().await;
            let mut servers = Vec::new();

            for (server_id, pool) in connections.iter() {
                let healthy_count = pool
                    .iter()
                    .filter(|_conn| {
                        // We need to handle async operations here
                        true // Placeholder - we'll need to refactor this
                    })
                    .count();

                // If we have fewer than half the max connections healthy, attempt recovery
                if healthy_count < self.config.max_connections_per_server / 2 {
                    servers.push(server_id.clone());
                }
            }

            servers
        };

        for server_id in servers_to_recover {
            tracing::info!("Attempting to recover connections for server {}", server_id);

            // Try to create a new connection to replace failed ones
            match self.create_connection_with_retry(&server_id).await {
                Ok(_) => {
                    tracing::info!("Successfully recovered connection for server {}", server_id);
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to recover connection for server {}: {}",
                        server_id,
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Update load balancer weights based on current pool health
    pub async fn update_load_balancer_weights(self: &Arc<Self>) -> Result<(), WorkflowError> {
        let pool_stats = self.get_pool_stats().await;
        let load_balancer = self.load_balancer.write().await;
        load_balancer.update_server_weights(pool_stats).await;
        Ok(())
    }

    /// Start the health check task for monitoring and recovery
    async fn start_health_check_task(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let pool = Arc::clone(self);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval);

            loop {
                interval.tick().await;

                // Perform health checks
                if let Err(e) = pool.perform_health_checks().await {
                    tracing::error!("Health check task failed: {}", e);
                }

                // Attempt to recover failed connections
                if config.enable_auto_reconnect {
                    if let Err(e) = pool.recover_failed_connections().await {
                        tracing::error!("Connection recovery task failed: {}", e);
                    }
                }

                // Update load balancer weights
                if let Err(e) = pool.update_load_balancer_weights().await {
                    tracing::error!("Load balancer weight update failed: {}", e);
                }

                tracing::debug!("Health check task completed");
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub connected_connections: usize,
    pub busy_connections: usize,
    pub average_age_seconds: u64,
    pub total_use_count: u64,
}

/// Server-specific health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHealthInfo {
    pub pool_stats: PoolStats,
    pub circuit_state: crate::mcp::core::error::circuit_breaker::CircuitState,
    // pub circuit_metrics: crate::core::error::circuit_breaker::CircuitBreakerMetrics,
}

/// Detailed health information for the entire pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedHealthInfo {
    pub server_health: HashMap<String, ServerHealthInfo>,
    pub overall_summary: crate::mcp::health::HealthSummary,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::transport::TransportType;

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let config = ConnectionConfig::default();
        let pool = MCPConnectionPool::new(config);

        let stats = pool.get_pool_stats().await;
        assert!(stats.is_empty());
    }

    #[tokio::test]
    async fn test_server_registration() {
        let pool = MCPConnectionPool::new(ConnectionConfig::default());

        pool.register_server(
            "test-server".to_string(),
            TransportType::Stdio {
                command: "echo".to_string(),
                args: vec!["hello".to_string()],
                auto_restart: true,
                max_restarts: 3,
            },
            "test-client".to_string(),
            "1.0.0".to_string(),
        )
        .await;

        let configs = pool.server_configs.read().await;
        assert!(configs.contains_key("test-server"));
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let pool = MCPConnectionPool::new(ConnectionConfig::default());

        pool.register_server(
            "test-server".to_string(),
            TransportType::WebSocket {
                url: "ws://localhost:8080".to_string(),
                heartbeat_interval: None,
                reconnect_config: crate::mcp::transport::ReconnectConfig::default(),
            },
            "test-client".to_string(),
            "1.0.0".to_string(),
        )
        .await;

        let stats = pool.get_pool_stats().await;
        assert_eq!(stats.len(), 0); // No connections created yet
    }

    #[tokio::test]
    async fn test_cleanup_expired_connections() {
        let pool = MCPConnectionPool::new(ConnectionConfig::default());

        let cleaned = pool.cleanup_expired_connections().await.unwrap();
        assert_eq!(cleaned, 0); // No connections to clean
    }
}
