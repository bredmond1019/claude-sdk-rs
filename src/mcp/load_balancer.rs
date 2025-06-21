//! # MCP Load Balancer
//!
//! This module provides load balancing strategies for selecting MCP connections
//! from the connection pool, including round-robin, least-connections, and health-based selection.

use rand::{seq::SliceRandom, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::mcp::connection_pool::{LoadBalancingStrategy, PoolStats};
use crate::mcp::core::error::WorkflowError;
use crate::mcp::health::HealthStatus;

/// Load balancer for MCP connections
pub struct MCPLoadBalancer {
    strategy: LoadBalancingStrategy,
    round_robin_counters: Arc<RwLock<HashMap<String, usize>>>,
    server_weights: Arc<RwLock<HashMap<String, f64>>>,
}

/// Connection information for load balancing decisions
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub connection_id: String,
    pub server_id: String,
    pub health_status: HealthStatus,
    pub response_time: Option<std::time::Duration>,
    pub use_count: u64,
    pub is_available: bool,
}

/// Load balancing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingMetrics {
    pub total_requests: u64,
    pub requests_per_server: HashMap<String, u64>,
    pub average_response_time: std::time::Duration,
    pub load_distribution: HashMap<String, f64>,
}

impl MCPLoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            round_robin_counters: Arc::new(RwLock::new(HashMap::new())),
            server_weights: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Select the best connection based on the configured strategy
    pub async fn select_connection(
        &self,
        server_id: &str,
        available_connections: &[ConnectionInfo],
    ) -> Result<Option<String>, WorkflowError> {
        if available_connections.is_empty() {
            return Ok(None);
        }

        let connection_id = match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.select_round_robin(server_id, available_connections)
                    .await
            }
            LoadBalancingStrategy::Random => self.select_random(available_connections).await,
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(available_connections).await
            }
            LoadBalancingStrategy::HealthBased => {
                self.select_health_based(available_connections).await
            }
        };

        Ok(connection_id)
    }

    /// Round-robin selection
    async fn select_round_robin(
        &self,
        server_id: &str,
        connections: &[ConnectionInfo],
    ) -> Option<String> {
        let mut counters = self.round_robin_counters.write().await;
        let counter = counters.entry(server_id.to_string()).or_insert(0);

        let available_connections: Vec<_> = connections
            .iter()
            .filter(|conn| conn.is_available)
            .collect();

        if available_connections.is_empty() {
            return None;
        }

        let selected = &available_connections[*counter % available_connections.len()];
        *counter = (*counter + 1) % available_connections.len();

        Some(selected.connection_id.clone())
    }

    /// Random selection
    async fn select_random(&self, connections: &[ConnectionInfo]) -> Option<String> {
        let available_connections: Vec<_> = connections
            .iter()
            .filter(|conn| conn.is_available)
            .collect();

        if available_connections.is_empty() {
            return None;
        }

        let mut rng = thread_rng();
        available_connections
            .choose(&mut rng)
            .map(|conn| conn.connection_id.clone())
    }

    /// Least connections selection (based on use count)
    async fn select_least_connections(&self, connections: &[ConnectionInfo]) -> Option<String> {
        connections
            .iter()
            .filter(|conn| conn.is_available)
            .min_by_key(|conn| conn.use_count)
            .map(|conn| conn.connection_id.clone())
    }

    /// Health-based selection with weighted random selection
    async fn select_health_based(&self, connections: &[ConnectionInfo]) -> Option<String> {
        let mut weighted_connections = Vec::new();

        for conn in connections.iter().filter(|c| c.is_available) {
            let weight = self.calculate_health_weight(conn).await;
            if weight > 0.0 {
                weighted_connections.push((conn, weight));
            }
        }

        if weighted_connections.is_empty() {
            return None;
        }

        // Weighted random selection
        let total_weight: f64 = weighted_connections.iter().map(|(_, w)| w).sum();
        let mut rng = thread_rng();
        let random_weight: f64 = rng.r#gen::<f64>() * total_weight;

        let mut cumulative_weight = 0.0;
        for (conn, weight) in &weighted_connections {
            cumulative_weight += weight;
            if random_weight <= cumulative_weight {
                return Some(conn.connection_id.clone());
            }
        }

        // Fallback to the last connection
        weighted_connections
            .last()
            .map(|(conn, _)| conn.connection_id.clone())
    }

    /// Calculate health-based weight for a connection
    async fn calculate_health_weight(&self, conn: &ConnectionInfo) -> f64 {
        let mut weight: f64 = match conn.health_status {
            HealthStatus::Healthy => 1.0,
            HealthStatus::Degraded => 0.5,
            HealthStatus::Unhealthy => 0.1,
            HealthStatus::Disconnected => 0.0,
        };

        // Adjust weight based on response time
        if let Some(response_time) = conn.response_time {
            let response_ms = response_time.as_millis() as f64;
            // Reduce weight for slower connections
            if response_ms > 1000.0 {
                weight *= 0.5;
            } else if response_ms > 500.0 {
                weight *= 0.8;
            }
        }

        // Reduce weight for heavily used connections
        if conn.use_count > 100 {
            weight *= 0.9;
        }

        weight.max(0.0)
    }

    /// Update server weights based on performance metrics
    pub async fn update_server_weights(&self, server_metrics: HashMap<String, PoolStats>) {
        let mut weights = self.server_weights.write().await;

        for (server_id, stats) in server_metrics {
            let health_ratio = if stats.total_connections > 0 {
                stats.healthy_connections as f64 / stats.total_connections as f64
            } else {
                0.0
            };

            let load_ratio = if stats.total_connections > 0 {
                1.0 - (stats.busy_connections as f64 / stats.total_connections as f64)
            } else {
                1.0
            };

            // Combine health and load ratios for overall weight
            let weight = (health_ratio * 0.7) + (load_ratio * 0.3);
            weights.insert(server_id, weight);
        }
    }

    /// Get server weights for monitoring
    pub async fn get_server_weights(&self) -> HashMap<String, f64> {
        self.server_weights.read().await.clone()
    }

    /// Reset round-robin counters (useful for testing)
    pub async fn reset_counters(&self) {
        let mut counters = self.round_robin_counters.write().await;
        counters.clear();
    }

    /// Get current load balancing strategy
    pub fn strategy(&self) -> LoadBalancingStrategy {
        self.strategy
    }

    /// Update the load balancing strategy
    pub fn set_strategy(&mut self, strategy: LoadBalancingStrategy) {
        self.strategy = strategy;
    }
}

/// Advanced load balancer with support for multiple servers and connection types
pub struct AdvancedMCPLoadBalancer {
    balancer: MCPLoadBalancer,
    server_priorities: Arc<RwLock<HashMap<String, i32>>>,
    connection_affinities: Arc<RwLock<HashMap<String, String>>>, // client_id -> preferred_server_id
    metrics: Arc<RwLock<LoadBalancingMetrics>>,
}

impl AdvancedMCPLoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            balancer: MCPLoadBalancer::new(strategy),
            server_priorities: Arc::new(RwLock::new(HashMap::new())),
            connection_affinities: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(LoadBalancingMetrics {
                total_requests: 0,
                requests_per_server: HashMap::new(),
                average_response_time: std::time::Duration::from_millis(0),
                load_distribution: HashMap::new(),
            })),
        }
    }

    /// Select connection with server priorities and affinity
    pub async fn select_connection_advanced(
        &self,
        client_id: Option<&str>,
        available_servers: &HashMap<String, Vec<ConnectionInfo>>,
    ) -> Result<Option<(String, String)>, WorkflowError> {
        // Returns (server_id, connection_id)
        // Check for client affinity first
        if let Some(client_id) = client_id {
            let affinities = self.connection_affinities.read().await;
            if let Some(preferred_server) = affinities.get(client_id) {
                if let Some(connections) = available_servers.get(preferred_server) {
                    if let Some(conn_id) = self
                        .balancer
                        .select_connection(preferred_server, connections)
                        .await?
                    {
                        self.record_request(preferred_server).await;
                        return Ok(Some((preferred_server.clone(), conn_id)));
                    }
                }
            }
        }

        // Find the best server based on priorities and health
        let priorities = self.server_priorities.read().await;
        let mut server_scores: Vec<(String, f64)> = Vec::new();

        for (server_id, connections) in available_servers {
            if connections.is_empty() {
                continue;
            }

            let priority = priorities.get(server_id).unwrap_or(&0);
            let available_count = connections.iter().filter(|c| c.is_available).count();

            if available_count == 0 {
                continue;
            }

            let health_score = connections
                .iter()
                .map(|c| match c.health_status {
                    HealthStatus::Healthy => 1.0,
                    HealthStatus::Degraded => 0.5,
                    HealthStatus::Unhealthy => 0.1,
                    HealthStatus::Disconnected => 0.0,
                })
                .sum::<f64>()
                / connections.len() as f64;

            let load_score = 1.0
                - (connections.iter().map(|c| c.use_count).sum::<u64>() as f64
                    / (connections.len() as f64 * 100.0)); // Normalize by 100 max uses

            let score = (*priority as f64 * 0.4) + (health_score * 0.4) + (load_score * 0.2);
            server_scores.push((server_id.clone(), score));
        }

        // Sort by score (highest first)
        server_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select connection from the best server
        for (server_id, _score) in server_scores {
            if let Some(connections) = available_servers.get(&server_id) {
                if let Some(conn_id) = self
                    .balancer
                    .select_connection(&server_id, connections)
                    .await?
                {
                    self.record_request(&server_id).await;
                    return Ok(Some((server_id, conn_id)));
                }
            }
        }

        Ok(None)
    }

    /// Set server priority (higher numbers = higher priority)
    pub async fn set_server_priority(&self, server_id: String, priority: i32) {
        let mut priorities = self.server_priorities.write().await;
        priorities.insert(server_id, priority);
    }

    /// Set client affinity to a specific server
    pub async fn set_client_affinity(&self, client_id: String, server_id: String) {
        let mut affinities = self.connection_affinities.write().await;
        affinities.insert(client_id, server_id);
    }

    /// Record a request for metrics
    async fn record_request(&self, server_id: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        *metrics
            .requests_per_server
            .entry(server_id.to_string())
            .or_insert(0) += 1;
    }

    /// Get load balancing metrics
    pub async fn get_metrics(&self) -> LoadBalancingMetrics {
        self.metrics.read().await.clone()
    }

    /// Update server weights
    pub async fn update_server_weights(&self, server_metrics: HashMap<String, PoolStats>) {
        self.balancer.update_server_weights(server_metrics).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_connection(
        id: &str,
        server_id: &str,
        health: HealthStatus,
        use_count: u64,
        response_time: Option<Duration>,
    ) -> ConnectionInfo {
        ConnectionInfo {
            connection_id: id.to_string(),
            server_id: server_id.to_string(),
            health_status: health,
            response_time,
            use_count,
            is_available: health != HealthStatus::Disconnected,
        }
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let balancer = MCPLoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let connections = vec![
            create_test_connection("conn1", "server1", HealthStatus::Healthy, 0, None),
            create_test_connection("conn2", "server1", HealthStatus::Healthy, 0, None),
            create_test_connection("conn3", "server1", HealthStatus::Healthy, 0, None),
        ];

        // Test round-robin selection
        let first = balancer
            .select_connection("server1", &connections)
            .await
            .unwrap();
        let second = balancer
            .select_connection("server1", &connections)
            .await
            .unwrap();
        let third = balancer
            .select_connection("server1", &connections)
            .await
            .unwrap();
        let fourth = balancer
            .select_connection("server1", &connections)
            .await
            .unwrap();

        assert!(first.is_some());
        assert!(second.is_some());
        assert!(third.is_some());
        assert_eq!(first, fourth); // Should wrap around
    }

    #[tokio::test]
    async fn test_least_connections_selection() {
        let balancer = MCPLoadBalancer::new(LoadBalancingStrategy::LeastConnections);

        let connections = vec![
            create_test_connection("conn1", "server1", HealthStatus::Healthy, 5, None),
            create_test_connection("conn2", "server1", HealthStatus::Healthy, 2, None),
            create_test_connection("conn3", "server1", HealthStatus::Healthy, 8, None),
        ];

        let selected = balancer
            .select_connection("server1", &connections)
            .await
            .unwrap();
        assert_eq!(selected, Some("conn2".to_string())); // Least used connection
    }

    #[tokio::test]
    async fn test_health_based_selection() {
        let balancer = MCPLoadBalancer::new(LoadBalancingStrategy::HealthBased);

        let connections = vec![
            create_test_connection(
                "conn1",
                "server1",
                HealthStatus::Degraded,
                0,
                Some(Duration::from_millis(200)),
            ),
            create_test_connection(
                "conn2",
                "server1",
                HealthStatus::Healthy,
                0,
                Some(Duration::from_millis(100)),
            ),
            create_test_connection(
                "conn3",
                "server1",
                HealthStatus::Unhealthy,
                0,
                Some(Duration::from_millis(300)),
            ),
        ];

        // Health-based selection should prefer healthy connections
        // We'll run this multiple times since it's weighted random
        let mut healthy_selected = 0;
        for _ in 0..10 {
            if let Some(selected) = balancer
                .select_connection("server1", &connections)
                .await
                .unwrap()
            {
                if selected == "conn2" {
                    healthy_selected += 1;
                }
            }
        }

        // Healthy connection should be selected more often
        assert!(healthy_selected > 3);
    }

    #[tokio::test]
    async fn test_no_available_connections() {
        let balancer = MCPLoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let connections =
            vec![create_test_connection("conn1", "server1", HealthStatus::Disconnected, 0, None)];

        let selected = balancer
            .select_connection("server1", &connections)
            .await
            .unwrap();
        assert!(selected.is_none());
    }

    #[tokio::test]
    async fn test_advanced_load_balancer() {
        let balancer = AdvancedMCPLoadBalancer::new(LoadBalancingStrategy::HealthBased);

        // Set server priority
        balancer
            .set_server_priority("server1".to_string(), 10)
            .await;
        balancer.set_server_priority("server2".to_string(), 5).await;

        // Set client affinity
        balancer
            .set_client_affinity("client1".to_string(), "server1".to_string())
            .await;

        let mut available_servers = HashMap::new();
        available_servers.insert(
            "server1".to_string(),
            vec![create_test_connection("conn1", "server1", HealthStatus::Healthy, 0, None)],
        );
        available_servers.insert(
            "server2".to_string(),
            vec![create_test_connection("conn2", "server2", HealthStatus::Healthy, 0, None)],
        );

        // Test affinity selection
        let result = balancer
            .select_connection_advanced(Some("client1"), &available_servers)
            .await
            .unwrap();

        assert!(result.is_some());
        let (server_id, _) = result.unwrap();
        assert_eq!(server_id, "server1"); // Should prefer server1 due to affinity
    }
}
