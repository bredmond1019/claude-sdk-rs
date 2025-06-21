#[cfg(test)]
mod tests {
    use claude_sdk_rs_mcp::config::*;
    use claude_sdk_rs_mcp::transport::TransportType;
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn test_mcp_config_default() {
        let config = MCPConfig::default();

        assert!(!config.enabled);
        assert_eq!(config.client_name, "ai-workflow-system");
        assert_eq!(config.client_version, "1.0.0");
        assert!(config.servers.is_empty());
    }

    #[test]
    fn test_mcp_server_config() {
        let mut servers = HashMap::new();
        servers.insert(
            "test-server".to_string(),
            MCPServerConfig {
                name: "test-server".to_string(),
                enabled: true,
                transport: TransportType::WebSocket {
                    url: "ws://localhost:8080".to_string(),
                    heartbeat_interval: Some(Duration::from_secs(30)),
                    reconnect_config: claude_ai_mcp::transport::ReconnectConfig::default(),
                },
                auto_connect: true,
                retry_on_failure: true,
            },
        );

        let config = MCPConfig {
            enabled: true,
            client_name: "test-client".to_string(),
            client_version: "1.0.0".to_string(),
            connection_pool: Default::default(),
            servers,
        };

        assert_eq!(config.servers.len(), 1);
        assert!(config.servers.contains_key("test-server"));

        let server_config = &config.servers["test-server"];
        assert!(server_config.enabled);
        assert!(server_config.auto_connect);
        assert!(server_config.retry_on_failure);
    }

    #[test]
    fn test_connection_config() {
        let config = claude_ai_mcp::connection_pool::ConnectionConfig {
            max_connections_per_server: 10,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            retry_attempts: 3,
            retry_delay: Duration::from_millis(1000),
            health_check_interval: Duration::from_secs(60),
            enable_load_balancing: true,
            load_balancing_strategy:
                claude_ai_mcp::connection_pool::LoadBalancingStrategy::RoundRobin,
            circuit_breaker: Default::default(),
            health_monitoring: Default::default(),
            enable_auto_reconnect: true,
            backoff_config: Default::default(),
        };

        assert_eq!(config.max_connections_per_server, 10);
        assert_eq!(config.connection_timeout, Duration::from_secs(30));
        assert_eq!(config.idle_timeout, Duration::from_secs(300));
        assert_eq!(config.retry_attempts, 3);
        assert_eq!(config.retry_delay, Duration::from_millis(1000));
        assert_eq!(config.health_check_interval, Duration::from_secs(60));
    }

    #[test]
    fn test_backoff_config() {
        let config = claude_ai_mcp::connection_pool::BackoffConfig {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            jitter: false,
        };

        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(60));
        assert_eq!(config.multiplier, 2.0);
        assert!(!config.jitter);
    }

    #[test]
    fn test_load_balancing_strategy() {
        let strategies = vec![
            claude_ai_mcp::connection_pool::LoadBalancingStrategy::RoundRobin,
            claude_ai_mcp::connection_pool::LoadBalancingStrategy::Random,
            claude_ai_mcp::connection_pool::LoadBalancingStrategy::LeastConnections,
            claude_ai_mcp::connection_pool::LoadBalancingStrategy::HealthBased,
        ];

        for strategy in strategies {
            match strategy {
                claude_ai_mcp::connection_pool::LoadBalancingStrategy::RoundRobin => {}
                claude_ai_mcp::connection_pool::LoadBalancingStrategy::Random => {}
                claude_ai_mcp::connection_pool::LoadBalancingStrategy::LeastConnections => {}
                claude_ai_mcp::connection_pool::LoadBalancingStrategy::HealthBased => {}
            }
        }
    }
}
