use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::time::Duration;

use crate::mcp::connection_pool::ConnectionConfig;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::transport::TransportType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    pub enabled: bool,
    pub client_name: String,
    pub client_version: String,
    pub connection_pool: ConnectionConfig,
    pub servers: HashMap<String, MCPServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    pub name: String,
    pub enabled: bool,
    pub transport: TransportType,
    pub auto_connect: bool,
    pub retry_on_failure: bool,
}

impl MCPConfig {
    pub fn from_env() -> Result<Self, WorkflowError> {
        let enabled = Self::get_enabled_from_env();
        let client_name = Self::get_client_name_from_env();
        let client_version = Self::get_client_version_from_env();
        let connection_pool = Self::get_connection_pool_from_env();
        let servers = Self::load_servers_from_env()?;

        Ok(MCPConfig {
            enabled,
            client_name,
            client_version,
            connection_pool,
            servers,
        })
    }

    fn get_enabled_from_env() -> bool {
        env::var("MCP_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false)
    }

    fn get_client_name_from_env() -> String {
        env::var("MCP_CLIENT_NAME").unwrap_or_else(|_| "ai-workflow-system".to_string())
    }

    fn get_client_version_from_env() -> String {
        env::var("MCP_CLIENT_VERSION").unwrap_or_else(|_| "1.0.0".to_string())
    }

    fn get_connection_pool_from_env() -> ConnectionConfig {
        ConnectionConfig {
            max_connections_per_server: Self::get_env_var_or_default(
                "MCP_MAX_CONNECTIONS_PER_SERVER",
                "5",
                5,
            ),
            connection_timeout: Duration::from_secs(Self::get_env_var_or_default(
                "MCP_CONNECTION_TIMEOUT_SECONDS",
                "30",
                30,
            )),
            idle_timeout: Duration::from_secs(Self::get_env_var_or_default(
                "MCP_IDLE_TIMEOUT_SECONDS",
                "300",
                300,
            )),
            retry_attempts: Self::get_env_var_or_default("MCP_RETRY_ATTEMPTS", "3", 3),
            retry_delay: Duration::from_millis(Self::get_env_var_or_default(
                "MCP_RETRY_DELAY_MS",
                "1000",
                1000,
            )),
            health_check_interval: Duration::from_secs(Self::get_env_var_or_default(
                "MCP_HEALTH_CHECK_INTERVAL_SECONDS",
                "60",
                60,
            )),
            enable_load_balancing: Self::get_env_var_or_default(
                "MCP_ENABLE_LOAD_BALANCING",
                "true",
                true,
            ),
            load_balancing_strategy:
                crate::mcp::connection_pool::LoadBalancingStrategy::HealthBased,
            circuit_breaker:
                crate::mcp::core::error::circuit_breaker::CircuitBreakerConfig::default(),
            health_monitoring: crate::mcp::health::HealthConfig::default(),
            enable_auto_reconnect: Self::get_env_var_or_default(
                "MCP_ENABLE_AUTO_RECONNECT",
                "true",
                true,
            ),
            backoff_config: crate::mcp::connection_pool::BackoffConfig::default(),
        }
    }

    fn get_env_var_or_default<T: std::str::FromStr + std::default::Default>(
        key: &str,
        default: &str,
        fallback: T,
    ) -> T
    where
        T::Err: std::fmt::Debug,
    {
        env::var(key)
            .unwrap_or_else(|_| default.to_string())
            .parse()
            .unwrap_or(fallback)
    }

    fn load_servers_from_env() -> Result<HashMap<String, MCPServerConfig>, WorkflowError> {
        let mut servers = HashMap::new();

        Self::load_customer_support_server(&mut servers)?;
        Self::load_external_servers(&mut servers)?;

        Ok(servers)
    }

    fn load_customer_support_server(
        servers: &mut HashMap<String, MCPServerConfig>,
    ) -> Result<(), WorkflowError> {
        if !Self::get_env_var_or_default("MCP_CUSTOMER_SUPPORT_ENABLED", "false", false) {
            return Ok(());
        }

        let transport = Self::create_customer_support_transport()?;

        servers.insert(
            "customer-support".to_string(),
            MCPServerConfig {
                name: "customer-support".to_string(),
                enabled: true,
                transport,
                auto_connect: true,
                retry_on_failure: true,
            },
        );

        Ok(())
    }

    fn create_customer_support_transport() -> Result<TransportType, WorkflowError> {
        let transport_type =
            env::var("MCP_CUSTOMER_SUPPORT_TRANSPORT").unwrap_or_else(|_| "stdio".to_string());

        match transport_type.as_str() {
            "stdio" => {
                let command = env::var("MCP_CUSTOMER_SUPPORT_COMMAND")
                    .unwrap_or_else(|_| "python".to_string());
                let args_str = env::var("MCP_CUSTOMER_SUPPORT_ARGS")
                    .unwrap_or_else(|_| "scripts/customer_support_server.py".to_string());
                let args: Vec<String> =
                    args_str.split_whitespace().map(|s| s.to_string()).collect();

                Ok(TransportType::Stdio {
                    command,
                    args,
                    auto_restart: true,
                    max_restarts: 3,
                })
            }
            "websocket" => {
                let url = env::var("MCP_CUSTOMER_SUPPORT_URI")
                    .unwrap_or_else(|_| "ws://localhost:8080/mcp".to_string());
                Ok(TransportType::WebSocket {
                    url,
                    heartbeat_interval: Some(std::time::Duration::from_secs(30)),
                    reconnect_config: crate::mcp::transport::ReconnectConfig::default(),
                })
            }
            _ => Err(WorkflowError::MCPError {
                message: "Invalid transport type for customer support server".to_string(),
            }),
        }
    }

    fn load_external_servers(
        servers: &mut HashMap<String, MCPServerConfig>,
    ) -> Result<(), WorkflowError> {
        let mut server_index = 1;

        while let Ok(name) = env::var(format!("MCP_EXTERNAL_SERVER_{}_NAME", server_index)) {
            if Self::get_env_var_or_default(
                &format!("MCP_EXTERNAL_SERVER_{}_ENABLED", server_index),
                "false",
                false,
            ) {
                let server_config = Self::create_external_server_config(server_index, &name)?;
                servers.insert(name, server_config);
            }
            server_index += 1;
        }

        Ok(())
    }

    fn create_external_server_config(
        server_index: u32,
        name: &str,
    ) -> Result<MCPServerConfig, WorkflowError> {
        let uri_key = format!("MCP_EXTERNAL_SERVER_{}_URI", server_index);
        let transport_key = format!("MCP_EXTERNAL_SERVER_{}_TRANSPORT", server_index);

        let uri = env::var(&uri_key).map_err(|_| WorkflowError::MCPError {
            message: format!("Missing URI for external server {}", name),
        })?;

        let transport_str = env::var(&transport_key).unwrap_or_else(|_| "websocket".to_string());
        let transport =
            Self::create_transport_for_external_server(server_index, &transport_str, uri)?;

        Ok(MCPServerConfig {
            name: name.to_string(),
            enabled: true,
            transport,
            auto_connect: true,
            retry_on_failure: true,
        })
    }

    fn create_transport_for_external_server(
        server_index: u32,
        transport_str: &str,
        uri: String,
    ) -> Result<TransportType, WorkflowError> {
        match transport_str {
            "websocket" => Ok(TransportType::WebSocket {
                url: uri,
                heartbeat_interval: Some(std::time::Duration::from_secs(30)),
                reconnect_config: crate::mcp::transport::ReconnectConfig::default(),
            }),
            "stdio" => {
                let command_key = format!("MCP_EXTERNAL_SERVER_{}_COMMAND", server_index);
                let args_key = format!("MCP_EXTERNAL_SERVER_{}_ARGS", server_index);

                let command = env::var(&command_key).unwrap_or_else(|_| "node".to_string());
                let args_str = env::var(&args_key).unwrap_or_else(|_| "".to_string());
                let args: Vec<String> = if args_str.is_empty() {
                    vec![]
                } else {
                    args_str.split_whitespace().map(|s| s.to_string()).collect()
                };

                Ok(TransportType::Stdio {
                    command,
                    args,
                    auto_restart: true,
                    max_restarts: 3,
                })
            }
            "http" => Ok(TransportType::Http {
                base_url: uri,
                pool_config: crate::mcp::transport::HttpPoolConfig::default(),
            }),
            _ => Err(WorkflowError::MCPError {
                message: format!(
                    "Invalid transport type '{}' for server {}",
                    transport_str, server_index
                ),
            }),
        }
    }

    pub fn get_server_config(&self, server_name: &str) -> Option<&MCPServerConfig> {
        self.servers.get(server_name)
    }

    pub fn get_enabled_servers(&self) -> Vec<&MCPServerConfig> {
        self.servers
            .values()
            .filter(|config| config.enabled)
            .collect()
    }

    pub fn is_server_enabled(&self, server_name: &str) -> bool {
        self.servers
            .get(server_name)
            .map(|config| config.enabled)
            .unwrap_or(false)
    }
}

impl Default for MCPConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            client_name: "ai-workflow-system".to_string(),
            client_version: "1.0.0".to_string(),
            connection_pool: ConnectionConfig::default(),
            servers: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::env;

    #[test]
    fn test_mcp_config_default() {
        let config = MCPConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.client_name, "ai-workflow-system");
        assert_eq!(config.client_version, "1.0.0");
        assert!(config.servers.is_empty());
    }

    #[test]
    #[serial]
    fn test_mcp_config_from_env_disabled() {
        // Clear all MCP environment variables
        unsafe {
            env::remove_var("MCP_ENABLED");
            env::remove_var("MCP_CLIENT_NAME");
            env::remove_var("MCP_CLIENT_VERSION");
            env::remove_var("MCP_CUSTOMER_SUPPORT_ENABLED");
            // Clean up any external server variables
            for i in 1..10 {
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_NAME", i));
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_ENABLED", i));
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_URI", i));
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_TRANSPORT", i));
            }
        };

        let config = MCPConfig::from_env().unwrap();
        assert!(!config.enabled);
    }

    #[test]
    #[serial]
    fn test_mcp_config_from_env_enabled() {
        unsafe {
            env::set_var("MCP_ENABLED", "true");
            env::set_var("MCP_CLIENT_NAME", "test-client");
            env::set_var("MCP_CLIENT_VERSION", "2.0.0");
        }

        let config = MCPConfig::from_env().unwrap();
        assert!(config.enabled);
        assert_eq!(config.client_name, "test-client");
        assert_eq!(config.client_version, "2.0.0");

        // Cleanup
        unsafe {
            env::remove_var("MCP_ENABLED");
            env::remove_var("MCP_CLIENT_NAME");
            env::remove_var("MCP_CLIENT_VERSION");
        }
    }

    #[test]
    #[serial]
    fn test_customer_support_server_config() {
        // Clean up environment variables first
        unsafe {
            env::remove_var("MCP_CUSTOMER_SUPPORT_ENABLED");
            env::remove_var("MCP_CUSTOMER_SUPPORT_TRANSPORT");
            env::remove_var("MCP_CUSTOMER_SUPPORT_COMMAND");
            env::remove_var("MCP_CUSTOMER_SUPPORT_ARGS");
            // Clean up any external server variables that might interfere
            for i in 1..10 {
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_NAME", i));
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_ENABLED", i));
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_URI", i));
                env::remove_var(format!("MCP_EXTERNAL_SERVER_{}_TRANSPORT", i));
            }
        }

        unsafe {
            env::set_var("MCP_CUSTOMER_SUPPORT_ENABLED", "true");
            env::set_var("MCP_CUSTOMER_SUPPORT_TRANSPORT", "stdio");
            env::set_var("MCP_CUSTOMER_SUPPORT_COMMAND", "python3");
            env::set_var("MCP_CUSTOMER_SUPPORT_ARGS", "scripts/server.py --port 8080");
        }

        let config = MCPConfig::from_env().unwrap();

        assert!(config.is_server_enabled("customer-support"));
        let server_config = config.get_server_config("customer-support").unwrap();
        assert_eq!(server_config.name, "customer-support");
        assert!(server_config.enabled);

        match &server_config.transport {
            TransportType::Stdio { command, args, .. } => {
                assert_eq!(command, "python3");
                assert_eq!(args, &vec!["scripts/server.py", "--port", "8080"]);
            }
            _ => panic!("Expected Stdio transport"),
        }

        // Cleanup
        unsafe {
            env::remove_var("MCP_CUSTOMER_SUPPORT_ENABLED");
            env::remove_var("MCP_CUSTOMER_SUPPORT_TRANSPORT");
            env::remove_var("MCP_CUSTOMER_SUPPORT_COMMAND");
            env::remove_var("MCP_CUSTOMER_SUPPORT_ARGS");
        }
    }

    #[test]
    #[serial]
    fn test_external_server_config() {
        // Cleanup any existing environment variables first
        unsafe {
            env::remove_var("MCP_EXTERNAL_SERVER_1_NAME");
            env::remove_var("MCP_EXTERNAL_SERVER_1_ENABLED");
            env::remove_var("MCP_EXTERNAL_SERVER_1_URI");
            env::remove_var("MCP_EXTERNAL_SERVER_1_TRANSPORT");
            env::remove_var("MCP_CUSTOMER_SUPPORT_ENABLED");
        }

        unsafe {
            env::set_var("MCP_EXTERNAL_SERVER_1_NAME", "test-server");
            env::set_var("MCP_EXTERNAL_SERVER_1_ENABLED", "true");
            env::set_var("MCP_EXTERNAL_SERVER_1_URI", "ws://localhost:9090/mcp");
            env::set_var("MCP_EXTERNAL_SERVER_1_TRANSPORT", "websocket");
        }

        let config = MCPConfig::from_env().unwrap();

        assert!(config.is_server_enabled("test-server"));
        let server_config = config.get_server_config("test-server").unwrap();
        assert_eq!(server_config.name, "test-server");

        match &server_config.transport {
            TransportType::WebSocket { url, .. } => {
                assert_eq!(url, "ws://localhost:9090/mcp");
            }
            _ => panic!("Expected WebSocket transport"),
        }

        // Cleanup
        unsafe {
            env::remove_var("MCP_EXTERNAL_SERVER_1_NAME");
            env::remove_var("MCP_EXTERNAL_SERVER_1_ENABLED");
            env::remove_var("MCP_EXTERNAL_SERVER_1_URI");
            env::remove_var("MCP_EXTERNAL_SERVER_1_TRANSPORT");
        }
    }

    #[test]
    #[serial]
    fn test_get_enabled_servers() {
        // Cleanup any existing environment variables first
        unsafe {
            env::remove_var("MCP_EXTERNAL_SERVER_1_NAME");
            env::remove_var("MCP_EXTERNAL_SERVER_1_ENABLED");
            env::remove_var("MCP_EXTERNAL_SERVER_1_URI");
            env::remove_var("MCP_EXTERNAL_SERVER_1_TRANSPORT");
            env::remove_var("MCP_EXTERNAL_SERVER_2_NAME");
            env::remove_var("MCP_EXTERNAL_SERVER_2_ENABLED");
            env::remove_var("MCP_EXTERNAL_SERVER_2_URI");
            env::remove_var("MCP_EXTERNAL_SERVER_2_TRANSPORT");
            env::remove_var("MCP_CUSTOMER_SUPPORT_ENABLED");
        }

        unsafe {
            env::set_var("MCP_EXTERNAL_SERVER_1_NAME", "server1");
            env::set_var("MCP_EXTERNAL_SERVER_1_ENABLED", "true");
            env::set_var("MCP_EXTERNAL_SERVER_1_URI", "ws://localhost:8080");
            env::set_var("MCP_EXTERNAL_SERVER_1_TRANSPORT", "websocket");

            env::set_var("MCP_EXTERNAL_SERVER_2_NAME", "server2");
            env::set_var("MCP_EXTERNAL_SERVER_2_ENABLED", "false");
            env::set_var("MCP_EXTERNAL_SERVER_2_URI", "ws://localhost:8081");
            env::set_var("MCP_EXTERNAL_SERVER_2_TRANSPORT", "websocket");
        }

        let config = MCPConfig::from_env().unwrap();
        let enabled_servers = config.get_enabled_servers();

        assert_eq!(enabled_servers.len(), 1);
        assert_eq!(enabled_servers[0].name, "server1");

        // Cleanup
        unsafe {
            env::remove_var("MCP_EXTERNAL_SERVER_1_NAME");
            env::remove_var("MCP_EXTERNAL_SERVER_1_ENABLED");
            env::remove_var("MCP_EXTERNAL_SERVER_1_URI");
            env::remove_var("MCP_EXTERNAL_SERVER_1_TRANSPORT");
            env::remove_var("MCP_EXTERNAL_SERVER_2_NAME");
            env::remove_var("MCP_EXTERNAL_SERVER_2_ENABLED");
            env::remove_var("MCP_EXTERNAL_SERVER_2_URI");
            env::remove_var("MCP_EXTERNAL_SERVER_2_TRANSPORT");
        }
    }
}
