//! Model Context Protocol (MCP) implementation for Claude AI SDK

// Internal modules
mod core;

// Re-export main modules
pub mod clients;
pub mod config;
pub mod connection_pool;
pub mod health;
pub mod load_balancer;
pub mod metrics;
pub mod protocol;
pub mod server;
pub mod service_config;
#[cfg(test)]
pub mod testing;
pub mod transport;

// Re-export key types
pub use clients::{MCPClient, MCPConnection};
pub use config::{MCPConfig, MCPServerConfig};
pub use protocol::{MCPMessage, MCPRequest, MCPResponse, ToolDefinition};
pub use server::{MCPToolServer, ToolMetadata};
pub use service_config::{ServiceConfig, ServiceConfigBuilder};
pub use transport::{MCPTransport, TransportType};

// Keep the original simple config types for backwards compatibility
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub servers: Vec<McpServer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServer {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: std::collections::HashMap<String, String>,
}

impl McpServer {
    pub fn new(command: impl Into<String>, args: Vec<&str>) -> Self {
        Self {
            name: String::new(),
            command: command.into(),
            args: args.into_iter().map(String::from).collect(),
            env: std::collections::HashMap::new(),
        }
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
}
