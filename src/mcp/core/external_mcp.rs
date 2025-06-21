use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

use crate::mcp::clients::MCPClient;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::protocol::{CallToolResult, ToolDefinition};
use crate::mcp::transport::TransportType;

/// Authentication configuration for external MCP servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub api_key: Option<String>,
    pub token: Option<String>,
    pub headers: Option<HashMap<String, String>>,
}

/// Retry configuration for external MCP operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            exponential_base: 2.0,
        }
    }
}

/// Configuration for external MCP clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalMCPConfig {
    pub service_name: String,
    pub transport: TransportType,
    pub auth: Option<AuthConfig>,
    pub retry_config: RetryConfig,
}

/// Base implementation for external MCP clients
pub struct BaseExternalMCPClient {
    config: ExternalMCPConfig,
    client: Option<Box<dyn MCPClient + Send + Sync>>,
}

impl std::fmt::Debug for BaseExternalMCPClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseExternalMCPClient")
            .field("config", &self.config)
            .field("client", &self.client.is_some())
            .finish()
    }
}

impl BaseExternalMCPClient {
    pub fn new(config: ExternalMCPConfig) -> Self {
        Self {
            config,
            client: None,
        }
    }

    pub fn get_config(&self) -> &ExternalMCPConfig {
        &self.config
    }

    pub async fn connect(&mut self) -> Result<(), WorkflowError> {
        // Create appropriate client based on transport type
        let mut client = crate::mcp::clients::create_mcp_client(self.config.transport.clone())?;
        client.connect().await?;

        // Initialize with service name
        client
            .initialize(&self.config.service_name, "1.0.0")
            .await?;

        self.client = Some(client);
        Ok(())
    }

    pub async fn disconnect(&mut self) -> Result<(), WorkflowError> {
        if let Some(mut client) = self.client.take() {
            client.disconnect().await?;
        }
        Ok(())
    }

    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    pub async fn execute_tool(
        &mut self,
        tool_name: &str,
        arguments: Option<HashMap<String, Value>>,
    ) -> Result<CallToolResult, WorkflowError> {
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| WorkflowError::MCPError {
                message: "Client not connected".to_string(),
            })?;

        client.call_tool(tool_name, arguments).await
    }

    pub async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, WorkflowError> {
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| WorkflowError::MCPError {
                message: "Client not connected".to_string(),
            })?;

        client.list_tools().await
    }
}

#[async_trait]
impl Node for BaseExternalMCPClient {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract tool name and arguments from input
        let tool_name = input.get("tool").and_then(|v| v.as_str()).ok_or_else(|| {
            WorkflowError::ValidationError {
                message: "Missing tool name in input".to_string(),
            }
        })?;

        let arguments = input
            .get("arguments")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect());

        // Clone self to get a mutable version for the tool call
        let mut client = BaseExternalMCPClient::new(self.config.clone());
        if !client.is_connected() {
            client.connect().await?;
        }

        let result = client.execute_tool(tool_name, arguments).await?;

        Ok(serde_json::to_value(result)?)
    }

    fn name(&self) -> &str {
        &self.config.service_name
    }
}

/// Trait for external MCP client nodes
#[async_trait]
pub trait ExternalMCPClientNode: Send + Sync {
    fn get_config(&self) -> &ExternalMCPConfig;

    async fn connect(&mut self) -> Result<(), WorkflowError>;

    async fn execute_tool(
        &mut self,
        tool_name: &str,
        arguments: Option<HashMap<String, Value>>,
    ) -> Result<CallToolResult, WorkflowError>;

    async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, WorkflowError>;

    async fn disconnect(&mut self) -> Result<(), WorkflowError>;

    fn is_connected(&self) -> bool;
}
