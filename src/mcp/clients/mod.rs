use async_trait::async_trait;
use std::collections::HashMap;

pub mod connection;
pub mod helpscout;
pub mod http;
pub mod notion;
pub mod services;
pub mod slack;
pub mod stdio;
pub mod websocket;

pub use connection::MCPConnection;
pub use http::HttpMCPClient;
pub use stdio::StdioMCPClient;
pub use websocket::WebSocketMCPClient;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::protocol::{CallToolResult, ToolDefinition};
use crate::mcp::transport::TransportType;

#[async_trait]
pub trait MCPClient: Send + Sync + std::fmt::Debug {
    async fn connect(&mut self) -> Result<(), WorkflowError>;
    async fn initialize(
        &mut self,
        client_name: &str,
        client_version: &str,
    ) -> Result<(), WorkflowError>;
    async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, WorkflowError>;
    async fn call_tool(
        &mut self,
        name: &str,
        arguments: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<CallToolResult, WorkflowError>;
    async fn disconnect(&mut self) -> Result<(), WorkflowError>;
    fn is_connected(&self) -> bool;
}

/// Create an MCP client based on transport type
pub fn create_mcp_client(
    transport: TransportType,
) -> Result<Box<dyn MCPClient + Send + Sync>, WorkflowError> {
    match transport {
        TransportType::Stdio { command, args, .. } => {
            Ok(Box::new(StdioMCPClient::new(command, args)))
        }
        TransportType::WebSocket { url, .. } => Ok(Box::new(WebSocketMCPClient::new(url))),
        TransportType::Http { base_url, .. } => Ok(Box::new(HttpMCPClient::new(base_url))),
    }
}
