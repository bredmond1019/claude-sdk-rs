use async_trait::async_trait;
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

use crate::mcp::clients::MCPClient;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::protocol::{
    CallToolResult, ClientCapabilities, ClientInfo, InitializeParams, MCPRequest, MCPResponse,
    ResponseResult, ToolCallParams, ToolDefinition,
};
use crate::mcp::transport::HttpTransport;

/// HTTP-based MCP client for cross-system communication
///
/// This client enables communication with MCP servers via HTTP requests,
/// making it suitable for cross-system communication where services
/// need to interact with remote MCP servers over HTTP APIs.
///
/// Unlike WebSocket or Stdio transports, HTTP transport is stateless
/// and each operation is a separate request-response cycle.
#[derive(Debug)]
pub struct HttpMCPClient {
    transport: HttpTransport,
    base_url: String,
    is_initialized: bool,
    client_name: String,
    client_version: String,
}

impl HttpMCPClient {
    /// Create a new HTTP MCP client
    pub fn new(base_url: String) -> Self {
        let transport = HttpTransport::new(base_url.clone());
        Self {
            transport,
            base_url,
            is_initialized: false,
            client_name: "ai-workflow-system".to_string(),
            client_version: "1.0.0".to_string(),
        }
    }

    /// Create a new HTTP MCP client with authentication token
    pub fn with_auth_token(base_url: String, auth_token: String) -> Self {
        let transport = HttpTransport::new(base_url.clone()).with_auth_token(auth_token);
        Self {
            transport,
            base_url,
            is_initialized: false,
            client_name: "ai-workflow-system".to_string(),
            client_version: "1.0.0".to_string(),
        }
    }

    /// Set the authentication token
    pub fn set_auth_token(&mut self, token: Option<String>) {
        self.transport.set_auth_token(token);
    }

    /// Set client identification
    pub fn set_client_info(&mut self, name: String, version: String) {
        self.client_name = name;
        self.client_version = version;
    }

    /// Send a request and get response using HTTP transport
    async fn send_http_request(&self, request: MCPRequest) -> Result<MCPResponse, WorkflowError> {
        self.transport
            .send_request(request)
            .await
            .map_err(|e| WorkflowError::MCPTransportError {
                message: format!("HTTP request failed: {}", e),
            })
    }

    /// Check if client is properly initialized
    fn ensure_initialized(&self) -> Result<(), WorkflowError> {
        if !self.is_initialized {
            return Err(WorkflowError::MCPError {
                message: "Client not initialized. Call initialize() first.".to_string(),
            });
        }
        Ok(())
    }
}

#[async_trait]
impl MCPClient for HttpMCPClient {
    async fn connect(&mut self) -> Result<(), WorkflowError> {
        // HTTP transport doesn't need explicit connection establishment
        // We consider it "connected" if the base URL is valid
        // The actual connectivity test happens during the first request

        debug!("HTTP MCP Client connected to: {}", self.base_url);
        Ok(())
    }

    async fn initialize(
        &mut self,
        client_name: &str,
        client_version: &str,
    ) -> Result<(), WorkflowError> {
        let request = MCPRequest::Initialize {
            id: Uuid::new_v4().to_string(),
            params: InitializeParams {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ClientCapabilities {
                    roots: None,
                    sampling: None,
                },
                client_info: ClientInfo {
                    name: client_name.to_string(),
                    version: client_version.to_string(),
                },
            },
        };

        let response = self.send_http_request(request).await?;

        match response {
            MCPResponse::Result {
                result: ResponseResult::Initialize(_),
                ..
            } => {
                self.is_initialized = true;
                self.client_name = client_name.to_string();
                self.client_version = client_version.to_string();

                // For HTTP transport, we don't need to send a separate initialized notification
                // as each request is stateless

                info!("HTTP MCP Client initialized successfully");
                Ok(())
            }
            MCPResponse::Error { error, .. } => Err(WorkflowError::MCPError {
                message: format!("Initialize failed: {}", error.message),
            }),
            _ => Err(WorkflowError::MCPProtocolError {
                message: "Unexpected response to initialize".to_string(),
            }),
        }
    }

    async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, WorkflowError> {
        self.ensure_initialized()?;

        let request = MCPRequest::ListTools {
            id: Uuid::new_v4().to_string(),
        };

        let response = self.send_http_request(request).await?;

        match response {
            MCPResponse::Result {
                result: ResponseResult::ListTools(tools_result),
                ..
            } => {
                debug!(
                    "Listed {} tools from HTTP MCP server",
                    tools_result.tools.len()
                );
                Ok(tools_result.tools)
            }
            MCPResponse::Error { error, .. } => Err(WorkflowError::MCPError {
                message: format!("List tools failed: {}", error.message),
            }),
            _ => Err(WorkflowError::MCPProtocolError {
                message: "Unexpected response to list_tools".to_string(),
            }),
        }
    }

    async fn call_tool(
        &mut self,
        name: &str,
        arguments: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<CallToolResult, WorkflowError> {
        self.ensure_initialized()?;

        let request = MCPRequest::CallTool {
            id: Uuid::new_v4().to_string(),
            params: ToolCallParams {
                name: name.to_string(),
                arguments,
            },
        };

        debug!("Calling tool '{}' via HTTP MCP", name);
        let response = self.send_http_request(request).await?;

        match response {
            MCPResponse::Result {
                result: ResponseResult::CallTool(call_result),
                ..
            } => {
                debug!("Tool '{}' called successfully via HTTP MCP", name);
                Ok(call_result)
            }
            MCPResponse::Error { error, .. } => Err(WorkflowError::MCPError {
                message: format!("Tool call '{}' failed: {}", name, error.message),
            }),
            _ => Err(WorkflowError::MCPProtocolError {
                message: format!("Unexpected response to call_tool '{}'", name),
            }),
        }
    }

    async fn disconnect(&mut self) -> Result<(), WorkflowError> {
        // HTTP transport doesn't need explicit disconnection
        self.is_initialized = false;
        debug!("HTTP MCP Client disconnected");
        Ok(())
    }

    fn is_connected(&self) -> bool {
        // For HTTP transport, we consider it "connected" if we can make requests
        // The actual connectivity test happens during requests
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_mcp_client_creation() {
        let client = HttpMCPClient::new("http://localhost:8080".to_string());
        assert_eq!(client.base_url, "http://localhost:8080");
        assert!(!client.is_initialized);
        assert!(client.is_connected()); // HTTP is always "connected"
    }

    #[test]
    fn test_http_mcp_client_with_auth() {
        let client = HttpMCPClient::with_auth_token(
            "http://localhost:8080".to_string(),
            "test-token".to_string(),
        );
        assert_eq!(client.base_url, "http://localhost:8080");
        assert!(!client.is_initialized);
    }

    #[test]
    fn test_client_info_setting() {
        let mut client = HttpMCPClient::new("http://localhost:8080".to_string());
        client.set_client_info("test-client".to_string(), "2.0.0".to_string());
        assert_eq!(client.client_name, "test-client");
        assert_eq!(client.client_version, "2.0.0");
    }

    #[tokio::test]
    async fn test_ensure_initialized_fails_when_not_initialized() {
        let client = HttpMCPClient::new("http://localhost:8080".to_string());
        let result = client.ensure_initialized();
        assert!(result.is_err());

        if let Err(WorkflowError::MCPError { message }) = result {
            assert!(message.contains("not initialized"));
        } else {
            panic!("Expected MCPError");
        }
    }
}
