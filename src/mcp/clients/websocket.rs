use async_trait::async_trait;
use std::collections::HashMap;
// use std::sync::Arc;
// use tokio::sync::Mutex;
use uuid::Uuid;

use crate::mcp::clients::connection::MCPConnection;
use crate::mcp::clients::MCPClient;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::protocol::{
    CallToolResult, ClientCapabilities, ClientInfo, InitializeParams, MCPRequest, MCPResponse,
    ResponseResult, ToolCallParams, ToolDefinition,
};
use crate::mcp::transport::WebSocketTransport;

#[derive(Debug)]
pub struct WebSocketMCPClient {
    connection: Option<MCPConnection>,
    url: String,
}

impl WebSocketMCPClient {
    pub fn new(url: String) -> Self {
        Self {
            connection: None,
            url,
        }
    }
}

#[async_trait]
impl MCPClient for WebSocketMCPClient {
    async fn connect(&mut self) -> Result<(), WorkflowError> {
        let transport = Box::new(WebSocketTransport::new(self.url.clone()));
        let mut connection = MCPConnection::new(transport);

        connection.transport.connect().await?;
        connection.is_connected = true;

        self.connection = Some(connection);
        Ok(())
    }

    async fn initialize(
        &mut self,
        client_name: &str,
        client_version: &str,
    ) -> Result<(), WorkflowError> {
        let connection =
            self.connection
                .as_mut()
                .ok_or_else(|| WorkflowError::MCPConnectionError {
                    message: "Not connected".to_string(),
                })?;

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

        let response = connection.send_request(request).await?;
        match response {
            MCPResponse::Result {
                result: ResponseResult::Initialize(_),
                ..
            } => {
                connection.is_initialized = true;

                // Send initialized notification
                let initialized = MCPRequest::Initialized;
                connection.transport.send(initialized).await?;

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
        let connection =
            self.connection
                .as_mut()
                .ok_or_else(|| WorkflowError::MCPConnectionError {
                    message: "Not connected".to_string(),
                })?;

        if !connection.is_initialized {
            return Err(WorkflowError::MCPError {
                message: "Client not initialized".to_string(),
            });
        }

        let request = MCPRequest::ListTools {
            id: Uuid::new_v4().to_string(),
        };

        let response = connection.send_request(request).await?;
        match response {
            MCPResponse::Result {
                result: ResponseResult::ListTools(tools_result),
                ..
            } => Ok(tools_result.tools),
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
        let connection =
            self.connection
                .as_mut()
                .ok_or_else(|| WorkflowError::MCPConnectionError {
                    message: "Not connected".to_string(),
                })?;

        if !connection.is_initialized {
            return Err(WorkflowError::MCPError {
                message: "Client not initialized".to_string(),
            });
        }

        let request = MCPRequest::CallTool {
            id: Uuid::new_v4().to_string(),
            params: ToolCallParams {
                name: name.to_string(),
                arguments,
            },
        };

        let response = connection.send_request(request).await?;
        match response {
            MCPResponse::Result {
                result: ResponseResult::CallTool(call_result),
                ..
            } => Ok(call_result),
            MCPResponse::Error { error, .. } => Err(WorkflowError::MCPError {
                message: format!("Tool call failed: {}", error.message),
            }),
            _ => Err(WorkflowError::MCPProtocolError {
                message: "Unexpected response to call_tool".to_string(),
            }),
        }
    }

    async fn disconnect(&mut self) -> Result<(), WorkflowError> {
        if let Some(mut connection) = self.connection.take() {
            connection.transport.disconnect().await?;
            connection.is_connected = false;
            connection.is_initialized = false;
        }
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connection
            .as_ref()
            .map(|c| c.is_connected)
            .unwrap_or(false)
    }
}
