// use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::protocol::{
    MCPRequest,
    MCPResponse,
    // CallToolResult, ClientCapabilities, ClientInfo, InitializeParams,
    // ResponseResult, ToolCallParams, ToolDefinition,
};
use crate::mcp::transport::MCPTransport;
// StdioTransport, WebSocketTransport unused for now

pub struct MCPConnection {
    pub transport: Box<dyn MCPTransport>,
    pub is_connected: bool,
    pub is_initialized: bool,
    pending_requests: Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<MCPResponse>>>>,
}

impl std::fmt::Debug for MCPConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MCPConnection")
            .field("is_connected", &self.is_connected)
            .field("is_initialized", &self.is_initialized)
            .finish()
    }
}

impl MCPConnection {
    pub fn new(transport: Box<dyn MCPTransport>) -> Self {
        Self {
            transport,
            is_connected: false,
            is_initialized: false,
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn send_request(
        &mut self,
        request: MCPRequest,
    ) -> Result<MCPResponse, WorkflowError> {
        let id = request
            .get_id()
            .map(|id| id.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.insert(id.clone(), tx);
        }

        self.transport.send(request).await?;

        match rx.await {
            Ok(response) => Ok(response),
            Err(_) => Err(WorkflowError::MCPError {
                message: "Request timeout or connection closed".to_string(),
            }),
        }
    }

    async fn receive_response(&mut self) -> Result<(), WorkflowError> {
        let response = self.transport.receive().await?;
        let id = response.get_id().to_string();

        let mut pending = self.pending_requests.lock().await;
        if let Some(tx) = pending.remove(&id) {
            let _ = tx.send(response);
        }

        Ok(())
    }
}
