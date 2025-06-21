// Mock MCP client for testing

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::mcp::clients::MCPClient;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::protocol::{CallToolResult, ToolDefinition};

/// Mock MCP client for testing
#[derive(Debug, Clone)]
pub struct MockMCPClient {
    connected: Arc<Mutex<bool>>,
    tools: Arc<Mutex<Vec<ToolDefinition>>>,
    tool_responses: Arc<Mutex<HashMap<String, CallToolResult>>>,
    call_history: Arc<Mutex<Vec<ToolCall>>>,
    should_fail: Arc<Mutex<Option<WorkflowError>>>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Option<HashMap<String, serde_json::Value>>,
    pub timestamp: std::time::Instant,
}

impl MockMCPClient {
    pub fn new() -> Self {
        Self {
            connected: Arc::new(Mutex::new(false)),
            tools: Arc::new(Mutex::new(Vec::new())),
            tool_responses: Arc::new(Mutex::new(HashMap::new())),
            call_history: Arc::new(Mutex::new(Vec::new())),
            should_fail: Arc::new(Mutex::new(None)),
        }
    }

    /// Configure the mock to return specific tools
    pub async fn with_tools(&self, tools: Vec<ToolDefinition>) {
        let mut mock_tools = self.tools.lock().await;
        *mock_tools = tools;
    }

    /// Configure a response for a specific tool call
    pub async fn expect_tool_call(&self, name: &str, response: CallToolResult) {
        let mut responses = self.tool_responses.lock().await;
        responses.insert(name.to_string(), response);
    }

    /// Configure the mock to fail with a specific error
    pub async fn fail_with(&self, error: WorkflowError) {
        let mut should_fail = self.should_fail.lock().await;
        *should_fail = Some(error);
    }

    /// Get the call history
    pub async fn get_call_history(&self) -> Vec<ToolCall> {
        let history = self.call_history.lock().await;
        history.clone()
    }

    /// Verify a tool was called with specific arguments
    pub async fn verify_tool_called(
        &self,
        name: &str,
        expected_args: Option<HashMap<String, serde_json::Value>>,
    ) -> bool {
        let history = self.call_history.lock().await;
        history.iter().any(|call| {
            call.name == name
                && match (&call.arguments, &expected_args) {
                    (None, None) => true,
                    (Some(actual), Some(expected)) => {
                        expected.iter().all(|(k, v)| actual.get(k) == Some(v))
                    }
                    _ => false,
                }
        })
    }

    /// Reset the mock to initial state
    pub async fn reset(&self) {
        *self.connected.lock().await = false;
        self.tools.lock().await.clear();
        self.tool_responses.lock().await.clear();
        self.call_history.lock().await.clear();
        *self.should_fail.lock().await = None;
    }
}

#[async_trait]
impl MCPClient for MockMCPClient {
    async fn connect(&mut self) -> Result<(), WorkflowError> {
        if let Some(ref error) = *self.should_fail.lock().await {
            return Err(WorkflowError::ExecutionError(format!("{}", error)));
        }

        *self.connected.lock().await = true;
        Ok(())
    }

    async fn initialize(
        &mut self,
        _client_name: &str,
        _client_version: &str,
    ) -> Result<(), WorkflowError> {
        if let Some(ref error) = *self.should_fail.lock().await {
            return Err(WorkflowError::ExecutionError(format!("{}", error)));
        }

        if !*self.connected.lock().await {
            return Err(WorkflowError::ExecutionError("Not connected".to_string()));
        }

        Ok(())
    }

    async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, WorkflowError> {
        if let Some(ref error) = *self.should_fail.lock().await {
            return Err(WorkflowError::ExecutionError(format!("{}", error)));
        }

        if !*self.connected.lock().await {
            return Err(WorkflowError::ExecutionError("Not connected".to_string()));
        }

        let tools = self.tools.lock().await;
        Ok(tools.clone())
    }

    async fn call_tool(
        &mut self,
        name: &str,
        arguments: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<CallToolResult, WorkflowError> {
        if let Some(ref error) = *self.should_fail.lock().await {
            return Err(WorkflowError::ExecutionError(format!("{}", error)));
        }

        if !*self.connected.lock().await {
            return Err(WorkflowError::ExecutionError("Not connected".to_string()));
        }

        // Record the call
        let mut history = self.call_history.lock().await;
        history.push(ToolCall {
            name: name.to_string(),
            arguments: arguments.clone(),
            timestamp: std::time::Instant::now(),
        });

        // Return configured response or default
        let responses = self.tool_responses.lock().await;
        if let Some(response) = responses.get(name) {
            Ok(response.clone())
        } else {
            Ok(CallToolResult {
                content: vec![crate::mcp::protocol::ToolContent::Text {
                    text: format!("Mock response for tool: {}", name),
                }],
                is_error: Some(false),
            })
        }
    }

    async fn disconnect(&mut self) -> Result<(), WorkflowError> {
        if let Some(ref error) = *self.should_fail.lock().await {
            return Err(WorkflowError::ExecutionError(format!("{}", error)));
        }

        *self.connected.lock().await = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        // Use try_lock to avoid async in sync context
        self.connected
            .try_lock()
            .map(|guard| *guard)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_client_basic_flow() {
        let mut client = MockMCPClient::new();

        // Test connection
        assert!(!client.is_connected());
        client.connect().await.unwrap();
        assert!(client.is_connected());

        // Test initialization
        client.initialize("test", "1.0").await.unwrap();

        // Test tools
        let tools = vec![ToolDefinition {
            name: "test_tool".to_string(),
            description: Some("Test tool".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }];
        client.with_tools(tools.clone()).await;

        let listed_tools = client.list_tools().await.unwrap();
        assert_eq!(listed_tools.len(), 1);
        assert_eq!(listed_tools[0].name, "test_tool");

        // Test tool call
        let response = CallToolResult {
            content: vec![crate::mcp::protocol::ToolContent::Text {
                text: "Success".to_string(),
            }],
            is_error: Some(false),
        };
        client.expect_tool_call("test_tool", response.clone()).await;

        let result = client.call_tool("test_tool", None).await.unwrap();
        assert_eq!(result.content, response.content);

        // Verify call history
        assert!(client.verify_tool_called("test_tool", None).await);

        // Test disconnection
        client.disconnect().await.unwrap();
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_mock_client_with_failure() {
        let mut client = MockMCPClient::new();

        // Configure to fail
        let error = WorkflowError::ExecutionError("Test error".to_string());
        client.fail_with(error).await;

        // All operations should fail
        assert!(client.connect().await.is_err());
        assert!(client.initialize("test", "1.0").await.is_err());
        assert!(client.list_tools().await.is_err());
        assert!(client.call_tool("test", None).await.is_err());
    }
}
