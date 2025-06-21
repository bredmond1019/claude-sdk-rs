#[cfg(test)]
mod tests {
    use crate::mcp::protocol::ToolDefinition;
    use crate::mcp::clients::{MCPClient, MCPConnection, StdioMCPClient, WebSocketMCPClient};
    use crate::mcp::protocol::{
        CallToolResult, ListToolsResult, MCPRequest, MCPResponse, ResponseResult, ToolContent,
    };
    use crate::mcp::transport::{MCPTransport, TransportError};
    use async_trait::async_trait;

    struct MockTransport {
        responses: Vec<MCPResponse>,
        response_index: usize,
        connected: bool,
    }

    impl MockTransport {
        fn new(responses: Vec<MCPResponse>) -> Self {
            Self {
                responses,
                response_index: 0,
                connected: false,
            }
        }
    }

    #[async_trait]
    impl MCPTransport for MockTransport {
        async fn connect(&mut self) -> Result<(), TransportError> {
            self.connected = true;
            Ok(())
        }

        async fn send(&mut self, _message: MCPRequest) -> Result<(), TransportError> {
            if !self.connected {
                return Err(TransportError::ConnectionError("Not connected".to_string()));
            }
            Ok(())
        }

        async fn receive(&mut self) -> Result<MCPResponse, TransportError> {
            if !self.connected {
                return Err(TransportError::ConnectionError("Not connected".to_string()));
            }

            if self.response_index >= self.responses.len() {
                return Err(TransportError::ConnectionError(
                    "No more responses".to_string(),
                ));
            }

            let response = self.responses[self.response_index].clone();
            self.response_index += 1;
            Ok(response)
        }

        async fn disconnect(&mut self) -> Result<(), TransportError> {
            self.connected = false;
            Ok(())
        }
        
        fn is_connected(&self) -> bool {
            self.connected
        }
        
        async fn health_check(&mut self) -> Result<crate::mcp::transport::TransportHealth, TransportError> {
            Ok(crate::mcp::transport::TransportHealth {
                is_connected: self.connected,
                last_ping: None,
                connection_age: std::time::Duration::from_secs(0),
                bytes_sent: 0,
                bytes_received: 0,
                messages_sent: 0,
                messages_received: 0,
                last_error: None,
            })
        }
        
        async fn ping(&mut self) -> Result<std::time::Duration, TransportError> {
            if !self.connected {
                return Err(TransportError::ConnectionError("Not connected".to_string()));
            }
            Ok(std::time::Duration::from_millis(0))
        }
        
        fn get_metrics(&self) -> crate::mcp::transport::TransportMetrics {
            crate::mcp::transport::TransportMetrics::default()
        }
        
        async fn reconnect(&mut self) -> Result<(), TransportError> {
            self.connect().await
        }
    }

    #[tokio::test]
    async fn test_mcp_connection_creation() {
        let transport = Box::new(MockTransport::new(vec![]));
        let connection = MCPConnection::new(transport);

        assert!(!connection.is_connected);
        assert!(!connection.is_initialized);
    }

    #[tokio::test]
    async fn test_stdio_mcp_client_creation() {
        let client = StdioMCPClient::new("echo".to_string(), vec!["hello".to_string()]);
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_websocket_mcp_client_creation() {
        let client = WebSocketMCPClient::new("ws://localhost:8080".to_string());
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_mcp_client_trait_methods() {
        // Test that we can create and use MCP clients through the trait
        let client: Box<dyn MCPClient> = Box::new(StdioMCPClient::new(
            "echo".to_string(),
            vec!["hello".to_string()],
        ));

        // Initially not connected
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_mcp_connection_with_mock_transport() {
        use crate::mcp::protocol::{InitializeResult, ServerCapabilities, ServerInfo};

        let init_response = MCPResponse::Result {
            id: "test-id".to_string(),
            result: ResponseResult::Initialize(InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ServerCapabilities {
                    logging: None,
                    prompts: None,
                    resources: None,
                    tools: None,
                },
                server_info: ServerInfo {
                    name: "test-server".to_string(),
                    version: "1.0.0".to_string(),
                },
            }),
        };

        let transport = Box::new(MockTransport::new(vec![init_response]));
        let mut connection = MCPConnection::new(transport);

        // Test connection
        connection.transport.connect().await.unwrap();
        assert!(
            connection
                .transport
                .send(MCPRequest::Initialized)
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_tool_definitions() {
        let tool_def = ToolDefinition {
            name: "test_tool".to_string(),
            description: Some("A test tool".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }),
        };

        assert_eq!(tool_def.name, "test_tool");
        assert!(tool_def.description.is_some());
    }

    #[tokio::test]
    async fn test_tool_call_result() {
        let result = CallToolResult {
            content: vec![ToolContent::Text {
                text: "Hello, World!".to_string(),
            }],
            is_error: Some(false),
        };

        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ToolContent::Text { text } => assert_eq!(text, "Hello, World!"),
            _ => panic!("Expected text content"),
        }
    }

    #[tokio::test]
    async fn test_mcp_request_id_extraction() {
        let request = MCPRequest::ListTools {
            id: "test-123".to_string(),
        };

        assert_eq!(request.get_id(), Some("test-123"));

        let notification = MCPRequest::Initialized;
        assert_eq!(notification.get_id(), None);
    }

    #[tokio::test]
    async fn test_mcp_response_id_extraction() {
        let response = MCPResponse::Result {
            id: "response-456".to_string(),
            result: ResponseResult::ListTools(ListToolsResult { tools: vec![] }),
        };

        assert_eq!(response.get_id(), "response-456");
    }
}
