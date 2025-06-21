#[cfg(test)]
mod tests {
    use claude_sdk_rs_mcp::{
        config::{MCPConfig, MCPServerConfig},
        protocol::*,
        transport::TransportType,
    };
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn test_basic_mcp_workflow() {
        // Create a tool definition
        let tool = ToolDefinition {
            name: "calculate_sum".to_string(),
            description: Some("Calculate the sum of two numbers".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" }
                },
                "required": ["a", "b"]
            }),
        };

        // Serialize and deserialize to ensure compatibility
        let serialized = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tool.name, deserialized.name);

        // Create a tool call request
        let request = MCPRequest::CallTool {
            id: "call-456".to_string(),
            params: ToolCallParams {
                name: "calculate_sum".to_string(),
                arguments: Some(
                    vec![("a".to_string(), json!(5)), ("b".to_string(), json!(3))]
                        .into_iter()
                        .collect(),
                ),
            },
        };

        // Verify request serialization
        let request_json = serde_json::to_value(&request).unwrap();
        assert_eq!(request_json["method"], "tools/call");
        assert_eq!(request_json["params"]["name"], "calculate_sum");

        // Create a response
        let response = MCPResponse::Result {
            id: "call-456".to_string(),
            result: ResponseResult::CallTool(CallToolResult {
                content: vec![ToolContent::Text {
                    text: json!({ "sum": 8 }).to_string(),
                }],
                is_error: Some(false),
            }),
        };

        // Verify response serialization
        let response_json = serde_json::to_value(&response).unwrap();
        assert_eq!(response_json["type"], "result");
        assert_eq!(response_json["id"], "call-456");
    }

    #[test]
    fn test_mcp_message_envelope() {
        let request = MCPRequest::ListTools {
            id: "test-123".to_string(),
        };
        let message = MCPMessage::Request(request.clone());

        match message {
            MCPMessage::Request(req) => {
                assert_eq!(req.get_id(), Some("test-123"));
            }
            _ => panic!("Expected Request message"),
        }
    }

    #[test]
    fn test_server_capabilities() {
        let capabilities = ServerCapabilities {
            tools: Some(ToolsCapability {
                list_changed: Some(true),
            }),
            prompts: None,
            resources: None,
            logging: None,
        };

        assert!(capabilities.tools.is_some());
        assert!(capabilities.prompts.is_none());
        assert!(capabilities.resources.is_none());

        let json = serde_json::to_value(&capabilities).unwrap();
        assert!(json["tools"].is_object());
        assert!(json["prompts"].is_null());
        assert!(json["resources"].is_null());
    }

    #[test]
    fn test_error_handling() {
        let error = MCPError {
            code: -32601,
            message: "Method not found".to_string(),
            data: Some(json!({
                "method": "unknown_method"
            })),
        };

        let response = MCPResponse::Error {
            id: "error-test".to_string(),
            error: error.clone(),
        };

        let message = MCPMessage::Response(response);

        match message {
            MCPMessage::Response(MCPResponse::Error { error, .. }) => {
                assert_eq!(error.code, -32601);
                assert_eq!(error.message, "Method not found");
            }
            _ => panic!("Expected error response"),
        }
    }

    #[tokio::test]
    async fn test_config_with_transport() {
        let config = MCPConfig {
            enabled: true,
            client_name: "integration-test".to_string(),
            client_version: "1.0.0".to_string(),
            connection_pool: Default::default(),
            servers: vec![(
                "test-server".to_string(),
                MCPServerConfig {
                    name: "test-server".to_string(),
                    enabled: true,
                    transport: TransportType::WebSocket {
                        url: "ws://localhost:8080/test".to_string(),
                        heartbeat_interval: Some(Duration::from_secs(30)),
                        reconnect_config: claude_ai_mcp::transport::ReconnectConfig::default(),
                    },
                    auto_connect: false,
                    retry_on_failure: true,
                },
            )]
            .into_iter()
            .collect(),
        };

        assert_eq!(config.servers.len(), 1);
        assert!(config.servers.contains_key("test-server"));

        let server = &config.servers["test-server"];
        match &server.transport {
            TransportType::WebSocket { url, .. } => {
                assert_eq!(url, "ws://localhost:8080/test");
            }
            _ => panic!("Expected WebSocket transport"),
        }
    }
}
