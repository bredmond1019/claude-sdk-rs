#[cfg(test)]
mod tests {
    use claude_sdk_rs_mcp::protocol::*;
    use serde_json::json;

    #[test]
    fn test_mcp_request_serialization() {
        let request = MCPRequest::Initialize {
            id: "test-123".to_string(),
            params: InitializeParams {
                protocol_version: "2024-11-05".to_string(),
                client_info: ClientInfo {
                    name: "test-client".to_string(),
                    version: "1.0.0".to_string(),
                },
                capabilities: ClientCapabilities {
                    roots: None,
                    sampling: None,
                },
            },
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("initialize"));
        assert!(serialized.contains("test-client"));
    }

    #[test]
    fn test_mcp_response_deserialization() {
        let response_json = json!({
            "type": "result",
            "id": "test-123",
            "result": {
                "protocol_version": "2024-11-05",
                "server_info": {
                    "name": "test-server",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": {"list_changed": true},
                    "prompts": null,
                    "resources": null,
                    "logging": null
                }
            }
        });

        let response: MCPResponse = serde_json::from_value(response_json).unwrap();
        match response {
            MCPResponse::Result { id, result } => {
                assert_eq!(id, "test-123");
                match result {
                    ResponseResult::Initialize(init_result) => {
                        assert_eq!(init_result.server_info.name, "test-server");
                        assert_eq!(init_result.server_info.version, "1.0.0");
                        assert!(init_result.capabilities.tools.is_some());
                    }
                    _ => panic!("Expected Initialize result"),
                }
            }
            _ => panic!("Expected Result response"),
        }
    }

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition {
            name: "test_tool".to_string(),
            description: Some("A test tool".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string"
                    }
                }
            }),
        };

        let serialized = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.name, "test_tool");
        assert_eq!(deserialized.description, Some("A test tool".to_string()));
    }

    #[test]
    fn test_call_tool_request() {
        let request = MCPRequest::CallTool {
            id: "call-123".to_string(),
            params: ToolCallParams {
                name: "analyze_sentiment".to_string(),
                arguments: Some(
                    vec![
                        ("text".to_string(), json!("I love this product!")),
                        ("language".to_string(), json!("en")),
                    ]
                    .into_iter()
                    .collect(),
                ),
            },
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("tools/call"));
        assert!(serialized.contains("analyze_sentiment"));
        assert!(serialized.contains("I love this product!"));
    }

    #[test]
    fn test_mcp_error() {
        let error = MCPError {
            code: -32000,
            message: "Server error".to_string(),
            data: Some(json!({
                "details": "Connection failed"
            })),
        };

        let serialized = serde_json::to_string(&error).unwrap();
        let deserialized: MCPError = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.code, -32000);
        assert_eq!(deserialized.message, "Server error");
        assert!(deserialized.data.is_some());
    }

    #[test]
    fn test_call_tool_result() {
        let result = CallToolResult {
            content: vec![ToolContent::Text {
                text: "Analysis complete: positive sentiment".to_string(),
            }],
            is_error: Some(false),
        };

        let response = MCPResponse::Result {
            id: "call-123".to_string(),
            result: ResponseResult::CallTool(result),
        };

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: MCPResponse = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            MCPResponse::Result {
                result: ResponseResult::CallTool(tool_result),
                ..
            } => {
                assert_eq!(tool_result.content.len(), 1);
                assert_eq!(tool_result.is_error, Some(false));
            }
            _ => panic!("Expected CallTool result"),
        }
    }
}
