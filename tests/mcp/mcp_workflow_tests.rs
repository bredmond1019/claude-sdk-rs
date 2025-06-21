#[cfg(test)]
mod tests {
    use claude_sdk_rs_mcp::config::*;
    use claude_sdk_rs_mcp::protocol::*;
    use claude_sdk_rs_mcp::transport::*;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn test_mcp_initialize_workflow() {
        // Test the complete MCP initialization workflow
        let init_request = MCPRequest::Initialize {
            id: "init-1".to_string(),
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

        // Verify request structure
        assert_eq!(init_request.get_id(), Some("init-1"));

        let json = serde_json::to_value(&init_request).unwrap();
        assert_eq!(json["method"], "initialize");
        assert_eq!(json["params"]["protocol_version"], "2024-11-05");
        assert_eq!(json["params"]["client_info"]["name"], "test-client");

        // Create corresponding response
        let init_response = MCPResponse::Result {
            id: "init-1".to_string(),
            result: ResponseResult::Initialize(InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                server_info: ServerInfo {
                    name: "test-server".to_string(),
                    version: "2.0.0".to_string(),
                },
                capabilities: ServerCapabilities {
                    tools: Some(ToolsCapability {
                        list_changed: Some(true),
                    }),
                    prompts: None,
                    resources: None,
                    logging: None,
                },
            }),
        };

        assert_eq!(init_response.get_id(), "init-1");

        let response_json = serde_json::to_value(&init_response).unwrap();
        assert_eq!(response_json["type"], "result");
        assert_eq!(response_json["id"], "init-1");
    }

    #[test]
    fn test_mcp_tool_listing_workflow() {
        // Test listing tools workflow
        let list_request = MCPRequest::ListTools {
            id: "list-1".to_string(),
        };

        assert_eq!(list_request.get_id(), Some("list-1"));

        // Create response with tools
        let tools = vec![
            ToolDefinition {
                name: "file_reader".to_string(),
                description: Some("Read file contents".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }),
            },
            ToolDefinition {
                name: "calculator".to_string(),
                description: Some("Perform calculations".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }),
            },
        ];

        let list_response = MCPResponse::Result {
            id: "list-1".to_string(),
            result: ResponseResult::ListTools(ListToolsResult { tools }),
        };

        let response_json = serde_json::to_value(&list_response).unwrap();
        assert_eq!(response_json["type"], "result");
        assert_eq!(
            response_json["result"]["tools"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn test_mcp_tool_execution_workflow() {
        // Test tool execution workflow
        let call_request = MCPRequest::CallTool {
            id: "call-1".to_string(),
            params: ToolCallParams {
                name: "file_reader".to_string(),
                arguments: Some(
                    vec![("path".to_string(), json!("/tmp/test.txt"))]
                        .into_iter()
                        .collect(),
                ),
            },
        };

        let json = serde_json::to_value(&call_request).unwrap();
        assert_eq!(json["method"], "tools/call");
        assert_eq!(json["params"]["name"], "file_reader");
        assert_eq!(json["params"]["arguments"]["path"], "/tmp/test.txt");

        // Success response
        let success_response = MCPResponse::Result {
            id: "call-1".to_string(),
            result: ResponseResult::CallTool(CallToolResult {
                content: vec![ToolContent::Text {
                    text: "File contents: Hello, World!".to_string(),
                }],
                is_error: Some(false),
            }),
        };

        match success_response {
            MCPResponse::Result {
                result: ResponseResult::CallTool(tool_result),
                ..
            } => {
                assert_eq!(tool_result.content.len(), 1);
                assert_eq!(tool_result.is_error, Some(false));
                match &tool_result.content[0] {
                    ToolContent::Text { text } => {
                        assert!(text.contains("Hello, World!"));
                    }
                    _ => panic!("Expected text content"),
                }
            }
            _ => panic!("Expected CallTool result"),
        }

        // Error response
        let error_response = MCPResponse::Error {
            id: "call-1".to_string(),
            error: MCPError {
                code: -32000,
                message: "File not found".to_string(),
                data: Some(json!({
                    "file_path": "/tmp/test.txt"
                })),
            },
        };

        match error_response {
            MCPResponse::Error { error, .. } => {
                assert_eq!(error.code, -32000);
                assert_eq!(error.message, "File not found");
                assert!(error.data.is_some());
            }
            _ => panic!("Expected error response"),
        }
    }

    #[test]
    fn test_transport_configuration() {
        // Test different transport configurations
        let stdio_transport = TransportType::Stdio {
            command: "python".to_string(),
            args: vec!["server.py".to_string()],
            auto_restart: true,
            max_restarts: 5,
        };

        match stdio_transport {
            TransportType::Stdio {
                command,
                args,
                auto_restart,
                max_restarts,
            } => {
                assert_eq!(command, "python");
                assert_eq!(args, vec!["server.py"]);
                assert!(auto_restart);
                assert_eq!(max_restarts, 5);
            }
            _ => panic!("Expected Stdio transport"),
        }

        let websocket_transport = TransportType::WebSocket {
            url: "wss://api.example.com/mcp".to_string(),
            heartbeat_interval: Some(Duration::from_secs(30)),
            reconnect_config: ReconnectConfig {
                enabled: true,
                max_attempts: 3,
                initial_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },
        };

        match websocket_transport {
            TransportType::WebSocket {
                url,
                heartbeat_interval,
                reconnect_config,
            } => {
                assert_eq!(url, "wss://api.example.com/mcp");
                assert_eq!(heartbeat_interval, Some(Duration::from_secs(30)));
                assert!(reconnect_config.enabled);
                assert_eq!(reconnect_config.max_attempts, 3);
            }
            _ => panic!("Expected WebSocket transport"),
        }

        let http_transport = TransportType::Http {
            base_url: "https://api.example.com".to_string(),
            pool_config: HttpPoolConfig {
                max_connections_per_host: 20,
                connect_timeout: Duration::from_secs(10),
                request_timeout: Duration::from_secs(60),
                keep_alive_timeout: Duration::from_secs(300),
            },
        };

        match http_transport {
            TransportType::Http {
                base_url,
                pool_config,
            } => {
                assert_eq!(base_url, "https://api.example.com");
                assert_eq!(pool_config.max_connections_per_host, 20);
                assert_eq!(pool_config.connect_timeout, Duration::from_secs(10));
            }
            _ => panic!("Expected HTTP transport"),
        }
    }

    #[test]
    fn test_mcp_server_configuration() {
        // Test MCP server configuration
        let server_config = MCPServerConfig {
            name: "test-server".to_string(),
            enabled: true,
            transport: TransportType::WebSocket {
                url: "ws://localhost:8080".to_string(),
                heartbeat_interval: Some(Duration::from_secs(45)),
                reconnect_config: ReconnectConfig::default(),
            },
            auto_connect: true,
            retry_on_failure: true,
        };

        assert_eq!(server_config.name, "test-server");
        assert!(server_config.enabled);
        assert!(server_config.auto_connect);
        assert!(server_config.retry_on_failure);

        let mcp_config = MCPConfig {
            enabled: true,
            client_name: "integration-test".to_string(),
            client_version: "1.2.0".to_string(),
            connection_pool: Default::default(),
            servers: vec![("test-server".to_string(), server_config)]
                .into_iter()
                .collect(),
        };

        assert!(mcp_config.enabled);
        assert_eq!(mcp_config.client_name, "integration-test");
        assert_eq!(mcp_config.client_version, "1.2.0");
        assert_eq!(mcp_config.servers.len(), 1);
        assert!(mcp_config.is_server_enabled("test-server"));
        assert!(!mcp_config.is_server_enabled("non-existent-server"));

        let enabled_servers = mcp_config.get_enabled_servers();
        assert_eq!(enabled_servers.len(), 1);
        assert_eq!(enabled_servers[0].name, "test-server");
    }

    #[test]
    fn test_mcp_error_types() {
        // Test various MCP error scenarios
        let errors = vec![
            MCPError {
                code: -32700,
                message: "Parse error".to_string(),
                data: None,
            },
            MCPError {
                code: -32600,
                message: "Invalid Request".to_string(),
                data: Some(json!({"invalid_field": "value"})),
            },
            MCPError {
                code: -32601,
                message: "Method not found".to_string(),
                data: Some(json!({"method": "unknown_method"})),
            },
            MCPError {
                code: -32602,
                message: "Invalid params".to_string(),
                data: Some(json!({"expected": "string", "got": "number"})),
            },
            MCPError {
                code: -32603,
                message: "Internal error".to_string(),
                data: Some(json!({"stack_trace": "..."})),
            },
        ];

        for error in errors {
            let response = MCPResponse::Error {
                id: "error-test".to_string(),
                error: error.clone(),
            };

            let json = serde_json::to_value(&response).unwrap();
            assert_eq!(json["type"], "error");
            assert_eq!(json["error"]["code"], error.code);
            assert_eq!(json["error"]["message"], error.message);

            // Verify serialization roundtrip
            let deserialized: MCPResponse = serde_json::from_value(json).unwrap();
            match deserialized {
                MCPResponse::Error {
                    error: deserialized_error,
                    ..
                } => {
                    assert_eq!(deserialized_error.code, error.code);
                    assert_eq!(deserialized_error.message, error.message);
                }
                _ => panic!("Expected error response"),
            }
        }
    }

    #[test]
    fn test_tool_content_types() {
        // Test different tool content types
        let text_content = ToolContent::Text {
            text: "This is text content".to_string(),
        };

        let image_content = ToolContent::Image {
            data: "base64encodeddata".to_string(),
            mime_type: "image/png".to_string(),
        };

        let resource_content = ToolContent::Resource {
            resource: ResourceReference {
                uri: "file:///tmp/data.json".to_string(),
                mime_type: Some("application/json".to_string()),
            },
        };

        let tool_result = CallToolResult {
            content: vec![text_content, image_content, resource_content],
            is_error: Some(false),
        };

        assert_eq!(tool_result.content.len(), 3);
        assert_eq!(tool_result.is_error, Some(false));

        match &tool_result.content[0] {
            ToolContent::Text { text } => assert_eq!(text, "This is text content"),
            _ => panic!("Expected text content"),
        }

        match &tool_result.content[1] {
            ToolContent::Image { data, mime_type } => {
                assert_eq!(data, "base64encodeddata");
                assert_eq!(mime_type, "image/png");
            }
            _ => panic!("Expected image content"),
        }

        match &tool_result.content[2] {
            ToolContent::Resource { resource } => {
                assert_eq!(resource.uri, "file:///tmp/data.json");
                assert_eq!(resource.mime_type, Some("application/json".to_string()));
            }
            _ => panic!("Expected resource content"),
        }
    }
}
