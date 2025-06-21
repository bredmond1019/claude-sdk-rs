//! Integration tests for claude-ai-mcp crate
//!
//! These tests verify that the core MCP functionality works correctly
//! without requiring external MCP servers to be running.

use claude_sdk_rs_mcp::health::HealthStatus;
use claude_sdk_rs_mcp::protocol::MCPRequest;
use claude_sdk_rs_mcp::transport::{HttpPoolConfig, TransportType};
use claude_sdk_rs_mcp::{McpConfig, McpServer};

#[test]
fn test_transport_type_serialization() {
    let http_transport = TransportType::Http {
        base_url: "http://example.com".to_string(),
        pool_config: HttpPoolConfig::default(),
    };

    let serialized = serde_json::to_value(&http_transport).unwrap();
    let deserialized: TransportType = serde_json::from_value(serialized).unwrap();

    match deserialized {
        TransportType::Http { base_url, .. } => {
            assert_eq!(base_url, "http://example.com");
        }
        _ => panic!("Expected HTTP transport"),
    }
}

#[test]
fn test_mcp_request_creation() {
    let request = MCPRequest::ListTools {
        id: "test-123".to_string(),
    };

    // Verify request can be serialized
    let serialized = serde_json::to_value(&request).unwrap();
    assert!(serialized.is_object());
    assert_eq!(serialized["id"], "test-123");
}

#[test]
fn test_health_status_enum() {
    // Test all health status variants
    let statuses = vec![
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Unhealthy,
    ];

    for status in statuses {
        // Verify each status can be serialized/deserialized
        let serialized = serde_json::to_value(&status).unwrap();
        let deserialized: HealthStatus = serde_json::from_value(serialized).unwrap();
        assert_eq!(
            std::mem::discriminant(&status),
            std::mem::discriminant(&deserialized)
        );
    }
}

#[test]
fn test_legacy_config_compatibility() {
    // Test the backwards compatibility types
    let server =
        McpServer::new("python", vec!["-m", "test_server"]).with_env("API_KEY", "test-key");

    let config = McpConfig {
        servers: vec![server],
    };

    assert_eq!(config.servers.len(), 1);
    assert_eq!(config.servers[0].command, "python");
    assert_eq!(config.servers[0].args, vec!["-m", "test_server"]);
    assert_eq!(
        config.servers[0].env.get("API_KEY"),
        Some(&"test-key".to_string())
    );
}

#[test]
fn test_stdio_transport_creation() {
    let stdio_transport = TransportType::Stdio {
        command: "python".to_string(),
        args: vec!["-m".to_string(), "server".to_string()],
        auto_restart: true,
        max_restarts: 3,
    };

    let serialized = serde_json::to_value(&stdio_transport).unwrap();
    let deserialized: TransportType = serde_json::from_value(serialized).unwrap();

    match deserialized {
        TransportType::Stdio {
            command,
            args,
            auto_restart,
            max_restarts,
        } => {
            assert_eq!(command, "python");
            assert_eq!(args, vec!["-m", "server"]);
            assert_eq!(auto_restart, true);
            assert_eq!(max_restarts, 3);
        }
        _ => panic!("Expected Stdio transport"),
    }
}

#[test]
fn test_websocket_transport_creation() {
    let ws_transport = TransportType::WebSocket {
        url: "ws://localhost:8080".to_string(),
        heartbeat_interval: Some(std::time::Duration::from_secs(30)),
        reconnect_config: claude_ai_mcp::transport::ReconnectConfig {
            enabled: true,
            max_attempts: 5,
            initial_delay: std::time::Duration::from_millis(100),
            max_delay: std::time::Duration::from_secs(30),
            backoff_multiplier: 2.0,
        },
    };

    let serialized = serde_json::to_value(&ws_transport).unwrap();
    let deserialized: TransportType = serde_json::from_value(serialized).unwrap();

    match deserialized {
        TransportType::WebSocket { url, .. } => {
            assert_eq!(url, "ws://localhost:8080");
        }
        _ => panic!("Expected WebSocket transport"),
    }
}

#[test]
fn test_mcp_request_variants() {
    // Test Initialize request
    let init_request = MCPRequest::Initialize {
        id: "init-1".to_string(),
        params: claude_ai_mcp::protocol::InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            capabilities: claude_ai_mcp::protocol::ClientCapabilities {
                roots: None,
                sampling: None,
            },
            client_info: claude_ai_mcp::protocol::ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
        },
    };

    let serialized = serde_json::to_value(&init_request).unwrap();
    assert_eq!(serialized["id"], "init-1");
    assert_eq!(serialized["method"], "initialize");

    // Test CallTool request - simplify to just use empty arguments
    let call_request = MCPRequest::CallTool {
        id: "call-1".to_string(),
        params: claude_ai_mcp::protocol::ToolCallParams {
            name: "test-tool".to_string(),
            arguments: None,
        },
    };

    let serialized = serde_json::to_value(&call_request).unwrap();
    assert_eq!(serialized["id"], "call-1");
    assert_eq!(serialized["method"], "tools/call");
}
