//! Critical path tests for Claude AI Core
//!
//! This module contains comprehensive tests for all critical paths in the core library,
//! targeting >95% test coverage of essential functionality.

use claude_sdk_rs_core::*;

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let config = Config::default();

        assert!(config.system_prompt.is_none());
        assert!(config.model.is_none());
        assert!(config.stream_format == StreamFormat::Text);
        assert_eq!(config.timeout_secs, Some(30)); // Default timeout is 30 seconds
        assert!(config.max_tokens.is_none());
        assert!(config.allowed_tools.is_none());
        assert!(config.mcp_config_path.is_none());
        assert!(!config.verbose);
    }

    #[test]
    fn test_config_builder_all_fields() {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .system_prompt("Test system prompt")
            .stream_format(StreamFormat::Json)
            .timeout_secs(60)
            .max_tokens(1000)
            .allowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
            .mcp_config("./config.json")
            .verbose(true)
            .build()
            .expect("Config should build successfully");

        assert_eq!(config.model, Some("claude-3-opus-20240229".to_string()));
        assert_eq!(config.system_prompt, Some("Test system prompt".to_string()));
        assert_eq!(config.stream_format, StreamFormat::Json);
        assert_eq!(config.timeout_secs, Some(60));
        assert_eq!(config.max_tokens, Some(1000));
        assert_eq!(
            config.allowed_tools,
            Some(vec!["bash".to_string(), "filesystem".to_string()])
        );
        assert!(!config.mcp_config_path.is_none());
        assert!(config.verbose);
    }

    #[test]
    fn test_config_validation_system_prompt_too_long() {
        let long_prompt = "a".repeat(10_001);
        let result = Config::builder().system_prompt(&long_prompt).build();

        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidInput(msg) => {
                assert!(msg.contains("System prompt exceeds maximum length"))
            }
            _ => panic!("Expected InvalidInput for long system prompt"),
        }
    }

    #[test]
    fn test_config_validation_timeout_too_small() {
        let result = Config::builder().timeout_secs(0).build();

        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidInput(msg) => assert!(msg.contains("Timeout must be between")),
            _ => panic!("Expected InvalidInput for invalid timeout"),
        }
    }

    #[test]
    fn test_config_validation_timeout_too_large() {
        let result = Config::builder().timeout_secs(3601).build();

        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidInput(msg) => assert!(msg.contains("Timeout must be between")),
            _ => panic!("Expected InvalidInput for invalid timeout"),
        }
    }

    #[test]
    fn test_config_validation_max_tokens_too_large() {
        let result = Config::builder().max_tokens(200_001).build();

        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidInput(msg) => assert!(msg.contains("Max tokens must be between")),
            _ => panic!("Expected InvalidInput for invalid max_tokens"),
        }
    }

    #[test]
    fn test_config_validation_tool_name_too_long() {
        let long_tool_name = "a".repeat(101);
        let result = Config::builder()
            .allowed_tools(vec![long_tool_name])
            .build();

        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidInput(msg) => assert!(
                msg.contains("Tool name") || msg.contains("too long") || msg.contains("exceeds")
            ),
            _ => panic!("Expected InvalidInput for long tool name"),
        }
    }

    #[test]
    fn test_query_validation_empty() {
        let result = validate_query("");
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidInput(msg) => assert!(msg.contains("Query cannot be empty")),
            _ => panic!("Expected InvalidInput for empty query"),
        }
    }

    #[test]
    fn test_query_validation_whitespace_only() {
        // Whitespace-only queries are actually allowed by the current implementation
        let result = validate_query("   \n\t  ");
        assert!(result.is_ok());
    }

    #[test]
    fn test_query_validation_too_long() {
        let long_query = "a".repeat(100_001);
        let result = validate_query(&long_query);
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidInput(msg) => assert!(msg.contains("Query exceeds maximum length")),
            _ => panic!("Expected InvalidInput for long query"),
        }
    }

    #[test]
    fn test_query_validation_valid() {
        let result = validate_query("Valid query");
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_format_variants() {
        // Test all stream format variants
        assert_eq!(format!("{:?}", StreamFormat::Text), "Text");
        assert_eq!(format!("{:?}", StreamFormat::Json), "Json");
        assert_eq!(format!("{:?}", StreamFormat::StreamJson), "StreamJson");
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_error_codes_unique() {
        // Test that all error codes are unique
        let mut codes = std::collections::HashSet::new();

        let errors = vec![
            Error::BinaryNotFound,
            Error::SessionNotFound("test".to_string()),
            Error::PermissionDenied("test".to_string()),
            Error::McpError("test".to_string()),
            Error::ConfigError("test".to_string()),
            Error::InvalidInput("test".to_string()),
            Error::Timeout(30),
            Error::ProcessError("test".to_string()),
            Error::StreamClosed,
            Error::NotAuthenticated,
            Error::RateLimitExceeded,
        ];

        for error in errors {
            let code = error.code();
            assert!(codes.insert(code), "Duplicate error code: {:?}", code);
        }
    }

    #[test]
    fn test_error_recoverability() {
        // Test recoverable errors
        assert!(Error::Timeout(30).is_recoverable());
        assert!(Error::RateLimitExceeded.is_recoverable());
        assert!(Error::StreamClosed.is_recoverable());
        assert!(Error::ProcessError("temp failure".to_string()).is_recoverable());

        // Test non-recoverable errors
        assert!(!Error::BinaryNotFound.is_recoverable());
        assert!(!Error::ConfigError("invalid".to_string()).is_recoverable());
        assert!(!Error::InvalidInput("bad".to_string()).is_recoverable());
        assert!(!Error::NotAuthenticated.is_recoverable());
        assert!(!Error::PermissionDenied("denied".to_string()).is_recoverable());
    }

    #[test]
    fn test_error_display_includes_codes() {
        let error = Error::BinaryNotFound;
        assert!(error.to_string().contains("[C001]"));

        let error = Error::Timeout(30);
        assert!(error.to_string().contains("[C007]"));
        assert!(error.to_string().contains("30s"));

        let error = Error::SessionNotFound("session123".to_string());
        assert!(error.to_string().contains("[C002]"));
        assert!(error.to_string().contains("session123"));
    }

    #[test]
    fn test_error_from_conversions() {
        // Test From implementations
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error: Error = io_error.into();
        assert_eq!(error.code(), ErrorCode::IoError);

        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let error: Error = json_error.into();
        assert_eq!(error.code(), ErrorCode::SerializationError);
    }

    #[test]
    fn test_error_clone() {
        let original = Error::SessionNotFound("test123".to_string());
        let cloned = original.clone();

        assert_eq!(original.code(), cloned.code());
        assert_eq!(original.to_string(), cloned.to_string());
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;

    #[test]
    fn test_session_id_creation() {
        let id = SessionId::new("test-session");
        assert_eq!(id.as_str(), "test-session");
        assert_eq!(id.to_string(), "test-session");
    }

    #[test]
    fn test_session_id_basic_functionality() {
        let id1 = SessionId::new("session-1");
        let id2 = SessionId::new("session-2");

        // Different IDs should be different
        assert_ne!(id1.as_str(), id2.as_str());

        // IDs should be displayed correctly
        assert_eq!(format!("{}", id1), "session-1");
        assert_eq!(format!("{}", id2), "session-2");
    }

    #[tokio::test]
    async fn test_session_manager_creation() {
        let manager = SessionManager::new();

        // Should start with no sessions
        let sessions = manager.list().await.expect("Should list sessions");
        assert!(sessions.is_empty());
    }

    #[test]
    fn test_session_manager_debug() {
        let manager = SessionManager::new();

        // Should be debuggable
        let debug_str = format!("{:?}", manager);
        assert!(debug_str.contains("SessionManager"));
    }
}

#[cfg(test)]
mod message_tests {
    use claude_sdk_rs_core::message::{
        ConversationStats, Message, MessageMeta, MessageType, TokenUsage,
    };
    use std::time::SystemTime;

    #[test]
    fn test_message_type_variants() {
        let init = MessageType::Init;
        let user = MessageType::User;
        let assistant = MessageType::Assistant;
        let result = MessageType::Result;
        let system = MessageType::System;
        let tool = MessageType::Tool;
        let tool_result = MessageType::ToolResult;

        // Test that all variants can be created
        assert_ne!(format!("{:?}", init), "");
        assert_ne!(format!("{:?}", user), "");
        assert_ne!(format!("{:?}", assistant), "");
        assert_ne!(format!("{:?}", result), "");
        assert_ne!(format!("{:?}", system), "");
        assert_ne!(format!("{:?}", tool), "");
        assert_ne!(format!("{:?}", tool_result), "");
    }

    #[test]
    fn test_message_type_equality() {
        assert_eq!(MessageType::User, MessageType::User);
        assert_ne!(MessageType::User, MessageType::Assistant);

        // Test clone
        let msg_type = MessageType::Assistant;
        let cloned = msg_type.clone();
        assert_eq!(msg_type, cloned);
    }

    #[test]
    fn test_message_type_serialization() {
        let msg_type = MessageType::User;
        let json = serde_json::to_string(&msg_type).unwrap();
        let deserialized: MessageType = serde_json::from_str(&json).unwrap();
        assert_eq!(msg_type, deserialized);
    }

    #[test]
    fn test_message_meta_creation() {
        let meta = MessageMeta {
            session_id: "test-session".to_string(),
            timestamp: Some(SystemTime::now()),
            cost_usd: Some(0.001),
            duration_ms: Some(1500),
            tokens_used: Some(TokenUsage {
                input: 100,
                output: 200,
                total: 300,
            }),
        };

        assert_eq!(meta.session_id, "test-session");
        assert!(meta.timestamp.is_some());
        assert_eq!(meta.cost_usd, Some(0.001));
        assert_eq!(meta.duration_ms, Some(1500));
        assert!(meta.tokens_used.is_some());
    }

    #[test]
    fn test_token_usage_structure() {
        let usage = TokenUsage {
            input: 150,
            output: 300,
            total: 450,
        };

        assert_eq!(usage.input, 150);
        assert_eq!(usage.output, 300);
        assert_eq!(usage.total, 450);

        // Test serialization
        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage.input, deserialized.input);
        assert_eq!(usage.output, deserialized.output);
        assert_eq!(usage.total, deserialized.total);
    }

    #[test]
    fn test_conversation_stats() {
        let stats = ConversationStats {
            total_messages: 5,
            total_cost_usd: 0.015,
            total_duration_ms: 3000,
            total_tokens: TokenUsage {
                input: 500,
                output: 1000,
                total: 1500,
            },
        };

        assert_eq!(stats.total_messages, 5);
        assert_eq!(stats.total_cost_usd, 0.015);
        assert_eq!(stats.total_duration_ms, 3000);
        assert_eq!(stats.total_tokens.total, 1500);
    }

    #[test]
    fn test_message_user_creation() {
        let meta = MessageMeta {
            session_id: "session1".to_string(),
            timestamp: None,
            cost_usd: None,
            duration_ms: None,
            tokens_used: None,
        };

        let message = Message::User {
            content: "Hello, Claude!".to_string(),
            meta: meta.clone(),
        };

        match message {
            Message::User {
                content,
                meta: msg_meta,
            } => {
                assert_eq!(content, "Hello, Claude!");
                assert_eq!(msg_meta.session_id, "session1");
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_message_assistant_creation() {
        let meta = MessageMeta {
            session_id: "session1".to_string(),
            timestamp: Some(SystemTime::now()),
            cost_usd: Some(0.001),
            duration_ms: Some(1200),
            tokens_used: None,
        };

        let message = Message::Assistant {
            content: "Hello! How can I help you?".to_string(),
            meta,
        };

        assert_eq!(message.message_type(), MessageType::Assistant);
        assert_eq!(message.content(), "Hello! How can I help you?");
        assert_eq!(message.meta().session_id, "session1");
    }

    #[test]
    fn test_message_tool_creation() {
        let meta = MessageMeta {
            session_id: "session1".to_string(),
            timestamp: None,
            cost_usd: None,
            duration_ms: None,
            tokens_used: None,
        };

        let params = serde_json::json!({"query": "SELECT * FROM users"});
        let message = Message::Tool {
            name: "database_query".to_string(),
            parameters: params.clone(),
            meta,
        };

        assert_eq!(message.message_type(), MessageType::Tool);
        let content = message.content();
        assert!(content.contains("database_query"));
        assert!(content.contains("SELECT * FROM users"));
    }

    #[test]
    fn test_message_tool_result_creation() {
        let meta = MessageMeta {
            session_id: "session1".to_string(),
            timestamp: None,
            cost_usd: None,
            duration_ms: None,
            tokens_used: None,
        };

        let result = serde_json::json!({"rows": [{"id": 1, "name": "Alice"}]});
        let message = Message::ToolResult {
            tool_name: "database_query".to_string(),
            result: result.clone(),
            meta,
        };

        assert_eq!(message.message_type(), MessageType::ToolResult);
        let content = message.content();
        assert!(content.contains("database_query"));
        assert!(content.contains("Alice"));
    }

    #[test]
    fn test_message_init_creation() {
        let meta = MessageMeta {
            session_id: "session1".to_string(),
            timestamp: Some(SystemTime::now()),
            cost_usd: None,
            duration_ms: None,
            tokens_used: None,
        };

        let message = Message::Init { meta };

        assert_eq!(message.message_type(), MessageType::Init);
        assert_eq!(message.content(), "Session initialized");
    }

    #[test]
    fn test_message_result_creation() {
        let meta = MessageMeta {
            session_id: "session1".to_string(),
            timestamp: None,
            cost_usd: None,
            duration_ms: None,
            tokens_used: None,
        };

        let stats = ConversationStats {
            total_messages: 3,
            total_cost_usd: 0.005,
            total_duration_ms: 2500,
            total_tokens: TokenUsage {
                input: 200,
                output: 400,
                total: 600,
            },
        };

        let message = Message::Result { meta, stats };

        assert_eq!(message.message_type(), MessageType::Result);
        let content = message.content();
        assert!(content.contains("3 messages"));
        assert!(content.contains("0.005"));
    }

    #[test]
    fn test_message_system_creation() {
        let meta = MessageMeta {
            session_id: "session1".to_string(),
            timestamp: None,
            cost_usd: None,
            duration_ms: None,
            tokens_used: None,
        };

        let message = Message::System {
            content: "You are a helpful assistant".to_string(),
            meta,
        };

        assert_eq!(message.message_type(), MessageType::System);
        assert_eq!(message.content(), "You are a helpful assistant");
    }

    #[test]
    fn test_message_serialization_roundtrip() {
        let meta = MessageMeta {
            session_id: "test".to_string(),
            timestamp: None,
            cost_usd: Some(0.001),
            duration_ms: Some(1000),
            tokens_used: Some(TokenUsage {
                input: 10,
                output: 20,
                total: 30,
            }),
        };

        let message = Message::User {
            content: "Test message".to_string(),
            meta,
        };

        let json = serde_json::to_string(&message).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(message.message_type(), deserialized.message_type());
        assert_eq!(message.content(), deserialized.content());
        assert_eq!(message.meta().session_id, deserialized.meta().session_id);
    }
}

#[cfg(test)]
mod types_tests {
    use claude_sdk_rs_core::types::{
        ClaudeCliResponse, ClaudeResponse, Cost, ResponseMetadata, TokenUsage as TypesTokenUsage,
        ToolPermission,
    };
    use serde_json::json;

    #[test]
    fn test_claude_cli_response_creation() {
        let response = ClaudeCliResponse {
            response_type: "assistant_response".to_string(),
            subtype: "completion".to_string(),
            cost_usd: Some(0.001),
            is_error: false,
            duration_ms: 1500,
            duration_api_ms: Some(1200),
            num_turns: 1,
            result: "Hello, world!".to_string(),
            total_cost: Some(0.001),
            session_id: "session_123".to_string(),
        };

        assert_eq!(response.response_type, "assistant_response");
        assert_eq!(response.result, "Hello, world!");
        assert_eq!(response.cost_usd, Some(0.001));
        assert!(!response.is_error);
        assert_eq!(response.session_id, "session_123");
    }

    #[test]
    fn test_claude_response_text_creation() {
        let response = ClaudeResponse::text("Simple response".to_string());

        assert_eq!(response.content, "Simple response");
        assert!(response.raw_json.is_none());
        assert!(response.metadata.is_none());
    }

    #[test]
    fn test_claude_response_with_json() {
        let raw_json = json!({
            "session_id": "test_session",
            "cost_usd": 0.002,
            "duration_ms": 1800,
            "message": {
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 100
                },
                "model": "claude-sonnet-4-20250514"
            }
        });

        let response = ClaudeResponse::with_json("Test response".to_string(), raw_json.clone());

        assert_eq!(response.content, "Test response");
        assert!(response.raw_json.is_some());
        assert!(response.metadata.is_some());

        let metadata = response.metadata.unwrap();
        assert_eq!(metadata.session_id, "test_session");
        assert_eq!(metadata.cost_usd, Some(0.002));
        assert_eq!(metadata.duration_ms, Some(1800));
        assert!(metadata.tokens_used.is_some());
        assert_eq!(metadata.model, Some("claude-sonnet-4-20250514".to_string()));

        let tokens = metadata.tokens_used.unwrap();
        assert_eq!(tokens.input_tokens, Some(50));
        assert_eq!(tokens.output_tokens, Some(100));
    }

    #[test]
    fn test_claude_response_extract_metadata_minimal() {
        let json = json!({
            "session_id": "minimal_session"
        });

        let response = ClaudeResponse::with_json("Minimal".to_string(), json);
        let metadata = response.metadata.unwrap();

        assert_eq!(metadata.session_id, "minimal_session");
        assert!(metadata.cost_usd.is_none());
        assert!(metadata.duration_ms.is_none());
        assert!(metadata.tokens_used.is_none());
        assert!(metadata.model.is_none());
    }

    #[test]
    fn test_claude_response_extract_metadata_missing_session() {
        let json = json!({
            "cost_usd": 0.001,
            "duration_ms": 1000
        });

        let response = ClaudeResponse::with_json("No session".to_string(), json);

        // Should be None because session_id is required
        assert!(response.metadata.is_none());
    }

    #[test]
    fn test_response_metadata_creation() {
        let metadata = ResponseMetadata {
            session_id: "test_session".to_string(),
            cost_usd: Some(0.003),
            duration_ms: Some(2000),
            tokens_used: Some(TypesTokenUsage {
                input_tokens: Some(75),
                output_tokens: Some(150),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: Some(25),
            }),
            model: Some("claude-haiku-3-20250307".to_string()),
        };

        assert_eq!(metadata.session_id, "test_session");
        assert_eq!(metadata.cost_usd, Some(0.003));
        assert_eq!(metadata.duration_ms, Some(2000));
        assert!(metadata.tokens_used.is_some());
        assert_eq!(metadata.model, Some("claude-haiku-3-20250307".to_string()));
    }

    #[test]
    fn test_token_usage_all_fields() {
        let usage = TypesTokenUsage {
            input_tokens: Some(100),
            output_tokens: Some(200),
            cache_creation_input_tokens: Some(50),
            cache_read_input_tokens: Some(25),
        };

        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(200));
        assert_eq!(usage.cache_creation_input_tokens, Some(50));
        assert_eq!(usage.cache_read_input_tokens, Some(25));

        // Test serialization
        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: TypesTokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage.input_tokens, deserialized.input_tokens);
        assert_eq!(usage.output_tokens, deserialized.output_tokens);
    }

    #[test]
    fn test_tool_permission_mcp() {
        let permission = ToolPermission::mcp("database", "query");

        match &permission {
            ToolPermission::Mcp { server, tool } => {
                assert_eq!(server, "database");
                assert_eq!(tool, "query");
            }
            _ => panic!("Expected MCP permission"),
        }

        assert_eq!(permission.to_cli_format(), "mcp__database__query");
    }

    #[test]
    fn test_tool_permission_mcp_wildcard() {
        let permission = ToolPermission::mcp("filesystem", "*");
        assert_eq!(permission.to_cli_format(), "mcp__filesystem__*");
    }

    #[test]
    fn test_tool_permission_bash() {
        let permission = ToolPermission::bash("ls -la");

        match &permission {
            ToolPermission::Bash { command } => {
                assert_eq!(command, "ls -la");
            }
            _ => panic!("Expected Bash permission"),
        }

        assert_eq!(permission.to_cli_format(), "bash:ls -la");
    }

    #[test]
    fn test_tool_permission_all() {
        let permission = ToolPermission::All;
        assert_eq!(permission.to_cli_format(), "*");
    }

    #[test]
    fn test_tool_permission_equality() {
        let perm1 = ToolPermission::mcp("server", "tool");
        let perm2 = ToolPermission::mcp("server", "tool");
        let perm3 = ToolPermission::bash("ls");

        assert_eq!(perm1, perm2);
        assert_ne!(perm1, perm3);
    }

    #[test]
    fn test_tool_permission_serialization() {
        let permissions = vec![
            ToolPermission::mcp("db", "query"),
            ToolPermission::bash("pwd"),
            ToolPermission::All,
        ];

        for permission in permissions {
            let json = serde_json::to_string(&permission).unwrap();
            let deserialized: ToolPermission = serde_json::from_str(&json).unwrap();
            assert_eq!(permission, deserialized);
        }
    }

    #[test]
    fn test_cost_creation() {
        let cost = Cost::new(0.001234);
        assert_eq!(cost.usd, 0.001234);

        let zero_cost = Cost::zero();
        assert_eq!(zero_cost.usd, 0.0);
    }

    #[test]
    fn test_cost_addition() {
        let cost1 = Cost::new(0.001);
        let cost2 = Cost::new(0.002);
        let total = cost1.add(&cost2);

        assert_eq!(total.usd, 0.003);

        // Original costs should be unchanged
        assert_eq!(cost1.usd, 0.001);
        assert_eq!(cost2.usd, 0.002);
    }

    #[test]
    fn test_cost_serialization() {
        let cost = Cost::new(0.00456);

        let json = serde_json::to_string(&cost).unwrap();
        let deserialized: Cost = serde_json::from_str(&json).unwrap();

        assert_eq!(cost.usd, deserialized.usd);
    }

    #[test]
    fn test_cost_copy_and_clone() {
        let cost = Cost::new(0.123);
        let copied = cost; // Test Copy trait
        let cloned = cost.clone(); // Test Clone trait

        assert_eq!(cost.usd, copied.usd);
        assert_eq!(cost.usd, cloned.usd);
    }
}
