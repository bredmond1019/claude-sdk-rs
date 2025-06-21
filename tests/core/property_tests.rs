//! Property-based tests for Claude AI Core
//!
//! This module uses property-based testing to verify invariants and properties
//! that should hold for all valid inputs to core types and functions.

use claude_sdk_rs_core::*;
use proptest::prelude::*;

// Property test strategies for generating test data

/// Strategy for generating valid session IDs
fn valid_session_id() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_-]{1,100}".prop_map(|s| s)
}

/// Strategy for generating valid system prompts
fn valid_system_prompt() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 .,!?]{1,1000}".prop_map(|s| s)
}

/// Strategy for generating valid query strings
fn valid_query() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 .,!?]{1,1000}".prop_map(|s| s)
}

/// Strategy for generating valid timeout values
fn valid_timeout() -> impl Strategy<Value = u64> {
    1u64..=3600u64
}

/// Strategy for generating valid max token values
fn valid_max_tokens() -> impl Strategy<Value = usize> {
    1usize..=200_000usize
}

/// Strategy for generating valid tool names
fn valid_tool_name() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_-]{1,50}".prop_map(|s| s)
}

/// Strategy for generating lists of tools
fn valid_tools() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(valid_tool_name(), 0..=10)
}

#[cfg(test)]
mod config_property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_config_builder_always_succeeds_with_valid_inputs(
            system_prompt in prop::option::of(valid_system_prompt()),
            timeout in prop::option::of(valid_timeout()),
            max_tokens in prop::option::of(valid_max_tokens()),
            tools in prop::option::of(valid_tools()),
            verbose in any::<bool>(),
        ) {
            let mut builder = Config::builder();

            if let Some(prompt) = system_prompt {
                builder = builder.system_prompt(&prompt);
            }

            if let Some(t) = timeout {
                builder = builder.timeout_secs(t);
            }

            if let Some(mt) = max_tokens {
                builder = builder.max_tokens(mt);
            }

            if let Some(tool_list) = tools {
                builder = builder.allowed_tools(tool_list);
            }

            builder = builder.verbose(verbose);

            // Building with valid inputs should always succeed
            let result = builder.build();
            prop_assert!(result.is_ok(), "Config build failed with valid inputs: {:?}", result);
        }

        #[test]
        fn test_config_timeout_bounds(timeout in any::<u64>()) {
            let result = Config::builder()
                .timeout_secs(timeout)
                .build();

            if timeout >= 1 && timeout <= 3600 {
                prop_assert!(result.is_ok(), "Valid timeout {} should succeed", timeout);
            } else {
                prop_assert!(result.is_err(), "Invalid timeout {} should fail", timeout);
            }
        }

        #[test]
        fn test_config_max_tokens_bounds(max_tokens in any::<usize>()) {
            let result = Config::builder()
                .max_tokens(max_tokens)
                .build();

            if max_tokens >= 1 && max_tokens <= 200_000 {
                prop_assert!(result.is_ok(), "Valid max_tokens {} should succeed", max_tokens);
            } else {
                prop_assert!(result.is_err(), "Invalid max_tokens {} should fail", max_tokens);
            }
        }

        #[test]
        fn test_config_system_prompt_length(prompt_length in 0usize..=15_000) {
            let prompt = "a".repeat(prompt_length);
            let result = Config::builder()
                .system_prompt(&prompt)
                .build();

            if prompt_length <= 10_000 {
                prop_assert!(result.is_ok(), "System prompt of length {} should succeed", prompt_length);
            } else {
                prop_assert!(result.is_err(), "System prompt of length {} should fail", prompt_length);
            }
        }

        #[test]
        fn test_config_serialization_roundtrip(
            system_prompt in prop::option::of(valid_system_prompt()),
            timeout in prop::option::of(valid_timeout()),
            max_tokens in prop::option::of(valid_max_tokens()),
            verbose in any::<bool>(),
        ) {
            let mut builder = Config::builder().verbose(verbose);

            if let Some(prompt) = system_prompt {
                builder = builder.system_prompt(&prompt);
            }

            if let Some(t) = timeout {
                builder = builder.timeout_secs(t);
            }

            if let Some(mt) = max_tokens {
                builder = builder.max_tokens(mt);
            }

            let config = builder.build().unwrap();

            // Serialize and deserialize
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: Config = serde_json::from_str(&json).unwrap();

            // Properties should be preserved
            prop_assert_eq!(config.system_prompt, deserialized.system_prompt);
            prop_assert_eq!(config.timeout_secs, deserialized.timeout_secs);
            prop_assert_eq!(config.max_tokens, deserialized.max_tokens);
            prop_assert_eq!(config.verbose, deserialized.verbose);
            prop_assert_eq!(config.stream_format, deserialized.stream_format);
        }

        #[test]
        fn test_stream_format_serialization_roundtrip(format in prop::sample::select(vec![
            StreamFormat::Text,
            StreamFormat::Json,
            StreamFormat::StreamJson,
        ])) {
            let config = Config::builder()
                .stream_format(format.clone())
                .build()
                .unwrap();

            let json = serde_json::to_string(&config).unwrap();
            let deserialized: Config = serde_json::from_str(&json).unwrap();

            prop_assert_eq!(config.stream_format, deserialized.stream_format);
        }
    }
}

#[cfg(test)]
mod query_validation_property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_query_validation_length_bounds(query_length in 0usize..=150_000) {
            let query = "a".repeat(query_length);
            let result = validate_query(&query);

            if query_length == 0 {
                prop_assert!(result.is_err(), "Empty query should fail validation");
            } else if query_length <= 100_000 {
                prop_assert!(result.is_ok(), "Query of length {} should pass validation", query_length);
            } else {
                prop_assert!(result.is_err(), "Query of length {} should fail validation", query_length);
            }
        }

        #[test]
        fn test_valid_queries_always_pass(query in valid_query()) {
            let result = validate_query(&query);
            prop_assert!(result.is_ok(), "Valid query '{}' should pass validation", query);
        }

        #[test]
        fn test_query_validation_idempotent(query in any::<String>()) {
            let result1 = validate_query(&query);
            let result2 = validate_query(&query);

            prop_assert_eq!(result1.is_ok(), result2.is_ok(),
                "Query validation should be deterministic");
        }

        #[test]
        fn test_non_empty_queries_with_printable_chars(
            query in "[!-~\\s]{1,1000}"
        ) {
            let result = validate_query(&query);

            // Non-empty queries with printable characters should generally pass
            // (unless they contain malicious patterns, which is tested separately)
            if query.len() <= 100_000 {
                // We can't assert success here because some patterns might be flagged as malicious
                // But we can assert that the function doesn't panic
                let _result = result;
            }
        }
    }
}

#[cfg(test)]
mod session_property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_session_id_creation_preserves_input(id in valid_session_id()) {
            let session_id = SessionId::new(&id);
            prop_assert_eq!(session_id.as_str(), &id);
            prop_assert_eq!(session_id.to_string(), id);
        }

        #[test]
        fn test_session_id_equality(id in valid_session_id()) {
            let session_id1 = SessionId::new(&id);
            let session_id2 = SessionId::new(&id);

            prop_assert_eq!(session_id1.as_str(), session_id2.as_str());
            prop_assert_eq!(session_id1, session_id2);
        }

        #[test]
        fn test_session_id_serialization_roundtrip(id in valid_session_id()) {
            let session_id = SessionId::new(&id);

            let json = serde_json::to_string(&session_id).unwrap();
            let deserialized: SessionId = serde_json::from_str(&json).unwrap();

            prop_assert_eq!(session_id.as_str(), deserialized.as_str());
        }

        #[test]
        fn test_session_creation_basic_properties(
            id in valid_session_id(),
            system_prompt in prop::option::of(valid_system_prompt()),
        ) {
            let session_id = SessionId::new(&id);
            let mut session = Session::new(session_id.clone());

            if let Some(ref prompt) = system_prompt {
                session = session.with_system_prompt(prompt);
            }

            // Properties should be preserved
            prop_assert_eq!(session.id(), &session_id);

            if system_prompt.is_some() {
                prop_assert!(session.system_prompt.is_some());
            }
        }
    }

    #[tokio::test]
    async fn test_session_manager_operations() {
        // This is a more complex property test using tokio
        let manager = SessionManager::new();

        // Property: newly created manager should be empty
        let sessions = manager.list().await.unwrap();
        assert!(sessions.is_empty());

        // Property: get() on empty manager should return None
        let id = SessionId::new("test");
        let result = manager.get(&id).await.unwrap();
        assert!(result.is_none());

        // Property: resume() on non-existent session should fail
        let result = manager.resume(&id).await;
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod error_property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_error_code_consistency(error_variant in prop::sample::select(vec![
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
        ])) {
            let code = error_variant.code();
            let display = error_variant.to_string();

            // Error display should include the error code
            let code_str = format!("{}", code);
            let expected_pattern = format!("[{}]", code_str);
            prop_assert!(display.contains(&expected_pattern));
        }

        #[test]
        fn test_error_recoverability_is_consistent(error_variant in prop::sample::select(vec![
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
        ])) {
            // Recoverability should be consistent for multiple calls
            let recoverable1 = error_variant.is_recoverable();
            let recoverable2 = error_variant.clone().is_recoverable();

            prop_assert_eq!(recoverable1, recoverable2);
        }

        #[test]
        fn test_timeout_error_preserves_value(timeout_value in 1u64..=10000) {
            let error = Error::Timeout(timeout_value);
            let display = error.to_string();

            // Timeout error should include the timeout value
            let timeout_str = format!("{}s", timeout_value);
            prop_assert!(display.contains(&timeout_str));
        }

        #[test]
        fn test_session_not_found_preserves_id(session_id in valid_session_id()) {
            let error = Error::SessionNotFound(session_id.clone());
            let display = error.to_string();

            // SessionNotFound error should include the session ID
            prop_assert!(display.contains(&session_id));
        }
    }
}

#[cfg(test)]
mod types_property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_token_usage_consistency(
            input in 0u64..=100_000,
            output in 0u64..=100_000,
        ) {
            let usage = TokenUsage {
                input,
                output,
                total: input + output,
            };

            // Total should equal input + output
            prop_assert_eq!(usage.total, usage.input + usage.output);

            // Total should be at least as large as either component
            prop_assert!(usage.total >= usage.input);
            prop_assert!(usage.total >= usage.output);
        }

        #[test]
        fn test_tool_permission_creation(
            server in valid_tool_name(),
            tool in valid_tool_name(),
        ) {
            // Test that different tool permission types can be created
            let bash_permission = ToolPermission::bash(&tool);
            let mcp_permission = ToolPermission::mcp(&server, &tool);

            // Permissions should be debuggable
            let bash_debug = format!("{:?}", bash_permission);
            let mcp_debug = format!("{:?}", mcp_permission);

            prop_assert!(bash_debug.contains("Bash"));
            prop_assert!(mcp_debug.contains("Mcp"));
        }

        #[test]
        fn test_message_type_serialization_roundtrip(
            message_type in prop::sample::select(vec![
                MessageType::Init,
                MessageType::User,
                MessageType::Assistant,
                MessageType::Result,
                MessageType::System,
                MessageType::Tool,
                MessageType::ToolResult,
            ])
        ) {
            let json = serde_json::to_string(&message_type).unwrap();
            let deserialized: MessageType = serde_json::from_str(&json).unwrap();

            prop_assert_eq!(message_type, deserialized);
        }
    }
}
