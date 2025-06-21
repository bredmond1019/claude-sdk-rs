// Test utilities for integration testing

use serde_json::json;
use std::env;
use std::path::PathBuf;

use super::IntegrationTestConfig;
use crate::mcp::config::MCPConfig;
use crate::mcp::protocol::ToolDefinition;

/// Create a test configuration with defaults
pub fn create_test_config() -> IntegrationTestConfig {
    IntegrationTestConfig {
        use_mocks: env::var("USE_REAL_SERVICES").is_err(),
        mock_registry: None, // Will be created by harness
        timeout_ms: 5000,
        retry_attempts: 3,
        log_level: env::var("TEST_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
    }
}

/// Setup test environment variables
pub fn setup_test_environment() {
    // Set test-specific environment variables
    env::set_var("MCP_TEST_MODE", "true");

    // Set service credentials if not using mocks
    if env::var("USE_REAL_SERVICES").is_ok() {
        // HelpScout
        if env::var("HELPSCOUT_API_KEY").is_err() {
            env::set_var("HELPSCOUT_API_KEY", "test-api-key");
        }

        // Notion
        if env::var("NOTION_API_KEY").is_err() {
            env::set_var("NOTION_API_KEY", "test-api-key");
        }

        // Slack
        if env::var("SLACK_BOT_TOKEN").is_err() {
            env::set_var("SLACK_BOT_TOKEN", "xoxb-test-token");
        }
    }
}

/// Create test MCP configuration
pub fn create_test_mcp_config(name: &str) -> MCPConfig {
    let mut config = MCPConfig::default();
    config.client_name = format!("test-{}", name);
    config.enabled = true;
    config
}

/// Create common test tools
pub fn create_test_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "echo".to_string(),
            description: Some("Echo back the input".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    }
                },
                "required": ["message"]
            }),
        },
        ToolDefinition {
            name: "add".to_string(),
            description: Some("Add two numbers".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }),
        },
    ]
}

/// Test data generators
pub mod test_data {
    use serde_json::json;

    /// Generate a test HelpScout conversation
    pub fn helpscout_conversation(id: i32, status: &str) -> serde_json::Value {
        json!({
            "id": id,
            "status": status,
            "subject": format!("Test conversation {}", id),
            "mailbox": {
                "id": 1,
                "name": "Support"
            },
            "customer": {
                "id": 100 + id,
                "email": format!("customer{}@example.com", id)
            },
            "threads": []
        })
    }

    /// Generate a test Notion page
    pub fn notion_page(id: &str, title: &str) -> serde_json::Value {
        json!({
            "id": id,
            "object": "page",
            "created_time": "2023-01-01T00:00:00Z",
            "last_edited_time": "2023-01-01T00:00:00Z",
            "properties": {
                "title": {
                    "title": [{
                        "text": {
                            "content": title
                        }
                    }]
                }
            }
        })
    }

    /// Generate a test Slack channel
    pub fn slack_channel(id: &str, name: &str) -> serde_json::Value {
        json!({
            "id": id,
            "name": name,
            "is_channel": true,
            "created": 1234567890,
            "creator": "U123456",
            "is_archived": false,
            "is_general": name == "general",
            "is_member": true,
            "num_members": 10
        })
    }

    /// Generate a test customer support ticket
    pub fn support_ticket(id: &str, status: &str, priority: &str) -> serde_json::Value {
        json!({
            "id": id,
            "status": status,
            "priority": priority,
            "subject": format!("Test ticket {}", id),
            "description": "This is a test support ticket",
            "customer": {
                "name": "Test Customer",
                "email": "test@example.com"
            },
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        })
    }
}

/// Assertion helpers
pub mod assertions {
    use serde_json::Value;

    /// Assert JSON values are equal, ignoring specified fields
    pub fn assert_json_eq_ignore(actual: &Value, expected: &Value, ignore_fields: &[&str]) {
        match (actual, expected) {
            (Value::Object(a), Value::Object(e)) => {
                for (key, expected_value) in e {
                    if !ignore_fields.contains(&key.as_str()) {
                        let actual_value = a
                            .get(key)
                            .expect(&format!("Expected field '{}' not found", key));
                        assert_json_eq_ignore(actual_value, expected_value, ignore_fields);
                    }
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                assert_eq!(a.len(), e.len(), "Array lengths don't match");
                for (actual_item, expected_item) in a.iter().zip(e.iter()) {
                    assert_json_eq_ignore(actual_item, expected_item, ignore_fields);
                }
            }
            _ => {
                assert_eq!(actual, expected, "Values don't match");
            }
        }
    }

    /// Assert a result contains an expected error message
    pub fn assert_error_contains<T, E: std::fmt::Debug>(
        result: Result<T, E>,
        expected_message: &str,
    ) {
        match result {
            Ok(_) => panic!(
                "Expected error containing '{}', but got Ok",
                expected_message
            ),
            Err(e) => {
                let error_str = format!("{:?}", e);
                assert!(
                    error_str.contains(expected_message),
                    "Error message '{}' does not contain '{}'",
                    error_str,
                    expected_message
                );
            }
        }
    }
}

/// Test fixtures
pub mod fixtures {
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Create a temporary directory for testing
    pub fn temp_dir() -> TempDir {
        tempfile::tempdir().expect("Failed to create temp dir")
    }

    /// Create a test file with content
    pub fn create_test_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        use std::fs;
        let path = dir.path().join(name);
        fs::write(&path, content).expect("Failed to write test file");
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_config() {
        let config = create_test_config();
        assert!(config.use_mocks);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.retry_attempts, 3);
    }

    #[test]
    fn test_json_assertions() {
        use assertions::assert_json_eq_ignore;

        let actual = json!({
            "id": "123",
            "name": "test",
            "timestamp": "2023-01-01T00:00:00Z"
        });

        let expected = json!({
            "id": "123",
            "name": "test",
            "timestamp": "2023-01-02T00:00:00Z" // Different timestamp
        });

        // Should pass when ignoring timestamp
        assert_json_eq_ignore(&actual, &expected, &["timestamp"]);
    }

    #[test]
    fn test_error_assertion() {
        use assertions::assert_error_contains;

        let result: Result<(), String> = Err("Something went wrong: network error".to_string());
        assert_error_contains(result, "network error");
    }
}
