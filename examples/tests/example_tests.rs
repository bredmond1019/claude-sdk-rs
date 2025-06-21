//! Tests for the example applications
//!
//! These tests ensure that the examples compile and can be instantiated correctly.
//! They don't test full functionality (which would require Claude CLI) but verify
//! the code structure is sound.

#[cfg(test)]
mod tests {
    use claude_sdk_rs::{Client, Config, SessionId, StreamFormat, ToolPermission};
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn test_basic_client_creation() {
        // Test from example 01
        let client = Client::builder().build().unwrap();
        let _query = client.query("test");

        let client2 = Client::builder()
            .model("claude-3-sonnet-20240229")
            .timeout_secs(30)
            .build();
        let _query2 = client2.query("test");
    }

    #[test]
    fn test_config_builder() {
        // Test configuration from examples
        let config = Config::builder()
            .model("claude-3-sonnet-20240229")
            .system_prompt("Test prompt")
            .stream_format(StreamFormat::Json)
            .timeout_secs(60)
            .build();

        assert_eq!(config.model, Some("claude-3-sonnet-20240229".to_string()));
        assert_eq!(config.system_prompt, Some("Test prompt".to_string()));
        assert_eq!(config.stream_format, StreamFormat::Json);
        assert_eq!(config.timeout_secs, Some(60));
    }

    #[test]
    fn test_session_id_creation() {
        // Test from example 02
        let session_id = SessionId::new(Uuid::new_v4().to_string());
        assert!(!session_id.as_str().is_empty());

        // Test multiple sessions
        let rust_session = SessionId::new(Uuid::new_v4().to_string());
        let python_session = SessionId::new(Uuid::new_v4().to_string());
        assert_ne!(rust_session.as_str(), python_session.as_str());
    }

    #[test]
    fn test_stream_format_configuration() {
        // Test from example 03
        let client1 = Client::builder()
            .stream_format(StreamFormat::StreamJson)
            .build();

        let client2 = Client::builder().stream_format(StreamFormat::Text).build();

        let client3 = Client::builder().stream_format(StreamFormat::Json).build();

        // All clients should be created successfully
        let _q1 = client1.query("test");
        let _q2 = client2.query("test");
        let _q3 = client3.query("test");
    }

    #[test]
    fn test_tool_permissions() {
        // Test from example 04
        let bash_perm = ToolPermission::bash("ls");
        assert_eq!(bash_perm.to_cli_format(), "bash:ls");

        let mcp_perm = ToolPermission::mcp("filesystem", "read");
        assert_eq!(mcp_perm.to_cli_format(), "mcp__filesystem__read");

        let all_perm = ToolPermission::All;
        assert_eq!(all_perm.to_cli_format(), "*");

        // Test client with tools
        let client = Client::builder()
            .allowed_tools(vec![
                ToolPermission::bash("date").to_cli_format(),
                ToolPermission::mcp("filesystem", "read").to_cli_format(),
            ])
            .build();

        let _query = client.query("test");
    }

    #[test]
    fn test_cost_tracker() {
        // Test from example 05
        #[derive(Debug)]
        struct CostTracker {
            total_cost: f64,
            session_costs: HashMap<String, f64>,
        }

        impl CostTracker {
            fn new() -> Self {
                Self {
                    total_cost: 0.0,
                    session_costs: HashMap::new(),
                }
            }

            fn record_cost(&mut self, session_id: &str, cost: f64) {
                self.total_cost += cost;
                *self
                    .session_costs
                    .entry(session_id.to_string())
                    .or_insert(0.0) += cost;
            }

            fn get_session_cost(&self, session_id: &str) -> f64 {
                self.session_costs.get(session_id).copied().unwrap_or(0.0)
            }
        }

        let mut tracker = CostTracker::new();
        tracker.record_cost("session1", 0.001);
        tracker.record_cost("session1", 0.002);
        tracker.record_cost("session2", 0.003);

        assert_eq!(tracker.total_cost, 0.006);
        assert_eq!(tracker.get_session_cost("session1"), 0.003);
        assert_eq!(tracker.get_session_cost("session2"), 0.003);
        assert_eq!(tracker.get_session_cost("session3"), 0.0);
    }

    #[test]
    fn test_query_builder_methods() {
        let client = Client::builder().build().unwrap();
        let session_id = SessionId::new("test-session".to_string());

        // Test query builder chain - just ensure it builds without errors
        let _query = client
            .query("test query")
            .session(session_id.clone())
            .format(StreamFormat::Json);

        // Can't test private fields, but we can test the build succeeds
        assert!(!session_id.as_str().is_empty());
    }

    #[test]
    fn test_example_helper_functions() {
        // Test truncate function from examples
        fn truncate_response(text: &str, max_len: usize) -> String {
            if text.len() <= max_len {
                text.to_string()
            } else {
                format!("{}...", &text[..max_len])
            }
        }

        assert_eq!(truncate_response("short", 10), "short");
        assert_eq!(
            truncate_response("this is a long text", 10),
            "this is a ..."
        );

        // Test indent function
        fn indent_text(text: &str, spaces: usize) -> String {
            let indent = " ".repeat(spaces);
            text.lines()
                .map(|line| format!("{}{}", indent, line))
                .collect::<Vec<_>>()
                .join("\n")
        }

        assert_eq!(indent_text("line1\nline2", 2), "  line1\n  line2");
    }

    #[test]
    fn test_app_modes() {
        #[derive(Debug, Clone, PartialEq)]
        enum AppMode {
            Chat,
            Development,
            Analysis,
        }

        let mode1 = AppMode::Chat;
        let mode2 = AppMode::Development;
        let mode3 = AppMode::Analysis;

        assert_ne!(mode1, mode2);
        assert_ne!(mode2, mode3);
        assert_eq!(mode1.clone(), AppMode::Chat);
    }
}
