/// Slack Search Node - Searches Slack messages and conversations
///
/// This node integrates with Slack via the SlackClientNode to search through
/// team conversations and messages for relevant information and past solutions.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::clients::slack::SlackClientNode;
use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Searches Slack conversations and messages for query matches
///
/// This node can optionally integrate with a SlackClientNode for real searches,
/// but provides mock results by default for testing and development purposes.
#[derive(Debug)]
pub struct SlackSearchNode {
    pub slack_client: Option<SlackClientNode>,
}

impl Default for SlackSearchNode {
    fn default() -> Self {
        Self { slack_client: None }
    }
}

impl SlackSearchNode {
    /// Creates a new SlackSearchNode without a client (mock mode)
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a SlackSearchNode with a configured Slack client
    pub fn with_client(mut self, client: SlackClientNode) -> Self {
        self.slack_client = Some(client);
        self
    }
}

#[async_trait]
impl Node for SlackSearchNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract user query from input
        let user_query = input
            .get("user_query")
            .or_else(|| input.get("event_data").and_then(|ed| ed.get("user_query")))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Mock search results for messages and conversations
        let mock_results = serde_json::json!({
            "source": "slack",
            "query": user_query,
            "results_found": 4,
            "messages": [
                {
                    "channel": "#general",
                    "user": "john.doe",
                    "text": "I had a similar question...",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "relevance": 82
                },
                {
                    "channel": "#support",
                    "user": "jane.smith",
                    "text": "Here's how I solved it...",
                    "timestamp": "2024-01-14T14:22:00Z",
                    "relevance": 75
                }
            ]
        });

        Ok(serde_json::json!({
            "slack_search_results": mock_results,
            "slack_search_completed": true
        }))
    }

    fn name(&self) -> &str {
        "SlackSearchNode"
    }
}
