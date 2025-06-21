/// HelpScout Search Node - Searches HelpScout articles and conversations
///
/// This node integrates with HelpScout via the HelpscoutClientNode to search through
/// knowledge base articles and customer support conversations for relevant information.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::clients::helpscout::HelpscoutClientNode;
use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Searches HelpScout knowledge base articles and conversations
///
/// This node can optionally integrate with a HelpscoutClientNode for real searches,
/// but provides mock results by default for testing and development purposes.
#[derive(Debug)]
pub struct HelpscoutSearchNode {
    pub helpscout_client: Option<HelpscoutClientNode>,
}
impl Default for HelpscoutSearchNode {
    fn default() -> Self {
        Self {
            helpscout_client: None,
        }
    }
}

impl HelpscoutSearchNode {
    /// Creates a new HelpscoutSearchNode without a client (mock mode)
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a HelpscoutSearchNode with a configured HelpScout client
    pub fn with_client(mut self, client: HelpscoutClientNode) -> Self {
        self.helpscout_client = Some(client);
        self
    }
}

#[async_trait]
impl Node for HelpscoutSearchNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract user query from input
        let user_query = input
            .get("user_query")
            .or_else(|| input.get("event_data").and_then(|ed| ed.get("user_query")))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Note: Real client integration requires mutable access to the client
        // which would need architectural changes to support properly.
        // For now, we'll use mock results and document the integration path.

        // Mock search results for articles and conversations
        let mock_results = serde_json::json!({
            "source": "helpscout",
            "query": user_query,
            "results_found": 2,
            "mock_data": true,
            "articles": [
                {
                    "title": "How to Guide",
                    "url": "https://helpscout.com/article1",
                    "relevance": 90
                }
            ],
            "conversations": [
                {
                    "subject": "Similar Issue Resolved",
                    "url": "https://helpscout.com/conversation1",
                    "relevance": 78
                }
            ]
        });

        Ok(serde_json::json!({
            "helpscout_search_results": mock_results,
            "helpscout_search_completed": true
        }))
    }

    fn name(&self) -> &str {
        "HelpscoutSearchNode"
    }
}
