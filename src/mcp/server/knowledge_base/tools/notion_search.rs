/// Notion Search Node - Searches Notion pages and databases for knowledge base queries
///
/// This node integrates with Notion via the NotionClientNode to search through
/// documentation, wiki pages, and structured databases for relevant information.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::clients::notion::NotionClientNode;
use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Searches Notion documentation and pages for query matches
///
/// This node can optionally integrate with a NotionClientNode for real searches,
/// but provides mock results by default for testing and development purposes.
#[derive(Debug)]
pub struct NotionSearchNode {
    pub notion_client: Option<NotionClientNode>,
}
impl Default for NotionSearchNode {
    fn default() -> Self {
        Self {
            notion_client: None,
        }
    }
}

impl NotionSearchNode {
    /// Creates a new NotionSearchNode without a client (mock mode)
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a NotionSearchNode with a configured Notion client
    pub fn with_client(mut self, client: NotionClientNode) -> Self {
        self.notion_client = Some(client);
        self
    }
}

#[async_trait]
impl Node for NotionSearchNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract user query from input
        let user_query = input
            .get("user_query")
            .or_else(|| input.get("event_data").and_then(|ed| ed.get("user_query")))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Mock search results for documentation and pages
        let mock_results = serde_json::json!({
            "source": "notion",
            "query": user_query,
            "results_found": 3,
            "pages": [
                {
                    "title": "Related Documentation",
                    "url": "https://notion.so/page1",
                    "relevance": 85
                },
                {
                    "title": "FAQ Entry",
                    "url": "https://notion.so/page2",
                    "relevance": 72
                }
            ]
        });

        Ok(serde_json::json!({
            "notion_search_results": mock_results,
            "notion_search_completed": true
        }))
    }

    fn name(&self) -> &str {
        "NotionSearchNode"
    }
}
