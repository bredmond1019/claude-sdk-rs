/// Search Router Node - Initiates parallel searches across knowledge sources
///
/// This node acts as a router that validates query processing status and
/// initiates parallel searches across multiple knowledge sources including
/// Notion documentation, HelpScout articles, and Slack conversations.
use serde_json::Value;

use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

use async_trait::async_trait;

/// Routes validated queries to parallel search operations
///
/// Responsibilities:
/// - Validates that the query passed validation and spam checks
/// - Sets up search context for parallel execution
/// - Prepares search readiness flags for downstream search nodes
#[derive(Debug, Clone)]
pub struct SearchRouterNode;
#[async_trait]
impl Node for SearchRouterNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Check validation results
        let is_valid = input
            .get("query_valid")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let is_spam = input
            .get("is_spam")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !is_valid || is_spam {
            return Err(WorkflowError::ValidationError {
                message: "Query failed validation or was detected as spam".to_string(),
            });
        }

        // Return search readiness flags
        Ok(serde_json::json!({
            "search_initiated": true,
            "notion_search_ready": true,
            "helpscout_search_ready": true,
            "slack_search_ready": true,
            "event_data": input.get("event_data").cloned().unwrap_or(input.clone())
        }))
    }

    fn name(&self) -> &str {
        "SearchRouterNode"
    }
}
