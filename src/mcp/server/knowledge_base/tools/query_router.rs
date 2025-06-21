/// Query Router Node - Prepares and routes user queries for knowledge base search
///
/// This node acts as the entry point for knowledge base queries, performing initial
/// processing and keyword extraction to prepare the query for downstream search operations.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};
use crate::mcp::server::knowledge_base::KnowledgeBaseEventData;

use super::extract_keywords;

/// Routes and prepares user queries for knowledge base processing
///
/// Responsibilities:
/// - Validates that a user query is present
/// - Extracts meaningful keywords for better search results
/// - Sets up the task context for downstream processing
#[derive(Debug, Clone)]
pub struct QueryRouterNode;
#[async_trait]
impl Node for QueryRouterNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: KnowledgeBaseEventData =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse event data: {}", e),
            })?;

        if event_data.user_query.trim().is_empty() {
            return Err(WorkflowError::ValidationError {
                message: "User query cannot be empty".to_string(),
            });
        }

        // Extract keywords for better search
        let keywords = extract_keywords(&event_data.user_query);

        // Return the processed data
        Ok(serde_json::json!({
            "event_data": event_data,
            "search_keywords": keywords,
            "query_processed": true,
            "ready_for_search": true
        }))
    }

    fn name(&self) -> &str {
        "QueryRouterNode"
    }
}
