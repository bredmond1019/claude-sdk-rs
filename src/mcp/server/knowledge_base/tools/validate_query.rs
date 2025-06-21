/// Query Validation Node - Validates user queries for knowledge base processing
///
/// This node performs basic validation checks on user queries to ensure they meet
/// minimum requirements for processing through the knowledge base workflow.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};
use crate::mcp::server::knowledge_base::KnowledgeBaseEventData;

/// Validates user queries before processing
///
/// Performs validation checks including:
/// - Query length validation (3-1000 characters)
/// - Non-empty query validation
/// - Sets validation status in the task context
#[derive(Debug, Clone)]
pub struct ValidateQueryNode;
#[async_trait]
impl Node for ValidateQueryNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: KnowledgeBaseEventData =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse event data: {}", e),
            })?;

        // Basic validation checks
        let is_valid = !event_data.user_query.trim().is_empty()
            && event_data.user_query.len() >= 3
            && event_data.user_query.len() <= 1000;

        let mut result = serde_json::json!({
            "event_data": event_data,
            "query_valid": is_valid
        });

        if !is_valid {
            result["validation_error"] =
                Value::String("Query must be between 3 and 1000 characters".to_string());
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "ValidateQueryNode"
    }
}
