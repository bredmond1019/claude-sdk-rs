/// Spam Filter Node - Filters out spam queries from knowledge base processing
///
/// This node implements basic spam detection to prevent malicious or irrelevant
/// queries from consuming knowledge base resources.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};
use crate::mcp::server::knowledge_base::KnowledgeBaseEventData;

/// Filters spam queries before knowledge base processing
///
/// Uses a simple keyword-based approach to detect common spam indicators.
/// Queries identified as spam are marked and can be rejected by downstream nodes.
#[derive(Debug, Clone)]
pub struct FilterSpamQueryNode;
#[async_trait]
impl Node for FilterSpamQueryNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: KnowledgeBaseEventData =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse event data: {}", e),
            })?;

        // Simple spam detection using common spam indicators
        let spam_indicators =
            ["viagra", "lottery", "winner", "congratulations", "click here", "free money"];
        let is_spam = spam_indicators
            .iter()
            .any(|&indicator| event_data.user_query.to_lowercase().contains(indicator));

        Ok(serde_json::json!({
            "event_data": event_data,
            "is_spam": is_spam,
            "spam_check_completed": true
        }))
    }

    fn name(&self) -> &str {
        "FilterSpamQueryNode"
    }
}
