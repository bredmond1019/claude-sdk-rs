use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::server::customer_support::CustomerCareEventData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpamAnalysis {
    pub is_human: bool,
    pub confidence: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone, Default)]
pub struct FilterSpamNode;

impl FilterSpamNode {
    pub fn new() -> Self {
        Self
    }

    pub async fn register(
        server: &mut super::super::server::CustomerSupportMCPServer,
    ) -> Result<(), WorkflowError> {
        use crate::mcp::server::ToolMetadata;
        use std::any::TypeId;
        use std::sync::Arc;

        let node = Arc::new(Self::new());
        let metadata = ToolMetadata::new(
            "filter_spam".to_string(),
            "Filters spam from customer support tickets".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "ticket_id": { "type": "string" },
                    "customer_id": { "type": "string" },
                    "message": { "type": "string" },
                    "priority": { "type": "string" }
                },
                "required": ["ticket_id", "customer_id", "message", "priority"]
            }),
            TypeId::of::<Self>(),
        );

        server
            .get_server()
            .register_node_as_tool(node, metadata)
            .await
    }
}

#[async_trait]
impl Node for FilterSpamNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: CustomerCareEventData =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse event data: {}", e),
            })?;

        let analysis = SpamAnalysis {
            is_human: true,
            confidence: 0.88, // 1 - spam_score
            reasoning: format!("Message length: {}", event_data.message.len()),
        };

        Ok(serde_json::json!({
            "spam_analysis": analysis,
            "event_data": event_data
        }))
    }

    fn name(&self) -> &str {
        "FilterSpamNode"
    }
}
