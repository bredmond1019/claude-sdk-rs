use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::server::customer_support::CustomerCareEventData;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CustomerIntent {
    ProblemSolving,
    BillingInquiry,
    GeneralQuestion,
    AccountManagement,
    FeatureRequest,
    FeedbackComplaint,
    Other,
}

#[derive(Debug, Clone, Default)]
pub struct DetermineTicketIntentNode;

impl DetermineTicketIntentNode {
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
            "determine_intent".to_string(),
            "Determines the intent of customer support tickets".to_string(),
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

    fn determine_intent(
        &self,
        event_data: CustomerCareEventData,
    ) -> Result<IntentAnalysis, WorkflowError> {
        // Simple keyword-based intent detection
        let message_lower = event_data.message.to_lowercase();

        let (intent, confidence) = if message_lower.contains("bill")
            || message_lower.contains("invoice")
            || message_lower.contains("charge")
        {
            (CustomerIntent::BillingInquiry, 0.85)
        } else if message_lower.contains("problem")
            || message_lower.contains("issue")
            || message_lower.contains("error")
        {
            (CustomerIntent::ProblemSolving, 0.80)
        } else if message_lower.contains("feature") || message_lower.contains("request") {
            (CustomerIntent::FeatureRequest, 0.75)
        } else if message_lower.contains("account")
            || message_lower.contains("password")
            || message_lower.contains("login")
        {
            (CustomerIntent::AccountManagement, 0.80)
        } else if message_lower.contains("complaint")
            || message_lower.contains("disappointed")
            || message_lower.contains("unhappy")
        {
            (CustomerIntent::FeedbackComplaint, 0.85)
        } else if message_lower.contains("how")
            || message_lower.contains("what")
            || message_lower.contains("where")
        {
            (CustomerIntent::GeneralQuestion, 0.70)
        } else {
            (CustomerIntent::Other, 0.50)
        };

        Ok(IntentAnalysis {
            reasoning: format!("Detected intent based on keywords in message"),
            intent,
            confidence,
            escalate: matches!(intent, CustomerIntent::FeedbackComplaint) || confidence < 0.6,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentAnalysis {
    pub reasoning: String,
    pub intent: CustomerIntent,
    pub confidence: f32,
    pub escalate: bool,
}

#[async_trait]
impl Node for DetermineTicketIntentNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: CustomerCareEventData =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse event data: {}", e),
            })?;

        let analysis = self.determine_intent(event_data.clone())?;

        Ok(serde_json::json!({
            "intent_analysis": analysis,
            "event_data": event_data
        }))
    }

    fn name(&self) -> &str {
        "DetermineTicketIntentNode"
    }
}
