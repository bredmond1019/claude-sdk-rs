use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::server::customer_support::CustomerCareEventData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedResponse {
    pub response_text: String,
    pub tone: String,
    pub includes_next_steps: bool,
    pub estimated_resolution_time: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct GenerateResponseNode;

impl GenerateResponseNode {
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
            "generate_response".to_string(),
            "Generates responses for customer support tickets".to_string(),
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
impl Node for GenerateResponseNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: CustomerCareEventData =
            serde_json::from_value(input.get("event_data").cloned().unwrap_or(input.clone()))
                .map_err(|e| WorkflowError::ValidationError {
                    message: format!("Failed to parse event data: {}", e),
                })?;

        // Get previous analysis results from the input
        let ticket_analysis = input.get("ticket_analysis");
        let intent_analysis = input.get("intent_analysis");

        // Generate a mock response based on the analyses
        let response = GeneratedResponse {
            response_text: format!(
                "Thank you for contacting us regarding your inquiry. We understand your concern and are here to help. {}",
                "Our team will review your request and get back to you within 24 hours."
            ),
            tone: "professional".to_string(),
            includes_next_steps: true,
            estimated_resolution_time: Some("24 hours".to_string()),
        };

        Ok(serde_json::json!({
            "generated_response": response,
            "event_data": event_data,
            "ticket_analysis": ticket_analysis,
            "intent_analysis": intent_analysis
        }))
    }

    fn name(&self) -> &str {
        "GenerateResponseNode"
    }
}
