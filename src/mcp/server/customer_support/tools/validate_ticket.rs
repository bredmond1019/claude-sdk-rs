use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::server::customer_support::CustomerCareEventData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicketValidation {
    pub is_valid: bool,
    pub missing_fields: Vec<String>,
    pub validation_message: String,
}

#[derive(Debug, Clone, Default)]
pub struct ValidateTicketNode;

impl ValidateTicketNode {
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
            "validate_ticket".to_string(),
            "Validates customer support ticket data".to_string(),
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
impl Node for ValidateTicketNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: CustomerCareEventData =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse event data: {}", e),
            })?;

        let mut missing_fields = Vec::new();
        if event_data.ticket_id.is_empty() {
            missing_fields.push("ticket_id".to_string());
        }
        if event_data.customer_id.is_empty() {
            missing_fields.push("customer_id".to_string());
        }

        let validation = TicketValidation {
            is_valid: missing_fields.is_empty(),
            missing_fields: missing_fields.clone(),
            validation_message: if missing_fields.is_empty() {
                "Ticket is valid".to_string()
            } else {
                format!("Missing required fields: {}", missing_fields.join(", "))
            },
        };

        Ok(serde_json::json!({
            "validation": validation,
            "event_data": event_data
        }))
    }

    fn name(&self) -> &str {
        "ValidateTicketNode"
    }
}
