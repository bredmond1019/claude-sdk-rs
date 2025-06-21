use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRequest {
    pub ticket_id: String,
    pub customer_id: String,
    pub reason: String,
    pub priority: EscalationPriority,
    pub assigned_to: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationResult {
    pub escalation_id: String,
    pub ticket_id: String,
    pub escalated_to: String,
    pub priority: EscalationPriority,
    pub timestamp: DateTime<Utc>,
    pub estimated_response_time: String,
    pub notification_sent: bool,
}

#[derive(Debug, Clone, Default)]
pub struct EscalateTicketNode;

impl EscalateTicketNode {
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
            "escalate_ticket".to_string(),
            "Escalates customer support tickets to higher priority".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "ticket_id": { "type": "string" },
                    "customer_id": { "type": "string" },
                    "reason": { "type": "string" },
                    "priority": { "type": "string" }
                },
                "required": ["ticket_id", "customer_id", "reason", "priority"]
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
impl Node for EscalateTicketNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract escalation request data
        let request: EscalationRequest =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse escalation request: {}", e),
            })?;

        // Determine escalation target based on priority
        let escalated_to = match request.priority {
            EscalationPriority::Critical => "Senior Management",
            EscalationPriority::High => "Team Lead",
            EscalationPriority::Medium => "Senior Support",
            EscalationPriority::Low => "Support Queue",
        };

        // Generate escalation ID
        let escalation_id = format!("ESC-{}-{}", request.ticket_id, Utc::now().timestamp());

        // Calculate estimated response time based on priority
        let estimated_response_time = match request.priority {
            EscalationPriority::Critical => "30 minutes",
            EscalationPriority::High => "2 hours",
            EscalationPriority::Medium => "4 hours",
            EscalationPriority::Low => "24 hours",
        }
        .to_string();

        let result = EscalationResult {
            escalation_id,
            ticket_id: request.ticket_id,
            escalated_to: request
                .assigned_to
                .unwrap_or_else(|| escalated_to.to_string()),
            priority: request.priority,
            timestamp: Utc::now(),
            estimated_response_time,
            notification_sent: true,
        };

        Ok(serde_json::json!({
            "escalation_result": result,
            "escalation_completed": true
        }))
    }

    fn name(&self) -> &str {
        "EscalateTicketNode"
    }
}
