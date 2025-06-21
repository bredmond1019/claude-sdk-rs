use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseTicketRequest {
    pub ticket_id: String,
    pub customer_id: String,
    pub resolution: String,
    pub resolution_code: ResolutionCode,
    pub customer_satisfaction: Option<i32>,
    pub follow_up_required: bool,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionCode {
    Resolved,
    WorkaroundProvided,
    NoActionRequired,
    CustomerWithdrew,
    Duplicate,
    CannotReproduce,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseTicketResult {
    pub ticket_id: String,
    pub customer_id: String,
    pub closed_at: DateTime<Utc>,
    pub resolution_code: ResolutionCode,
    pub time_to_resolution: String,
    pub satisfaction_score: Option<i32>,
    pub survey_sent: bool,
    pub knowledge_base_updated: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CloseTicketNode;

impl CloseTicketNode {
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
            "close_ticket".to_string(),
            "Closes customer support tickets".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "ticket_id": { "type": "string" },
                    "customer_id": { "type": "string" },
                    "resolution": { "type": "string" }
                },
                "required": ["ticket_id", "customer_id", "resolution"]
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
impl Node for CloseTicketNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract close ticket request data
        let request: CloseTicketRequest =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse close ticket request: {}", e),
            })?;

        // Calculate time to resolution (mock)
        let time_to_resolution = "2 hours 15 minutes".to_string();

        // Determine if survey should be sent
        let survey_sent = request.customer_satisfaction.is_none();

        // Check if knowledge base should be updated based on resolution
        let knowledge_base_updated = matches!(
            request.resolution_code,
            ResolutionCode::Resolved | ResolutionCode::WorkaroundProvided
        );

        let result = CloseTicketResult {
            ticket_id: request.ticket_id,
            customer_id: request.customer_id,
            closed_at: Utc::now(),
            resolution_code: request.resolution_code,
            time_to_resolution,
            satisfaction_score: request.customer_satisfaction,
            survey_sent,
            knowledge_base_updated,
        };

        Ok(serde_json::json!({
            "close_ticket_result": result,
            "ticket_closed": true,
            "follow_up_required": request.follow_up_required
        }))
    }

    fn name(&self) -> &str {
        "CloseTicketNode"
    }
}
