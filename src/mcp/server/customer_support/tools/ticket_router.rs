use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};
use crate::mcp::server::ToolMetadata;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub target_department: String,
    pub assigned_agent: Option<String>,
    pub escalation_level: u8,
    pub priority: String,
    pub routing_reason: String,
    pub estimated_response_time: String,
}

#[derive(Debug, Clone, Default)]
pub struct TicketRouterNode;

impl TicketRouterNode {
    pub fn new() -> Self {
        Self
    }

    pub async fn register(
        server: &mut super::super::server::CustomerSupportMCPServer,
    ) -> Result<(), WorkflowError> {
        use std::any::TypeId;

        let node = Arc::new(Self::new());
        let metadata = ToolMetadata::new(
            "ticket_router".to_string(),
            "Routes customer support tickets to appropriate departments".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "ticket_id": { "type": "string" },
                    "department": { "type": "string" },
                    "priority": { "type": "string" }
                },
                "required": ["ticket_id", "department", "priority"]
            }),
            TypeId::of::<Self>(),
        );

        server
            .get_server()
            .register_node_as_tool(node, metadata)
            .await
    }

    fn determine_routing(
        &self,
        _ticket_id: &str,
        department: &str,
        priority: &str,
        routing_reason: &str,
        agent_id: Option<String>,
    ) -> Result<RoutingDecision, WorkflowError> {
        // Intelligent routing logic based on department and priority
        let (target_department, escalation_level) = match department.to_lowercase().as_str() {
            "billing" => ("Billing Department", 1),
            "technical" | "tech" => ("Technical Support", 2),
            "general" => ("Customer Service", 1),
            "complaint" => ("Escalation Team", 3),
            _ => ("Customer Service", 1),
        };

        let estimated_response_time = match (priority.to_lowercase().as_str(), escalation_level) {
            ("urgent", 3) => "30 minutes",
            ("urgent", _) => "1 hour",
            ("high", _) => "2 hours",
            ("normal", _) => "4 hours",
            ("low", _) => "24 hours",
            _ => "4 hours",
        };

        Ok(RoutingDecision {
            target_department: target_department.to_string(),
            assigned_agent: agent_id,
            escalation_level,
            priority: priority.to_string(),
            routing_reason: routing_reason.to_string(),
            estimated_response_time: estimated_response_time.to_string(),
        })
    }

    fn validate_department(&self, department: &str) -> Result<String, WorkflowError> {
        let valid_departments =
            ["billing", "technical", "tech", "general", "complaint", "sales", "support"];

        let dept_lower = department.to_lowercase();
        if valid_departments.contains(&dept_lower.as_str()) {
            Ok(department.to_string())
        } else {
            Err(WorkflowError::ValidationError {
                message: format!(
                    "Invalid department: {}. Valid departments are: {:?}",
                    department, valid_departments
                ),
            })
        }
    }
}

#[async_trait]
impl Node for TicketRouterNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract routing parameters from input
        let ticket_id = input
            .get("ticket_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::ValidationError {
                message: "Missing required field: ticket_id".to_string(),
            })?;

        let department = input
            .get("department")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::ValidationError {
                message: "Missing required field: department".to_string(),
            })?;

        let routing_reason = input
            .get("routing_reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::ValidationError {
                message: "Missing required field: routing_reason".to_string(),
            })?;

        let priority = input
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("normal");

        let agent_id = input
            .get("agent_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Validate department
        let validated_department = self.validate_department(department)?;

        // Implement intelligent routing logic
        let routing_decision = self.determine_routing(
            ticket_id,
            &validated_department,
            priority,
            routing_reason,
            agent_id,
        )?;

        // Log the routing decision
        tracing::info!(
            "Routing ticket {} to {} (Priority: {}, Escalation Level: {})",
            ticket_id,
            routing_decision.target_department,
            routing_decision.priority,
            routing_decision.escalation_level
        );

        // Return the routing decision
        Ok(serde_json::json!({
            "routing_decision": routing_decision,
            "ticket_routed": true,
            "routed_at": chrono::Utc::now(),
            "ticket_id": ticket_id
        }))
    }

    fn name(&self) -> &str {
        "TicketRouterNode"
    }
}
