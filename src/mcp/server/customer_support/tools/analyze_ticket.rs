use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::agent::{AgentConfig, ModelProvider};
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::server::customer_support::CustomerCareEventData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicketAnalysis {
    pub sentiment: String,
    pub urgency: String,
    pub category: String,
    pub key_issues: Vec<String>,
    pub suggested_action: String,
    pub requires_human_review: bool,
}

#[derive(Debug, Clone)]
pub struct AnalyzeTicketNode {
    pub agent_config: Option<AgentConfig>,
}

impl Default for AnalyzeTicketNode {
    fn default() -> Self {
        Self { agent_config: None }
    }
}

impl AnalyzeTicketNode {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_agent_config(mut self, config: AgentConfig) -> Self {
        self.agent_config = Some(config);
        self
    }

    pub async fn register(
        server: &mut super::super::server::CustomerSupportMCPServer,
    ) -> Result<(), WorkflowError> {
        use crate::mcp::server::ToolMetadata;
        use std::any::TypeId;
        use std::sync::Arc;

        let node = Arc::new(Self::new());
        let metadata = ToolMetadata::new(
            "analyze_ticket".to_string(),
            "Analyzes customer support tickets for sentiment, urgency, and category".to_string(),
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
impl Node for AnalyzeTicketNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let event_data: CustomerCareEventData =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse event data: {}", e),
            })?;

        // Create AI agent configuration for ticket analysis
        let _agent_config = AgentConfig {
            model: "claude-3-sonnet-20240229".to_string(),
            provider: ModelProvider::Anthropic,
            temperature: Some(0.3),
            max_tokens: Some(4000),
            system_prompt: Some("You are an expert customer support analyst. Analyze customer support tickets to extract sentiment, urgency, category, and other relevant metrics. Respond with a JSON object containing your analysis.".to_string()),
        };

        // For now, provide a mock analysis
        let analysis = TicketAnalysis {
            sentiment: "neutral".to_string(),
            urgency: "medium".to_string(),
            category: "technical_support".to_string(),
            key_issues: vec!["configuration issue".to_string()],
            suggested_action: "Provide troubleshooting steps".to_string(),
            requires_human_review: false,
        };

        Ok(serde_json::json!({
            "ticket_analysis": analysis,
            "event_data": event_data
        }))
    }

    fn name(&self) -> &str {
        "AnalyzeTicketNode"
    }
}
