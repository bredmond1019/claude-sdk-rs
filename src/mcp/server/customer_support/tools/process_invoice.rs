use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvoiceRequest {
    pub invoice_id: String,
    pub customer_id: String,
    pub action: InvoiceAction,
    pub amount: Option<f64>,
    pub reason: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvoiceAction {
    Refund,
    Credit,
    Adjustment,
    Cancel,
    Resend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvoiceResult {
    pub transaction_id: String,
    pub invoice_id: String,
    pub action_taken: InvoiceAction,
    pub amount_processed: Option<f64>,
    pub original_amount: f64,
    pub new_balance: f64,
    pub timestamp: DateTime<Utc>,
    pub status: String,
    pub confirmation_sent: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ProcessInvoiceNode;

impl ProcessInvoiceNode {
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
            "process_invoice".to_string(),
            "Processes invoice-related customer support requests".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "invoice_id": { "type": "string" },
                    "customer_id": { "type": "string" },
                    "action": { "type": "string" }
                },
                "required": ["invoice_id", "customer_id", "action"]
            }),
            TypeId::of::<Self>(),
        );

        server
            .get_server()
            .register_node_as_tool(node, metadata)
            .await
    }

    fn process_invoice(&self, request: InvoiceRequest) -> Result<InvoiceResult, WorkflowError> {
        // Mock implementation
        let original_amount = 149.99; // Mock original invoice amount

        let (amount_processed, new_balance) = match request.action {
            InvoiceAction::Refund => {
                let refund_amount = request.amount.unwrap_or(original_amount);
                (Some(refund_amount), original_amount - refund_amount)
            }
            InvoiceAction::Credit => {
                let credit_amount = request.amount.unwrap_or(original_amount * 0.1);
                (Some(credit_amount), original_amount - credit_amount)
            }
            InvoiceAction::Adjustment => {
                let adjustment = request.amount.unwrap_or(0.0);
                (Some(adjustment.abs()), original_amount + adjustment)
            }
            InvoiceAction::Cancel => (None, 0.0),
            InvoiceAction::Resend => (None, original_amount),
        };

        Ok(InvoiceResult {
            transaction_id: format!("TXN-{}", uuid::Uuid::new_v4()),
            invoice_id: request.invoice_id,
            action_taken: request.action,
            amount_processed,
            original_amount,
            new_balance,
            timestamp: Utc::now(),
            status: "completed".to_string(),
            confirmation_sent: true,
        })
    }
}

#[async_trait]
impl Node for ProcessInvoiceNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Parse the invoice request
        let request: InvoiceRequest =
            serde_json::from_value(input.clone()).map_err(|e| WorkflowError::ValidationError {
                message: format!("Failed to parse invoice request: {}", e),
            })?;

        // Process the invoice
        let result = self.process_invoice(request)?;

        Ok(serde_json::to_value(result)?)
    }

    fn name(&self) -> &str {
        "ProcessInvoiceNode"
    }
}
