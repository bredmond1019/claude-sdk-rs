use crate::mcp::core::error::WorkflowError;
use serde::{Deserialize, Serialize};

use super::server::CustomerSupportMCPServer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerCareEventData {
    pub ticket_id: String,
    pub customer_id: String,
    pub message: String,
    pub priority: String,
}

mod analyze_ticket;
mod close_ticket;
mod determine_intent;
mod escalate_ticket;
mod filter_spam;
mod generate_response;
mod process_invoice;
mod send_reply;
mod ticket_router;
mod validate_ticket;

pub use analyze_ticket::AnalyzeTicketNode;
pub use close_ticket::CloseTicketNode;
pub use determine_intent::DetermineTicketIntentNode;
pub use escalate_ticket::EscalateTicketNode;
pub use filter_spam::FilterSpamNode;
pub use generate_response::GenerateResponseNode;
pub use process_invoice::ProcessInvoiceNode;
pub use send_reply::SendReplyNode;
pub use ticket_router::TicketRouterNode;
pub use validate_ticket::ValidateTicketNode;

pub async fn register_customer_support_tools(
    server: &mut CustomerSupportMCPServer,
) -> Result<(), WorkflowError> {
    ValidateTicketNode::register(server).await?;
    FilterSpamNode::register(server).await?;
    DetermineTicketIntentNode::register(server).await?;
    AnalyzeTicketNode::register(server).await?;
    GenerateResponseNode::register(server).await?;
    EscalateTicketNode::register(server).await?;
    ProcessInvoiceNode::register(server).await?;
    CloseTicketNode::register(server).await?;
    SendReplyNode::register(server).await?;
    TicketRouterNode::register(server).await?;
    Ok(())
}
