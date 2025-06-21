use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};
use crate::mcp::server::ToolMetadata;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplyResult {
    pub ticket_id: String,
    pub message_id: String,
    pub channel: String,
    pub delivery_status: String,
    pub sent_at: String,
    pub priority: String,
}

#[derive(Debug, Clone, Default)]
pub struct SendReplyNode;

impl SendReplyNode {
    pub fn new() -> Self {
        Self
    }

    pub async fn register(
        server: &mut super::super::server::CustomerSupportMCPServer,
    ) -> Result<(), WorkflowError> {
        use std::any::TypeId;

        let node = Arc::new(Self::new());
        let metadata = ToolMetadata::new(
            "send_reply".to_string(),
            "Sends replies to customers through various channels".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "ticket_id": { "type": "string" },
                    "message": { "type": "string" },
                    "channel": { "type": "string" },
                    "priority": { "type": "string" }
                },
                "required": ["ticket_id", "message", "channel", "priority"]
            }),
            TypeId::of::<Self>(),
        );

        server
            .get_server()
            .register_node_as_tool(node, metadata)
            .await
    }

    fn validate_message_content(&self, message: &str) -> Result<(), WorkflowError> {
        if message.trim().is_empty() {
            return Err(WorkflowError::ValidationError {
                message: "Reply message cannot be empty".to_string(),
            });
        }

        if message.len() > 10000 {
            return Err(WorkflowError::ValidationError {
                message: "Reply message exceeds maximum length of 10000 characters".to_string(),
            });
        }

        // Check for inappropriate content (basic validation)
        let message_lower = message.to_lowercase();
        let inappropriate_words = ["spam", "scam", "inappropriate"];

        for word in &inappropriate_words {
            if message_lower.contains(word) {
                tracing::warn!(
                    "Potentially inappropriate content detected in reply: {}",
                    word
                );
            }
        }

        Ok(())
    }

    fn validate_channel(&self, channel: &str) -> Result<String, WorkflowError> {
        match channel.to_lowercase().as_str() {
            "email" => Ok("email".to_string()),
            "chat" | "live_chat" => Ok("chat".to_string()),
            "phone" | "call" => Ok("phone".to_string()),
            "sms" | "text" => Ok("sms".to_string()),
            _ => Err(WorkflowError::ValidationError {
                message: format!("Unsupported communication channel: {}", channel),
            }),
        }
    }

    fn send_reply_through_channel(
        &self,
        ticket_id: &str,
        message: &str,
        channel: &str,
        priority: &str,
    ) -> Result<ReplyResult, WorkflowError> {
        let message_id = self.generate_message_id(ticket_id, channel);
        let sent_at = chrono::Utc::now().to_rfc3339();

        // Simulate sending through different channels
        let delivery_status = match channel {
            "email" => self.send_email_reply(ticket_id, message, priority)?,
            "chat" => self.send_chat_reply(ticket_id, message, priority)?,
            "phone" => self.queue_phone_callback(ticket_id, message, priority)?,
            "sms" => self.send_sms_reply(ticket_id, message, priority)?,
            _ => "unknown".to_string(),
        };

        Ok(ReplyResult {
            ticket_id: ticket_id.to_string(),
            message_id,
            channel: channel.to_string(),
            delivery_status,
            sent_at,
            priority: priority.to_string(),
        })
    }

    fn generate_message_id(&self, ticket_id: &str, channel: &str) -> String {
        format!(
            "MSG-{}-{}-{}",
            ticket_id,
            channel,
            chrono::Utc::now().timestamp()
        )
    }

    fn send_email_reply(
        &self,
        _ticket_id: &str,
        _message: &str,
        _priority: &str,
    ) -> Result<String, WorkflowError> {
        // Mock implementation - in real system would integrate with email service
        Ok("delivered".to_string())
    }

    fn send_chat_reply(
        &self,
        _ticket_id: &str,
        _message: &str,
        _priority: &str,
    ) -> Result<String, WorkflowError> {
        // Mock implementation - in real system would integrate with chat platform
        Ok("delivered".to_string())
    }

    fn queue_phone_callback(
        &self,
        _ticket_id: &str,
        _message: &str,
        _priority: &str,
    ) -> Result<String, WorkflowError> {
        // Mock implementation - in real system would queue for phone callback
        Ok("queued".to_string())
    }

    fn send_sms_reply(
        &self,
        _ticket_id: &str,
        _message: &str,
        _priority: &str,
    ) -> Result<String, WorkflowError> {
        // Mock implementation - in real system would integrate with SMS gateway
        Ok("delivered".to_string())
    }
}

#[async_trait]
impl Node for SendReplyNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract reply parameters from input
        let ticket_id = input
            .get("ticket_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::ValidationError {
                message: "Missing required field: ticket_id".to_string(),
            })?;

        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::ValidationError {
                message: "Missing required field: message".to_string(),
            })?;

        let channel = input
            .get("channel")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::ValidationError {
                message: "Missing required field: channel".to_string(),
            })?;

        let priority = input
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("normal");

        // Validate message content
        self.validate_message_content(message)?;

        // Validate channel
        let validated_channel = self.validate_channel(channel)?;

        // Send the reply through the appropriate channel
        let reply_result =
            self.send_reply_through_channel(ticket_id, message, &validated_channel, priority)?;

        // Log the reply sending
        tracing::info!(
            "Sent reply for ticket {} via {} (Message ID: {})",
            ticket_id,
            reply_result.channel,
            reply_result.message_id
        );

        // Return the reply result
        Ok(serde_json::json!({
            "reply_result": reply_result,
            "reply_sent": true,
            "sent_at": chrono::Utc::now()
        }))
    }

    fn name(&self) -> &str {
        "SendReplyNode"
    }
}
