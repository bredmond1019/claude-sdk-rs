/// Slack Service Integration - Provides real API access for Slack operations
///
/// This service acts as a bridge between the workflow nodes and the actual Slack API,
/// handling authentication, error handling, and response formatting.
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::mcp::clients::services::SlackApiService;
use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Service node that provides real Slack API integration
#[derive(Debug)]
pub struct SlackServiceNode {
    api_service: Arc<RwLock<SlackApiService>>,
}

impl SlackServiceNode {
    /// Create a new Slack service node with bot token
    pub fn new(bot_token: String) -> Self {
        Self {
            api_service: Arc::new(RwLock::new(SlackApiService::new(bot_token))),
        }
    }

    /// Search messages using the real Slack API
    pub async fn search_messages(&self, query: &str) -> Result<Value, WorkflowError> {
        let api = self.api_service.read().await;
        let messages = api.search_messages(query, Some(20), None).await?;

        // Convert to workflow-friendly format
        let results = messages
            .into_iter()
            .map(|msg| {
                serde_json::json!({
                    "ts": msg.ts,
                    "text": msg.text,
                    "user": msg.user,
                    "channel": msg.channel,
                    "thread_ts": msg.thread_ts
                })
            })
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "source": "slack",
            "query": query,
            "results_found": results.len(),
            "messages": results,
            "real_api": true
        }))
    }

    /// Send a message to a channel
    pub async fn send_message(
        &self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> Result<Value, WorkflowError> {
        let api = self.api_service.read().await;
        let result = api.post_message(channel, text, thread_ts, None).await?;

        Ok(serde_json::json!({
            "success": true,
            "ts": result.ts,
            "channel": result.channel,
            "message": {
                "text": result.message.text,
                "user": result.message.user,
                "ts": result.message.ts
            }
        }))
    }

    /// List channels
    pub async fn list_channels(&self) -> Result<Value, WorkflowError> {
        let api = self.api_service.read().await;
        let channels = api
            .list_channels(true, Some("public_channel,private_channel"), Some(100))
            .await?;

        let channel_list = channels
            .into_iter()
            .map(|ch| {
                serde_json::json!({
                    "id": ch.id,
                    "name": ch.name,
                    "is_private": ch.is_private,
                    "is_archived": ch.is_archived,
                    "num_members": ch.num_members,
                    "topic": ch.topic.map(|t| t.value),
                    "purpose": ch.purpose.map(|p| p.value)
                })
            })
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "channels": channel_list,
            "count": channel_list.len()
        }))
    }

    /// Get user information
    pub async fn get_user_info(&self, user_id: &str) -> Result<Value, WorkflowError> {
        let api = self.api_service.read().await;
        let user = api.get_user_info(user_id).await?;

        Ok(serde_json::json!({
            "id": user.id,
            "name": user.name,
            "real_name": user.real_name,
            "is_bot": user.is_bot,
            "is_admin": user.is_admin,
            "email": user.profile.email,
            "display_name": user.profile.display_name,
            "status_text": user.profile.status_text,
            "status_emoji": user.profile.status_emoji
        }))
    }
}

#[async_trait]
impl Node for SlackServiceNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract operation type
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("search");

        match operation {
            "search" | "search_messages" => {
                let query = input
                    .get("query")
                    .or_else(|| input.get("user_query"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing query parameter".to_string())
                    })?;

                let results = self.search_messages(query).await?;

                Ok(serde_json::json!({
                    "slack_search_results": results,
                    "slack_search_completed": true
                }))
            }
            "send_message" => {
                let channel = input
                    .get("channel")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing channel parameter".to_string())
                    })?;

                let text = input.get("text").and_then(|v| v.as_str()).ok_or_else(|| {
                    WorkflowError::InvalidInput("Missing text parameter".to_string())
                })?;

                let thread_ts = input.get("thread_ts").and_then(|v| v.as_str());

                let result = self.send_message(channel, text, thread_ts).await?;

                Ok(serde_json::json!({
                    "send_result": result,
                    "operation_completed": true
                }))
            }
            "list_channels" => {
                let result = self.list_channels().await?;

                Ok(serde_json::json!({
                    "channels": result,
                    "operation_completed": true
                }))
            }
            "get_user_info" => {
                let user_id = input
                    .get("user_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing user_id parameter".to_string())
                    })?;

                let result = self.get_user_info(user_id).await?;

                Ok(serde_json::json!({
                    "user": result,
                    "operation_completed": true
                }))
            }
            _ => Err(WorkflowError::InvalidInput(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }

    fn name(&self) -> &str {
        "SlackServiceNode"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_search_messages() {
        let mock_server = MockServer::start().await;
        let api_service =
            SlackApiService::with_base_url("test-bot-token".to_string(), mock_server.uri());

        let node = SlackServiceNode {
            api_service: Arc::new(RwLock::new(api_service)),
        };

        Mock::given(method("GET"))
            .and(path("/search.messages"))
            .and(header("Authorization", "Bearer test-bot-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "messages": {
                    "matches": [{
                        "ts": "1234567890.123456",
                        "text": "Test message",
                        "user": "U1234567890",
                        "channel": "C1234567890"
                    }]
                }
            })))
            .mount(&mock_server)
            .await;

        let input = serde_json::json!({
            "operation": "search_messages",
            "query": "test query"
        });

        let result = node.execute(input, &TaskContext::default()).await.unwrap();
        let search_results = result.get("slack_search_results").unwrap();

        assert_eq!(search_results.get("query").unwrap(), "test query");
        assert_eq!(search_results.get("results_found").unwrap(), 1);
        assert!(search_results.get("real_api").unwrap().as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_send_message() {
        let mock_server = MockServer::start().await;
        let api_service =
            SlackApiService::with_base_url("test-bot-token".to_string(), mock_server.uri());

        let node = SlackServiceNode {
            api_service: Arc::new(RwLock::new(api_service)),
        };

        Mock::given(method("POST"))
            .and(path("/chat.postMessage"))
            .and(header("Authorization", "Bearer test-bot-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "ts": "1234567890.123456",
                "channel": "C1234567890",
                "message": {
                    "ts": "1234567890.123456",
                    "user": "U1234567890",
                    "text": "Hello, world!",
                    "channel": "C1234567890"
                }
            })))
            .mount(&mock_server)
            .await;

        let input = serde_json::json!({
            "operation": "send_message",
            "channel": "C1234567890",
            "text": "Hello, world!"
        });

        let result = node.execute(input, &TaskContext::default()).await.unwrap();
        let send_result = result.get("send_result").unwrap();

        assert!(send_result.get("success").unwrap().as_bool().unwrap());
        assert_eq!(send_result.get("channel").unwrap(), "C1234567890");
    }
}
