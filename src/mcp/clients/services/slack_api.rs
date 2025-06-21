use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::mcp::core::error::WorkflowError;

/// Slack API service for real API integration
///
/// This service provides actual HTTP API calls to Slack's Web API.
/// Documentation: https://api.slack.com/web
#[derive(Debug, Clone)]
pub struct SlackApiService {
    client: Client,
    bot_token: String,
    base_url: String,
}

/// Represents a Slack channel
#[derive(Debug, Serialize, Deserialize)]
pub struct Channel {
    pub id: String,
    pub name: String,
    pub is_channel: bool,
    pub is_group: bool,
    pub is_im: bool,
    pub is_mpim: bool,
    pub is_private: bool,
    pub created: i64,
    pub is_archived: bool,
    pub is_general: bool,
    pub name_normalized: String,
    pub is_shared: bool,
    pub is_org_shared: bool,
    pub is_member: Option<bool>,
    pub num_members: Option<i32>,
    pub topic: Option<ChannelTopic>,
    pub purpose: Option<ChannelPurpose>,
}

/// Channel topic
#[derive(Debug, Serialize, Deserialize)]
pub struct ChannelTopic {
    pub value: String,
    pub creator: String,
    pub last_set: i64,
}

/// Channel purpose
#[derive(Debug, Serialize, Deserialize)]
pub struct ChannelPurpose {
    pub value: String,
    pub creator: String,
    pub last_set: i64,
}

/// Represents a Slack user
#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub team_id: String,
    pub name: String,
    pub deleted: bool,
    pub real_name: String,
    pub tz: Option<String>,
    pub tz_label: Option<String>,
    pub is_admin: Option<bool>,
    pub is_owner: Option<bool>,
    pub is_primary_owner: Option<bool>,
    pub is_restricted: Option<bool>,
    pub is_ultra_restricted: Option<bool>,
    pub is_bot: bool,
    pub is_app_user: bool,
    pub profile: UserProfile,
}

/// User profile information
#[derive(Debug, Serialize, Deserialize)]
pub struct UserProfile {
    pub title: Option<String>,
    pub phone: Option<String>,
    pub skype: Option<String>,
    pub real_name: String,
    pub display_name: String,
    pub status_text: Option<String>,
    pub status_emoji: Option<String>,
    pub email: Option<String>,
}

/// Represents a Slack message
#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub ts: String,
    pub thread_ts: Option<String>,
    pub user: Option<String>,
    pub text: String,
    pub attachments: Option<Vec<Attachment>>,
    pub blocks: Option<Vec<serde_json::Value>>,
    pub channel: Option<String>,
}

/// Message attachment
#[derive(Debug, Serialize, Deserialize)]
pub struct Attachment {
    pub color: Option<String>,
    pub fallback: Option<String>,
    pub title: Option<String>,
    pub text: Option<String>,
    pub fields: Option<Vec<AttachmentField>>,
}

/// Attachment field
#[derive(Debug, Serialize, Deserialize)]
pub struct AttachmentField {
    pub title: String,
    pub value: String,
    pub short: bool,
}

/// API response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct SlackResponse<T> {
    pub ok: bool,
    pub error: Option<String>,
    #[serde(flatten)]
    pub data: Option<T>,
}

/// Channel list response
#[derive(Debug, Serialize, Deserialize)]
pub struct ChannelListResponse {
    pub channels: Vec<Channel>,
}

/// User info response
#[derive(Debug, Serialize, Deserialize)]
pub struct UserInfoResponse {
    pub user: User,
}

/// Message post response
#[derive(Debug, Serialize, Deserialize)]
pub struct PostMessageResponse {
    pub ts: String,
    pub channel: String,
    pub message: Message,
}

/// Channel history response
#[derive(Debug, Serialize, Deserialize)]
pub struct ChannelHistoryResponse {
    pub messages: Vec<Message>,
    pub has_more: bool,
}

impl SlackApiService {
    /// Create a new Slack API service
    pub fn new(bot_token: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            bot_token,
            base_url: "https://slack.com/api".to_string(),
        }
    }

    /// Create a new Slack API service with custom base URL (for testing)
    pub fn with_base_url(bot_token: String, base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            bot_token,
            base_url,
        }
    }

    /// Send a message to a channel
    pub async fn post_message(
        &self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
        blocks: Option<Vec<serde_json::Value>>,
    ) -> Result<PostMessageResponse, WorkflowError> {
        let url = format!("{}/chat.postMessage", self.base_url);

        let mut body = serde_json::json!({
            "channel": channel,
            "text": text
        });

        if let Some(ts) = thread_ts {
            body["thread_ts"] = serde_json::json!(ts);
        }

        if let Some(blocks) = blocks {
            body["blocks"] = serde_json::json!(blocks);
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!("Failed to post message: {}", e),
            })?;

        if response.status() == StatusCode::UNAUTHORIZED {
            return Err(WorkflowError::AuthenticationError {
                message: "Invalid Slack bot token".to_string(),
            });
        }

        let slack_response: SlackResponse<PostMessageResponse> =
            response.json().await.map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse response: {}", e))
            })?;

        if !slack_response.ok {
            return Err(WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!(
                    "Slack API error: {}",
                    slack_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string())
                ),
            });
        }

        slack_response
            .data
            .ok_or_else(|| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: "No data in successful response".to_string(),
            })
    }

    /// List channels in the workspace
    pub async fn list_channels(
        &self,
        exclude_archived: bool,
        types: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<Channel>, WorkflowError> {
        let url = format!("{}/conversations.list", self.base_url);

        let mut params = HashMap::new();
        params.insert("exclude_archived", exclude_archived.to_string());
        if let Some(types) = types {
            params.insert("types", types.to_string());
        }
        if let Some(limit) = limit {
            params.insert("limit", limit.to_string());
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!("Failed to list channels: {}", e),
            })?;

        if response.status() == StatusCode::UNAUTHORIZED {
            return Err(WorkflowError::AuthenticationError {
                message: "Invalid Slack bot token".to_string(),
            });
        }

        let slack_response: SlackResponse<ChannelListResponse> =
            response.json().await.map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse response: {}", e))
            })?;

        if !slack_response.ok {
            return Err(WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!(
                    "Slack API error: {}",
                    slack_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string())
                ),
            });
        }

        Ok(slack_response.data.map(|d| d.channels).unwrap_or_default())
    }

    /// Get user information
    pub async fn get_user_info(&self, user_id: &str) -> Result<User, WorkflowError> {
        let url = format!("{}/users.info", self.base_url);

        let mut params = HashMap::new();
        params.insert("user", user_id.to_string());

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!("Failed to get user info: {}", e),
            })?;

        let slack_response: SlackResponse<UserInfoResponse> =
            response.json().await.map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse response: {}", e))
            })?;

        if !slack_response.ok {
            if slack_response.error.as_deref() == Some("user_not_found") {
                return Err(WorkflowError::NotFound {
                    resource: format!("User with ID: {}", user_id),
                });
            }
            return Err(WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!(
                    "Slack API error: {}",
                    slack_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string())
                ),
            });
        }

        slack_response
            .data
            .map(|d| d.user)
            .ok_or_else(|| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: "No data in successful response".to_string(),
            })
    }

    /// Get channel information
    pub async fn get_channel_info(&self, channel_id: &str) -> Result<Channel, WorkflowError> {
        let url = format!("{}/conversations.info", self.base_url);

        let mut params = HashMap::new();
        params.insert("channel", channel_id.to_string());

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!("Failed to get channel info: {}", e),
            })?;

        let slack_response: SlackResponse<serde_json::Value> =
            response.json().await.map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse response: {}", e))
            })?;

        if !slack_response.ok {
            if slack_response.error.as_deref() == Some("channel_not_found") {
                return Err(WorkflowError::NotFound {
                    resource: format!("Channel with ID: {}", channel_id),
                });
            }
            return Err(WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!(
                    "Slack API error: {}",
                    slack_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string())
                ),
            });
        }

        slack_response
            .data
            .and_then(|d| d.get("channel").cloned())
            .ok_or_else(|| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: "No channel data in response".to_string(),
            })
            .and_then(|c| {
                serde_json::from_value(c).map_err(|e| {
                    WorkflowError::SerializationError(format!("Failed to parse channel: {}", e))
                })
            })
    }

    /// Get channel history
    pub async fn get_channel_history(
        &self,
        channel: &str,
        limit: Option<u32>,
        oldest: Option<&str>,
        latest: Option<&str>,
    ) -> Result<ChannelHistoryResponse, WorkflowError> {
        let url = format!("{}/conversations.history", self.base_url);

        let mut params = HashMap::new();
        params.insert("channel", channel.to_string());
        if let Some(limit) = limit {
            params.insert("limit", limit.to_string());
        }
        if let Some(oldest) = oldest {
            params.insert("oldest", oldest.to_string());
        }
        if let Some(latest) = latest {
            params.insert("latest", latest.to_string());
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!("Failed to get channel history: {}", e),
            })?;

        let slack_response: SlackResponse<ChannelHistoryResponse> =
            response.json().await.map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse response: {}", e))
            })?;

        if !slack_response.ok {
            return Err(WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!(
                    "Slack API error: {}",
                    slack_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string())
                ),
            });
        }

        slack_response
            .data
            .ok_or_else(|| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: "No data in successful response".to_string(),
            })
    }

    /// Search messages in the workspace
    pub async fn search_messages(
        &self,
        query: &str,
        count: Option<u32>,
        page: Option<u32>,
    ) -> Result<Vec<Message>, WorkflowError> {
        let url = format!("{}/search.messages", self.base_url);

        let mut params = HashMap::new();
        params.insert("query", query.to_string());
        if let Some(count) = count {
            params.insert("count", count.to_string());
        }
        if let Some(page) = page {
            params.insert("page", page.to_string());
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!("Failed to search messages: {}", e),
            })?;

        let slack_response: SlackResponse<serde_json::Value> =
            response.json().await.map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse response: {}", e))
            })?;

        if !slack_response.ok {
            return Err(WorkflowError::ExternalServiceError {
                service: "Slack".to_string(),
                message: format!(
                    "Slack API error: {}",
                    slack_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string())
                ),
            });
        }

        // Extract messages from search results
        Ok(slack_response
            .data
            .and_then(|d| d.get("messages").cloned())
            .and_then(|m| m.get("matches").cloned())
            .and_then(|matches| serde_json::from_value::<Vec<Message>>(matches).ok())
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_post_message() {
        let mock_server = MockServer::start().await;
        let api = SlackApiService::with_base_url("test-bot-token".to_string(), mock_server.uri());

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

        let result = api
            .post_message("C1234567890", "Hello, world!", None, None)
            .await
            .unwrap();
        assert_eq!(result.channel, "C1234567890");
        assert_eq!(result.message.text, "Hello, world!");
    }

    #[tokio::test]
    async fn test_list_channels() {
        let mock_server = MockServer::start().await;
        let api = SlackApiService::with_base_url("test-bot-token".to_string(), mock_server.uri());

        Mock::given(method("GET"))
            .and(path("/conversations.list"))
            .and(header("Authorization", "Bearer test-bot-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "channels": [{
                    "id": "C1234567890",
                    "name": "general",
                    "is_channel": true,
                    "is_group": false,
                    "is_im": false,
                    "is_mpim": false,
                    "is_private": false,
                    "created": 1449252889,
                    "is_archived": false,
                    "is_general": true,
                    "name_normalized": "general",
                    "is_shared": false,
                    "is_org_shared": false
                }]
            })))
            .mount(&mock_server)
            .await;

        let result = api.list_channels(true, None, None).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "general");
    }

    #[tokio::test]
    async fn test_authentication_error() {
        let mock_server = MockServer::start().await;
        let api = SlackApiService::with_base_url("invalid-token".to_string(), mock_server.uri());

        Mock::given(method("POST"))
            .and(path("/chat.postMessage"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock_server)
            .await;

        let result = api.post_message("C1234567890", "Hello", None, None).await;
        assert!(matches!(
            result.unwrap_err(),
            WorkflowError::AuthenticationError { .. }
        ));
    }
}
