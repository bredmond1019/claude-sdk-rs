use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;

use crate::mcp::clients::services::SlackApiService;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::external_mcp::{
    AuthConfig, BaseExternalMCPClient, ExternalMCPClientNode, ExternalMCPConfig, RetryConfig,
};
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::protocol::{CallToolResult, ToolContent, ToolDefinition};
use crate::mcp::transport::{HttpPoolConfig, TransportType};

/// Configuration specific to Slack MCP client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackClientConfig {
    /// Base URL for the Slack MCP server
    pub server_url: String,

    /// Bot token for Slack authentication
    pub bot_token: Option<String>,

    /// User token for Slack authentication (if needed for certain operations)
    pub user_token: Option<String>,

    /// Transport type to use for connection
    pub transport: TransportType,

    /// Retry configuration
    pub retry_config: Option<RetryConfig>,
}

impl Default for SlackClientConfig {
    fn default() -> Self {
        Self {
            server_url: env::var("SLACK_MCP_URL")
                .unwrap_or_else(|_| "http://localhost:8003".to_string()),
            bot_token: env::var("SLACK_BOT_TOKEN").ok(),
            user_token: env::var("SLACK_USER_TOKEN").ok(),
            transport: TransportType::Http {
                base_url: env::var("SLACK_MCP_URL")
                    .unwrap_or_else(|_| "http://localhost:8003".to_string()),
                pool_config: crate::mcp::transport::HttpPoolConfig::default(),
            },
            retry_config: None,
        }
    }
}

/// Slack client node for connecting to external Slack MCP servers
#[derive(Debug)]
pub struct SlackClientNode {
    base_client: BaseExternalMCPClient,
    slack_config: SlackClientConfig,
    api_service: Option<SlackApiService>,
}

impl SlackClientNode {
    /// Create a new SlackClientNode with the given configuration
    pub fn new(config: SlackClientConfig) -> Self {
        let mut auth = None;
        let mut headers = HashMap::new();

        // Add authorization headers based on available tokens
        if let Some(ref bot_token) = config.bot_token {
            headers.insert("Authorization".to_string(), format!("Bearer {}", bot_token));
        }

        if let Some(ref user_token) = config.user_token {
            headers.insert("X-Slack-User-Token".to_string(), user_token.clone());
        }

        if !headers.is_empty() {
            auth = Some(AuthConfig {
                api_key: config.bot_token.clone(),
                token: config.bot_token.clone(),
                headers: Some(headers),
            });
        }

        let external_config = ExternalMCPConfig {
            service_name: "slack".to_string(),
            transport: config.transport.clone(),
            auth,
            retry_config: config.retry_config.clone().unwrap_or_default(),
        };

        let api_service = config
            .bot_token
            .as_ref()
            .map(|token| SlackApiService::new(token.clone()));

        Self {
            base_client: BaseExternalMCPClient::new(external_config),
            slack_config: config,
            api_service,
        }
    }

    /// Create a new SlackClientNode with default configuration
    pub fn with_defaults() -> Self {
        Self::new(SlackClientConfig::default())
    }

    /// Create a new SlackClientNode with HTTP transport
    pub fn with_http_transport(
        base_url: String,
        bot_token: Option<String>,
        user_token: Option<String>,
    ) -> Self {
        let config = SlackClientConfig {
            server_url: base_url.clone(),
            bot_token,
            user_token,
            transport: TransportType::Http {
                base_url,
                pool_config: HttpPoolConfig::default(),
            },
            retry_config: None,
        };
        Self::new(config)
    }

    /// Create a new SlackClientNode with WebSocket transport
    pub fn with_websocket_transport(
        url: String,
        bot_token: Option<String>,
        user_token: Option<String>,
    ) -> Self {
        let config = SlackClientConfig {
            server_url: url.clone(),
            bot_token,
            user_token,
            transport: TransportType::WebSocket {
                url,
                heartbeat_interval: Some(std::time::Duration::from_secs(30)),
                reconnect_config: crate::mcp::transport::ReconnectConfig::default(),
            },
            retry_config: None,
        };
        Self::new(config)
    }

    /// Create a new SlackClientNode with stdio transport
    pub fn with_stdio_transport(
        command: String,
        args: Vec<String>,
        bot_token: Option<String>,
        user_token: Option<String>,
    ) -> Self {
        let config = SlackClientConfig {
            server_url: format!("stdio://{}", command),
            bot_token,
            user_token,
            transport: TransportType::Stdio {
                command,
                args,
                auto_restart: true,
                max_restarts: 3,
            },
            retry_config: None,
        };
        Self::new(config)
    }

    /// Send a message to a Slack channel
    pub async fn send_message(
        &mut self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> Result<CallToolResult, WorkflowError> {
        // Use real API if available
        if let Some(ref api_service) = self.api_service {
            let result = api_service
                .post_message(channel, text, thread_ts, None)
                .await?;

            // Convert API response to CallToolResult format
            let content = vec![ToolContent::Text {
                text: serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| "Failed to serialize message result".to_string()),
            }];

            return Ok(CallToolResult {
                content,
                is_error: Some(false),
            });
        }

        // Fall back to MCP server
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel.to_string()),
        );
        args.insert(
            "text".to_string(),
            serde_json::Value::String(text.to_string()),
        );

        if let Some(thread_ts) = thread_ts {
            args.insert(
                "thread_ts".to_string(),
                serde_json::Value::String(thread_ts.to_string()),
            );
        }

        self.execute_tool("send_message", Some(args)).await
    }

    /// List channels in the workspace
    pub async fn list_channels(
        &mut self,
        exclude_archived: Option<bool>,
        types: Option<Vec<String>>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();

        if let Some(exclude_archived) = exclude_archived {
            args.insert(
                "exclude_archived".to_string(),
                serde_json::Value::Bool(exclude_archived),
            );
        }

        if let Some(types) = types {
            args.insert(
                "types".to_string(),
                serde_json::Value::Array(
                    types
                        .into_iter()
                        .map(|t| serde_json::Value::String(t))
                        .collect(),
                ),
            );
        }

        self.execute_tool("list_channels", Some(args)).await
    }

    /// Get information about a user
    pub async fn get_user_info(&mut self, user_id: &str) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "user".to_string(),
            serde_json::Value::String(user_id.to_string()),
        );

        self.execute_tool("get_user_info", Some(args)).await
    }

    /// Get information about a channel
    pub async fn get_channel_info(
        &mut self,
        channel_id: &str,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel_id.to_string()),
        );

        self.execute_tool("get_channel_info", Some(args)).await
    }

    /// Get message history from a channel
    pub async fn get_channel_history(
        &mut self,
        channel: &str,
        limit: Option<u32>,
        oldest: Option<&str>,
        latest: Option<&str>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel.to_string()),
        );

        if let Some(limit) = limit {
            args.insert("limit".to_string(), serde_json::Value::Number(limit.into()));
        }

        if let Some(oldest) = oldest {
            args.insert(
                "oldest".to_string(),
                serde_json::Value::String(oldest.to_string()),
            );
        }

        if let Some(latest) = latest {
            args.insert(
                "latest".to_string(),
                serde_json::Value::String(latest.to_string()),
            );
        }

        self.execute_tool("get_channel_history", Some(args)).await
    }

    /// Update a message
    pub async fn update_message(
        &mut self,
        channel: &str,
        ts: &str,
        text: &str,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel.to_string()),
        );
        args.insert("ts".to_string(), serde_json::Value::String(ts.to_string()));
        args.insert(
            "text".to_string(),
            serde_json::Value::String(text.to_string()),
        );

        self.execute_tool("update_message", Some(args)).await
    }

    /// Delete a message
    pub async fn delete_message(
        &mut self,
        channel: &str,
        ts: &str,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel.to_string()),
        );
        args.insert("ts".to_string(), serde_json::Value::String(ts.to_string()));

        self.execute_tool("delete_message", Some(args)).await
    }

    /// Add a reaction to a message
    pub async fn add_reaction(
        &mut self,
        channel: &str,
        timestamp: &str,
        name: &str,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel.to_string()),
        );
        args.insert(
            "timestamp".to_string(),
            serde_json::Value::String(timestamp.to_string()),
        );
        args.insert(
            "name".to_string(),
            serde_json::Value::String(name.to_string()),
        );

        self.execute_tool("add_reaction", Some(args)).await
    }

    /// Remove a reaction from a message
    pub async fn remove_reaction(
        &mut self,
        channel: &str,
        timestamp: &str,
        name: &str,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel.to_string()),
        );
        args.insert(
            "timestamp".to_string(),
            serde_json::Value::String(timestamp.to_string()),
        );
        args.insert(
            "name".to_string(),
            serde_json::Value::String(name.to_string()),
        );

        self.execute_tool("remove_reaction", Some(args)).await
    }

    /// Search for messages
    pub async fn search_messages(
        &mut self,
        query: &str,
        sort: Option<&str>,
        sort_dir: Option<&str>,
        count: Option<u32>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            serde_json::Value::String(query.to_string()),
        );

        if let Some(sort) = sort {
            args.insert(
                "sort".to_string(),
                serde_json::Value::String(sort.to_string()),
            );
        }

        if let Some(sort_dir) = sort_dir {
            args.insert(
                "sort_dir".to_string(),
                serde_json::Value::String(sort_dir.to_string()),
            );
        }

        if let Some(count) = count {
            args.insert("count".to_string(), serde_json::Value::Number(count.into()));
        }

        self.execute_tool("search_messages", Some(args)).await
    }

    /// Create a new channel
    pub async fn create_channel(
        &mut self,
        name: &str,
        is_private: Option<bool>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "name".to_string(),
            serde_json::Value::String(name.to_string()),
        );

        if let Some(is_private) = is_private {
            args.insert(
                "is_private".to_string(),
                serde_json::Value::Bool(is_private),
            );
        }

        self.execute_tool("create_channel", Some(args)).await
    }

    /// Invite users to a channel
    pub async fn invite_to_channel(
        &mut self,
        channel: &str,
        users: Vec<String>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "channel".to_string(),
            serde_json::Value::String(channel.to_string()),
        );
        args.insert(
            "users".to_string(),
            serde_json::Value::Array(
                users
                    .into_iter()
                    .map(|u| serde_json::Value::String(u))
                    .collect(),
            ),
        );

        self.execute_tool("invite_to_channel", Some(args)).await
    }

    /// Get Slack configuration
    pub fn get_slack_config(&self) -> &SlackClientConfig {
        &self.slack_config
    }
}

#[async_trait]
impl ExternalMCPClientNode for SlackClientNode {
    fn get_config(&self) -> &ExternalMCPConfig {
        self.base_client.get_config()
    }

    async fn connect(&mut self) -> Result<(), WorkflowError> {
        self.base_client.connect().await
    }

    async fn execute_tool(
        &mut self,
        tool_name: &str,
        arguments: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<CallToolResult, WorkflowError> {
        self.base_client.execute_tool(tool_name, arguments).await
    }

    async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, WorkflowError> {
        self.base_client.list_tools().await
    }

    async fn disconnect(&mut self) -> Result<(), WorkflowError> {
        self.base_client.disconnect().await
    }

    fn is_connected(&self) -> bool {
        self.base_client.is_connected()
    }
}

#[async_trait]
impl Node for SlackClientNode {
    async fn execute(&self, input: Value, context: &TaskContext) -> Result<Value, WorkflowError> {
        // Pass through to base client
        self.base_client.execute(input, context).await
    }

    fn name(&self) -> &str {
        "SlackClientNode"
    }
}

// #[cfg(test)]
// mod tests;
