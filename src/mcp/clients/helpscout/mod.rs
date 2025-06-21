use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::time::Duration;

use crate::mcp::clients::services::HelpScoutApiService;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::external_mcp::{
    AuthConfig, BaseExternalMCPClient, ExternalMCPClientNode, ExternalMCPConfig, RetryConfig,
};
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::protocol::{CallToolResult, ToolContent, ToolDefinition};
use crate::mcp::transport::{HttpPoolConfig, ReconnectConfig, TransportType};

/// Configuration specific to HelpScout MCP client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelpscoutClientConfig {
    /// Base URL for the HelpScout MCP server
    pub server_url: String,

    /// API key for HelpScout authentication
    pub api_key: Option<String>,

    /// Transport type to use for connection
    pub transport: TransportType,

    /// Retry configuration
    pub retry_config: Option<RetryConfig>,
}

impl Default for HelpscoutClientConfig {
    fn default() -> Self {
        Self {
            server_url: env::var("HELPSCOUT_MCP_URL")
                .unwrap_or_else(|_| "http://localhost:8001".to_string()),
            api_key: env::var("HELPSCOUT_API_KEY").ok(),
            transport: TransportType::Http {
                base_url: env::var("HELPSCOUT_MCP_URL")
                    .unwrap_or_else(|_| "http://localhost:8001".to_string()),
                pool_config: HttpPoolConfig::default(),
            },
            retry_config: None,
        }
    }
}

/// HelpScout client node for connecting to external HelpScout MCP servers
#[derive(Debug)]
pub struct HelpscoutClientNode {
    base_client: BaseExternalMCPClient,
    helpscout_config: HelpscoutClientConfig,
    api_service: Option<HelpScoutApiService>,
}

impl HelpscoutClientNode {
    /// Create a new HelpscoutClientNode with the given configuration
    pub fn new(config: HelpscoutClientConfig) -> Self {
        let mut auth = None;
        if let Some(ref api_key) = config.api_key {
            let mut headers = HashMap::new();
            headers.insert("X-API-Key".to_string(), api_key.clone());
            auth = Some(AuthConfig {
                api_key: Some(api_key.clone()),
                token: Some(api_key.clone()),
                headers: Some(headers),
            });
        }

        let external_config = ExternalMCPConfig {
            service_name: "helpscout".to_string(),
            transport: config.transport.clone(),
            auth,
            retry_config: config.retry_config.clone().unwrap_or_default(),
        };

        let api_service = config
            .api_key
            .as_ref()
            .map(|key| HelpScoutApiService::new(key.clone()));

        Self {
            base_client: BaseExternalMCPClient::new(external_config),
            helpscout_config: config,
            api_service,
        }
    }

    /// Create a new HelpscoutClientNode with default configuration
    pub fn with_defaults() -> Self {
        Self::new(HelpscoutClientConfig::default())
    }

    /// Create a new HelpscoutClientNode with HTTP transport
    pub fn with_http_transport(base_url: String, api_key: Option<String>) -> Self {
        let config = HelpscoutClientConfig {
            server_url: base_url.clone(),
            api_key,
            transport: TransportType::Http {
                base_url,
                pool_config: HttpPoolConfig::default(),
            },
            retry_config: None,
        };
        Self::new(config)
    }

    /// Create a new HelpscoutClientNode with WebSocket transport
    pub fn with_websocket_transport(url: String, api_key: Option<String>) -> Self {
        let config = HelpscoutClientConfig {
            server_url: url.clone(),
            api_key,
            transport: TransportType::WebSocket {
                url,
                heartbeat_interval: Some(Duration::from_secs(30)),
                reconnect_config: ReconnectConfig::default(),
            },
            retry_config: None,
        };
        Self::new(config)
    }

    /// Create a new HelpscoutClientNode with stdio transport
    pub fn with_stdio_transport(
        command: String,
        args: Vec<String>,
        api_key: Option<String>,
    ) -> Self {
        let config = HelpscoutClientConfig {
            server_url: format!("stdio://{}", command),
            api_key,
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

    /// Search for articles in HelpScout knowledge base
    pub async fn search_articles(
        &mut self,
        keywords: &str,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<CallToolResult, WorkflowError> {
        // Use real API if available
        if let Some(ref api_service) = self.api_service {
            let search_result = api_service
                .search_articles(keywords, None, page, per_page)
                .await?;

            // Convert API response to CallToolResult format
            let content = vec![ToolContent::Text {
                text: serde_json::to_string_pretty(&search_result)
                    .unwrap_or_else(|_| "Failed to serialize search results".to_string()),
            }];

            return Ok(CallToolResult {
                content,
                is_error: Some(false),
            });
        }

        // Fall back to MCP server
        let mut args = HashMap::new();
        args.insert(
            "keywords".to_string(),
            serde_json::Value::String(keywords.to_string()),
        );

        if let Some(page) = page {
            args.insert("page".to_string(), serde_json::Value::Number(page.into()));
        }

        if let Some(per_page) = per_page {
            args.insert(
                "per_page".to_string(),
                serde_json::Value::Number(per_page.into()),
            );
        }

        self.execute_tool("search_articles", Some(args)).await
    }

    /// Get a specific article by ID
    pub async fn get_article(&mut self, article_id: &str) -> Result<CallToolResult, WorkflowError> {
        // Use real API if available
        if let Some(ref api_service) = self.api_service {
            let article = api_service.get_article(article_id).await?;

            // Convert API response to CallToolResult format
            let content = vec![ToolContent::Text {
                text: serde_json::to_string_pretty(&article)
                    .unwrap_or_else(|_| "Failed to serialize article".to_string()),
            }];

            return Ok(CallToolResult {
                content,
                is_error: Some(false),
            });
        }

        // Fall back to MCP server
        let mut args = HashMap::new();
        args.insert(
            "article_id".to_string(),
            serde_json::Value::String(article_id.to_string()),
        );

        self.execute_tool("get_article", Some(args)).await
    }

    /// List all articles
    pub async fn list_articles(
        &mut self,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();

        if let Some(page) = page {
            args.insert("page".to_string(), serde_json::Value::Number(page.into()));
        }

        if let Some(per_page) = per_page {
            args.insert(
                "per_page".to_string(),
                serde_json::Value::Number(per_page.into()),
            );
        }

        self.execute_tool("list_articles", Some(args)).await
    }

    /// List all collections
    pub async fn list_collections(
        &mut self,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();

        if let Some(page) = page {
            args.insert("page".to_string(), serde_json::Value::Number(page.into()));
        }

        if let Some(per_page) = per_page {
            args.insert(
                "per_page".to_string(),
                serde_json::Value::Number(per_page.into()),
            );
        }

        self.execute_tool("list_collections", Some(args)).await
    }

    /// Get a specific collection by ID
    pub async fn get_collection(
        &mut self,
        collection_id: &str,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "collection_id".to_string(),
            serde_json::Value::String(collection_id.to_string()),
        );

        self.execute_tool("get_collection", Some(args)).await
    }

    /// Create a new article
    pub async fn create_article(
        &mut self,
        title: &str,
        content: &str,
        collection_id: &str,
        tags: Option<Vec<String>>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "title".to_string(),
            serde_json::Value::String(title.to_string()),
        );
        args.insert(
            "content".to_string(),
            serde_json::Value::String(content.to_string()),
        );
        args.insert(
            "collection_id".to_string(),
            serde_json::Value::String(collection_id.to_string()),
        );

        if let Some(tags) = tags {
            args.insert(
                "tags".to_string(),
                serde_json::Value::Array(
                    tags.into_iter()
                        .map(|tag| serde_json::Value::String(tag))
                        .collect(),
                ),
            );
        }

        self.execute_tool("create_article", Some(args)).await
    }

    /// Update an existing article
    pub async fn update_article(
        &mut self,
        article_id: &str,
        title: Option<&str>,
        content: Option<&str>,
        tags: Option<Vec<String>>,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "article_id".to_string(),
            serde_json::Value::String(article_id.to_string()),
        );

        if let Some(title) = title {
            args.insert(
                "title".to_string(),
                serde_json::Value::String(title.to_string()),
            );
        }

        if let Some(content) = content {
            args.insert(
                "content".to_string(),
                serde_json::Value::String(content.to_string()),
            );
        }

        if let Some(tags) = tags {
            args.insert(
                "tags".to_string(),
                serde_json::Value::Array(
                    tags.into_iter()
                        .map(|tag| serde_json::Value::String(tag))
                        .collect(),
                ),
            );
        }

        self.execute_tool("update_article", Some(args)).await
    }

    /// Delete an article
    pub async fn delete_article(
        &mut self,
        article_id: &str,
    ) -> Result<CallToolResult, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "article_id".to_string(),
            serde_json::Value::String(article_id.to_string()),
        );

        self.execute_tool("delete_article", Some(args)).await
    }

    /// Get HelpScout configuration
    pub fn get_helpscout_config(&self) -> &HelpscoutClientConfig {
        &self.helpscout_config
    }
}

#[async_trait]
impl ExternalMCPClientNode for HelpscoutClientNode {
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
impl Node for HelpscoutClientNode {
    async fn execute(&self, input: Value, context: &TaskContext) -> Result<Value, WorkflowError> {
        // Pass through to base client
        self.base_client.execute(input, context).await
    }

    fn name(&self) -> &str {
        "HelpscoutClientNode"
    }
}

#[cfg(test)]
mod tests;
