use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::mcp::clients::services::NotionApiService;
use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::external_mcp::{
    AuthConfig, BaseExternalMCPClient, ExternalMCPClientNode, ExternalMCPConfig, RetryConfig,
};
use crate::mcp::core::nodes::Node;
use crate::mcp::core::task::TaskContext;
use crate::mcp::protocol::{CallToolResult, ToolDefinition};

/// Notion-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotionConfig {
    /// Base configuration for external MCP connection
    pub base_config: ExternalMCPConfig,

    /// Notion workspace ID (optional)
    pub workspace_id: Option<String>,

    /// Default database ID for operations (optional)
    pub default_database_id: Option<String>,
}

impl NotionConfig {
    /// Create a new Notion configuration with HTTP transport
    pub fn new_http(base_url: String, api_key: Option<String>) -> Self {
        let auth = api_key.map(|key| AuthConfig {
            api_key: Some(key.clone()),
            token: Some(key),
            headers: None,
        });

        Self {
            base_config: ExternalMCPConfig {
                service_name: "notion".to_string(),
                transport: crate::mcp::transport::TransportType::Http {
                    base_url,
                    pool_config: crate::mcp::transport::HttpPoolConfig::default(),
                },
                auth,
                retry_config: RetryConfig::default(),
            },
            workspace_id: None,
            default_database_id: None,
        }
    }

    /// Create a new Notion configuration with WebSocket transport
    pub fn new_websocket(url: String) -> Self {
        Self {
            base_config: ExternalMCPConfig {
                service_name: "notion".to_string(),
                transport: crate::mcp::transport::TransportType::WebSocket {
                    url,
                    heartbeat_interval: Some(std::time::Duration::from_secs(30)),
                    reconnect_config: crate::mcp::transport::ReconnectConfig::default(),
                },
                auth: None,
                retry_config: RetryConfig::default(),
            },
            workspace_id: None,
            default_database_id: None,
        }
    }

    /// Create a new Notion configuration with stdio transport
    pub fn new_stdio(command: String, args: Vec<String>) -> Self {
        Self {
            base_config: ExternalMCPConfig {
                service_name: "notion".to_string(),
                transport: crate::mcp::transport::TransportType::Stdio {
                    command,
                    args,
                    auto_restart: true,
                    max_restarts: 3,
                },
                auth: None,
                retry_config: RetryConfig::default(),
            },
            workspace_id: None,
            default_database_id: None,
        }
    }
}

/// Notion client node for interacting with Notion via MCP
#[derive(Debug)]
pub struct NotionClientNode {
    pub config: NotionConfig,
    base_client: BaseExternalMCPClient,
    api_service: Option<NotionApiService>,
}

impl NotionClientNode {
    pub fn new(config: NotionConfig) -> Self {
        let base_client = BaseExternalMCPClient::new(config.base_config.clone());

        // Extract API key from auth config
        let api_service = config
            .base_config
            .auth
            .as_ref()
            .and_then(|auth| auth.api_key.clone().or(auth.token.clone()))
            .map(|key| NotionApiService::new(key));

        Self {
            config,
            base_client,
            api_service,
        }
    }

    /// Search for pages in Notion
    pub async fn search_pages(
        &mut self,
        query: &str,
        limit: Option<u32>,
    ) -> Result<Value, WorkflowError> {
        // Use real API if available
        if let Some(ref api_service) = self.api_service {
            let search_result = api_service
                .search(query, Some("page"), None, None, limit)
                .await?;

            return Ok(serde_json::to_value(search_result)?);
        }

        // Fall back to MCP server
        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String(query.to_string()));
        if let Some(limit) = limit {
            args.insert("limit".to_string(), Value::Number(limit.into()));
        }

        let result = self
            .base_client
            .execute_tool("search_pages", Some(args))
            .await?;
        Ok(serde_json::to_value(result.content)?)
    }

    /// Create a new page in Notion
    pub async fn create_page(
        &mut self,
        title: &str,
        content: &str,
        parent_id: Option<&str>,
    ) -> Result<Value, WorkflowError> {
        let mut args = HashMap::new();
        args.insert("title".to_string(), Value::String(title.to_string()));
        args.insert("content".to_string(), Value::String(content.to_string()));

        if let Some(parent) = parent_id {
            args.insert("parent_id".to_string(), Value::String(parent.to_string()));
        } else if let Some(ref db_id) = self.config.default_database_id {
            args.insert("parent_id".to_string(), Value::String(db_id.clone()));
        }

        let result = self
            .base_client
            .execute_tool("create_page", Some(args))
            .await?;
        Ok(serde_json::to_value(result.content)?)
    }

    /// Create a research documentation page with structured content
    pub async fn create_research_page(
        &mut self,
        title: &str,
        summary: &str,
        key_points: &[String],
        sources: &[Value],
        parent_id: Option<&str>,
        properties: Option<HashMap<String, Value>>,
    ) -> Result<Value, WorkflowError> {
        // Build structured content for research documentation
        let mut content_parts = Vec::new();

        // Add title
        content_parts.push(format!("# {}\n", title));

        // Add summary section
        content_parts.push("## Summary\n".to_string());
        content_parts.push(format!("{}\n\n", summary));

        // Add key insights
        if !key_points.is_empty() {
            content_parts.push("## Key Insights\n".to_string());
            for point in key_points {
                content_parts.push(format!("- {}\n", point));
            }
            content_parts.push("\n".to_string());
        }

        // Add sources section
        if !sources.is_empty() {
            content_parts.push("## Sources\n".to_string());
            for (i, source) in sources.iter().enumerate() {
                match source {
                    Value::String(url) => {
                        content_parts.push(format!("{}. {}\n", i + 1, url));
                    }
                    Value::Object(obj) => {
                        let Some(title) = obj.get("title") else {
                            continue;
                        };
                        let Some(url) = obj.get("url") else {
                            continue;
                        };

                        content_parts.push(format!(
                            "{}. [{}]({})\n",
                            i + 1,
                            title.as_str().unwrap_or("Unknown Title"),
                            url.as_str().unwrap_or("No URL")
                        ));
                    }
                    _ => {
                        content_parts.push(format!(
                            "{}. {}\n",
                            i + 1,
                            serde_json::to_string(source).unwrap_or_default()
                        ));
                    }
                }
            }
            content_parts.push("\n".to_string());
        }

        // Add timestamp
        content_parts.push(format!(
            "---\n*Generated on: {}*\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        let content = content_parts.join("");

        // Create the page with enhanced arguments
        let mut args = HashMap::new();
        args.insert("title".to_string(), Value::String(title.to_string()));
        args.insert("content".to_string(), Value::String(content));

        if let Some(parent) = parent_id {
            args.insert("parent_id".to_string(), Value::String(parent.to_string()));
        } else if let Some(ref db_id) = self.config.default_database_id {
            args.insert("parent_id".to_string(), Value::String(db_id.clone()));
        }

        // Add properties if provided
        if let Some(props) = properties {
            args.insert(
                "properties".to_string(),
                Value::Object(props.into_iter().collect()),
            );
        }

        let result = self
            .base_client
            .execute_tool("create_page", Some(args))
            .await?;
        Ok(serde_json::to_value(result.content)?)
    }

    /// Update an existing page in Notion
    pub async fn update_page(
        &mut self,
        page_id: &str,
        updates: HashMap<String, Value>,
    ) -> Result<Value, WorkflowError> {
        let mut args = HashMap::new();
        args.insert("page_id".to_string(), Value::String(page_id.to_string()));
        args.insert(
            "updates".to_string(),
            Value::Object(updates.into_iter().collect()),
        );

        let result = self
            .base_client
            .execute_tool("update_page", Some(args))
            .await?;
        Ok(serde_json::to_value(result.content)?)
    }

    /// Get a page by ID
    pub async fn get_page(&mut self, page_id: &str) -> Result<Value, WorkflowError> {
        let mut args = HashMap::new();
        args.insert("page_id".to_string(), Value::String(page_id.to_string()));

        let result = self
            .base_client
            .execute_tool("get_page", Some(args))
            .await?;
        Ok(serde_json::to_value(result.content)?)
    }

    /// List databases in the workspace
    pub async fn list_databases(&mut self) -> Result<Value, WorkflowError> {
        let result = self
            .base_client
            .execute_tool("list_databases", None)
            .await?;
        Ok(serde_json::to_value(result.content)?)
    }

    /// Query a database
    pub async fn query_database(
        &mut self,
        database_id: &str,
        filter: Option<Value>,
        sorts: Option<Value>,
        limit: Option<u32>,
    ) -> Result<Value, WorkflowError> {
        let mut args = HashMap::new();
        args.insert(
            "database_id".to_string(),
            Value::String(database_id.to_string()),
        );

        if let Some(filter) = filter {
            args.insert("filter".to_string(), filter);
        }
        if let Some(sorts) = sorts {
            args.insert("sorts".to_string(), sorts);
        }
        if let Some(limit) = limit {
            args.insert("limit".to_string(), Value::Number(limit.into()));
        }

        let result = self
            .base_client
            .execute_tool("query_database", Some(args))
            .await?;
        Ok(serde_json::to_value(result.content)?)
    }

    /// Helper method to parse Notion-specific errors
    pub fn parse_notion_error(&self, error: &WorkflowError) -> WorkflowError {
        match error {
            WorkflowError::MCPError { message } => {
                // Check for common Notion API errors
                if message.contains("unauthorized") || message.contains("401") {
                    WorkflowError::MCPError {
                        message: format!("Notion authentication failed: {}", message),
                    }
                } else if message.contains("not_found") || message.contains("404") {
                    WorkflowError::MCPError {
                        message: format!("Notion resource not found: {}", message),
                    }
                } else if message.contains("rate_limit") || message.contains("429") {
                    WorkflowError::MCPError {
                        message: format!("Notion rate limit exceeded: {}", message),
                    }
                } else {
                    WorkflowError::MCPError {
                        message: message.clone(),
                    }
                }
            }
            _ => WorkflowError::MCPError {
                message: format!("Notion error: {:?}", error),
            },
        }
    }
}

#[async_trait]
impl Node for NotionClientNode {
    async fn execute(&self, input: Value, context: &TaskContext) -> Result<Value, WorkflowError> {
        // Pass through to base client
        self.base_client.execute(input, context).await
    }

    fn name(&self) -> &str {
        "NotionClientNode"
    }
}

#[async_trait]
impl ExternalMCPClientNode for NotionClientNode {
    fn get_config(&self) -> &ExternalMCPConfig {
        &self.config.base_config
    }

    async fn connect(&mut self) -> Result<(), WorkflowError> {
        self.base_client
            .connect()
            .await
            .map_err(|e| self.parse_notion_error(&e))
    }

    async fn execute_tool(
        &mut self,
        tool_name: &str,
        arguments: Option<HashMap<String, Value>>,
    ) -> Result<CallToolResult, WorkflowError> {
        self.base_client
            .execute_tool(tool_name, arguments)
            .await
            .map_err(|e| self.parse_notion_error(&e))
    }

    async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, WorkflowError> {
        self.base_client
            .list_tools()
            .await
            .map_err(|e| self.parse_notion_error(&e))
    }

    async fn disconnect(&mut self) -> Result<(), WorkflowError> {
        self.base_client.disconnect().await
    }

    fn is_connected(&self) -> bool {
        self.base_client.is_connected()
    }
}

/// Builder for NotionClientNode
pub struct NotionClientBuilder {
    config: NotionConfig,
}

impl NotionClientBuilder {
    pub fn new_http(base_url: String) -> Self {
        Self {
            config: NotionConfig::new_http(base_url, None),
        }
    }

    pub fn new_websocket(url: String) -> Self {
        Self {
            config: NotionConfig::new_websocket(url),
        }
    }

    pub fn new_stdio(command: String, args: Vec<String>) -> Self {
        Self {
            config: NotionConfig::new_stdio(command, args),
        }
    }

    pub fn with_api_key(mut self, api_key: String) -> Self {
        if self.config.base_config.auth.is_none() {
            self.config.base_config.auth = Some(AuthConfig {
                api_key: Some(api_key.clone()),
                token: Some(api_key),
                headers: None,
            });
        } else if let Some(ref mut auth) = self.config.base_config.auth {
            auth.token = Some(api_key);
        }
        self
    }

    pub fn with_workspace_id(mut self, workspace_id: String) -> Self {
        self.config.workspace_id = Some(workspace_id);
        self
    }

    pub fn with_default_database_id(mut self, database_id: String) -> Self {
        self.config.default_database_id = Some(database_id);
        self
    }

    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.config.base_config.retry_config = retry_config;
        self
    }

    pub fn build(self) -> NotionClientNode {
        NotionClientNode::new(self.config)
    }
}

// #[cfg(test)]
// mod tests;
