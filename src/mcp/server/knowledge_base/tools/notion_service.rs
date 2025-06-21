/// Notion Service Integration - Provides real API access for Notion operations
///
/// This service acts as a bridge between the workflow nodes and the actual Notion API,
/// handling authentication, error handling, and response formatting.
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::mcp::clients::services::NotionApiService;
use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Service node that provides real Notion API integration
#[derive(Debug)]
pub struct NotionServiceNode {
    api_service: Arc<RwLock<NotionApiService>>,
}

impl NotionServiceNode {
    /// Create a new Notion service node with API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_service: Arc::new(RwLock::new(NotionApiService::new(api_key))),
        }
    }

    /// Search pages using the real Notion API
    pub async fn search_pages(&self, query: &str) -> Result<Value, WorkflowError> {
        let api = self.api_service.read().await;
        let search_result = api
            .search(query, Some("page"), None, None, Some(20))
            .await?;

        // Convert to workflow-friendly format
        let pages = search_result
            .results
            .into_iter()
            .filter_map(|result| {
                match result {
                    crate::mcp::clients::services::notion_api::SearchResult::Page(page) => {
                        // Extract title from properties
                        let title = Self::extract_page_title(&page.properties);

                        Some(serde_json::json!({
                            "id": page.id,
                            "title": title,
                            "url": page.url,
                            "created_time": page.created_time,
                            "last_edited_time": page.last_edited_time
                        }))
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "source": "notion",
            "query": query,
            "results_found": pages.len(),
            "pages": pages,
            "real_api": true
        }))
    }

    /// Search databases using the real Notion API
    pub async fn search_databases(&self, query: &str) -> Result<Value, WorkflowError> {
        let api = self.api_service.read().await;
        let search_result = api
            .search(query, Some("database"), None, None, Some(20))
            .await?;

        // Convert to workflow-friendly format
        let databases = search_result
            .results
            .into_iter()
            .filter_map(|result| match result {
                crate::mcp::clients::services::notion_api::SearchResult::Database(db) => {
                    let title = db
                        .title
                        .first()
                        .and_then(|rt| rt.plain_text.clone())
                        .unwrap_or_else(|| "Untitled".to_string());

                    Some(serde_json::json!({
                        "id": db.id,
                        "title": title,
                        "url": db.url,
                        "created_time": db.created_time,
                        "last_edited_time": db.last_edited_time
                    }))
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "source": "notion",
            "query": query,
            "results_found": databases.len(),
            "databases": databases,
            "real_api": true
        }))
    }

    /// Helper to extract page title from properties
    fn extract_page_title(properties: &Value) -> String {
        // Try common property names for title
        let title_keys = ["Name", "Title", "title", "name"];

        for key in &title_keys {
            if let Some(prop) = properties.get(key) {
                if let Some(title_array) = prop.get("title").and_then(|t| t.as_array()) {
                    if let Some(first) = title_array.first() {
                        if let Some(text) = first.get("plain_text").and_then(|t| t.as_str()) {
                            return text.to_string();
                        }
                    }
                }
            }
        }

        "Untitled".to_string()
    }
}

#[async_trait]
impl Node for NotionServiceNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract operation type
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("search");

        match operation {
            "search" | "search_pages" => {
                let query = input
                    .get("query")
                    .or_else(|| input.get("user_query"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing query parameter".to_string())
                    })?;

                let results = self.search_pages(query).await?;

                Ok(serde_json::json!({
                    "notion_search_results": results,
                    "notion_search_completed": true
                }))
            }
            "search_databases" => {
                let query = input.get("query").and_then(|v| v.as_str()).ok_or_else(|| {
                    WorkflowError::InvalidInput("Missing query parameter".to_string())
                })?;

                let results = self.search_databases(query).await?;

                Ok(serde_json::json!({
                    "notion_search_results": results,
                    "notion_search_completed": true
                }))
            }
            "get_page" => {
                let page_id = input
                    .get("page_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing page_id parameter".to_string())
                    })?;

                let api = self.api_service.read().await;
                let page = api.get_page(page_id).await?;

                Ok(serde_json::json!({
                    "page": page,
                    "operation_completed": true
                }))
            }
            "create_page" => {
                let title = input.get("title").and_then(|v| v.as_str()).ok_or_else(|| {
                    WorkflowError::InvalidInput("Missing title parameter".to_string())
                })?;

                let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

                let parent_id = input
                    .get("parent_id")
                    .or_else(|| input.get("database_id"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing parent_id parameter".to_string())
                    })?;

                let api = self.api_service.read().await;

                // Create parent reference
                let parent = crate::mcp::clients::services::notion_api::Parent::Database {
                    database_id: parent_id.to_string(),
                };

                // Create properties with title
                let properties = serde_json::json!({
                    "Name": NotionApiService::create_title_property(title)
                });

                // Create content blocks if content is provided
                let children = if !content.is_empty() {
                    Some(vec![
                        crate::mcp::clients::services::notion_api::Block::Paragraph {
                            paragraph: crate::mcp::clients::services::notion_api::ParagraphBlock {
                                rich_text: NotionApiService::create_rich_text(content),
                            },
                        },
                    ])
                } else {
                    None
                };

                let page = api.create_page(parent, properties, children).await?;

                Ok(serde_json::json!({
                    "page": page,
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
        "NotionServiceNode"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_search_pages() {
        let mock_server = MockServer::start().await;
        let api_service =
            NotionApiService::with_base_url("test-api-key".to_string(), mock_server.uri());

        let node = NotionServiceNode {
            api_service: Arc::new(RwLock::new(api_service)),
        };

        Mock::given(method("POST"))
            .and(path("/v1/search"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "object": "page",
                    "id": "123",
                    "created_time": "2024-01-01T00:00:00Z",
                    "last_edited_time": "2024-01-01T00:00:00Z",
                    "archived": false,
                    "properties": {
                        "Name": {
                            "title": [{
                                "plain_text": "Test Page"
                            }]
                        }
                    },
                    "parent": {
                        "type": "workspace",
                        "workspace": true
                    },
                    "url": "https://notion.so/123"
                }],
                "has_more": false,
                "next_cursor": null
            })))
            .mount(&mock_server)
            .await;

        let input = serde_json::json!({
            "operation": "search_pages",
            "query": "test query"
        });

        let result = node.execute(input, &TaskContext::default()).await.unwrap();
        let search_results = result.get("notion_search_results").unwrap();

        assert_eq!(search_results.get("query").unwrap(), "test query");
        assert_eq!(search_results.get("results_found").unwrap(), 1);
        assert!(search_results.get("real_api").unwrap().as_bool().unwrap());

        let pages = search_results.get("pages").unwrap().as_array().unwrap();
        assert_eq!(pages[0].get("title").unwrap(), "Test Page");
    }
}
