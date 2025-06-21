use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::mcp::core::error::WorkflowError;

/// Notion API service for real API integration
///
/// This service provides actual HTTP API calls to Notion's API v1.
/// Documentation: https://developers.notion.com/reference/intro
#[derive(Debug, Clone)]
pub struct NotionApiService {
    client: Client,
    api_key: String,
    base_url: String,
    version: String,
}

/// Represents a Notion page
#[derive(Debug, Serialize, Deserialize)]
pub struct Page {
    pub id: String,
    pub created_time: String,
    pub last_edited_time: String,
    pub archived: bool,
    pub properties: serde_json::Value,
    pub parent: Parent,
    pub url: String,
}

/// Represents a page parent (database or page)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Parent {
    #[serde(rename = "database_id")]
    Database { database_id: String },
    #[serde(rename = "page_id")]
    Page { page_id: String },
    #[serde(rename = "workspace")]
    Workspace { workspace: bool },
}

/// Represents a Notion database
#[derive(Debug, Serialize, Deserialize)]
pub struct Database {
    pub id: String,
    pub created_time: String,
    pub last_edited_time: String,
    pub title: Vec<RichText>,
    pub properties: serde_json::Value,
    pub parent: Parent,
    pub url: String,
}

/// Represents rich text in Notion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichText {
    #[serde(rename = "type")]
    pub text_type: String,
    pub text: Option<TextContent>,
    pub plain_text: Option<String>,
}

/// Text content within rich text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    pub content: String,
    pub link: Option<serde_json::Value>,
}

/// Search response from Notion
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub has_more: bool,
    pub next_cursor: Option<String>,
}

/// Individual search result
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "object")]
pub enum SearchResult {
    #[serde(rename = "page")]
    Page(Page),
    #[serde(rename = "database")]
    Database(Database),
}

/// Query database response
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<Page>,
    pub has_more: bool,
    pub next_cursor: Option<String>,
}

/// Block types for page content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Block {
    #[serde(rename = "paragraph")]
    Paragraph { paragraph: ParagraphBlock },
    #[serde(rename = "heading_1")]
    Heading1 { heading_1: HeadingBlock },
    #[serde(rename = "heading_2")]
    Heading2 { heading_2: HeadingBlock },
    #[serde(rename = "heading_3")]
    Heading3 { heading_3: HeadingBlock },
    #[serde(rename = "bulleted_list_item")]
    BulletedListItem { bulleted_list_item: TextBlock },
    #[serde(rename = "numbered_list_item")]
    NumberedListItem { numbered_list_item: TextBlock },
}

/// Paragraph block content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParagraphBlock {
    pub rich_text: Vec<RichText>,
}

/// Heading block content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadingBlock {
    pub rich_text: Vec<RichText>,
}

/// Text block content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub rich_text: Vec<RichText>,
}

impl NotionApiService {
    /// Create a new Notion API service
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            base_url: "https://api.notion.com".to_string(),
            version: "2022-06-28".to_string(),
        }
    }

    /// Create a new Notion API service with custom base URL (for testing)
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            base_url,
            version: "2022-06-28".to_string(),
        }
    }

    /// Search for pages and databases
    pub async fn search(
        &self,
        query: &str,
        filter: Option<&str>,
        sort: Option<&str>,
        start_cursor: Option<&str>,
        page_size: Option<u32>,
    ) -> Result<SearchResponse, WorkflowError> {
        let url = format!("{}/v1/search", self.base_url);

        let mut body = serde_json::json!({
            "query": query
        });

        if let Some(filter_type) = filter {
            body["filter"] = serde_json::json!({
                "property": "object",
                "value": filter_type
            });
        }

        if let Some(sort_order) = sort {
            body["sort"] = serde_json::json!({
                "direction": sort_order,
                "timestamp": "last_edited_time"
            });
        }

        if let Some(cursor) = start_cursor {
            body["start_cursor"] = serde_json::json!(cursor);
        }

        if let Some(size) = page_size {
            body["page_size"] = serde_json::json!(size.min(100));
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Notion-Version", &self.version)
            .json(&body)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Failed to search: {}", e),
            })?;

        if response.status() == StatusCode::UNAUTHORIZED {
            return Err(WorkflowError::AuthenticationError {
                message: "Invalid Notion API key".to_string(),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Search failed with status: {}", response.status()),
            });
        }

        response.json::<SearchResponse>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse search results: {}", e))
        })
    }

    /// Get a page by ID
    pub async fn get_page(&self, page_id: &str) -> Result<Page, WorkflowError> {
        let url = format!("{}/v1/pages/{}", self.base_url, page_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Notion-Version", &self.version)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Failed to get page: {}", e),
            })?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(WorkflowError::NotFound {
                resource: format!("Page with ID: {}", page_id),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Get page failed with status: {}", response.status()),
            });
        }

        response
            .json::<Page>()
            .await
            .map_err(|e| WorkflowError::SerializationError(format!("Failed to parse page: {}", e)))
    }

    /// Create a new page
    pub async fn create_page(
        &self,
        parent: Parent,
        properties: serde_json::Value,
        children: Option<Vec<Block>>,
    ) -> Result<Page, WorkflowError> {
        let url = format!("{}/v1/pages", self.base_url);

        let mut body = serde_json::json!({
            "parent": parent,
            "properties": properties
        });

        if let Some(blocks) = children {
            body["children"] = serde_json::json!(blocks);
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Notion-Version", &self.version)
            .json(&body)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Failed to create page: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Create page failed with status: {}", response.status()),
            });
        }

        response.json::<Page>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse created page: {}", e))
        })
    }

    /// Update page properties
    pub async fn update_page(
        &self,
        page_id: &str,
        properties: serde_json::Value,
    ) -> Result<Page, WorkflowError> {
        let url = format!("{}/v1/pages/{}", self.base_url, page_id);

        let body = serde_json::json!({
            "properties": properties
        });

        let response = self
            .client
            .patch(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Notion-Version", &self.version)
            .json(&body)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Failed to update page: {}", e),
            })?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(WorkflowError::NotFound {
                resource: format!("Page with ID: {}", page_id),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Update page failed with status: {}", response.status()),
            });
        }

        response.json::<Page>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse updated page: {}", e))
        })
    }

    /// Query a database
    pub async fn query_database(
        &self,
        database_id: &str,
        filter: Option<serde_json::Value>,
        sorts: Option<Vec<serde_json::Value>>,
        start_cursor: Option<&str>,
        page_size: Option<u32>,
    ) -> Result<QueryResponse, WorkflowError> {
        let url = format!("{}/v1/databases/{}/query", self.base_url, database_id);

        let mut body = serde_json::json!({});

        if let Some(f) = filter {
            body["filter"] = f;
        }

        if let Some(s) = sorts {
            body["sorts"] = serde_json::json!(s);
        }

        if let Some(cursor) = start_cursor {
            body["start_cursor"] = serde_json::json!(cursor);
        }

        if let Some(size) = page_size {
            body["page_size"] = serde_json::json!(size.min(100));
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Notion-Version", &self.version)
            .json(&body)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Failed to query database: {}", e),
            })?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(WorkflowError::NotFound {
                resource: format!("Database with ID: {}", database_id),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "Notion".to_string(),
                message: format!("Query database failed with status: {}", response.status()),
            });
        }

        response.json::<QueryResponse>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse query results: {}", e))
        })
    }

    /// Helper function to create rich text
    pub fn create_rich_text(content: &str) -> Vec<RichText> {
        vec![RichText {
            text_type: "text".to_string(),
            text: Some(TextContent {
                content: content.to_string(),
                link: None,
            }),
            plain_text: Some(content.to_string()),
        }]
    }

    /// Helper function to create a title property
    pub fn create_title_property(title: &str) -> serde_json::Value {
        serde_json::json!({
            "title": Self::create_rich_text(title)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_search() {
        let mock_server = MockServer::start().await;
        let api = NotionApiService::with_base_url("test-api-key".to_string(), mock_server.uri());

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
                    "properties": {},
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

        let result = api
            .search("test query", None, None, None, None)
            .await
            .unwrap();
        assert_eq!(result.results.len(), 1);
    }

    #[tokio::test]
    async fn test_create_page() {
        let mock_server = MockServer::start().await;
        let api = NotionApiService::with_base_url("test-api-key".to_string(), mock_server.uri());

        Mock::given(method("POST"))
            .and(path("/v1/pages"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "456",
                "created_time": "2024-01-01T00:00:00Z",
                "last_edited_time": "2024-01-01T00:00:00Z",
                "archived": false,
                "properties": {
                    "title": {
                        "title": [{
                            "type": "text",
                            "text": {
                                "content": "Test Page"
                            },
                            "plain_text": "Test Page"
                        }]
                    }
                },
                "parent": {
                    "type": "database_id",
                    "database_id": "789"
                },
                "url": "https://notion.so/456"
            })))
            .mount(&mock_server)
            .await;

        let parent = Parent::Database {
            database_id: "789".to_string(),
        };

        let properties = serde_json::json!({
            "Name": NotionApiService::create_title_property("Test Page")
        });

        let result = api.create_page(parent, properties, None).await.unwrap();
        assert_eq!(result.id, "456");
    }

    #[tokio::test]
    async fn test_authentication_error() {
        let mock_server = MockServer::start().await;
        let api = NotionApiService::with_base_url("invalid-key".to_string(), mock_server.uri());

        Mock::given(method("POST"))
            .and(path("/v1/search"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock_server)
            .await;

        let result = api.search("test", None, None, None, None).await;
        assert!(matches!(
            result.unwrap_err(),
            WorkflowError::AuthenticationError { .. }
        ));
    }
}
