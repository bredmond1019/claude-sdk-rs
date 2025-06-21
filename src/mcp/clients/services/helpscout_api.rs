use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::mcp::core::error::WorkflowError;

/// HelpScout API service for real API integration
///
/// This service provides actual HTTP API calls to HelpScout's REST API v2.
/// Documentation: https://developer.helpscout.com/docs-api/
#[derive(Debug, Clone)]
pub struct HelpScoutApiService {
    client: Client,
    api_key: String,
    base_url: String,
}

/// Represents a HelpScout article
#[derive(Debug, Serialize, Deserialize)]
pub struct Article {
    pub id: String,
    pub collection_id: String,
    pub name: String,
    pub text: String,
    pub status: String,
    pub slug: String,
    pub created_at: String,
    pub updated_at: String,
    pub published_at: Option<String>,
    pub author: Option<Author>,
    pub tags: Vec<String>,
}

/// Represents an article author
#[derive(Debug, Serialize, Deserialize)]
pub struct Author {
    pub id: String,
    pub first_name: String,
    pub last_name: String,
    pub email: String,
}

/// Represents a HelpScout collection
#[derive(Debug, Serialize, Deserialize)]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub visibility: String,
    pub order: i32,
    pub article_count: i32,
    pub created_at: String,
    pub updated_at: String,
}

/// Represents a search result
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub articles: Vec<ArticleSearchHit>,
    pub total_count: i32,
    pub page: i32,
    pub pages: i32,
}

/// Represents an article search hit
#[derive(Debug, Serialize, Deserialize)]
pub struct ArticleSearchHit {
    pub id: String,
    pub collection_id: String,
    pub name: String,
    pub preview: String,
    pub url: String,
    pub score: f32,
}

/// Represents a HelpScout conversation
#[derive(Debug, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub subject: String,
    pub status: String,
    pub mailbox_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub preview: String,
    pub customer: Customer,
}

/// Represents a customer in a conversation
#[derive(Debug, Serialize, Deserialize)]
pub struct Customer {
    pub id: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub email: String,
}

/// Response wrapper for paginated results
#[derive(Debug, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub page: i32,
    pub pages: i32,
    pub count: i32,
    pub items: Vec<T>,
}

impl HelpScoutApiService {
    /// Create a new HelpScout API service
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            base_url: "https://api.helpscout.net/v2".to_string(),
        }
    }

    /// Create a new HelpScout API service with custom base URL (for testing)
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            base_url,
        }
    }

    /// Search for articles in the knowledge base
    pub async fn search_articles(
        &self,
        query: &str,
        collection_id: Option<&str>,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<SearchResult, WorkflowError> {
        let mut params = HashMap::new();
        params.insert("query", query.to_string());

        if let Some(cid) = collection_id {
            params.insert("collectionId", cid.to_string());
        }

        params.insert("page", page.unwrap_or(1).to_string());
        params.insert("perPage", per_page.unwrap_or(20).to_string());

        let url = format!("{}/search/articles", self.base_url);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to search articles: {}", e),
            })?;

        if response.status() == StatusCode::UNAUTHORIZED {
            return Err(WorkflowError::AuthenticationError {
                message: "Invalid HelpScout API key".to_string(),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Search failed with status: {}", response.status()),
            });
        }

        response.json::<SearchResult>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse search results: {}", e))
        })
    }

    /// Get a specific article by ID
    pub async fn get_article(&self, article_id: &str) -> Result<Article, WorkflowError> {
        let url = format!("{}/articles/{}", self.base_url, article_id);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to get article: {}", e),
            })?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(WorkflowError::NotFound {
                resource: format!("Article with ID: {}", article_id),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Get article failed with status: {}", response.status()),
            });
        }

        response.json::<Article>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse article: {}", e))
        })
    }

    /// List all articles with pagination
    pub async fn list_articles(
        &self,
        collection_id: Option<&str>,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<PaginatedResponse<Article>, WorkflowError> {
        let mut url = format!("{}/articles", self.base_url);
        let mut params = HashMap::new();

        if let Some(cid) = collection_id {
            url = format!("{}/collections/{}/articles", self.base_url, cid);
        }

        params.insert("page", page.unwrap_or(1).to_string());
        params.insert("perPage", per_page.unwrap_or(20).to_string());

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to list articles: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("List articles failed with status: {}", response.status()),
            });
        }

        response
            .json::<PaginatedResponse<Article>>()
            .await
            .map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse articles list: {}", e))
            })
    }

    /// List all collections
    pub async fn list_collections(
        &self,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<PaginatedResponse<Collection>, WorkflowError> {
        let url = format!("{}/collections", self.base_url);
        let mut params = HashMap::new();

        params.insert("page", page.unwrap_or(1).to_string());
        params.insert("perPage", per_page.unwrap_or(20).to_string());

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to list collections: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("List collections failed with status: {}", response.status()),
            });
        }

        response
            .json::<PaginatedResponse<Collection>>()
            .await
            .map_err(|e| {
                WorkflowError::SerializationError(format!(
                    "Failed to parse collections list: {}",
                    e
                ))
            })
    }

    /// Get a specific collection by ID
    pub async fn get_collection(&self, collection_id: &str) -> Result<Collection, WorkflowError> {
        let url = format!("{}/collections/{}", self.base_url, collection_id);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to get collection: {}", e),
            })?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(WorkflowError::NotFound {
                resource: format!("Collection with ID: {}", collection_id),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Get collection failed with status: {}", response.status()),
            });
        }

        response.json::<Collection>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse collection: {}", e))
        })
    }

    /// Create a new article
    pub async fn create_article(
        &self,
        collection_id: &str,
        name: &str,
        text: &str,
        status: Option<&str>,
        tags: Option<Vec<String>>,
    ) -> Result<Article, WorkflowError> {
        let url = format!("{}/articles", self.base_url);

        let mut body = serde_json::json!({
            "collectionId": collection_id,
            "name": name,
            "text": text,
            "status": status.unwrap_or("draft")
        });

        if let Some(tags) = tags {
            body["tags"] = serde_json::json!(tags);
        }

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to create article: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Create article failed with status: {}", response.status()),
            });
        }

        response.json::<Article>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse created article: {}", e))
        })
    }

    /// Update an existing article
    pub async fn update_article(
        &self,
        article_id: &str,
        name: Option<&str>,
        text: Option<&str>,
        status: Option<&str>,
        tags: Option<Vec<String>>,
    ) -> Result<Article, WorkflowError> {
        let url = format!("{}/articles/{}", self.base_url, article_id);

        let mut body = serde_json::json!({});

        if let Some(name) = name {
            body["name"] = serde_json::json!(name);
        }
        if let Some(text) = text {
            body["text"] = serde_json::json!(text);
        }
        if let Some(status) = status {
            body["status"] = serde_json::json!(status);
        }
        if let Some(tags) = tags {
            body["tags"] = serde_json::json!(tags);
        }

        let response = self
            .client
            .patch(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to update article: {}", e),
            })?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(WorkflowError::NotFound {
                resource: format!("Article with ID: {}", article_id),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Update article failed with status: {}", response.status()),
            });
        }

        response.json::<Article>().await.map_err(|e| {
            WorkflowError::SerializationError(format!("Failed to parse updated article: {}", e))
        })
    }

    /// Delete an article
    pub async fn delete_article(&self, article_id: &str) -> Result<(), WorkflowError> {
        let url = format!("{}/articles/{}", self.base_url, article_id);

        let response = self
            .client
            .delete(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to delete article: {}", e),
            })?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(WorkflowError::NotFound {
                resource: format!("Article with ID: {}", article_id),
            });
        }

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Delete article failed with status: {}", response.status()),
            });
        }

        Ok(())
    }

    /// Search conversations
    pub async fn search_conversations(
        &self,
        query: &str,
        mailbox_id: Option<&str>,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<PaginatedResponse<Conversation>, WorkflowError> {
        let mut params = HashMap::new();
        params.insert("query", query.to_string());

        if let Some(mid) = mailbox_id {
            params.insert("mailbox", mid.to_string());
        }

        params.insert("page", page.unwrap_or(1).to_string());
        params.insert("perPage", per_page.unwrap_or(20).to_string());

        let url = format!("{}/conversations", self.base_url);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .query(&params)
            .send()
            .await
            .map_err(|e| WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!("Failed to search conversations: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(WorkflowError::ExternalServiceError {
                service: "HelpScout".to_string(),
                message: format!(
                    "Search conversations failed with status: {}",
                    response.status()
                ),
            });
        }

        response
            .json::<PaginatedResponse<Conversation>>()
            .await
            .map_err(|e| {
                WorkflowError::SerializationError(format!("Failed to parse conversations: {}", e))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_search_articles() {
        let mock_server = MockServer::start().await;
        let api = HelpScoutApiService::with_base_url("test-api-key".to_string(), mock_server.uri());

        Mock::given(method("GET"))
            .and(path("/search/articles"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "articles": [{
                    "id": "123",
                    "collection_id": "456",
                    "name": "Test Article",
                    "preview": "This is a test article",
                    "url": "https://example.com/article/123",
                    "score": 0.95
                }],
                "total_count": 1,
                "page": 1,
                "pages": 1
            })))
            .mount(&mock_server)
            .await;

        let result = api.search_articles("test", None, None, None).await.unwrap();
        assert_eq!(result.articles.len(), 1);
        assert_eq!(result.articles[0].name, "Test Article");
    }

    #[tokio::test]
    async fn test_get_article() {
        let mock_server = MockServer::start().await;
        let api = HelpScoutApiService::with_base_url("test-api-key".to_string(), mock_server.uri());

        Mock::given(method("GET"))
            .and(path("/articles/123"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "123",
                "collection_id": "456",
                "name": "Test Article",
                "text": "Article content",
                "status": "published",
                "slug": "test-article",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "tags": ["test", "example"]
            })))
            .mount(&mock_server)
            .await;

        let result = api.get_article("123").await.unwrap();
        assert_eq!(result.id, "123");
        assert_eq!(result.name, "Test Article");
    }

    #[tokio::test]
    async fn test_create_article() {
        let mock_server = MockServer::start().await;
        let api = HelpScoutApiService::with_base_url("test-api-key".to_string(), mock_server.uri());

        Mock::given(method("POST"))
            .and(path("/articles"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "id": "789",
                "collection_id": "456",
                "name": "New Article",
                "text": "New content",
                "status": "draft",
                "slug": "new-article",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "tags": []
            })))
            .mount(&mock_server)
            .await;

        let result = api
            .create_article("456", "New Article", "New content", None, None)
            .await
            .unwrap();
        assert_eq!(result.id, "789");
        assert_eq!(result.name, "New Article");
    }

    #[tokio::test]
    async fn test_authentication_error() {
        let mock_server = MockServer::start().await;
        let api = HelpScoutApiService::with_base_url("invalid-key".to_string(), mock_server.uri());

        Mock::given(method("GET"))
            .and(path("/search/articles"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock_server)
            .await;

        let result = api.search_articles("test", None, None, None).await;
        assert!(matches!(
            result.unwrap_err(),
            WorkflowError::AuthenticationError { .. }
        ));
    }
}
