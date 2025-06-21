/// HelpScout Service Integration - Provides real API access for HelpScout operations
///
/// This service acts as a bridge between the workflow nodes and the actual HelpScout API,
/// handling authentication, error handling, and response formatting.
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::mcp::clients::services::HelpScoutApiService;
use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Service node that provides real HelpScout API integration
#[derive(Debug)]
pub struct HelpscoutServiceNode {
    api_service: Arc<RwLock<HelpScoutApiService>>,
}

impl HelpscoutServiceNode {
    /// Create a new HelpScout service node with API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_service: Arc::new(RwLock::new(HelpScoutApiService::new(api_key))),
        }
    }

    /// Search articles using the real HelpScout API
    pub async fn search_articles(&self, query: &str) -> Result<Value, WorkflowError> {
        let api = self.api_service.read().await;
        let search_result = api.search_articles(query, None, None, None).await?;

        // Convert to workflow-friendly format
        Ok(serde_json::json!({
            "source": "helpscout",
            "query": query,
            "results_found": search_result.total_count,
            "articles": search_result.articles.iter().map(|article| {
                serde_json::json!({
                    "id": article.id,
                    "title": article.name,
                    "url": article.url,
                    "preview": article.preview,
                    "relevance": article.score * 100.0
                })
            }).collect::<Vec<_>>(),
            "real_api": true
        }))
    }
}

#[async_trait]
impl Node for HelpscoutServiceNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Extract operation type
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("search");

        match operation {
            "search" => {
                let query = input
                    .get("query")
                    .or_else(|| input.get("user_query"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing query parameter".to_string())
                    })?;

                let results = self.search_articles(query).await?;

                Ok(serde_json::json!({
                    "helpscout_search_results": results,
                    "helpscout_search_completed": true
                }))
            }
            "get_article" => {
                let article_id = input
                    .get("article_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        WorkflowError::InvalidInput("Missing article_id parameter".to_string())
                    })?;

                let api = self.api_service.read().await;
                let article = api.get_article(article_id).await?;

                Ok(serde_json::json!({
                    "article": article,
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
        "HelpscoutServiceNode"
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
        let api_service =
            HelpScoutApiService::with_base_url("test-api-key".to_string(), mock_server.uri());

        let node = HelpscoutServiceNode {
            api_service: Arc::new(RwLock::new(api_service)),
        };

        Mock::given(method("GET"))
            .and(path("/search/articles"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "articles": [{
                    "id": "123",
                    "collection_id": "456",
                    "name": "Test Article",
                    "preview": "This is a test",
                    "url": "https://example.com/article/123",
                    "score": 0.95
                }],
                "total_count": 1,
                "page": 1,
                "pages": 1
            })))
            .mount(&mock_server)
            .await;

        let input = serde_json::json!({
            "operation": "search",
            "query": "test query"
        });

        let result = node.execute(input, &TaskContext::default()).await.unwrap();
        let search_results = result.get("helpscout_search_results").unwrap();

        assert_eq!(search_results.get("query").unwrap(), "test query");
        assert_eq!(search_results.get("results_found").unwrap(), 1);
        assert!(search_results.get("real_api").unwrap().as_bool().unwrap());
    }
}
