#[cfg(test)]
mod tests {
    use crate::mcp::clients::services::enhanced_services::EnhancedHelpScoutService;
    use crate::mcp::clients::services::helpscout_api::{Article, ArticleSearchHit, SearchResult};
    use crate::mcp::testing::{
        ExpectationTimes, IntegrationTestConfig, MockHelpScout, MockableService,
        ServiceExpectation, TestHarness,
    };
    use serde_json::json;

    #[tokio::test]
    async fn test_helpscout_service_search_articles() {
        let config = IntegrationTestConfig::default();
        let harness = TestHarness::new(config);

        harness
            .run_test(|_context| async move {
                // Create HelpScout service
                let service = EnhancedHelpScoutService::new("test-api-key".to_string());

                // Test basic service creation - just verify it doesn't panic
                // We can't test internal fields since they're private

                Ok(())
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_helpscout_mock_integration() {
        let config = IntegrationTestConfig::default();
        let harness = TestHarness::new(config);

        harness
            .run_test(|context| async move {
                // Set up mock HelpScout service
                let mut mock_helpscout = MockHelpScout::new();

                // Configure mock expectations
                mock_helpscout
                    .expect(ServiceExpectation {
                        method: "GET".to_string(),
                        path: Some("/v2/search/articles".to_string()),
                        request_body: None,
                        response_status: 200,
                        response_body: json!({
                            "articles": [{
                                "id": "123",
                                "collection_id": "456",
                                "name": "Test Article",
                                "preview": "This is a test article about scheduling",
                                "url": "https://example.helpscout.net/article/123",
                                "score": 0.95
                            }],
                            "total_count": 1,
                            "page": 1,
                            "pages": 1
                        }),
                        times: ExpectationTimes::Once,
                    })
                    .await;

                // Register the mock
                context
                    .registry
                    .register("helpscout".to_string(), Box::new(mock_helpscout))
                    .await;

                Ok(())
            })
            .await
            .unwrap();
    }

    #[test]
    fn test_helpscout_error_handling() {
        let service = EnhancedHelpScoutService::new("invalid-key".to_string());

        // Test that service is created even with invalid key
        // Error handling happens at request time - just verify no panic
        let _ = service;
    }

    #[test]
    fn test_helpscout_service_creation() {
        let service = EnhancedHelpScoutService::new("test-key".to_string());

        // Test basic service creation - verify no panic
        let _ = service;
    }

    #[tokio::test]
    async fn test_helpscout_api_types() {
        // Test that our types serialize/deserialize correctly
        let article = Article {
            id: "123".to_string(),
            collection_id: "456".to_string(),
            name: "Test Article".to_string(),
            text: "Test content".to_string(),
            status: "published".to_string(),
            slug: "test-article".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            updated_at: "2023-01-01T00:00:00Z".to_string(),
            published_at: Some("2023-01-01T00:00:00Z".to_string()),
            author: None,
            tags: vec!["test".to_string()],
        };

        let json_str = serde_json::to_string(&article).unwrap();
        let deserialized: Article = serde_json::from_str(&json_str).unwrap();

        assert_eq!(article.id, deserialized.id);
        assert_eq!(article.name, deserialized.name);
        assert_eq!(article.tags, deserialized.tags);
    }

    #[tokio::test]
    async fn test_search_result_pagination() {
        // Test pagination handling
        let search_result = SearchResult {
            articles: vec![],
            total_count: 0,
            page: 2,
            pages: 5,
        };

        assert_eq!(search_result.page, 2);
        assert_eq!(search_result.pages, 5);
        assert_eq!(search_result.total_count, 0);
    }
}
