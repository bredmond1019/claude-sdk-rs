/// Enhanced service implementations with comprehensive error handling, rate limiting, and circuit breakers
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::mcp::clients::services::{
    ErrorHandler, ErrorHandlingConfig, HelpScoutApiService, NotionApiService, RateLimitConfig,
    RateLimiter, SlackApiService,
};
use crate::mcp::core::error::WorkflowError;

/// Enhanced HelpScout service with error handling and rate limiting
#[derive(Debug)]
pub struct EnhancedHelpScoutService {
    api_service: HelpScoutApiService,
    error_handler: Arc<RwLock<ErrorHandler>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

impl EnhancedHelpScoutService {
    pub fn new(api_key: String) -> Self {
        let error_config = ErrorHandlingConfig::default();
        let rate_config = RateLimitConfig {
            requests_per_second: 10.0, // HelpScout rate limit
            burst_size: 20,
        };

        Self {
            api_service: HelpScoutApiService::new(api_key),
            error_handler: Arc::new(RwLock::new(ErrorHandler::new(error_config))),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(rate_config))),
        }
    }

    pub async fn search_articles(
        &self,
        query: &str,
        collection_id: Option<&str>,
        page: Option<u32>,
        per_page: Option<u32>,
    ) -> Result<crate::mcp::clients::services::helpscout_api::SearchResult, WorkflowError> {
        // Apply rate limiting
        self.rate_limiter.write().await.wait_if_needed().await;

        // Execute with retry logic
        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(
                || async {
                    api.search_articles(query, collection_id, page, per_page)
                        .await
                },
                "HelpScout",
            )
            .await
    }

    pub async fn get_article(
        &self,
        article_id: &str,
    ) -> Result<crate::mcp::clients::services::helpscout_api::Article, WorkflowError> {
        self.rate_limiter.write().await.wait_if_needed().await;

        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(|| async { api.get_article(article_id).await }, "HelpScout")
            .await
    }
}

/// Enhanced Notion service with error handling and rate limiting
#[derive(Debug)]
pub struct EnhancedNotionService {
    api_service: NotionApiService,
    error_handler: Arc<RwLock<ErrorHandler>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

impl EnhancedNotionService {
    pub fn new(api_key: String) -> Self {
        let error_config = ErrorHandlingConfig::default();
        let rate_config = RateLimitConfig {
            requests_per_second: 3.0, // Notion rate limit
            burst_size: 10,
        };

        Self {
            api_service: NotionApiService::new(api_key),
            error_handler: Arc::new(RwLock::new(ErrorHandler::new(error_config))),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(rate_config))),
        }
    }

    pub async fn search(
        &self,
        query: &str,
        filter: Option<&str>,
        sort: Option<&str>,
        start_cursor: Option<&str>,
        page_size: Option<u32>,
    ) -> Result<crate::mcp::clients::services::notion_api::SearchResponse, WorkflowError> {
        self.rate_limiter.write().await.wait_if_needed().await;

        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(
                || async {
                    api.search(query, filter, sort, start_cursor, page_size)
                        .await
                },
                "Notion",
            )
            .await
    }

    pub async fn get_page(
        &self,
        page_id: &str,
    ) -> Result<crate::mcp::clients::services::notion_api::Page, WorkflowError> {
        self.rate_limiter.write().await.wait_if_needed().await;

        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(|| async { api.get_page(page_id).await }, "Notion")
            .await
    }

    pub async fn create_page(
        &self,
        parent: crate::mcp::clients::services::notion_api::Parent,
        properties: serde_json::Value,
        children: Option<Vec<crate::mcp::clients::services::notion_api::Block>>,
    ) -> Result<crate::mcp::clients::services::notion_api::Page, WorkflowError> {
        self.rate_limiter.write().await.wait_if_needed().await;

        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(
                || async {
                    api.create_page(parent.clone(), properties.clone(), children.clone())
                        .await
                },
                "Notion",
            )
            .await
    }
}

/// Enhanced Slack service with error handling and rate limiting
#[derive(Debug)]
pub struct EnhancedSlackService {
    api_service: SlackApiService,
    error_handler: Arc<RwLock<ErrorHandler>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

impl EnhancedSlackService {
    pub fn new(bot_token: String) -> Self {
        let mut error_config = ErrorHandlingConfig::default();
        // Slack is more resilient, reduce retry attempts
        error_config.retry_policy.max_attempts = 2;

        let rate_config = RateLimitConfig {
            requests_per_second: 20.0, // Slack Tier 2 rate limit
            burst_size: 100,
        };

        Self {
            api_service: SlackApiService::new(bot_token),
            error_handler: Arc::new(RwLock::new(ErrorHandler::new(error_config))),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(rate_config))),
        }
    }

    pub async fn post_message(
        &self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
        blocks: Option<Vec<serde_json::Value>>,
    ) -> Result<crate::mcp::clients::services::slack_api::PostMessageResponse, WorkflowError> {
        self.rate_limiter.write().await.wait_if_needed().await;

        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(
                || async {
                    api.post_message(channel, text, thread_ts, blocks.clone())
                        .await
                },
                "Slack",
            )
            .await
    }

    pub async fn list_channels(
        &self,
        exclude_archived: bool,
        types: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<crate::mcp::clients::services::slack_api::Channel>, WorkflowError> {
        self.rate_limiter.write().await.wait_if_needed().await;

        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(
                || async { api.list_channels(exclude_archived, types, limit).await },
                "Slack",
            )
            .await
    }

    pub async fn search_messages(
        &self,
        query: &str,
        count: Option<u32>,
        page: Option<u32>,
    ) -> Result<Vec<crate::mcp::clients::services::slack_api::Message>, WorkflowError> {
        self.rate_limiter.write().await.wait_if_needed().await;

        let api = &self.api_service;
        self.error_handler
            .write()
            .await
            .execute_with_retry(
                || async { api.search_messages(query, count, page).await },
                "Slack",
            )
            .await
    }
}

/// Service health monitor for tracking the health of all external services
#[derive(Debug)]
pub struct ServiceHealthMonitor {
    helpscout: Option<Arc<EnhancedHelpScoutService>>,
    notion: Option<Arc<EnhancedNotionService>>,
    slack: Option<Arc<EnhancedSlackService>>,
}

impl ServiceHealthMonitor {
    pub fn new() -> Self {
        Self {
            helpscout: None,
            notion: None,
            slack: None,
        }
    }

    pub fn with_helpscout(mut self, service: Arc<EnhancedHelpScoutService>) -> Self {
        self.helpscout = Some(service);
        self
    }

    pub fn with_notion(mut self, service: Arc<EnhancedNotionService>) -> Self {
        self.notion = Some(service);
        self
    }

    pub fn with_slack(mut self, service: Arc<EnhancedSlackService>) -> Self {
        self.slack = Some(service);
        self
    }

    /// Check health of all configured services
    pub async fn check_all_services(&self) -> Vec<ServiceHealthStatus> {
        let mut results = Vec::new();

        // Check HelpScout
        if let Some(ref _helpscout) = self.helpscout {
            let status = self.check_helpscout_health().await;
            results.push(status);
        }

        // Check Notion
        if let Some(ref _notion) = self.notion {
            let status = self.check_notion_health().await;
            results.push(status);
        }

        // Check Slack
        if let Some(ref _slack) = self.slack {
            let status = self.check_slack_health().await;
            results.push(status);
        }

        results
    }

    async fn check_helpscout_health(&self) -> ServiceHealthStatus {
        let start = std::time::Instant::now();

        // Try to list a single article as health check
        let health = if let Some(ref service) = self.helpscout {
            match service
                .api_service
                .list_articles(None, Some(1), Some(1))
                .await
            {
                Ok(_) => ServiceHealth::Healthy,
                Err(e) => ServiceHealth::Unhealthy(format!("Health check failed: {:?}", e)),
            }
        } else {
            ServiceHealth::Unhealthy("Service not configured".to_string())
        };

        ServiceHealthStatus {
            service: "HelpScout".to_string(),
            health,
            response_time: start.elapsed(),
            last_checked: std::time::SystemTime::now(),
        }
    }

    async fn check_notion_health(&self) -> ServiceHealthStatus {
        let start = std::time::Instant::now();

        // Try a minimal search as health check
        let health = if let Some(ref service) = self.notion {
            match service
                .api_service
                .search("", None, None, None, Some(1))
                .await
            {
                Ok(_) => ServiceHealth::Healthy,
                Err(e) => ServiceHealth::Unhealthy(format!("Health check failed: {:?}", e)),
            }
        } else {
            ServiceHealth::Unhealthy("Service not configured".to_string())
        };

        ServiceHealthStatus {
            service: "Notion".to_string(),
            health,
            response_time: start.elapsed(),
            last_checked: std::time::SystemTime::now(),
        }
    }

    async fn check_slack_health(&self) -> ServiceHealthStatus {
        let start = std::time::Instant::now();

        // Try to list channels as health check
        let health = if let Some(ref service) = self.slack {
            match service.api_service.list_channels(true, None, Some(1)).await {
                Ok(_) => ServiceHealth::Healthy,
                Err(e) => ServiceHealth::Unhealthy(format!("Health check failed: {:?}", e)),
            }
        } else {
            ServiceHealth::Unhealthy("Service not configured".to_string())
        };

        ServiceHealthStatus {
            service: "Slack".to_string(),
            health,
            response_time: start.elapsed(),
            last_checked: std::time::SystemTime::now(),
        }
    }
}

/// Service health status
#[derive(Debug, Clone)]
pub struct ServiceHealthStatus {
    pub service: String,
    pub health: ServiceHealth,
    pub response_time: std::time::Duration,
    pub last_checked: std::time::SystemTime,
}

/// Service health state
#[derive(Debug, Clone)]
pub enum ServiceHealth {
    Healthy,
    Degraded(String),
    Unhealthy(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_service_creation() {
        let _ = EnhancedHelpScoutService::new("test-key".to_string());
        let _ = EnhancedNotionService::new("test-key".to_string());
        let _ = EnhancedSlackService::new("test-token".to_string());
    }

    #[test]
    fn test_health_monitor_builder() {
        let monitor = ServiceHealthMonitor::new()
            .with_helpscout(Arc::new(EnhancedHelpScoutService::new("key".to_string())))
            .with_notion(Arc::new(EnhancedNotionService::new("key".to_string())))
            .with_slack(Arc::new(EnhancedSlackService::new("token".to_string())));

        assert!(monitor.helpscout.is_some());
        assert!(monitor.notion.is_some());
        assert!(monitor.slack.is_some());
    }
}
