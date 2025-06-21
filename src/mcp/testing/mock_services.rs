// Mock implementations for external services

use async_trait::async_trait;
use reqwest::{Response, StatusCode};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::{ExpectationTimes, MockableService, ServiceExpectation};

/// Mock HelpScout service
#[derive(Debug, Clone)]
pub struct MockHelpScout {
    expectations: Arc<Mutex<Vec<(ServiceExpectation, usize)>>>,
    unmatched_calls: Arc<Mutex<Vec<UnmatchedCall>>>,
}

/// Mock Notion service
#[derive(Debug, Clone)]
pub struct MockNotion {
    expectations: Arc<Mutex<Vec<(ServiceExpectation, usize)>>>,
    unmatched_calls: Arc<Mutex<Vec<UnmatchedCall>>>,
}

/// Mock Slack service
#[derive(Debug, Clone)]
pub struct MockSlack {
    expectations: Arc<Mutex<Vec<(ServiceExpectation, usize)>>>,
    unmatched_calls: Arc<Mutex<Vec<UnmatchedCall>>>,
}

#[derive(Debug, Clone)]
struct UnmatchedCall {
    method: String,
    path: String,
    body: Option<serde_json::Value>,
    timestamp: std::time::Instant,
}

// Common implementation for all mock services
macro_rules! impl_mock_service {
    ($type:ty) => {
        impl $type {
            pub fn new() -> Self {
                Self {
                    expectations: Arc::new(Mutex::new(Vec::new())),
                    unmatched_calls: Arc::new(Mutex::new(Vec::new())),
                }
            }

            pub async fn handle_request(
                &self,
                method: &str,
                path: &str,
                body: Option<serde_json::Value>,
            ) -> Result<(u16, serde_json::Value), String> {
                let mut expectations = self.expectations.lock().await;

                // Find matching expectation
                for (exp, times_called) in expectations.iter_mut() {
                    if exp.method == method {
                        if let Some(exp_path) = &exp.path {
                            if exp_path != path {
                                continue;
                            }
                        }

                        if let Some(exp_body) = &exp.request_body {
                            if let Some(req_body) = &body {
                                if exp_body != req_body {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        // Check if expectation can be used
                        let can_use = match exp.times {
                            ExpectationTimes::Once => *times_called == 0,
                            ExpectationTimes::Exactly(n) => *times_called < n,
                            ExpectationTimes::AtLeast(_) => true,
                            ExpectationTimes::AtMost(n) => *times_called < n,
                            ExpectationTimes::Any => true,
                        };

                        if can_use {
                            *times_called += 1;
                            return Ok((exp.response_status, exp.response_body.clone()));
                        }
                    }
                }

                // No matching expectation found
                let mut unmatched = self.unmatched_calls.lock().await;
                unmatched.push(UnmatchedCall {
                    method: method.to_string(),
                    path: path.to_string(),
                    body,
                    timestamp: std::time::Instant::now(),
                });

                Err(format!("No matching expectation for {} {}", method, path))
            }
        }

        #[async_trait]
        impl MockableService for $type {
            async fn expect(&mut self, expectation: ServiceExpectation) {
                let mut expectations = self.expectations.lock().await;
                expectations.push((expectation, 0));
            }

            async fn verify(&self) -> Result<(), String> {
                let expectations = self.expectations.lock().await;
                let unmatched = self.unmatched_calls.lock().await;

                let mut errors = Vec::new();

                // Check expectations were met
                for (exp, times_called) in expectations.iter() {
                    let valid = match exp.times {
                        ExpectationTimes::Once => *times_called == 1,
                        ExpectationTimes::Exactly(n) => *times_called == n,
                        ExpectationTimes::AtLeast(n) => *times_called >= n,
                        ExpectationTimes::AtMost(n) => *times_called <= n,
                        ExpectationTimes::Any => true,
                    };

                    if !valid {
                        errors.push(format!(
                            "Expectation not met: {} {} (called {} times, expected {:?})",
                            exp.method,
                            exp.path.as_deref().unwrap_or("*"),
                            times_called,
                            exp.times
                        ));
                    }
                }

                // Report unmatched calls
                if !unmatched.is_empty() {
                    errors.push(format!(
                        "Unexpected calls: {}",
                        unmatched
                            .iter()
                            .map(|c| format!("{} {}", c.method, c.path))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }

                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors.join("\n"))
                }
            }

            async fn reset(&mut self) {
                self.expectations.lock().await.clear();
                self.unmatched_calls.lock().await.clear();
            }
        }
    };
}

impl_mock_service!(MockHelpScout);
impl_mock_service!(MockNotion);
impl_mock_service!(MockSlack);

// Service-specific helper methods
impl MockHelpScout {
    pub async fn expect_create_conversation(&mut self, customer_id: &str, mailbox_id: i32) {
        self.expect(ServiceExpectation {
            method: "POST".to_string(),
            path: Some("/v2/conversations".to_string()),
            request_body: Some(json!({
                "customer": { "id": customer_id },
                "mailbox": { "id": mailbox_id },
                "status": "active",
                "type": "email"
            })),
            response_status: 201,
            response_body: json!({
                "id": 12345,
                "status": "active",
                "mailbox": { "id": mailbox_id }
            }),
            times: ExpectationTimes::Once,
        })
        .await;
    }

    pub async fn expect_list_conversations(&mut self, conversations: Vec<serde_json::Value>) {
        self.expect(ServiceExpectation {
            method: "GET".to_string(),
            path: Some("/v2/conversations".to_string()),
            request_body: None,
            response_status: 200,
            response_body: json!({
                "conversations": conversations,
                "page": 1,
                "pages": 1,
                "count": conversations.len()
            }),
            times: ExpectationTimes::Any,
        })
        .await;
    }
}

impl MockNotion {
    pub async fn expect_create_page(&mut self, parent_id: &str, title: &str) {
        self.expect(ServiceExpectation {
            method: "POST".to_string(),
            path: Some("/v1/pages".to_string()),
            request_body: Some(json!({
                "parent": { "database_id": parent_id },
                "properties": {
                    "title": {
                        "title": [{
                            "text": { "content": title }
                        }]
                    }
                }
            })),
            response_status: 200,
            response_body: json!({
                "id": "page-123",
                "object": "page",
                "created_time": "2023-01-01T00:00:00Z"
            }),
            times: ExpectationTimes::Once,
        })
        .await;
    }

    pub async fn expect_query_database(
        &mut self,
        database_id: &str,
        results: Vec<serde_json::Value>,
    ) {
        self.expect(ServiceExpectation {
            method: "POST".to_string(),
            path: Some(format!("/v1/databases/{}/query", database_id)),
            request_body: None,
            response_status: 200,
            response_body: json!({
                "results": results,
                "has_more": false
            }),
            times: ExpectationTimes::Any,
        })
        .await;
    }
}

impl MockSlack {
    pub async fn expect_post_message(&mut self, channel: &str, text: &str) {
        self.expect(ServiceExpectation {
            method: "POST".to_string(),
            path: Some("/api/chat.postMessage".to_string()),
            request_body: Some(json!({
                "channel": channel,
                "text": text
            })),
            response_status: 200,
            response_body: json!({
                "ok": true,
                "ts": "1234567890.123456",
                "channel": channel
            }),
            times: ExpectationTimes::Once,
        })
        .await;
    }

    pub async fn expect_list_channels(&mut self, channels: Vec<serde_json::Value>) {
        self.expect(ServiceExpectation {
            method: "GET".to_string(),
            path: Some("/api/conversations.list".to_string()),
            request_body: None,
            response_status: 200,
            response_body: json!({
                "ok": true,
                "channels": channels,
                "response_metadata": {
                    "next_cursor": ""
                }
            }),
            times: ExpectationTimes::Any,
        })
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_helpscout_expectations() {
        let mut mock = MockHelpScout::new();

        // Set up expectation
        mock.expect_create_conversation("cust-123", 1).await;

        // Make matching request
        let (status, body) = mock
            .handle_request(
                "POST",
                "/v2/conversations",
                Some(json!({
                    "customer": { "id": "cust-123" },
                    "mailbox": { "id": 1 },
                    "status": "active",
                    "type": "email"
                })),
            )
            .await
            .unwrap();

        assert_eq!(status, 201);
        assert_eq!(body["id"], 12345);

        // Verify expectations were met
        assert!(mock.verify().await.is_ok());
    }

    #[tokio::test]
    async fn test_mock_notion_query() {
        let mut mock = MockNotion::new();

        // Set up expectation
        let results = vec![
            json!({"id": "page-1", "properties": {}}),
            json!({"id": "page-2", "properties": {}}),
        ];
        mock.expect_query_database("db-123", results).await;

        // Make request
        let (status, body) = mock
            .handle_request("POST", "/v1/databases/db-123/query", None)
            .await
            .unwrap();

        assert_eq!(status, 200);
        assert_eq!(body["results"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_mock_slack_post_message() {
        let mut mock = MockSlack::new();

        // Set up expectation
        mock.expect_post_message("#general", "Hello, world!").await;

        // Make request
        let (status, body) = mock
            .handle_request(
                "POST",
                "/api/chat.postMessage",
                Some(json!({
                    "channel": "#general",
                    "text": "Hello, world!"
                })),
            )
            .await
            .unwrap();

        assert_eq!(status, 200);
        assert_eq!(body["ok"], true);

        // Verify
        assert!(mock.verify().await.is_ok());
    }
}
