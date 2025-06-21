// Integration testing framework for MCP services
// Provides mocking capabilities for external services and test utilities

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::mcp::core::error::WorkflowError;
use crate::mcp::protocol::{CallToolResult, ToolDefinition};

pub mod mock_client;
pub mod mock_services;
pub mod test_harness;
pub mod test_utils;

pub use mock_client::MockMCPClient;
pub use mock_services::{MockHelpScout, MockNotion, MockSlack};
pub use test_harness::TestHarness;
pub use test_utils::{create_test_config, setup_test_environment};

/// Trait for mockable external services
#[async_trait]
pub trait MockableService: Send + Sync + std::fmt::Debug {
    /// Configure expected behavior for the mock
    async fn expect(&mut self, expectation: ServiceExpectation);

    /// Verify all expectations were met
    async fn verify(&self) -> Result<(), String>;

    /// Reset the mock to initial state
    async fn reset(&mut self);
}

/// Expectation configuration for mock services
#[derive(Debug, Clone)]
pub struct ServiceExpectation {
    pub method: String,
    pub path: Option<String>,
    pub request_body: Option<serde_json::Value>,
    pub response_status: u16,
    pub response_body: serde_json::Value,
    pub times: ExpectationTimes,
}

/// How many times an expectation should be matched
#[derive(Debug, Clone, Copy)]
pub enum ExpectationTimes {
    Once,
    Exactly(usize),
    AtLeast(usize),
    AtMost(usize),
    Any,
}

/// Mock service registry for dependency injection
#[derive(Debug)]
pub struct MockServiceRegistry {
    services: Arc<Mutex<HashMap<String, Box<dyn MockableService>>>>,
}

impl MockServiceRegistry {
    pub fn new() -> Self {
        Self {
            services: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn register(&self, name: String, service: Box<dyn MockableService>) {
        let mut services = self.services.lock().await;
        services.insert(name, service);
    }

    pub async fn get(&self, name: &str) -> Option<Box<dyn MockableService>> {
        let services = self.services.lock().await;
        services.get(name).map(|s| {
            // Clone the service for thread safety
            // In real implementation, we'd use Arc<Mutex<>> for the service
            panic!("MockableService should be wrapped in Arc<Mutex<>>")
        })
    }

    pub async fn verify_all(&self) -> Result<(), Vec<String>> {
        let services = self.services.lock().await;
        let mut errors = Vec::new();

        for (name, service) in services.iter() {
            if let Err(e) = service.verify().await {
                errors.push(format!("{}: {}", name, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub async fn reset_all(&self) {
        let mut services = self.services.lock().await;
        for (_, service) in services.iter_mut() {
            service.reset().await;
        }
    }
}

/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    pub use_mocks: bool,
    pub mock_registry: Option<Arc<MockServiceRegistry>>,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
    pub log_level: String,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            use_mocks: true,
            mock_registry: Some(Arc::new(MockServiceRegistry::new())),
            timeout_ms: 5000,
            retry_attempts: 3,
            log_level: "debug".to_string(),
        }
    }
}

/// Test context for integration tests
pub struct TestContext {
    pub config: IntegrationTestConfig,
    pub registry: Arc<MockServiceRegistry>,
    pub captured_logs: Arc<Mutex<Vec<String>>>,
}

impl TestContext {
    pub fn new(config: IntegrationTestConfig) -> Self {
        let registry = config
            .mock_registry
            .clone()
            .unwrap_or_else(|| Arc::new(MockServiceRegistry::new()));

        Self {
            config,
            registry,
            captured_logs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn capture_log(&self, message: String) {
        let mut logs = self.captured_logs.lock().await;
        logs.push(message);
    }

    pub async fn get_logs(&self) -> Vec<String> {
        let logs = self.captured_logs.lock().await;
        logs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_registry() {
        let registry = MockServiceRegistry::new();

        // Test registration and verification
        // Actual mock implementations would be added in mock_services module
    }

    #[tokio::test]
    async fn test_expectation_times() {
        let once = ExpectationTimes::Once;
        let exactly_5 = ExpectationTimes::Exactly(5);
        let at_least_2 = ExpectationTimes::AtLeast(2);
        let at_most_10 = ExpectationTimes::AtMost(10);
        let any = ExpectationTimes::Any;

        // Verify expectation matching logic
        // Implementation would be in the mock services
    }
}
