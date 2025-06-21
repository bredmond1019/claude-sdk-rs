// Test harness for integration testing

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::{IntegrationTestConfig, MockServiceRegistry, TestContext};
use crate::mcp::clients::MCPClient;
use crate::mcp::core::error::WorkflowError;

/// Test harness for running integration tests
pub struct TestHarness {
    context: Arc<TestContext>,
    clients: Arc<Mutex<HashMap<String, Box<dyn MCPClient + Send + Sync>>>>,
    cleanup_tasks: Arc<Mutex<Vec<Box<dyn CleanupTask>>>>,
}

#[async_trait::async_trait]
trait CleanupTask: Send + Sync {
    async fn cleanup(&self) -> Result<(), String>;
}

impl TestHarness {
    pub fn new(config: IntegrationTestConfig) -> Self {
        Self {
            context: Arc::new(TestContext::new(config)),
            clients: Arc::new(Mutex::new(HashMap::new())),
            cleanup_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get the test context
    pub fn context(&self) -> Arc<TestContext> {
        self.context.clone()
    }

    /// Register an MCP client for testing
    pub async fn register_client(&self, name: String, client: Box<dyn MCPClient + Send + Sync>) {
        let mut clients = self.clients.lock().await;
        clients.insert(name, client);
    }

    /// Get a registered client
    pub async fn get_client(&self, name: &str) -> Option<Box<dyn MCPClient + Send + Sync>> {
        let clients = self.clients.lock().await;
        // In a real implementation, we'd return a reference or use Arc
        None // Placeholder
    }

    /// Run a test with automatic setup and cleanup
    pub async fn run_test<F, Fut, T>(&self, test_fn: F) -> Result<T, String>
    where
        F: FnOnce(Arc<TestContext>) -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        // Setup phase
        self.setup().await?;

        // Run test
        let result = test_fn(self.context.clone()).await;

        // Cleanup phase
        self.cleanup().await?;

        // Verify all mocks
        if let Err(errors) = self.context.registry.verify_all().await {
            return Err(format!("Mock verification failed:\n{}", errors.join("\n")));
        }

        result
    }

    /// Setup test environment
    async fn setup(&self) -> Result<(), String> {
        // Initialize logging
        if self.context.config.log_level == "debug" {
            self.context
                .capture_log("Test harness initialized in debug mode".to_string())
                .await;
        }

        // Connect all registered clients
        let mut clients = self.clients.lock().await;
        for (name, client) in clients.iter_mut() {
            client
                .connect()
                .await
                .map_err(|e| format!("Failed to connect client {}: {:?}", name, e))?;

            client
                .initialize("test-harness", "1.0.0")
                .await
                .map_err(|e| format!("Failed to initialize client {}: {:?}", name, e))?;
        }

        Ok(())
    }

    /// Cleanup test environment
    async fn cleanup(&self) -> Result<(), String> {
        let mut errors = Vec::new();

        // Disconnect all clients
        let mut clients = self.clients.lock().await;
        for (name, client) in clients.iter_mut() {
            if let Err(e) = client.disconnect().await {
                errors.push(format!("Failed to disconnect client {}: {:?}", name, e));
            }
        }

        // Run cleanup tasks
        let cleanup_tasks = self.cleanup_tasks.lock().await;
        for task in cleanup_tasks.iter() {
            if let Err(e) = task.cleanup().await {
                errors.push(e);
            }
        }

        // Reset all mocks
        self.context.registry.reset_all().await;

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }

    /// Add a cleanup task
    pub async fn add_cleanup<F>(&self, cleanup_fn: F)
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        struct SimpleCleanup<F> {
            cleanup_fn: F,
        }

        #[async_trait::async_trait]
        impl<F> CleanupTask for SimpleCleanup<F>
        where
            F: Fn() -> Result<(), String> + Send + Sync,
        {
            async fn cleanup(&self) -> Result<(), String> {
                (self.cleanup_fn)()
            }
        }

        let mut tasks = self.cleanup_tasks.lock().await;
        tasks.push(Box::new(SimpleCleanup { cleanup_fn }));
    }
}

/// Builder for creating test scenarios
pub struct TestScenarioBuilder {
    name: String,
    config: IntegrationTestConfig,
    setup_steps: Vec<Box<dyn SetupStep>>,
    assertions: Vec<Box<dyn Assertion>>,
}

#[async_trait::async_trait]
trait SetupStep: Send + Sync {
    async fn execute(&self, context: &TestContext) -> Result<(), String>;
}

#[async_trait::async_trait]
trait Assertion: Send + Sync {
    async fn assert(&self, context: &TestContext) -> Result<(), String>;
}

impl TestScenarioBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: IntegrationTestConfig::default(),
            setup_steps: Vec::new(),
            assertions: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: IntegrationTestConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_setup<F>(mut self, setup_fn: F) -> Self
    where
        F: Fn(&TestContext) -> Result<(), String> + Send + Sync + 'static,
    {
        struct SimpleSetup<F> {
            setup_fn: F,
        }

        #[async_trait::async_trait]
        impl<F> SetupStep for SimpleSetup<F>
        where
            F: Fn(&TestContext) -> Result<(), String> + Send + Sync,
        {
            async fn execute(&self, context: &TestContext) -> Result<(), String> {
                (self.setup_fn)(context)
            }
        }

        self.setup_steps.push(Box::new(SimpleSetup { setup_fn }));
        self
    }

    pub fn assert<F>(mut self, assertion_fn: F) -> Self
    where
        F: Fn(&TestContext) -> Result<(), String> + Send + Sync + 'static,
    {
        struct SimpleAssertion<F> {
            assertion_fn: F,
        }

        #[async_trait::async_trait]
        impl<F> Assertion for SimpleAssertion<F>
        where
            F: Fn(&TestContext) -> Result<(), String> + Send + Sync,
        {
            async fn assert(&self, context: &TestContext) -> Result<(), String> {
                (self.assertion_fn)(context)
            }
        }

        self.assertions
            .push(Box::new(SimpleAssertion { assertion_fn }));
        self
    }

    pub async fn run(self) -> Result<(), String> {
        let harness = TestHarness::new(self.config);

        harness
            .run_test(|context| async move {
                // Execute setup steps
                for step in &self.setup_steps {
                    step.execute(&context).await?;
                }

                // Run assertions
                for assertion in &self.assertions {
                    assertion.assert(&context).await?;
                }

                Ok(())
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::testing::MockMCPClient;

    #[tokio::test]
    async fn test_harness_basic_flow() {
        let config = IntegrationTestConfig::default();
        let harness = TestHarness::new(config);

        // Register a mock client
        let client = Box::new(MockMCPClient::new());
        harness
            .register_client("test-client".to_string(), client)
            .await;

        // Run a simple test
        let result = harness
            .run_test(|context| async move {
                context.capture_log("Test running".to_string()).await;
                Ok(())
            })
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scenario_builder() {
        let result = TestScenarioBuilder::new("test scenario")
            .with_setup(|_context| {
                // Setup logic
                Ok(())
            })
            .assert(|_context| {
                // Assertion logic
                Ok(())
            })
            .run()
            .await;

        assert!(result.is_ok());
    }
}
