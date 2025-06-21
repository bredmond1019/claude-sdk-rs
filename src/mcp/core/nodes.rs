use crate::mcp::core::error::WorkflowError;
use crate::mcp::core::task::TaskContext;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[async_trait]
pub trait Node: Send + Sync {
    async fn execute(&self, input: Value, context: &TaskContext) -> Result<Value, WorkflowError>;

    fn name(&self) -> &str;

    fn description(&self) -> &str {
        ""
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({})
    }

    fn output_schema(&self) -> Value {
        serde_json::json!({})
    }
}

pub mod external_mcp_client {
    use super::*;
    use crate::mcp::core::external_mcp::BaseExternalMCPClient;
    use std::sync::Arc;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RetryConfig {
        pub max_attempts: u32,
        pub initial_delay: Duration,
        pub max_delay: Duration,
        pub exponential_base: f64,
    }

    impl Default for RetryConfig {
        fn default() -> Self {
            Self {
                max_attempts: 3,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(10),
                exponential_base: 2.0,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AuthConfig {
        pub api_key: Option<String>,
        pub token: Option<String>,
        pub headers: Option<std::collections::HashMap<String, String>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ExternalMCPConfig {
        pub name: String,
        pub auth: Option<AuthConfig>,
        pub retry: RetryConfig,
        pub timeout: Duration,
    }

    pub struct ExternalMCPClientNode {
        pub client: Arc<tokio::sync::Mutex<BaseExternalMCPClient>>,
        pub tool_name: String,
    }

    #[async_trait]
    impl Node for ExternalMCPClientNode {
        async fn execute(
            &self,
            input: Value,
            _context: &TaskContext,
        ) -> Result<Value, WorkflowError> {
            use std::collections::HashMap;

            let args_map = if let Value::Object(map) = input {
                Some(map.into_iter().collect::<HashMap<String, Value>>())
            } else {
                None
            };

            let mut client = self.client.lock().await;
            let result = client.execute_tool(&self.tool_name, args_map).await?;

            Ok(serde_json::to_value(result)?)
        }

        fn name(&self) -> &str {
            &self.tool_name
        }
    }
}

pub mod agent {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ModelProvider {
        Anthropic,
        OpenAI,
        Google,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AgentConfig {
        pub model: String,
        pub provider: ModelProvider,
        pub temperature: Option<f32>,
        pub max_tokens: Option<u32>,
        pub system_prompt: Option<String>,
    }
}
