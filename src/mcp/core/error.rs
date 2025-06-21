use thiserror::Error;

#[derive(Error, Debug)]
pub enum WorkflowError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Transport error: {0}")]
    TransportError(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("Service error: {0}")]
    ServiceError(String),

    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),

    #[error("Circuit breaker open")]
    CircuitBreakerOpen,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Node execution error: {0}")]
    NodeExecutionError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("MCP error: {message}")]
    MCPError { message: String },

    #[error("MCP protocol error: {message}")]
    MCPProtocolError { message: String },

    #[error("MCP transport error: {message}")]
    MCPTransportError { message: String },

    #[error("MCP connection error: {message}")]
    MCPConnectionError { message: String },

    #[error("Not found: {resource}")]
    NotFound { resource: String },

    #[error("External service error: {service} - {message}")]
    ExternalServiceError { service: String, message: String },

    #[error("Authentication error: {message}")]
    AuthenticationError { message: String },

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl From<serde_json::Error> for WorkflowError {
    fn from(err: serde_json::Error) -> Self {
        WorkflowError::SerializationError(err.to_string())
    }
}

impl From<crate::mcp::transport::TransportError> for WorkflowError {
    fn from(err: crate::mcp::transport::TransportError) -> Self {
        WorkflowError::TransportError(err.to_string())
    }
}

pub mod circuit_breaker {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::RwLock;

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum CircuitState {
        Closed,
        Open,
        HalfOpen,
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct CircuitBreakerConfig {
        pub failure_threshold: u32,
        pub success_threshold: u32,
        pub timeout: Duration,
        pub half_open_requests: u32,
    }

    impl Default for CircuitBreakerConfig {
        fn default() -> Self {
            Self {
                failure_threshold: 5,
                success_threshold: 3,
                timeout: Duration::from_secs(60),
                half_open_requests: 3,
            }
        }
    }

    #[derive(Debug)]
    pub struct CircuitBreaker {
        config: CircuitBreakerConfig,
        state: CircuitState,
        failure_count: u32,
        success_count: u32,
        last_failure_time: Option<Instant>,
        half_open_attempts: u32,
    }

    impl CircuitBreaker {
        pub fn new(config: CircuitBreakerConfig) -> Self {
            Self {
                config,
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                half_open_attempts: 0,
            }
        }

        pub fn state(&self) -> CircuitState {
            self.state
        }

        pub fn record_success(&mut self) {
            match self.state {
                CircuitState::Closed => {
                    self.failure_count = 0;
                }
                CircuitState::HalfOpen => {
                    self.success_count += 1;
                    if self.success_count >= self.config.success_threshold {
                        self.state = CircuitState::Closed;
                        self.failure_count = 0;
                        self.success_count = 0;
                        self.half_open_attempts = 0;
                    }
                }
                CircuitState::Open => {}
            }
        }

        pub fn record_failure(&mut self) {
            match self.state {
                CircuitState::Closed => {
                    self.failure_count += 1;
                    if self.failure_count >= self.config.failure_threshold {
                        self.state = CircuitState::Open;
                        self.last_failure_time = Some(Instant::now());
                    }
                }
                CircuitState::HalfOpen => {
                    self.state = CircuitState::Open;
                    self.last_failure_time = Some(Instant::now());
                    self.success_count = 0;
                    self.half_open_attempts = 0;
                }
                CircuitState::Open => {}
            }
        }

        pub fn should_allow_request(&mut self) -> bool {
            match self.state {
                CircuitState::Closed => true,
                CircuitState::Open => {
                    let Some(last_failure) = self.last_failure_time else {
                        return false;
                    };

                    if last_failure.elapsed() >= self.config.timeout {
                        self.state = CircuitState::HalfOpen;
                        self.half_open_attempts = 0;
                        self.success_count = 0;
                        true
                    } else {
                        false
                    }
                }
                CircuitState::HalfOpen => {
                    if self.half_open_attempts < self.config.half_open_requests {
                        self.half_open_attempts += 1;
                        true
                    } else {
                        false
                    }
                }
            }
        }

        pub fn reset(&mut self) {
            self.state = CircuitState::Closed;
            self.failure_count = 0;
            self.success_count = 0;
            self.half_open_attempts = 0;
            self.last_failure_time = None;
        }
    }

    #[derive(Clone)]
    pub struct CircuitBreakerRegistry {
        breakers: Arc<RwLock<HashMap<String, Arc<RwLock<CircuitBreaker>>>>>,
    }

    impl CircuitBreakerRegistry {
        pub fn new() -> Self {
            Self {
                breakers: Arc::new(RwLock::new(HashMap::new())),
            }
        }

        pub async fn get_or_create(
            &self,
            key: &str,
            config: CircuitBreakerConfig,
        ) -> Arc<RwLock<CircuitBreaker>> {
            let mut breakers = self.breakers.write().await;
            breakers
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(RwLock::new(CircuitBreaker::new(config))))
                .clone()
        }

        pub async fn get(&self, key: &str) -> Option<Arc<RwLock<CircuitBreaker>>> {
            let breakers = self.breakers.read().await;
            breakers.get(key).cloned()
        }
    }
}
