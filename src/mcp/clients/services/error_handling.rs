use crate::mcp::core::error::WorkflowError;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, warn};

/// Retry policy for external service calls
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_backoff: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            exponential_backoff: true,
        }
    }
}

/// Error handling configuration for external services
#[derive(Debug, Clone)]
pub struct ErrorHandlingConfig {
    pub retry_policy: RetryPolicy,
    pub circuit_breaker_enabled: bool,
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub log_errors: bool,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            retry_policy: RetryPolicy::default(),
            circuit_breaker_enabled: true,
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            log_errors: true,
        }
    }
}

/// Enhanced error handling for external service operations
#[derive(Debug)]
pub struct ErrorHandler {
    config: ErrorHandlingConfig,
    consecutive_failures: u32,
    circuit_open_until: Option<std::time::Instant>,
}

impl ErrorHandler {
    pub fn new(config: ErrorHandlingConfig) -> Self {
        Self {
            config,
            consecutive_failures: 0,
            circuit_open_until: None,
        }
    }

    /// Check if circuit breaker is open
    pub fn is_circuit_open(&self) -> bool {
        if let Some(open_until) = self.circuit_open_until {
            std::time::Instant::now() < open_until
        } else {
            false
        }
    }

    /// Record a successful operation
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.circuit_open_until = None;
    }

    /// Record a failed operation
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;

        if self.config.circuit_breaker_enabled
            && self.consecutive_failures >= self.config.failure_threshold
        {
            self.circuit_open_until =
                Some(std::time::Instant::now() + self.config.recovery_timeout);
            if self.config.log_errors {
                error!(
                    "Circuit breaker opened after {} consecutive failures. Will retry after {:?}",
                    self.consecutive_failures, self.config.recovery_timeout
                );
            }
        }
    }

    /// Execute an operation with retry logic
    pub async fn execute_with_retry<F, T, Fut>(
        &mut self,
        operation: F,
        service_name: &str,
    ) -> Result<T, WorkflowError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, WorkflowError>>,
    {
        if self.is_circuit_open() {
            return Err(WorkflowError::ExternalServiceError {
                service: service_name.to_string(),
                message: "Circuit breaker is open due to repeated failures".to_string(),
            });
        }

        let mut attempt = 0;
        let mut delay = self.config.retry_policy.initial_delay;

        loop {
            attempt += 1;

            match operation().await {
                Ok(result) => {
                    self.record_success();
                    return Ok(result);
                }
                Err(err) => {
                    let is_retryable = match &err {
                        WorkflowError::ExternalServiceError { .. } => true,
                        WorkflowError::TimeoutError(_) => true,
                        WorkflowError::ConnectionError(_) => true,
                        WorkflowError::AuthenticationError { .. } => false,
                        WorkflowError::NotFound { .. } => false,
                        _ => false,
                    };

                    if !is_retryable || attempt >= self.config.retry_policy.max_attempts {
                        self.record_failure();
                        if self.config.log_errors {
                            error!(
                                "Failed to execute operation for {} after {} attempts: {:?}",
                                service_name, attempt, err
                            );
                        }
                        return Err(err);
                    }

                    if self.config.log_errors {
                        warn!(
                            "Attempt {} failed for {}: {:?}. Retrying after {:?}",
                            attempt, service_name, err, delay
                        );
                    }

                    tokio::time::sleep(delay).await;

                    // Update delay for next attempt
                    if self.config.retry_policy.exponential_backoff {
                        delay = std::cmp::min(delay * 2, self.config.retry_policy.max_delay);
                    }
                }
            }
        }
    }
}

/// Fallback strategies for when services are unavailable
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Return a default value
    ReturnDefault(serde_json::Value),
    /// Use cached data if available
    UseCache,
    /// Fail immediately
    FailFast,
    /// Return empty results
    ReturnEmpty,
}

/// Service health status
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceHealth {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
}

/// Health check result
#[derive(Debug)]
pub struct HealthCheckResult {
    pub service: String,
    pub status: ServiceHealth,
    pub last_check: std::time::Instant,
    pub response_time: Option<Duration>,
}

/// Trait for services that support health checks
#[async_trait::async_trait]
pub trait HealthCheckable {
    async fn health_check(&self) -> HealthCheckResult;
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub requests_per_second: f64,
    pub burst_size: u32,
}

/// Rate limiter implementation
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    last_request: Option<std::time::Instant>,
    tokens: f64,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        let tokens = config.burst_size as f64;
        Self {
            config,
            last_request: None,
            tokens,
        }
    }

    /// Wait if necessary to respect rate limits
    pub async fn wait_if_needed(&mut self) {
        let now = std::time::Instant::now();

        if let Some(last) = self.last_request {
            let elapsed = now.duration_since(last).as_secs_f64();
            self.tokens = (self.tokens + elapsed * self.config.requests_per_second)
                .min(self.config.burst_size as f64);
        }

        if self.tokens < 1.0 {
            let wait_time = (1.0 - self.tokens) / self.config.requests_per_second;
            tokio::time::sleep(Duration::from_secs_f64(wait_time)).await;
            self.tokens = 1.0;
        }

        self.tokens -= 1.0;
        self.last_request = Some(now);
    }
}

/// Common error transformations for external services
pub fn transform_http_error(status: reqwest::StatusCode, service: &str) -> WorkflowError {
    match status {
        reqwest::StatusCode::UNAUTHORIZED => WorkflowError::AuthenticationError {
            message: format!("{} authentication failed", service),
        },
        reqwest::StatusCode::FORBIDDEN => WorkflowError::AuthenticationError {
            message: format!("{} access forbidden", service),
        },
        reqwest::StatusCode::NOT_FOUND => WorkflowError::NotFound {
            resource: format!("{} resource", service),
        },
        reqwest::StatusCode::TOO_MANY_REQUESTS => WorkflowError::RateLimitExceeded,
        reqwest::StatusCode::SERVICE_UNAVAILABLE | reqwest::StatusCode::GATEWAY_TIMEOUT => {
            WorkflowError::ExternalServiceError {
                service: service.to_string(),
                message: "Service temporarily unavailable".to_string(),
            }
        }
        _ => WorkflowError::ExternalServiceError {
            service: service.to_string(),
            message: format!("HTTP error: {}", status),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = ErrorHandlingConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            ..Default::default()
        };

        let mut handler = ErrorHandler::new(config);

        // Record failures to open circuit
        handler.record_failure();
        assert!(!handler.is_circuit_open());

        handler.record_failure();
        assert!(handler.is_circuit_open());

        // Wait for recovery timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(!handler.is_circuit_open());
    }

    #[tokio::test]
    async fn test_retry_logic() {
        let config = ErrorHandlingConfig::default();
        let mut handler = ErrorHandler::new(config);

        let attempt_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = handler
            .execute_with_retry(
                move || {
                    let count =
                        attempt_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    async move {
                        if count < 3 {
                            Err(WorkflowError::ExternalServiceError {
                                service: "test".to_string(),
                                message: "temporary failure".to_string(),
                            })
                        } else {
                            Ok("success")
                        }
                    }
                },
                "test_service",
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(attempt_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let config = RateLimitConfig {
            requests_per_second: 10.0,
            burst_size: 2,
        };

        let mut limiter = RateLimiter::new(config);

        let start = std::time::Instant::now();

        // First two requests should be immediate (burst)
        limiter.wait_if_needed().await;
        limiter.wait_if_needed().await;

        // Third request should wait
        limiter.wait_if_needed().await;

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(90)); // Should wait ~100ms
    }
}
