// Service configuration for external API integrations

use serde::{Deserialize, Serialize};
use std::env;

use crate::mcp::core::error::WorkflowError;

/// Configuration for all external services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub helpscout: HelpScoutConfig,
    pub notion: NotionConfig,
    pub slack: SlackConfig,
    pub general: GeneralServiceConfig,
}

/// HelpScout service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelpScoutConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub rate_limit: RateLimitConfig,
    pub retry: RetryConfig,
}

/// Notion service configuration  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotionConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub base_url: String,
    pub workspace_id: Option<String>,
    pub default_database_id: Option<String>,
    pub timeout_seconds: u64,
    pub rate_limit: RateLimitConfig,
    pub retry: RetryConfig,
}

/// Slack service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    pub enabled: bool,
    pub bot_token: Option<String>,
    pub app_token: Option<String>,
    pub base_url: String,
    pub default_channel: Option<String>,
    pub timeout_seconds: u64,
    pub rate_limit: RateLimitConfig,
    pub retry: RetryConfig,
}

/// General configuration applied to all services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralServiceConfig {
    pub use_mocks: bool,
    pub log_requests: bool,
    pub log_responses: bool,
    pub validate_responses: bool,
    pub cache_enabled: bool,
    pub cache_ttl_seconds: u64,
    pub metrics_enabled: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_second: f64,
    pub burst_size: u32,
    pub backoff_multiplier: f64,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub enabled: bool,
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_backoff: bool,
    pub jitter: bool,
}

impl ServiceConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, WorkflowError> {
        Ok(ServiceConfig {
            helpscout: HelpScoutConfig::from_env()?,
            notion: NotionConfig::from_env()?,
            slack: SlackConfig::from_env()?,
            general: GeneralServiceConfig::from_env()?,
        })
    }

    /// Load configuration from a file
    pub fn from_file(path: &str) -> Result<Self, WorkflowError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            WorkflowError::ConfigurationError(format!("Failed to read config file: {}", e))
        })?;

        let config: ServiceConfig = serde_json::from_str(&content).map_err(|e| {
            WorkflowError::ConfigurationError(format!("Failed to parse config: {}", e))
        })?;

        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), WorkflowError> {
        let content = serde_json::to_string_pretty(self).map_err(|e| {
            WorkflowError::ConfigurationError(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(path, content).map_err(|e| {
            WorkflowError::ConfigurationError(format!("Failed to write config file: {}", e))
        })?;

        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), WorkflowError> {
        // Validate HelpScout config
        if self.helpscout.enabled && self.helpscout.api_key.is_none() {
            return Err(WorkflowError::ConfigurationError(
                "HelpScout is enabled but no API key provided".to_string(),
            ));
        }

        // Validate Notion config
        if self.notion.enabled && self.notion.api_key.is_none() {
            return Err(WorkflowError::ConfigurationError(
                "Notion is enabled but no API key provided".to_string(),
            ));
        }

        // Validate Slack config
        if self.slack.enabled && self.slack.bot_token.is_none() {
            return Err(WorkflowError::ConfigurationError(
                "Slack is enabled but no bot token provided".to_string(),
            ));
        }

        // Validate rate limits
        for (service, rate_limit) in [
            ("HelpScout", &self.helpscout.rate_limit),
            ("Notion", &self.notion.rate_limit),
            ("Slack", &self.slack.rate_limit),
        ] {
            if rate_limit.enabled && rate_limit.requests_per_second <= 0.0 {
                return Err(WorkflowError::ConfigurationError(format!(
                    "{} rate limit requests_per_second must be positive",
                    service
                )));
            }
        }

        Ok(())
    }
}

impl HelpScoutConfig {
    fn from_env() -> Result<Self, WorkflowError> {
        Ok(HelpScoutConfig {
            enabled: get_env_bool("HELPSCOUT_ENABLED", false),
            api_key: env::var("HELPSCOUT_API_KEY").ok(),
            base_url: env::var("HELPSCOUT_BASE_URL")
                .unwrap_or_else(|_| "https://api.helpscout.net".to_string()),
            timeout_seconds: get_env_u64("HELPSCOUT_TIMEOUT_SECONDS", 30),
            rate_limit: RateLimitConfig {
                enabled: get_env_bool("HELPSCOUT_RATE_LIMIT_ENABLED", true),
                requests_per_second: get_env_f64("HELPSCOUT_RATE_LIMIT_RPS", 10.0),
                burst_size: get_env_u32("HELPSCOUT_RATE_LIMIT_BURST", 5),
                backoff_multiplier: get_env_f64("HELPSCOUT_RATE_LIMIT_BACKOFF", 2.0),
            },
            retry: RetryConfig {
                enabled: get_env_bool("HELPSCOUT_RETRY_ENABLED", true),
                max_attempts: get_env_u32("HELPSCOUT_RETRY_MAX_ATTEMPTS", 3),
                initial_delay_ms: get_env_u64("HELPSCOUT_RETRY_INITIAL_DELAY_MS", 1000),
                max_delay_ms: get_env_u64("HELPSCOUT_RETRY_MAX_DELAY_MS", 30000),
                exponential_backoff: get_env_bool("HELPSCOUT_RETRY_EXPONENTIAL", true),
                jitter: get_env_bool("HELPSCOUT_RETRY_JITTER", true),
            },
        })
    }
}

impl NotionConfig {
    fn from_env() -> Result<Self, WorkflowError> {
        Ok(NotionConfig {
            enabled: get_env_bool("NOTION_ENABLED", false),
            api_key: env::var("NOTION_API_KEY").ok(),
            base_url: env::var("NOTION_BASE_URL")
                .unwrap_or_else(|_| "https://api.notion.com".to_string()),
            workspace_id: env::var("NOTION_WORKSPACE_ID").ok(),
            default_database_id: env::var("NOTION_DEFAULT_DATABASE_ID").ok(),
            timeout_seconds: get_env_u64("NOTION_TIMEOUT_SECONDS", 30),
            rate_limit: RateLimitConfig {
                enabled: get_env_bool("NOTION_RATE_LIMIT_ENABLED", true),
                requests_per_second: get_env_f64("NOTION_RATE_LIMIT_RPS", 3.0),
                burst_size: get_env_u32("NOTION_RATE_LIMIT_BURST", 1),
                backoff_multiplier: get_env_f64("NOTION_RATE_LIMIT_BACKOFF", 2.0),
            },
            retry: RetryConfig {
                enabled: get_env_bool("NOTION_RETRY_ENABLED", true),
                max_attempts: get_env_u32("NOTION_RETRY_MAX_ATTEMPTS", 3),
                initial_delay_ms: get_env_u64("NOTION_RETRY_INITIAL_DELAY_MS", 1000),
                max_delay_ms: get_env_u64("NOTION_RETRY_MAX_DELAY_MS", 30000),
                exponential_backoff: get_env_bool("NOTION_RETRY_EXPONENTIAL", true),
                jitter: get_env_bool("NOTION_RETRY_JITTER", true),
            },
        })
    }
}

impl SlackConfig {
    fn from_env() -> Result<Self, WorkflowError> {
        Ok(SlackConfig {
            enabled: get_env_bool("SLACK_ENABLED", false),
            bot_token: env::var("SLACK_BOT_TOKEN").ok(),
            app_token: env::var("SLACK_APP_TOKEN").ok(),
            base_url: env::var("SLACK_BASE_URL")
                .unwrap_or_else(|_| "https://slack.com/api".to_string()),
            default_channel: env::var("SLACK_DEFAULT_CHANNEL").ok(),
            timeout_seconds: get_env_u64("SLACK_TIMEOUT_SECONDS", 30),
            rate_limit: RateLimitConfig {
                enabled: get_env_bool("SLACK_RATE_LIMIT_ENABLED", true),
                requests_per_second: get_env_f64("SLACK_RATE_LIMIT_RPS", 1.0),
                burst_size: get_env_u32("SLACK_RATE_LIMIT_BURST", 1),
                backoff_multiplier: get_env_f64("SLACK_RATE_LIMIT_BACKOFF", 2.0),
            },
            retry: RetryConfig {
                enabled: get_env_bool("SLACK_RETRY_ENABLED", true),
                max_attempts: get_env_u32("SLACK_RETRY_MAX_ATTEMPTS", 3),
                initial_delay_ms: get_env_u64("SLACK_RETRY_INITIAL_DELAY_MS", 1000),
                max_delay_ms: get_env_u64("SLACK_RETRY_MAX_DELAY_MS", 30000),
                exponential_backoff: get_env_bool("SLACK_RETRY_EXPONENTIAL", true),
                jitter: get_env_bool("SLACK_RETRY_JITTER", true),
            },
        })
    }
}

impl GeneralServiceConfig {
    fn from_env() -> Result<Self, WorkflowError> {
        Ok(GeneralServiceConfig {
            use_mocks: get_env_bool("SERVICES_USE_MOCKS", false),
            log_requests: get_env_bool("SERVICES_LOG_REQUESTS", false),
            log_responses: get_env_bool("SERVICES_LOG_RESPONSES", false),
            validate_responses: get_env_bool("SERVICES_VALIDATE_RESPONSES", true),
            cache_enabled: get_env_bool("SERVICES_CACHE_ENABLED", false),
            cache_ttl_seconds: get_env_u64("SERVICES_CACHE_TTL_SECONDS", 300),
            metrics_enabled: get_env_bool("SERVICES_METRICS_ENABLED", true),
        })
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        ServiceConfig {
            helpscout: HelpScoutConfig::default(),
            notion: NotionConfig::default(),
            slack: SlackConfig::default(),
            general: GeneralServiceConfig::default(),
        }
    }
}

impl Default for HelpScoutConfig {
    fn default() -> Self {
        HelpScoutConfig {
            enabled: false,
            api_key: None,
            base_url: "https://api.helpscout.net".to_string(),
            timeout_seconds: 30,
            rate_limit: RateLimitConfig::default_helpscout(),
            retry: RetryConfig::default(),
        }
    }
}

impl Default for NotionConfig {
    fn default() -> Self {
        NotionConfig {
            enabled: false,
            api_key: None,
            base_url: "https://api.notion.com".to_string(),
            workspace_id: None,
            default_database_id: None,
            timeout_seconds: 30,
            rate_limit: RateLimitConfig::default_notion(),
            retry: RetryConfig::default(),
        }
    }
}

impl Default for SlackConfig {
    fn default() -> Self {
        SlackConfig {
            enabled: false,
            bot_token: None,
            app_token: None,
            base_url: "https://slack.com/api".to_string(),
            default_channel: None,
            timeout_seconds: 30,
            rate_limit: RateLimitConfig::default_slack(),
            retry: RetryConfig::default(),
        }
    }
}

impl Default for GeneralServiceConfig {
    fn default() -> Self {
        GeneralServiceConfig {
            use_mocks: false,
            log_requests: false,
            log_responses: false,
            validate_responses: true,
            cache_enabled: false,
            cache_ttl_seconds: 300,
            metrics_enabled: true,
        }
    }
}

impl RateLimitConfig {
    fn default_helpscout() -> Self {
        RateLimitConfig {
            enabled: true,
            requests_per_second: 10.0,
            burst_size: 5,
            backoff_multiplier: 2.0,
        }
    }

    fn default_notion() -> Self {
        RateLimitConfig {
            enabled: true,
            requests_per_second: 3.0,
            burst_size: 1,
            backoff_multiplier: 2.0,
        }
    }

    fn default_slack() -> Self {
        RateLimitConfig {
            enabled: true,
            requests_per_second: 1.0,
            burst_size: 1,
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            enabled: true,
            max_attempts: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            exponential_backoff: true,
            jitter: true,
        }
    }
}

// Helper functions for parsing environment variables
fn get_env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

fn get_env_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

fn get_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

fn get_env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

/// Builder for creating service configurations
pub struct ServiceConfigBuilder {
    config: ServiceConfig,
}

impl ServiceConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ServiceConfig::default(),
        }
    }

    pub fn helpscout_enabled(mut self, enabled: bool) -> Self {
        self.config.helpscout.enabled = enabled;
        self
    }

    pub fn helpscout_api_key(mut self, api_key: String) -> Self {
        self.config.helpscout.api_key = Some(api_key);
        self
    }

    pub fn notion_enabled(mut self, enabled: bool) -> Self {
        self.config.notion.enabled = enabled;
        self
    }

    pub fn notion_api_key(mut self, api_key: String) -> Self {
        self.config.notion.api_key = Some(api_key);
        self
    }

    pub fn slack_enabled(mut self, enabled: bool) -> Self {
        self.config.slack.enabled = enabled;
        self
    }

    pub fn slack_bot_token(mut self, bot_token: String) -> Self {
        self.config.slack.bot_token = Some(bot_token);
        self
    }

    pub fn use_mocks(mut self, use_mocks: bool) -> Self {
        self.config.general.use_mocks = use_mocks;
        self
    }

    pub fn build(self) -> Result<ServiceConfig, WorkflowError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_service_config_from_env() {
        // Set up environment
        env::set_var("HELPSCOUT_ENABLED", "true");
        env::set_var("HELPSCOUT_API_KEY", "test-key");
        env::set_var("NOTION_ENABLED", "true");
        env::set_var("NOTION_API_KEY", "test-key");
        env::set_var("SLACK_ENABLED", "true");
        env::set_var("SLACK_BOT_TOKEN", "test-token");

        let config = ServiceConfig::from_env().unwrap();

        assert!(config.helpscout.enabled);
        assert_eq!(config.helpscout.api_key, Some("test-key".to_string()));
        assert!(config.notion.enabled);
        assert_eq!(config.notion.api_key, Some("test-key".to_string()));
        assert!(config.slack.enabled);
        assert_eq!(config.slack.bot_token, Some("test-token".to_string()));

        // Cleanup
        env::remove_var("HELPSCOUT_ENABLED");
        env::remove_var("HELPSCOUT_API_KEY");
        env::remove_var("NOTION_ENABLED");
        env::remove_var("NOTION_API_KEY");
        env::remove_var("SLACK_ENABLED");
        env::remove_var("SLACK_BOT_TOKEN");
    }

    #[test]
    fn test_service_config_validation() {
        let mut config = ServiceConfig::default();

        // Should validate with all services disabled
        assert!(config.validate().is_ok());

        // Should fail with enabled service but no API key
        config.helpscout.enabled = true;
        assert!(config.validate().is_err());

        // Should pass with API key
        config.helpscout.api_key = Some("test-key".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_service_config_builder() {
        let config = ServiceConfigBuilder::new()
            .helpscout_enabled(true)
            .helpscout_api_key("test-key".to_string())
            .notion_enabled(true)
            .notion_api_key("test-key".to_string())
            .use_mocks(true)
            .build()
            .unwrap();

        assert!(config.helpscout.enabled);
        assert!(config.notion.enabled);
        assert!(config.general.use_mocks);
    }

    #[test]
    fn test_config_file_operations() {
        let config = ServiceConfig::default();
        let temp_file = "/tmp/test_service_config.json";

        // Save to file
        config.save_to_file(temp_file).unwrap();

        // Load from file
        let loaded_config = ServiceConfig::from_file(temp_file).unwrap();

        assert_eq!(config.helpscout.enabled, loaded_config.helpscout.enabled);
        assert_eq!(config.notion.enabled, loaded_config.notion.enabled);
        assert_eq!(config.slack.enabled, loaded_config.slack.enabled);

        // Cleanup
        std::fs::remove_file(temp_file).ok();
    }
}
