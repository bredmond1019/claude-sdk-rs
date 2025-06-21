/// Real API service implementations for external integrations
pub mod enhanced_services;
pub mod error_handling;
pub mod helpscout_api;
pub mod notion_api;
pub mod slack_api;

pub use enhanced_services::{
    EnhancedHelpScoutService, EnhancedNotionService, EnhancedSlackService, ServiceHealth,
    ServiceHealthMonitor, ServiceHealthStatus,
};
pub use error_handling::{
    ErrorHandler, ErrorHandlingConfig, RateLimitConfig, RateLimiter, RetryPolicy,
};
pub use helpscout_api::HelpScoutApiService;
pub use notion_api::NotionApiService;
pub use slack_api::SlackApiService;
