use crate::cli::error::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Configuration settings for claude-sdk-rs CLI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default timeout for operations in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Default verbosity level
    #[serde(default)]
    pub verbose: bool,

    /// Default quiet mode
    #[serde(default)]
    pub quiet: bool,

    /// Custom data directory (if not using system default)
    pub data_dir: Option<PathBuf>,

    /// Default session settings
    #[serde(default)]
    pub session: SessionDefaults,

    /// Default Claude configuration
    #[serde(default)]
    pub claude: ClaudeDefaults,

    /// Output formatting preferences
    #[serde(default)]
    pub output: OutputDefaults,

    /// Analytics settings
    #[serde(default)]
    pub analytics: AnalyticsDefaults,
}

/// Default session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDefaults {
    /// Default model to use for new sessions
    pub model: Option<String>,

    /// Default system prompt for new sessions
    pub system_prompt: Option<String>,

    /// Default allowed tools
    pub allowed_tools: Option<Vec<String>>,

    /// Auto-archive sessions after this many days of inactivity
    pub auto_archive_days: Option<u32>,
}

/// Default Claude AI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeDefaults {
    /// Default streaming format
    #[serde(default)]
    pub stream_format: String,

    /// Default max tokens
    pub max_tokens: Option<usize>,

    /// Default timeout for Claude API calls
    pub api_timeout_secs: Option<u64>,
}

/// Output formatting defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputDefaults {
    /// Enable colored output by default
    #[serde(default = "default_true")]
    pub color: bool,

    /// Show progress indicators by default
    #[serde(default = "default_true")]
    pub progress: bool,

    /// Default output format for commands
    #[serde(default)]
    pub format: String,

    /// Show timestamps in output
    #[serde(default)]
    pub timestamps: bool,
}

/// Analytics configuration defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsDefaults {
    /// Enable analytics collection
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Retention period for analytics data in days
    #[serde(default = "default_analytics_retention")]
    pub retention_days: u32,

    /// Enable automatic report generation
    #[serde(default)]
    pub auto_reports: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            timeout_secs: default_timeout(),
            verbose: false,
            quiet: false,
            data_dir: None,
            session: SessionDefaults::default(),
            claude: ClaudeDefaults::default(),
            output: OutputDefaults::default(),
            analytics: AnalyticsDefaults::default(),
        }
    }
}

impl Default for SessionDefaults {
    fn default() -> Self {
        Self {
            model: None,
            system_prompt: None,
            allowed_tools: None,
            auto_archive_days: Some(30),
        }
    }
}

impl Default for ClaudeDefaults {
    fn default() -> Self {
        Self {
            stream_format: "text".to_string(),
            max_tokens: None,
            api_timeout_secs: Some(30),
        }
    }
}

impl Default for OutputDefaults {
    fn default() -> Self {
        Self {
            color: default_true(),
            progress: default_true(),
            format: "pretty".to_string(),
            timestamps: false,
        }
    }
}

impl Default for AnalyticsDefaults {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            retention_days: default_analytics_retention(),
            auto_reports: false,
        }
    }
}

impl Config {
    /// Load configuration from file, with fallback to defaults
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            // Create default config file
            let config = Self::default();
            config.save_to_file(path)?;
            return Ok(config);
        }

        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content).map_err(|e| {
            crate::cli::error::InteractiveError::Configuration(format!(
                "Failed to parse config file: {}",
                e
            ))
        })?;

        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self).map_err(|e| {
            crate::cli::error::InteractiveError::Configuration(format!(
                "Failed to serialize config: {}",
                e
            ))
        })?;

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the default config file path
    pub fn default_path() -> Result<PathBuf> {
        let data_dir = crate::cli::ensure_data_dir()?;
        Ok(data_dir.join("config.toml"))
    }

    /// Load configuration from the default location
    pub fn load_default() -> Result<Self> {
        let path = Self::default_path()?;
        Self::load_from_file(path)
    }

    /// Merge with command-line arguments, giving priority to CLI args
    pub fn merge_with_cli_args(mut self, cli_args: &crate::cli::cli::Cli) -> Self {
        // CLI args override config file settings
        if cli_args.verbose {
            self.verbose = true;
        }
        if cli_args.quiet {
            self.quiet = true;
        }
        if cli_args.timeout != default_timeout() {
            self.timeout_secs = cli_args.timeout;
        }
        if let Some(ref data_dir) = cli_args.data_dir {
            self.data_dir = Some(data_dir.clone());
        }

        self
    }
}

// Helper functions for default values
fn default_timeout() -> u64 {
    60
}

fn default_true() -> bool {
    true
}

fn default_analytics_retention() -> u32 {
    90
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.timeout_secs, 60);
        assert!(!config.verbose);
        assert!(!config.quiet);
        assert!(config.output.color);
        assert!(config.analytics.enabled);
    }

    #[test]
    fn test_config_save_load() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        let mut config = Config::default();
        config.verbose = true;
        config.timeout_secs = 120;
        config.session.model = Some("claude-3-opus-20240229".to_string());

        // Save config
        config.save_to_file(&config_path).unwrap();

        // Load config
        let loaded_config = Config::load_from_file(&config_path).unwrap();

        assert_eq!(loaded_config.verbose, true);
        assert_eq!(loaded_config.timeout_secs, 120);
        assert_eq!(
            loaded_config.session.model,
            Some("claude-3-opus-20240229".to_string())
        );
    }

    #[test]
    fn test_config_file_creation() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        // Should create default config file if it doesn't exist
        let config = Config::load_from_file(&config_path).unwrap();

        assert!(config_path.exists());
        assert_eq!(config.timeout_secs, default_timeout());
    }
}
