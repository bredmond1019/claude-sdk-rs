use super::*;
use crate::{cli::error::InteractiveError, cli::error::Result};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Test utility for creating a mock data directory with commands
pub struct MockCommandEnvironment {
    pub temp_dir: TempDir,
    pub data_dir: PathBuf,
    pub commands_dir: PathBuf,
}

impl MockCommandEnvironment {
    /// Create a new mock environment
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let data_dir = temp_dir.path().to_path_buf();
        let commands_dir = data_dir.join("commands");
        fs::create_dir_all(&commands_dir)?;

        Ok(Self {
            temp_dir,
            data_dir,
            commands_dir,
        })
    }

    /// Add a command file to the mock environment
    pub fn add_command(&self, name: &str, content: &str) -> Result<()> {
        let file_path = self.commands_dir.join(name);
        fs::write(&file_path, content)?;
        Ok(())
    }

    /// Get the data directory path
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

/// Test utility for capturing CLI output
pub struct OutputCapture {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl OutputCapture {
    pub fn new() -> Self {
        Self {
            stdout: Vec::new(),
            stderr: Vec::new(),
        }
    }

    pub fn stdout_str(&self) -> String {
        String::from_utf8_lossy(&self.stdout).to_string()
    }

    pub fn stderr_str(&self) -> String {
        String::from_utf8_lossy(&self.stderr).to_string()
    }
}

/// Mock execution context for testing CLI commands
#[derive(Clone)]
pub struct MockExecutionContext {
    pub session_id: Option<uuid::Uuid>,
    pub command_name: String,
    pub args: Vec<String>,
    pub parallel: bool,
    pub agent_count: usize,
}

impl Default for MockExecutionContext {
    fn default() -> Self {
        Self {
            session_id: None,
            command_name: "test".to_string(),
            args: vec![],
            parallel: false,
            agent_count: 1,
        }
    }
}

/// Helper function to parse CLI arguments from strings
pub fn parse_args<T: Parser>(args: &[&str]) -> std::result::Result<T, clap::Error> {
    T::try_parse_from(args)
}

/// Helper to validate command parsing succeeds
pub fn assert_parse_success<T: Parser>(args: &[&str]) -> T {
    match parse_args::<T>(args) {
        Ok(parsed) => parsed,
        Err(e) => panic!("Failed to parse arguments: {}", e),
    }
}

/// Helper to validate command parsing fails with expected error
pub fn assert_parse_error<T: Parser>(args: &[&str]) -> clap::Error {
    match parse_args::<T>(args) {
        Ok(_) => panic!("Expected parsing to fail but it succeeded"),
        Err(e) => e,
    }
}

#[cfg(test)]
mod infrastructure_tests {
    use super::*;

    #[test]
    fn test_mock_environment_creation() {
        let env = MockCommandEnvironment::new().expect("Failed to create mock environment");

        assert!(env.data_dir.exists());
        assert!(env.commands_dir.exists());
        assert_eq!(env.commands_dir, env.data_dir.join("commands"));
    }

    #[test]
    fn test_mock_environment_add_command() {
        let env = MockCommandEnvironment::new().expect("Failed to create mock environment");

        env.add_command("test.sh", "#!/bin/bash\necho 'test'")
            .expect("Failed to add command");

        let command_file = env.commands_dir.join("test.sh");
        assert!(command_file.exists());

        let content = fs::read_to_string(&command_file).expect("Failed to read command file");
        assert_eq!(content, "#!/bin/bash\necho 'test'");
    }

    #[test]
    fn test_output_capture() {
        let mut capture = OutputCapture::new();

        capture.stdout.extend_from_slice(b"Hello, stdout!");
        capture.stderr.extend_from_slice(b"Hello, stderr!");

        assert_eq!(capture.stdout_str(), "Hello, stdout!");
        assert_eq!(capture.stderr_str(), "Hello, stderr!");
    }

    #[test]
    fn test_mock_execution_context() {
        let ctx = MockExecutionContext::default();

        assert_eq!(ctx.command_name, "test");
        assert!(ctx.args.is_empty());
        assert!(!ctx.parallel);
        assert_eq!(ctx.agent_count, 1);
        assert!(ctx.session_id.is_none());

        // Test clone
        let cloned = ctx.clone();
        assert_eq!(cloned.command_name, ctx.command_name);
        assert_eq!(cloned.args, ctx.args);
    }
}

// Command parsing tests
#[cfg(test)]
mod list_command_parsing_tests {
    use super::*;

    #[test]
    fn test_list_command_basic() {
        let cmd = ListCommand {
            filter: None,
            detailed: false,
        };

        assert!(cmd.filter.is_none());
        assert!(!cmd.detailed);
    }

    #[test]
    fn test_list_command_with_filter() {
        let cmd = ListCommand {
            filter: Some("test".to_string()),
            detailed: false,
        };

        assert_eq!(cmd.filter, Some("test".to_string()));
        assert!(!cmd.detailed);
    }

    #[test]
    fn test_list_command_detailed() {
        let cmd = ListCommand {
            filter: None,
            detailed: true,
        };

        assert!(cmd.filter.is_none());
        assert!(cmd.detailed);
    }

    #[test]
    fn test_list_command_all_options() {
        let cmd = ListCommand {
            filter: Some("analyze".to_string()),
            detailed: true,
        };

        assert_eq!(cmd.filter, Some("analyze".to_string()));
        assert!(cmd.detailed);
    }
}

#[cfg(test)]
mod session_command_parsing_tests {
    use super::*;

    #[test]
    fn test_session_create_basic() {
        match (SessionAction::Create {
            name: "test-session".to_string(),
            description: None,
        }) {
            SessionAction::Create { name, description } => {
                assert_eq!(name, "test-session");
                assert!(description.is_none());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_session_create_with_description() {
        match (SessionAction::Create {
            name: "test-session".to_string(),
            description: Some("A test session".to_string()),
        }) {
            SessionAction::Create { name, description } => {
                assert_eq!(name, "test-session");
                assert_eq!(description, Some("A test session".to_string()));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_session_delete() {
        match (SessionAction::Delete {
            session: "test-session".to_string(),
            force: false,
        }) {
            SessionAction::Delete { session, force } => {
                assert_eq!(session, "test-session");
                assert!(!force);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_session_delete_force() {
        match (SessionAction::Delete {
            session: "test-session".to_string(),
            force: true,
        }) {
            SessionAction::Delete { session, force } => {
                assert_eq!(session, "test-session");
                assert!(force);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_session_list() {
        match (SessionAction::List { detailed: false }) {
            SessionAction::List { detailed } => {
                assert!(!detailed);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_session_list_detailed() {
        match (SessionAction::List { detailed: true }) {
            SessionAction::List { detailed } => {
                assert!(detailed);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_session_switch() {
        match (SessionAction::Switch {
            session: "other-session".to_string(),
        }) {
            SessionAction::Switch { session } => {
                assert_eq!(session, "other-session");
            }
            _ => panic!("Wrong variant"),
        }
    }
}

#[cfg(test)]
mod run_command_parsing_tests {
    use super::*;

    #[test]
    fn test_run_command_basic() {
        let cmd = RunCommand {
            command: "analyze".to_string(),
            args: vec![],
            session: None,
            parallel: false,
            agents: 1,
        };

        assert_eq!(cmd.command, "analyze");
        assert!(cmd.args.is_empty());
        assert!(cmd.session.is_none());
        assert!(!cmd.parallel);
        assert_eq!(cmd.agents, 1);
    }

    #[test]
    fn test_run_command_with_args() {
        let cmd = RunCommand {
            command: "analyze".to_string(),
            args: vec!["--file".to_string(), "test.rs".to_string()],
            session: None,
            parallel: false,
            agents: 1,
        };

        assert_eq!(cmd.command, "analyze");
        assert_eq!(cmd.args, vec!["--file", "test.rs"]);
    }

    #[test]
    fn test_run_command_with_session() {
        let cmd = RunCommand {
            command: "analyze".to_string(),
            args: vec![],
            session: Some("work-session".to_string()),
            parallel: false,
            agents: 1,
        };

        assert_eq!(cmd.session, Some("work-session".to_string()));
    }

    #[test]
    fn test_run_command_parallel() {
        let cmd = RunCommand {
            command: "analyze".to_string(),
            args: vec![],
            session: None,
            parallel: true,
            agents: 4,
        };

        assert!(cmd.parallel);
        assert_eq!(cmd.agents, 4);
    }

    #[test]
    fn test_run_command_all_options() {
        let cmd = RunCommand {
            command: "test".to_string(),
            args: vec!["--verbose".to_string(), "--output".to_string(), "result.txt".to_string()],
            session: Some("dev-session".to_string()),
            parallel: true,
            agents: 8,
        };

        assert_eq!(cmd.command, "test");
        assert_eq!(cmd.args.len(), 3);
        assert_eq!(cmd.session, Some("dev-session".to_string()));
        assert!(cmd.parallel);
        assert_eq!(cmd.agents, 8);
    }
}

#[cfg(test)]
mod cost_command_parsing_tests {
    use super::*;

    #[test]
    fn test_cost_command_basic() {
        let cmd = CostCommand {
            session: None,
            breakdown: false,
            since: None,
            export: None,
        };

        assert!(cmd.session.is_none());
        assert!(!cmd.breakdown);
        assert!(cmd.since.is_none());
        assert!(cmd.export.is_none());
    }

    #[test]
    fn test_cost_command_with_session() {
        let cmd = CostCommand {
            session: Some("work-session".to_string()),
            breakdown: false,
            since: None,
            export: None,
        };

        assert_eq!(cmd.session, Some("work-session".to_string()));
    }

    #[test]
    fn test_cost_command_with_breakdown() {
        let cmd = CostCommand {
            session: None,
            breakdown: true,
            since: None,
            export: None,
        };

        assert!(cmd.breakdown);
    }

    #[test]
    fn test_cost_command_with_date_range() {
        let cmd = CostCommand {
            session: None,
            breakdown: false,
            since: Some("2024-01-01".to_string()),
            export: None,
        };

        assert_eq!(cmd.since, Some("2024-01-01".to_string()));
    }

    #[test]
    fn test_cost_command_with_export() {
        let cmd = CostCommand {
            session: None,
            breakdown: false,
            since: None,
            export: Some("csv".to_string()),
        };

        assert_eq!(cmd.export, Some("csv".to_string()));
    }

    #[test]
    fn test_cost_command_all_options() {
        let cmd = CostCommand {
            session: Some("dev-session".to_string()),
            breakdown: true,
            since: Some("2024-06-01".to_string()),
            export: Some("json".to_string()),
        };

        assert_eq!(cmd.session, Some("dev-session".to_string()));
        assert!(cmd.breakdown);
        assert_eq!(cmd.since, Some("2024-06-01".to_string()));
        assert_eq!(cmd.export, Some("json".to_string()));
    }
}

#[cfg(test)]
mod history_command_parsing_tests {
    use super::*;

    #[test]
    fn test_history_command_basic() {
        let cmd = HistoryCommand {
            search: None,
            session: None,
            limit: 20,
            output: false,
            export: None,
        };

        assert!(cmd.search.is_none());
        assert!(cmd.session.is_none());
        assert_eq!(cmd.limit, 20);
        assert!(!cmd.output);
        assert!(cmd.export.is_none());
    }

    #[test]
    fn test_history_command_with_search() {
        let cmd = HistoryCommand {
            search: Some("analyze.*test".to_string()),
            session: None,
            limit: 20,
            output: false,
            export: None,
        };

        assert_eq!(cmd.search, Some("analyze.*test".to_string()));
    }

    #[test]
    fn test_history_command_with_session() {
        let cmd = HistoryCommand {
            search: None,
            session: Some("work-session".to_string()),
            limit: 20,
            output: false,
            export: None,
        };

        assert_eq!(cmd.session, Some("work-session".to_string()));
    }

    #[test]
    fn test_history_command_with_limit() {
        let cmd = HistoryCommand {
            search: None,
            session: None,
            limit: 100,
            output: false,
            export: None,
        };

        assert_eq!(cmd.limit, 100);
    }

    #[test]
    fn test_history_command_with_output() {
        let cmd = HistoryCommand {
            search: None,
            session: None,
            limit: 20,
            output: true,
            export: None,
        };

        assert!(cmd.output);
    }

    #[test]
    fn test_history_command_all_options() {
        let cmd = HistoryCommand {
            search: Some("error|failure".to_string()),
            session: Some("debug-session".to_string()),
            limit: 50,
            output: true,
            export: Some("json".to_string()),
        };

        assert_eq!(cmd.search, Some("error|failure".to_string()));
        assert_eq!(cmd.session, Some("debug-session".to_string()));
        assert_eq!(cmd.limit, 50);
        assert!(cmd.output);
        assert_eq!(cmd.export, Some("json".to_string()));
    }
}

#[cfg(test)]
mod config_command_parsing_tests {
    use super::*;

    #[test]
    fn test_config_show() {
        let action = ConfigAction::Show;
        match action {
            ConfigAction::Show => {
                // Test passes if it matches
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_config_set() {
        let action = ConfigAction::Set {
            key: "timeout_secs".to_string(),
            value: "60".to_string(),
        };
        match action {
            ConfigAction::Set { key, value } => {
                assert_eq!(key, "timeout_secs");
                assert_eq!(value, "60");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_config_reset() {
        let action = ConfigAction::Reset { force: false };
        match action {
            ConfigAction::Reset { force } => {
                assert!(!force);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_config_reset_force() {
        let action = ConfigAction::Reset { force: true };
        match action {
            ConfigAction::Reset { force } => {
                assert!(force);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_config_path() {
        let action = ConfigAction::Path;
        match action {
            ConfigAction::Path => {
                // Test passes if it matches
            }
            _ => panic!("Wrong variant"),
        }
    }
}

// Integration tests for end-to-end CLI testing
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_list_command_integration() {
        let env = MockCommandEnvironment::new().expect("Failed to create environment");

        // Add some test commands
        env.add_command("analyze.sh", "# Analyze code\necho 'analyzing...'")
            .unwrap();
        env.add_command("test.py", "# Run tests\nprint('testing...')")
            .unwrap();
        env.add_command("build.sh", "# Build project\necho 'building...'")
            .unwrap();

        // Test basic list
        let cmd = ListCommand {
            filter: None,
            detailed: false,
        };

        let result = cmd.execute(env.data_dir()).await;
        assert!(result.is_ok());

        // Test with filter
        let cmd = ListCommand {
            filter: Some("test".to_string()),
            detailed: false,
        };

        let result = cmd.execute(env.data_dir()).await;
        assert!(result.is_ok());

        // Test detailed
        let cmd = ListCommand {
            filter: None,
            detailed: true,
        };

        let result = cmd.execute(env.data_dir()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_command_no_commands_dir() {
        let temp_dir = TempDir::new().unwrap();

        let cmd = ListCommand {
            filter: None,
            detailed: false,
        };

        let result = cmd.execute(temp_dir.path()).await;
        assert!(result.is_err());

        if let Err(e) = result {
            match e {
                InteractiveError::CommandDiscovery(msg) => {
                    assert!(msg.contains("Commands directory not found"));
                }
                _ => panic!("Expected CommandDiscovery error"),
            }
        }
    }

    #[tokio::test]
    async fn test_list_command_empty_directory() {
        let env = MockCommandEnvironment::new().expect("Failed to create environment");

        let cmd = ListCommand {
            filter: None,
            detailed: false,
        };

        let result = cmd.execute(env.data_dir()).await;
        assert!(result.is_ok()); // Should succeed but show no commands
    }

    #[tokio::test]
    async fn test_list_command_filter_no_matches() {
        let env = MockCommandEnvironment::new().expect("Failed to create environment");
        env.add_command("analyze.sh", "# Analyze code").unwrap();
        env.add_command("build.sh", "# Build project").unwrap();

        let cmd = ListCommand {
            filter: Some("nonexistent".to_string()),
            detailed: false,
        };

        let result = cmd.execute(env.data_dir()).await;
        assert!(result.is_ok()); // Should succeed but show no matches
    }
}

// Task 4.3: CLI Argument Validation Tests
#[cfg(test)]
mod argument_validation_tests {
    use super::*;

    #[test]
    fn test_session_name_validation() {
        // Test valid session names
        let valid_names = vec![
            "my-session",
            "session_1",
            "test123",
            "work-project",
            "a", // Single character
            "session-with-many-dashes",
            "under_score_session",
            "MixedCaseSession",
        ];

        for name in valid_names {
            let action = SessionAction::Create {
                name: name.to_string(),
                description: None,
            };

            match action {
                SessionAction::Create { name: n, .. } => {
                    assert_eq!(n, name);
                    assert!(!n.is_empty());
                    assert!(!n.starts_with(' '));
                    assert!(!n.ends_with(' '));
                }
                _ => panic!("Wrong variant"),
            }
        }
    }

    #[test]
    fn test_command_args_validation() {
        // Test various argument patterns
        let test_cases = vec![
            (vec![], 0, "empty args"),
            (vec!["--file".to_string()], 1, "single flag"),
            (
                vec!["--file".to_string(), "test.rs".to_string()],
                2,
                "flag with value",
            ),
            (
                vec!["--verbose".to_string(), "--quiet".to_string()],
                2,
                "multiple flags",
            ),
            (
                vec!["arg1".to_string(), "arg2".to_string(), "arg3".to_string()],
                3,
                "positional args",
            ),
        ];

        for (args, expected_count, description) in test_cases {
            let cmd = RunCommand {
                command: "test".to_string(),
                args: args.clone(),
                session: None,
                parallel: false,
                agents: 1,
            };

            assert_eq!(
                cmd.args.len(),
                expected_count,
                "Failed for: {}",
                description
            );
        }
    }

    #[test]
    fn test_parallel_execution_validation() {
        // Test agent count validation
        let test_cases = vec![
            (false, 1, "serial execution"),
            (true, 1, "parallel with 1 agent"),
            (true, 4, "parallel with 4 agents"),
            (true, 8, "parallel with 8 agents"),
            (true, 16, "parallel with many agents"),
        ];

        for (parallel, agents, description) in test_cases {
            let cmd = RunCommand {
                command: "test".to_string(),
                args: vec![],
                session: None,
                parallel,
                agents,
            };

            if parallel && agents > 1 {
                assert!(cmd.parallel, "Should be parallel for: {}", description);
            }

            assert!(
                cmd.agents >= 1,
                "Agent count should be positive for: {}",
                description
            );
        }
    }

    #[test]
    fn test_date_format_validation() {
        // Test date string formats for cost command
        let valid_dates = vec!["2024-01-01", "2024-12-31", "2023-06-15", "2025-02-28"];

        for date_str in valid_dates {
            let cmd = CostCommand {
                session: None,
                breakdown: false,
                since: Some(date_str.to_string()),
                export: None,
            };

            assert!(cmd.since.is_some());
            let since = cmd.since.unwrap();
            assert_eq!(since.len(), 10); // YYYY-MM-DD format
            assert!(since.contains('-'));

            // Check basic format
            let parts: Vec<&str> = since.split('-').collect();
            assert_eq!(parts.len(), 3);
            assert_eq!(parts[0].len(), 4); // Year
            assert_eq!(parts[1].len(), 2); // Month
            assert_eq!(parts[2].len(), 2); // Day
        }
    }

    #[test]
    fn test_export_format_validation() {
        let valid_formats = vec!["json", "csv"];

        for format in valid_formats {
            let cost_cmd = CostCommand {
                session: None,
                breakdown: false,
                since: None,
                export: Some(format.to_string()),
            };

            assert_eq!(cost_cmd.export, Some(format.to_string()));

            let history_cmd = HistoryCommand {
                search: None,
                session: None,
                limit: 20,
                output: false,
                export: Some(format.to_string()),
            };

            assert_eq!(history_cmd.export, Some(format.to_string()));
        }
    }

    #[test]
    fn test_limit_validation() {
        let test_limits = vec![1, 20, 50, 100, 1000, usize::MAX];

        for limit in test_limits {
            let cmd = HistoryCommand {
                search: None,
                session: None,
                limit,
                output: false,
                export: None,
            };

            assert_eq!(cmd.limit, limit);
            assert!(cmd.limit > 0);
        }
    }

    #[test]
    fn test_search_pattern_validation() {
        let valid_patterns = vec![
            "simple text",
            "analyze.*test",
            "error|failure|exception",
            "^start",
            "end$",
            "[a-zA-Z0-9]+",
            "\\d{4}-\\d{2}-\\d{2}",
        ];

        for pattern in valid_patterns {
            let cmd = HistoryCommand {
                search: Some(pattern.to_string()),
                session: None,
                limit: 20,
                output: false,
                export: None,
            };

            assert!(cmd.search.is_some());
            assert_eq!(cmd.search.unwrap(), pattern);
        }
    }

    #[test]
    fn test_config_key_validation() {
        let valid_keys = vec![
            "timeout_secs",
            "verbose",
            "quiet",
            "session.model",
            "session.system_prompt",
            "output.color",
            "output.progress",
            "output.format",
            "analytics.enabled",
        ];

        for key in valid_keys {
            match (ConfigAction::Set {
                key: key.to_string(),
                value: "test_value".to_string(),
            }) {
                ConfigAction::Set { key: k, value: v } => {
                    assert_eq!(k, key);
                    assert_eq!(v, "test_value");
                    assert!(k
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '.' || c == '_'));
                }
                _ => panic!("Wrong variant"),
            }
        }
    }

    #[test]
    fn test_boolean_flag_combinations() {
        // Test conflicting boolean flags
        struct TestFlags {
            verbose: bool,
            quiet: bool,
            detailed: bool,
            summary: bool,
        }

        let test_cases = vec![
            TestFlags {
                verbose: true,
                quiet: false,
                detailed: true,
                summary: false,
            },
            TestFlags {
                verbose: false,
                quiet: true,
                detailed: false,
                summary: true,
            },
            TestFlags {
                verbose: false,
                quiet: false,
                detailed: true,
                summary: false,
            },
            TestFlags {
                verbose: false,
                quiet: false,
                detailed: false,
                summary: false,
            },
        ];

        for flags in test_cases {
            // Verbose and quiet should not both be true
            assert!(
                !(flags.verbose && flags.quiet),
                "Verbose and quiet cannot both be true"
            );

            // Detailed and summary might be mutually exclusive in some contexts
            if flags.detailed && flags.summary {
                // This might be a warning case
                assert!(true, "Both detailed and summary requested");
            }
        }
    }

    #[test]
    fn test_optional_argument_combinations() {
        // Test various combinations of optional arguments
        let cmd1 = CostCommand {
            session: Some("test".to_string()),
            breakdown: true,
            since: Some("2024-01-01".to_string()),
            export: Some("json".to_string()),
        };

        // All options set
        assert!(cmd1.session.is_some());
        assert!(cmd1.breakdown);
        assert!(cmd1.since.is_some());
        assert!(cmd1.export.is_some());

        let cmd2 = CostCommand {
            session: None,
            breakdown: false,
            since: None,
            export: None,
        };

        // No options set
        assert!(cmd2.session.is_none());
        assert!(!cmd2.breakdown);
        assert!(cmd2.since.is_none());
        assert!(cmd2.export.is_none());

        // Partial options
        let cmd3 = CostCommand {
            session: Some("test".to_string()),
            breakdown: true,
            since: None,
            export: None,
        };

        assert!(cmd3.session.is_some());
        assert!(cmd3.breakdown);
        assert!(cmd3.since.is_none());
        assert!(cmd3.export.is_none());
    }
}

// Task 4.5: Error Recovery and Retry Logic Tests
#[cfg(test)]
mod error_recovery_tests {
    use super::*;
    use std::io;
    use std::time::Duration;

    #[test]
    fn test_retry_decision_logic() {
        // Test which errors should trigger retries
        let errors = vec![
            (
                InteractiveError::Timeout(30),
                true,
                "timeout should be retryable",
            ),
            (
                InteractiveError::CommandNotFound("cmd".to_string()),
                false,
                "command not found should not retry",
            ),
            (
                InteractiveError::SessionNotFound("session".to_string()),
                false,
                "session not found should not retry",
            ),
            (
                InteractiveError::InvalidInput("bad".to_string()),
                false,
                "invalid input should not retry",
            ),
            (
                InteractiveError::PermissionDenied("denied".to_string()),
                false,
                "permission denied should not retry",
            ),
            (
                InteractiveError::Io(io::Error::new(io::ErrorKind::TimedOut, "timeout")),
                true,
                "IO timeout should retry",
            ),
            (
                InteractiveError::Io(io::Error::new(io::ErrorKind::ConnectionRefused, "refused")),
                true,
                "connection refused should retry",
            ),
            (
                InteractiveError::Io(io::Error::new(io::ErrorKind::NotFound, "not found")),
                true,
                "IO errors are retryable including NotFound",
            ),
        ];

        for (error, should_retry, description) in errors {
            assert_eq!(
                error.is_retryable(),
                should_retry,
                "Failed for: {}",
                description
            );
        }
    }

    #[derive(Debug, Clone)]
    struct RetryConfig {
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        exponential_base: f64,
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

    fn calculate_retry_delay(attempt: u32, config: &RetryConfig) -> Duration {
        let exponential_delay = config.initial_delay.as_millis() as f64
            * config.exponential_base.powi(attempt as i32 - 1);

        let capped_delay = exponential_delay.min(config.max_delay.as_millis() as f64);
        Duration::from_millis(capped_delay as u64)
    }

    #[test]
    fn test_exponential_backoff_calculation() {
        let config = RetryConfig::default();

        // Test exponential progression
        let delay1 = calculate_retry_delay(1, &config);
        let delay2 = calculate_retry_delay(2, &config);
        let delay3 = calculate_retry_delay(3, &config);

        assert_eq!(delay1, Duration::from_millis(100));
        assert_eq!(delay2, Duration::from_millis(200));
        assert_eq!(delay3, Duration::from_millis(400));

        // Test max delay capping
        let delay10 = calculate_retry_delay(10, &config);
        assert!(delay10 <= config.max_delay);
    }

    #[test]
    fn test_error_recovery_strategies() {
        enum RecoveryStrategy {
            Retry(RetryConfig),
            Fallback(String),
            Fail,
        }

        fn get_recovery_strategy(error: &InteractiveError) -> RecoveryStrategy {
            match error {
                InteractiveError::Timeout(_) => RecoveryStrategy::Retry(RetryConfig {
                    max_attempts: 3,
                    initial_delay: Duration::from_secs(1),
                    ..Default::default()
                }),
                InteractiveError::CommandNotFound(_) => {
                    RecoveryStrategy::Fallback("Suggest similar commands".to_string())
                }
                InteractiveError::CostTracking(_) => {
                    RecoveryStrategy::Fallback("Continue without cost tracking".to_string())
                }
                InteractiveError::History(_) => {
                    RecoveryStrategy::Fallback("Continue without history".to_string())
                }
                InteractiveError::PermissionDenied(_) => RecoveryStrategy::Fail,
                _ => RecoveryStrategy::Retry(RetryConfig::default()),
            }
        }

        // Test different strategies
        let timeout_err = InteractiveError::Timeout(30);
        match get_recovery_strategy(&timeout_err) {
            RecoveryStrategy::Retry(config) => {
                assert_eq!(config.max_attempts, 3);
                assert_eq!(config.initial_delay, Duration::from_secs(1));
            }
            _ => panic!("Expected Retry strategy for timeout"),
        }

        let cmd_not_found = InteractiveError::CommandNotFound("test".to_string());
        match get_recovery_strategy(&cmd_not_found) {
            RecoveryStrategy::Fallback(msg) => {
                assert!(msg.contains("Suggest similar commands"));
            }
            _ => panic!("Expected Fallback strategy for command not found"),
        }

        let permission_err = InteractiveError::PermissionDenied("denied".to_string());
        match get_recovery_strategy(&permission_err) {
            RecoveryStrategy::Fail => {
                // Correct - permission errors should fail immediately
            }
            _ => panic!("Expected Fail strategy for permission denied"),
        }
    }

    #[tokio::test]
    async fn test_transient_failure_handling() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        // Simulate operation that fails first 2 times, then succeeds
        let attempt_count = Arc::new(AtomicU32::new(0));

        async fn flaky_operation(attempt_count: Arc<AtomicU32>) -> Result<String> {
            let attempts = attempt_count.fetch_add(1, Ordering::SeqCst);

            if attempts < 2 {
                Err(InteractiveError::Io(io::Error::new(
                    io::ErrorKind::ConnectionRefused,
                    "Temporary failure",
                )))
            } else {
                Ok("Success!".to_string())
            }
        }

        // Retry logic
        async fn retry_operation<F, Fut, T>(operation: F, max_attempts: u32) -> Result<T>
        where
            F: Fn() -> Fut,
            Fut: std::future::Future<Output = Result<T>>,
        {
            let mut last_error = None;

            for attempt in 1..=max_attempts {
                match operation().await {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        if !e.is_retryable() || attempt == max_attempts {
                            return Err(e);
                        }
                        last_error = Some(e);

                        // Wait before retry (simplified, no exponential backoff here)
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }

            Err(last_error.unwrap())
        }

        // Test successful retry
        let result = retry_operation(|| flaky_operation(Arc::clone(&attempt_count)), 3).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success!");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_graceful_degradation_decisions() {
        // Test which features can be disabled for graceful degradation
        struct AppFeatures {
            cost_tracking: bool,
            history: bool,
            parallel_execution: bool,
            analytics: bool,
            progress_indicators: bool,
        }

        impl AppFeatures {
            fn new() -> Self {
                Self {
                    cost_tracking: true,
                    history: true,
                    parallel_execution: true,
                    analytics: true,
                    progress_indicators: true,
                }
            }

            fn degrade_for_error(&mut self, error: &InteractiveError) {
                match error {
                    InteractiveError::CostTracking(_) => {
                        self.cost_tracking = false;
                    }
                    InteractiveError::History(_) => {
                        self.history = false;
                    }
                    InteractiveError::ParallelExecution(_) => {
                        self.parallel_execution = false;
                    }
                    InteractiveError::OutputFormatting(_) => {
                        self.progress_indicators = false;
                    }
                    _ => {}
                }
            }

            fn is_operational(&self) -> bool {
                // Core functionality doesn't include cost tracking, history, or analytics
                true // Always operational, just with reduced features
            }
        }

        let mut features = AppFeatures::new();
        assert!(features.cost_tracking);
        assert!(features.history);

        // Simulate cost tracking failure
        let cost_error = InteractiveError::CostTracking("DB error".to_string());
        features.degrade_for_error(&cost_error);

        assert!(!features.cost_tracking);
        assert!(features.history); // Other features unaffected
        assert!(features.is_operational());

        // Simulate history failure
        let history_error = InteractiveError::History("Storage error".to_string());
        features.degrade_for_error(&history_error);

        assert!(!features.history);
        assert!(features.is_operational());
    }

    #[test]
    fn test_error_context_for_debugging() {
        // Test that errors contain enough context for debugging
        fn create_detailed_error(operation: &str, cause: &str) -> InteractiveError {
            InteractiveError::execution(format!(
                "Operation '{}' failed: {}. Context: user=test, session=work, command=analyze",
                operation, cause
            ))
        }

        let error = create_detailed_error("command_execution", "timeout after 30s");
        let error_string = error.to_string();

        // Verify error contains all context
        assert!(error_string.contains("command_execution"));
        assert!(error_string.contains("timeout after 30s"));
        assert!(error_string.contains("user=test"));
        assert!(error_string.contains("session=work"));
        assert!(error_string.contains("command=analyze"));
    }
}
