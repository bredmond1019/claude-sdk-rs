//! Mock implementations for testing
//!
//! This module provides mock implementations of external dependencies:
//! - Mock Claude CLI responses
//! - Mock file system operations
//! - Mock network requests
//! - Test data generators

use crate::{
    cost::{CostEntry, CostSummary},
    error::InteractiveError,
    history::{HistoryEntry, HistoryStats},
    session::{Session, SessionId, SessionMetadata},
    Result,
};
use chrono::{Duration, Utc};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Mock Claude CLI response generator
pub struct MockClaudeClient {
    responses: Arc<Mutex<Vec<MockResponse>>>,
    response_delay_ms: u64,
    failure_rate: f64,
}

/// Mock response configuration
#[derive(Debug, Clone)]
pub struct MockResponse {
    pub command_pattern: String,
    pub response_text: String,
    pub cost_usd: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub duration_ms: u64,
    pub success: bool,
}

impl MockClaudeClient {
    /// Create a new mock Claude client
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            response_delay_ms: 100,
            failure_rate: 0.0,
        }
    }

    /// Add a mock response for a command pattern
    pub fn add_response(&self, response: MockResponse) {
        let mut responses = self.responses.lock().unwrap();
        responses.push(response);
    }

    /// Set artificial response delay
    pub fn set_response_delay(&mut self, delay_ms: u64) {
        self.response_delay_ms = delay_ms;
    }

    /// Set failure rate (0.0 to 1.0)
    pub fn set_failure_rate(&mut self, rate: f64) {
        self.failure_rate = rate.clamp(0.0, 1.0);
    }

    /// Execute a mock command
    pub async fn execute_command(
        &self,
        command: &str,
        args: &[String],
    ) -> Result<MockCommandResult> {
        // Simulate response delay
        if self.response_delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.response_delay_ms)).await;
        }

        // Simulate random failures
        if self.failure_rate > 0.0 && rand::random::<f64>() < self.failure_rate {
            return Err(InteractiveError::execution("Mock command failure"));
        }

        // Find matching response
        let responses = self.responses.lock().unwrap();
        let full_command = format!("{} {}", command, args.join(" "));

        for response in responses.iter() {
            if full_command.contains(&response.command_pattern) {
                return Ok(MockCommandResult {
                    output: response.response_text.clone(),
                    cost_usd: response.cost_usd,
                    input_tokens: response.input_tokens,
                    output_tokens: response.output_tokens,
                    duration_ms: response.duration_ms,
                    success: response.success,
                });
            }
        }

        // Default response if no pattern matches
        Ok(MockCommandResult {
            output: "Mock command executed successfully".to_string(),
            cost_usd: 0.01,
            input_tokens: 50,
            output_tokens: 100,
            duration_ms: 1000,
            success: true,
        })
    }

    /// Setup common test responses
    pub fn setup_default_responses(&self) {
        self.add_response(MockResponse {
            command_pattern: "list".to_string(),
            response_text: "Available commands:\n- analyze\n- generate\n- help".to_string(),
            cost_usd: 0.005,
            input_tokens: 20,
            output_tokens: 40,
            duration_ms: 500,
            success: true,
        });

        self.add_response(MockResponse {
            command_pattern: "analyze".to_string(),
            response_text: "Analysis complete. Found 3 issues to address.".to_string(),
            cost_usd: 0.025,
            input_tokens: 100,
            output_tokens: 200,
            duration_ms: 2000,
            success: true,
        });

        self.add_response(MockResponse {
            command_pattern: "generate".to_string(),
            response_text:
                "Generated code:\n\n```rust\nfn hello() {\n    println!(\"Hello, world!\");\n}\n```"
                    .to_string(),
            cost_usd: 0.035,
            input_tokens: 80,
            output_tokens: 300,
            duration_ms: 3000,
            success: true,
        });

        self.add_response(MockResponse {
            command_pattern: "error".to_string(),
            response_text: "".to_string(),
            cost_usd: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            duration_ms: 100,
            success: false,
        });
    }
}

/// Result of a mock command execution
#[derive(Debug, Clone)]
pub struct MockCommandResult {
    pub output: String,
    pub cost_usd: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub duration_ms: u64,
    pub success: bool,
}

/// Test data generator for sessions
pub struct SessionDataGenerator;

impl SessionDataGenerator {
    /// Generate a test session
    pub fn generate_session(name: Option<String>) -> Session {
        let id = Uuid::new_v4();
        let name = name.unwrap_or_else(|| format!("test-session-{}", id));

        Session {
            id,
            name,
            description: Some("Generated test session".to_string()),
            created_at: Utc::now() - Duration::hours(rand::random::<i64>() % 24),
            last_active: Utc::now() - Duration::minutes(rand::random::<i64>() % 60),
            metadata: SessionMetadata {
                total_commands: rand::random::<usize>() % 50,
                total_cost: rand::random::<f64>() * 10.0,
                tags: vec!["test".to_string(), "generated".to_string()],
            },
        }
    }

    /// Generate multiple test sessions
    pub fn generate_sessions(count: usize) -> Vec<Session> {
        (0..count)
            .map(|i| Self::generate_session(Some(format!("test-session-{}", i))))
            .collect()
    }
}

/// Test data generator for cost entries
pub struct CostDataGenerator;

impl CostDataGenerator {
    /// Generate a test cost entry
    pub fn generate_cost_entry(session_id: Option<SessionId>) -> CostEntry {
        let session_id = session_id.unwrap_or_else(Uuid::new_v4);
        let commands = ["analyze", "generate", "list", "help", "debug"];
        let models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"];

        CostEntry::new(
            session_id,
            commands[rand::random::<usize>() % commands.len()].to_string(),
            rand::random::<f64>() * 0.1,        // $0 to $0.10
            rand::random::<u32>() % 200 + 50,   // 50-250 input tokens
            rand::random::<u32>() % 500 + 100,  // 100-600 output tokens
            rand::random::<u64>() % 5000 + 500, // 500-5500ms duration
            models[rand::random::<usize>() % models.len()].to_string(),
        )
    }

    /// Generate multiple cost entries for a session
    pub fn generate_session_costs(session_id: SessionId, count: usize) -> Vec<CostEntry> {
        (0..count)
            .map(|_| Self::generate_cost_entry(Some(session_id)))
            .collect()
    }

    /// Generate a mock cost summary
    pub fn generate_cost_summary() -> CostSummary {
        let commands = ["analyze", "generate", "list", "help", "debug"];
        let models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"];

        let mut by_command = HashMap::new();
        let mut by_model = HashMap::new();
        let mut total_cost = 0.0;

        for command in commands {
            let cost = rand::random::<f64>() * 5.0;
            by_command.insert(command.to_string(), cost);
            total_cost += cost;
        }

        for model in models {
            let cost = rand::random::<f64>() * 3.0;
            by_model.insert(model.to_string(), cost);
        }

        let command_count = rand::random::<usize>() % 100 + 10;

        CostSummary {
            total_cost,
            command_count,
            average_cost: total_cost / command_count as f64,
            total_tokens: rand::random::<u32>() % 10000 + 1000,
            date_range: (Utc::now() - Duration::days(7), Utc::now()),
            by_command,
            by_model,
        }
    }
}

/// Test data generator for history entries
pub struct HistoryDataGenerator;

impl HistoryDataGenerator {
    /// Generate a test history entry
    pub fn generate_history_entry(session_id: Option<SessionId>) -> HistoryEntry {
        let session_id = session_id.unwrap_or_else(Uuid::new_v4);
        let commands = ["analyze", "generate", "list", "help", "debug"];
        let success_rate = 0.85; // 85% success rate

        let command = commands[rand::random::<usize>() % commands.len()];
        let success = rand::random::<f64>() < success_rate;

        let mut entry = HistoryEntry::new(
            session_id,
            command.to_string(),
            vec!["--option".to_string(), "value".to_string()],
            if success {
                format!("{} executed successfully with sample output", command)
            } else {
                "Command failed with error".to_string()
            },
            success,
            rand::random::<u64>() % 5000 + 500,
        );

        if success {
            entry = entry.with_cost(
                rand::random::<f64>() * 0.1,
                rand::random::<u32>() % 200 + 50,
                rand::random::<u32>() % 500 + 100,
                "claude-3-opus".to_string(),
            );
        } else {
            entry = entry.with_error("Mock command execution failed".to_string());
        }

        entry.with_tags(vec!["test".to_string(), "generated".to_string()])
    }

    /// Generate multiple history entries for a session
    pub fn generate_session_history(session_id: SessionId, count: usize) -> Vec<HistoryEntry> {
        (0..count)
            .map(|_| Self::generate_history_entry(Some(session_id)))
            .collect()
    }

    /// Generate mock history statistics
    pub fn generate_history_stats() -> HistoryStats {
        let total_entries = rand::random::<usize>() % 100 + 20;
        let successful_commands = (total_entries as f64 * 0.85) as usize;
        let failed_commands = total_entries - successful_commands;

        let mut command_counts = HashMap::new();
        let mut model_usage = HashMap::new();

        let commands = ["analyze", "generate", "list", "help", "debug"];
        for command in commands {
            command_counts.insert(command.to_string(), rand::random::<usize>() % 20 + 1);
        }

        let models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"];
        for model in models {
            model_usage.insert(model.to_string(), rand::random::<usize>() % 15 + 1);
        }

        HistoryStats {
            total_entries,
            successful_commands,
            failed_commands,
            success_rate: (successful_commands as f64 / total_entries as f64) * 100.0,
            total_cost: rand::random::<f64>() * 50.0,
            total_duration_ms: rand::random::<u64>() % 100000 + 10000,
            average_duration_ms: rand::random::<f64>() * 2000.0 + 500.0,
            average_cost: rand::random::<f64>() * 0.5,
            command_counts,
            model_usage,
            date_range: (Utc::now() - Duration::days(30), Utc::now()),
        }
    }
}

/// Mock file system for testing
pub struct MockFileSystem {
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    read_only: bool,
}

impl MockFileSystem {
    /// Create a new mock file system
    pub fn new() -> Self {
        Self {
            files: Arc::new(Mutex::new(HashMap::new())),
            read_only: false,
        }
    }

    /// Set read-only mode
    pub fn set_read_only(&mut self, read_only: bool) {
        self.read_only = read_only;
    }

    /// Write a file
    pub fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        if self.read_only {
            return Err(InteractiveError::PermissionDenied(
                "File system is read-only".to_string(),
            ));
        }

        let mut files = self.files.lock().unwrap();
        files.insert(path.to_string(), content.to_vec());
        Ok(())
    }

    /// Read a file
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let files = self.files.lock().unwrap();
        files
            .get(path)
            .cloned()
            .ok_or_else(|| InteractiveError::invalid_input(format!("File not found: {}", path)))
    }

    /// Check if file exists
    pub fn exists(&self, path: &str) -> bool {
        let files = self.files.lock().unwrap();
        files.contains_key(path)
    }

    /// List all files
    pub fn list_files(&self) -> Vec<String> {
        let files = self.files.lock().unwrap();
        files.keys().cloned().collect()
    }

    /// Create test files
    pub fn create_test_files(&self) -> Result<()> {
        self.write_file("test_config.json", br#"{"test": true}"#)?;
        self.write_file("test_data.txt", b"Sample test data")?;
        self.write_file("empty_file.txt", b"")?;
        Ok(())
    }
}

/// Mock network client for testing external API calls
pub struct MockNetworkClient {
    responses: Arc<Mutex<HashMap<String, MockHttpResponse>>>,
    delay_ms: u64,
    failure_rate: f64,
}

/// Mock HTTP response
#[derive(Debug, Clone)]
pub struct MockHttpResponse {
    pub status_code: u16,
    pub body: String,
    pub headers: HashMap<String, String>,
}

impl MockNetworkClient {
    /// Create a new mock network client
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(HashMap::new())),
            delay_ms: 50,
            failure_rate: 0.0,
        }
    }

    /// Add a mock response for a URL
    pub fn add_response(&self, url: &str, response: MockHttpResponse) {
        let mut responses = self.responses.lock().unwrap();
        responses.insert(url.to_string(), response);
    }

    /// Set network delay
    pub fn set_delay(&mut self, delay_ms: u64) {
        self.delay_ms = delay_ms;
    }

    /// Set failure rate
    pub fn set_failure_rate(&mut self, rate: f64) {
        self.failure_rate = rate.clamp(0.0, 1.0);
    }

    /// Make a mock HTTP request
    pub async fn get(&self, url: &str) -> Result<MockHttpResponse> {
        // Simulate network delay
        if self.delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
        }

        // Simulate random failures
        if self.failure_rate > 0.0 && rand::random::<f64>() < self.failure_rate {
            return Err(InteractiveError::invalid_input("Mock network failure"));
        }

        let responses = self.responses.lock().unwrap();
        if let Some(response) = responses.get(url) {
            Ok(response.clone())
        } else {
            Ok(MockHttpResponse {
                status_code: 404,
                body: "Not Found".to_string(),
                headers: HashMap::new(),
            })
        }
    }

    /// Setup common test responses
    pub fn setup_default_responses(&self) {
        // Mock Claude API health check
        self.add_response(
            "https://api.anthropic.com/health",
            MockHttpResponse {
                status_code: 200,
                body: r#"{"status": "healthy"}"#.to_string(),
                headers: {
                    let mut headers = HashMap::new();
                    headers.insert("content-type".to_string(), "application/json".to_string());
                    headers
                },
            },
        );

        // Mock version check
        self.add_response(
            "https://api.github.com/repos/anthropics/claude-cli/releases/latest",
            MockHttpResponse {
                status_code: 200,
                body: r#"{"tag_name": "v1.0.0", "name": "Latest Release"}"#.to_string(),
                headers: {
                    let mut headers = HashMap::new();
                    headers.insert("content-type".to_string(), "application/json".to_string());
                    headers
                },
            },
        );
    }
}
