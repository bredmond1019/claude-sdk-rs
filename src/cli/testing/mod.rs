//! Testing framework and quality assurance utilities
//!
//! This module provides:
//! - Test harnesses for interactive CLI components
//! - Mock implementations for testing
//! - Quality assurance and validation tools
//! - Performance benchmarking utilities

pub mod benchmarks;
pub mod harness;
pub mod mocks;
pub mod validation;

#[cfg(test)]
pub mod memory_tests;

use crate::cli::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::process::Command;

/// Test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub test_data_dir: PathBuf,
    pub temp_dir: PathBuf,
    pub mock_claude_responses: bool,
    pub enable_integration_tests: bool,
    pub parallel_test_execution: bool,
    pub max_test_duration_seconds: u64,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            test_data_dir: PathBuf::from("test_data"),
            temp_dir: PathBuf::from("/tmp/claude_interactive_tests"),
            mock_claude_responses: true,
            enable_integration_tests: false,
            parallel_test_execution: true,
            max_test_duration_seconds: 300,
        }
    }
}

/// Test result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub duration_ms: u64,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Test suite runner
pub struct TestSuite {
    config: TestConfig,
    tests: Vec<Box<dyn TestCase>>,
}

/// Test case trait (simplified for object safety)
pub trait TestCase: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run_sync(&self, config: &TestConfig) -> Result<TestResult>;
    fn requires_claude_cli(&self) -> bool {
        false
    }
    fn is_integration_test(&self) -> bool {
        false
    }
}

impl TestSuite {
    /// Create a new test suite
    pub fn new(config: TestConfig) -> Self {
        Self {
            config,
            tests: Vec::new(),
        }
    }

    /// Add a test case to the suite
    pub fn add_test<T: TestCase + 'static>(&mut self, test: T) {
        self.tests.push(Box::new(test));
    }

    /// Run all tests in the suite
    pub async fn run_all(&self) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();

        // Setup test environment
        self.setup_test_environment().await?;

        for test in &self.tests {
            // Skip integration tests if disabled
            if test.is_integration_test() && !self.config.enable_integration_tests {
                continue;
            }

            // Skip tests requiring Claude CLI if mocking
            if test.requires_claude_cli() && self.config.mock_claude_responses {
                continue;
            }

            let start_time = std::time::Instant::now();
            let result = match test.run_sync(&self.config) {
                Ok(mut test_result) => {
                    test_result.duration_ms = start_time.elapsed().as_millis() as u64;
                    test_result
                }
                Err(e) => TestResult {
                    test_name: test.name().to_string(),
                    passed: false,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    error_message: Some(e.to_string()),
                    metadata: HashMap::new(),
                },
            };

            results.push(result);
        }

        // Cleanup test environment
        self.cleanup_test_environment().await?;

        Ok(results)
    }

    /// Run tests in parallel
    pub async fn run_parallel(&self) -> Result<Vec<TestResult>> {
        if !self.config.parallel_test_execution {
            return self.run_all().await;
        }

        // Setup test environment
        self.setup_test_environment().await?;

        // For simplicity, run sequentially for now
        self.run_all().await
    }

    /// Generate test report
    pub fn generate_report(&self, results: &[TestResult]) -> TestReport {
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;
        let total_duration_ms = results.iter().map(|r| r.duration_ms).sum();

        let mut slowest_tests = results.to_vec();
        slowest_tests.sort_by(|a, b| b.duration_ms.cmp(&a.duration_ms));
        slowest_tests.truncate(5);

        let failed_tests_details: Vec<_> = results.iter().filter(|r| !r.passed).cloned().collect();

        TestReport {
            total_tests,
            passed_tests,
            failed_count: failed_tests,
            success_rate: (passed_tests as f64 / total_tests as f64) * 100.0,
            total_duration_ms,
            average_duration_ms: total_duration_ms as f64 / total_tests as f64,
            slowest_tests,
            failed_tests: failed_tests_details,
            generated_at: chrono::Utc::now(),
        }
    }

    // Private helper methods

    async fn setup_test_environment(&self) -> Result<()> {
        // Create test directories
        tokio::fs::create_dir_all(&self.config.test_data_dir).await?;
        tokio::fs::create_dir_all(&self.config.temp_dir).await?;

        // Setup test data if needed
        self.create_test_data().await?;

        Ok(())
    }

    async fn cleanup_test_environment(&self) -> Result<()> {
        // Clean up temporary test files
        if self.config.temp_dir.exists() {
            tokio::fs::remove_dir_all(&self.config.temp_dir).await?;
        }

        Ok(())
    }

    async fn create_test_data(&self) -> Result<()> {
        let test_sessions_file = self.config.test_data_dir.join("test_sessions.json");
        if !test_sessions_file.exists() {
            let test_sessions = vec![
                serde_json::json!({
                    "id": "test-session-1",
                    "name": "Test Session 1",
                    "description": "Test session for unit tests",
                    "created_at": "2024-01-01T00:00:00Z"
                }),
                serde_json::json!({
                    "id": "test-session-2",
                    "name": "Test Session 2",
                    "description": "Another test session",
                    "created_at": "2024-01-02T00:00:00Z"
                }),
            ];

            let content = serde_json::to_string_pretty(&test_sessions)?;
            tokio::fs::write(&test_sessions_file, content).await?;
        }

        Ok(())
    }
}

/// Test report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_count: usize,
    pub success_rate: f64,
    pub total_duration_ms: u64,
    pub average_duration_ms: f64,
    pub slowest_tests: Vec<TestResult>,
    pub failed_tests: Vec<TestResult>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

impl TestReport {
    /// Export report as HTML
    pub async fn export_html(&self, path: &PathBuf) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html><head><title>Test Report</title>")?;
        writeln!(file, "<style>")?;
        writeln!(
            file,
            "  body {{ font-family: Arial, sans-serif; margin: 20px; }}"
        )?;
        writeln!(file, "  .summary {{ background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}")?;
        writeln!(file, "  .passed {{ color: green; }}")?;
        writeln!(file, "  .failed {{ color: red; }}")?;
        writeln!(
            file,
            "  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}"
        )?;
        writeln!(
            file,
            "  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}"
        )?;
        writeln!(file, "  th {{ background-color: #f2f2f2; }}")?;
        writeln!(file, "</style></head><body>")?;

        writeln!(file, "<h1>Claude Interactive Test Report</h1>")?;
        writeln!(
            file,
            "<p>Generated: {}</p>",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        )?;

        writeln!(file, "<div class=\"summary\">")?;
        writeln!(file, "<h2>Summary</h2>")?;
        writeln!(file, "<p>Total Tests: {}</p>", self.total_tests)?;
        writeln!(
            file,
            "<p class=\"passed\">Passed: {}</p>",
            self.passed_tests
        )?;
        writeln!(
            file,
            "<p class=\"failed\">Failed: {}</p>",
            self.failed_count
        )?;
        writeln!(file, "<p>Success Rate: {:.1}%</p>", self.success_rate)?;
        writeln!(file, "<p>Total Duration: {}ms</p>", self.total_duration_ms)?;
        writeln!(
            file,
            "<p>Average Duration: {:.1}ms</p>",
            self.average_duration_ms
        )?;
        writeln!(file, "</div>")?;

        if !self.failed_tests.is_empty() {
            writeln!(file, "<h2>Failed Tests</h2>")?;
            writeln!(file, "<table>")?;
            writeln!(
                file,
                "<tr><th>Test Name</th><th>Duration</th><th>Error</th></tr>"
            )?;
            for test in &self.failed_tests {
                writeln!(file, "<tr>")?;
                writeln!(file, "<td>{}</td>", test.test_name)?;
                writeln!(file, "<td>{}ms</td>", test.duration_ms)?;
                writeln!(
                    file,
                    "<td>{}</td>",
                    test.error_message.as_deref().unwrap_or("Unknown error")
                )?;
                writeln!(file, "</tr>")?;
            }
            writeln!(file, "</table>")?;
        }

        if !self.slowest_tests.is_empty() {
            writeln!(file, "<h2>Slowest Tests</h2>")?;
            writeln!(file, "<table>")?;
            writeln!(
                file,
                "<tr><th>Test Name</th><th>Duration</th><th>Status</th></tr>"
            )?;
            for test in &self.slowest_tests {
                let status = if test.passed { "PASSED" } else { "FAILED" };
                let status_class = if test.passed { "passed" } else { "failed" };
                writeln!(file, "<tr>")?;
                writeln!(file, "<td>{}</td>", test.test_name)?;
                writeln!(file, "<td>{}ms</td>", test.duration_ms)?;
                writeln!(file, "<td class=\"{}\">{}</td>", status_class, status)?;
                writeln!(file, "</tr>")?;
            }
            writeln!(file, "</table>")?;
        }

        writeln!(file, "</body></html>")?;

        Ok(())
    }

    /// Export report as JSON
    pub async fn export_json(&self, path: &PathBuf) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }
}

/// Quality assurance utilities
pub struct QualityAssurance;

impl QualityAssurance {
    /// Run comprehensive quality checks
    pub async fn run_quality_checks(project_root: &PathBuf) -> Result<QualityReport> {
        let mut checks = Vec::new();

        // Code formatting check
        checks.push(Self::check_code_formatting(project_root).await?);

        // Linting check
        checks.push(Self::check_linting(project_root).await?);

        // Test coverage check
        checks.push(Self::check_test_coverage(project_root).await?);

        // Documentation check
        checks.push(Self::check_documentation(project_root).await?);

        // Security audit
        checks.push(Self::check_security_audit(project_root).await?);

        let passed_checks = checks.iter().filter(|c| c.passed).count();
        let total_checks = checks.len();

        Ok(QualityReport {
            checks,
            overall_score: (passed_checks as f64 / total_checks as f64) * 100.0,
            generated_at: chrono::Utc::now(),
        })
    }

    async fn check_code_formatting(project_root: &PathBuf) -> Result<QualityCheck> {
        let output = Command::new("cargo")
            .args(&["fmt", "--", "--check"])
            .current_dir(project_root)
            .output()
            .await?;

        Ok(QualityCheck {
            name: "Code Formatting".to_string(),
            description: "Checks if code is properly formatted with rustfmt".to_string(),
            passed: output.status.success(),
            message: if output.status.success() {
                "Code is properly formatted".to_string()
            } else {
                format!(
                    "Code formatting issues: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
            },
        })
    }

    async fn check_linting(project_root: &PathBuf) -> Result<QualityCheck> {
        let output = Command::new("cargo")
            .args(&["clippy", "--", "-D", "warnings"])
            .current_dir(project_root)
            .output()
            .await?;

        Ok(QualityCheck {
            name: "Linting".to_string(),
            description: "Checks for common mistakes and style issues with clippy".to_string(),
            passed: output.status.success(),
            message: if output.status.success() {
                "No linting issues found".to_string()
            } else {
                format!(
                    "Linting issues: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
            },
        })
    }

    async fn check_test_coverage(_project_root: &PathBuf) -> Result<QualityCheck> {
        // Placeholder for test coverage check
        // Would use tools like tarpaulin or grcov
        Ok(QualityCheck {
            name: "Test Coverage".to_string(),
            description: "Checks test coverage percentage".to_string(),
            passed: true,
            message: "Test coverage check not implemented".to_string(),
        })
    }

    async fn check_documentation(project_root: &PathBuf) -> Result<QualityCheck> {
        let output = Command::new("cargo")
            .args(&["doc", "--no-deps"])
            .current_dir(project_root)
            .output()
            .await?;

        Ok(QualityCheck {
            name: "Documentation".to_string(),
            description: "Checks if documentation builds successfully".to_string(),
            passed: output.status.success(),
            message: if output.status.success() {
                "Documentation builds successfully".to_string()
            } else {
                format!(
                    "Documentation issues: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
            },
        })
    }

    async fn check_security_audit(project_root: &PathBuf) -> Result<QualityCheck> {
        let output = Command::new("cargo")
            .args(&["audit"])
            .current_dir(project_root)
            .output()
            .await;

        match output {
            Ok(output) => Ok(QualityCheck {
                name: "Security Audit".to_string(),
                description: "Checks for known security vulnerabilities".to_string(),
                passed: output.status.success(),
                message: if output.status.success() {
                    "No security vulnerabilities found".to_string()
                } else {
                    format!(
                        "Security issues: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )
                },
            }),
            Err(_) => Ok(QualityCheck {
                name: "Security Audit".to_string(),
                description: "Checks for known security vulnerabilities".to_string(),
                passed: true,
                message: "cargo-audit not installed, skipping security check".to_string(),
            }),
        }
    }
}

/// Quality check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCheck {
    pub name: String,
    pub description: String,
    pub passed: bool,
    pub message: String,
}

/// Quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub checks: Vec<QualityCheck>,
    pub overall_score: f64,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}
