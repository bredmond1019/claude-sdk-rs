# Tutorial: Building a Code Review Bot with claude-sdk-rs

This tutorial demonstrates how to build a comprehensive code review bot using the claude-sdk-rs SDK. The bot will analyze code changes, provide feedback on code quality, security issues, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Project Setup](#project-setup)
3. [Core Code Review Logic](#core-code-review-logic)
4. [Git Integration](#git-integration)
5. [Web API for GitHub/GitLab Integration](#web-api-for-githubgitlab-integration)
6. [Configuration and Customization](#configuration-and-customization)
7. [Testing and Deployment](#testing-and-deployment)

## Overview

Our code review bot will include the following features:

- **Multi-language support** - Review code in Rust, Python, JavaScript, and more
- **Security analysis** - Detect common security vulnerabilities
- **Best practices** - Enforce coding standards and conventions
- **Git integration** - Analyze diffs and commit messages
- **Web API** - Integrate with GitHub, GitLab, or custom CI/CD
- **Configurable rules** - Customize review criteria per project

## Project Setup

First, create a new Rust project and add the necessary dependencies:

```toml
[package]
name = "code-review-bot"
version = "0.1.0"
edition = "2021"

[dependencies]
claude-sdk-rs = { version = "0.1", features = ["tools"] }
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
axum = { version = "0.7", features = ["ws"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
tracing = "0.1"
tracing-subscriber = "0.3"
clap = { version = "4.0", features = ["derive"] }
git2 = "0.18"
regex = "1.0"
anyhow = "1.0"
uuid = { version = "1.0", features = ["v4"] }
```

## Core Code Review Logic

Let's start by building the core code review engine:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission, StreamFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeReviewRequest {
    pub language: String,
    pub code: String,
    pub filename: String,
    pub diff: Option<String>,
    pub context: Option<ReviewContext>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReviewContext {
    pub project_type: String,      // "web-api", "library", "cli", etc.
    pub security_level: String,    // "high", "medium", "low"
    pub coding_standards: Vec<String>, // Style guides to enforce
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeReviewResult {
    pub overall_score: u8,         // 0-100
    pub issues: Vec<ReviewIssue>,
    pub suggestions: Vec<String>,
    pub security_concerns: Vec<SecurityIssue>,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReviewIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub line_number: Option<u32>,
    pub description: String,
    pub suggestion: String,
    pub code_snippet: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub severity: SecuritySeverity,
    pub vulnerability_type: String,
    pub description: String,
    pub mitigation: String,
    pub line_number: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
    Style,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum IssueCategory {
    Performance,
    Maintainability,
    Readability,
    ErrorHandling,
    Testing,
    Documentation,
    Architecture,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Critical,
    High,
    Medium,
    Low,
}

pub struct CodeReviewBot {
    claude_client: Client,
    language_configs: HashMap<String, LanguageConfig>,
}

#[derive(Debug, Clone)]
pub struct LanguageConfig {
    pub file_extensions: Vec<String>,
    pub specific_rules: Vec<String>,
    pub security_patterns: Vec<String>,
}

impl CodeReviewBot {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Json)
            .system_prompt(include_str!("../prompts/code_review_system.txt"))
            .timeout_secs(120)
            .allowed_tools(vec![
                ToolPermission::bash("grep").to_cli_format(),
                ToolPermission::bash("find").to_cli_format(),
                ToolPermission::mcp("filesystem", "read").to_cli_format(),
            ])
            .build();

        let claude_client = Client::new(config);
        let language_configs = Self::initialize_language_configs();

        Ok(Self {
            claude_client,
            language_configs,
        })
    }

    fn initialize_language_configs() -> HashMap<String, LanguageConfig> {
        let mut configs = HashMap::new();

        configs.insert("rust".to_string(), LanguageConfig {
            file_extensions: vec!["rs".to_string()],
            specific_rules: vec![
                "Check for proper error handling with Result<>".to_string(),
                "Ensure no use of unwrap() in production code".to_string(),
                "Verify proper lifetime annotations".to_string(),
                "Check for potential panic sources".to_string(),
                "Ensure proper use of ownership and borrowing".to_string(),
            ],
            security_patterns: vec![
                "unsafe blocks".to_string(),
                "raw pointers".to_string(),
                "env::args() without validation".to_string(),
                "file operations without error handling".to_string(),
            ],
        });

        configs.insert("python".to_string(), LanguageConfig {
            file_extensions: vec!["py".to_string()],
            specific_rules: vec![
                "Check for proper exception handling".to_string(),
                "Verify type hints usage".to_string(),
                "Ensure PEP 8 compliance".to_string(),
                "Check for potential security vulnerabilities".to_string(),
            ],
            security_patterns: vec![
                "eval() usage".to_string(),
                "exec() usage".to_string(),
                "pickle.loads() without validation".to_string(),
                "SQL query string concatenation".to_string(),
            ],
        });

        configs.insert("javascript".to_string(), LanguageConfig {
            file_extensions: vec!["js".to_string(), "ts".to_string()],
            specific_rules: vec![
                "Check for proper async/await usage".to_string(),
                "Verify error handling in promises".to_string(),
                "Ensure proper TypeScript types (if applicable)".to_string(),
                "Check for ESLint compliance".to_string(),
            ],
            security_patterns: vec![
                "eval() usage".to_string(),
                "innerHTML assignment".to_string(),
                "document.write()".to_string(),
                "unvalidated user input".to_string(),
            ],
        });

        configs
    }

    pub async fn review_code(&self, request: CodeReviewRequest) -> Result<CodeReviewResult, Box<dyn std::error::Error>> {
        let language_config = self.language_configs.get(&request.language)
            .ok_or_else(|| format!("Unsupported language: {}", request.language))?;

        let review_prompt = self.build_review_prompt(&request, language_config);
        
        let response = self.claude_client
            .query(&review_prompt)
            .send_full()
            .await?;

        let review_result = self.parse_review_response(&response.content)?;
        Ok(review_result)
    }

    fn build_review_prompt(&self, request: &CodeReviewRequest, config: &LanguageConfig) -> String {
        let mut prompt = format!(
            "Please review this {} code and provide a comprehensive analysis.\n\n",
            request.language
        );

        prompt.push_str(&format!("File: {}\n", request.filename));

        if let Some(context) = &request.context {
            prompt.push_str(&format!(
                "Project Context:\n- Type: {}\n- Security Level: {}\n- Standards: {}\n\n",
                context.project_type,
                context.security_level,
                context.coding_standards.join(", ")
            ));
        }

        if let Some(diff) = &request.diff {
            prompt.push_str("Git Diff:\n```diff\n");
            prompt.push_str(diff);
            prompt.push_str("\n```\n\n");
        }

        prompt.push_str("Code to Review:\n```");
        prompt.push_str(&request.language);
        prompt.push('\n');
        prompt.push_str(&request.code);
        prompt.push_str("\n```\n\n");

        prompt.push_str("Language-Specific Rules to Check:\n");
        for rule in &config.specific_rules {
            prompt.push_str(&format!("- {}\n", rule));
        }

        prompt.push_str("\nSecurity Patterns to Look For:\n");
        for pattern in &config.security_patterns {
            prompt.push_str(&format!("- {}\n", pattern));
        }

        prompt.push_str("\nPlease provide your review in this JSON format:\n");
        prompt.push_str(r#"{
  "overall_score": 85,
  "issues": [
    {
      "severity": "Warning",
      "category": "ErrorHandling",
      "line_number": 42,
      "description": "Using unwrap() could cause panic",
      "suggestion": "Use match or if let to handle the Result properly",
      "code_snippet": "result.unwrap()"
    }
  ],
  "suggestions": [
    "Consider adding unit tests for this function",
    "Add documentation comments for public functions"
  ],
  "security_concerns": [
    {
      "severity": "High",
      "vulnerability_type": "Input Validation",
      "description": "User input is not validated before processing",
      "mitigation": "Add input validation and sanitization",
      "line_number": 15
    }
  ],
  "summary": "Overall the code is well-structured but needs attention to error handling and input validation."
}"#);

        prompt
    }

    fn parse_review_response(&self, response: &str) -> Result<CodeReviewResult, Box<dyn std::error::Error>> {
        // Extract JSON from the response (Claude might include additional text)
        let json_start = response.find('{').ok_or("No JSON found in response")?;
        let json_end = response.rfind('}').ok_or("No JSON found in response")? + 1;
        let json_str = &response[json_start..json_end];

        let result: CodeReviewResult = serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse review result: {}", e))?;

        Ok(result)
    }

    pub async fn review_multiple_files(&self, files: Vec<CodeReviewRequest>) -> Result<Vec<CodeReviewResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for file_request in files {
            match self.review_code(file_request).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Failed to review file: {}", e);
                    // Continue with other files
                }
            }
        }

        Ok(results)
    }
}
```

## Git Integration

Now let's add Git integration to analyze repository changes:

```rust
use git2::{Repository, Diff, DiffOptions};
use std::path::Path;

pub struct GitAnalyzer {
    repo: Repository,
}

impl GitAnalyzer {
    pub fn new(repo_path: &Path) -> Result<Self, git2::Error> {
        let repo = Repository::open(repo_path)?;
        Ok(Self { repo })
    }

    pub fn get_recent_commits(&self, count: usize) -> Result<Vec<CommitInfo>, git2::Error> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;
        
        let mut commits = Vec::new();
        
        for (i, oid) in revwalk.enumerate() {
            if i >= count {
                break;
            }
            
            let oid = oid?;
            let commit = self.repo.find_commit(oid)?;
            
            commits.push(CommitInfo {
                id: oid.to_string(),
                message: commit.message().unwrap_or("").to_string(),
                author: commit.author().name().unwrap_or("").to_string(),
                timestamp: commit.time().seconds(),
            });
        }
        
        Ok(commits)
    }

    pub fn get_diff_for_commit(&self, commit_id: &str) -> Result<Vec<FileDiff>, Box<dyn std::error::Error>> {
        let oid = git2::Oid::from_str(commit_id)?;
        let commit = self.repo.find_commit(oid)?;
        
        let tree = commit.tree()?;
        let parent_tree = if commit.parent_count() > 0 {
            Some(commit.parent(0)?.tree()?)
        } else {
            None
        };

        let mut diff_opts = DiffOptions::new();
        diff_opts.context_lines(3);
        
        let diff = self.repo.diff_tree_to_tree(
            parent_tree.as_ref(),
            Some(&tree),
            Some(&mut diff_opts)
        )?;

        let mut file_diffs = Vec::new();

        diff.foreach(
            &mut |delta, _progress| {
                if let Some(new_file) = delta.new_file().path() {
                    file_diffs.push(FileDiff {
                        filename: new_file.to_string_lossy().to_string(),
                        status: match delta.status() {
                            git2::Delta::Added => "added".to_string(),
                            git2::Delta::Deleted => "deleted".to_string(),
                            git2::Delta::Modified => "modified".to_string(),
                            _ => "unknown".to_string(),
                        },
                        diff_content: String::new(), // Will be filled by diff callback
                    });
                }
                true
            },
            None,
            None,
            None,
        )?;

        Ok(file_diffs)
    }

    pub async fn analyze_commit_with_claude(&self, commit_id: &str, review_bot: &CodeReviewBot) -> Result<CommitReview, Box<dyn std::error::Error>> {
        let commit_info = self.get_commit_info(commit_id)?;
        let file_diffs = self.get_diff_for_commit(commit_id)?;
        
        let mut file_reviews = Vec::new();
        
        for file_diff in file_diffs {
            // Skip non-code files
            if !self.is_code_file(&file_diff.filename) {
                continue;
            }

            // Get the full file content for review
            if let Ok(file_content) = self.get_file_content_at_commit(commit_id, &file_diff.filename) {
                let language = self.detect_language(&file_diff.filename);
                
                let review_request = CodeReviewRequest {
                    language,
                    code: file_content,
                    filename: file_diff.filename.clone(),
                    diff: Some(file_diff.diff_content.clone()),
                    context: Some(ReviewContext {
                        project_type: "repository".to_string(),
                        security_level: "medium".to_string(),
                        coding_standards: vec!["standard".to_string()],
                    }),
                };

                match review_bot.review_code(review_request).await {
                    Ok(review) => file_reviews.push(FileReview {
                        filename: file_diff.filename,
                        status: file_diff.status,
                        review_result: review,
                    }),
                    Err(e) => eprintln!("Failed to review {}: {}", file_diff.filename, e),
                }
            }
        }

        Ok(CommitReview {
            commit_info,
            file_reviews,
            overall_assessment: self.generate_overall_assessment(&file_reviews),
        })
    }

    fn get_commit_info(&self, commit_id: &str) -> Result<CommitInfo, git2::Error> {
        let oid = git2::Oid::from_str(commit_id)?;
        let commit = self.repo.find_commit(oid)?;
        
        Ok(CommitInfo {
            id: commit_id.to_string(),
            message: commit.message().unwrap_or("").to_string(),
            author: commit.author().name().unwrap_or("").to_string(),
            timestamp: commit.time().seconds(),
        })
    }

    fn get_file_content_at_commit(&self, commit_id: &str, filename: &str) -> Result<String, git2::Error> {
        let oid = git2::Oid::from_str(commit_id)?;
        let commit = self.repo.find_commit(oid)?;
        let tree = commit.tree()?;
        
        let entry = tree.get_path(Path::new(filename))?;
        let blob = self.repo.find_blob(entry.id())?;
        
        Ok(String::from_utf8_lossy(blob.content()).to_string())
    }

    fn is_code_file(&self, filename: &str) -> bool {
        let code_extensions = [
            "rs", "py", "js", "ts", "java", "cpp", "c", "h", "hpp",
            "go", "rb", "php", "swift", "kt", "scala", "cs"
        ];
        
        if let Some(extension) = Path::new(filename).extension() {
            if let Some(ext_str) = extension.to_str() {
                return code_extensions.contains(&ext_str);
            }
        }
        false
    }

    fn detect_language(&self, filename: &str) -> String {
        if let Some(extension) = Path::new(filename).extension() {
            match extension.to_str() {
                Some("rs") => "rust".to_string(),
                Some("py") => "python".to_string(),
                Some("js") => "javascript".to_string(),
                Some("ts") => "typescript".to_string(),
                Some("java") => "java".to_string(),
                Some("cpp") | Some("cc") | Some("cxx") => "cpp".to_string(),
                Some("c") => "c".to_string(),
                Some("go") => "go".to_string(),
                Some("rb") => "ruby".to_string(),
                Some("php") => "php".to_string(),
                _ => "unknown".to_string(),
            }
        } else {
            "unknown".to_string()
        }
    }

    fn generate_overall_assessment(&self, file_reviews: &[FileReview]) -> String {
        let total_files = file_reviews.len();
        let avg_score: f64 = file_reviews.iter()
            .map(|fr| fr.review_result.overall_score as f64)
            .sum::<f64>() / total_files as f64;

        let total_issues: usize = file_reviews.iter()
            .map(|fr| fr.review_result.issues.len())
            .sum();

        let security_concerns: usize = file_reviews.iter()
            .map(|fr| fr.review_result.security_concerns.len())
            .sum();

        format!(
            "Reviewed {} files. Average score: {:.1}/100. Found {} issues and {} security concerns.",
            total_files, avg_score, total_issues, security_concerns
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitInfo {
    pub id: String,
    pub message: String,
    pub author: String,
    pub timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileDiff {
    pub filename: String,
    pub status: String,
    pub diff_content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileReview {
    pub filename: String,
    pub status: String,
    pub review_result: CodeReviewResult,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitReview {
    pub commit_info: CommitInfo,
    pub file_reviews: Vec<FileReview>,
    pub overall_assessment: String,
}
```

## Web API for GitHub/GitLab Integration

Now let's create a web API that can be used with GitHub webhooks or GitLab CI:

```rust
use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::Json as ResponseJson,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub review_bot: Arc<CodeReviewBot>,
}

#[derive(Debug, Deserialize)]
pub struct GitHubWebhookPayload {
    pub action: String,
    pub pull_request: Option<PullRequest>,
    pub repository: Repository,
}

#[derive(Debug, Deserialize)]
pub struct PullRequest {
    pub number: u32,
    pub head: GitRef,
    pub base: GitRef,
    pub diff_url: String,
}

#[derive(Debug, Deserialize)]
pub struct GitRef {
    pub sha: String,
    #[serde(rename = "ref")]
    pub git_ref: String,
}

#[derive(Debug, Deserialize)]
pub struct Repository {
    pub name: String,
    pub full_name: String,
    pub clone_url: String,
}

#[derive(Debug, Deserialize)]
pub struct ReviewRequestQuery {
    pub repo_url: Option<String>,
    pub commit_id: Option<String>,
    pub pr_number: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ReviewResponse {
    pub success: bool,
    pub review_id: String,
    pub summary: String,
    pub details: Option<CommitReview>,
    pub error: Option<String>,
}

// GitHub webhook endpoint
pub async fn github_webhook_handler(
    State(state): State<AppState>,
    Json(payload): Json<GitHubWebhookPayload>,
) -> Result<ResponseJson<ReviewResponse>, (StatusCode, String)> {
    match payload.action.as_str() {
        "opened" | "synchronize" => {
            if let Some(pr) = payload.pull_request {
                // Review the pull request
                match review_pull_request(&state, &payload.repository, &pr).await {
                    Ok(review) => Ok(ResponseJson(ReviewResponse {
                        success: true,
                        review_id: uuid::Uuid::new_v4().to_string(),
                        summary: review.overall_assessment.clone(),
                        details: Some(review),
                        error: None,
                    })),
                    Err(e) => Ok(ResponseJson(ReviewResponse {
                        success: false,
                        review_id: uuid::Uuid::new_v4().to_string(),
                        summary: "Review failed".to_string(),
                        details: None,
                        error: Some(e.to_string()),
                    })),
                }
            } else {
                Err((StatusCode::BAD_REQUEST, "No pull request found".to_string()))
            }
        }
        _ => Ok(ResponseJson(ReviewResponse {
            success: true,
            review_id: uuid::Uuid::new_v4().to_string(),
            summary: "Action ignored".to_string(),
            details: None,
            error: None,
        })),
    }
}

// Manual review endpoint
pub async fn manual_review_handler(
    State(state): State<AppState>,
    Query(params): Query<ReviewRequestQuery>,
    Json(request): Json<CodeReviewRequest>,
) -> Result<ResponseJson<ReviewResponse>, (StatusCode, String)> {
    match state.review_bot.review_code(request).await {
        Ok(result) => Ok(ResponseJson(ReviewResponse {
            success: true,
            review_id: uuid::Uuid::new_v4().to_string(),
            summary: result.summary.clone(),
            details: None, // Individual file review
            error: None,
        })),
        Err(e) => Ok(ResponseJson(ReviewResponse {
            success: false,
            review_id: uuid::Uuid::new_v4().to_string(),
            summary: "Review failed".to_string(),
            details: None,
            error: Some(e.to_string()),
        })),
    }
}

// Health check endpoint
pub async fn health_check() -> ResponseJson<serde_json::Value> {
    ResponseJson(serde_json::json!({
        "status": "healthy",
        "service": "code-review-bot",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn review_pull_request(
    state: &AppState,
    repo: &Repository,
    pr: &PullRequest,
) -> Result<CommitReview, Box<dyn std::error::Error>> {
    // In a real implementation, you would:
    // 1. Clone the repository or use existing clone
    // 2. Fetch the specific commits
    // 3. Generate diff between base and head
    // 4. Review changed files
    
    // For this example, we'll simulate the process
    let commit_info = CommitInfo {
        id: pr.head.sha.clone(),
        message: format!("PR #{}: Review requested", pr.number),
        author: "github-user".to_string(),
        timestamp: chrono::Utc::now().timestamp(),
    };

    // This would normally analyze the actual diff
    let file_reviews = Vec::new(); // Placeholder

    Ok(CommitReview {
        commit_info,
        file_reviews,
        overall_assessment: "Pull request review completed".to_string(),
    })
}

pub fn create_app(review_bot: CodeReviewBot) -> Router {
    let state = AppState {
        review_bot: Arc::new(review_bot),
    };

    Router::new()
        .route("/health", get(health_check))
        .route("/webhooks/github", post(github_webhook_handler))
        .route("/review", post(manual_review_handler))
        .with_state(state)
        .layer(
            tower::ServiceBuilder::new()
                .layer(tower_http::cors::CorsLayer::permissive())
                .layer(tower_http::trace::TraceLayer::new_for_http())
        )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::init();

    // Create the review bot
    let review_bot = CodeReviewBot::new()?;

    // Create the web app
    let app = create_app(review_bot);

    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Code Review Bot server running on http://0.0.0.0:3000");

    axum::serve(listener, app).await?;

    Ok(())
}
```

## Configuration and Customization

Create a configuration system to customize the review bot:

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct BotConfig {
    pub claude: ClaudeConfig,
    pub git: GitConfig,
    pub rules: ReviewRules,
    pub integrations: IntegrationConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub model: String,
    pub timeout_secs: u64,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GitConfig {
    pub default_branch: String,
    pub max_diff_size: usize,
    pub ignore_patterns: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReviewRules {
    pub language_configs: HashMap<String, LanguageRules>,
    pub security_level: String,
    pub fail_on_security_issues: bool,
    pub max_issues_per_file: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LanguageRules {
    pub enabled: bool,
    pub custom_rules: Vec<String>,
    pub ignore_rules: Vec<String>,
    pub severity_overrides: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrationConfig {
    pub github: Option<GitHubConfig>,
    pub gitlab: Option<GitLabConfig>,
    pub webhook_secret: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GitHubConfig {
    pub token: String,
    pub app_id: Option<u64>,
    pub private_key_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GitLabConfig {
    pub token: String,
    pub base_url: String,
}

impl BotConfig {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: BotConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn load_from_env() -> Result<Self, Box<dyn std::error::Error>> {
        // Load configuration from environment variables
        let config = BotConfig {
            claude: ClaudeConfig {
                model: std::env::var("CLAUDE_MODEL")
                    .unwrap_or_else(|_| "claude-3-opus-20240229".to_string()),
                timeout_secs: std::env::var("CLAUDE_TIMEOUT")
                    .unwrap_or_else(|_| "120".to_string())
                    .parse()?,
                max_tokens: std::env::var("CLAUDE_MAX_TOKENS")
                    .ok()
                    .and_then(|s| s.parse().ok()),
            },
            git: GitConfig {
                default_branch: std::env::var("GIT_DEFAULT_BRANCH")
                    .unwrap_or_else(|_| "main".to_string()),
                max_diff_size: std::env::var("GIT_MAX_DIFF_SIZE")
                    .unwrap_or_else(|_| "10000".to_string())
                    .parse()?,
                ignore_patterns: std::env::var("GIT_IGNORE_PATTERNS")
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            },
            rules: ReviewRules {
                language_configs: HashMap::new(),
                security_level: std::env::var("SECURITY_LEVEL")
                    .unwrap_or_else(|_| "medium".to_string()),
                fail_on_security_issues: std::env::var("FAIL_ON_SECURITY")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()?,
                max_issues_per_file: std::env::var("MAX_ISSUES_PER_FILE")
                    .unwrap_or_else(|_| "20".to_string())
                    .parse()?,
            },
            integrations: IntegrationConfig {
                github: std::env::var("GITHUB_TOKEN").ok().map(|token| GitHubConfig {
                    token,
                    app_id: std::env::var("GITHUB_APP_ID").ok().and_then(|s| s.parse().ok()),
                    private_key_path: std::env::var("GITHUB_PRIVATE_KEY_PATH").ok(),
                }),
                gitlab: std::env::var("GITLAB_TOKEN").ok().map(|token| GitLabConfig {
                    token,
                    base_url: std::env::var("GITLAB_BASE_URL")
                        .unwrap_or_else(|_| "https://gitlab.com".to_string()),
                }),
                webhook_secret: std::env::var("WEBHOOK_SECRET").ok(),
            },
        };

        Ok(config)
    }
}
```

## Testing and Deployment

Finally, let's add comprehensive testing:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rust_code_review() {
        let bot = CodeReviewBot::new().unwrap();
        
        let request = CodeReviewRequest {
            language: "rust".to_string(),
            code: r#"
fn main() {
    let result = std::fs::read_to_string("file.txt").unwrap();
    println!("{}", result);
}
"#.to_string(),
            filename: "main.rs".to_string(),
            diff: None,
            context: Some(ReviewContext {
                project_type: "cli".to_string(),
                security_level: "high".to_string(),
                coding_standards: vec!["standard".to_string()],
            }),
        };

        let result = bot.review_code(request).await;
        assert!(result.is_ok());
        
        let review = result.unwrap();
        assert!(!review.issues.is_empty()); // Should find the unwrap() issue
    }

    #[tokio::test]
    async fn test_python_security_review() {
        let bot = CodeReviewBot::new().unwrap();
        
        let request = CodeReviewRequest {
            language: "python".to_string(),
            code: r#"
import os
user_input = input("Enter command: ")
os.system(user_input)  # Security vulnerability
"#.to_string(),
            filename: "script.py".to_string(),
            diff: None,
            context: Some(ReviewContext {
                project_type: "web-api".to_string(),
                security_level: "high".to_string(),
                coding_standards: vec!["pep8".to_string()],
            }),
        };

        let result = bot.review_code(request).await;
        assert!(result.is_ok());
        
        let review = result.unwrap();
        assert!(!review.security_concerns.is_empty()); // Should find security issue
    }

    #[test]
    fn test_language_detection() {
        let analyzer = GitAnalyzer::new(Path::new(".")).unwrap();
        
        assert_eq!(analyzer.detect_language("main.rs"), "rust");
        assert_eq!(analyzer.detect_language("script.py"), "python");
        assert_eq!(analyzer.detect_language("app.js"), "javascript");
    }

    #[test]
    fn test_config_loading() {
        let config = BotConfig::load_from_env().unwrap();
        assert!(!config.claude.model.is_empty());
        assert!(config.claude.timeout_secs > 0);
    }
}
```

## Usage Examples

### CLI Usage

```bash
# Review a single file
cargo run -- review --file src/main.rs --language rust

# Review a git commit
cargo run -- review --commit abc123 --repo /path/to/repo

# Start the web server
cargo run -- server --port 3000

# Review with custom rules
cargo run -- review --file script.py --config custom-rules.toml
```

### GitHub Integration

Add a webhook to your GitHub repository pointing to `https://your-bot.com/webhooks/github`.

### Manual API Usage

```bash
# Review code via API
curl -X POST http://localhost:3000/review \
  -H "Content-Type: application/json" \
  -d '{
    "language": "rust",
    "code": "fn main() { println!(\"Hello\"); }",
    "filename": "main.rs"
  }'
```

## Deployment

Deploy using Docker:

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/code-review-bot /usr/local/bin/
EXPOSE 3000
CMD ["code-review-bot", "server"]
```

This comprehensive code review bot demonstrates the power of the claude-sdk-rs SDK for building sophisticated AI-powered tools. It can be customized and extended for specific use cases and integrated into existing development workflows.