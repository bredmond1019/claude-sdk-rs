//! # Example 05: Complete Application
//!
//! This example demonstrates building a complete application with `claude-sdk-rs`.
//! It combines multiple features to create a practical CLI assistant that:
//! - Manages sessions for different contexts
//! - Tracks costs manually
//! - Provides different modes of operation
//! - Integrates tools for development tasks

use claude_sdk_rs::{Client, Message, SessionId, StreamFormat, ToolPermission};
use futures::StreamExt;
use std::collections::HashMap;
use std::io::{self, Write};
use uuid::Uuid;

/// Simple cost tracker
struct CostTracker {
    total_cost: f64,
    session_costs: HashMap<String, f64>,
}

impl CostTracker {
    fn new() -> Self {
        Self {
            total_cost: 0.0,
            session_costs: HashMap::new(),
        }
    }

    fn record_cost(&mut self, session_id: &str, cost: f64) {
        self.total_cost += cost;
        *self
            .session_costs
            .entry(session_id.to_string())
            .or_insert(0.0) += cost;
    }

    fn get_session_cost(&self, session_id: &str) -> f64 {
        self.session_costs.get(session_id).copied().unwrap_or(0.0)
    }
}

/// Application modes
#[derive(Debug, Clone)]
enum AppMode {
    Chat,
    Development,
    Analysis,
}

/// Main application state
struct ClaudeAssistant {
    client: Client,
    mode: AppMode,
    current_session: SessionId,
    cost_tracker: CostTracker,
    sessions: HashMap<String, String>, // session_id -> description
}

impl ClaudeAssistant {
    fn new(mode: AppMode) -> claude_sdk_rs::Result<Self> {
        let client = match &mode {
            AppMode::Chat => create_chat_client()?,
            AppMode::Development => create_dev_client()?,
            AppMode::Analysis => create_analysis_client()?,
        };

        let session_id = SessionId::new(Uuid::new_v4().to_string());
        let mut sessions = HashMap::new();
        sessions.insert(
            session_id.as_str().to_string(),
            format!("{:?} Session", mode),
        );

        Ok(Self {
            client,
            mode,
            current_session: session_id,
            cost_tracker: CostTracker::new(),
            sessions,
        })
    }

    async fn run(&mut self) -> claude_sdk_rs::Result<()> {
        self.print_welcome();

        loop {
            print!("\n> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            // Handle commands
            match input {
                "/help" => self.show_help(),
                "/mode" => self.show_mode(),
                "/cost" => self.show_cost(),
                "/sessions" => self.list_sessions(),
                "/new" => self.new_session(),
                "/switch" => self.switch_session().await,
                "/quit" | "/exit" => break,
                _ if input.starts_with('/') => {
                    println!("Unknown command. Type /help for available commands.");
                }
                _ => {
                    // Process query
                    self.process_query(input).await?;
                }
            }
        }

        println!(
            "\nGoodbye! Total cost: ${:.6}",
            self.cost_tracker.total_cost
        );
        Ok(())
    }

    fn print_welcome(&self) {
        println!("🤖 Claude Assistant - {:?} Mode", self.mode);
        println!("{}", "=".repeat(50));
        println!("Type /help for commands, or start chatting!");

        match self.mode {
            AppMode::Chat => {
                println!("Chat mode: General conversation with Claude");
            }
            AppMode::Development => {
                println!("Development mode: Code assistance with file and tool access");
            }
            AppMode::Analysis => {
                println!("Analysis mode: Code review and project analysis");
            }
        }
    }

    fn show_help(&self) {
        println!("\nAvailable commands:");
        println!("  /help     - Show this help");
        println!("  /mode     - Show current mode");
        println!("  /cost     - Show cost information");
        println!("  /sessions - List all sessions");
        println!("  /new      - Create new session");
        println!("  /switch   - Switch to another session");
        println!("  /quit     - Exit the application");
    }

    fn show_mode(&self) {
        println!("\nCurrent mode: {:?}", self.mode);
        match self.mode {
            AppMode::Chat => {
                println!("  - No tool access");
                println!("  - General conversation");
            }
            AppMode::Development => {
                println!("  - File system access");
                println!("  - Development tools");
                println!("  - Git integration");
            }
            AppMode::Analysis => {
                println!("  - Read-only file access");
                println!("  - Code analysis tools");
                println!("  - Project inspection");
            }
        }
    }

    fn show_cost(&self) {
        println!("\n💰 Cost Information:");
        println!("  Total cost: ${:.6}", self.cost_tracker.total_cost);
        println!(
            "  Current session: ${:.6}",
            self.cost_tracker
                .get_session_cost(self.current_session.as_str())
        );

        if self.sessions.len() > 1 {
            println!("\n  Session breakdown:");
            for (id, desc) in &self.sessions {
                let cost = self.cost_tracker.get_session_cost(id);
                if cost > 0.0 {
                    println!("    {} - ${:.6}", desc, cost);
                }
            }
        }
    }

    fn list_sessions(&self) {
        println!("\n📁 Sessions:");
        for (id, desc) in &self.sessions {
            let marker = if id == self.current_session.as_str() {
                "▶"
            } else {
                " "
            };
            let cost = self.cost_tracker.get_session_cost(id);
            println!("{} {} (${:.6}) - {}", marker, &id[..8], cost, desc);
        }
    }

    fn new_session(&mut self) {
        print!("Session description: ");
        io::stdout().flush().unwrap();

        let mut desc = String::new();
        io::stdin().read_line(&mut desc).unwrap();
        let desc = desc.trim();

        let session_id = SessionId::new(Uuid::new_v4().to_string());
        self.sessions.insert(
            session_id.as_str().to_string(),
            if desc.is_empty() {
                format!("{:?} Session", self.mode)
            } else {
                desc.to_string()
            },
        );

        self.current_session = session_id;
        println!("Created new session: {}", self.current_session.as_str());
    }

    async fn switch_session(&mut self) {
        if self.sessions.len() <= 1 {
            println!("No other sessions to switch to.");
            return;
        }

        self.list_sessions();
        print!("\nEnter session ID (first 8 chars): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        for (id, _) in &self.sessions {
            if id.starts_with(input) {
                self.current_session = SessionId::new(id.clone());
                println!("Switched to session: {}", id);
                return;
            }
        }

        println!("Session not found.");
    }

    async fn process_query(&mut self, query: &str) -> claude_sdk_rs::Result<()> {
        // Stream the response
        let mut stream = self
            .client
            .query(query)
            .session(self.current_session.clone())
            .stream()
            .await?;

        let mut query_cost = 0.0;

        print!("\n");
        while let Some(message) = stream.next().await {
            match message? {
                Message::Assistant { content, meta } => {
                    print!("{}", content);
                    io::stdout().flush().unwrap();

                    if let Some(cost) = meta.cost_usd {
                        query_cost = cost;
                    }
                }
                Message::Tool { name, .. } => {
                    println!("\n🔧 Using tool: {}", name);
                }
                Message::ToolResult { tool_name, .. } => {
                    println!("✅ {} completed\n", tool_name);
                }
                Message::Result { stats, .. } => {
                    query_cost = stats.total_cost_usd;
                }
                _ => {}
            }
        }

        println!("\n");

        // Track cost
        if query_cost > 0.0 {
            self.cost_tracker
                .record_cost(self.current_session.as_str(), query_cost);
            println!("💰 Cost: ${:.6}", query_cost);
        }

        Ok(())
    }
}

// Client creation functions for different modes

fn create_chat_client() -> claude_sdk_rs::Result<Client> {
    Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .system_prompt("You are a helpful AI assistant.")
        .build()
}

fn create_dev_client() -> claude_sdk_rs::Result<Client> {
    Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .system_prompt(
            "You are a development assistant. Help with coding tasks, \
             debugging, and project management.",
        )
        .allowed_tools(vec![
            // File access
            ToolPermission::mcp("filesystem", "*").to_cli_format(),
            // Development tools
            ToolPermission::bash("cargo build").to_cli_format(),
            ToolPermission::bash("cargo test").to_cli_format(),
            ToolPermission::bash("cargo check").to_cli_format(),
            ToolPermission::bash("cargo clippy").to_cli_format(),
            // Git
            ToolPermission::bash("git status").to_cli_format(),
            ToolPermission::bash("git diff").to_cli_format(),
            ToolPermission::bash("git log").to_cli_format(),
        ])
        .build()
}

fn create_analysis_client() -> claude_sdk_rs::Result<Client> {
    Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .system_prompt(
            "You are a code analysis expert. Analyze code for quality, \
             security, and best practices.",
        )
        .allowed_tools(vec![
            // Read-only file access
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
            // Analysis tools
            ToolPermission::bash("find").to_cli_format(),
            ToolPermission::bash("grep").to_cli_format(),
            ToolPermission::bash("wc").to_cli_format(),
            ToolPermission::bash("cargo clippy").to_cli_format(),
        ])
        .build()
}

#[tokio::main]
async fn main() -> claude_sdk_rs::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let mode = if args.len() > 1 {
        match args[1].as_str() {
            "chat" => AppMode::Chat,
            "dev" => AppMode::Development,
            "analysis" => AppMode::Analysis,
            _ => {
                println!("Usage: {} [chat|dev|analysis]", args[0]);
                println!("  chat     - General conversation mode");
                println!("  dev      - Development mode with tools");
                println!("  analysis - Code analysis mode");
                return Ok(());
            }
        }
    } else {
        AppMode::Chat
    };

    let mut app = ClaudeAssistant::new(mode)?;
    app.run().await
}

// Example usage:
/*
$ cargo run --example 05_complete_app dev

🤖 Claude Assistant - Development Mode
==================================================
Type /help for commands, or start chatting!
Development mode: Code assistance with file and tool access

> What's the structure of this project?

🔧 Using tool: bash:find
✅ bash:find completed

This is a Rust workspace project with the following structure:

**Main Crates:**
- `claude-sdk-rs` - The main SDK for interacting with Claude
- `claude-sdk-rs-core` - Core types and configurations
- `claude-sdk-rs-runtime` - Runtime implementation for executing Claude
- `claude-sdk-rs-interactive` - Interactive CLI features
- `claude-sdk-rs-mcp` - Model Context Protocol support
- `claude-sdk-rs-macros` - Procedural macros

**Key Features:**
- Async/await support throughout
- Multiple response formats (Text, JSON, Streaming)
- Session management for conversations
- Tool integration for filesystem and bash commands
- Comprehensive error handling

The project follows Rust workspace best practices with shared dependencies and modular design.

💰 Cost: $0.001234

> /cost

💰 Cost Information:
  Total cost: $0.001234
  Current session: $0.001234

> /sessions

📁 Sessions:
▶ 550e8400 ($0.001234) - Development Session

> /help

Available commands:
  /help     - Show this help
  /mode     - Show current mode
  /cost     - Show cost information
  /sessions - List all sessions
  /new      - Create new session
  /switch   - Switch to another session
  /quit     - Exit the application

> /quit

Goodbye! Total cost: $0.001234
*/
