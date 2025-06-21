# Part 5: Tool Integration

Claude's real power shines when you enable it to interact with external tools and systems. This tutorial covers how to safely and effectively integrate tools with the claude-sdk-rs SDK, including filesystem access, bash commands, and Model Context Protocol (MCP) servers.

## What are Tools?

Tools extend Claude's capabilities beyond text generation by allowing it to:
- Execute system commands
- Access and modify files
- Query databases
- Interact with APIs
- Perform calculations
- And much more

The claude-sdk-rs SDK provides fine-grained control over which tools Claude can access, ensuring security while maximizing functionality.

## Tool Permission System

The SDK uses the `ToolPermission` enum to define exactly what Claude can do:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .allowed_tools(vec![
            // Allow specific MCP tools
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
            ToolPermission::mcp("database", "query").to_cli_format(),
            
            // Allow specific bash commands
            ToolPermission::bash("ls").to_cli_format(),
            ToolPermission::bash("git status").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);
    let client = Client::new(config);

    let response = client
        .query("What files are in the current directory?")
        .send()
        .await?;

    println!("Response: {}", response);
    Ok(())
}
```

## Basic Tool Configuration

### 1. Simple Tool Access

For basic filesystem and command access:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .system_prompt("You are a helpful development assistant")
        .allowed_tools(vec![
            // Filesystem access
            ToolPermission::mcp("filesystem", "*").to_cli_format(),
            
            // Basic commands
            ToolPermission::bash("ls").to_cli_format(),
            ToolPermission::bash("pwd").to_cli_format(),
            ToolPermission::bash("cat").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);
    let client = Client::new(config);

    // Claude can now read files and list directories
    let response = client
        .query("Show me the contents of the README.md file")
        .send()
        .await?;

    println!("{}", response);
    Ok(())
}
```

### 2. Development Tools Configuration

For software development workflows:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

async fn create_dev_client() -> Client {
    let config = Config::builder()
        .system_prompt(
            "You are an expert software engineer. You can read files, \
             run tests, and execute build commands to help with development tasks."
        )
        .allowed_tools(vec![
            // File system access
            ToolPermission::mcp("filesystem", "*").to_cli_format(),
            
            // Git commands
            ToolPermission::bash("git status").to_cli_format(),
            ToolPermission::bash("git diff").to_cli_format(),
            ToolPermission::bash("git log").to_cli_format(),
            
            // Build and test commands
            ToolPermission::bash("cargo build").to_cli_format(),
            ToolPermission::bash("cargo test").to_cli_format(),
            ToolPermission::bash("cargo clippy").to_cli_format(),
            
            // Node.js commands
            ToolPermission::bash("npm install").to_cli_format(),
            ToolPermission::bash("npm test").to_cli_format(),
            ToolPermission::bash("npm run").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);
    Client::new(config)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = create_dev_client().await;
    
    let response = client
        .query("Check the git status and run the tests for this project")
        .send()
        .await?;
    
    println!("Development Report:\n{}", response);
    Ok(())
}
```

## MCP (Model Context Protocol) Integration

MCP enables Claude to interact with specialized servers that provide domain-specific tools.

### Setting Up MCP Servers

First, create an MCP configuration file:

```json
{
  "servers": {
    "filesystem": {
      "command": "mcp-filesystem-server",
      "args": ["--root", "/safe/project/directory"]
    },
    "database": {
      "command": "mcp-database-server", 
      "args": ["--connection", "postgresql://user:pass@localhost/db"]
    },
    "web": {
      "command": "mcp-web-server",
      "args": ["--allowed-domains", "api.example.com"]
    }
  }
}
```

### Using MCP in Your Application

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .mcp_config(PathBuf::from("./mcp-config.json"))
        .allowed_tools(vec![
            // Database operations
            ToolPermission::mcp("database", "query").to_cli_format(),
            ToolPermission::mcp("database", "schema").to_cli_format(),
            
            // File operations (sandboxed to safe directory)
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
            ToolPermission::mcp("filesystem", "write").to_cli_format(),
            
            // Web requests (limited to allowed domains)
            ToolPermission::mcp("web", "fetch").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);

    let response = client
        .query(
            "Query the users table to get the top 10 most recent signups, \
             then save the results to a JSON file"
        )
        .send()
        .await?;

    println!("Database Query Results:\n{}", response);
    Ok(())
}
```

## Bash Command Integration

### Safe Command Execution

Always use specific commands rather than wildcards:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

fn create_safe_bash_client() -> Client {
    let config = Config::builder()
        .system_prompt("You can only run the specifically allowed commands")
        .allowed_tools(vec![
            // Safe read-only commands
            ToolPermission::bash("ls").to_cli_format(),
            ToolPermission::bash("pwd").to_cli_format(),
            ToolPermission::bash("whoami").to_cli_format(),
            ToolPermission::bash("date").to_cli_format(),
            
            // Specific development commands
            ToolPermission::bash("cargo --version").to_cli_format(),
            ToolPermission::bash("rustc --version").to_cli_format(),
            
            // Version control (read-only)
            ToolPermission::bash("git status").to_cli_format(),
            ToolPermission::bash("git log --oneline -10").to_cli_format(),
        ])
        .build();
    Client::new(config)
}
```

### Development Environment Setup

For development environments where more access is needed:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

fn create_dev_bash_client() -> Client {
    let config = Config::builder()
        .system_prompt(
            "You are a development assistant. You can run build commands, \
             tests, and development tools, but be careful with destructive operations."
        )
        .allowed_tools(vec![
            // Build system
            ToolPermission::bash("cargo build").to_cli_format(),
            ToolPermission::bash("cargo test").to_cli_format(),
            ToolPermission::bash("cargo check").to_cli_format(),
            ToolPermission::bash("cargo clippy").to_cli_format(),
            ToolPermission::bash("cargo fmt").to_cli_format(),
            
            // Package management
            ToolPermission::bash("npm install").to_cli_format(),
            ToolPermission::bash("npm test").to_cli_format(),
            ToolPermission::bash("npm run build").to_cli_format(),
            
            // Git operations (careful with these)
            ToolPermission::bash("git add").to_cli_format(),
            ToolPermission::bash("git commit").to_cli_format(),
            ToolPermission::bash("git push").to_cli_format(),
        ])
        .build();
    Client::new(config)
}
```

## Practical Tool Examples

### 1. Code Review Assistant

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

async fn create_code_review_assistant() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .system_prompt(
            "You are a code review assistant. Analyze code files, \
             run static analysis tools, and provide feedback on code quality."
        )
        .allowed_tools(vec![
            // File access
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
            
            // Analysis tools
            ToolPermission::bash("cargo clippy").to_cli_format(),
            ToolPermission::bash("cargo audit").to_cli_format(),
            ToolPermission::bash("cargo outdated").to_cli_format(),
            
            // Git information
            ToolPermission::bash("git diff").to_cli_format(),
            ToolPermission::bash("git show").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);

    let response = client
        .query(
            "Review the changes in the current git branch. \
             Check for potential issues, run clippy, and provide suggestions."
        )
        .send()
        .await?;

    println!("Code Review:\n{}", response);
    Ok(())
}
```

### 2. Project Analysis Tool

```rust
use claude_sdk_rs::{Client, Config, ToolPermission, StreamFormat};

async fn analyze_project() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .stream_format(StreamFormat::Json)
        .system_prompt(
            "You are a project analyst. Examine project structure, \
             dependencies, and provide insights about the codebase."
        )
        .allowed_tools(vec![
            // Project structure
            ToolPermission::bash("find . -type f -name '*.rs' | head -20").to_cli_format(),
            ToolPermission::bash("find . -type f -name '*.toml'").to_cli_format(),
            
            // Dependencies
            ToolPermission::bash("cargo tree").to_cli_format(),
            ToolPermission::bash("cargo metadata").to_cli_format(),
            
            // Statistics
            ToolPermission::bash("wc -l **/*.rs").to_cli_format(),
            ToolPermission::bash("git log --oneline | wc -l").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);

    let response = client
        .query(
            "Analyze this Rust project. What's the structure, \
             what are the main dependencies, and what insights \
             can you provide about the codebase?"
        )
        .send_full()
        .await?;

    println!("Project Analysis:\n{}", response.content);
    println!("Analysis cost: ${:.6}", response.metadata.unwrap().cost_usd.unwrap_or(0.0));
    
    Ok(())
}
```

### 3. Deployment Assistant

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

async fn create_deployment_assistant() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .system_prompt(
            "You are a deployment assistant. You can run builds, \
             execute tests, and deploy applications safely."
        )
        .allowed_tools(vec![
            // Pre-deployment checks
            ToolPermission::bash("cargo test").to_cli_format(),
            ToolPermission::bash("cargo clippy -- -D warnings").to_cli_format(),
            
            // Build
            ToolPermission::bash("cargo build --release").to_cli_format(),
            
            // Docker operations (if using Docker)
            ToolPermission::bash("docker build").to_cli_format(),
            ToolPermission::bash("docker push").to_cli_format(),
            
            // Deployment commands
            ToolPermission::bash("kubectl apply").to_cli_format(),
            ToolPermission::bash("kubectl get pods").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);

    let response = client
        .query(
            "Prepare this application for deployment: \
             1. Run all tests \
             2. Check for any clippy warnings \
             3. Build the release version \
             4. If everything passes, build and push Docker image"
        )
        .send()
        .await?;

    println!("Deployment Process:\n{}", response);
    Ok(())
}
```

## Error Handling for Tool Operations

Tool operations can fail in various ways. Handle them gracefully:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission, Error};

async fn handle_tool_errors() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .allowed_tools(vec![
            ToolPermission::bash("cargo test").to_cli_format(),
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
        ])
        .timeout_secs(120) // Longer timeout for tool operations
        .build();
    let client = Client::new(config);

    match client
        .query("Run the tests and analyze any failures")
        .send()
        .await
    {
        Ok(response) => {
            println!("Tool operation completed:\n{}", response);
        }
        Err(Error::Timeout) => {
            eprintln!("Tool operation timed out. Consider increasing timeout or breaking into smaller tasks.");
        }
        Err(Error::ProcessError(msg)) => {
            eprintln!("Tool execution failed: {}", msg);
            // Could retry with different approach
        }
        Err(e) => {
            eprintln!("Other error: {:?}", e);
        }
    }

    Ok(())
}
```

## Security Considerations

### 1. Principle of Least Privilege

Only grant the minimum permissions necessary:

```rust
// Good: Specific permissions
let config = Config::builder()
    .allowed_tools(vec![
        ToolPermission::bash("cargo test").to_cli_format(),
        ToolPermission::mcp("filesystem", "read").to_cli_format(),
    ])
    .build();
    let client = Client::new(config);

// Dangerous: Too broad permissions
let config = Config::builder()
    .allowed_tools(vec![
        ToolPermission::All.to_cli_format(), // Avoid this!
    ])
    .build();
    let client = Client::new(config);
```

### 2. Sandboxed Environments

Use MCP servers with restricted access:

```json
{
  "servers": {
    "filesystem": {
      "command": "mcp-filesystem-server",
      "args": [
        "--root", "/safe/project/directory",
        "--read-only", "true"
      ]
    }
  }
}
```

### 3. Input Validation

Validate and sanitize any user input that might influence tool commands:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

fn sanitize_filename(filename: &str) -> Option<String> {
    // Only allow alphanumeric, dots, hyphens, underscores
    if filename.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_') {
        Some(filename.to_string())
    } else {
        None
    }
}

async fn safe_file_operation(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let safe_filename = sanitize_filename(filename)
        .ok_or_else(|| claude_sdk_rs::Error::ProcessError("Invalid filename".to_string()))?;
    
    let config = Config::builder()
        .allowed_tools(vec![
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);

    let query = format!("Read and analyze the file: {}", safe_filename);
    let response = client.query(&query).send().await?;
    
    println!("File analysis: {}", response);
    Ok(())
}
```

## Streaming with Tools

Tools work seamlessly with streaming responses:

```rust
use claude_sdk_rs::{Client, ToolPermission, StreamFormat, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .stream_format(StreamFormat::StreamJson)
        .allowed_tools(vec![
            ToolPermission::bash("cargo test").to_cli_format(),
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);

    let mut stream = client
        .query("Run tests and analyze the results, explaining each step")
        .stream()
        .await?;

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Message::Tool { name, parameters, .. } => {
                println!("\n🔧 Executing tool: {} with params: {}", name, parameters);
            }
            Message::ToolResult { tool_name, result, .. } => {
                println!("✅ Tool {} completed: {}", tool_name, result);
            }
            Message::Result { stats, .. } => {
                println!("\n\n📊 Final stats: ${:.6} total cost", stats.total_cost_usd);
            }
            _ => {}
        }
    }

    Ok(())
}
```

## Advanced Tool Integration Patterns

### 1. Conditional Tool Access

Enable different tools based on context:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission};

enum Environment {
    Development,
    Staging,
    Production,
}

fn create_client_for_env(env: Environment) -> Client {
    let base_tools = vec![
        ToolPermission::mcp("filesystem", "read").to_cli_format(),
        ToolPermission::bash("git status").to_cli_format(),
    ];

    let mut allowed_tools = base_tools;

    match env {
        Environment::Development => {
            allowed_tools.extend(vec![
                ToolPermission::bash("cargo test").to_cli_format(),
                ToolPermission::bash("cargo build").to_cli_format(),
                ToolPermission::mcp("filesystem", "write").to_cli_format(),
            ]);
        }
        Environment::Staging => {
            allowed_tools.extend(vec![
                ToolPermission::bash("cargo test").to_cli_format(),
            ]);
        }
        Environment::Production => {
            // Only read-only tools in production
        }
    }

    let config = Config::builder()
        .allowed_tools(allowed_tools)
        .build()
}
```

### 2. Tool Result Processing

Process and validate tool results:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission, StreamFormat};

async fn process_tool_results() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .stream_format(StreamFormat::Json)
        .allowed_tools(vec![
            ToolPermission::bash("cargo test").to_cli_format(),
        ])
        .build();
    let client = Client::new(config);

    let response = client
        .query("Run cargo test and tell me if all tests pass")
        .send_full()
        .await?;

    // Check if tests passed based on response content
    if response.content.contains("test result: ok") {
        println!("✅ All tests passed!");
    } else if response.content.contains("test result: FAILED") {
        println!("❌ Some tests failed!");
        // Could trigger additional actions
    }

    println!("Full response: {}", response.content);
    Ok(())
}
```

## Testing Tool Integration

Create tests for your tool-enabled workflows:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safe_tool_config() {
        let config = Config::builder()
            .allowed_tools(vec![
                ToolPermission::bash("echo hello").to_cli_format(),
            ])
            .build();
    let client = Client::new(config);

        let response = client
            .query("Execute the echo command")
            .send()
            .await;

        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_filesystem_access() {
        let config = Config::builder()
            .allowed_tools(vec![
                ToolPermission::mcp("filesystem", "read").to_cli_format(),
            ])
            .build();
    let client = Client::new(config);

        let response = client
            .query("List files in current directory")
            .send()
            .await;

        assert!(response.is_ok());
    }
}
```

## Best Practices Summary

1. **Start Restrictive**: Begin with minimal permissions and add tools as needed
2. **Be Specific**: Use exact command permissions rather than wildcards
3. **Validate Input**: Always sanitize user input that affects tool commands
4. **Use Sandboxing**: Leverage MCP servers with restricted environments
5. **Monitor Usage**: Track tool usage and costs in production
6. **Test Thoroughly**: Test tool integrations in safe environments first
7. **Handle Errors**: Implement proper error handling for tool failures
8. **Document Permissions**: Clearly document why each tool permission is needed

## Next Steps

Tool integration opens up endless possibilities for Claude applications. Consider exploring:

- Custom MCP server development for domain-specific tools
- Integration with CI/CD pipelines
- Database query and analysis workflows
- File processing and transformation tasks
- System monitoring and alerting

The `with_tools.rs` example in `claude-sdk-rs/examples/` provides a simple starting point for experimentation.

Remember: with great power comes great responsibility. Always prioritize security when enabling tool access!