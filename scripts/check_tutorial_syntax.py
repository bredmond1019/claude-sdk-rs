#!/usr/bin/env python3
"""
Script to check Rust syntax in tutorial code snippets.
This validates that code examples are syntactically correct without requiring compilation.
"""

import os
import re
import subprocess
import tempfile
import sys
from pathlib import Path

def extract_rust_code_blocks(file_path):
    """Extract all Rust code blocks from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all ```rust code blocks
    pattern = r'```rust\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    code_blocks = []
    for i, match in enumerate(matches):
        # Skip incomplete examples that are just fragments
        if any(fragment in match for fragment in [
            'async fn main() -> Result<(), Box<dyn std::error::Error>> {',
            'use claude_sdk_rs::{',
            '#[tokio::main]'
        ]):
            code_blocks.append((i, match))
    
    return code_blocks

def create_dummy_cargo_toml():
    """Create a dummy Cargo.toml for syntax checking."""
    return """[package]
name = "syntax-test"
version = "0.1.0"
edition = "2021"

[dependencies]
# Dummy dependencies for syntax checking
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
futures = "0.3"
"""

def create_dummy_lib_rs():
    """Create a dummy lib.rs with basic types to satisfy imports."""
    return """
// Dummy claude-sdk-rs types for syntax checking
pub struct Client;
pub struct Config;
pub struct SessionManager;
pub struct SessionBuilder;
pub struct SessionId;
pub enum StreamFormat { Text, Json, StreamJson }
pub enum StorageBackend { File(std::path::PathBuf), Sqlite(std::path::PathBuf) }
pub enum ToolPermission { Bash(String), Mcp(String, String) }
pub enum Message { 
    Assistant { content: String }, 
    Tool { name: String, parameters: serde_json::Value },
    ToolResult { tool_name: String, result: serde_json::Value },
    Result { stats: ConversationStats },
    Init,
    User { content: String },
    System { content: String },
}
pub struct ConversationStats { pub total_cost_usd: f64 }
pub struct ClaudeResponse { pub content: String, pub metadata: Option<ResponseMetadata> }
pub struct ResponseMetadata { pub cost_usd: Option<f64> }
pub type Error = Box<dyn std::error::Error>;

impl Client {
    pub fn new(_config: Config) -> Self { Self }
    pub fn query(&self, _query: &str) -> QueryBuilder { QueryBuilder }
}

impl Config {
    pub fn default() -> Self { Self }
    pub fn builder() -> ConfigBuilder { ConfigBuilder }
}

impl ConfigBuilder {
    pub fn stream_format(self, _format: StreamFormat) -> Self { self }
    pub fn timeout_secs(self, _secs: u64) -> Self { self }
    pub fn model<S: Into<String>>(self, _model: S) -> Self { self }
    pub fn system_prompt<S: Into<String>>(self, _prompt: S) -> Self { self }
    pub fn allowed_tools(self, _tools: Vec<String>) -> Self { self }
    pub fn max_tokens(self, _tokens: usize) -> Self { self }
    pub fn mcp_config(self, _path: std::path::PathBuf) -> Self { self }
    pub fn build(self) -> Config { Config }
}

pub struct ConfigBuilder;
pub struct QueryBuilder;

impl QueryBuilder {
    pub fn session(self, _id: SessionId) -> Self { self }
    pub fn system_prompt<S: Into<String>>(self, _prompt: S) -> Self { self }
    pub async fn send(self) -> Result<String, Error> { Ok("response".to_string()) }
    pub async fn send_full(self) -> Result<ClaudeResponse, Error> { 
        Ok(ClaudeResponse { 
            content: "response".to_string(), 
            metadata: None 
        }) 
    }
    pub async fn stream(self) -> Result<MessageStream, Error> { Ok(MessageStream) }
}

pub struct MessageStream;

impl futures::Stream for MessageStream {
    type Item = Result<Message, Error>;
    fn poll_next(self: std::pin::Pin<&mut Self>, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Ready(None)
    }
}

impl SessionManager {
    pub fn new() -> Self { Self }
    pub fn with_storage(_backend: StorageBackend) -> Self { Self }
    pub async fn with_storage_async(_backend: StorageBackend) -> Result<Self, Error> { Ok(Self) }
    pub fn create_session(&self) -> SessionBuilder { SessionBuilder }
    pub async fn get(&self, _id: &SessionId) -> Result<Option<Session>, Error> { Ok(None) }
    pub async fn list(&self) -> Result<Vec<SessionId>, Error> { Ok(vec![]) }
    pub async fn resume(&self, _id: &SessionId) -> Result<Session, Error> { Ok(Session) }
    pub async fn delete(&self, _id: &SessionId) -> Result<(), Error> { Ok(()) }
    pub async fn clear(&self) -> Result<(), Error> { Ok(()) }
}

impl SessionBuilder {
    pub fn new() -> Self { Self }
    pub fn with_id<S: Into<String>>(id: S) -> Self { Self }
    pub fn with_system_prompt<S: Into<String>>(self, _prompt: S) -> Self { self }
    pub fn with_metadata<S: Into<String>>(self, _key: S, _value: serde_json::Value) -> Self { self }
    pub async fn build(self) -> Result<Session, Error> { Ok(Session) }
}

impl SessionId {
    pub fn new<S: Into<String>>(_id: S) -> Self { Self }
}

impl ToolPermission {
    pub fn bash<S: Into<String>>(cmd: S) -> Self { Self::Bash(cmd.into()) }
    pub fn mcp<S: Into<String>>(server: S, tool: S) -> Self { Self::Mcp(server.into(), tool.into()) }
    pub fn to_cli_format(&self) -> String { "tool".to_string() }
}

pub struct Session;

impl Session {
    pub fn id(&self) -> &SessionId { &SessionId }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "session-id")
    }
}
"""

def make_code_testable(code):
    """Make a code snippet testable by wrapping it if needed."""
    # Replace claude_sdk_rs with crate for local testing
    code = code.replace("use claude_sdk_rs::", "use crate::")
    
    # If it's already a complete program, return as-is
    if '#[tokio::main]' in code or 'fn main()' in code:
        return code
    
    # If it's just imports and types, wrap in a simple main
    if not ('async fn' in code or 'fn ' in code or 'impl ' in code or 'struct ' in code):
        return f"""
{code}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    println!("Code compiles successfully!");
    Ok(())
}}
"""
    
    # For function definitions, add a main that can call them
    return f"""
{code}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    println!("Functions compile successfully!");
    Ok(())
}}
"""

def check_syntax(code, block_index, temp_dir):
    """Check syntax of a single code block."""
    # Create src directory
    src_dir = temp_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Write Cargo.toml
    cargo_toml = temp_dir / "Cargo.toml"
    cargo_toml.write_text(create_dummy_cargo_toml())
    
    # Write lib.rs with dummy types
    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text(create_dummy_lib_rs())
    
    # Write the code to main.rs
    main_rs = src_dir / "main.rs"
    testable_code = make_code_testable(code)
    main_rs.write_text(testable_code)
    
    # Try to check syntax
    try:
        result = subprocess.run(
            ["cargo", "check", "--quiet"], 
            cwd=temp_dir, 
            capture_output=True, 
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            return True, "OK"
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Syntax check timeout"
    except Exception as e:
        return False, f"Error: {e}"

def check_tutorial_file(file_path):
    """Check all code blocks in a tutorial file."""
    print(f"\n=== Checking {file_path.name} ===")
    
    code_blocks = extract_rust_code_blocks(file_path)
    if not code_blocks:
        print("  No testable Rust code blocks found")
        return True
    
    print(f"  Found {len(code_blocks)} code blocks to check")
    
    all_passed = True
    
    for block_index, code in code_blocks:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            success, message = check_syntax(code, block_index, temp_path)
            
            if success:
                print(f"  ✅ Block {block_index + 1}: OK")
            else:
                print(f"  ❌ Block {block_index + 1}: FAILED")
                print(f"     Error: {message[:200]}...")
                # Print the problematic code for debugging
                print(f"     Code preview:")
                lines = code.split('\n')[:3]
                for line in lines:
                    print(f"       {line}")
                total_lines = len(code.split('\n'))
                if total_lines > 3:
                    print(f"       ... ({total_lines - 3} more lines)")
                all_passed = False
    
    return all_passed

def main():
    # Find the project root (where this script is located)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tutorials_dir = project_root / "docs" / "tutorials"
    
    if not tutorials_dir.exists():
        print(f"Error: Tutorials directory not found at {tutorials_dir}")
        sys.exit(1)
    
    print("Checking syntax of all tutorial code snippets...")
    print(f"Project root: {project_root}")
    print(f"Tutorials directory: {tutorials_dir}")
    
    # Find all tutorial files
    tutorial_files = sorted(tutorials_dir.glob("*.md"))
    
    if not tutorial_files:
        print("No tutorial files found!")
        sys.exit(1)
    
    all_tutorials_passed = True
    
    for tutorial_file in tutorial_files:
        passed = check_tutorial_file(tutorial_file)
        if not passed:
            all_tutorials_passed = False
    
    print("\n" + "="*50)
    if all_tutorials_passed:
        print("✅ All tutorial code snippets have valid syntax!")
        sys.exit(0)
    else:
        print("❌ Some tutorial code snippets have syntax issues!")
        print("Please fix the issues above and run the check again.")
        sys.exit(1)

if __name__ == "__main__":
    main()