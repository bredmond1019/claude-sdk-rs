# Claude AI Examples

This directory contains comprehensive examples demonstrating the capabilities of the `claude-sdk-rs` SDK.

## Overview

The examples are organized to show progression from simple SDK usage to complete applications:

1. **Basic SDK Usage** (`01_basic_sdk.rs`) - Simple queries and configuration
2. **Session Management** (`02_sdk_sessions.rs`) - Using sessions for context
3. **Streaming Responses** (`03_streaming.rs`) - Real-time response processing
4. **Tool Integration** (`04_tools.rs`) - Using filesystem and bash tools
5. **Complete Application** (`05_complete_app.rs`) - Full CLI assistant

## Running the Examples

```bash
# Run individual examples
cargo run --example 01_basic_sdk
cargo run --example 02_sdk_sessions
cargo run --example 03_streaming
cargo run --example 04_tools
cargo run --example 05_complete_app

# Run the complete app with different modes
cargo run --example 05_complete_app chat
cargo run --example 05_complete_app dev
cargo run --example 05_complete_app analysis

# Run tests
cargo test
```

## Features Demonstrated

### Example 1: Basic SDK Usage
- Client initialization
- Simple queries
- Configuration options
- Different response formats
- Error handling

### Example 2: Session Management  
- Creating session IDs
- Maintaining conversation context
- Multi-session management
- Manual session tracking

### Example 3: Streaming Responses
- Real-time response processing
- Message type handling
- Cost and token tracking
- Tool integration in streams

### Example 4: Tool Integration
- MCP tool configuration
- Bash command execution
- Safe tool patterns
- Development workflows

### Example 5: Complete Application
- Interactive CLI interface
- Multiple operation modes
- Cost tracking
- Session switching
- Command system

## Example Structure

Each example includes:
- Comprehensive documentation
- Step-by-step code comments
- Error handling patterns
- Example output
- Helper functions

## Prerequisites

1. **Claude Code CLI** must be installed and authenticated
   ```bash
   # Install Claude Code CLI first
   # Then authenticate
   claude login
   ```

2. **Rust 1.70+** 
   ```bash
   rustc --version
   ```

3. **Environment Setup**
   - Examples work out of the box after authentication
   - Some examples use filesystem/git tools

## Testing

Run the test suite to verify examples work correctly:

```bash
cargo test
```

Tests validate:
- Client creation and configuration
- Session ID management
- Tool permission setup
- Helper function behavior
- Cost tracking logic

## Learning Path

**Recommended order:**

1. Start with `01_basic_sdk.rs` for SDK fundamentals
2. Learn session management with `02_sdk_sessions.rs`  
3. Explore real-time features in `03_streaming.rs`
4. Add tool capabilities with `04_tools.rs`
5. See it all together in `05_complete_app.rs`

Each example builds on previous concepts while introducing new functionality.

## Interactive Features

The examples also include:
- **Interactive shell script** (`run_examples.sh`) for guided exploration
- **Library comparison guide** (`LIBRARY_COMPARISON.md`) explaining architecture
- **Comprehensive tests** validating example correctness