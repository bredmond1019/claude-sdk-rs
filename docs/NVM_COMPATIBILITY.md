# NVM Compatibility Guide

## Problem

The Claude CLI requires Node.js 18+ to function properly. When using nvm (Node Version Manager), the correct Node version must be activated before running the Claude CLI. However, when the SDK spawns the `claude` process, it doesn't inherit the nvm environment, leading to errors like:

```
Error: Process completed with non-zero exit code
stderr: file:///Users/.../node_modules/@anthropic-ai/claude-code/cli.js:319
```

## Solution

The SDK now supports specifying a custom Claude binary path via the `CLAUDE_BINARY` environment variable. We provide a wrapper script that handles nvm activation automatically.

## Usage

### Method 1: Using the Provided Wrapper Script

1. The wrapper script is located at `scripts/claude-wrapper.sh`
2. Set the environment variable before running your application:

```bash
export CLAUDE_BINARY="/path/to/claude-code-sdk-rs/scripts/claude-wrapper.sh"
cargo run
```

### Method 2: In Your Application

Set the environment variable programmatically:

```rust
use std::env;
use claude_sdk_rs::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set the wrapper path (adjust path as needed)
    env::set_var("CLAUDE_BINARY", "./scripts/claude-wrapper.sh");
    
    let client = Client::builder().build()?;
    let response = client.query("Hello").send().await?;
    println!("{}", response);
    
    Ok(())
}
```

### Method 3: System-wide Configuration

Add to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`):

```bash
export CLAUDE_BINARY="$HOME/path/to/claude-code-sdk-rs/scripts/claude-wrapper.sh"
```

## How It Works

1. The SDK checks for the `CLAUDE_BINARY` environment variable
2. If set, it uses that path instead of searching for `claude` in PATH
3. The wrapper script:
   - Loads the nvm environment
   - Switches to Node.js 18
   - Executes the actual Claude CLI with all arguments

## Alternative Solutions

### Install Node.js 18 System-wide

Instead of using nvm, install Node.js 18 directly:

```bash
# macOS with Homebrew
brew install node@18

# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Create Your Own Wrapper

You can create a custom wrapper tailored to your environment:

```bash
#!/bin/bash
# Custom wrapper example
export PATH="/usr/local/opt/node@18/bin:$PATH"
exec /usr/local/bin/claude "$@"
```

## Troubleshooting

1. **Permission Denied**: Ensure the wrapper script is executable:
   ```bash
   chmod +x scripts/claude-wrapper.sh
   ```

2. **Wrapper Not Found**: Use an absolute path for `CLAUDE_BINARY`

3. **Still Getting Node Errors**: Check that nvm is properly installed and Node 18 is available:
   ```bash
   nvm list
   nvm install 18  # if not installed
   ```