# Command Line SDK Tutorial

This tutorial will guide you through using the Claude Code SDK from the command line, including how to use slash commands and various features.

## Prerequisites

Before you begin, make sure you have:

1. **Claude Code CLI installed**: `npm install -g @anthropic-ai/claude-code`
2. **Anthropic API key**: Set the `ANTHROPIC_API_KEY` environment variable
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

## Basic Usage

### Simple Prompts

The most basic way to use Claude Code is with the `-p` (print) flag for non-interactive mode:

```bash
# Ask Claude to write code
claude -p "Write a function to calculate Fibonacci numbers"

# Pipe input to Claude
echo "Explain this code: def add(a, b): return a + b" | claude -p

# Get help with a specific task
claude -p "How do I sort a list in Python?"
```

### Output Formats

You can control how Claude responds using different output formats:

```bash
# Default text output
claude -p "Generate a hello world function"

# JSON output with metadata (cost, duration, session ID)
claude -p "Generate a hello world function" --output-format json

# Streaming JSON for real-time updates
claude -p "Build a React component" --output-format stream-json
```

## Working with Files

### Reading Files

```bash
# Analyze a single file
cat mycode.py | claude -p "Review this code for bugs"

# Process multiple files
for file in *.js; do
    echo "Processing $file..."
    claude -p "Add JSDoc comments to this file:" < "$file" > "${file}.documented"
done
```

### Batch Processing

```bash
# Find and fix TODOs in Python files
grep -l "TODO" *.py | while read file; do
    claude -p "Fix all TODO items in this file" < "$file"
done
```

## Multi-Turn Conversations

### Continuing Conversations

```bash
# Start a conversation
claude -p "Create a Python web server"

# Continue the most recent conversation
claude --continue "Now add authentication"

# Continue with a specific prompt
claude -p --continue "Add error handling"
```

### Session Management

```bash
# Start a session and save the ID
claude -p "Initialize a new project" --output-format json | jq -r '.session_id' > session.txt

# Resume a specific session
claude -p --resume "$(cat session.txt)" "Add unit tests"

# Resume in interactive mode
claude --resume 550e8400-e29b-41d4-a716-446655440000
```

## Using Slash Commands

While the documentation doesn't explicitly mention slash commands for the CLI SDK, you can use Claude Code interactively (without the `-p` flag) to access slash commands:

```bash
# Start interactive mode
claude

# Then use slash commands like:
# /help - Get help
# /model - Change model
# /settings - View settings
# /memory - Manage memory
# /exit - Exit the session
```

## Advanced Features

### Custom System Prompts

Guide Claude's behavior with custom instructions:

```bash
# Override the default system prompt
claude -p "Build a REST API" --system-prompt "You are a senior backend engineer. Focus on security, performance, and maintainability."

# Append to the default prompt
claude -p "Create a database schema" --append-system-prompt "Always include proper indexing and use PostgreSQL best practices."
```

### Tool Permissions

Control which tools Claude can use:

```bash
# Allow specific tools
claude -p "Analyze this project" --allowedTools "Read,Grep,Glob"

# Disallow specific tools
claude -p "Review code" --disallowedTools "Bash,Write"

# Allow specific bash commands
claude -p "Install dependencies" --allowedTools "Bash(npm install),Read"
```

### Limiting Turns

Control how many times Claude can iterate:

```bash
# Limit to 3 turns for simple tasks
claude -p "Fix the syntax errors" --max-turns 3

# More turns for complex tasks
claude -p "Refactor this entire module" --max-turns 10
```

## Practical Examples

### Code Review Workflow

```bash
#!/bin/bash
# review.sh - Automated code review script

# Review changes before committing
git diff --staged | claude -p "Review these changes for bugs, security issues, and best practices"

# Get suggestions for improvement
claude -p "Suggest improvements for this code" < main.py
```

### Documentation Generator

```bash
# Generate documentation for all Python files
for file in src/**/*.py; do
    echo "Documenting $file..."
    claude -p "Add comprehensive docstrings to all functions and classes" < "$file" > "$file.tmp"
    mv "$file.tmp" "$file"
done
```

### Test Generation

```bash
# Generate tests for a module
claude -p "Write comprehensive unit tests for this module" < app/models.py > tests/test_models.py
```

## Best Practices

1. **Use JSON output for scripts**:
   ```bash
   result=$(claude -p "Generate code" --output-format json)
   code=$(echo "$result" | jq -r '.result')
   cost=$(echo "$result" | jq -r '.total_cost_usd')
   echo "Generated code (cost: $cost USD)"
   ```

2. **Handle errors gracefully**:
   ```bash
   if ! claude -p "$prompt" 2>error.log; then
       echo "Error occurred:" >&2
       cat error.log >&2
       exit 1
   fi
   ```

3. **Add timeouts for long operations**:
   ```bash
   timeout 300 claude -p "Complex refactoring task" || echo "Timed out after 5 minutes"
   ```

4. **Use verbose mode for debugging**:
   ```bash
   claude -p "Debug this issue" --verbose
   ```

## Integration with Development Tools

### Git Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
# Automatically review changes before commit

changes=$(git diff --staged)
if [ -n "$changes" ]; then
    echo "$changes" | claude -p "Review for issues. Reply with 'OK' if good, or list problems" --output-format text
fi
```

### CI/CD Pipeline

```bash
# In your CI script
claude -p "Review PR changes and suggest improvements" \
  --system-prompt "Focus on performance, security, and maintainability" \
  --max-turns 5 \
  --output-format json
```

## Tips and Tricks

1. **Save common prompts**: Create aliases for frequently used commands
   ```bash
   alias claude-review='claude -p "Review this code for best practices and potential issues"'
   alias claude-test='claude -p "Write unit tests for this code"'
   ```

2. **Chain commands**: Use Claude's output as input for other tools
   ```bash
   claude -p "Generate SQL migration" | psql -d mydb
   ```

3. **Monitor costs**: Track your API usage
   ```bash
   claude -p "Task" --output-format json | jq '.total_cost_usd' >> costs.log
   ```

## Next Steps

- Explore the [TypeScript SDK](https://www.npmjs.com/package/@anthropic-ai/claude-code) for programmatic integration
- Try the [Python SDK](https://pypi.org/project/claude-code-sdk/) for Python applications
- Check out [MCP (Model Context Protocol)](https://docs.anthropic.com/en/docs/claude-code/claude-code-sdk-doc#mcp-configuration) for extending Claude with custom tools
- Read about [GitHub Actions integration](https://docs.anthropic.com/en/docs/claude-code/github-actions) for automated workflows

Remember to always respect rate limits and use session management for complex multi-step tasks to maintain context between operations.