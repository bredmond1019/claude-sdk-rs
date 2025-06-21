---
name: 🐛 Bug Report
about: Report a bug or unexpected behavior in the claude-sdk-rs SDK
title: '[BUG] '
labels: ['bug', 'triage']
assignees: ''
---

## Bug Description

**A clear and concise description of what the bug is.**

## Environment

- **SDK Version**: [e.g., 1.0.0]
- **Rust Version**: [e.g., 1.75.0]
- **Operating System**: [e.g., macOS 14.5, Ubuntu 22.04, Windows 11]
- **Architecture**: [e.g., x86_64, arm64]
- **Claude CLI Version**: [output of `claude --version`]

## Expected Behavior

**What you expected to happen.**

## Actual Behavior

**What actually happened instead.**

## Reproduction Steps

**Steps to reproduce the behavior:**

1. Go to '...'
2. Run '...'
3. See error

## Code Sample

**Minimal code example that reproduces the issue:**

```rust
// Please provide a minimal, complete example
use claude_ai::{Client, Config};

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    // Your reproduction code here
    Ok(())
}
```

## Error Output

**If applicable, paste the complete error message and stack trace:**

```
Error output here
```

## Additional Context

**Add any other context about the problem here.**

- Configuration details
- Network conditions
- Concurrent usage patterns
- Any workarounds you've tried

## Possible Solution

**If you have suggestions on how to fix this, please describe them here.**

## Impact

**How is this affecting your work?**

- [ ] Blocks development completely
- [ ] Significant inconvenience
- [ ] Minor annoyance
- [ ] Just noticed it

## Checklist

- [ ] I have searched for similar issues before creating this one
- [ ] I have included all relevant environment information
- [ ] I have provided a minimal reproduction example
- [ ] I have included complete error messages
- [ ] I have tested with the latest version of the SDK