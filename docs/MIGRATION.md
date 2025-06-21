# Migration Guide

## Migration from Pre-1.0 to 1.0

*Release Date: 2024-12-19*

### Overview

Version 1.0 represents a major milestone for claude-sdk-rs. The most significant change is the consolidation from a multi-crate workspace to a single crate with feature flags. This simplifies dependency management while maintaining modularity.

### Breaking Changes

#### 1. Crate Consolidation

**Before (Pre-1.0):**
```toml
[dependencies]
claude-sdk-rs = "0.x"
claude-sdk-rs-core = "0.x"
claude-sdk-rs-runtime = "0.x"
claude-sdk-rs-mcp = "0.x"
```

**After (1.0):**
```toml
[dependencies]
claude-sdk-rs = "1.0"

# Or with features
claude-sdk-rs = { version = "1.0", features = ["mcp", "sqlite"] }
```

**Migration Steps:**
1. Remove all separate `claude-sdk-rs-*` dependencies from `Cargo.toml`
2. Add the single `claude-sdk-rs` crate with appropriate features
3. Update import paths (see below)

#### 2. Import Path Changes

**Before:**
```rust
use claude_sdk_rs_core::{Config, Error};
use claude_sdk_rs_runtime::Client;
use claude_sdk_rs_mcp::McpServer;
```

**After:**
```rust
use claude_sdk_rs::{Config, Error, Client};
use claude_sdk_rs::mcp::McpServer; // When mcp feature is enabled
```

**Automated Migration:**
```bash
# Use find and replace in your editor or:
find . -name "*.rs" -type f -exec sed -i 's/claude_sdk_rs_core/claude_sdk_rs/g' {} +
find . -name "*.rs" -type f -exec sed -i 's/claude_sdk_rs_runtime/claude_sdk_rs/g' {} +
```

#### 3. Feature-Gated Functionality

Some functionality now requires explicit feature flags:

**MCP Support:**
```toml
# Before: Always available with claude-sdk-rs-mcp
# After: Requires feature flag
claude-sdk-rs = { version = "1.0", features = ["mcp"] }
```

**SQLite Storage:**
```toml
# Before: Part of core
# After: Requires feature flag
claude-sdk_rs = { version = "1.0", features = ["sqlite"] }
```

### New Features in 1.0

1. **Simplified Dependency Management**: Single crate to manage
2. **Reduced Binary Size**: Only include features you need
3. **CLI Binary**: Available with `cli` feature flag
4. **Analytics**: Usage tracking with `analytics` feature
5. **Improved Documentation**: Comprehensive examples and guides

### Common Migration Issues

#### Issue 1: Unresolved Import Errors

**Symptom:**
```
error[E0432]: unresolved import `claude_sdk_rs_core`
```

**Solution:**
Replace all `claude_sdk_rs_*` imports with `claude_sdk_rs`:
```rust
// Old
use claude_sdk_rs_core::Config;

// New
use claude_sdk_rs::Config;
```

#### Issue 2: Missing Types (MCP)

**Symptom:**
```
error[E0412]: cannot find type `McpServer` in this scope
```

**Solution:**
Enable the `mcp` feature in `Cargo.toml`:
```toml
claude-sdk-rs = { version = "1.0", features = ["mcp"] }
```

#### Issue 3: SQLite Functions Not Found

**Symptom:**
```
error: no function `with_sqlite_storage` found
```

**Solution:**
Enable the `sqlite` feature:
```toml
claude-sdk-rs = { version = "1.0", features = ["sqlite"] }
```

### Migration Checklist

- [ ] Update `Cargo.toml` to use single crate
- [ ] Add necessary feature flags
- [ ] Update all import statements
- [ ] Run `cargo clean` before building
- [ ] Run `cargo build` to check for errors
- [ ] Run test suite with `cargo test --all-features`
- [ ] Update CI/CD configurations
- [ ] Review and update documentation

### Performance Considerations

- Feature flags reduce binary size and compilation time
- Core functionality has zero additional dependencies
- No performance regression from consolidation

### Getting Help

- [GitHub Issues](https://github.com/bredmond1019/claude-sdk-rust/issues)
- [Documentation](https://docs.rs/claude-sdk-rs)
- [Examples](https://github.com/bredmond1019/claude-sdk-rust/tree/main/examples)

### Full Changelog

See [CHANGELOG.md](../CHANGELOG.md) for complete details of all changes in version 1.0.