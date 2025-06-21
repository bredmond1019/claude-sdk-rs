# Testing Guide for claude-sdk-rs

This document describes the testing strategy, organization, and best practices for the claude-sdk-rs workspace.

## Table of Contents

- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Mock vs Real CLI Testing](#mock-vs-real-cli-testing)
- [Coverage Requirements](#coverage-requirements)
- [CI/CD Integration](#cicd-integration)

## Test Organization

The claude-sdk-rs workspace follows a consistent testing structure across all crates:

### Directory Structure

```
crate-name/
├── src/
│   ├── lib.rs
│   ├── module.rs
│   └── module_test.rs      # Unit tests for module
└── tests/
    ├── integration_test.rs  # Integration tests
    └── specific_test.rs     # Focused integration tests
```

### Test Categories

#### 1. Unit Tests
Located in `src/*_test.rs` files, these test individual functions and types in isolation.

**Example:** `src/core/config_test.rs` (58 tests)
- Configuration builder patterns and validation
- Default value verification
- Edge cases and boundary conditions
- Property-based testing with arbitrary inputs

#### 2. Integration Tests
Located in `tests/` directories, these test interactions between modules and external dependencies.

**Example:** `tests/runtime/integration_tests.rs` (31 tests)
- Client creation and configuration
- Streaming functionality and backpressure
- Error propagation and recovery patterns
- Concurrent request handling

#### 3. Critical Path Tests
Comprehensive tests for core functionality across modules.

**Example:** `tests/core/critical_path_tests.rs` (52 tests)
- End-to-end workflow testing
- Performance-critical operations
- Security and reliability validation

#### 3. Property-Based Tests
Uses the `proptest` crate for testing with arbitrary inputs.

**Example:** Property tests in `config_test.rs`
- Tests configuration with random inputs
- Validates invariants hold for all inputs
- Finds edge cases automatically

#### 4. Documentation Tests
Tests embedded in documentation comments.

```rust
/// # Example
/// ```
/// use claude_ai::Client;
/// let client = Client::new(Default::default());
/// ```
```

### Test Count by Module

**Total Test Functions: 1,172** (as of 2025-06-19)

| Module | Test Count | Primary Test Files |
|--------|------------|-------------------|
| Core Configuration | 58 | `src/core/config_test.rs` |
| CLI History | 75 | `src/cli/history/store_test.rs` |
| CLI Commands | 57 | `src/cli/cli/commands_test.rs` |
| Runtime Integration | 31 | `tests/runtime/integration_tests.rs` |
| Critical Path | 91 | `tests/core/critical_path_tests.rs`, `tests/runtime/critical_path_tests.rs` |
| Analytics & Profiling | 68 | Various analytics test files |
| Error Handling | 24 | Core and runtime error tests |
| Session Management | 33 | `src/core/session_test.rs` |
| **Other Modules** | 735+ | Various test files across MCP, CLI, and interactive features | |

## Running Tests

### Basic Commands

```bash
# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test -p claude-sdk-rs-core

# Run specific test
cargo test test_config_default

# Run tests with output
cargo test -- --nocapture

# Run tests in release mode
cargo test --release
```

### Using Make Commands

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run doc tests
make test-doc

# Count tests
make test-count

# Generate coverage report
make test-coverage
```

### Test Filtering

```bash
# Run tests matching pattern
cargo test config

# Run ignored tests
cargo test -- --ignored

# Run all tests including ignored
cargo test -- --include-ignored
```

## Writing Tests

### Test Naming Conventions

```rust
#[test]
fn test_module_function_scenario() {
    // Test names should be descriptive
}

#[tokio::test]
async fn test_async_operation_success() {
    // Async tests use tokio::test
}
```

### Test Structure

Follow the Arrange-Act-Assert pattern:

```rust
#[test]
fn test_config_builder_with_all_fields() {
    // Arrange
    let expected_model = "claude-3-opus";
    let expected_timeout = 60;
    
    // Act
    let config = Config::builder()
        .model(expected_model)
        .timeout_secs(expected_timeout)
        .build();
    
    // Assert
    assert_eq!(config.model, Some(expected_model.to_string()));
    assert_eq!(config.timeout_secs, Some(expected_timeout));
}
```

### Async Test Patterns

```rust
#[tokio::test]
async fn test_streaming_response() {
    use futures::StreamExt;
    
    // Create test stream
    let (tx, rx) = mpsc::channel(10);
    let stream = MessageStream::new(rx, StreamFormat::Text);
    
    // Send test data
    tx.send(Ok(test_message())).await.unwrap();
    drop(tx);
    
    // Collect and verify
    let result = stream.collect_full_response().await;
    assert!(result.is_ok());
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_config_with_arbitrary_strings(
        model in any::<String>(),
        prompt in any::<String>(),
    ) {
        let config = Config::builder()
            .model(model)
            .system_prompt(prompt)
            .build();
        
        // Config should build successfully with any strings
        assert!(config.timeout_secs.is_some());
    }
}
```

### Error Testing

```rust
#[test]
fn test_error_handling() {
    // Test specific error types
    match some_operation() {
        Err(Error::Timeout(secs)) => {
            assert_eq!(secs, 30);
        }
        Err(e) => panic!("Unexpected error: {}", e),
        Ok(_) => panic!("Expected error"),
    }
}
```

## Mock vs Real CLI Testing

### When to Use Mocks

**Use mocks for:**
- Unit tests that need deterministic behavior
- Testing error conditions
- CI/CD environments without Claude CLI
- Testing specific edge cases

**Example Mock Test:**
```rust
#[tokio::test]
async fn test_mock_claude_timeout() {
    let mock = MockClaude::new()
        .with_delay(Duration::from_secs(5))
        .with_response("Test response");
    
    let config = Config::builder()
        .timeout_secs(1)
        .build();
    
    let result = mock.execute(&config, "test").await;
    assert!(matches!(result, Err(Error::Timeout(_))));
}
```

### When to Use Real CLI

**Use real CLI for:**
- Integration tests
- Performance benchmarks
- Manual testing
- Verifying actual Claude behavior

**Example Real CLI Test:**
```rust
#[tokio::test]
async fn test_real_claude_query() {
    // Skip if Claude not available
    if which::which("claude").is_err() {
        eprintln!("Skipping: Claude CLI not found");
        return;
    }
    
    let client = Client::new(Config::default());
    let result = client.query("Say hello").send().await;
    
    // Handle both success and expected failures
    match result {
        Ok(response) => assert!(!response.is_empty()),
        Err(e) => eprintln!("Expected error: {}", e),
    }
}
```

### Hybrid Approach

Most test files use a hybrid approach:

```rust
#[cfg(test)]
mod tests {
    // Real CLI tests when available
    mod integration_tests {
        // Tests that use actual Claude CLI
    }
    
    // Mock tests for edge cases
    mod mock_tests {
        // Tests using mock implementations
    }
}
```

## Coverage Requirements

### Target Coverage

- **Overall:** 80% minimum
- **Core modules:** 85% recommended  
- **New code:** 90% for PRs
- **Critical paths:** 95% required

### Current Coverage Status

*Note: Full coverage reports may require extended execution time due to the comprehensive test suite (1,172+ tests). For development, use focused testing on specific modules.*

### Checking Coverage

```bash
# Generate local coverage report
make test-coverage

# View HTML report
open target/coverage/index.html

# Check coverage in CI
# Coverage is automatically checked in CI/CD pipeline
```

### Coverage Configuration

See `tarpaulin.toml` for coverage settings:
- **Branch coverage:** Enabled for comprehensive analysis
- **Timeout:** 300s per test (accommodates large test suite)
- **Output formats:** HTML, XML, and LCOV
- **Failure threshold:** 80% minimum coverage
- **Excluded files:** Tests, examples, benchmarks, build scripts

**Note:** Due to the extensive test suite (1,172+ tests), coverage generation may take several minutes. For faster feedback during development, run coverage on specific modules:

```bash
# Run coverage on core module only
cargo tarpaulin --lib --timeout 120

# Run with reduced timeout for development
cargo tarpaulin --timeout 60 --out Json
```

### Improving Coverage

1. **Identify gaps:** Use coverage reports to find untested code
2. **Add edge cases:** Test error paths and boundary conditions
3. **Test all branches:** Ensure all if/else paths are covered
4. **Property tests:** Use proptest for better input coverage

## CI/CD Integration

### Test Matrix

Tests run on multiple platforms and Rust versions:
- **OS:** Ubuntu, Windows, macOS
- **Rust:** Stable, Beta, MSRV (1.70.0)

### CI Test Stages

1. **Unit Tests:** Fast feedback (core module tests ~83 tests in 0.12s)
2. **Integration Tests:** End-to-end validation (31 runtime integration tests)
3. **Critical Path Tests:** Core functionality validation (91 tests)
4. **Doc Tests:** Verify documentation examples
5. **Coverage:** Generate reports (may require extended timeout)
6. **Quality:** Clippy lints and format checks

**Performance Notes:**
- Core tests execute in ~0.12s
- Full test suite may take several minutes
- Some telemetry tests may require extended timeouts
- Use `cargo test --lib` for faster development feedback

### Pre-commit Testing

Run before committing:
```bash
make pre-commit
```

This runs:
- Format check
- Clippy lints
- All tests

### Debugging CI Failures

1. **Check logs:** Look for specific test failures
2. **Reproduce locally:** Use same Rust version
3. **Platform-specific:** Test on Docker if needed
4. **Environment:** Check for missing env vars

## Best Practices

### DO:
- ✅ Write descriptive test names
- ✅ Test both success and failure cases
- ✅ Use property tests for complex inputs
- ✅ Keep tests independent and isolated
- ✅ Mock external dependencies in unit tests
- ✅ Document why a test exists if not obvious

### DON'T:
- ❌ Skip error cases
- ❌ Depend on test execution order
- ❌ Use hard-coded timeouts
- ❌ Ignore flaky tests
- ❌ Test implementation details
- ❌ Leave commented-out tests

### Test Maintenance

- **Review tests** during code reviews
- **Update tests** when changing behavior  
- **Remove obsolete tests** when removing features
- **Fix flaky tests** immediately (some telemetry tests may need timeout adjustments)
- **Refactor tests** to reduce duplication
- **Monitor test performance** - investigate tests running >60s
- **Update test inventory** when adding significant test suites

### Known Issues

- Some telemetry and streaming tests may hang under certain conditions
- Coverage generation can be slow due to large test suite
- Timeouts may need adjustment in CI environments

---

For more information, see:
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [tests/TEST_INVENTORY.md](../tests/TEST_INVENTORY.md) - Complete test listing
- [PERFORMANCE.md](PERFORMANCE.md) - Performance testing and benchmarks
- [SECURITY.md](SECURITY.md) - Security testing guidelines
- [Rust Testing Book](https://doc.rust-lang.org/book/ch11-00-testing.html) - Official Rust testing guide