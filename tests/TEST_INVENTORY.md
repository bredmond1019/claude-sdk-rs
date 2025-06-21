# Test Inventory

## Summary
- **Total Test Functions**: 837
- **Test Count Method**: `find . -path ./target -prune -o -name "*.rs" -type f -exec grep -c "^\s*#\[\(test\|tokio::test\)\]" {} \;`

## Test Distribution by Crate

| Crate | Test Count |
|-------|------------|
| claude-sdk-rs-core | 80 |
| claude-sdk-rs-runtime | 57 |
| claude-sdk-rs | 29 |
| claude-sdk-rs-mcp | 73 |
| claude-sdk-rs-interactive | 583 |
| **Total** | **837** |

## Top Test Files

1. `claude-sdk-rs-interactive/src/history/store_test.rs` - 75 tests
2. `claude-sdk-rs-interactive/src/cli/commands_test.rs` - 57 tests
3. `claude-sdk-rs-core/src/config_test.rs` - 48 tests
4. `claude-sdk-rs-interactive/src/profiling/profiling_test.rs` - 34 tests
5. `claude-sdk-rs-interactive/src/cost/tracker_test.rs` - 34 tests
6. `claude-sdk-rs-runtime/tests/integration_tests.rs` - 31 tests
7. `claude-sdk-rs-core/src/session_test.rs` - 30 tests
8. `claude-sdk-rs-interactive/src/execution/execution_test.rs` - 29 tests
9. `claude-sdk-rs-interactive/src/analytics/dashboard_tests.rs` - 29 tests
10. `claude-sdk-rs-interactive/src/cost/trend_tests.rs` - 28 tests

## New Tests Added by Agent 2

### claude-sdk-rs-core (7 new tests)
- `test_config_with_invalid_timeout_zero` - Tests zero timeout configuration
- `test_config_with_negative_max_tokens_workaround` - Tests edge case for max tokens
- `test_config_with_invalid_mcp_path` - Tests invalid MCP configuration path
- `test_config_with_conflicting_stream_formats` - Tests multiple stream format settings
- `test_config_with_invalid_tool_names` - Tests invalid tool name handling
- `test_config_max_values` - Tests maximum value boundaries
- `test_config_model_name_edge_cases` - Tests edge cases in model names

### claude-sdk-rs-runtime (13 new tests)
#### Streaming Timeout Tests (3)
- `test_streaming_timeout_immediate` - Tests immediate timeout scenarios
- `test_streaming_timeout_during_messages` - Tests timeout during message streaming
- `test_streaming_timeout_recovery` - Tests recovery after timeout

#### Concurrent Request Tests (3)
- `test_concurrent_client_creation` - Tests creating multiple clients concurrently
- `test_concurrent_message_processing` - Tests processing messages from multiple streams
- `test_concurrent_error_handling` - Tests concurrent streams with mixed success/error

#### Malformed Output Tests (3)
- `test_malformed_json_single_quote` - Tests handling of JSON with single quotes
- `test_malformed_json_missing_closing_brace` - Tests incomplete JSON handling
- `test_malformed_stream_json_mixed_format` - Tests mixed valid/invalid JSON lines

#### Error Recovery Tests (4)
- `test_error_recovery_after_timeout` - Tests recovery after timeout errors
- `test_error_recovery_after_process_error` - Tests recovery after process errors
- `test_error_recovery_with_retry_pattern` - Tests retry pattern implementation
- `test_error_recovery_circuit_breaker_pattern` - Tests circuit breaker pattern

## Testing Commands

```bash
# Count all tests
make test-count

# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p claude-sdk-rs-core
cargo test -p claude-sdk-rs-runtime

# Run specific test
cargo test test_config_with_invalid_timeout_zero

# Run tests with output
cargo test -- --nocapture
```

## Test Coverage Areas

1. **Configuration Validation** - Comprehensive edge case testing
2. **Streaming Operations** - Timeout and concurrent stream handling
3. **Error Handling** - Recovery patterns and malformed data
4. **Concurrency** - Multiple client and request handling
5. **Data Parsing** - JSON parsing and validation

## Notes

- All tests are real, functional tests with meaningful assertions
- No placeholder or trivial tests
- Tests follow Rust testing best practices
- Uses both `#[test]` and `#[tokio::test]` annotations
- Property-based testing with `proptest` in some modules