# Release Notes

## v1.0.0 (2025-06-17)

### 🎉 Initial Stable Release

We're excited to announce the first stable release of `claude-sdk-rs`, a type-safe, async-first Rust SDK for Claude Code. This release marks the culmination of extensive development, testing, and refinement to provide a production-ready SDK for building AI-powered applications with Claude.

### ✨ Key Features

#### Type-Safe API
- Strongly typed requests and responses with compile-time guarantees
- Comprehensive error types for robust error handling
- Builder patterns for intuitive configuration

#### Async-First Architecture
- Built on Tokio for efficient concurrent operations
- Support for streaming responses with async iterators
- Non-blocking I/O throughout the stack

#### Rich Response Handling
- Three response modes: Simple text, full metadata, and streaming
- Access to cost information, token usage, and timing data
- Raw JSON access for advanced use cases

#### Session Management
- Automatic session persistence across conversations
- Context preservation for multi-turn interactions
- Session metadata tracking and analytics

#### Tool Integration
- Model Context Protocol (MCP) support
- Configurable tool permissions
- Extensible architecture for custom tools

### 📦 Crate Structure

The SDK is organized into 5 specialized crates:

- **`claude-sdk-rs`** (1.0.0) - Main SDK facade and public API
- **`claude-sdk-rs-core`** (1.0.0) - Core types, configuration, and errors
- **`claude-sdk-rs-runtime`** (1.0.0) - Process execution and streaming
- **`claude-sdk-rs-mcp`** (1.0.0) - Model Context Protocol implementation
- **`claude-sdk-rs-interactive`** (1.0.0) - Interactive terminal interface

### 🚀 Performance

Based on comprehensive benchmarking:

- **Message parsing**: ~350ns for JSON, ~41ns for text (8.5x faster)
- **Streaming throughput**: 10-30µs per message depending on size
- **Optimal buffer size**: 100-200 messages for best latency/memory trade-off
- **Memory footprint**: <1KB per client instance

### 🧪 Testing & Quality

- **84 comprehensive tests** across all crates
- **100% test pass rate**
- **Extensive documentation** with examples
- **Integration tests** for real-world scenarios
- **Property-based testing** for core types

### 📚 Documentation

This release includes comprehensive documentation:

- [Quick Start Guide](QUICK_START.md) - Get running in minutes
- [Tutorial](TUTORIAL.md) - In-depth guide with examples
- [Performance Guide](docs/PERFORMANCE.md) - Optimization tips
- [FAQ](FAQ.md) - Common questions answered
- [API Documentation](https://docs.rs/claude-sdk-rs) - Full API reference

### 🔧 Requirements

- **Rust 1.70+** - For modern async/await support
- **Claude Code CLI** - Must be installed and authenticated
- **Tokio runtime** - Required for async operations

### 💻 Platform Support

- Linux (x86_64, ARM64)
- macOS (x86_64, Apple Silicon)
- Windows (x86_64, ARM64)

### 🙏 Acknowledgments

Thank you to all contributors who helped make this release possible. Special thanks to the Anthropic team for Claude Code and the Rust community for their excellent tooling and libraries.

### 🚦 Migration from Pre-1.0

If you were using a pre-1.0 version, please note:

1. The API is now stable and follows semantic versioning
2. Some builder method names have been standardized
3. Error types have been consolidated in `claude-sdk-rs-core`
4. Session management is now automatic by default

### 📈 What's Next

Future releases will focus on:

- Performance optimizations (SIMD JSON parsing, zero-copy deserialization)
- Enhanced MCP tool support
- Connection pooling for improved efficiency
- Batch processing capabilities
- Additional language model support

### 🐛 Known Issues

- MCP tools are in early development stage
- Some advanced streaming configurations may require manual tuning
- Windows support for certain MCP tools is limited

### 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
claude-sdk-rs = "1.0.0"
tokio = { version = "1.0", features = ["full"] }
```

### 🔗 Links

- [GitHub Repository](https://github.com/frgmt0/claude-sdk-rs)
- [Crates.io](https://crates.io/crates/claude-sdk-rs)
- [Documentation](https://docs.rs/claude-sdk-rs)
- [Issue Tracker](https://github.com/frgmt0/claude-sdk-rs/issues)

---

For questions or support, please open an issue on GitHub or reach out through the community channels.