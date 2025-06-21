# Contributing to claude-sdk-rs

Thank you for your interest in contributing to claude-sdk-rs! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct: be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/claude-sdk-rs.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test`
6. Commit your changes
7. Push to your fork
8. Open a pull request

## Development Setup

### Prerequisites

- Rust 1.70 or later
- Claude Code CLI installed
- Valid Anthropic API key (for integration tests)

### Building

```bash
cargo build
```

### Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with verbose output
cargo test -- --nocapture

# Run tests for specific crate
cargo test -p claude-sdk-rs-core

# Check test coverage
make test-coverage
```

#### Writing Tests

When contributing new features or fixes, please include appropriate tests:

**Unit Test Example:**
```rust
#[test]
fn test_config_builder_with_timeout() {
    let config = Config::builder()
        .timeout_secs(120)
        .build();
    
    assert_eq!(config.timeout_secs, Some(120));
}
```

**Async Test Example:**
```rust
#[tokio::test]
async fn test_client_query() {
    let client = Client::new(Config::default());
    
    // Skip if Claude CLI not available
    if which::which("claude").is_err() {
        return;
    }
    
    let result = client.query("test").send().await;
    assert!(result.is_ok() || matches!(result, Err(Error::ClaudeNotAuthenticated)));
}
```

**Property Test Example:**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_session_id_roundtrip(s in "[a-zA-Z0-9-]{1,50}") {
        let id = SessionId::new(&s);
        assert_eq!(id.as_str(), s);
    }
}
```

See [docs/TESTING.md](docs/TESTING.md) for comprehensive testing guidelines.

### Code Style & Formatting

#### Formatting Requirements

**All code must be properly formatted before committing.** We use `rustfmt` with default settings to ensure consistent code style across the project.

**Before committing, always run:**
```bash
# Format all code
cargo fmt --all

# Verify formatting is correct
cargo fmt --check
```

**Pre-commit Hook:**
The repository includes a pre-commit hook that automatically checks formatting. If your code isn't properly formatted, the commit will be rejected with helpful instructions.

#### Style Guidelines

- Follow Rust standard style guidelines (enforced by `rustfmt`)
- Use `cargo fmt` before committing (required)
- Ensure `cargo clippy` passes without warnings
- Add documentation for public APIs
- Include tests for new functionality
- Use meaningful variable and function names
- Keep functions reasonably sized and focused

#### Code Quality Tools

```bash
# Format code (required)
cargo fmt --all

# Check for common issues and style violations
cargo clippy --all-features

# Check for security vulnerabilities
cargo audit

# Run all checks together
cargo fmt --all && cargo clippy --all-features && cargo test
```

## Project Structure

- `claude-sdk-rs-core/`: Core types and traits
- `claude-sdk-rs-runtime/`: Process management and runtime
- `claude-sdk-rs-mcp/`: MCP protocol implementation
- `claude-sdk-rs-macros/`: Procedural macros
- `claude-sdk-rs/`: Main SDK crate
- `examples/`: Usage examples

## Pull Request Process

1. **Format your code**: Run `cargo fmt --all` and ensure `cargo fmt --check` passes
2. **Ensure all tests pass**: Run `cargo test --workspace`
3. **Check for linting issues**: Run `cargo clippy --all-features`
4. **Verify security**: Run `cargo audit` to check for vulnerabilities
5. Update documentation as needed
6. Add an entry to CHANGELOG.md (if applicable)
7. Ensure your commits are well-described
8. Link any related issues

**Note:** Pull requests with formatting violations will be automatically rejected. Please run the pre-commit checks locally before submitting.

## Areas for Contribution

- Implementing missing features from the spec
- Adding more examples
- Improving documentation
- Writing tests
- Performance optimizations
- Bug fixes

## Publishing Process

### Publishing to crates.io

The project uses a automated publish script that handles the complex dependency chain between crates. The crates must be published in a specific order due to their interdependencies:

1. `claude-sdk-rs-core` (no internal dependencies)
2. `claude-sdk-rs-mcp` (depends on core)
3. `claude-sdk-rs-runtime` (depends on core and mcp)
4. `claude-sdk-rs` (depends on core, runtime, and optionally mcp)
5. `claude-sdk-rs-interactive` (depends on claude-sdk-rs and core)

#### Using the Publish Script

```bash
# Dry run (recommended first step)
DRY_RUN=true ./scripts/publish.sh

# Actual publish
./scripts/publish.sh

# With custom options
SLEEP_TIME=60 ./scripts/publish.sh  # Wait 60s between publishes
FORCE_CONTINUE=true ./scripts/publish.sh  # Continue on errors
VERIFY_DEPENDENCIES=false DRY_RUN=true ./scripts/publish.sh  # Skip dependency checks in dry run
```

#### Environment Variables

- `DRY_RUN`: Set to `true` to perform a dry run without actually publishing (default: false)
- `SLEEP_TIME`: Seconds to wait between publishing crates for crates.io processing (default: 30)
- `FORCE_CONTINUE`: Set to `true` to continue publishing even if a crate fails (default: false)
- `VERIFY_DEPENDENCIES`: Set to `false` to skip dependency verification in dry runs (default: true)
- `CARGO_REGISTRY_TOKEN`: Your crates.io API token (required for actual publishing)

#### Manual Publishing

If you need to publish crates manually, ensure you follow the dependency order:

```bash
# 1. First publish core (no dependencies)
cd claude-sdk-rs-core && cargo publish && cd ..

# 2. Wait for crates.io to process (important!)
sleep 30

# 3. Then publish crates that depend only on core
cd claude-sdk-rs-mcp && cargo publish && cd ..
sleep 30

# 4. Then runtime (depends on core and mcp)
cd claude-sdk-rs-runtime && cargo publish && cd ..
sleep 30

# 5. Then main SDK
cd claude-sdk-rs && cargo publish && cd ..
sleep 30

# 6. Finally the CLI
cd claude-sdk-rs-interactive && cargo publish && cd ..
```

#### Pre-publish Checklist

Before publishing a new version:

1. **Update version numbers** in all Cargo.toml files
2. **Update CHANGELOG.md** with the new version and changes
3. **Run all tests**: `cargo test --workspace`
4. **Check formatting**: `cargo fmt --check`
5. **Run clippy**: `cargo clippy --all-features`
6. **Security audit**: `cargo audit`
7. **Dry run publish**: `DRY_RUN=true ./scripts/publish.sh`
8. **Tag the release**: `git tag v1.0.0 && git push --tags`

#### Troubleshooting

**"no matching package named" errors during dry run:**
This is expected if the dependencies aren't published yet. Use `VERIFY_DEPENDENCIES=false` for dry runs.

**"crate version is already uploaded" error:**
The version number already exists on crates.io. Bump the version in Cargo.toml.

**Network/API errors:**
The script will retry automatically. If it continues to fail, check your internet connection and crates.io status.

**"Failed to change to directory" error:**
Ensure you're running the script from the workspace root directory.

## Types of Contributions

We welcome many different types of contributions:

### 🐛 Bug Reports

When filing a bug report, please include:
- A clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Rust version, claude-sdk-rs version)
- Any error messages or logs

Use the **Bug Report** issue template for consistency.

### ✨ Feature Requests

Before requesting a new feature:
- Check if a similar feature already exists or is planned
- Search existing issues and discussions
- Consider if the feature fits the project's goals
- Think about backward compatibility implications

Use the **Feature Request** issue template and include:
- Clear description of the desired functionality
- Use cases and motivation
- Proposed API design (if applicable)
- Willingness to implement (optional)

### 📚 Documentation Improvements

Documentation contributions are highly valued:
- API documentation and examples
- Tutorials and guides
- README improvements
- Code comments
- Migration guides

### 🧪 Testing

Help improve test coverage:
- Add tests for existing functionality
- Create integration tests
- Add property-based tests
- Improve test performance
- Add regression tests for bug fixes

### 🎯 Performance Improvements

Performance contributions should include:
- Benchmarks showing the improvement
- Analysis of trade-offs
- Tests ensuring correctness is maintained

## Community Guidelines

### Communication

- Be respectful and inclusive in all interactions
- Use clear, constructive language
- Focus on the issue, not the person
- Assume good intentions
- Ask questions when unsure

### Issue Etiquette

- Search existing issues before creating new ones
- Use descriptive titles
- Provide complete information
- Stay on topic in discussions
- Update issues if your situation changes

### Pull Request Guidelines

#### Before Creating a PR

1. **Discuss major changes**: For significant features or changes, open an issue first
2. **Follow the development setup**: Ensure your environment is properly configured
3. **Run pre-commit checks**: Use our git hooks or run checks manually
4. **Review the code**: Ensure your code meets our standards

#### PR Description

Include in your PR description:
- Summary of changes
- Related issue numbers
- Testing performed
- Breaking changes (if any)
- Screenshots (for UI changes)

#### Reviewing Process

- PRs require at least one approving review from a maintainer
- Automated checks must pass
- Breaking changes require special review
- Large PRs may be broken into smaller ones

### Commit Message Guidelines

We follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(core): add timeout configuration option

Add support for custom timeout values in Config builder.
This allows users to set different timeout values for
different use cases.

Closes #123
```

```
fix(runtime): handle process termination gracefully

Previously, the runtime would panic if the Claude CLI process
terminated unexpectedly. Now it returns a proper error.

Fixes #456
```

## Versioning and Releases

### Semantic Versioning

We follow [semantic versioning](https://semver.org/):
- **Patch** (1.0.X): Bug fixes, internal improvements
- **Minor** (1.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes

### API Compatibility

- **Backward compatibility** is maintained within major versions
- **Breaking changes** require major version bumps
- **Deprecation warnings** precede removal by at least 2 minor versions

See our [API Evolution Guidelines](API_EVOLUTION_GUIDELINES.md) for detailed policies.

### Release Process

1. **Version Planning**: Major versions are planned with community input
2. **Beta Testing**: Pre-releases allow community testing
3. **Migration Support**: Breaking changes include migration guides
4. **LTS Support**: Long-term support for major versions

## Security

### Reporting Vulnerabilities

**Do not report security vulnerabilities through public GitHub issues.**

Instead:
1. Email security concerns to [maintainer email]
2. Include detailed reproduction steps
3. Allow time for investigation and patching
4. Coordinate public disclosure

See [SECURITY.md](SECURITY.md) for our security policy.

### Security Considerations

When contributing:
- Be mindful of input validation
- Avoid hardcoded secrets
- Consider security implications of changes
- Follow secure coding practices

## Recognition

### Contributor Recognition

We value all contributions and recognize contributors through:
- **Git history**: Your commits are permanently part of the project
- **Release notes**: Significant contributions are mentioned
- **Contributor list**: Active contributors are recognized
- **Maintainer status**: Long-term contributors may become maintainers

### Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Development Workflow

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/claude-sdk-rs.git
   cd claude-sdk-rs
   ```

2. **Install dependencies**:
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install Claude Code CLI
   # Follow instructions at https://claude.ai/code
   ```

3. **Set up git hooks**:
   ```bash
   ./scripts/setup-git-hooks.sh
   ```

4. **Verify setup**:
   ```bash
   cargo build --workspace
   cargo test --workspace
   ```

### Branch Strategy

- **main**: Stable, release-ready code
- **feature/**: New features and improvements
- **fix/**: Bug fixes
- **docs/**: Documentation changes

### Development Cycle

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow coding standards and guidelines
3. **Run tests**: Ensure all tests pass
4. **Commit changes**: Use conventional commit format
5. **Push branch**: `git push origin feature/your-feature`
6. **Create PR**: Use the pull request template
7. **Address feedback**: Make requested changes
8. **Merge**: Maintainer merges after approval

## Troubleshooting

### Common Issues

**Build failures:**
- Ensure Rust 1.70+ is installed
- Run `cargo clean` and try again
- Check for missing system dependencies

**Test failures:**
- Ensure Claude CLI is installed and authenticated
- Check environment variables
- Run individual test suites to isolate issues

**Permission errors:**
- Ensure git hooks are executable
- Check file permissions
- Verify workspace ownership

**Formatting issues:**
- Run `cargo fmt --all`
- Check for unsaved editor changes
- Ensure rustfmt is installed

### Getting Help

1. **Check documentation**: README, guides, and API docs
2. **Search issues**: Look for similar problems
3. **Ask questions**: Open a GitHub issue with the "question" label
4. **Join discussions**: Participate in GitHub Discussions
5. **Contact maintainers**: For complex issues or private concerns

## Useful Resources

### Documentation
- [API Documentation](https://docs.rs/claude-sdk-rs)
- [Testing Guide](docs/TESTING.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Performance Guide](docs/PERFORMANCE.md)

### Development Tools
- [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks): Breaking change detection
- [cargo-audit](https://github.com/RustSec/rustsec/tree/main/cargo-audit): Security auditing
- [cargo-tarpaulin](https://github.com/xd009642/tarpaulin): Code coverage

### External Resources
- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Claude Code Documentation](https://claude.ai/code)

## Questions?

Feel free to:
- Open an issue for questions or discussions
- Start a GitHub Discussion for broader topics
- Contact maintainers for private concerns
- Join our community channels (see README for links)

Thank you for contributing to claude-sdk-rs! 🚀