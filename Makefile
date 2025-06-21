# Makefile for claude-sdk-rs development

# Configuration
CARGO = cargo
RUST_LOG ?= info
RUST_BACKTRACE ?= 1

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Default target
.PHONY: help
help:
	@echo "$(GREEN)claude-sdk-rs Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Building:$(NC)"
	@echo "  make build          - Build the crate"
	@echo "  make build-release  - Build in release mode with optimizations"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-doc       - Run documentation tests"
	@echo "  make test-all       - Run all tests with all features"
	@echo "  make test-coverage  - Generate test coverage report"
	@echo "  make test-count     - Count tests across all crates"
	@echo ""
	@echo "$(YELLOW)Code Quality:$(NC)"
	@echo "  make fmt            - Format code with rustfmt"
	@echo "  make fmt-check      - Check code formatting"
	@echo "  make lint           - Run clippy lints"
	@echo "  make check          - Run cargo check"
	@echo "  make audit          - Run security audit"
	@echo ""
	@echo "$(YELLOW)Performance:$(NC)"
	@echo "  make bench          - Run all benchmarks"
	@echo "  make bench-stream   - Run streaming benchmarks"
	@echo "  make bench-client   - Run client benchmarks"
	@echo "  make bench-compare  - Compare benchmarks with baseline"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  make docs           - Build documentation"
	@echo "  make docs-open      - Build and open documentation"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make examples       - Build all examples"
	@echo "  make run-basic      - Run basic example"
	@echo "  make run-streaming  - Run streaming example"
	@echo "  make run-tools      - Run tools example"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make dev            - Run development workflow (fmt, lint, test)"
	@echo "  make pre-commit     - Run pre-commit checks"
	@echo "  make watch          - Watch for changes and rebuild"
	@echo ""
	@echo "$(YELLOW)Release:$(NC)"
	@echo "  make publish-dry    - Dry run of publishing to crates.io"
	@echo "  make publish        - Publish all crates to crates.io"

# Building targets
.PHONY: build
build:
	@echo "$(GREEN)Building the crate...$(NC)"
	$(CARGO) build

.PHONY: build-release
build-release:
	@echo "$(GREEN)Building in release mode...$(NC)"
	$(CARGO) build --release

.PHONY: clean
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	$(CARGO) clean

# Testing targets
.PHONY: test
test:
	@echo "$(GREEN)Running all tests...$(NC)"
	RUST_LOG=$(RUST_LOG) RUST_BACKTRACE=$(RUST_BACKTRACE) $(CARGO) test

.PHONY: test-unit
test-unit:
	@echo "$(GREEN)Running unit tests...$(NC)"
	RUST_LOG=$(RUST_LOG) $(CARGO) test --lib

.PHONY: test-integration
test-integration:
	@echo "$(GREEN)Running integration tests...$(NC)"
	RUST_LOG=$(RUST_LOG) $(CARGO) test --test '*'

.PHONY: test-doc
test-doc:
	@echo "$(GREEN)Running documentation tests...$(NC)"
	$(CARGO) test --doc

.PHONY: test-all
test-all:
	@echo "$(GREEN)Running all tests with all features...$(NC)"
	RUST_LOG=$(RUST_LOG) RUST_BACKTRACE=$(RUST_BACKTRACE) $(CARGO) test --all-features

.PHONY: test-coverage
test-coverage:
	@echo "$(GREEN)Generating test coverage...$(NC)"
	@command -v cargo-tarpaulin >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing cargo-tarpaulin...$(NC)"; \
		$(CARGO) install cargo-tarpaulin; \
	}
	$(CARGO) tarpaulin --config tarpaulin.toml
	@echo "$(GREEN)Coverage report generated at target/coverage/index.html$(NC)"
	@echo "$(GREEN)To view: open target/coverage/index.html$(NC)"

.PHONY: test-count
test-count:
	@echo "$(GREEN)Counting tests in the crate...$(NC)"
	@echo ""
	@echo ""
	@total=$$(find . -path ./target -prune -o -name "*.rs" -type f -exec grep -c "^\s*#\[\(test\|tokio::test\)\]" {} \; | awk '{sum+=$$1} END {print sum}'); \
	echo "$(YELLOW)Total test functions: $$total$(NC)"
	@echo ""
	@echo "$(GREEN)Test breakdown by file (top 10):$(NC)"
	@find . -path ./target -prune -o -name "*.rs" -type f -exec sh -c 'count=$$(grep -c "^\s*#\[\(test\|tokio::test\)\]" "$$1"); if [ $$count -gt 0 ]; then echo "$$count $$1"; fi' _ {} \; | sort -nr | head -10

# Code quality targets
.PHONY: fmt
fmt:
	@echo "$(GREEN)Formatting code...$(NC)"
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check:
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(CARGO) fmt --all -- --check

.PHONY: lint
lint:
	@echo "$(GREEN)Running clippy...$(NC)"
	$(CARGO) clippy --all-targets --all-features -- -D warnings

.PHONY: check
check:
	@echo "$(GREEN)Running cargo check...$(NC)"
	$(CARGO) check --all-features

.PHONY: audit
audit:
	@echo "$(GREEN)Running security audit...$(NC)"
	@command -v cargo-audit >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing cargo-audit...$(NC)"; \
		$(CARGO) install cargo-audit; \
	}
	$(CARGO) audit

# Documentation targets
.PHONY: docs
docs:
	@echo "$(GREEN)Building documentation...$(NC)"
	$(CARGO) doc --no-deps --all-features

.PHONY: docs-open
docs-open: docs
	@echo "$(GREEN)Opening documentation...$(NC)"
	$(CARGO) doc --no-deps --all-features --open

# Example targets
.PHONY: examples
examples:
	@echo "$(GREEN)Building all examples...$(NC)"
	$(CARGO) build --examples

.PHONY: run-basic
run-basic:
	@echo "$(GREEN)Running basic example...$(NC)"
	$(CARGO) run --example basic_usage

.PHONY: run-streaming
run-streaming:
	@echo "$(GREEN)Running streaming example...$(NC)"
	$(CARGO) run --example streaming

.PHONY: run-tools
run-tools:
	@echo "$(GREEN)Running tools example...$(NC)"
	$(CARGO) run --example 04_tools

# Development workflow targets
.PHONY: dev
dev: fmt lint test
	@echo "$(GREEN)Development checks complete!$(NC)"

.PHONY: pre-commit
pre-commit: fmt-check lint test
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

.PHONY: watch
watch:
	@echo "$(GREEN)Watching for changes...$(NC)"
	@command -v cargo-watch >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing cargo-watch...$(NC)"; \
		$(CARGO) install cargo-watch; \
	}
	cargo-watch -x build

# Release targets
.PHONY: publish-dry
publish-dry:
	@echo "$(YELLOW)Dry run of publishing to crates.io...$(NC)"
	@echo "$(YELLOW)Checking claude-sdk-rs...$(NC)"
	$(CARGO) publish --dry-run
	@echo "$(GREEN)Crate ready for publishing!$(NC)"

.PHONY: publish
publish:
	@echo "$(RED)Publishing to crates.io...$(NC)"
	@echo "$(RED)This will publish the crate to crates.io.$(NC)"
	@echo "$(RED)Press Ctrl+C to cancel, or Enter to continue.$(NC)"
	@read -r
	$(CARGO) publish

# Performance benchmarking targets
.PHONY: bench
bench:
	@echo "$(GREEN)Running all benchmarks...$(NC)"
	$(CARGO) bench

.PHONY: bench-stream
bench-stream:
	@echo "$(GREEN)Running streaming benchmarks...$(NC)"
	$(CARGO) bench --bench streaming_bench

.PHONY: bench-client
bench-client:
	@echo "$(GREEN)Running client benchmarks...$(NC)"
	$(CARGO) bench --bench client_bench

.PHONY: bench-compare
bench-compare:
	@echo "$(GREEN)Comparing benchmarks with baseline...$(NC)"
	@echo "$(YELLOW)This will compare against the last saved baseline$(NC)"
	$(CARGO) bench -- --baseline baseline

.PHONY: bench-save
bench-save:
	@echo "$(GREEN)Saving benchmark baseline...$(NC)"
	$(CARGO) bench -- --save-baseline baseline

# Install development tools
.PHONY: install-tools
install-tools:
	@echo "$(GREEN)Installing development tools...$(NC)"
	$(CARGO) install cargo-watch
	$(CARGO) install cargo-tarpaulin
	$(CARGO) install cargo-audit
	$(CARGO) install cargo-outdated
	$(CARGO) install cargo-edit
	@echo "$(GREEN)Development tools installed!$(NC)"