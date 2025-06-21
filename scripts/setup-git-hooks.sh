#!/bin/bash
# Setup git hooks for claude-ai development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_info "Setting up git hooks for claude-ai development..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not in a git repository"
    exit 1
fi

# Check if we're in the workspace root
if [ ! -f "Cargo.toml" ] || ! grep -q "\[workspace\]" Cargo.toml; then
    print_error "Must be run from the workspace root directory"
    exit 1
fi

# Create git hooks directory if it doesn't exist
if [ ! -d ".git/hooks" ]; then
    mkdir -p .git/hooks
    print_info "Created .git/hooks directory"
fi

# Install pre-commit hook
if [ -f ".githooks/pre-commit" ]; then
    cp .githooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    print_success "Installed pre-commit hook"
else
    print_error "Pre-commit hook source not found at .githooks/pre-commit"
    exit 1
fi

# Check if cargo-semver-checks is installed
print_info "Checking for cargo-semver-checks..."
if command -v cargo-semver-checks >/dev/null 2>&1; then
    print_success "cargo-semver-checks is installed"
else
    print_warning "cargo-semver-checks not found"
    echo ""
    echo "To install cargo-semver-checks for breaking change detection:"
    echo "  cargo install cargo-semver-checks"
    echo ""
    echo "This tool is used by CI/CD for automated breaking change detection."
fi

# Make sure version tools are executable
if [ -f "scripts/version-tools.sh" ]; then
    chmod +x scripts/version-tools.sh
    print_success "Made version tools executable"
fi

# Test the pre-commit hook
print_info "Testing pre-commit hook..."
if .git/hooks/pre-commit; then
    print_success "Pre-commit hook test passed"
else
    print_warning "Pre-commit hook test failed - you may need to fix issues before committing"
fi

echo ""
print_success "Git hooks setup completed!"
echo ""
echo "The following hooks are now active:"
echo "  • pre-commit: Checks for breaking changes, formatting, and basic compatibility"
echo ""
echo "Before each commit, the hook will:"
echo "  1. Check version consistency across workspace"
echo "  2. Verify code formatting (cargo fmt)"
echo "  3. Run clippy for basic linting"
echo "  4. Test workspace build"
echo "  5. Run API compatibility tests"
echo "  6. Scan for potential breaking changes"
echo ""
echo "To temporarily bypass hooks (not recommended):"
echo "  git commit --no-verify"
echo ""
echo "For more information, see:"
echo "  • API_EVOLUTION_GUIDELINES.md"
echo "  • DEPRECATION_POLICY.md"
echo "  • BACKWARD_COMPATIBILITY.md"