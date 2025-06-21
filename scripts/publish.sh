#!/bin/bash
set -e

# publish.sh - Publish claude-sdk-rs crate to crates.io
#
# This script publishes the single crate claude-sdk-rs to crates.io

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=${DRY_RUN:-false}
FORCE_CONTINUE=${FORCE_CONTINUE:-false}

print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

publish_crate() {
    local max_retries=3
    local retry_count=0
    
    print_step "Publishing claude-sdk-rs"
    
    # Check that we're in the right directory
    if [ ! -f "Cargo.toml" ] || ! grep -q "^name = \"claude-sdk-rs\"" Cargo.toml; then
        print_error "Must be run from the claude-sdk-rs crate root directory"
        return 1
    fi
    
    # Check if we're in a git repository and if there are uncommitted changes
    if git rev-parse --git-dir > /dev/null 2>&1; then
        if ! git diff-index --quiet HEAD --; then
            print_warning "Uncommitted changes detected, using --allow-dirty"
            ALLOW_DIRTY="--allow-dirty"
        else
            ALLOW_DIRTY=""
        fi
    else
        print_warning "Not in a git repository, using --allow-dirty"
        ALLOW_DIRTY="--allow-dirty"
    fi
    
    # Build the publish command
    if [ "$DRY_RUN" = "true" ]; then
        cmd="cargo publish --dry-run $ALLOW_DIRTY"
        print_warning "DRY RUN: $cmd"
    else
        cmd="cargo publish $ALLOW_DIRTY"
        print_step "REAL PUBLISH: $cmd"
    fi
    
    # Execute the command with retry logic
    while [ $retry_count -lt $max_retries ]; do
        if eval $cmd 2>&1 | tee /tmp/publish_claude_sdk_rs.log; then
            print_success "Successfully published claude-sdk-rs"
            return 0
        else
            retry_count=$((retry_count + 1))
            local log_content=$(cat /tmp/publish_claude_sdk_rs.log)
            
            # Analyze different error types
            if echo "$log_content" | grep -q "no matching package named"; then
                print_error "Missing dependency: $(echo "$log_content" | grep "no matching package named" | head -1)"
            elif echo "$log_content" | grep -q "crate version .* is already uploaded"; then
                print_warning "Version already published on crates.io"
                return 0
            elif echo "$log_content" | grep -q "api errors"; then
                print_error "API error from crates.io: $(echo "$log_content" | grep -A2 "api errors")"
            elif echo "$log_content" | grep -q "failed to get a 200 OK response"; then
                print_error "Network/connection error to crates.io"
            elif echo "$log_content" | grep -q "failed to read"; then
                print_error "Failed to read crate files: $(echo "$log_content" | grep "failed to read")"
            else
                print_error "Unknown error publishing claude-sdk-rs. See /tmp/publish_claude_sdk_rs.log for details"
            fi
            
            if [ $retry_count -lt $max_retries ]; then
                print_warning "Publish failed, retrying in 10s... (attempt $retry_count/$max_retries)"
                sleep 10
            else
                print_error "Failed to publish claude-sdk-rs after $max_retries attempts"
                echo ""
                echo "Last error output:"
                echo "=================="
                tail -20 /tmp/publish_claude_sdk_rs.log
                echo "=================="
                
                if [ "$FORCE_CONTINUE" = "true" ]; then
                    print_warning "Continuing despite failure (FORCE_CONTINUE=true)"
                    return 1
                else
                    print_error "Aborting publication process. Fix the issue and retry."
                    exit 1
                fi
            fi
        fi
    done
    
    return 1
}

verify_crate() {
    print_step "Verifying crate structure"
    
    # Check that we're in the right directory
    if [ ! -f "Cargo.toml" ]; then
        print_error "Must be run from the crate root directory"
        exit 1
    fi
    
    # Verify crate has package information
    if ! grep -q "^\[package\]" "Cargo.toml"; then
        print_error "No [package] section found in Cargo.toml"
        exit 1
    fi
    
    # Verify crate name
    if ! grep -q "^name = \"claude-sdk-rs\"" "Cargo.toml"; then
        print_error "Crate name is not claude-sdk-rs"
        exit 1
    fi
    
    print_success "Crate structure verified"
}

check_credentials() {
    if [ "$DRY_RUN" != "true" ]; then
        print_step "Checking crates.io credentials"
        
        if [ -z "$CARGO_REGISTRY_TOKEN" ] && ! cargo login --help > /dev/null 2>&1; then
            print_error "No crates.io credentials found. Run 'cargo login' or set CARGO_REGISTRY_TOKEN"
            exit 1
        fi
        
        print_success "Credentials check passed"
    fi
}

run_tests() {
    print_step "Running tests before publish"
    
    if cargo test --all-features; then
        print_success "All tests passed"
    else
        print_error "Tests failed. Please fix before publishing."
        exit 1
    fi
}

get_crate_version() {
    local version=$(grep "^version" "Cargo.toml" | head -n1 | cut -d'"' -f2)
    echo "$version"
}

main() {
    print_step "Starting claude-sdk-rs crate publication"
    
    # Show configuration
    echo "Configuration:"
    echo "  DRY_RUN: $DRY_RUN"
    echo "  FORCE_CONTINUE: $FORCE_CONTINUE"
    echo ""
    
    # Verify environment
    verify_crate
    check_credentials
    
    # Run tests (skip in dry run for speed)
    if [ "$DRY_RUN" != "true" ]; then
        run_tests
    fi
    
    # Publish the crate
    if publish_crate; then
        local version=$(get_crate_version)
        print_success "claude-sdk-rs v$version published successfully!"
        
        if [ "$DRY_RUN" != "true" ]; then
            echo ""
            print_step "Publication Complete"
            echo "claude-sdk-rs v$version has been published to crates.io"
            echo ""
            echo "You can now install it with:"
            echo "  cargo add claude-sdk-rs"
            echo ""
            echo "Or install the CLI with:"
            echo "  cargo install claude-sdk-rs --features cli"
        else
            echo ""
            print_success "Dry run completed successfully!"
            echo "To publish for real, run:"
            echo "  DRY_RUN=false ./scripts/publish.sh"
        fi
    else
        print_error "Publication failed"
        exit 1
    fi
}

# Help text
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Publish claude-sdk-rs crate to crates.io."
    echo ""
    echo "Environment variables:"
    echo "  DRY_RUN=true       - Perform dry run only (default: false)"
    echo "  CARGO_REGISTRY_TOKEN=<token> - crates.io API token"
    echo ""
    echo "Examples:"
    echo "  $0                 - Publish the crate"
    echo "  DRY_RUN=true $0    - Dry run only"
    echo ""
    exit 0
fi

# Run main function
main "$@"