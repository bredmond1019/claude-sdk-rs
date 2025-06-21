#!/bin/bash
# scripts/version-tools.sh - Version management utilities for claude-ai workspace

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Get current workspace version
get_current_version() {
    if [ ! -f "Cargo.toml" ]; then
        print_error "Cargo.toml not found. Run from workspace root."
        exit 1
    fi
    
    grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2
}

# Validate version consistency across workspace
validate_versions() {
    local workspace_version=$(get_current_version)
    local errors=0
    
    print_info "Validating version consistency..."
    print_info "Workspace version: $workspace_version"
    echo ""
    
    # Check all crate Cargo.toml files
    find . -name "Cargo.toml" -not -path "./target/*" -not -path "./Cargo.toml" | sort | while read -r cargo_file; do
        local crate_name=$(basename $(dirname "$cargo_file"))
        
        print_info "Checking $crate_name..."
        
        # Check if crate uses workspace version
        if grep -q "version.workspace = true" "$cargo_file"; then
            print_success "$crate_name uses workspace version"
        else
            # Check explicit version
            local crate_version=$(grep "^version = " "$cargo_file" | head -n1 | cut -d'"' -f2 2>/dev/null || echo "")
            if [ -n "$crate_version" ]; then
                if [ "$crate_version" = "$workspace_version" ]; then
                    print_success "$crate_name has matching explicit version: $crate_version"
                else
                    print_error "$crate_name version mismatch: expected $workspace_version, found $crate_version"
                    errors=$((errors + 1))
                fi
            else
                print_warning "$crate_name has no version field"
            fi
        fi
        
        # Check internal dependencies
        local dep_errors=0
        grep -E "claude-ai[^[:space:]]* = \\{ version = \"[^\"]*\"" "$cargo_file" 2>/dev/null | while read -r line; do
            local dep_name=$(echo "$line" | cut -d' ' -f1)
            local dep_version=$(echo "$line" | cut -d'"' -f2)
            if [ "$dep_version" != "$workspace_version" ]; then
                print_error "$crate_name dependency $dep_name version mismatch: expected $workspace_version, found $dep_version"
                dep_errors=$((dep_errors + 1))
            else
                print_success "$crate_name dependency $dep_name: $dep_version"
            fi
        done
        
        errors=$((errors + dep_errors))
        echo ""
    done
    
    echo ""
    if [ $errors -eq 0 ]; then
        print_success "All versions are consistent across the workspace"
        return 0
    else
        print_error "Found $errors version inconsistencies"
        return 1
    fi
}

# Check if version is already published on crates.io
check_published() {
    local version=$1
    if [ -z "$version" ]; then
        version=$(get_current_version)
    fi
    
    print_info "Checking if version $version is published on crates.io..."
    echo ""
    
    local crates=("claude-ai-core" "claude-ai-mcp" "claude-ai-runtime" "claude-ai" "claude-ai-interactive")
    local all_published=true
    
    for crate in "${crates[@]}"; do
        print_info "Checking $crate..."
        
        local response=$(curl -s "https://crates.io/api/v1/crates/$crate" 2>/dev/null || echo "")
        
        if [ -z "$response" ]; then
            print_error "$crate: Failed to fetch crate info from crates.io"
            all_published=false
            continue
        fi
        
        if echo "$response" | grep -q "\"name\":\"$crate\""; then
            if echo "$response" | grep -q "\"num\":\"$version\""; then
                print_success "$crate v$version is published"
            else
                print_warning "$crate v$version is NOT published"
                all_published=false
                
                # Show available versions
                local versions=$(echo "$response" | grep -o "\"num\":\"[^\"]*\"" | head -5 | cut -d'"' -f4)
                if [ -n "$versions" ]; then
                    print_info "Available versions: $(echo $versions | tr '\n' ' ')"
                fi
            fi
        else
            print_error "$crate: Crate not found on crates.io"
            all_published=false
        fi
        echo ""
    done
    
    if [ "$all_published" = true ]; then
        print_success "All crates version $version are published on crates.io"
        return 0
    else
        print_warning "Some crates are not published or have different versions"
        return 1
    fi
}

# Compare two versions and determine if it's a breaking change
compare_versions() {
    local old_version=$1
    local new_version=$2
    
    if [ -z "$old_version" ] || [ -z "$new_version" ]; then
        echo "Usage: compare_versions <old-version> <new-version>"
        return 1
    fi
    
    print_info "Comparing versions: $old_version → $new_version"
    
    # Extract major, minor, patch versions
    local old_major=$(echo "$old_version" | cut -d'.' -f1)
    local old_minor=$(echo "$old_version" | cut -d'.' -f2)
    local old_patch=$(echo "$old_version" | cut -d'.' -f3 | cut -d'-' -f1)
    
    local new_major=$(echo "$new_version" | cut -d'.' -f1)
    local new_minor=$(echo "$new_version" | cut -d'.' -f2)
    local new_patch=$(echo "$new_version" | cut -d'.' -f3 | cut -d'-' -f1)
    
    if [ "$new_major" -gt "$old_major" ]; then
        print_warning "MAJOR version bump: Breaking changes expected"
        echo "  - Public API may have breaking changes"
        echo "  - Migration guide required"
        echo "  - Thorough testing recommended"
    elif [ "$new_major" -eq "$old_major" ] && [ "$new_minor" -gt "$old_minor" ]; then
        print_info "MINOR version bump: New features, backward compatible"
        echo "  - New functionality added"
        echo "  - Existing API remains compatible"
        echo "  - Safe to upgrade"
    elif [ "$new_major" -eq "$old_major" ] && [ "$new_minor" -eq "$old_minor" ] && [ "$new_patch" -gt "$old_patch" ]; then
        print_info "PATCH version bump: Bug fixes, backward compatible"
        echo "  - Bug fixes and improvements"
        echo "  - No API changes"
        echo "  - Safe to upgrade"
    else
        print_error "Invalid version progression"
        echo "  - New version should be higher than old version"
        echo "  - Check version format and ordering"
        return 1
    fi
    
    # Check for pre-release versions
    if echo "$new_version" | grep -q "-"; then
        local prerelease=$(echo "$new_version" | cut -d'-' -f2-)
        print_info "Pre-release version: $prerelease"
        echo "  - Not recommended for production use"
        echo "  - May have unstable APIs or incomplete features"
    fi
}

# List all versions across workspace
list_versions() {
    print_info "Version summary for claude-ai workspace"
    echo ""
    
    local workspace_version=$(get_current_version)
    print_info "Workspace version: $workspace_version"
    echo ""
    
    print_info "Crate versions:"
    find . -name "Cargo.toml" -not -path "./target/*" -not -path "./Cargo.toml" | sort | while read -r cargo_file; do
        local crate_name=$(basename $(dirname "$cargo_file"))
        
        if grep -q "version.workspace = true" "$cargo_file"; then
            echo "  $crate_name: $workspace_version (workspace)"
        else
            local crate_version=$(grep "^version = " "$cargo_file" | head -n1 | cut -d'"' -f2 2>/dev/null || echo "not specified")
            echo "  $crate_name: $crate_version"
        fi
    done
    echo ""
    
    print_info "Internal dependencies:"
    find . -name "Cargo.toml" -not -path "./target/*" -not -path "./Cargo.toml" | while read -r cargo_file; do
        local crate_name=$(basename $(dirname "$cargo_file"))
        local deps=$(grep -E "claude-ai[^[:space:]]* = \\{ version = \"[^\"]*\"" "$cargo_file" 2>/dev/null | cut -d' ' -f1 | tr '\n' ' ')
        if [ -n "$deps" ]; then
            echo "  $crate_name depends on: $deps"
        fi
    done
}

# Get next version based on change type
next_version() {
    local current_version=$(get_current_version)
    local change_type=$1
    
    if [ -z "$change_type" ]; then
        echo "Usage: next_version <major|minor|patch>"
        return 1
    fi
    
    # Extract version components
    local major=$(echo "$current_version" | cut -d'.' -f1)
    local minor=$(echo "$current_version" | cut -d'.' -f2)
    local patch=$(echo "$current_version" | cut -d'.' -f3 | cut -d'-' -f1)
    
    case "$change_type" in
        "major")
            echo "$((major + 1)).0.0"
            ;;
        "minor")
            echo "$major.$((minor + 1)).0"
            ;;
        "patch")
            echo "$major.$minor.$((patch + 1))"
            ;;
        *)
            print_error "Invalid change type: $change_type"
            echo "Use: major, minor, or patch"
            return 1
            ;;
    esac
}

# Show help
show_help() {
    echo "Version management tools for claude-ai workspace"
    echo ""
    echo "Usage: $0 <command> [arguments]"
    echo ""
    echo "Commands:"
    echo "  current                          Show current workspace version"
    echo "  validate                         Check version consistency across workspace"
    echo "  check-published [version]        Check if version is published on crates.io"
    echo "  compare <old> <new>              Compare two versions and show change type"
    echo "  list                             List all versions in workspace"
    echo "  next <major|minor|patch>         Show next version for given change type"
    echo "  help                             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 current                       # Show current version"
    echo "  $0 validate                      # Check version consistency"
    echo "  $0 check-published 1.0.0        # Check if v1.0.0 is published"
    echo "  $0 compare 1.0.0 1.1.0           # Compare versions"
    echo "  $0 next minor                    # Show next minor version"
    echo ""
}

# Main command dispatch
case "$1" in
    "current")
        get_current_version
        ;;
    "validate")
        validate_versions
        ;;
    "check-published")
        check_published "$2"
        ;;
    "compare")
        compare_versions "$2" "$3"
        ;;
    "list")
        list_versions
        ;;
    "next")
        next_version "$2"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac