#!/bin/bash
# scripts/release.sh - Complete release process for claude-ai workspace

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

VERSION=$1
CHANGELOG_ENTRY=$2
DRY_RUN=${DRY_RUN:-false}

show_help() {
    echo "Complete release process for claude-ai workspace"
    echo ""
    echo "Usage: $0 <version> [changelog-entry]"
    echo ""
    echo "Arguments:"
    echo "  version           New version number (e.g., 1.1.0, 2.0.0-beta.1)"
    echo "  changelog-entry   Optional description of changes"
    echo ""
    echo "Environment variables:"
    echo "  DRY_RUN=true     Perform dry run without making changes"
    echo ""
    echo "Examples:"
    echo "  $0 1.1.0 'Add streaming support and performance improvements'"
    echo "  $0 2.0.0-beta.1 'Breaking changes for v2 API'"
    echo "  DRY_RUN=true $0 1.0.1 'Bug fixes'"
    echo ""
    echo "The script will:"
    echo "  1. Validate version format and git status"
    echo "  2. Update version across all crates"
    echo "  3. Update CHANGELOG.md (if entry provided)"
    echo "  4. Run comprehensive validation tests"
    echo "  5. Commit changes and create git tag"
    echo "  6. Optionally publish to crates.io"
    echo ""
}

if [ -z "$VERSION" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Validate version format
if ! echo "$VERSION" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$'; then
    print_error "Invalid semantic version format: $VERSION"
    echo "Use format: X.Y.Z or X.Y.Z-prerelease"
    exit 1
fi

# Check if we're in the workspace root
if [ ! -f "Cargo.toml" ] || ! grep -q "\[workspace\]" Cargo.toml; then
    print_error "Must be run from the workspace root directory"
    exit 1
fi

# Get current version for comparison
CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2)

print_step "Preparing release $VERSION (current: $CURRENT_VERSION)"

if [ "$DRY_RUN" = "true" ]; then
    print_warning "DRY RUN MODE - No changes will be made"
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_error "Uncommitted changes detected"
    echo ""
    echo "Please commit or stash changes before releasing:"
    git status --porcelain
    exit 1
fi

# Check if we're on the main branch (or master)
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
    print_warning "Not on main/master branch (currently on: $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Validate version progression
print_step "Validating version progression"

# Extract version components
OLD_MAJOR=$(echo "$CURRENT_VERSION" | cut -d'.' -f1)
OLD_MINOR=$(echo "$CURRENT_VERSION" | cut -d'.' -f2)
OLD_PATCH=$(echo "$CURRENT_VERSION" | cut -d'.' -f3 | cut -d'-' -f1)

NEW_MAJOR=$(echo "$VERSION" | cut -d'.' -f1)
NEW_MINOR=$(echo "$VERSION" | cut -d'.' -f2)
NEW_PATCH=$(echo "$VERSION" | cut -d'.' -f3 | cut -d'-' -f1)

# Check version progression is valid
if [ "$NEW_MAJOR" -lt "$OLD_MAJOR" ] || \
   ([ "$NEW_MAJOR" -eq "$OLD_MAJOR" ] && [ "$NEW_MINOR" -lt "$OLD_MINOR" ]) || \
   ([ "$NEW_MAJOR" -eq "$OLD_MAJOR" ] && [ "$NEW_MINOR" -eq "$OLD_MINOR" ] && [ "$NEW_PATCH" -le "$OLD_PATCH" ]); then
    # Exception: allow same version if it's adding/changing pre-release suffix
    if ! ([ "$NEW_MAJOR" -eq "$OLD_MAJOR" ] && [ "$NEW_MINOR" -eq "$OLD_MINOR" ] && [ "$NEW_PATCH" -eq "$OLD_PATCH" ] && echo "$VERSION" | grep -q "-"); then
        print_error "New version ($VERSION) is not greater than current version ($CURRENT_VERSION)"
        exit 1
    fi
fi

# Determine change type
if [ "$NEW_MAJOR" -gt "$OLD_MAJOR" ]; then
    CHANGE_TYPE="major"
    print_warning "MAJOR version bump - breaking changes expected"
elif [ "$NEW_MINOR" -gt "$OLD_MINOR" ]; then
    CHANGE_TYPE="minor"
    print_success "MINOR version bump - new features, backward compatible"
else
    CHANGE_TYPE="patch"
    print_success "PATCH version bump - bug fixes, backward compatible"
fi

if echo "$VERSION" | grep -q "-"; then
    PRERELEASE=$(echo "$VERSION" | cut -d'-' -f2-)
    print_warning "Pre-release version: $PRERELEASE"
fi

# Update version
print_step "Updating version to $VERSION"

if [ "$DRY_RUN" = "true" ]; then
    print_warning "Would update version from $CURRENT_VERSION to $VERSION"
else
    ./scripts/bump-version.sh "$VERSION"
    print_success "Version updated successfully"
fi

# Update changelog if entry provided
if [ -n "$CHANGELOG_ENTRY" ]; then
    print_step "Updating CHANGELOG.md"
    
    if [ "$DRY_RUN" = "true" ]; then
        print_warning "Would add changelog entry: $CHANGELOG_ENTRY"
    else
        # Check if CHANGELOG.md exists
        if [ ! -f "CHANGELOG.md" ]; then
            print_warning "CHANGELOG.md not found, creating..."
            cat > CHANGELOG.md << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [$VERSION] - $(date +%Y-%m-%d)
- $CHANGELOG_ENTRY
EOF
        else
            # Insert new version entry after [Unreleased]
            if grep -q "## \[Unreleased\]" CHANGELOG.md; then
                sed -i.bak "/## \[Unreleased\]/a\\
\\
## [$VERSION] - $(date +%Y-%m-%d)\\
- $CHANGELOG_ENTRY" CHANGELOG.md
                rm -f CHANGELOG.md.bak
            else
                print_warning "Could not find [Unreleased] section in CHANGELOG.md"
                print_warning "Please update CHANGELOG.md manually"
            fi
        fi
        print_success "CHANGELOG.md updated"
    fi
fi

# Run comprehensive validation
print_step "Running validation checks"

if [ "$DRY_RUN" = "true" ]; then
    print_warning "Would run validation checks"
else
    # Check version consistency
    print_step "Checking version consistency..."
    if ! ./scripts/version-tools.sh validate; then
        print_error "Version consistency check failed"
        exit 1
    fi
    
    # Update Cargo.lock
    print_step "Updating Cargo.lock..."
    cargo update --workspace
    
    # Check that everything builds
    print_step "Building workspace..."
    if ! cargo build --workspace --all-features; then
        print_error "Build failed"
        exit 1
    fi
    
    # Run tests
    print_step "Running test suite..."
    if ! cargo test --workspace --all-features; then
        print_error "Tests failed"
        exit 1
    fi
    
    # Run clippy
    print_step "Running clippy..."
    if ! cargo clippy --workspace --all-targets --all-features -- -D warnings; then
        print_error "Clippy warnings found"
        exit 1
    fi
    
    # Check formatting
    print_step "Checking code formatting..."
    if ! cargo fmt --check; then
        print_error "Code formatting issues found. Run 'cargo fmt' to fix."
        exit 1
    fi
    
    # Build documentation
    print_step "Building documentation..."
    if ! cargo doc --workspace --all-features --no-deps; then
        print_error "Documentation build failed"
        exit 1
    fi
    
    print_success "All validation checks passed"
fi

# Commit changes and create tag
print_step "Creating git commit and tag"

if [ "$DRY_RUN" = "true" ]; then
    print_warning "Would create commit and tag v$VERSION"
else
    # Commit changes
    git add -A
    
    COMMIT_MSG="Release v$VERSION"
    if [ -n "$CHANGELOG_ENTRY" ]; then
        COMMIT_MSG="$COMMIT_MSG

Changes: $CHANGELOG_ENTRY"
    fi
    
    COMMIT_MSG="$COMMIT_MSG

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    git commit -m "$COMMIT_MSG"
    
    # Create tag
    TAG_MSG="Release v$VERSION"
    if [ -n "$CHANGELOG_ENTRY" ]; then
        TAG_MSG="$TAG_MSG

$CHANGELOG_ENTRY"
    fi
    
    git tag -a "v$VERSION" -m "$TAG_MSG"
    
    print_success "Created commit and tag v$VERSION"
fi

# Show next steps
print_step "Release v$VERSION prepared successfully!"

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    print_success "Dry run completed successfully"
    echo ""
    echo "To perform actual release:"
    echo "  DRY_RUN=false $0 $VERSION $([ -n "$CHANGELOG_ENTRY" ] && echo "\"$CHANGELOG_ENTRY\"")"
else
    echo ""
    echo "Next steps:"
    echo "  1. Review the changes:"
    echo "     git show --name-only"
    echo "     git log --oneline -5"
    echo ""
    echo "  2. Push to remote repository:"
    echo "     git push origin $(git branch --show-current)"
    echo "     git push origin v$VERSION"
    echo ""
    echo "  3. Publish to crates.io:"
    echo "     ./scripts/publish.sh"
    echo "  OR for dry run first:"
    echo "     DRY_RUN=true ./scripts/publish.sh"
    echo ""
    echo "  4. Create GitHub release (if applicable):"
    echo "     gh release create v$VERSION --title 'Release v$VERSION' --notes-file RELEASE_NOTES.md"
    echo ""
    
    # Ask if user wants to publish now
    echo -n "Publish to crates.io now? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_step "Publishing to crates.io..."
        if ./scripts/publish.sh; then
            print_success "Successfully published to crates.io!"
        else
            print_error "Publication failed. You can retry with: ./scripts/publish.sh"
        fi
    fi
fi