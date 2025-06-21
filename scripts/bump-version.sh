#!/bin/bash
# scripts/bump-version.sh - Automated version bumping for claude-ai workspace

set -e

NEW_VERSION=$1

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 1.1.0"
    echo ""
    echo "Supported formats:"
    echo "  Major: 2.0.0"
    echo "  Minor: 1.1.0"
    echo "  Patch: 1.0.1"
    echo "  Pre-release: 1.1.0-beta.1"
    exit 1
fi

# Validate semantic version format
if ! echo "$NEW_VERSION" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$'; then
    echo "Error: Invalid semantic version format. Use X.Y.Z or X.Y.Z-prerelease"
    exit 1
fi

# Get current version for comparison
CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2)

echo "Bumping version from $CURRENT_VERSION to $NEW_VERSION..."

# Check if we're in the workspace root
if [ ! -f "Cargo.toml" ] || ! grep -q "\[workspace\]" Cargo.toml; then
    echo "Error: Must be run from the workspace root directory"
    exit 1
fi

# Create backup before making changes
echo "Creating backup..."
cp Cargo.toml Cargo.toml.backup

# Update workspace version
echo "Updating workspace version..."
sed -i.tmp "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml
rm -f Cargo.toml.tmp

# Update internal dependency versions in all crate Cargo.toml files
echo "Updating internal dependency versions..."
find . -name "Cargo.toml" -not -path "./target/*" -not -path "./Cargo.toml" | while read -r cargo_file; do
    echo "  Updating $cargo_file..."
    
    # Create backup
    cp "$cargo_file" "${cargo_file}.backup"
    
    # Update claude-ai-* dependencies
    # Handle both path and version dependencies
    sed -i.tmp -E "s/(claude-ai[^[:space:]]*)[[:space:]]*=[[:space:]]*\\{[[:space:]]*version[[:space:]]*=[[:space:]]*\"[^\"]*\"/\\1 = { version = \"$NEW_VERSION\"/" "$cargo_file"
    
    # Clean up temporary file
    rm -f "${cargo_file}.tmp"
done

# Verify changes
echo "Verifying version consistency..."
WORKSPACE_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2)

if [ "$WORKSPACE_VERSION" != "$NEW_VERSION" ]; then
    echo "Error: Workspace version update failed"
    echo "Expected: $NEW_VERSION"
    echo "Found: $WORKSPACE_VERSION"
    exit 1
fi

# Check internal dependencies
ERRORS=0
find . -name "Cargo.toml" -not -path "./target/*" -not -path "./Cargo.toml" | while read -r cargo_file; do
    # Check claude-ai dependencies
    grep -E "claude-ai[^[:space:]]* = \\{ version = \"[^\"]*\"" "$cargo_file" | while read -r line; do
        DEP_VERSION=$(echo "$line" | cut -d'"' -f2)
        if [ "$DEP_VERSION" != "$NEW_VERSION" ]; then
            echo "Error: Version mismatch in $cargo_file"
            echo "  Expected: $NEW_VERSION"
            echo "  Found: $DEP_VERSION"
            ERRORS=$((ERRORS + 1))
        fi
    done
done

# Run basic validation
echo "Running basic validation..."
if ! cargo check --workspace >/dev/null 2>&1; then
    echo "Warning: 'cargo check' failed. Please review the changes."
    echo "You may need to run 'cargo update' to refresh the lock file."
else
    echo "✓ Basic validation passed"
fi

# Clean up backups if everything went well
echo "Cleaning up backups..."
rm -f Cargo.toml.backup
find . -name "Cargo.toml.backup" -not -path "./target/*" -delete

echo ""
echo "✓ Version successfully bumped from $CURRENT_VERSION to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Test the build: cargo test --workspace"
echo "  3. Update CHANGELOG.md"
echo "  4. Commit changes: git add -A && git commit -m 'Bump version to $NEW_VERSION'"
echo "  5. Create tag: git tag -a 'v$NEW_VERSION' -m 'Release v$NEW_VERSION'"
echo ""