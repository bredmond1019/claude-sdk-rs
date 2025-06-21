#!/usr/bin/env python3
"""
Script to extract and test all Rust code snippets from tutorials.
This ensures all examples compile and are syntactically correct.
"""

import os
import re
import subprocess
import tempfile
import sys
from pathlib import Path

def extract_rust_code_blocks(file_path):
    """Extract all Rust code blocks from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all ```rust code blocks
    pattern = r'```rust\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    code_blocks = []
    for i, match in enumerate(matches):
        # Skip incomplete examples that are just fragments
        if any(fragment in match for fragment in [
            'async fn main() -> Result<(), Box<dyn std::error::Error>> {',
            'use claude_sdk_rs::{',
            '#[tokio::main]'
        ]):
            code_blocks.append((i, match))
    
    return code_blocks

def create_test_cargo_toml():
    """Create a basic Cargo.toml for testing."""
    return """[package]
name = "tutorial-test"
version = "0.1.0"
edition = "2021"

[dependencies]
claude-sdk-rs = { path = "..", features = ["full"] }
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
futures = "0.3"
axum = { version = "0.7", optional = true }
tower = { version = "0.4", optional = true }
tower-http = { version = "0.5", optional = true }
chrono = { version = "0.4", features = ["serde"], optional = true }
prometheus = { version = "0.13", optional = true }
tracing = { version = "0.1", optional = true }
tracing-subscriber = { version = "0.3", optional = true }
thiserror = "1.0"
walkdir = "2.0"
sysinfo = "0.30"
"""

def make_code_testable(code):
    """Make a code snippet testable by wrapping it if needed."""
    # If it's already a complete program, return as-is
    if '#[tokio::main]' in code or 'fn main()' in code:
        return code
    
    # If it's just imports and types, wrap in a simple main
    if not ('async fn' in code or 'fn ' in code or 'impl ' in code):
        return f"""
{code}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    println!("Code compiles successfully!");
    Ok(())
}}
"""
    
    # For function definitions, add a main that can call them
    return f"""
{code}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    println!("Functions compile successfully!");
    Ok(())
}}
"""

def test_code_block(code, block_index, temp_dir):
    """Test a single code block for compilation."""
    # Create src directory
    src_dir = temp_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Write Cargo.toml
    cargo_toml = temp_dir / "Cargo.toml"
    cargo_toml.write_text(create_test_cargo_toml())
    
    # Write the code to main.rs
    main_rs = src_dir / "main.rs"
    testable_code = make_code_testable(code)
    main_rs.write_text(testable_code)
    
    # Try to compile
    try:
        result = subprocess.run(
            ["cargo", "check", "--quiet"], 
            cwd=temp_dir, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, "OK"
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except Exception as e:
        return False, f"Error: {e}"

def test_tutorial_file(file_path):
    """Test all code blocks in a tutorial file."""
    print(f"\n=== Testing {file_path.name} ===")
    
    code_blocks = extract_rust_code_blocks(file_path)
    if not code_blocks:
        print("  No testable Rust code blocks found")
        return True
    
    print(f"  Found {len(code_blocks)} code blocks to test")
    
    all_passed = True
    
    for block_index, code in code_blocks:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            success, message = test_code_block(code, block_index, temp_path)
            
            if success:
                print(f"  ✅ Block {block_index + 1}: OK")
            else:
                print(f"  ❌ Block {block_index + 1}: FAILED")
                print(f"     Error: {message}")
                # Print the problematic code for debugging
                print(f"     Code preview:")
                lines = code.split('\n')[:5]
                for line in lines:
                    print(f"       {line}")
                total_lines = len(code.split('\n'))
                if total_lines > 5:
                    print(f"       ... ({total_lines - 5} more lines)")
                all_passed = False
    
    return all_passed

def main():
    # Find the project root (where this script is located)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tutorials_dir = project_root / "docs" / "tutorials"
    
    if not tutorials_dir.exists():
        print(f"Error: Tutorials directory not found at {tutorials_dir}")
        sys.exit(1)
    
    print("Testing all tutorial code snippets...")
    print(f"Project root: {project_root}")
    print(f"Tutorials directory: {tutorials_dir}")
    
    # Find all tutorial files
    tutorial_files = sorted(tutorials_dir.glob("*.md"))
    
    if not tutorial_files:
        print("No tutorial files found!")
        sys.exit(1)
    
    all_tutorials_passed = True
    
    for tutorial_file in tutorial_files:
        passed = test_tutorial_file(tutorial_file)
        if not passed:
            all_tutorials_passed = False
    
    print("\n" + "="*50)
    if all_tutorials_passed:
        print("✅ All tutorial code snippets compiled successfully!")
        sys.exit(0)
    else:
        print("❌ Some tutorial code snippets failed to compile!")
        print("Please fix the issues above and run the test again.")
        sys.exit(1)

if __name__ == "__main__":
    main()