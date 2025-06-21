#!/bin/bash
# Script to run all examples demonstrating claude-ai and claude-interactive

echo "🚀 Claude AI Examples Runner"
echo "=========================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run an example
run_example() {
    local example_name=$1
    local description=$2
    
    echo -e "${BLUE}Running Example: ${example_name}${NC}"
    echo "Description: ${description}"
    echo "---"
    
    # Note: Some examples require interactive input or Claude CLI authentication
    cargo run --example $example_name 2>/dev/null || {
        echo "Note: This example requires Claude CLI to be authenticated"
        echo "Run 'claude login' first if you haven't already"
    }
    
    echo ""
    echo "Press Enter to continue..."
    read
    clear
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Please run this script from the examples directory"
    exit 1
fi

# Main menu
while true; do
    echo -e "${GREEN}Claude AI Examples${NC}"
    echo "=================="
    echo ""
    echo "1. Basic SDK Usage - Simple queries and configuration"
    echo "2. Session Management - Using sessions for context"
    echo "3. Streaming Responses - Real-time response processing"
    echo "4. Tool Integration - Using filesystem and bash tools"
    echo "5. Complete Application - Full CLI assistant"
    echo "6. Run All Examples"
    echo "0. Exit"
    echo ""
    echo -n "Select an example (0-6): "
    
    read choice
    
    case $choice in
        1)
            clear
            run_example "01_basic_sdk" "Simple queries and configuration"
            ;;
        2)
            clear
            run_example "02_sdk_sessions" "Using sessions for context"
            ;;
        3)
            clear
            run_example "03_streaming" "Real-time response processing"
            ;;
        4)
            clear
            run_example "04_tools" "Using filesystem and bash tools"
            ;;
        5)
            clear
            echo "Note: The complete app example has multiple modes:"
            echo "  cargo run --example 05_complete_app chat"
            echo "  cargo run --example 05_complete_app dev" 
            echo "  cargo run --example 05_complete_app analysis"
            echo ""
            echo "Try: cargo run --example 05_complete_app dev"
            echo ""
            echo "Press Enter to continue..."
            read
            clear
            ;;
        6)
            clear
            echo "Running all examples..."
            echo ""
            for i in {1..4}; do
                case $i in
                    1) run_example "01_basic_sdk" "Basic SDK usage" ;;
                    2) run_example "02_sdk_sessions" "Session management" ;;
                    3) run_example "03_streaming" "Streaming responses" ;;
                    4) run_example "04_tools" "Tool integration" ;;
                esac
            done
            ;;
        0)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please select 0-6."
            echo "Press Enter to continue..."
            read
            clear
            ;;
    esac
done