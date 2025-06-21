//! Integration tests for examples
//!
//! These tests verify that examples can be built and basic functionality works.
//! They don't require Claude CLI authentication but test the code paths.

#[cfg(test)]
mod integration {
    use std::process::Command;

    #[test]
    fn test_examples_compile() {
        // Test that all examples compile successfully
        let output = Command::new("cargo")
            .args(&["build", "--examples"])
            .output()
            .expect("Failed to run cargo build");

        if !output.status.success() {
            panic!(
                "Examples failed to compile:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn test_example_help_output() {
        // Test that the complete app example shows help correctly
        let output = Command::new("cargo")
            .args(&["run", "--example", "05_complete_app", "--", "invalid"])
            .output()
            .expect("Failed to run example");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Usage:") || stdout.contains("chat") || output.status.success());
    }

    #[test]
    fn test_basic_sdk_structure() {
        // Test that the basic SDK example has the expected structure
        let example_content =
            std::fs::read_to_string("01_basic_sdk.rs").expect("Failed to read basic SDK example");

        // Check for key components
        assert!(example_content.contains("Client::new"));
        assert!(example_content.contains("Config::default"));
        assert!(example_content.contains("query"));
        assert!(example_content.contains("send"));
    }

    #[test]
    fn test_session_example_structure() {
        // Test that the session example has expected structure
        let example_content =
            std::fs::read_to_string("02_sdk_sessions.rs").expect("Failed to read session example");

        assert!(example_content.contains("SessionId"));
        assert!(example_content.contains("session"));
        assert!(example_content.contains("Uuid::new_v4"));
    }

    #[test]
    fn test_streaming_example_structure() {
        // Test that the streaming example has expected structure
        let example_content =
            std::fs::read_to_string("03_streaming.rs").expect("Failed to read streaming example");

        assert!(example_content.contains("StreamFormat::StreamJson"));
        assert!(example_content.contains("stream()"));
        assert!(example_content.contains("Message::"));
    }

    #[test]
    fn test_tools_example_structure() {
        // Test that the tools example has expected structure
        let example_content =
            std::fs::read_to_string("04_tools.rs").expect("Failed to read tools example");

        assert!(example_content.contains("ToolPermission"));
        assert!(example_content.contains("allowed_tools"));
        assert!(example_content.contains("bash"));
    }
}
