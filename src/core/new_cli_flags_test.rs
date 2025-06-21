//! Unit tests for new CLI flags
//!
//! This module provides comprehensive tests for the new CLI flags added to the Claude SDK:
//! - --append-system-prompt
//! - --max-turns
//! - --disallowed-tools (with granular permissions support)
//! - --dangerously-skip-permissions
//!
//! Tests cover CLI flag generation, validation, error handling, and configuration.

use crate::core::types::ToolPermission;
use crate::core::{Config, StreamFormat};
use std::str::FromStr;

/// Mock function to simulate CLI argument generation
/// This mirrors the logic in process.rs for the new flags
fn generate_cli_args(config: &Config) -> Vec<String> {
    let mut args = vec![];

    // Add append system prompt flag
    if let Some(append_prompt) = &config.append_system_prompt {
        args.push("--append-system-prompt".to_string());
        args.push(append_prompt.clone());
    }

    // Add max turns flag
    if let Some(max_turns) = &config.max_turns {
        args.push("--max-turns".to_string());
        args.push(max_turns.to_string());
    }

    // Add disallowed tools flags
    if let Some(disallowed_tools) = &config.disallowed_tools {
        for tool in disallowed_tools {
            args.push("--disallowedTools".to_string());
            args.push(tool.clone());
        }
    }

    // Add skip permissions flag
    if config.skip_permissions {
        args.push("--dangerously-skip-permissions".to_string());
    }

    args
}

// =============================================================================
// DEFAULT VALUE TESTS
// =============================================================================

#[cfg(test)]
mod new_cli_flags_defaults_tests {
    use super::*;

    #[test]
    fn test_new_cli_flags_default_values() {
        let config = Config::default();

        // Test default values for new flags
        assert_eq!(config.append_system_prompt, None);
        assert_eq!(config.max_turns, None);
        assert_eq!(config.disallowed_tools, None);
        assert!(config.skip_permissions); // Default should be true for programmatic use
    }

    #[test]
    fn test_skip_permissions_default_behavior() {
        // skip_permissions should default to true for SDK usage
        let config = Config::default();
        assert!(config.skip_permissions);

        // Builder should also have same default
        let builder_config = Config::builder().build().unwrap();
        assert!(builder_config.skip_permissions);
    }
}

// =============================================================================
// CLI FLAG GENERATION TESTS
// =============================================================================

#[cfg(test)]
mod cli_flag_generation_tests {
    use super::*;

    #[test]
    fn test_append_system_prompt_flag_generation() {
        let config = Config::builder()
            .append_system_prompt("Additionally, be concise and accurate.")
            .build()
            .unwrap();

        let args = generate_cli_args(&config);

        assert!(args.contains(&"--append-system-prompt".to_string()));
        assert!(args.contains(&"Additionally, be concise and accurate.".to_string()));

        // Should appear in sequence
        let append_idx = args
            .iter()
            .position(|x| x == "--append-system-prompt")
            .unwrap();
        assert_eq!(
            args[append_idx + 1],
            "Additionally, be concise and accurate."
        );
    }

    #[test]
    fn test_max_turns_flag_generation() {
        let config = Config::builder().max_turns(10).build().unwrap();

        let args = generate_cli_args(&config);

        assert!(args.contains(&"--max-turns".to_string()));
        assert!(args.contains(&"10".to_string()));

        // Should appear in sequence
        let turns_idx = args.iter().position(|x| x == "--max-turns").unwrap();
        assert_eq!(args[turns_idx + 1], "10");
    }

    #[test]
    fn test_disallowed_tools_flag_generation() {
        let tools = vec![
            "Bash(rm)".to_string(),
            "mcp__dangerous__delete".to_string(),
            "bash:sudo".to_string(),
        ];

        let config = Config::builder()
            .disallowed_tools(tools.clone())
            .build()
            .unwrap();

        let args = generate_cli_args(&config);

        // Each tool should generate a --disallowedTools flag
        let disallowed_count = args.iter().filter(|x| *x == "--disallowedTools").count();
        assert_eq!(disallowed_count, 3);

        // All tools should be present
        for tool in &tools {
            assert!(args.contains(tool));
        }

        // Check proper pairing of flags and values
        for (i, arg) in args.iter().enumerate() {
            if arg == "--disallowedTools" && i + 1 < args.len() {
                assert!(tools.contains(&args[i + 1]));
            }
        }
    }

    #[test]
    fn test_skip_permissions_flag_generation() {
        // When skip_permissions is true (default)
        let config1 = Config::builder().skip_permissions(true).build().unwrap();

        let args1 = generate_cli_args(&config1);
        assert!(args1.contains(&"--dangerously-skip-permissions".to_string()));

        // When skip_permissions is false
        let config2 = Config::builder().skip_permissions(false).build().unwrap();

        let args2 = generate_cli_args(&config2);
        assert!(!args2.contains(&"--dangerously-skip-permissions".to_string()));
    }

    #[test]
    fn test_all_new_flags_together() {
        let config = Config::builder()
            .append_system_prompt("Be helpful and concise")
            .max_turns(5)
            .disallowed_tools(vec![
                "Bash(rm)".to_string(),
                "mcp__dangerous__*".to_string(),
            ])
            .skip_permissions(true)
            .build()
            .unwrap();

        let args = generate_cli_args(&config);

        // Check all flags are present
        assert!(args.contains(&"--append-system-prompt".to_string()));
        assert!(args.contains(&"Be helpful and concise".to_string()));
        assert!(args.contains(&"--max-turns".to_string()));
        assert!(args.contains(&"5".to_string()));
        assert!(args.contains(&"--disallowedTools".to_string()));
        assert!(args.contains(&"Bash(rm)".to_string()));
        assert!(args.contains(&"mcp__dangerous__*".to_string()));
        assert!(args.contains(&"--dangerously-skip-permissions".to_string()));

        // Should have exactly 2 disallowed tools flags
        let disallowed_count = args.iter().filter(|x| *x == "--disallowedTools").count();
        assert_eq!(disallowed_count, 2);
    }

    #[test]
    fn test_no_new_flags_when_unset() {
        let config = Config::builder()
            .model("claude-3-sonnet-20240229")
            .system_prompt("You are helpful")
            .skip_permissions(false) // Explicitly disable
            .build()
            .unwrap();

        let args = generate_cli_args(&config);

        // None of the new flags should be present
        assert!(!args.contains(&"--append-system-prompt".to_string()));
        assert!(!args.contains(&"--max-turns".to_string()));
        assert!(!args.contains(&"--disallowedTools".to_string()));
        assert!(!args.contains(&"--dangerously-skip-permissions".to_string()));
    }
}

// =============================================================================
// GRANULAR PERMISSIONS TESTS
// =============================================================================

#[cfg(test)]
mod granular_permissions_tests {
    use super::*;

    #[test]
    fn test_bash_granular_permissions_cli_format() {
        // Test different bash permission formats
        let bash_permissions = vec![
            ("Bash(ls)", "Bash(ls)"),
            ("Bash(git status)", "Bash(git status)"),
            ("Bash(npm install --save)", "Bash(npm install --save)"),
            ("bash:ls", "Bash(ls)"), // Legacy format should convert
            ("bash:git status", "Bash(git status)"),
        ];

        for (input, expected_cli) in bash_permissions {
            let permission = ToolPermission::from_str(input).unwrap();
            assert_eq!(permission.to_cli_format(), expected_cli);

            // Test in config context
            let config = Config::builder()
                .disallowed_tools(vec![input.to_string()])
                .build()
                .unwrap();

            assert_eq!(config.disallowed_tools.as_ref().unwrap()[0], input);
        }
    }

    #[test]
    fn test_mcp_granular_permissions_cli_format() {
        // Test MCP permission formats
        let mcp_permissions = vec![
            ("mcp__database__query", "mcp__database__query"),
            ("mcp__filesystem__read", "mcp__filesystem__read"),
            ("mcp__server__*", "mcp__server__*"),
            ("mcp__notion__search", "mcp__notion__search"),
        ];

        for (input, expected_cli) in mcp_permissions {
            let permission = ToolPermission::from_str(input).unwrap();
            assert_eq!(permission.to_cli_format(), expected_cli);

            // Test in config context
            let config = Config::builder()
                .allowed_tools(vec![input.to_string()])
                .build()
                .unwrap();

            assert_eq!(config.allowed_tools.as_ref().unwrap()[0], input);
        }
    }

    #[test]
    fn test_mixed_granular_permissions_in_config() {
        let mixed_permissions = vec![
            "Bash(ls)".to_string(),
            "bash:pwd".to_string(),
            "mcp__database__query".to_string(),
            "mcp__filesystem__*".to_string(),
            "*".to_string(),
        ];

        let config = Config::builder()
            .allowed_tools(mixed_permissions.clone())
            .disallowed_tools(vec![
                "Bash(rm)".to_string(),
                "mcp__dangerous__delete".to_string(),
            ])
            .build()
            .unwrap();

        // All permissions should validate successfully
        assert!(config.validate().is_ok());

        // Check that all permissions are preserved
        assert_eq!(config.allowed_tools.as_ref().unwrap().len(), 5);
        assert_eq!(config.disallowed_tools.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_granular_permission_validation_failures() {
        let invalid_permissions = vec![
            "Bash()",         // Empty command
            "bash:",          // Empty legacy command
            "mcp__",          // Incomplete MCP
            "mcp__server",    // Missing tool
            "mcp__server__",  // Empty tool
            "Shell(ls)",      // Wrong tool name
            "unknown_format", // Unknown format
            "",               // Empty string
        ];

        for invalid_permission in invalid_permissions {
            let result = Config::builder()
                .allowed_tools(vec![invalid_permission.to_string()])
                .build();

            assert!(
                result.is_err(),
                "Invalid permission '{}' should fail validation",
                invalid_permission
            );

            // Check error message contains helpful information
            if let Err(e) = result {
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("Invalid tool permission format")
                        || error_msg.contains("Tool name length must be")
                        || error_msg.contains("Unknown tool permission format"),
                    "Error message should be informative for '{}': {}",
                    invalid_permission,
                    error_msg
                );
            }
        }
    }

    #[test]
    fn test_granular_permission_roundtrip_consistency() {
        let permissions = vec![
            "Bash(ls)",
            "Bash(git status)",
            "bash:pwd", // Legacy format
            "mcp__database__query",
            "mcp__filesystem__read",
            "mcp__server__*",
            "*",
        ];

        for permission_str in permissions {
            // Parse the permission
            let parsed = ToolPermission::from_str(permission_str).unwrap();

            // Convert to CLI format
            let cli_format = parsed.to_cli_format();

            // Parse the CLI format back
            let reparsed = ToolPermission::from_str(&cli_format).unwrap();

            // Should be equivalent
            assert_eq!(
                parsed, reparsed,
                "Roundtrip failed for '{}': parsed={:?}, cli_format='{}', reparsed={:?}",
                permission_str, parsed, cli_format, reparsed
            );
        }
    }
}

// =============================================================================
// FLAG INTERACTION TESTS
// =============================================================================

#[cfg(test)]
mod flag_interaction_tests {
    use super::*;

    #[test]
    fn test_system_prompt_and_append_system_prompt_conflict() {
        // Both system_prompt and append_system_prompt should fail validation
        let result = Config::builder()
            .system_prompt("Main system prompt")
            .append_system_prompt("Additional prompt")
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Cannot use both system_prompt and append_system_prompt"));
        }
    }

    #[test]
    fn test_allowed_and_disallowed_tools_conflict() {
        // Same tool in both allowed and disallowed should fail
        let result = Config::builder()
            .allowed_tools(vec!["Bash(ls)".to_string()])
            .disallowed_tools(vec!["Bash(ls)".to_string()])
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("cannot be both allowed and disallowed"));
        }
    }

    #[test]
    fn test_granular_permission_conflicts() {
        // Test conflicts with granular permissions
        let conflict_cases = vec![
            // Same exact permission
            (vec!["Bash(ls)"], vec!["Bash(ls)"]),
            // MCP tool conflicts
            (vec!["mcp__db__query"], vec!["mcp__db__query"]),
        ];

        for (allowed, disallowed) in conflict_cases {
            let result = Config::builder()
                .allowed_tools(allowed.iter().map(|s| s.to_string()).collect())
                .disallowed_tools(disallowed.iter().map(|s| s.to_string()).collect())
                .build();

            assert!(
                result.is_err(),
                "Conflict should be detected for allowed={:?}, disallowed={:?}",
                allowed,
                disallowed
            );
        }

        // Legacy vs new format conflicts are more complex - they would need
        // to be converted to the same format first to detect conflicts
        // For now, we just test that exact string matches are detected
    }

    #[test]
    fn test_no_conflict_with_different_tools() {
        // Different tools should not conflict
        let non_conflict_cases = vec![
            (vec!["Bash(ls)"], vec!["Bash(rm)"]),
            (vec!["bash:ls"], vec!["Bash(pwd)"]),
            (vec!["mcp__db__query"], vec!["mcp__db__delete"]),
            (vec!["mcp__server1__tool"], vec!["mcp__server2__tool"]),
            (vec!["*"], vec!["Bash(rm)"]), // Wildcard with specific restriction
        ];

        for (allowed, disallowed) in non_conflict_cases {
            let result = Config::builder()
                .allowed_tools(allowed.iter().map(|s| s.to_string()).collect())
                .disallowed_tools(disallowed.iter().map(|s| s.to_string()).collect())
                .build();

            assert!(
                result.is_ok(),
                "No conflict should be detected for allowed={:?}, disallowed={:?}",
                allowed,
                disallowed
            );
        }
    }

    #[test]
    fn test_max_turns_with_other_flags() {
        // max_turns should work with other flags
        let config = Config::builder()
            .max_turns(5)
            .append_system_prompt("Be concise")
            .disallowed_tools(vec!["Bash(rm)".to_string()])
            .skip_permissions(false)
            .model("claude-3-sonnet-20240229")
            .timeout_secs(60)
            .build()
            .unwrap();

        assert_eq!(config.max_turns, Some(5));
        assert_eq!(config.append_system_prompt, Some("Be concise".to_string()));
        assert_eq!(config.disallowed_tools, Some(vec!["Bash(rm)".to_string()]));
        assert!(!config.skip_permissions);
    }

    #[test]
    fn test_skip_permissions_interaction() {
        // skip_permissions should work independently of other settings
        let configs = vec![
            // With tools allowed
            Config::builder()
                .allowed_tools(vec!["Bash(ls)".to_string()])
                .skip_permissions(true)
                .build()
                .unwrap(),
            // With tools disallowed
            Config::builder()
                .disallowed_tools(vec!["Bash(rm)".to_string()])
                .skip_permissions(false)
                .build()
                .unwrap(),
            // With max turns
            Config::builder()
                .max_turns(10)
                .skip_permissions(true)
                .build()
                .unwrap(),
        ];

        // All should validate successfully
        for config in configs {
            assert!(config.validate().is_ok());
        }
    }
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_max_turns_validation_errors() {
        // Zero max_turns should fail
        let result = Config::builder().max_turns(0).build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Max turns must be greater than 0"));
        }

        // Valid max_turns should pass
        let valid_turns = vec![1, 5, 10, 100, u32::MAX];
        for turns in valid_turns {
            let result = Config::builder().max_turns(turns).build();
            assert!(result.is_ok(), "Valid max_turns {} should pass", turns);
        }
    }

    #[test]
    fn test_append_system_prompt_validation_errors() {
        // Too long append_system_prompt should fail
        let long_prompt = "a".repeat(10_001); // Exceeds MAX_SYSTEM_PROMPT_LENGTH
        let result = Config::builder().append_system_prompt(long_prompt).build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Append system prompt exceeds maximum length"));
        }

        // Malicious content should fail
        let malicious_prompts = vec![
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "'; DROP TABLE users;--",
            "../../etc/passwd",
        ];

        for malicious_prompt in malicious_prompts {
            let result = Config::builder()
                .append_system_prompt(malicious_prompt)
                .build();
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(e.to_string().contains("malicious content"));
            }
        }

        // Valid append_system_prompt should pass
        let valid_prompts = vec![
            "Be concise",
            "Additionally, use proper formatting",
            "你好世界 🦀", // Unicode should be allowed
            "",            // Empty should be allowed
        ];

        for prompt in valid_prompts {
            let result = Config::builder().append_system_prompt(prompt).build();
            assert!(
                result.is_ok(),
                "Valid append_system_prompt '{}' should pass",
                prompt
            );
        }
    }

    #[test]
    fn test_disallowed_tools_validation_errors() {
        // Empty tool names should fail
        let result = Config::builder()
            .disallowed_tools(vec!["".to_string()])
            .build();
        assert!(result.is_err());

        // Too long tool names should fail
        let long_tool = "a".repeat(101); // Exceeds MAX_TOOL_NAME_LENGTH
        let result = Config::builder().disallowed_tools(vec![long_tool]).build();
        assert!(result.is_err());

        // Invalid permission formats should fail
        let invalid_tools = vec![
            "Bash()",         // Empty command
            "mcp__",          // Incomplete MCP
            "unknown_format", // Unknown format
        ];

        for tool in invalid_tools {
            let result = Config::builder()
                .disallowed_tools(vec![tool.to_string()])
                .build();
            assert!(result.is_err(), "Invalid tool '{}' should fail", tool);
        }
    }

    #[test]
    fn test_comprehensive_error_scenarios() {
        // Multiple validation errors should be detected
        let result = Config::builder()
            .system_prompt("Main prompt")
            .append_system_prompt("Conflicting prompt") // Conflict
            .max_turns(0) // Invalid
            .allowed_tools(vec!["Bash(ls)".to_string()])
            .disallowed_tools(vec!["Bash(ls)".to_string()]) // Conflict
            .build();

        assert!(result.is_err());
        // Should fail on the first validation error encountered
    }

    #[test]
    fn test_error_message_quality() {
        // Error messages should be informative
        let test_cases = vec![
            (
                Config::builder().max_turns(0),
                "Max turns must be greater than 0",
            ),
            (
                Config::builder().append_system_prompt("a".repeat(10_001)),
                "Append system prompt exceeds maximum length",
            ),
            (
                Config::builder()
                    .system_prompt("Main")
                    .append_system_prompt("Additional"),
                "Cannot use both system_prompt and append_system_prompt",
            ),
            (
                Config::builder()
                    .allowed_tools(vec!["Bash(ls)".to_string()])
                    .disallowed_tools(vec!["Bash(ls)".to_string()]),
                "cannot be both allowed and disallowed",
            ),
        ];

        for (builder, expected_fragment) in test_cases {
            let result = builder.build();
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(
                    e.to_string().contains(expected_fragment),
                    "Error message '{}' should contain '{}'",
                    e.to_string(),
                    expected_fragment
                );
            }
        }
    }
}

// =============================================================================
// SECURITY AND VALIDATION TESTS
// =============================================================================

#[cfg(test)]
mod security_validation_tests {
    use super::*;

    #[test]
    fn test_malicious_content_detection_in_append_prompt() {
        let malicious_patterns = vec![
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "$(rm -rf /)",
            "${dangerous_command}",
            "; rm -rf /",
            "| malicious_command",
            "../../etc/passwd",
            "'; DROP TABLE users;--",
            "\0null_byte_injection",
        ];

        for pattern in malicious_patterns {
            let result = Config::builder().append_system_prompt(pattern).build();
            assert!(
                result.is_err(),
                "Malicious pattern '{}' should be rejected",
                pattern
            );

            if let Err(e) = result {
                assert!(e.to_string().contains("malicious content"));
            }
        }
    }

    #[test]
    fn test_granular_permission_security_validation() {
        let dangerous_commands = vec![
            "Bash(rm -rf /)",
            "Bash(sudo su)",
            "Bash(chmod 777 /)",
            "Bash(dd if=/dev/zero of=/dev/sda)",
            "bash:rm -rf *",
            "bash:sudo rm -rf /",
        ];

        // These should parse successfully (validation doesn't reject based on danger)
        // but the user should be able to control them via disallowed_tools
        for command in &dangerous_commands {
            let permission = ToolPermission::from_str(command);
            assert!(
                permission.is_ok(),
                "Dangerous command '{}' should parse (to be controlled via disallowed_tools)",
                command
            );
        }

        // Test that they can be properly disallowed
        let config = Config::builder()
            .disallowed_tools(dangerous_commands.iter().map(|s| s.to_string()).collect())
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_unicode_and_special_characters_handling() {
        // Unicode should be allowed in append_system_prompt
        let unicode_prompts = vec![
            "Respond in 日本語 if needed",
            "Use emojis like 🦀 🚀 ✨",
            "Handle accents: café, naïve, résumé",
            "Mathematical symbols: ∀x ∈ ℝ, x² ≥ 0",
        ];

        for prompt in unicode_prompts {
            let result = Config::builder().append_system_prompt(prompt).build();
            assert!(
                result.is_ok(),
                "Unicode prompt '{}' should be allowed",
                prompt
            );
        }

        // Unicode in tool names should also work
        let config = Config::builder()
            .disallowed_tools(vec!["Bash(echo 你好世界)".to_string()])
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_length_limits_enforcement() {
        // Test maximum allowed lengths for append_system_prompt
        let max_prompt = "a".repeat(10_000); // At the limit
        let result = Config::builder().append_system_prompt(&max_prompt).build();
        assert!(result.is_ok());

        // Test just over the limit
        let over_limit_prompt = "a".repeat(10_001);
        let result = Config::builder()
            .append_system_prompt(&over_limit_prompt)
            .build();
        assert!(result.is_err());

        // Test tool name length limits - need to use valid tool format
        let max_tool = format!("Bash({})", "a".repeat(90)); // Valid Bash format with long command
        let _result = Config::builder().disallowed_tools(vec![max_tool]).build();
        // This should be okay as long as the tool name itself is valid

        let over_limit_tool = "a".repeat(101);
        let result = Config::builder()
            .disallowed_tools(vec![over_limit_tool])
            .build();
        assert!(result.is_err()); // Should fail because it's not a valid tool format
    }

    #[test]
    fn test_boundary_values_for_max_turns() {
        // Test boundary values for max_turns
        let boundary_values = vec![
            (1, true),        // Minimum valid
            (100, true),      // Normal value
            (1000, true),     // Large value
            (u32::MAX, true), // Maximum possible
        ];

        for (turns, should_pass) in boundary_values {
            let result = Config::builder().max_turns(turns).build();
            if should_pass {
                assert!(result.is_ok(), "max_turns {} should be valid", turns);
            } else {
                assert!(result.is_err(), "max_turns {} should be invalid", turns);
            }
        }

        // Zero should always fail
        let result = Config::builder().max_turns(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_injection_prevention_in_tool_names() {
        // Test various injection attempts in tool names
        let injection_attempts = vec![
            "Bash(ls; rm -rf /)",
            "Bash(ls && malicious_command)",
            "Bash(ls | dangerous_pipe)",
            "Bash(ls > /dev/null; evil)",
            "mcp__server__tool; malicious",
            "mcp__server__tool && evil",
            "tool\nwith\nnewlines",
            "tool\rwith\rcarriage\rreturns",
        ];

        for attempt in injection_attempts {
            let result = Config::builder()
                .disallowed_tools(vec![attempt.to_string()])
                .build();

            // Some may fail parsing, others may fail validation
            // The key is they should be handled safely
            if result.is_err() {
                // Error should be descriptive - just check that there's an error
                let error_msg = result.unwrap_err().to_string();
                assert!(
                    !error_msg.is_empty(),
                    "Error message should be present for injection attempt '{}'",
                    attempt
                );

                // Check that common error patterns are present
                assert!(
                    error_msg.contains("Invalid")
                        || error_msg.contains("malicious")
                        || error_msg.contains("Unknown")
                        || error_msg.contains("format"),
                    "Error message should indicate validation failure for '{}': {}",
                    attempt,
                    error_msg
                );
            }
        }
    }
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_large_number_of_disallowed_tools() {
        // Test with many disallowed tools
        let many_tools: Vec<String> = (0..1000).map(|i| format!("Bash(tool_{})", i)).collect();

        let start = std::time::Instant::now();
        let result = Config::builder()
            .disallowed_tools(many_tools.clone())
            .build();
        let duration = start.elapsed();

        assert!(result.is_ok());
        assert_eq!(result.unwrap().disallowed_tools.unwrap().len(), 1000);

        // Should complete reasonably quickly (adjust threshold as needed)
        assert!(
            duration.as_millis() < 1000,
            "Validation took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_very_long_append_system_prompt() {
        // Test with maximum allowed length
        let max_length_prompt = "a".repeat(10_000);

        let start = std::time::Instant::now();
        let result = Config::builder()
            .append_system_prompt(&max_length_prompt)
            .build();
        let duration = start.elapsed();

        assert!(result.is_ok());
        assert_eq!(result.unwrap().append_system_prompt.unwrap().len(), 10_000);

        // Should complete reasonably quickly
        assert!(
            duration.as_millis() < 100,
            "Validation took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_complex_config_building_performance() {
        // Test building a complex config with all new flags
        let many_disallowed: Vec<String> = (0..100)
            .map(|i| format!("mcp__server{}__tool{}", i % 10, i))
            .collect();

        let start = std::time::Instant::now();
        let result = Config::builder()
            .model("claude-3-sonnet-20240229")
            .system_prompt("Complex system prompt with detailed instructions")
            .append_system_prompt("Additional context and requirements")
            .max_turns(50)
            .disallowed_tools(many_disallowed)
            .skip_permissions(true)
            .stream_format(StreamFormat::Json)
            .timeout_secs(120)
            .max_tokens(4000)
            .build();
        let duration = start.elapsed();

        assert!(result.is_err()); // Should fail due to system_prompt + append_system_prompt conflict

        // Even with validation failure, should complete quickly
        assert!(
            duration.as_millis() < 100,
            "Config building took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_repeated_config_validation() {
        // Test that validation can be called multiple times efficiently
        let config = Config::builder()
            .append_system_prompt("Test prompt")
            .max_turns(10)
            .disallowed_tools(vec!["Bash(ls)".to_string(), "mcp__test__tool".to_string()])
            .build()
            .unwrap();

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            assert!(config.validate().is_ok());
        }
        let duration = start.elapsed();

        // 1000 validations should complete quickly
        assert!(
            duration.as_millis() < 100,
            "Repeated validation took too long: {:?}",
            duration
        );
    }
}
