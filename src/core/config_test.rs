//! Configuration tests
//!
//! This module provides comprehensive tests for the Config structure,
//! including builder pattern, validation, defaults, and edge cases.

use crate::core::config::{validate_query, Config, StreamFormat};
use crate::core::error::Error;

/// Test configuration defaults
#[cfg(test)]
mod config_defaults_tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let config = Config::default();

        assert_eq!(config.stream_format, StreamFormat::Text);
        assert_eq!(config.timeout_secs, Some(30));
        assert_eq!(config.model, None);
        assert_eq!(config.system_prompt, None);
        assert_eq!(config.allowed_tools, None);
        assert!(!config.verbose);
        assert!(config.non_interactive);
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.mcp_config_path, None);
    }

    #[test]
    fn test_stream_format_default() {
        let format = StreamFormat::default();
        assert_eq!(format, StreamFormat::Text);
    }

    #[test]
    fn test_stream_format_variants() {
        // Test all StreamFormat variants exist and are properly named
        match StreamFormat::Text {
            StreamFormat::Text => {}
            StreamFormat::Json => {}
            StreamFormat::StreamJson => {}
        }

        match StreamFormat::Json {
            StreamFormat::Text => {}
            StreamFormat::Json => {}
            StreamFormat::StreamJson => {}
        }

        match StreamFormat::StreamJson {
            StreamFormat::Text => {}
            StreamFormat::Json => {}
            StreamFormat::StreamJson => {}
        }
    }
}

/// Test configuration builder pattern
#[cfg(test)]
mod config_builder_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_builder_basic_construction() {
        let config = Config::builder().build().unwrap();

        // Builder should produce default values when no customization
        assert_eq!(config.stream_format, StreamFormat::Text);
        assert_eq!(config.timeout_secs, Some(30));
        assert_eq!(config.model, None);
        assert_eq!(config.system_prompt, None);
    }

    #[test]
    fn test_builder_with_model() {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .build()
            .unwrap();

        assert_eq!(config.model, Some("claude-3-opus-20240229".to_string()));
        assert_eq!(config.stream_format, StreamFormat::Text); // Other fields default
    }

    #[test]
    fn test_builder_with_system_prompt() {
        let prompt = "You are a helpful assistant specialized in code analysis.";
        let config = Config::builder().system_prompt(prompt).build().unwrap();

        assert_eq!(config.system_prompt, Some(prompt.to_string()));
    }

    #[test]
    fn test_builder_with_stream_format() {
        let config = Config::builder()
            .stream_format(StreamFormat::Json)
            .build()
            .unwrap();

        assert_eq!(config.stream_format, StreamFormat::Json);
    }

    #[test]
    fn test_builder_with_timeout_secs() {
        let config = Config::builder().timeout_secs(120).build().unwrap();

        assert_eq!(config.timeout_secs, Some(120));
    }

    #[test]
    fn test_builder_with_tools() {
        let tools = vec!["mcp__server__search".to_string(), "bash".to_string()];
        let config = Config::builder()
            .allowed_tools(tools.clone())
            .build()
            .unwrap();

        assert_eq!(config.allowed_tools, Some(tools));
    }

    #[test]
    fn test_builder_with_max_tokens() {
        let config = Config::builder().max_tokens(1000).build().unwrap();

        assert_eq!(config.max_tokens, Some(1000));
    }

    #[test]
    fn test_builder_with_mcp_config() {
        let path = PathBuf::from("/path/to/mcp.json");
        let config = Config::builder().mcp_config(path.clone()).build().unwrap();

        assert_eq!(config.mcp_config_path, Some(path));
    }

    #[test]
    fn test_builder_chaining_all_options() {
        let tools = vec!["tool1".to_string(), "tool2".to_string()];
        let mcp_path = PathBuf::from("/test/mcp.json");

        let config = Config::builder()
            .model("claude-3-sonnet-20240229")
            .system_prompt("Be concise and accurate")
            .stream_format(StreamFormat::StreamJson)
            .timeout_secs(180)
            .allowed_tools(tools.clone())
            .max_tokens(2000)
            .mcp_config(mcp_path.clone())
            .verbose(true)
            .non_interactive(false)
            .build()
            .unwrap();

        assert_eq!(config.model, Some("claude-3-sonnet-20240229".to_string()));
        assert_eq!(
            config.system_prompt,
            Some("Be concise and accurate".to_string())
        );
        assert_eq!(config.stream_format, StreamFormat::StreamJson);
        assert_eq!(config.timeout_secs, Some(180));
        assert_eq!(config.allowed_tools, Some(tools));
        assert_eq!(config.max_tokens, Some(2000));
        assert_eq!(config.mcp_config_path, Some(mcp_path));
        assert!(config.verbose);
        assert!(!config.non_interactive);
    }

    #[test]
    fn test_builder_overwrite_values() {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .model("claude-3-sonnet-20240229") // Overwrite previous
            .timeout_secs(60)
            .timeout_secs(120) // Overwrite previous
            .build()
            .unwrap();

        assert_eq!(config.model, Some("claude-3-sonnet-20240229".to_string()));
        assert_eq!(config.timeout_secs, Some(120));
    }
}

/// Test configuration validation
#[cfg(test)]
mod config_validation_tests {
    use super::*;

    #[test]
    fn test_valid_claude_models() {
        let valid_models = vec![
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
        ];

        for model in valid_models {
            let config = Config::builder().model(model).build().unwrap();

            assert_eq!(config.model, Some(model.to_string()));
        }
    }

    #[test]
    fn test_timeout_validation() {
        let timeouts = vec![1, 30, 300, 3600];

        for timeout in timeouts {
            let config = Config::builder().timeout_secs(timeout).build().unwrap();

            assert_eq!(config.timeout_secs, Some(timeout));
        }
    }

    #[test]
    fn test_zero_timeout() {
        let result = Config::builder().timeout_secs(0).build();

        // Zero timeout should fail validation
        assert!(result.is_err());
    }

    #[test]
    fn test_very_large_timeout() {
        let large_timeout = 86400; // 24 hours
        let result = Config::builder().timeout_secs(large_timeout).build();

        // 24 hours exceeds our max of 3600 seconds (1 hour)
        assert!(result.is_err());
    }

    #[test]
    fn test_max_tokens_validation() {
        let token_limits = vec![1, 100, 1000, 100_000];

        for limit in token_limits {
            let config = Config::builder().max_tokens(limit).build().unwrap();

            assert_eq!(config.max_tokens, Some(limit));
        }
    }
}

/// Test system prompt validation and edge cases
#[cfg(test)]
mod system_prompt_tests {
    use super::*;

    #[test]
    fn test_empty_system_prompt() {
        let config = Config::builder().system_prompt("").build().unwrap();

        assert_eq!(config.system_prompt, Some("".to_string()));
    }

    #[test]
    fn test_multiline_system_prompt() {
        let prompt = "You are a helpful assistant.\nBe concise.\nAlways be accurate.";
        let config = Config::builder().system_prompt(prompt).build().unwrap();

        assert_eq!(config.system_prompt, Some(prompt.to_string()));
    }

    #[test]
    fn test_unicode_system_prompt() {
        let prompt = "You are a helpful assistant 🤖. Respond in English, français, or 日本語.";
        let config = Config::builder().system_prompt(prompt).build().unwrap();

        assert_eq!(config.system_prompt, Some(prompt.to_string()));
    }

    #[test]
    fn test_very_long_system_prompt() {
        let long_prompt = "a".repeat(10000);
        let config = Config::builder()
            .system_prompt(&long_prompt)
            .build()
            .unwrap();

        assert_eq!(config.system_prompt, Some(long_prompt));
    }

    #[test]
    fn test_system_prompt_with_special_characters() {
        let prompt = r#"You are an AI assistant. Use JSON format: {"response": "content"}. Handle "quotes" and 'apostrophes'."#;
        let config = Config::builder().system_prompt(prompt).build().unwrap();

        assert_eq!(config.system_prompt, Some(prompt.to_string()));
    }
}

/// Test tool configuration
#[cfg(test)]
mod tool_configuration_tests {
    use super::*;

    #[test]
    fn test_empty_tools_list() {
        let config = Config::builder().allowed_tools(vec![]).build().unwrap();

        assert_eq!(config.allowed_tools, Some(vec![]));
    }

    #[test]
    fn test_single_tool() {
        let tools = vec!["mcp__server__search".to_string()];
        let config = Config::builder()
            .allowed_tools(tools.clone())
            .build()
            .unwrap();

        assert_eq!(config.allowed_tools, Some(tools));
    }

    #[test]
    fn test_multiple_tools() {
        let tools = vec![
            "mcp__server__search".to_string(),
            "mcp__server__filesystem".to_string(),
            "bash".to_string(),
        ];
        let config = Config::builder()
            .allowed_tools(tools.clone())
            .build()
            .unwrap();

        assert_eq!(config.allowed_tools, Some(tools));
    }

    #[test]
    fn test_duplicate_tools() {
        let tools = vec![
            "bash".to_string(),
            "bash".to_string(), // Duplicate
            "mcp__server__search".to_string(),
        ];
        let config = Config::builder()
            .allowed_tools(tools.clone())
            .build()
            .unwrap();

        // Duplicates should be preserved (filtering can happen at runtime)
        assert_eq!(config.allowed_tools, Some(tools));
    }

    #[test]
    fn test_tool_name_patterns() {
        let tools = vec![
            "bash".to_string(),                    // Simple name
            "mcp__server__filesystem".to_string(), // MCP pattern
            "custom-tool-name".to_string(),        // Hyphenated
            "tool_with_underscores".to_string(),   // Underscored
            "numeric123".to_string(),              // With numbers (but not starting with)
        ];
        let config = Config::builder()
            .allowed_tools(tools.clone())
            .build()
            .unwrap();

        assert_eq!(config.allowed_tools, Some(tools));
    }
}

/// Test configuration cloning and serialization
#[cfg(test)]
mod config_cloning_tests {
    use super::*;

    #[test]
    fn test_config_clone() {
        let original = Config::builder()
            .model("claude-3-opus-20240229")
            .system_prompt("Test prompt")
            .stream_format(StreamFormat::Json)
            .timeout_secs(60)
            .allowed_tools(vec!["tool1".to_string()])
            .build()
            .unwrap();

        let cloned = original.clone();

        assert_eq!(original.model, cloned.model);
        assert_eq!(original.system_prompt, cloned.system_prompt);
        assert_eq!(original.stream_format, cloned.stream_format);
        assert_eq!(original.timeout_secs, cloned.timeout_secs);
        assert_eq!(original.allowed_tools, cloned.allowed_tools);
    }

    #[test]
    fn test_config_debug_format() {
        let config = Config::builder()
            .model("claude-3-sonnet-20240229")
            .build()
            .unwrap();

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("claude-3-sonnet-20240229"));
    }

    #[test]
    fn test_stream_format_debug() {
        assert_eq!(format!("{:?}", StreamFormat::Text), "Text");
        assert_eq!(format!("{:?}", StreamFormat::Json), "Json");
        assert_eq!(format!("{:?}", StreamFormat::StreamJson), "StreamJson");
    }

    #[test]
    fn test_stream_format_clone() {
        let original = StreamFormat::StreamJson;
        let cloned = original;
        assert_eq!(original, cloned);
    }
}

/// Test configuration edge cases and error conditions
#[cfg(test)]
mod config_edge_cases {
    use super::*;

    #[test]
    fn test_config_with_none_values() {
        let config = Config {
            model: None,
            system_prompt: None,
            stream_format: StreamFormat::Text,
            timeout_secs: None,
            allowed_tools: None,
            mcp_config_path: None,
            non_interactive: true,
            verbose: false,
            max_tokens: None,
        };

        assert_eq!(config.model, None);
        assert_eq!(config.system_prompt, None);
        assert_eq!(config.timeout_secs, None);
    }

    #[test]
    fn test_empty_string_values() {
        let result = Config::builder().model("").system_prompt("").build();

        // Empty strings are allowed - no validation on model/prompt content
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.model, Some("".to_string()));
        assert_eq!(config.system_prompt, Some("".to_string()));
    }

    #[test]
    fn test_whitespace_only_values() {
        let config = Config::builder()
            .model("   ")
            .system_prompt("\t\n  \r\n")
            .build()
            .unwrap();

        assert_eq!(config.model, Some("   ".to_string()));
        assert_eq!(config.system_prompt, Some("\t\n  \r\n".to_string()));
    }

    #[test]
    fn test_boolean_flags() {
        let config1 = Config::builder()
            .verbose(true)
            .non_interactive(false)
            .build()
            .unwrap();

        assert!(config1.verbose);
        assert!(!config1.non_interactive);

        let config2 = Config::builder()
            .verbose(false)
            .non_interactive(true)
            .build()
            .unwrap();

        assert!(!config2.verbose);
        assert!(config2.non_interactive);
    }
}

/// Property-based tests for Config validation
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_config_builder_with_arbitrary_strings(
            model in any::<Option<String>>(),
            system_prompt in any::<Option<String>>(),
        ) {
            let mut builder = Config::builder();

            if let Some(m) = model {
                builder = builder.model(m);
            }

            if let Some(sp) = system_prompt {
                builder = builder.system_prompt(sp);
            }

            // Try to build - might fail due to validation
            let _ = builder.build();
        }

        #[test]
        fn test_config_builder_with_arbitrary_numbers(
            timeout in 1u64..=3600,
            max_tokens in 1usize..=200_000,
        ) {
            let result = Config::builder()
                .timeout_secs(timeout)
                .max_tokens(max_tokens)
                .build();

            // Valid ranges should always succeed
            assert!(result.is_ok());
            let config = result.unwrap();
            assert_eq!(config.timeout_secs, Some(timeout));
            assert_eq!(config.max_tokens, Some(max_tokens));
        }

        #[test]
        fn test_config_builder_with_arbitrary_tools(
            tools in prop::collection::vec(any::<String>(), 0..10),
        ) {
            // Note: arbitrary tools might fail validation
            let _ = Config::builder()
                .allowed_tools(tools.clone())
                .build();
        }

        #[test]
        fn test_config_clone_consistency(
            timeout in 1u64..=3600,
            verbose in any::<bool>(),
            non_interactive in any::<bool>(),
        ) {
            let result = Config::builder()
                .timeout_secs(timeout)
                .verbose(verbose)
                .non_interactive(non_interactive)
                .build();

            if let Ok(config) = result {
                let cloned = config.clone();

                assert_eq!(config.timeout_secs, cloned.timeout_secs);
                assert_eq!(config.verbose, cloned.verbose);
                assert_eq!(config.non_interactive, cloned.non_interactive);
                assert_eq!(config.stream_format, cloned.stream_format);
            }
        }

        #[test]
        fn test_config_builder_idempotence(
            model in any::<String>(),
            timeout in 1u64..=3600,
        ) {
            // Setting the same value multiple times should result in the last value
            let result = Config::builder()
                .model(&model)
                .model(&model)
                .timeout_secs(timeout)
                .timeout_secs(timeout)
                .build();

            if let Ok(config) = result {
                assert_eq!(config.model, Some(model));
                assert_eq!(config.timeout_secs, Some(timeout));
            }
        }
    }
}

/// Additional validation tests for Config edge cases
#[cfg(test)]
mod config_validation_edge_cases {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_config_with_invalid_timeout_zero() {
        // Zero timeout should fail validation
        let result = Config::builder().timeout_secs(0).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_negative_max_tokens_workaround() {
        // Since max_tokens is usize, we can't have negative values
        // Zero should fail validation
        let result = Config::builder().max_tokens(0).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_invalid_mcp_path() {
        // Config should accept any path, validation happens at runtime
        let invalid_path = PathBuf::from("/this/path/does/not/exist/mcp.json");
        let config = Config::builder()
            .mcp_config(invalid_path.clone())
            .build()
            .unwrap();

        assert_eq!(config.mcp_config_path, Some(invalid_path));
    }

    #[test]
    fn test_config_with_conflicting_stream_formats() {
        // Last format wins when setting multiple times
        let config = Config::builder()
            .stream_format(StreamFormat::Text)
            .stream_format(StreamFormat::Json)
            .stream_format(StreamFormat::StreamJson)
            .build()
            .unwrap();

        assert_eq!(config.stream_format, StreamFormat::StreamJson);
    }

    #[test]
    fn test_config_with_invalid_tool_names() {
        // Test that invalid tool names fail validation
        let invalid_tools = vec![
            vec!["".to_string()],                 // Empty
            vec![" ".to_string()],                // Whitespace
            vec!["tool with spaces".to_string()], // Spaces
            vec!["tool@#$%".to_string()],         // Special chars
            vec!["🔧tool".to_string()],           // Unicode
        ];

        for tools in invalid_tools {
            let result = Config::builder().allowed_tools(tools).build();

            assert!(result.is_err());
        }
    }

    #[test]
    fn test_config_max_values() {
        // Test maximum reasonable values - should fail validation
        let result1 = Config::builder().timeout_secs(u64::MAX).build();

        assert!(result1.is_err());

        let result2 = Config::builder().max_tokens(usize::MAX).build();

        assert!(result2.is_err());
    }

    #[test]
    fn test_config_model_name_edge_cases() {
        let long_model = format!("claude-3-opus-20240229{}", "x".repeat(1000));
        let edge_case_models = vec![
            "",            // Empty
            " ",           // Whitespace
            "\n\t",        // Special whitespace
            &long_model,   // Very long
            "模型名称",    // Unicode
            "model\0name", // Null byte
        ];

        for model_name in edge_case_models {
            let result = Config::builder().model(model_name).build();

            // Model names are not validated, so all should succeed
            assert!(result.is_ok());
        }
    }
}

/// Comprehensive validation tests
#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .system_prompt("You are a helpful assistant")
            .timeout_secs(60)
            .max_tokens(1000)
            .allowed_tools(vec!["bash".to_string(), "mcp__server__tool".to_string()])
            .build()
            .unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_system_prompt_too_long() {
        let long_prompt = "a".repeat(10_001); // 1 over limit
        let result = Config::builder().system_prompt(long_prompt).build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, Error::InvalidInput(_)));
            assert!(e
                .to_string()
                .contains("System prompt exceeds maximum length"));
        }
    }

    #[test]
    fn test_system_prompt_malicious_content() {
        let malicious_prompts = vec![
            "Execute this: <script>alert('xss')</script>",
            "Run command: $(rm -rf /)",
            "Inject SQL: '; DROP TABLE users;--",
            "Path traversal: ../../etc/passwd",
        ];

        for prompt in malicious_prompts {
            let result = Config::builder().system_prompt(prompt).build();

            assert!(result.is_err());
            if let Err(e) = result {
                assert!(matches!(e, Error::InvalidInput(_)));
                assert!(e.to_string().contains("malicious content"));
            }
        }
    }

    #[test]
    fn test_timeout_validation() {
        // Too small
        let result = Config::builder().timeout_secs(0).build();
        assert!(result.is_err());

        // Too large
        let result = Config::builder().timeout_secs(3601).build();
        assert!(result.is_err());

        // Valid range
        for timeout in [1, 30, 60, 3600] {
            let result = Config::builder().timeout_secs(timeout).build();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_max_tokens_validation() {
        // Zero tokens
        let result = Config::builder().max_tokens(0).build();
        assert!(result.is_err());

        // Too many tokens
        let result = Config::builder().max_tokens(200_001).build();
        assert!(result.is_err());

        // Valid range
        for tokens in [1, 100, 1000, 100_000, 200_000] {
            let result = Config::builder().max_tokens(tokens).build();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_tool_name_validation() {
        // Invalid tool names
        let invalid_tools = vec![
            vec!["".to_string()],  // Empty name
            vec!["a".repeat(101)], // Too long
            vec!["tool with spaces".to_string()],
            vec!["tool@#$".to_string()],
            vec!["tool;command".to_string()],
        ];

        for tools in invalid_tools {
            let result = Config::builder().allowed_tools(tools).build();
            assert!(result.is_err());
        }

        // Valid tool names
        let valid_tools = vec![
            vec!["bash".to_string()],
            vec!["mcp__server__tool".to_string()],
            vec!["tool-name".to_string()],
            vec!["tool_name".to_string()],
            vec!["tool:command".to_string()],
        ];

        for tools in valid_tools {
            let result = Config::builder().allowed_tools(tools).build();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_empty_mcp_path_validation() {
        let result = Config::builder()
            .mcp_config(std::path::PathBuf::from(""))
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, Error::InvalidInput(_)));
            assert!(e.to_string().contains("MCP config path cannot be empty"));
        }
    }

    #[test]
    fn test_query_validation() {
        // Empty query
        assert!(validate_query("").is_err());

        // Valid queries
        assert!(validate_query("Hello, Claude!").is_ok());
        assert!(validate_query("What is 2 + 2?").is_ok());

        // Very long query
        let long_query = "a".repeat(100_000);
        assert!(validate_query(&long_query).is_ok());

        // Too long query
        let too_long_query = "a".repeat(100_001);
        assert!(validate_query(&too_long_query).is_err());

        // Malicious queries
        let malicious_queries = vec![
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "'; DROP TABLE users;--",
            "../../etc/passwd",
        ];

        for query in malicious_queries {
            let result = validate_query(query);
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(matches!(e, Error::InvalidInput(_)));
                assert!(e.to_string().contains("malicious content"));
            }
        }
    }

    #[test]
    fn test_edge_case_validation() {
        // All validation at once
        let config = Config::builder()
            .system_prompt("Safe prompt")
            .timeout_secs(30)
            .max_tokens(1000)
            .allowed_tools(vec!["bash".to_string()])
            .mcp_config(std::path::PathBuf::from("/path/to/config"))
            .build()
            .unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_error_messages() {
        // Check that error messages are informative
        let result = Config::builder().system_prompt("a".repeat(10_001)).build();

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("10000")); // Shows limit
            assert!(error_msg.contains("10001")); // Shows actual
        }

        let result = Config::builder().timeout_secs(5000).build();

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("3600")); // Shows max
            assert!(error_msg.contains("5000")); // Shows actual
        }
    }
}
