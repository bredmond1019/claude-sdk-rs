# Security Best Practices

This guide outlines security best practices for using the claude-sdk-rs SDK in production environments.

## Authentication and API Key Management

### Claude CLI Integration

The SDK relies on the Claude CLI for authentication. **No API keys are handled directly by the SDK** - all authentication flows through the official Claude CLI tool.

**✅ Secure Authentication Flow:**
```rust
// The SDK automatically uses Claude CLI authentication
let client = Client::new(Config::default());
// No API key needed - delegates to `claude auth` credentials
```

**Authentication Requirements:**
- Claude CLI must be installed and authenticated (`claude auth`)
- SDK validates CLI availability before operations
- No direct API key handling reduces attack surface

### CLI Security Validation

The SDK includes security checks for CLI integration:

```rust
// CLI binary detection and validation
use which::which;

if which("claude").is_err() {
    return Err(Error::BinaryNotFound);
}
```

### Security Architecture

- **No direct API access** - All requests go through authenticated CLI
- **Process isolation** - CLI runs in separate process space
- **Secure communication** - Uses stdout/stderr for data exchange
- **Timeout protection** - Prevents hanging processes
- **Resource limits** - Configurable timeouts and size limits

## Input Validation and Sanitization

### Comprehensive Input Validation

The SDK includes extensive input validation with **570+ security tests** covering:

```rust
// Built-in query validation
let result = validate_query(user_input);
match result {
    Ok(validated) => client.query(&validated).send().await?,
    Err(Error::InvalidInput(msg)) => return Err(msg),
}
```

**Validation Checks Include:**
- Length limits (queries limited to 100,000 chars)
- Control character filtering (null bytes, newlines)
- Injection pattern detection
- Unicode security issues
- Format string attack prevention

### Security Test Coverage

**Security Tests Include:**
- **SQL Injection:** 15+ patterns including union, blind, and time-based attacks
- **Command Injection:** Shell metacharacters and dangerous commands
- **Path Traversal:** Directory traversal attempts (`../../../etc/passwd`)
- **Script Injection:** XSS patterns, template injection, code execution
- **Format String Attacks:** Printf-style format vulnerabilities  
- **Unicode Attacks:** RTL override, zero-width characters, BOM injection
- **Null Byte Injection:** File extension bypass attempts
- **DoS via Large Inputs:** 1MB+ payloads, memory exhaustion

```rust
// Example security test pattern
let malicious_inputs = vec![
    "'; DROP TABLE users; --",
    "../../../etc/passwd", 
    "<script>alert('XSS')</script>",
    "${system('rm -rf /')}",
];

for input in malicious_inputs {
    let result = validate_query(input);
    assert!(result.is_err() || is_safely_handled(&result));
}
```

## Output Handling

### Validate Responses

Never trust AI output for critical operations without validation:

```rust
let response = client.query("Generate SQL query").send().await?;

// ❌ Bad: Direct execution
// db.execute(&response)?;

// ✅ Good: Validate first
if is_valid_sql(&response) && is_safe_query(&response) {
    db.execute(&response)?;
} else {
    return Err("Invalid or unsafe SQL generated");
}
```

### Sensitive Information

Be careful about sensitive information in responses:

```rust
// Filter sensitive data from logs
let response = client.query(query).send().await?;
let sanitized_response = remove_sensitive_data(&response);
log::info!("Response: {}", sanitized_response);
```

## Network Security

### Use HTTPS

The Claude CLI uses HTTPS by default. Ensure your environment doesn't downgrade connections:

```rust
// Verify SSL/TLS is enabled in your environment
// The SDK handles this automatically through the CLI
```

### Timeout Configuration

Set appropriate timeouts to prevent resource exhaustion:

```rust
let client = Client::builder()
    .timeout_secs(30)  // Reasonable timeout
    .build();
```

### Rate Limiting

Implement rate limiting to prevent abuse:

```rust
use tokio::time::{sleep, Duration};

// Simple rate limiting
static LAST_REQUEST: Mutex<Instant> = Mutex::new(Instant::now());

async fn rate_limited_query(client: &Client, query: &str) -> Result<String> {
    let mut last = LAST_REQUEST.lock().unwrap();
    let elapsed = last.elapsed();
    
    if elapsed < Duration::from_millis(100) {
        sleep(Duration::from_millis(100) - elapsed).await;
    }
    
    *last = Instant::now();
    drop(last);
    
    client.query(query).send().await
}
```

## Error Handling

### Don't Leak Sensitive Information

Be careful with error messages:

```rust
match client.query(query).send().await {
    Ok(response) => Ok(response),
    Err(e) => {
        // ❌ Bad: May leak sensitive information
        // return Err(format!("Claude error: {:?}", e));
        
        // ✅ Good: Generic error for users
        log::error!("Claude API error: {:?}", e);  // Log full error
        Err("Service temporarily unavailable")      // Generic user error
    }
}
```

### Fail Securely

Always fail to a secure state:

```rust
// Default to restrictive behavior
let mut allow_action = false;

match check_permissions().await {
    Ok(permitted) => allow_action = permitted,
    Err(_) => {
        // Fail closed - deny on error
        allow_action = false;
        log::error!("Permission check failed, denying access");
    }
}
```

## Session Management

### Session Security

When using sessions, implement proper security:

```rust
// Use strong session identifiers
use uuid::Uuid;
let session_id = Uuid::new_v4().to_string();

// Implement session expiration
const SESSION_TIMEOUT: Duration = Duration::from_secs(3600); // 1 hour

// Validate session ownership
fn validate_session(session_id: &str, user_id: &str) -> bool {
    // Check that session belongs to user
    session_owner(session_id) == user_id
}
```

### Clean Up Sessions

Always clean up sessions when done:

```rust
// Use Drop trait or explicit cleanup
struct SecureSession {
    id: String,
    client: Client,
}

impl Drop for SecureSession {
    fn drop(&mut self) {
        // Clean up session data
        log::info!("Cleaning up session {}", self.id);
        // Remove any temporary data
    }
}
```

## Data Protection

### Encryption at Rest

If storing conversation history or responses:

```rust
// Encrypt sensitive data before storage
use encryption_crate::encrypt;

let encrypted_response = encrypt(&response, &key)?;
storage.save(&session_id, &encrypted_response)?;
```

### Memory Security

Clear sensitive data from memory when done:

```rust
// Use zeroize for sensitive data
use zeroize::Zeroize;

let mut sensitive_data = get_sensitive_data();
// Use sensitive_data
sensitive_data.zeroize();  // Clear from memory
```

## Audit and Monitoring

### Log Security Events

Implement comprehensive logging:

```rust
// Log security-relevant events
log::info!("Claude query from user: {}", user_id);
log::warn!("Rate limit exceeded for user: {}", user_id);
log::error!("Authentication failed for user: {}", user_id);
```

### Monitor Usage

Track usage patterns:

```rust
#[derive(Debug)]
struct UsageMetrics {
    user_id: String,
    timestamp: chrono::DateTime<Utc>,
    tokens_used: usize,
    cost: f64,
    session_id: String,
}

// Store metrics for analysis
async fn track_usage(metrics: UsageMetrics) {
    // Store in database or metrics system
    metrics_store.record(metrics).await;
}
```

## Tool and MCP Security

### Tool Permission Validation

The SDK includes comprehensive tool security validation:

```rust
// Tool permission validation with security checks
let dangerous_tools = vec![
    "bash:rm -rf /",
    "bash:cat /etc/passwd", 
    "mcp__filesystem__read_file:/etc/shadow",
];

// Built-in validation prevents dangerous tool usage
for tool in dangerous_tools {
    let permission = ToolPermission::parse(tool);
    assert!(permission.is_safe()); // Validates against known dangerous patterns
}
```

**Tool Security Features:**
- **Command validation** - Blocks dangerous shell commands
- **Path restriction** - Prevents access to sensitive system files
- **Permission scoping** - Tools limited to intended functionality
- **Audit logging** - Tool usage is logged for security monitoring

### Sandbox Tool Execution

If possible, run tools in sandboxed environments:

```rust
// Example: Restrict filesystem access
let client = Client::builder()
    .allowed_tools(vec!["mcp__sandboxed_fs__read".to_string()])
    .build();
```

## Production Deployment

### Environment Isolation

- Run the Claude CLI in isolated environments
- Use containers or VMs for additional isolation
- Limit network access to required endpoints only

### Resource Limits

Set resource limits to prevent DoS:

```rust
// In your deployment configuration
// Limit memory, CPU, and concurrent connections
```

### Security Updates

- Keep the Claude CLI updated
- Regularly update the claude-sdk-rs SDK
- Monitor security advisories

## Compliance Considerations

### Data Privacy

- Understand what data is sent to Claude
- Implement data retention policies
- Comply with GDPR, CCPA, etc.

```rust
// Example: Data minimization
fn prepare_query(user_data: &UserData) -> String {
    // Only send necessary information
    format!("Analyze this anonymized data: {}", 
            anonymize(user_data))
}
```

### Audit Trail

Maintain audit trails for compliance:

```rust
#[derive(Debug, Serialize)]
struct AuditLog {
    timestamp: chrono::DateTime<Utc>,
    user_id: String,
    action: String,
    session_id: String,
    success: bool,
}

async fn audit_log(entry: AuditLog) {
    // Store in append-only audit log
    audit_store.append(entry).await;
}
```

## Security Test Suite

### Automated Security Validation

Run comprehensive security tests:

```bash
# Run all security tests
cargo test security

# Run penetration tests  
cargo test penetration

# Run specific security categories
cargo test injection_attack_tests
cargo test access_control_tests
cargo test cryptographic_security_tests
```

**Test Coverage:**
- **Core security tests:** 570+ tests across multiple categories
- **Penetration testing:** Real-world attack simulation
- **Property-based testing:** Fuzzing with arbitrary inputs
- **Memory safety:** Bounds checking and resource management

## Security Checklist

Before deploying to production, ensure:

- [ ] **Authentication:** Claude CLI properly configured (`claude auth`)
- [ ] **Input validation:** All user input goes through `validate_query()`
- [ ] **Tool permissions:** Restricted to necessary tools only
- [ ] **Error handling:** No sensitive information in error messages
- [ ] **Resource limits:** Timeouts and size limits configured
- [ ] **Security tests:** All security test suites pass
- [ ] **Dependencies:** Regular security audits (`cargo audit`)
- [ ] **Monitoring:** Security events logged and monitored
- [ ] **Updates:** SDK and CLI kept up to date
- [ ] **Environment:** Production isolation and access controls

## Security Validation Tools

### Built-in Security Utilities

```rust
use claude_sdk_rs::security::*;

// Validate input security
let is_safe = SecurityValidator::validate_input(user_input)?;

// Check tool permissions
let tool_check = ToolSecurity::validate_permissions(&tools)?;

// Audit configuration
let audit = SecurityAudit::check_config(&config)?;
```

### Security Metrics

- **Test Coverage:** 570+ security-focused tests
- **Validation Speed:** Input validation typically <1ms
- **Memory Safety:** Zero-copy validation where possible
- **Error Handling:** Structured error codes with security context

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** open a public issue
2. Email security concerns to the maintainers
3. Include:
   - Description of the vulnerability
   - Steps to reproduce (with test cases)
   - Potential impact assessment
   - Suggested fix (if any)
4. **Reference existing tests:** Check `tests/core/security_tests.rs` and `tests/core/penetration_tests.rs`

### Security Test Contributions

When adding new security tests:
- Follow the pattern in `tests/core/security_tests.rs`
- Include both positive and negative test cases
- Document attack vectors being tested
- Ensure tests are deterministic and fast

## Additional Resources

### Security Documentation
- [Security Test Inventory](../tests/core/security_tests.rs) - Complete security test suite
- [Penetration Tests](../tests/core/penetration_tests.rs) - Ethical hacking simulations
- [TESTING.md](TESTING.md) - Testing guidelines including security tests
- [Error Handling Guide](../src/core/error.rs) - Secure error management

### External Resources
- [OWASP AI Security Guidelines](https://owasp.org/www-project-ai-security/)
- [Anthropic Safety Best Practices](https://www.anthropic.com/safety)
- [Rust Security Guidelines](https://anssi-fr.github.io/rust-guide/)
- [Claude CLI Security](https://claude.ai/docs/cli/security)

### Continuous Security

```bash
# Regular security maintenance
cargo audit              # Check for vulnerable dependencies
cargo test security      # Run security test suite
cargo clippy -- -D warnings  # Security-focused linting
```

**Remember:** Security is a continuous process. The SDK includes extensive automated security testing, but you should regularly review and update your security practices as new threats emerge.