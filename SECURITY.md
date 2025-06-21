# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

1. **DO NOT** open a public issue
2. Email security concerns to: security@example.com (placeholder)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: 1-2 weeks
  - High: 2-4 weeks
  - Medium: 1-2 months
  - Low: Next release

## Known Security Issues

### v1.0.0 (2025-06-17) - RESOLVED

**All critical vulnerabilities have been resolved as of 2025-06-17:**

✅ **RUSTSEC-2024-0421**: `idna 0.1.5` - **FIXED**
   - **Resolution**: Removed unused `jsonrpc-core` and `jsonrpc-core-client` dependencies
   - **Impact**: Eliminated the entire dependency chain causing the vulnerability
   - **Status**: No longer present in dependency tree

✅ **RUSTSEC-2024-0437**: `protobuf 2.28.0` - **FIXED**
   - **Resolution**: Updated `prometheus` dependency from `0.13` to `0.14`
   - **Impact**: Uses secure protobuf version (>=3.7.2) through updated prometheus
   - **Status**: No longer vulnerable

### Current Status

🟢 **0 critical vulnerabilities** (verified via `cargo audit`)
🟡 **1 warning**: `dotenv 0.15.0` (unmaintained, non-critical)

**Note**: The remaining dotenv warning is for development/testing only and poses no security risk to production deployments.

## Security Practices

### Dependencies

- Regular `cargo audit` runs in CI
- Dependabot enabled for security updates
- Conservative dependency updates
- Prefer well-maintained crates

### Code Review

- All changes require PR review
- Security-sensitive changes need 2 reviewers
- Automated security scanning in CI

### Safe Defaults

- Timeouts on all operations
- Input validation and sanitization
- No arbitrary code execution
- Careful subprocess handling

## Security Features

### Built-in Protections

1. **Process Isolation**: Claude CLI runs in subprocess
2. **Input Sanitization**: All user input is validated
3. **No Direct Execution**: No `eval` or dynamic code
4. **Timeout Protection**: Prevents resource exhaustion
5. **Error Sanitization**: No sensitive data in errors

### Best Practices for Users

1. **API Key Security**:
   ```bash
   # Use environment variables
   export CLAUDE_API_KEY="your-key"
   
   # Never commit keys to version control
   echo "CLAUDE_API_KEY" >> .gitignore
   ```

2. **Tool Permissions**:
   ```rust
   // Be explicit about allowed tools
   let client = Client::builder()
       .allowed_tools(vec!["mcp__filesystem__read"])
       .build();
   ```

3. **Session Security**:
   - Don't share session IDs
   - Clear sessions with sensitive data
   - Use unique sessions per user

## Audit Schedule

- **Monthly**: Dependency audit
- **Quarterly**: Full security review
- **Annually**: Third-party audit (planned)

## Compliance

While we strive for security best practices, this project:
- Is not certified for any compliance standards
- Should not be used for regulated data without assessment
- Provides no warranty or liability coverage

## Security Changelog

### 2025-06-17 - Critical Vulnerability Fixes

**Security Improvements:**
- ✅ Fixed RUSTSEC-2024-0421 (idna vulnerability) by removing unused jsonrpc dependencies
- ✅ Fixed RUSTSEC-2024-0437 (protobuf vulnerability) by updating prometheus to 0.14
- ✅ Fixed integration test compilation errors with updated Error enum variants
- ✅ Verified 0 critical vulnerabilities remain via cargo audit
- 🔧 Updated dependency management to prevent similar issues

**Technical Details:**
- Removed: `jsonrpc-core = "18.0"` and `jsonrpc-core-client = "18.0"` (unused)
- Updated: `prometheus = "0.13"` → `prometheus = "0.14"`
- Fixed: Integration tests now use correct Error enum variants
- Verified: All tests pass with security updates

**Impact:** Production deployments are now secure against these known vulnerabilities.

## Updates

This policy may be updated as the project evolves. Check back regularly and watch for security advisories in releases.

---

*Last updated: 2025-06-17*