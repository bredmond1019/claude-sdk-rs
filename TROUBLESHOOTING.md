# Troubleshooting Guide for claude-sdk-rs SDK

This comprehensive guide helps you diagnose and resolve common issues when using the claude-sdk-rs SDK. Each section includes symptoms, causes, and step-by-step solutions.

## Table of Contents

1. [Installation and Setup Issues](#installation-and-setup-issues)
2. [Authentication Problems](#authentication-problems)
3. [API Call Failures](#api-call-failures)
4. [Response Parsing Errors](#response-parsing-errors)
5. [Timeout Issues](#timeout-issues)
6. [Session Management Problems](#session-management-problems)
7. [Streaming Response Issues](#streaming-response-issues)
8. [Performance Problems](#performance-problems)
9. [Tool Integration Issues](#tool-integration-issues)
10. [Platform-Specific Issues](#platform-specific-issues)
11. [Debugging Techniques](#debugging-techniques)
12. [Getting Help](#getting-help)

## Installation and Setup Issues

### Issue: "claude-sdk-rs crate not found"

**Symptoms:**
- `cargo build` fails with "package 'claude-sdk-rs' not found"
- Compilation errors about missing dependencies

**Causes:**
- Incorrect crate name in Cargo.toml
- Network issues preventing crate download
- Version conflicts

**Solutions:**

1. **Verify crate name and version:**
   ```toml
   [dependencies]
   claude-sdk-rs = "1.0.0"  # Use exact version
   tokio = { version = "1.40", features = ["full"] }
   ```

2. **Update Cargo index:**
   ```bash
   cargo update
   cargo clean
   cargo build
   ```

3. **Check network connectivity:**
   ```bash
   # Test crates.io access
   curl -I https://crates.io
   
   # Check for proxy settings
   echo $https_proxy
   ```

4. **Use alternative registry if needed:**
   ```toml
   [dependencies]
   claude-sdk-rs = { version = "1.0.0", registry = "crates-io" }
   ```

### Issue: "Claude CLI not found"

**Symptoms:**
- Runtime error: `ClaudeNotFound`
- "Binary not found" when creating client

**Causes:**
- Claude CLI not installed
- Claude CLI not in PATH
- Wrong CLI binary name

**Solutions:**

1. **Install Claude CLI:**
   ```bash
   # macOS/Linux
   curl -sSL https://claude.ai/install.sh | sh
   
   # Windows (PowerShell)
   Invoke-RestMethod -Uri https://claude.ai/install.ps1 | Invoke-Expression
   ```

2. **Verify installation:**
   ```bash
   which claude
   claude --version
   ```

3. **Add to PATH if needed:**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export PATH="$PATH:/path/to/claude/cli"
   
   # Reload shell
   source ~/.bashrc
   ```

4. **Set custom binary path:**
   ```rust
   use claude_ai::{Client, Config};
   
   let config = Config::builder()
       .claude_binary_path("/custom/path/to/claude")
       .build()?;
   let client = Client::new(config);
   ```

### Issue: "Permission denied" errors

**Symptoms:**
- "Permission denied" when running Claude CLI
- SDK fails with permission errors

**Causes:**
- Claude CLI lacks execute permissions
- Insufficient file system permissions

**Solutions:**

1. **Make CLI executable:**
   ```bash
   chmod +x /path/to/claude
   ```

2. **Check file permissions:**
   ```bash
   ls -la $(which claude)
   ```

3. **Run with proper permissions:**
   ```bash
   sudo chmod 755 /usr/local/bin/claude
   ```

## Authentication Problems

### Issue: "Not authenticated" errors

**Symptoms:**
- `ClaudeNotAuthenticated` error
- "Please login to Claude" messages

**Causes:**
- Claude CLI not logged in
- Authentication token expired
- Wrong authentication method

**Solutions:**

1. **Login to Claude CLI:**
   ```bash
   claude auth login
   # Follow prompts to enter API key
   ```

2. **Check authentication status:**
   ```bash
   claude auth status
   ```

3. **Re-authenticate if needed:**
   ```bash
   claude auth logout
   claude auth login
   ```

4. **Set API key environment variable:**
   ```bash
   export CLAUDE_API_KEY="your-api-key-here"
   ```

### Issue: "Invalid API key" errors

**Symptoms:**
- Authentication fails with "invalid key"
- 401 Unauthorized responses

**Causes:**
- Incorrect API key
- API key format issues
- Key permissions problems

**Solutions:**

1. **Verify API key format:**
   - Should start with `sk-`
   - Check for extra spaces or characters

2. **Get new API key:**
   - Visit Anthropic console
   - Generate new API key
   - Update authentication

3. **Test key manually:**
   ```bash
   curl -H "Authorization: Bearer YOUR_API_KEY" \
        https://api.anthropic.com/v1/messages
   ```

### Issue: "Rate limit exceeded"

**Symptoms:**
- 429 Too Many Requests errors
- "Rate limit exceeded" messages

**Causes:**
- Too many requests in short time
- Account limits reached
- Burst request patterns

**Solutions:**

1. **Implement retry logic:**
   ```rust
   use tokio::time::{sleep, Duration};
   
   async fn request_with_retry(client: &Client, prompt: &str) -> claude_ai::Result<String> {
       for attempt in 1..=3 {
           match client.query(prompt).send().await {
               Ok(response) => return Ok(response),
               Err(claude_ai::Error::RateLimit) => {
                   if attempt < 3 {
                       sleep(Duration::from_secs(2_u64.pow(attempt))).await;
                       continue;
                   }
               }
               Err(e) => return Err(e),
           }
       }
       Err(claude_ai::Error::RateLimit)
   }
   ```

2. **Add delays between requests:**
   ```rust
   use tokio::time::{sleep, Duration};
   
   for query in queries {
       let response = client.query(&query).send().await?;
       sleep(Duration::from_millis(500)).await; // 500ms delay
   }
   ```

3. **Check account limits:**
   - Review Anthropic console
   - Upgrade plan if needed
   - Monitor usage patterns

## API Call Failures

### Issue: "Connection refused" errors

**Symptoms:**
- Network connection errors
- "Connection refused" messages
- Timeouts on API calls

**Causes:**
- Internet connectivity issues
- Firewall blocking connections
- Proxy configuration problems

**Solutions:**

1. **Test connectivity:**
   ```bash
   ping api.anthropic.com
   curl -I https://api.anthropic.com
   ```

2. **Check firewall settings:**
   ```bash
   # Allow HTTPS traffic
   sudo ufw allow out 443
   ```

3. **Configure proxy if needed:**
   ```bash
   export https_proxy=http://proxy.company.com:8080
   export http_proxy=http://proxy.company.com:8080
   ```

4. **Use custom HTTP client:**
   ```rust
   use claude_ai::{Client, Config};
   
   let config = Config::builder()
       .timeout(Duration::from_secs(60))
       .build()?;
   ```

### Issue: "Invalid request format" errors

**Symptoms:**
- 400 Bad Request errors
- "Invalid request" messages
- Malformed request errors

**Causes:**
- Invalid query parameters
- Incorrect request structure
- Missing required fields

**Solutions:**

1. **Validate query content:**
   ```rust
   // Ensure query is not empty
   if query_text.trim().is_empty() {
       return Err(claude_ai::Error::Custom("Query cannot be empty".to_string()));
   }
   
   // Check for valid UTF-8
   if !query_text.is_char_boundary(0) {
       return Err(claude_ai::Error::Custom("Invalid UTF-8 in query".to_string()));
   }
   ```

2. **Use builder pattern correctly:**
   ```rust
   let client = Client::builder()
       .model("claude-sonnet-4-20250514")  // Valid model name
       .timeout(Duration::from_secs(30))   // Reasonable timeout
       .build();
   ```

3. **Enable debug logging:**
   ```rust
   std::env::set_var("RUST_LOG", "claude_ai=debug");
   env_logger::init();
   ```

### Issue: "Model not available" errors

**Symptoms:**
- "Model not found" errors
- "Invalid model" messages
- Service unavailable errors

**Causes:**
- Incorrect model name
- Model temporarily unavailable
- Account permissions

**Solutions:**

1. **Use correct model names:**
   ```rust
   // Valid model names (as of 2025)
   let valid_models = vec![
       "claude-opus-4-20250514",
       "claude-sonnet-4-20250514", 
       "claude-haiku-3-20250307",
   ];
   
   let client = Client::builder()
       .model("claude-sonnet-4-20250514")
       .build();
   ```

2. **Fallback to default model:**
   ```rust
   async fn create_client_with_fallback() -> Client {
       let preferred_models = vec![
           "claude-opus-4-20250514",
           "claude-sonnet-4-20250514",
           "claude-haiku-3-20250307",
       ];
       
       for model in preferred_models {
           let client = Client::builder().model(model).build();
           // Test with a simple query
           match client.query("test").send().await {
               Ok(_) => return client,
               Err(_) => continue,
           }
       }
       
       // Fallback to default
       Client::new(Config::default())
   }
   ```

## Response Parsing Errors

### Issue: "JSON parsing failed" errors

**Symptoms:**
- `SerializationError` when using JSON format
- "Failed to parse response" messages
- Malformed JSON errors

**Causes:**
- Claude returned non-JSON response
- Partial JSON responses
- Invalid JSON structure

**Solutions:**

1. **Use text format for debugging:**
   ```rust
   let client = Client::builder()
       .stream_format(StreamFormat::Text)  // Use text instead of JSON
       .build();
   
   let response = client.query("test").send().await?;
   println!("Raw response: {}", response);
   ```

2. **Implement robust JSON parsing:**
   ```rust
   use serde_json::Value;
   
   fn parse_json_safely(text: &str) -> Result<Value, String> {
       // Try to find JSON in response
       if let Some(start) = text.find('{') {
           if let Some(end) = text.rfind('}') {
               let json_part = &text[start..=end];
               return serde_json::from_str(json_part)
                   .map_err(|e| format!("JSON parse error: {}", e));
           }
       }
       Err("No JSON found in response".to_string())
   }
   ```

3. **Handle streaming JSON carefully:**
   ```rust
   use futures::StreamExt;
   
   let mut stream = client.query("test").stream().await?;
   let mut json_buffer = String::new();
   
   while let Some(result) = stream.next().await {
       match result {
           Ok(message) => {
               json_buffer.push_str(&message.content);
               // Try to parse complete JSON objects
               while let Some(end) = json_buffer.find('\n') {
                   let line = &json_buffer[..end];
                   if let Ok(parsed) = serde_json::from_str::<Value>(line) {
                       // Process parsed JSON
                   }
                   json_buffer = json_buffer[end + 1..].to_string();
               }
           }
           Err(e) => eprintln!("Stream error: {}", e),
       }
   }
   ```

### Issue: "Incomplete response" errors

**Symptoms:**
- Responses cut off mid-sentence
- Missing data in structured responses
- Partial JSON objects

**Causes:**
- Timeout during response generation
- Token limits reached
- Network interruptions

**Solutions:**

1. **Increase timeout:**
   ```rust
   let client = Client::builder()
       .timeout(Duration::from_secs(120))  // 2 minutes
       .build();
   ```

2. **Use streaming for long responses:**
   ```rust
   let mut stream = client.query("long request").stream().await?;
   let mut complete_response = String::new();
   
   while let Some(result) = stream.next().await {
       match result {
           Ok(message) => complete_response.push_str(&message.content),
           Err(e) => {
               eprintln!("Stream interrupted: {}", e);
               break;
           }
       }
   }
   ```

3. **Split large requests:**
   ```rust
   async fn process_large_request(client: &Client, large_prompt: &str) -> claude_ai::Result<String> {
       let chunks = split_prompt(large_prompt, 4000); // Split into smaller chunks
       let mut results = Vec::new();
       
       for chunk in chunks {
           let response = client.query(&chunk).send().await?;
           results.push(response);
       }
       
       Ok(results.join("\n"))
   }
   ```

## Timeout Issues

### Issue: Frequent timeout errors

**Symptoms:**
- `Timeout` errors on most requests
- Operations taking too long
- Inconsistent response times

**Causes:**
- Timeout set too low
- Complex queries requiring more time
- Network latency issues

**Solutions:**

1. **Adjust timeout based on query complexity:**
   ```rust
   fn get_timeout_for_query(query: &str) -> Duration {
       let word_count = query.split_whitespace().count();
       match word_count {
           0..=100 => Duration::from_secs(30),
           101..=500 => Duration::from_secs(60),
           501..=1000 => Duration::from_secs(120),
           _ => Duration::from_secs(180),
       }
   }
   
   let timeout = get_timeout_for_query(&my_query);
   let client = Client::builder().timeout(timeout).build();
   ```

2. **Use different timeouts for different operations:**
   ```rust
   struct AdaptiveClient {
       quick_client: Client,
       standard_client: Client,
       long_client: Client,
   }
   
   impl AdaptiveClient {
       fn new() -> Self {
           Self {
               quick_client: Client::builder().timeout(Duration::from_secs(15)).build(),
               standard_client: Client::builder().timeout(Duration::from_secs(60)).build(),
               long_client: Client::builder().timeout(Duration::from_secs(300)).build(),
           }
       }
       
       async fn query_adaptive(&self, query: &str) -> claude_ai::Result<String> {
           // Try quick first
           if let Ok(response) = self.quick_client.query(query).send().await {
               return Ok(response);
           }
           
           // Fall back to standard
           if let Ok(response) = self.standard_client.query(query).send().await {
               return Ok(response);
           }
           
           // Finally try long timeout
           self.long_client.query(query).send().await
       }
   }
   ```

3. **Implement timeout with retry:**
   ```rust
   async fn query_with_progressive_timeout(
       client: &Client,
       query: &str
   ) -> claude_ai::Result<String> {
       let timeouts = vec![30, 60, 120]; // Progressive timeouts
       
       for (attempt, timeout_secs) in timeouts.iter().enumerate() {
           let timeout_client = Client::builder()
               .timeout(Duration::from_secs(*timeout_secs))
               .build();
           
           match timeout_client.query(query).send().await {
               Ok(response) => return Ok(response),
               Err(claude_ai::Error::Timeout) if attempt < timeouts.len() - 1 => {
                   println!("Timeout at {}s, retrying with longer timeout...", timeout_secs);
                   continue;
               }
               Err(e) => return Err(e),
           }
       }
       
       Err(claude_ai::Error::Timeout)
   }
   ```

## Session Management Problems

### Issue: "Session not found" errors

**Symptoms:**
- Session-based queries fail
- "Invalid session ID" errors
- Lost conversation context

**Causes:**
- Session ID not properly maintained
- Session expired or cleared
- Multiple session ID formats

**Solutions:**

1. **Proper session ID management:**
   ```rust
   use claude_ai_core::session::SessionId;
   
   struct ConversationManager {
       session_id: SessionId,
       client: Client,
   }
   
   impl ConversationManager {
       fn new() -> Self {
           Self {
               session_id: SessionId::new(),
               client: Client::new(Config::default()),
           }
       }
       
       async fn send_message(&self, message: &str) -> claude_ai::Result<String> {
           self.client
               .query(message)
               .session_id(&self.session_id)
               .send()
               .await
       }
   }
   ```

2. **Session persistence:**
   ```rust
   use std::fs;
   use serde::{Serialize, Deserialize};
   
   #[derive(Serialize, Deserialize)]
   struct SessionState {
       id: String,
       created_at: chrono::DateTime<chrono::Utc>,
       last_used: chrono::DateTime<chrono::Utc>,
   }
   
   impl SessionState {
       fn save(&self, path: &str) -> std::io::Result<()> {
           let json = serde_json::to_string_pretty(self)?;
           fs::write(path, json)?;
           Ok(())
       }
       
       fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
           let json = fs::read_to_string(path)?;
           let state: SessionState = serde_json::from_str(&json)?;
           Ok(state)
       }
   }
   ```

### Issue: Context lost between messages

**Symptoms:**
- Claude doesn't remember previous messages
- Conversation state resets unexpectedly
- Inconsistent responses

**Causes:**
- Session ID not consistently used
- Session data not persisted
- CLI session management issues

**Solutions:**

1. **Consistent session usage:**
   ```rust
   struct PersistentConversation {
       session_id: SessionId,
       client: Client,
       message_history: Vec<String>,
   }
   
   impl PersistentConversation {
       async fn send_with_context(&mut self, message: &str) -> claude_ai::Result<String> {
           let response = self.client
               .query(message)
               .session_id(&self.session_id)
               .send()
               .await?;
           
           // Track message history
           self.message_history.push(format!("User: {}", message));
           self.message_history.push(format!("Claude: {}", response));
           
           Ok(response)
       }
       
       fn get_conversation_history(&self) -> String {
           self.message_history.join("\n")
       }
   }
   ```

## Streaming Response Issues

### Issue: Stream disconnects or fails

**Symptoms:**
- Streaming stops mid-response
- "Stream ended unexpectedly" errors
- Incomplete streaming responses

**Causes:**
- Network interruptions
- Server-side stream termination
- Client-side connection issues

**Solutions:**

1. **Robust stream handling:**
   ```rust
   use futures::StreamExt;
   use tokio::time::{timeout, Duration};
   
   async fn handle_stream_with_recovery(
       client: &Client,
       query: &str
   ) -> claude_ai::Result<String> {
       let mut stream = client.query(query).stream().await?;
       let mut complete_response = String::new();
       let mut last_chunk_time = std::time::Instant::now();
       
       while let Some(result) = timeout(Duration::from_secs(30), stream.next()).await {
           match result {
               Ok(Some(Ok(message))) => {
                   complete_response.push_str(&message.content);
                   last_chunk_time = std::time::Instant::now();
               }
               Ok(Some(Err(e))) => {
                   eprintln!("Stream error: {}", e);
                   // Try to continue with partial response
                   break;
               }
               Ok(None) => break, // Stream ended normally
               Err(_) => {
                   // Timeout waiting for next chunk
                   if last_chunk_time.elapsed() > Duration::from_secs(60) {
                       eprintln!("Stream timeout, using partial response");
                       break;
                   }
               }
           }
       }
       
       Ok(complete_response)
   }
   ```

2. **Stream reconnection:**
   ```rust
   async fn stream_with_reconnect(
       client: &Client,
       query: &str,
       max_retries: u32
   ) -> claude_ai::Result<String> {
       for attempt in 0..max_retries {
           match client.query(query).stream().await {
               Ok(mut stream) => {
                   let mut response = String::new();
                   let mut success = true;
                   
                   while let Some(result) = stream.next().await {
                       match result {
                           Ok(message) => response.push_str(&message.content),
                           Err(e) => {
                               eprintln!("Stream error on attempt {}: {}", attempt + 1, e);
                               success = false;
                               break;
                           }
                       }
                   }
                   
                   if success {
                       return Ok(response);
                   }
               }
               Err(e) => eprintln!("Failed to start stream on attempt {}: {}", attempt + 1, e),
           }
           
           if attempt < max_retries - 1 {
               tokio::time::sleep(Duration::from_secs(2_u64.pow(attempt))).await;
           }
       }
       
       Err(claude_ai::Error::Custom("Max stream retries exceeded".to_string()))
   }
   ```

## Performance Problems

### Issue: Slow response times

**Symptoms:**
- Responses taking unusually long
- High latency on simple queries
- Inconsistent performance

**Causes:**
- Inefficient query patterns
- Network latency
- Suboptimal configuration

**Solutions:**

1. **Optimize query patterns:**
   ```rust
   // Bad: Multiple small queries
   async fn inefficient_queries(client: &Client, items: &[String]) -> claude_ai::Result<Vec<String>> {
       let mut results = Vec::new();
       for item in items {
           let result = client.query(&format!("Process: {}", item)).send().await?;
           results.push(result);
       }
       Ok(results)
   }
   
   // Good: Batch processing
   async fn efficient_batch_query(client: &Client, items: &[String]) -> claude_ai::Result<String> {
       let batch_query = format!(
           "Process these items in batch:\n{}",
           items.iter().enumerate()
               .map(|(i, item)| format!("{}. {}", i + 1, item))
               .collect::<Vec<_>>()
               .join("\n")
       );
       
       client.query(&batch_query).send().await
   }
   ```

2. **Use appropriate models for tasks:**
   ```rust
   struct ModelSelector;
   
   impl ModelSelector {
       fn select_model(task_type: &str, complexity: &str) -> &'static str {
           match (task_type, complexity) {
               ("simple", _) => "claude-haiku-3-20250307",      // Fast, cheap
               ("analysis", "low") => "claude-sonnet-4-20250514",  // Balanced
               ("creative", _) => "claude-opus-4-20250514",       // Most capable
               (_, "high") => "claude-opus-4-20250514",          // Complex tasks
               _ => "claude-sonnet-4-20250514",                   // Default
           }
       }
   }
   
   let model = ModelSelector::select_model("simple", "low");
   let client = Client::builder().model(model).build();
   ```

3. **Implement connection pooling:**
   ```rust
   use std::sync::Arc;
   use tokio::sync::Semaphore;
   
   struct ClientPool {
       clients: Vec<Client>,
       semaphore: Arc<Semaphore>,
   }
   
   impl ClientPool {
       fn new(pool_size: usize) -> Self {
           let clients = (0..pool_size)
               .map(|_| Client::new(Config::default()))
               .collect();
           
           Self {
               clients,
               semaphore: Arc::new(Semaphore::new(pool_size)),
           }
       }
       
       async fn execute<F, R>(&self, f: F) -> claude_ai::Result<R>
       where
           F: FnOnce(&Client) -> futures::future::BoxFuture<claude_ai::Result<R>>,
       {
           let _permit = self.semaphore.acquire().await.unwrap();
           let client = &self.clients[0]; // In practice, use round-robin
           f(client).await
       }
   }
   ```

### Issue: Memory usage growing over time

**Symptoms:**
- Increasing memory consumption
- Performance degradation over time
- Out of memory errors

**Causes:**
- Memory leaks in session management
- Accumulating response data
- Inefficient data structures

**Solutions:**

1. **Proper resource cleanup:**
   ```rust
   struct ManagedSession {
       client: Client,
       session_id: SessionId,
       message_count: usize,
       max_messages: usize,
   }
   
   impl ManagedSession {
       async fn send_message(&mut self, message: &str) -> claude_ai::Result<String> {
           let response = self.client
               .query(message)
               .session_id(&self.session_id)
               .send()
               .await?;
           
           self.message_count += 1;
           
           // Reset session if too many messages
           if self.message_count >= self.max_messages {
               self.session_id = SessionId::new();
               self.message_count = 0;
           }
           
           Ok(response)
       }
   }
   ```

2. **Memory monitoring:**
   ```rust
   use std::sync::atomic::{AtomicUsize, Ordering};
   
   static MEMORY_USAGE: AtomicUsize = AtomicUsize::new(0);
   
   struct MemoryAwareClient {
       client: Client,
       max_memory_mb: usize,
   }
   
   impl MemoryAwareClient {
       async fn query_with_memory_check(&self, query: &str) -> claude_ai::Result<String> {
           let current_memory = MEMORY_USAGE.load(Ordering::Relaxed);
           
           if current_memory > self.max_memory_mb * 1024 * 1024 {
               return Err(claude_ai::Error::Custom("Memory limit exceeded".to_string()));
           }
           
           let response = self.client.query(query).send().await?;
           
           // Update memory usage (simplified)
           MEMORY_USAGE.fetch_add(response.len(), Ordering::Relaxed);
           
           Ok(response)
       }
   }
   ```

## Tool Integration Issues

### Issue: MCP server connection failures

**Symptoms:**
- "MCP server not found" errors
- Tool execution failures
- Connection timeouts to MCP servers

**Causes:**
- MCP server not running
- Incorrect server configuration
- Network connectivity issues

**Solutions:**

1. **Verify MCP server status:**
   ```bash
   # Check if MCP server is running
   ps aux | grep mcp-server
   
   # Test MCP server directly
   curl -X POST http://localhost:3000/mcp/ping
   ```

2. **Robust MCP client:**
   ```rust
   use claude_ai::{Client, Config};
   
   async fn create_mcp_client_with_fallback() -> claude_ai::Result<Client> {
       let mcp_servers = vec![
           "http://localhost:3000",
           "http://localhost:3001",
           "http://backup-server:3000",
       ];
       
       for server in mcp_servers {
           let config = Config::builder()
               .mcp_server_url(server)
               .build()?;
           
           let client = Client::new(config);
           
           // Test connection
           match client.query("test mcp connection").send().await {
               Ok(_) => return Ok(client),
               Err(e) => eprintln!("MCP server {} failed: {}", server, e),
           }
       }
       
       // Fallback to no MCP
       Ok(Client::new(Config::default()))
   }
   ```

### Issue: Tool permission errors

**Symptoms:**
- "Tool not allowed" errors
- Permission denied for specific tools
- Restricted tool access

**Causes:**
- Tool not in allowed list
- Incorrect tool configuration
- Security restrictions

**Solutions:**

1. **Configure tool permissions:**
   ```rust
   let client = Client::builder()
       .allowed_tools(vec![
           "bash:ls".to_string(),
           "bash:cat".to_string(),
           "mcp__filesystem__read".to_string(),
           "mcp__web__search".to_string(),
       ])
       .build();
   ```

2. **Dynamic tool permission checking:**
   ```rust
   async fn execute_tool_safely(
       client: &Client,
       tool_name: &str,
       query: &str
   ) -> claude_ai::Result<String> {
       let safe_tools = vec![
           "bash:ls", "bash:pwd", "bash:echo",
           "mcp__filesystem__read",
           "mcp__web__search",
       ];
       
       if !safe_tools.contains(&tool_name) {
           return Err(claude_ai::Error::Custom(
               format!("Tool '{}' not in safe list", tool_name)
           ));
       }
       
       client.query(query).send().await
   }
   ```

## Platform-Specific Issues

### macOS Issues

**Issue: "Code signing" or "Gatekeeper" warnings**

**Solutions:**
```bash
# Allow unsigned Claude CLI
sudo spctl --master-disable

# Or sign the binary yourself
codesign -s - $(which claude)

# Check quarantine status
xattr -l $(which claude)
xattr -d com.apple.quarantine $(which claude)
```

### Windows Issues

**Issue: Path or execution problems**

**Solutions:**
```powershell
# Check PATH
$env:PATH -split ';' | Select-String claude

# Add to PATH permanently
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\path\to\claude", "User")

# Check execution policy
Get-ExecutionPolicy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux Issues

**Issue: Missing dependencies**

**Solutions:**
```bash
# Install common dependencies
sudo apt-get update
sudo apt-get install curl ca-certificates

# For older systems
sudo apt-get install libssl-dev

# Check shared library dependencies
ldd $(which claude)
```

## Debugging Techniques

### Enable Debug Logging

```rust
// Set environment variable
std::env::set_var("RUST_LOG", "claude_ai=debug");

// Initialize logger
env_logger::init();

// Or use specific components
std::env::set_var("RUST_LOG", "claude_ai::client=debug,claude_ai::process=trace");
```

### Request/Response Logging

```rust
use claude_ai::{Client, Config};

struct DebuggingClient {
    inner: Client,
    log_requests: bool,
    log_responses: bool,
}

impl DebuggingClient {
    fn new(log_requests: bool, log_responses: bool) -> Self {
        Self {
            inner: Client::new(Config::default()),
            log_requests,
            log_responses,
        }
    }
    
    async fn query_debug(&self, query: &str) -> claude_ai::Result<String> {
        if self.log_requests {
            println!("=== REQUEST ===\n{}\n", query);
        }
        
        let response = self.inner.query(query).send().await?;
        
        if self.log_responses {
            println!("=== RESPONSE ===\n{}\n", response);
        }
        
        Ok(response)
    }
}
```

### Performance Profiling

```rust
use std::time::Instant;

async fn profile_request(client: &Client, query: &str) -> claude_ai::Result<String> {
    let start = Instant::now();
    
    let response = client.query(query).send().await?;
    
    let duration = start.elapsed();
    println!("Query took: {:?}", duration);
    println!("Response length: {} chars", response.len());
    println!("Characters per second: {:.2}", response.len() as f64 / duration.as_secs_f64());
    
    Ok(response)
}
```

### Network Diagnostics

```rust
use std::process::Command;

fn diagnose_network() -> std::io::Result<()> {
    println!("=== Network Diagnostics ===");
    
    // Check DNS resolution
    let output = Command::new("nslookup")
        .arg("api.anthropic.com")
        .output()?;
    println!("DNS lookup: {}", String::from_utf8_lossy(&output.stdout));
    
    // Check connectivity
    let output = Command::new("ping")
        .args(&["-c", "3", "api.anthropic.com"])
        .output()?;
    println!("Ping test: {}", String::from_utf8_lossy(&output.stdout));
    
    // Check HTTP connectivity
    let output = Command::new("curl")
        .args(&["-I", "https://api.anthropic.com"])
        .output()?;
    println!("HTTP test: {}", String::from_utf8_lossy(&output.stdout));
    
    Ok(())
}
```

## Getting Help

### Collect Diagnostic Information

Before seeking help, collect this information:

```rust
use claude_ai::{Client, Config};

async fn collect_diagnostics() {
    println!("=== claude-sdk-rs SDK Diagnostics ===");
    println!("SDK Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Rust Version: {}", env!("RUSTC_VERSION"));
    println!("OS: {}", std::env::consts::OS);
    println!("Architecture: {}", std::env::consts::ARCH);
    
    // Check Claude CLI
    match std::process::Command::new("claude").arg("--version").output() {
        Ok(output) => {
            println!("Claude CLI: {}", String::from_utf8_lossy(&output.stdout));
        }
        Err(e) => println!("Claude CLI Error: {}", e),
    }
    
    // Check authentication
    match std::process::Command::new("claude").args(&["auth", "status"]).output() {
        Ok(output) => {
            println!("Auth Status: {}", String::from_utf8_lossy(&output.stdout));
        }
        Err(e) => println!("Auth Check Error: {}", e),
    }
    
    // Test basic functionality
    let client = Client::new(Config::default());
    match client.query("Hello").send().await {
        Ok(_) => println!("Basic functionality: ✅ Working"),
        Err(e) => println!("Basic functionality: ❌ Failed: {}", e),
    }
}
```

### Create Minimal Reproduction

```rust
// Create a minimal example that reproduces your issue
use claude_ai::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reproduce the issue with minimal code
    let client = Client::new(Config::default());
    
    match client.query("Your problematic query here").send().await {
        Ok(response) => println!("Success: {}", response),
        Err(e) => {
            eprintln!("Error: {:?}", e);
            eprintln!("Error type: {}", std::any::type_name_of_val(&e));
        }
    }
    
    Ok(())
}
```

### Where to Get Help

1. **GitHub Issues**: https://github.com/anthropics/claude-sdk-rs-sdk/issues
2. **Documentation**: Check the latest API documentation
3. **Community**: Join the Claude AI community forums
4. **Support**: Contact Anthropic support for account-related issues

### What to Include in Bug Reports

- SDK version and Rust version
- Operating system and architecture
- Complete error messages and stack traces
- Minimal reproduction code
- Expected vs actual behavior
- Steps to reproduce
- Any relevant configuration or environment details

Remember: Good bug reports help maintainers fix issues faster!