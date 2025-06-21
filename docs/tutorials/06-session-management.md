# Part 6: Session Management

Session management in the claude-sdk-rs SDK enables persistent conversation state, metadata tracking, and context preservation across multiple interactions. This tutorial covers all aspects of working with sessions, from basic usage to advanced storage backends.

## Session Management Overview

Sessions in claude-sdk-rs provide:

- **Persistent conversation state**: Maintain sessions across application restarts
- **Metadata tracking**: Store additional information with each session  
- **Multiple storage backends**: Memory, file system, and SQLite options
- **Builder pattern**: Fluent API for creating configured sessions
- **Session lifecycle**: Create, retrieve, update, and delete sessions

```rust
use claude_sdk_rs::{SessionManager, SessionBuilder, SessionId, StorageBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a session manager with file storage
    let manager = SessionManager::with_storage(StorageBackend::File("./sessions".into()));
    
    // Create a new session with metadata
    let session = manager.create_session()
        .with_system_prompt("You are a helpful Rust programming assistant")
        .with_metadata("project", serde_json::json!("my-rust-app"))
        .build()
        .await?;
    
    println!("Created session: {}", session.id());
    
    Ok(())
}
```

## Core Session Concepts

### SessionId and Session Types

```rust
use claude_sdk_rs::{SessionId, Session, SessionBuilder};
use serde_json::json;

// Session IDs are string-based identifiers
let session_id = SessionId::new("my-session-001");

// Sessions contain metadata and system prompts
let session = SessionBuilder::new()
    .with_system_prompt("You are a code review assistant")
    .with_metadata("language", json!("rust"))
    .with_metadata("project", json!("web-api"))
    .build()
    .await?;

println!("Session ID: {}", session.id());
println!("Created at: {}", session.created_at);
```

### SessionManager Initialization

The `SessionManager` handles all session operations and provides different storage backends:

```rust
use claude_sdk_rs::{SessionManager, StorageBackend};
use std::path::PathBuf;

// In-memory storage (default, non-persistent)
let memory_manager = SessionManager::new();

// File-based storage (persistent across restarts)
let file_manager = SessionManager::with_storage(
    StorageBackend::File(PathBuf::from("./my-sessions"))
);

// SQLite storage (requires "sqlite" feature)
#[cfg(feature = "sqlite")]
let sqlite_manager = SessionManager::with_storage_async(
    StorageBackend::Sqlite(PathBuf::from("sessions.db"))
).await?;
```

## Basic Session Operations

### Creating and Managing Sessions

```rust
use claude_sdk_rs::{SessionManager, StorageBackend};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = SessionManager::with_storage(
        StorageBackend::File("./sessions".into())
    );
    
    // Create a new session
    let session = manager.create_session()
        .with_system_prompt("You are a helpful geography teacher")
        .with_metadata("subject", json!("geography"))
        .with_metadata("level", json!("beginner"))
        .build()
        .await?;
    
    println!("Created session: {}", session.id());
    
    // List all session IDs
    let session_ids = manager.list().await?;
    println!("Total sessions: {}", session_ids.len());
    
    // Get a specific session
    let retrieved = manager.get(session.id()).await?;
    if let Some(s) = retrieved {
        println!("Found session with prompt: {:?}", s.system_prompt);
    }
    
    // Resume an existing session (errors if not found)
    let resumed = manager.resume(session.id()).await?;
    println!("Resumed session: {}", resumed.id());
    
    Ok(())
}
```

### Session Builder Patterns

```rust
use claude_sdk_rs::{SessionBuilder, SessionManager};
use serde_json::json;

async fn session_builder_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Standalone session (not managed)
    let standalone = SessionBuilder::new()
        .with_system_prompt("You are a code reviewer")
        .with_metadata("type", json!("code-review"))
        .build()
        .await?;
    
    // Session with custom ID
    let custom_id_session = SessionBuilder::with_id("review-2024-001")
        .with_system_prompt("You are reviewing authentication code")
        .with_metadata("module", json!("auth"))
        .build()
        .await?;
    
    // Managed session (automatically stored)
    let manager = SessionManager::new();
    let managed = manager.create_session()
        .with_system_prompt("You are a data analyst")
        .with_metadata("role", json!("analyst"))
        .build()
        .await?;
    
    println!("Standalone: {}", standalone.id());
    println!("Custom ID: {}", custom_id_session.id());
    println!("Managed: {}", managed.id());
    
    Ok(())
}
```

## Session-Based Conversations

### Using Sessions with Client

```rust
use claude_sdk_rs::{Client, Config, SessionManager, SessionId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = SessionManager::new();
    let client = Client::new(Config::default());
    
    // Create a session for this conversation
    let session = manager.create_session()
        .with_system_prompt("You are a Rust programming tutor")
        .build()
        .await?;
    
    // Use the session ID in queries
    let response = client
        .query("Explain ownership in Rust")
        .session(session.id().clone())
        .send()
        .await?;
    
    println!("Response: {}", response);
    
    // Continue the conversation in the same session
    let follow_up = client
        .query("Can you give me a practical example?")
        .session(session.id().clone())
        .send()
        .await?;
    
    println!("Follow-up: {}", follow_up);
    
    Ok(())
}
```

### Session Context Management

```rust
use claude_sdk_rs::{Client, Config, SessionManager, StorageBackend};
use serde_json::json;

async fn contextual_conversation() -> Result<(), Box<dyn std::error::Error>> {
    let manager = SessionManager::with_storage(
        StorageBackend::File("./conversations".into())
    );
    let client = Client::new(Config::default());
    
    // Create a session with specific context
    let session = manager.create_session()
        .with_system_prompt("You are reviewing authentication code for security issues")
        .with_metadata("review_type", json!("security"))
        .with_metadata("module", json!("authentication"))
        .build()
        .await?;
    
    let questions = vec![
        "Review this authentication function for security issues",
        "What are the main vulnerabilities you found?",
        "How would you fix the password hashing?",
        "Are there any other security best practices I should follow?",
    ];
    
    for question in questions {
        let response = client
            .query(question)
            .session(session.id().clone())
            .send()
            .await?;
        
        println!("Q: {}", question);
        println!("A: {}\n", response);
    }
    
    Ok(())
}
```

## Storage Backends

### File-Based Storage

```rust
use claude_sdk_rs::{SessionManager, StorageBackend};
use std::path::PathBuf;

fn setup_file_storage() -> SessionManager {
    let storage_path = PathBuf::from("./project-sessions");
    SessionManager::with_storage(StorageBackend::File(storage_path))
}

async fn file_storage_example() -> Result<(), Box<dyn std::error::Error>> {
    let manager = setup_file_storage();
    
    // Create and persist session
    let session = manager.create_session()
        .with_system_prompt("You are a project assistant")
        .build()
        .await?;
    
    println!("Session persisted to: ./project-sessions/{}.json", session.id());
    
    // Session will be available after application restart
    let retrieved = manager.get(session.id()).await?;
    println!("Session exists: {}", retrieved.is_some());
    
    Ok(())
}
```

### SQLite Storage (with feature flag)

```rust
#[cfg(feature = "sqlite")]
use claude_sdk_rs::{SessionManager, StorageBackend};

#[cfg(feature = "sqlite")]
async fn sqlite_storage_example() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    
    // SQLite requires async initialization
    let manager = SessionManager::with_storage_async(
        StorageBackend::Sqlite(PathBuf::from("sessions.db"))
    ).await?;
    
    // Create session (stored in SQLite database)
    let session = manager.create_session()
        .with_system_prompt("You are a database consultant")
        .build()
        .await?;
    
    println!("Session stored in SQLite: {}", session.id());
    
    // Query sessions from database
    let all_sessions = manager.list().await?;
    println!("Total sessions in database: {}", all_sessions.len());
    
    Ok(())
}
```

## Session Lifecycle Management

### Session Cleanup and Management

```rust
use claude_sdk_rs::{SessionManager, SessionId, StorageBackend};

struct SessionLifecycleManager {
    manager: SessionManager,
}

impl SessionLifecycleManager {
    fn new() -> Self {
        Self {
            manager: SessionManager::with_storage(
                StorageBackend::File("./sessions".into())
            ),
        }
    }
    
    async fn cleanup_all_sessions(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let session_ids = self.manager.list().await?;
        let count = session_ids.len();
        
        self.manager.clear().await?;
        
        println!("Cleaned up {} sessions", count);
        Ok(count)
    }
    
    async fn delete_specific_session(&self, id: &SessionId) -> Result<(), Box<dyn std::error::Error>> {
        self.manager.delete(id).await?;
        println!("Deleted session: {}", id);
        Ok(())
    }
    
    async fn session_exists(&self, id: &SessionId) -> Result<bool, Box<dyn std::error::Error>> {
        let session = self.manager.get(id).await?;
        Ok(session.is_some())
    }
}

async fn lifecycle_example() -> Result<(), Box<dyn std::error::Error>> {
    let lifecycle = SessionLifecycleManager::new();
    
    // Create some test sessions
    let session1 = lifecycle.manager.create_session()
        .with_system_prompt("Test session 1")
        .build()
        .await?;
        
    let session2 = lifecycle.manager.create_session()
        .with_system_prompt("Test session 2")  
        .build()
        .await?;
    
    // Check existence
    println!("Session 1 exists: {}", lifecycle.session_exists(session1.id()).await?);
    
    // Delete specific session
    lifecycle.delete_specific_session(session1.id()).await?;
    
    // Cleanup all remaining sessions
    lifecycle.cleanup_all_sessions().await?;
    
    Ok(())
}
```

## Advanced Session Patterns

### Session Pool Management

```rust
use claude_sdk_rs::{SessionManager, SessionId, Client, Config};
use std::collections::HashMap;
use serde_json::json;

struct SessionPool {
    manager: SessionManager,
    client: Client,
    sessions: HashMap<String, SessionId>,
}

impl SessionPool {
    fn new() -> Self {
        Self {
            manager: SessionManager::new(),
            client: Client::new(Config::default()),
            sessions: HashMap::new(),
        }
    }
    
    async fn get_or_create_session(&mut self, context: &str) -> claude_sdk_rs::Result<SessionId> {
        if let Some(session_id) = self.sessions.get(context) {
            // Verify session still exists
            if self.manager.get(session_id).await?.is_some() {
                return Ok(session_id.clone());
            }
        }
        
        // Create new session for this context
        let session = self.manager.create_session()
            .with_system_prompt(&format!("You are working in the {} context", context))
            .with_metadata("context", json!(context))
            .build()
            .await?;
        
        let session_id = session.id().clone();
        self.sessions.insert(context.to_string(), session_id.clone());
        Ok(session_id)
    }
    
    async fn query_in_context(&mut self, context: &str, query: &str) -> claude_sdk_rs::Result<String> {
        let session_id = self.get_or_create_session(context).await?;
        
        let response = self.client
            .query(query)
            .session(session_id)
            .send()
            .await?;
        
        Ok(response)
    }
}

async fn session_pool_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut pool = SessionPool::new();
    
    // Different contexts maintain separate conversations
    let response1 = pool.query_in_context("rust-learning", "Explain ownership").await?;
    let response2 = pool.query_in_context("project-planning", "Help me plan a web API").await?;
    let response3 = pool.query_in_context("rust-learning", "Give me an example of borrowing").await?;
    
    println!("Rust learning: {}", response1);
    println!("Project planning: {}", response2);  
    println!("Rust follow-up: {}", response3);
    
    Ok(())
}
```

### Session Templates

```rust
use claude_sdk_rs::{SessionManager, SessionBuilder};
use serde_json::json;

struct SessionTemplate {
    name_prefix: String,
    system_prompt_template: String,
    default_metadata: serde_json::Value,
}

impl SessionTemplate {
    fn new(name_prefix: &str, system_prompt: &str) -> Self {
        Self {
            name_prefix: name_prefix.to_string(),
            system_prompt_template: system_prompt.to_string(),
            default_metadata: json!({}),
        }
    }
    
    fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.default_metadata = metadata;
        self
    }
    
    async fn create_session(&self, manager: &SessionManager, suffix: &str) -> claude_sdk_rs::Result<claude_sdk_rs::Session> {
        let id = format!("{}_{}", self.name_prefix, suffix);
        let prompt = self.system_prompt_template.replace("{suffix}", suffix);
        
        let mut builder = manager.create_session()
            .with_system_prompt(prompt);
        
        // Add default metadata
        if let serde_json::Value::Object(map) = &self.default_metadata {
            for (key, value) in map {
                builder = builder.with_metadata(key, value.clone());
            }
        }
        
        builder = builder.with_metadata("template", json!(self.name_prefix));
        builder = builder.with_metadata("suffix", json!(suffix));
        
        builder.build().await
    }
}

async fn template_example() -> Result<(), Box<dyn std::error::Error>> {
    let manager = SessionManager::new();
    
    let code_review_template = SessionTemplate::new(
        "code_review",
        "You are reviewing {suffix} code for best practices and security"
    ).with_metadata(json!({
        "type": "code-review",
        "automated": false
    }));
    
    let learning_template = SessionTemplate::new(
        "learning",
        "You are teaching about {suffix} to a beginner"
    ).with_metadata(json!({
        "type": "education",
        "level": "beginner"
    }));
    
    // Create sessions from templates
    let review_session = code_review_template
        .create_session(&manager, "authentication")
        .await?;
    
    let learning_session = learning_template
        .create_session(&manager, "async_rust")
        .await?;
    
    println!("Created review session: {}", review_session.id());
    println!("Created learning session: {}", learning_session.id());
    
    Ok(())
}
```

## Error Handling for Session Operations

### Comprehensive Error Handling

```rust
use claude_sdk_rs::{SessionManager, SessionId, Error};

async fn robust_session_operations() -> Result<(), Box<dyn std::error::Error>> {
    let manager = SessionManager::new();
    
    // Handle session creation
    let session = manager.create_session()
        .with_system_prompt("Test session")
        .build()
        .await?;
    
    // Handle session retrieval
    let fake_id = SessionId::new("non-existent-session");
    match manager.get(&fake_id).await {
        Ok(Some(session)) => println!("Found session: {}", session.id()),
        Ok(None) => println!("Session not found: {}", fake_id),
        Err(e) => println!("Error getting session: {}", e),
    }
    
    // Handle session resumption (errors if not found)
    match manager.resume(&fake_id).await {
        Ok(session) => println!("Resumed session: {}", session.id()),
        Err(Error::SessionNotFound(id)) => {
            println!("Cannot resume - session not found: {}", id);
            // Create new session instead
            let new_session = manager.create_session()
                .with_system_prompt("Replacement session")
                .build()
                .await?;
            println!("Created replacement session: {}", new_session.id());
        }
        Err(e) => println!("Other error: {}", e),
    }
    
    // Handle storage errors gracefully
    match manager.list().await {
        Ok(session_ids) => {
            println!("Found {} sessions", session_ids.len());
            
            // Try to delete each session
            for id in session_ids {
                if let Err(e) = manager.delete(&id).await {
                    println!("Failed to delete session {}: {}", id, e);
                }
            }
        }
        Err(e) => {
            println!("Failed to list sessions: {}", e);
        }
    }
    
    Ok(())
}
```

## Best Practices and Performance

### Session Management Best Practices

1. **Use appropriate storage backends** for your use case:
   - Memory: For temporary sessions and testing
   - File: For persistent sessions with simple requirements  
   - SQLite: For advanced querying and concurrent access

2. **Set up regular cleanup** to prevent storage bloat:
   ```rust
   async fn cleanup_old_sessions(manager: &SessionManager) -> Result<(), Box<dyn std::error::Error>> {
       let session_ids = manager.list().await?;
       println!("Starting cleanup of {} sessions", session_ids.len());
       
       // In a real implementation, you'd filter by age/usage
       // For now, just demonstrate the cleanup mechanism
       if session_ids.len() > 100 {
           manager.clear().await?;
           println!("Cleaned up all sessions due to high count");
       }
       
       Ok(())
   }
   ```

3. **Use descriptive metadata** for session organization:
   ```rust
   let session = manager.create_session()
       .with_system_prompt("You are a code reviewer")
       .with_metadata("project", json!("web-api"))
       .with_metadata("language", json!("rust"))
       .with_metadata("created_by", json!("alice"))
       .with_metadata("purpose", json!("security-review"))
       .build()
       .await?;
   ```

### Performance Optimization

```rust
use claude_sdk_rs::SessionManager;
use std::time::Instant;

async fn optimized_session_operations() -> Result<(), Box<dyn std::error::Error>> {
    let manager = SessionManager::new();
    
    // Batch session creation
    let start = Instant::now();
    let mut sessions = Vec::new();
    
    for i in 0..10 {
        let session = manager.create_session()
            .with_system_prompt(&format!("Session {}", i))
            .build()
            .await?;
        sessions.push(session);
    }
    
    println!("Created {} sessions in {:?}", sessions.len(), start.elapsed());
    
    // Batch operations when possible
    let session_ids = manager.list().await?;
    println!("Retrieved {} session IDs in one operation", session_ids.len());
    
    // Use session IDs for efficient lookups
    for id in &session_ids[..5.min(session_ids.len())] {
        if let Some(session) = manager.get(id).await? {
            println!("Session {}: {:?}", id, session.system_prompt);
        }
    }
    
    Ok(())
}
```

## Next Steps

Now that you understand session management, explore:

- **Part 7**: [Advanced Usage](07-advanced-usage.md) - Complex SDK usage patterns
- **Production considerations** - Deploying claude-sdk-rs applications with session management

## Session Management Checklist

- ✅ **Choose appropriate storage backend** for your persistence needs
- ✅ **Create sessions with descriptive metadata** for organization
- ✅ **Implement proper error handling** for all session operations
- ✅ **Use session builders** for complex session configuration
- ✅ **Set up cleanup strategies** for long-running applications  
- ✅ **Use session pools** for multi-context applications

Session management provides the foundation for building stateful, context-aware Claude AI applications!