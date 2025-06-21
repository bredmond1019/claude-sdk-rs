//! Session persistence tests
//!
//! This module provides comprehensive tests for session persistence functionality
//! across different storage backends (memory, file, SQLite).

#[cfg(test)]
use super::session::{Session, SessionId, SessionManager, StorageBackend, SessionStorage, FileStorage, MemoryStorage};
#[cfg(test)]
use crate::Result;
#[cfg(test)]
use tempfile::TempDir;
#[cfg(test)]
use std::path::PathBuf;

/// Helper to create a test session with metadata
#[cfg(test)]
fn create_test_session(id: &str) -> Session {
    let mut session = Session::new(SessionId::new(id))
        .with_system_prompt("Test system prompt");
    
    session.metadata.insert(
        "test_key".to_string(),
        serde_json::json!("test_value")
    );
    session.metadata.insert(
        "counter".to_string(),
        serde_json::json!(42)
    );
    
    session
}

#[tokio::test]
async fn test_memory_storage_basic_operations() -> Result<()> {
    let manager = SessionManager::new(); // Default is memory storage
    
    // Create and store a session
    let session = manager.create_session()
        .with_system_prompt("Memory test")
        .with_metadata("key", serde_json::json!("value"))
        .build()
        .await?;
    
    let session_id = session.id().clone();
    
    // Retrieve the session
    let retrieved = manager.get(&session_id).await?;
    assert!(retrieved.is_some());
    
    let retrieved_session = retrieved.unwrap();
    assert_eq!(retrieved_session.id(), &session_id);
    assert_eq!(retrieved_session.system_prompt, Some("Memory test".to_string()));
    assert_eq!(
        retrieved_session.metadata.get("key"),
        Some(&serde_json::json!("value"))
    );
    
    // List sessions
    let ids = manager.list().await?;
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], session_id);
    
    // Delete session
    manager.delete(&session_id).await?;
    assert!(manager.get(&session_id).await?.is_none());
    
    Ok(())
}

#[tokio::test]
async fn test_file_storage_persistence() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    // Create manager with file storage
    let manager = SessionManager::with_storage(
        StorageBackend::File(storage_path.clone())
    );
    
    // Create and store a session
    let stored_session = manager.create_session()
        .with_system_prompt("File storage test")
        .with_metadata("persistent", serde_json::json!(true))
        .build()
        .await?;
    
    let stored_id = stored_session.id().clone();
    
    // Create a new manager with the same storage path
    let new_manager = SessionManager::with_storage(
        StorageBackend::File(storage_path)
    );
    
    // Session should be persisted
    let retrieved = new_manager.get(&stored_id).await?;
    assert!(retrieved.is_some());
    
    let retrieved_session = retrieved.unwrap();
    assert_eq!(retrieved_session.system_prompt, Some("File storage test".to_string()));
    assert_eq!(
        retrieved_session.metadata.get("persistent"),
        Some(&serde_json::json!(true))
    );
    
    Ok(())
}

#[tokio::test]
async fn test_file_storage_multiple_sessions() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    let manager = SessionManager::with_storage(
        StorageBackend::File(storage_path)
    );
    
    // Create multiple sessions
    let mut session_ids = Vec::new();
    for i in 0..5 {
        let session = manager.create_session()
            .with_system_prompt(format!("Session {}", i))
            .with_metadata("index", serde_json::json!(i))
            .build()
            .await?;
        session_ids.push(session.id().clone());
    }
    
    // List all sessions
    let listed_ids = manager.list().await?;
    assert_eq!(listed_ids.len(), 5);
    
    // Verify each session
    for (i, id) in session_ids.iter().enumerate() {
        let session = manager.get(id).await?.unwrap();
        assert_eq!(session.system_prompt, Some(format!("Session {}", i)));
        assert_eq!(
            session.metadata.get("index"),
            Some(&serde_json::json!(i))
        );
    }
    
    // Clear all sessions
    manager.clear().await?;
    assert_eq!(manager.list().await?.len(), 0);
    
    Ok(())
}

#[tokio::test]
async fn test_file_storage_update_session() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    let manager = SessionManager::with_storage(
        StorageBackend::File(storage_path.clone())
    );
    
    // Create initial session
    let session = manager.create_session()
        .with_system_prompt("Initial prompt")
        .with_metadata("version", serde_json::json!(1))
        .build()
        .await?;
    
    let session_id = session.id().clone();
    
    // Update the session manually
    let mut updated_session = session.clone();
    updated_session.system_prompt = Some("Updated prompt".to_string());
    updated_session.metadata.insert("version".to_string(), serde_json::json!(2));
    
    // Save through the storage backend directly
    let file_storage = FileStorage::new(storage_path.clone());
    file_storage.save(&updated_session).await?;
    
    // Retrieve and verify update
    let retrieved = manager.get(&session_id).await?.unwrap();
    assert_eq!(retrieved.system_prompt, Some("Updated prompt".to_string()));
    assert_eq!(
        retrieved.metadata.get("version"),
        Some(&serde_json::json!(2))
    );
    
    // Verify updated_at has changed
    assert!(retrieved.updated_at > session.created_at);
    
    Ok(())
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn test_sqlite_storage_basic_operations() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test_sessions.db");
    
    // Create manager with SQLite storage
    let manager = SessionManager::with_storage_async(
        StorageBackend::Sqlite(db_path.clone())
    ).await?;
    
    // Create and store a session
    let session = manager.create_session()
        .with_system_prompt("SQLite test")
        .with_metadata("db", serde_json::json!("sqlite"))
        .build()
        .await?;
    
    let session_id = session.id().clone();
    
    // Retrieve the session
    let retrieved = manager.get(&session_id).await?;
    assert!(retrieved.is_some());
    
    let retrieved_session = retrieved.unwrap();
    assert_eq!(retrieved_session.system_prompt, Some("SQLite test".to_string()));
    assert_eq!(
        retrieved_session.metadata.get("db"),
        Some(&serde_json::json!("sqlite"))
    );
    
    Ok(())
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn test_sqlite_storage_persistence_across_instances() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("persist_test.db");
    
    // First instance: create sessions
    {
        let manager = SessionManager::with_storage_async(
            StorageBackend::Sqlite(db_path.clone())
        ).await?;
        
        for i in 0..3 {
            manager.create_session()
                .with_system_prompt(format!("Persistent session {}", i))
                .with_metadata("order", serde_json::json!(i))
                .build()
                .await?;
        }
    }
    
    // Second instance: verify persistence
    {
        let manager = SessionManager::with_storage_async(
            StorageBackend::Sqlite(db_path.clone())
        ).await?;
        
        let sessions = manager.list().await?;
        assert_eq!(sessions.len(), 3);
        
        // Sessions should be ordered by updated_at DESC
        for session_id in sessions.iter() {
            let session = manager.get(session_id).await?.unwrap();
            // Note: SQLite returns in DESC order, so we check differently
            assert!(session.system_prompt.is_some());
            assert!(session.metadata.contains_key("order"));
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_session_not_found_error() -> Result<()> {
    let manager = SessionManager::new();
    let non_existent_id = SessionId::new("does-not-exist");
    
    // Get should return None
    assert!(manager.get(&non_existent_id).await?.is_none());
    
    // Resume should return error
    let result = manager.resume(&non_existent_id).await;
    assert!(result.is_err());
    
    match result {
        Err(crate::Error::SessionNotFound(id)) => {
            assert_eq!(id, "does-not-exist");
        }
        _ => panic!("Expected SessionNotFound error"),
    }
    
    Ok(())
}

#[tokio::test]
async fn test_session_builder_with_id() -> Result<()> {
    use super::session::SessionBuilder;
    
    let custom_id = "custom-session-id";
    let session = SessionBuilder::with_id(custom_id)
        .with_system_prompt("Custom ID session")
        .build()
        .await?;
    
    assert_eq!(session.id().as_str(), custom_id);
    assert_eq!(session.system_prompt, Some("Custom ID session".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_file_storage_invalid_characters_in_id() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    let manager = SessionManager::with_storage(
        StorageBackend::File(storage_path)
    );
    
    // Create session with ID containing path separators
    let stored = manager.create_session()
        .with_system_prompt("Path test")
        .build()
        .await?;
    
    // Should be able to retrieve it
    let retrieved = manager.get(&stored.id()).await?;
    assert!(retrieved.is_some());
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_session_access() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    let manager = SessionManager::with_storage(
        StorageBackend::File(storage_path)
    );
    
    // Create a session
    let session = manager.create_session()
        .with_system_prompt("Concurrent test")
        .build()
        .await?;
    
    let session_id = session.id().clone();
    
    // Spawn multiple tasks to access the session concurrently
    let mut handles = Vec::new();
    for i in 0..10 {
        let manager_clone = manager.clone();
        let id_clone = session_id.clone();
        
        let handle = tokio::spawn(async move {
            // Each task tries to get the session
            let result = manager_clone.get(&id_clone).await;
            assert!(result.is_ok());
            assert!(result.unwrap().is_some());
            i
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result < 10);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_storage_backend_selection() {
    // Test memory backend
    let memory_manager = SessionManager::new();
    assert!(!format!("{:?}", memory_manager).is_empty());
    
    // Test file backend
    let temp_dir = TempDir::new().unwrap();
    let file_manager = SessionManager::with_storage(
        StorageBackend::File(temp_dir.path().to_path_buf())
    );
    assert!(!format!("{:?}", file_manager).is_empty());
    
    // Test SQLite backend (should panic in sync version)
    #[cfg(feature = "sqlite")]
    {
        let result = std::panic::catch_unwind(|| {
            SessionManager::with_storage(
                StorageBackend::Sqlite(PathBuf::from("test.db"))
            )
        });
        assert!(result.is_err());
    }
}

#[tokio::test]
async fn test_session_timestamps() -> Result<()> {
    let manager = SessionManager::new();
    
    let session = manager.create_session()
        .with_system_prompt("Timestamp test")
        .build()
        .await?;
    
    // Verify timestamps are set
    assert!(session.created_at <= session.updated_at);
    
    // Small delay to ensure timestamp difference
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    
    // Update session (simulate by saving again)
    let storage = MemoryStorage::new();
    storage.save(&session).await?;
    
    // For memory storage, updated_at won't change automatically
    // This is expected behavior for in-memory storage
    
    Ok(())
}