//! Session management tests
//!
//! This module provides comprehensive tests for session creation, management,
//! and cleanup functionality in the Claude AI core system.

use crate::{Session, SessionId, SessionManager};
use std::collections::HashSet;

/// Test session creation and basic properties
#[cfg(test)]
mod session_creation_tests {
    use super::*;

    #[test]
    fn test_session_id_creation() {
        let id = SessionId::new("test-session-123");
        assert_eq!(id.as_str(), "test-session-123");
        assert_eq!(id.to_string(), "test-session-123");
    }

    #[test]
    fn test_session_id_uniqueness() {
        let mut ids = HashSet::new();

        // Generate many session IDs and verify uniqueness when different strings used
        for i in 0..1000 {
            let id = SessionId::new(format!("session-{}", i));
            assert!(ids.insert(id), "SessionId should be unique");
        }
    }

    #[test]
    fn test_session_creation_with_id() {
        let id = SessionId::new("test-session");
        let session = Session::new(id.clone());

        assert_eq!(session.id(), &id);
        assert_eq!(session.system_prompt, None);
        assert!(session.metadata.is_empty());
    }

    #[test]
    fn test_session_with_system_prompt() {
        let id = SessionId::new("test-session");
        let session = Session::new(id).with_system_prompt("You are a helpful assistant");

        assert_eq!(
            session.system_prompt,
            Some("You are a helpful assistant".to_string())
        );
    }

    #[test]
    fn test_session_id_string_representation() {
        let id = SessionId::new("my-session-id");
        let id_str = id.to_string();

        assert_eq!(id_str, "my-session-id");
        assert_eq!(id.as_str(), "my-session-id");
    }

    #[test]
    fn test_session_id_equality() {
        let id1 = SessionId::new("same-id");
        let id2 = SessionId::new("same-id");
        let id3 = SessionId::new("different-id");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(id2, id3);
    }
}

/// Test session manager functionality  
#[cfg(test)]
mod session_manager_tests {
    use super::*;

    #[test]
    fn test_session_manager_creation() {
        let manager = SessionManager::new();
        // SessionManager created successfully - basic smoke test
        assert!(!format!("{:?}", manager).is_empty());
    }

    #[test]
    fn test_session_manager_default() {
        let manager = SessionManager::default();
        // Default SessionManager created successfully
        assert!(!format!("{:?}", manager).is_empty());
    }

    #[tokio::test]
    async fn test_create_session_builder() {
        let manager = SessionManager::new();
        let session_builder = manager.create_session();

        // SessionBuilder created successfully - basic API test
        assert!(!format!("{:?}", session_builder).is_empty());
    }

    #[tokio::test]
    async fn test_session_get_operation() {
        let manager = SessionManager::new();
        let test_id = SessionId::new("test-session");

        // Test getting a non-existent session
        let result = manager.get(&test_id).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }

    #[tokio::test]
    async fn test_session_resume_operation() {
        let manager = SessionManager::new();
        let test_id = SessionId::new("non-existent-session");

        // Test resuming a non-existent session should return an error
        let result = manager.resume(&test_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let manager = SessionManager::new();

        // Test listing sessions (should be empty initially)
        let result = manager.list().await;
        assert!(result.is_ok());

        let sessions = result.unwrap();
        assert!(sessions.is_empty());
    }
}

/// Test session cloning and equality
#[cfg(test)]
mod session_cloning_tests {
    use super::*;

    #[test]
    fn test_session_id_hash() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        let id1 = SessionId::new("session1");
        let id2 = SessionId::new("session2");

        map.insert(id1.clone(), "session1");
        map.insert(id2.clone(), "session2");

        assert_eq!(map.get(&id1), Some(&"session1"));
        assert_eq!(map.get(&id2), Some(&"session2"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_session_clone() {
        let id = SessionId::new("test-session");
        let original = Session::new(id.clone()).with_system_prompt("Test prompt");

        let cloned = original.clone();

        assert_eq!(original.id(), cloned.id());
        assert_eq!(original.system_prompt, cloned.system_prompt);
        assert_eq!(original.metadata.len(), cloned.metadata.len());
    }

    #[test]
    fn test_session_debug_format() {
        let session = Session::new(SessionId::new("debug-test"));
        let debug_str = format!("{:?}", session);

        // Should contain session ID and basic info
        assert!(debug_str.contains("debug-test"));
    }

    #[test]
    fn test_session_id_debug_format() {
        let id = SessionId::new("debug-session-id");
        let debug_str = format!("{:?}", id);

        // Should contain the ID string
        assert!(debug_str.contains("debug-session-id"));
    }

    #[test]
    fn test_session_manager_clone() {
        let manager = SessionManager::new();
        let cloned_manager = manager.clone();

        // Both managers should be valid (basic smoke test)
        assert!(!format!("{:?}", manager).is_empty());
        assert!(!format!("{:?}", cloned_manager).is_empty());
    }
}

/// Test session metadata and attributes
#[cfg(test)]
mod session_metadata_tests {
    use super::*;

    #[test]
    fn test_session_metadata_modification() {
        let id = SessionId::new("metadata-test");
        let mut session = Session::new(id);

        // Add metadata
        session
            .metadata
            .insert("key1".to_string(), serde_json::json!("value1"));
        session
            .metadata
            .insert("key2".to_string(), serde_json::json!(42));

        assert_eq!(session.metadata.len(), 2);
        assert_eq!(
            session.metadata.get("key1"),
            Some(&serde_json::json!("value1"))
        );
        assert_eq!(session.metadata.get("key2"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_session_system_prompt_modification() {
        let id = SessionId::new("prompt-test");
        let session = Session::new(id).with_system_prompt("Initial prompt");

        assert_eq!(session.system_prompt, Some("Initial prompt".to_string()));

        // Create new session with different prompt
        let updated_session = session.with_system_prompt("Updated prompt");
        assert_eq!(
            updated_session.system_prompt,
            Some("Updated prompt".to_string())
        );
    }

    #[test]
    fn test_session_id_immutability() {
        let id = SessionId::new("immutable-test");
        let session = Session::new(id.clone());

        assert_eq!(session.id(), &id);

        // ID should remain the same even after modifications
        let modified_session = session.with_system_prompt("New prompt");
        assert_eq!(modified_session.id(), &id);
    }
}

/// Test edge cases and special scenarios
#[cfg(test)]
mod session_edge_cases {
    use super::*;

    #[test]
    fn test_empty_session_id() {
        let id = SessionId::new("");
        let session = Session::new(id.clone());

        assert_eq!(id.as_str(), "");
        assert_eq!(session.id().as_str(), "");
    }

    #[test]
    fn test_unicode_session_id() {
        let unicode_id = "セッション-🚀-тест";
        let id = SessionId::new(unicode_id);
        let session = Session::new(id.clone());

        assert_eq!(id.as_str(), unicode_id);
        assert_eq!(session.id().as_str(), unicode_id);
    }

    #[test]
    fn test_very_long_session_id() {
        let long_id = "a".repeat(10000);
        let id = SessionId::new(&long_id);
        let session = Session::new(id.clone());

        assert_eq!(id.as_str().len(), 10000);
        assert_eq!(session.id().as_str(), &long_id);
    }

    #[test]
    fn test_special_characters_in_session_id() {
        let special_id = "session!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let id = SessionId::new(special_id);
        let session = Session::new(id.clone());

        assert_eq!(id.as_str(), special_id);
        assert_eq!(session.id().as_str(), special_id);
    }

    #[test]
    fn test_empty_system_prompt() {
        let id = SessionId::new("empty-prompt-test");
        let session = Session::new(id).with_system_prompt("");

        assert_eq!(session.system_prompt, Some("".to_string()));
    }

    #[test]
    fn test_multiline_system_prompt() {
        let multiline_prompt = "Line 1\nLine 2\nLine 3";
        let id = SessionId::new("multiline-test");
        let session = Session::new(id).with_system_prompt(multiline_prompt);

        assert_eq!(session.system_prompt, Some(multiline_prompt.to_string()));
    }

    #[test]
    fn test_large_metadata() {
        let id = SessionId::new("large-metadata-test");
        let mut session = Session::new(id);

        // Add many metadata entries
        for i in 0..1000 {
            session.metadata.insert(
                format!("key{}", i),
                serde_json::json!(format!("value{}", i)),
            );
        }

        assert_eq!(session.metadata.len(), 1000);
        assert_eq!(
            session.metadata.get("key500"),
            Some(&serde_json::json!("value500"))
        );
    }

    #[test]
    fn test_complex_metadata_types() {
        let id = SessionId::new("complex-metadata-test");
        let mut session = Session::new(id);

        // Add various JSON types
        session
            .metadata
            .insert("string".to_string(), serde_json::json!("text"));
        session
            .metadata
            .insert("number".to_string(), serde_json::json!(42));
        session
            .metadata
            .insert("float".to_string(), serde_json::json!(3.5));
        session
            .metadata
            .insert("boolean".to_string(), serde_json::json!(true));
        session
            .metadata
            .insert("null".to_string(), serde_json::json!(null));
        session
            .metadata
            .insert("array".to_string(), serde_json::json!([1, 2, 3]));
        session
            .metadata
            .insert("object".to_string(), serde_json::json!({"nested": "value"}));

        assert_eq!(session.metadata.len(), 7);
        assert_eq!(
            session.metadata.get("string"),
            Some(&serde_json::json!("text"))
        );
        assert_eq!(session.metadata.get("number"), Some(&serde_json::json!(42)));
        assert_eq!(
            session.metadata.get("object"),
            Some(&serde_json::json!({"nested": "value"}))
        );
    }
}

/// Test session persistence across different storage backends
#[cfg(test)]
mod session_persistence_tests {
    use super::*;
    use tempfile::TempDir;
    use std::path::PathBuf;
    use crate::session::{StorageBackend, SessionStorage, FileStorage, MemoryStorage};

    #[tokio::test]
    async fn test_memory_storage_basic_operations() -> crate::Result<()> {
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
    async fn test_file_storage_persistence() -> crate::Result<()> {
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
    async fn test_session_timestamps() -> crate::Result<()> {
        let manager = SessionManager::new();
        
        let session = manager.create_session()
            .with_system_prompt("Timestamp test")
            .build()
            .await?;
        
        // Verify timestamps are set
        assert!(session.created_at <= session.updated_at);
        
        Ok(())
    }
}

/// Test concurrent access and thread safety
#[cfg(test)]
mod session_concurrency_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_session_id_thread_safety() {
        let id = Arc::new(SessionId::new("thread-safe-test"));
        let mut handles = vec![];

        // Spawn multiple threads that use the session ID
        for i in 0..10 {
            let id_clone = Arc::clone(&id);
            let handle = thread::spawn(move || {
                let session = Session::new((*id_clone).clone());
                // Verify session was created with correct ID
                assert_eq!(session.id().as_str(), "thread-safe-test");
                i
            });
            handles.push(handle);
        }

        // Wait for all threads and collect results
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result < 10);
        }
    }

    #[test]
    fn test_session_manager_thread_safety() {
        let manager = Arc::new(SessionManager::new());
        let mut handles = vec![];

        // Spawn multiple threads that use the session manager
        for i in 0..10 {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                // Just test that we can create session builders
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let _builder = manager_clone.create_session();
                });
                i
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result < 10);
        }
    }
}
