//! Session management for Claude Interactive
//!
//! This module provides functionality for managing Claude sessions,
//! including creating, storing, retrieving, and managing session state.

use crate::core::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Unique identifier for a session
pub type SessionId = Uuid;

/// Session metadata and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier
    pub id: SessionId,
    /// Human-readable session name
    pub name: String,
    /// Session description (optional)
    pub description: Option<String>,
    /// When the session was created
    pub created_at: DateTime<Utc>,
    /// When the session was last accessed
    pub last_accessed: DateTime<Utc>,
    /// Session configuration
    pub config: SessionConfig,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

/// Configuration for a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Working directory for the session
    pub working_dir: Option<PathBuf>,
    /// Environment variables for the session
    pub env_vars: HashMap<String, String>,
    /// Default timeout for operations
    pub default_timeout: Option<u64>,
    /// Whether to enable verbose logging
    pub verbose: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            working_dir: None,
            env_vars: HashMap::new(),
            default_timeout: Some(60),
            verbose: false,
        }
    }
}

/// Manages Claude sessions
#[derive(Debug)]
pub struct SessionManager {
    /// Directory where session data is stored
    data_dir: PathBuf,
    /// In-memory cache of loaded sessions
    sessions: HashMap<SessionId, Session>,
}

impl SessionManager {
    /// Create a new session manager with default storage location
    pub fn with_default_storage() -> Result<Self> {
        let data_dir = crate::cli::default_data_dir();
        Self::new(data_dir)
    }

    /// Create a new session manager
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        // Ensure the data directory exists
        if !data_dir.exists() {
            std::fs::create_dir_all(&data_dir)?;
        }

        let mut manager = Self {
            data_dir,
            sessions: HashMap::new(),
        };

        // Load existing sessions
        manager.load_sessions()?;

        Ok(manager)
    }

    /// Create a new session
    pub fn create_session(
        &mut self,
        name: String,
        description: Option<String>,
        config: Option<SessionConfig>,
    ) -> Result<SessionId> {
        let id = SessionId::new_v4();
        let now = Utc::now();

        let session = Session {
            id,
            name,
            description,
            created_at: now,
            last_accessed: now,
            config: config.unwrap_or_default(),
            metadata: HashMap::new(),
        };

        // Save to disk
        self.save_session(&session)?;

        // Add to in-memory cache
        self.sessions.insert(id, session);

        Ok(id)
    }

    /// Get a session by ID
    pub fn get_session(&mut self, id: SessionId) -> Result<Option<&Session>> {
        if let Some(session) = self.sessions.get(&id) {
            // Update last accessed time
            let mut updated_session = session.clone();
            updated_session.last_accessed = Utc::now();
            self.save_session(&updated_session)?;
            self.sessions.insert(id, updated_session);
            return Ok(self.sessions.get(&id));
        }

        Ok(None)
    }

    /// Get a session by name
    pub fn get_session_by_name(&mut self, name: &str) -> Result<Option<&Session>> {
        let id = self
            .sessions
            .values()
            .find(|session| session.name == name)
            .map(|session| session.id);

        if let Some(id) = id {
            self.get_session(id)
        } else {
            Ok(None)
        }
    }

    /// List all sessions
    pub fn list_sessions(&self) -> Vec<&Session> {
        self.sessions.values().collect()
    }

    /// Get the current session ID (most recently accessed)
    pub fn get_current_session_id(&self) -> Option<SessionId> {
        self.sessions
            .values()
            .max_by_key(|session| session.last_accessed)
            .map(|session| session.id)
    }

    /// Get the current session (most recently accessed)
    pub fn get_current_session(&self) -> Option<&Session> {
        self.sessions
            .values()
            .max_by_key(|session| session.last_accessed)
    }

    /// Switch to a session by ID
    pub fn switch_to_session(&mut self, id: SessionId) -> Result<()> {
        if let Some(session) = self.sessions.get_mut(&id) {
            session.last_accessed = Utc::now();
            // Create a clone to avoid borrow checker issues
            let session_to_save = session.clone();
            let _ = session; // Explicitly drop the mutable borrow
            self.save_session(&session_to_save)?;
        }
        Ok(())
    }

    /// Delete a session
    pub fn delete_session(&mut self, id: SessionId) -> Result<bool> {
        if self.sessions.remove(&id).is_some() {
            // Remove from disk
            let session_file = self.session_file_path(id);
            if session_file.exists() {
                std::fs::remove_file(session_file)?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Update session metadata
    pub fn update_session_metadata(
        &mut self,
        id: SessionId,
        key: String,
        value: String,
    ) -> Result<()> {
        if let Some(session) = self.sessions.get_mut(&id) {
            session.metadata.insert(key, value);
            session.last_accessed = Utc::now();
            let session_clone = session.clone();
            self.save_session(&session_clone)?;
        }
        Ok(())
    }

    /// Load all sessions from disk
    fn load_sessions(&mut self) -> Result<()> {
        let sessions_dir = self.data_dir.join("sessions");
        if !sessions_dir.exists() {
            std::fs::create_dir_all(&sessions_dir)?;
            return Ok(());
        }

        for entry in std::fs::read_dir(&sessions_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(session) = self.load_session_from_file(&path) {
                    self.sessions.insert(session.id, session);
                }
            }
        }

        Ok(())
    }

    /// Load a session from a file
    fn load_session_from_file(&self, path: &Path) -> Result<Session> {
        let content = std::fs::read_to_string(path)?;
        let session: Session = serde_json::from_str(&content)?;
        Ok(session)
    }

    /// Save a session to disk
    fn save_session(&self, session: &Session) -> Result<()> {
        let sessions_dir = self.data_dir.join("sessions");
        if !sessions_dir.exists() {
            std::fs::create_dir_all(&sessions_dir)?;
        }

        let session_file = self.session_file_path(session.id);
        let content = serde_json::to_string_pretty(session)?;
        std::fs::write(session_file, content)?;

        Ok(())
    }

    /// Get the file path for a session
    fn session_file_path(&self, id: SessionId) -> PathBuf {
        self.data_dir.join("sessions").join(format!("{}.json", id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_manager() -> (SessionManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let manager = SessionManager::new(temp_dir.path().to_path_buf()).unwrap();
        (manager, temp_dir)
    }

    #[test]
    fn test_create_session() {
        let (mut manager, _temp_dir) = create_test_manager();

        let session_id = manager
            .create_session("test-session".to_string(), None, None)
            .unwrap();

        assert!(manager.sessions.contains_key(&session_id));

        let session = manager.get_session(session_id).unwrap().unwrap();
        assert_eq!(session.name, "test-session");
        assert!(session.description.is_none());
    }

    #[test]
    fn test_get_session_by_name() {
        let (mut manager, _temp_dir) = create_test_manager();

        let _session_id = manager
            .create_session("test-session".to_string(), None, None)
            .unwrap();

        let session = manager
            .get_session_by_name("test-session")
            .unwrap()
            .unwrap();
        assert_eq!(session.name, "test-session");

        let not_found = manager.get_session_by_name("nonexistent").unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_list_sessions() {
        let (mut manager, _temp_dir) = create_test_manager();

        let _id1 = manager
            .create_session("session1".to_string(), None, None)
            .unwrap();
        let _id2 = manager
            .create_session("session2".to_string(), None, None)
            .unwrap();

        let sessions = manager.list_sessions();
        assert_eq!(sessions.len(), 2);

        let names: Vec<&str> = sessions.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"session1"));
        assert!(names.contains(&"session2"));
    }

    #[test]
    fn test_delete_session() {
        let (mut manager, _temp_dir) = create_test_manager();

        let session_id = manager
            .create_session("test-session".to_string(), None, None)
            .unwrap();

        assert!(manager.sessions.contains_key(&session_id));

        let deleted = manager.delete_session(session_id).unwrap();
        assert!(deleted);
        assert!(!manager.sessions.contains_key(&session_id));

        // Try to delete again - should return false
        let deleted_again = manager.delete_session(session_id).unwrap();
        assert!(!deleted_again);
    }

    #[test]
    fn test_session_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();

        let session_id = {
            let mut manager = SessionManager::new(data_dir.clone()).unwrap();
            manager
                .create_session("persistent-session".to_string(), None, None)
                .unwrap()
        };

        // Create a new manager instance - should load the session from disk
        let mut manager2 = SessionManager::new(data_dir).unwrap();
        let session = manager2.get_session(session_id).unwrap().unwrap();
        assert_eq!(session.name, "persistent-session");
    }

    #[test]
    fn test_update_session_metadata() {
        let (mut manager, _temp_dir) = create_test_manager();

        let session_id = manager
            .create_session("test-session".to_string(), None, None)
            .unwrap();

        manager
            .update_session_metadata(session_id, "key1".to_string(), "value1".to_string())
            .unwrap();

        let session = manager.get_session(session_id).unwrap().unwrap();
        assert_eq!(session.metadata.get("key1"), Some(&"value1".to_string()));
    }
}
