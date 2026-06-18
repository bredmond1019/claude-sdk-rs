use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "sqlite")]
use super::sqlite_storage::SqliteStorage;

/// Unique identifier for a Claude AI session
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Create a new session ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the session ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Represents an active Claude AI session
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    /// Unique identifier for this session
    pub id: SessionId,
    /// Optional system prompt for this session
    pub system_prompt: Option<String>,
    /// Additional metadata associated with the session
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation timestamp
    #[serde(default = "chrono::Utc::now")]
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    #[serde(default = "chrono::Utc::now")]
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl Session {
    /// Create a new session with the given ID
    pub fn new(id: SessionId) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            system_prompt: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the system prompt for this session
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Get a reference to the session ID
    pub fn id(&self) -> &SessionId {
        &self.id
    }
}

/// Trait for session storage backends
///
/// This trait defines the interface for persisting sessions across
/// different storage backends (e.g., file system, database).
#[async_trait]
pub trait SessionStorage: Send + Sync + std::fmt::Debug {
    /// Save a session to storage
    async fn save(&self, session: &Session) -> Result<()>;

    /// Load a session from storage by ID
    async fn load(&self, id: &SessionId) -> Result<Option<Session>>;

    /// Delete a session from storage
    async fn delete(&self, id: &SessionId) -> Result<()>;

    /// List all session IDs in storage
    async fn list_ids(&self) -> Result<Vec<SessionId>>;

    /// Clear all sessions from storage
    async fn clear(&self) -> Result<()>;
}

/// Storage backend selection
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// In-memory storage (default, not persistent)
    Memory,
    /// File-based storage with directory path
    File(PathBuf),
    /// SQLite database storage with file path
    #[cfg(feature = "sqlite")]
    Sqlite(PathBuf),
}

impl Default for StorageBackend {
    fn default() -> Self {
        StorageBackend::Memory
    }
}

/// In-memory session storage (non-persistent)
#[derive(Debug, Clone)]
pub struct MemoryStorage {
    sessions: Arc<RwLock<HashMap<SessionId, Session>>>,
}

impl MemoryStorage {
    /// Create a new in-memory storage backend
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl SessionStorage for MemoryStorage {
    async fn save(&self, session: &Session) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn load(&self, id: &SessionId) -> Result<Option<Session>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(id).cloned())
    }

    async fn delete(&self, id: &SessionId) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id);
        Ok(())
    }

    async fn list_ids(&self) -> Result<Vec<SessionId>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.keys().cloned().collect())
    }

    async fn clear(&self) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.clear();
        Ok(())
    }
}

/// File-based session storage
#[derive(Debug, Clone)]
pub struct FileStorage {
    base_path: PathBuf,
}

impl FileStorage {
    /// Create a new file-based storage backend
    pub fn new(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    fn session_file_path(&self, id: &SessionId) -> PathBuf {
        self.base_path.join(format!("{}.json", id.as_str()))
    }

    async fn ensure_directory(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.base_path)
            .await
            .map_err(|e| Error::Io(e))?;
        Ok(())
    }
}

#[async_trait]
impl SessionStorage for FileStorage {
    async fn save(&self, session: &Session) -> Result<()> {
        self.ensure_directory().await?;

        let mut updated_session = session.clone();
        updated_session.updated_at = chrono::Utc::now();

        let json = serde_json::to_string_pretty(&updated_session)?;
        let path = self.session_file_path(&session.id);

        tokio::fs::write(&path, json)
            .await
            .map_err(|e| Error::Io(e))?;

        Ok(())
    }

    async fn load(&self, id: &SessionId) -> Result<Option<Session>> {
        let path = self.session_file_path(id);

        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                let session: Session = serde_json::from_str(&content)?;
                Ok(Some(session))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Io(e)),
        }
    }

    async fn delete(&self, id: &SessionId) -> Result<()> {
        let path = self.session_file_path(id);

        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::Io(e)),
        }
    }

    async fn list_ids(&self) -> Result<Vec<SessionId>> {
        self.ensure_directory().await?;

        let mut ids = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.base_path)
            .await
            .map_err(|e| Error::Io(e))?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| Error::Io(e))? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(SessionId::new(stem));
                }
            }
        }

        Ok(ids)
    }

    async fn clear(&self) -> Result<()> {
        if self.base_path.exists() {
            let mut entries = tokio::fs::read_dir(&self.base_path)
                .await
                .map_err(|e| Error::Io(e))?;

            while let Some(entry) = entries.next_entry().await.map_err(|e| Error::Io(e))? {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    tokio::fs::remove_file(&path)
                        .await
                        .map_err(|e| Error::Io(e))?;
                }
            }
        }

        Ok(())
    }
}

/// Manages multiple Claude AI sessions with configurable storage
#[derive(Debug, Clone)]
pub struct SessionManager {
    storage: Arc<Box<dyn SessionStorage>>,
}

impl SessionManager {
    /// Create a new session manager with default in-memory storage
    pub fn new() -> Self {
        Self::with_storage(StorageBackend::Memory)
    }

    /// Create a new session manager with specified storage backend
    pub fn with_storage(backend: StorageBackend) -> Self {
        let storage: Box<dyn SessionStorage> = match backend {
            StorageBackend::Memory => Box::new(MemoryStorage::new()),
            StorageBackend::File(path) => Box::new(FileStorage::new(path)),
            #[cfg(feature = "sqlite")]
            StorageBackend::Sqlite(_) => {
                panic!("SQLite storage requires async initialization. Use SessionManager::with_storage_async instead.");
            }
        };

        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create a new session manager with specified storage backend (async version for SQLite)
    pub async fn with_storage_async(backend: StorageBackend) -> Result<Self> {
        let storage: Box<dyn SessionStorage> = match backend {
            StorageBackend::Memory => Box::new(MemoryStorage::new()),
            StorageBackend::File(path) => Box::new(FileStorage::new(path)),
            #[cfg(feature = "sqlite")]
            StorageBackend::Sqlite(path) => Box::new(SqliteStorage::new(path).await?),
        };

        Ok(Self {
            storage: Arc::new(storage),
        })
    }

    /// Create a new session builder
    pub fn builder() -> SessionBuilder {
        SessionBuilder::new()
    }

    /// Create a new session associated with this manager
    pub fn create_session(&self) -> SessionBuilder {
        SessionBuilder::with_manager(self.clone())
    }

    /// Get a session by ID, returns None if not found
    pub async fn get(&self, id: &SessionId) -> Result<Option<Session>> {
        self.storage.load(id).await
    }

    /// Resume an existing session, returns error if not found
    pub async fn resume(&self, id: &SessionId) -> Result<Session> {
        self.storage
            .load(id)
            .await?
            .ok_or_else(|| Error::SessionNotFound(id.to_string()))
    }

    /// List all session IDs
    pub async fn list(&self) -> Result<Vec<SessionId>> {
        self.storage.list_ids().await
    }

    /// Delete a session by ID
    pub async fn delete(&self, id: &SessionId) -> Result<()> {
        self.storage.delete(id).await
    }

    /// Clear all sessions
    pub async fn clear(&self) -> Result<()> {
        self.storage.clear().await
    }

    async fn store(&self, session: Session) -> Result<()> {
        self.storage.save(&session).await
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating sessions with custom configuration
///
/// `SessionBuilder` provides a fluent interface for constructing sessions with
/// custom system prompts and metadata. Sessions can be standalone or managed
/// by a `SessionManager` for persistence and retrieval.
///
/// # Examples
///
/// ```rust,no_run
/// # use claude_sdk_rs::core::{SessionBuilder, Result};
/// # use serde_json::json;
/// # #[tokio::main]
/// # async fn main() -> Result<()> {
/// // Create a standalone session
/// let session = SessionBuilder::new()
///     .with_system_prompt("You are a helpful coding assistant")
///     .with_metadata("project", json!("my-app"))
///     .with_metadata("language", json!("rust"))
///     .build()
///     .await?;
///
/// // Create a managed session (stored in SessionManager)
/// # use claude_sdk_rs::core::SessionManager;
/// let manager = SessionManager::new();
/// let session = manager.create_session()
///     .with_system_prompt("You are a data analyst")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SessionBuilder {
    session: Session,
    manager: Option<SessionManager>,
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBuilder {
    /// Create a new session builder
    pub fn new() -> Self {
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        Self {
            session: Session::new(id),
            manager: None,
        }
    }

    /// Create a session builder with a specific ID
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            session: Session::new(SessionId::new(id)),
            manager: None,
        }
    }

    fn with_manager(manager: SessionManager) -> Self {
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        Self {
            session: Session::new(id),
            manager: Some(manager),
        }
    }

    /// Set the system prompt for the session being built
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.session.system_prompt = Some(prompt.into());
        self
    }

    /// Add metadata to the session being built
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.session.metadata.insert(key.into(), value);
        self
    }

    /// Build and optionally store the session
    pub async fn build(self) -> Result<Session> {
        if let Some(manager) = self.manager {
            manager.store(self.session.clone()).await?;
        }
        Ok(self.session)
    }
}
