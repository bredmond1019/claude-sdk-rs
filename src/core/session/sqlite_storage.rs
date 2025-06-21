use super::session::{Session, SessionId, SessionStorage};
use crate::{Error, Result};
use async_trait::async_trait;
use sqlx::{
    sqlite::{SqlitePool, SqlitePoolOptions},
    Row,
};
use std::path::PathBuf;

/// SQLite-based session storage
#[derive(Debug)]
pub struct SqliteStorage {
    pool: SqlitePool,
}

impl SqliteStorage {
    /// Create a new SQLite storage backend
    pub async fn new(path: PathBuf) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| Error::Io(e))?;
        }

        let connection_string = format!("sqlite:{}", path.display());

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&connection_string)
            .await
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to connect to SQLite database: {}", e),
                ))
            })?;

        // Create sessions table if it doesn't exist
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY NOT NULL,
                system_prompt TEXT,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ",
        )
        .execute(&pool)
        .await
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create sessions table: {}", e),
            ))
        })?;

        Ok(Self { pool })
    }
}

impl Clone for SqliteStorage {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

#[async_trait]
impl SessionStorage for SqliteStorage {
    async fn save(&self, session: &Session) -> Result<()> {
        let metadata_json = serde_json::to_string(&session.metadata)?;
        let mut updated_session = session.clone();
        updated_session.updated_at = chrono::Utc::now();

        sqlx::query(
            r"
            INSERT INTO sessions (id, system_prompt, metadata, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(id) DO UPDATE SET
                system_prompt = excluded.system_prompt,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            ",
        )
        .bind(session.id.as_str())
        .bind(&session.system_prompt)
        .bind(&metadata_json)
        .bind(session.created_at.to_rfc3339())
        .bind(updated_session.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to save session: {}", e),
            ))
        })?;

        Ok(())
    }

    async fn load(&self, id: &SessionId) -> Result<Option<Session>> {
        let row = sqlx::query(
            r"
            SELECT id, system_prompt, metadata, created_at, updated_at
            FROM sessions
            WHERE id = ?1
            ",
        )
        .bind(id.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load session: {}", e),
            ))
        })?;

        match row {
            Some(row) => {
                let id_str: String = row.get("id");
                let system_prompt: Option<String> = row.get("system_prompt");
                let metadata_json: String = row.get("metadata");
                let created_at_str: String = row.get("created_at");
                let updated_at_str: String = row.get("updated_at");

                let metadata = serde_json::from_str(&metadata_json)?;
                let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                    .map_err(|e| {
                        Error::SerializationError(serde_json::Error::io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Invalid created_at timestamp: {}", e),
                        )))
                    })?
                    .with_timezone(&chrono::Utc);
                let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
                    .map_err(|e| {
                        Error::SerializationError(serde_json::Error::io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Invalid updated_at timestamp: {}", e),
                        )))
                    })?
                    .with_timezone(&chrono::Utc);

                Ok(Some(Session {
                    id: SessionId::new(id_str),
                    system_prompt,
                    metadata,
                    created_at,
                    updated_at,
                }))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &SessionId) -> Result<()> {
        sqlx::query("DELETE FROM sessions WHERE id = ?1")
            .bind(id.as_str())
            .execute(&self.pool)
            .await
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to delete session: {}", e),
                ))
            })?;

        Ok(())
    }

    async fn list_ids(&self) -> Result<Vec<SessionId>> {
        let rows = sqlx::query("SELECT id FROM sessions ORDER BY updated_at DESC")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to list sessions: {}", e),
                ))
            })?;

        let ids = rows
            .into_iter()
            .map(|row| {
                let id_str: String = row.get("id");
                SessionId::new(id_str)
            })
            .collect();

        Ok(ids)
    }

    async fn clear(&self) -> Result<()> {
        sqlx::query("DELETE FROM sessions")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to clear sessions: {}", e),
                ))
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_storage() -> Result<(SqliteStorage, TempDir)> {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_sessions.db");
        let storage = SqliteStorage::new(db_path).await?;
        Ok((storage, temp_dir))
    }

    #[tokio::test]
    async fn test_sqlite_storage_save_and_load() -> Result<()> {
        let (storage, _temp_dir) = create_test_storage().await?;

        let session =
            Session::new(SessionId::new("test-session-1")).with_system_prompt("Test prompt");

        // Save session
        storage.save(&session).await?;

        // Load session
        let loaded = storage.load(&session.id).await?;
        assert!(loaded.is_some());

        let loaded_session = loaded.unwrap();
        assert_eq!(loaded_session.id, session.id);
        assert_eq!(
            loaded_session.system_prompt,
            Some("Test prompt".to_string())
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_storage_update() -> Result<()> {
        let (storage, _temp_dir) = create_test_storage().await?;

        let mut session = Session::new(SessionId::new("test-session-2"));
        storage.save(&session).await?;

        // Update session
        session.system_prompt = Some("Updated prompt".to_string());
        session
            .metadata
            .insert("key".to_string(), serde_json::json!("value"));
        storage.save(&session).await?;

        // Load and verify update
        let loaded = storage.load(&session.id).await?.unwrap();
        assert_eq!(loaded.system_prompt, Some("Updated prompt".to_string()));
        assert_eq!(
            loaded.metadata.get("key"),
            Some(&serde_json::json!("value"))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_storage_delete() -> Result<()> {
        let (storage, _temp_dir) = create_test_storage().await?;

        let session = Session::new(SessionId::new("test-session-3"));
        storage.save(&session).await?;

        // Verify it exists
        assert!(storage.load(&session.id).await?.is_some());

        // Delete it
        storage.delete(&session.id).await?;

        // Verify it's gone
        assert!(storage.load(&session.id).await?.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_storage_list_and_clear() -> Result<()> {
        let (storage, _temp_dir) = create_test_storage().await?;

        // Create multiple sessions
        for i in 0..3 {
            let session = Session::new(SessionId::new(format!("test-session-{}", i)));
            storage.save(&session).await?;
        }

        // List sessions
        let ids = storage.list_ids().await?;
        assert_eq!(ids.len(), 3);

        // Clear all
        storage.clear().await?;

        // Verify empty
        let ids = storage.list_ids().await?;
        assert!(ids.is_empty());

        Ok(())
    }
}
