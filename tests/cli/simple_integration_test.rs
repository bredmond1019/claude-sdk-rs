//! Simple integration tests between modules
//!
//! These tests validate basic integration without complex concurrency

use claude_sdk_rs_interactive::{
    analytics::{AnalyticsConfig, AnalyticsEngine},
    cost::{CostEntry, CostFilter, CostTracker},
    history::{HistoryEntry, HistorySearch, HistoryStore},
    session::{SessionId, SessionManager},
};
use std::sync::Arc;
use tempfile::tempdir;
use tokio::sync::RwLock;
use uuid::Uuid;

#[tokio::test]
async fn test_basic_module_integration() {
    let temp_dir = tempdir().unwrap();

    // Initialize modules
    let session_manager = SessionManager::new();
    let cost_tracker = Arc::new(RwLock::new(
        CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
    ));
    let history_store = Arc::new(RwLock::new(
        HistoryStore::new(temp_dir.path().join("history.json")).unwrap(),
    ));

    // Create a session
    let session = session_manager
        .create_session("test-session".to_string(), None)
        .await
        .unwrap();

    // Record some cost data
    {
        let mut tracker = cost_tracker.write().await;
        for i in 0..5 {
            tracker
                .record_cost(CostEntry::new(
                    session.id,
                    format!("command_{}", i),
                    0.01 * (i as f64 + 1.0),
                    100 + i * 10,
                    200 + i * 20,
                    (1000 + i * 500) as u64,
                    "claude-3-opus".to_string(),
                ))
                .await
                .unwrap();
        }
    }

    // Record some history
    {
        let mut store = history_store.write().await;
        for i in 0..5 {
            let mut entry = HistoryEntry::new(
                session.id,
                format!("command_{}", i),
                vec![format!("arg_{}", i)],
                format!("Output {}", i),
                true,
                (1000 + i * 500) as u64,
            );
            entry.cost_usd = Some(0.01 * (i as f64 + 1.0));
            entry.input_tokens = Some((100 + i * 10) as u32);
            entry.output_tokens = Some((200 + i * 20) as u32);
            entry.model = Some("claude-3-opus".to_string());
            store.store_entry(entry).await.unwrap();
        }
    }

    // Verify data consistency
    let cost_summary = {
        let tracker = cost_tracker.read().await;
        tracker.get_session_summary(session.id).await.unwrap()
    };

    let history_stats = {
        let store = history_store.read().await;
        store.get_stats(None).await.unwrap()
    };

    assert_eq!(cost_summary.command_count, 5);
    assert_eq!(history_stats.total_entries, 5);
    assert!((cost_summary.total_cost - 0.15).abs() < 0.0001); // Sum of 0.01, 0.02, 0.03, 0.04, 0.05
}

#[tokio::test]
async fn test_analytics_integration() {
    let temp_dir = tempdir().unwrap();

    // Initialize modules
    let cost_tracker = Arc::new(RwLock::new(
        CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
    ));
    let history_store = Arc::new(RwLock::new(
        HistoryStore::new(temp_dir.path().join("history.json")).unwrap(),
    ));

    let config = AnalyticsConfig::default();
    let analytics_engine = AnalyticsEngine::new(
        Arc::clone(&cost_tracker),
        Arc::clone(&history_store),
        config,
    );

    // Add some test data
    let session_id = Uuid::new_v4();
    {
        let mut tracker = cost_tracker.write().await;
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "analyze".to_string(),
                0.05,
                500,
                1000,
                2000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();
    }

    {
        let mut store = history_store.write().await;
        let mut entry = HistoryEntry::new(
            session_id,
            "analyze".to_string(),
            vec!["file.rs".to_string()],
            "Analysis complete".to_string(),
            true,
            2000,
        );
        entry.cost_usd = Some(0.05);
        entry.input_tokens = Some(500);
        entry.output_tokens = Some(1000);
        entry.model = Some("claude-3-opus".to_string());
        store.store_entry(entry).await.unwrap();
    }

    // Generate analytics
    let summary = analytics_engine.generate_summary(7).await.unwrap();

    assert_eq!(summary.cost_summary.command_count, 1);
    assert_eq!(summary.history_stats.total_entries, 1);
    assert!((summary.cost_summary.total_cost - 0.05).abs() < 0.0001);
    assert_eq!(summary.history_stats.successful_commands, 1);
}

#[tokio::test]
async fn test_session_workflow() {
    let temp_dir = tempdir().unwrap();
    let session_manager = SessionManager::new();
    let cost_tracker = Arc::new(RwLock::new(
        CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
    ));

    // Create multiple sessions
    let session1 = session_manager
        .create_session("session-1".to_string(), None)
        .await
        .unwrap();
    let session2 = session_manager
        .create_session("session-2".to_string(), None)
        .await
        .unwrap();

    // Record costs for different sessions
    {
        let mut tracker = cost_tracker.write().await;

        // Session 1 costs
        tracker
            .record_cost(CostEntry::new(
                session1.id,
                "cmd1".to_string(),
                0.02,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        // Session 2 costs
        tracker
            .record_cost(CostEntry::new(
                session2.id,
                "cmd2".to_string(),
                0.03,
                150,
                250,
                1500,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();
    }

    // Verify session-specific data
    let session1_summary = {
        let tracker = cost_tracker.read().await;
        tracker.get_session_summary(session1.id).await.unwrap()
    };

    let session2_summary = {
        let tracker = cost_tracker.read().await;
        tracker.get_session_summary(session2.id).await.unwrap()
    };

    assert_eq!(session1_summary.command_count, 1);
    assert!((session1_summary.total_cost - 0.02).abs() < 0.0001);

    assert_eq!(session2_summary.command_count, 1);
    assert!((session2_summary.total_cost - 0.03).abs() < 0.0001);

    // List all sessions
    let sessions = session_manager.list_sessions().await.unwrap();
    assert!(sessions.len() >= 2);
}
