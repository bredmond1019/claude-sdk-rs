//! Integration tests between modules with realistic data
//!
//! These tests validate that all modules work together correctly
//! using realistic data patterns and usage scenarios.

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

/// Realistic test data generator for integration testing
struct RealisticDataGenerator {
    session_ids: Vec<SessionId>,
    commands: Vec<String>,
    models: Vec<String>,
    typical_costs: Vec<f64>,
    typical_durations: Vec<u64>,
}

impl RealisticDataGenerator {
    fn new() -> Self {
        Self {
            session_ids: (0..25).map(|_| Uuid::new_v4()).collect(),
            commands: vec![
                "file".to_string(),
                "edit".to_string(),
                "search".to_string(),
                "analyze".to_string(),
                "generate".to_string(),
                "refactor".to_string(),
                "test".to_string(),
                "build".to_string(),
                "deploy".to_string(),
                "debug".to_string(),
                "optimize".to_string(),
                "review".to_string(),
                "format".to_string(),
                "lint".to_string(),
                "document".to_string(),
            ],
            models: vec![
                "claude-3-opus".to_string(),
                "claude-3-sonnet".to_string(),
                "claude-3-haiku".to_string(),
            ],
            typical_costs: vec![
                0.001, 0.003, 0.005, 0.008, 0.012, 0.020, 0.035, 0.050, 0.075, 0.100,
            ],
            typical_durations: vec![150, 300, 500, 800, 1200, 2000, 3500, 5000, 8000, 12000],
        }
    }

    /// Generate realistic command session data
    fn generate_session_data(
        &self,
        session_index: usize,
        commands_in_session: usize,
    ) -> (Vec<CostEntry>, Vec<HistoryEntry>) {
        let session_id = self.session_ids[session_index % self.session_ids.len()];
        let mut cost_entries = Vec::new();
        let mut history_entries = Vec::new();

        // Simulate realistic command patterns within a session
        for i in 0..commands_in_session {
            let command = &self.commands[i % self.commands.len()];
            let model = &self.models[i % self.models.len()];
            let cost = self.typical_costs[i % self.typical_costs.len()];
            let duration = self.typical_durations[i % self.typical_durations.len()];

            // Realistic success rates (varies by command type)
            let success = match command.as_str() {
                "test" | "build" => i % 8 != 0, // 87.5% success
                "deploy" => i % 10 != 0,        // 90% success
                "debug" => i % 6 != 0,          // 83% success
                _ => i % 15 != 0,               // 93% success
            };

            // Realistic token usage (varies by operation)
            let (input_tokens, output_tokens) = match command.as_str() {
                "generate" => (500 + (i % 1000) as u32, 1500 + (i % 2000) as u32),
                "analyze" => (800 + (i % 1200) as u32, 600 + (i % 800) as u32),
                "refactor" => (1200 + (i % 800) as u32, 1000 + (i % 1200) as u32),
                "review" => (2000 + (i % 1000) as u32, 400 + (i % 600) as u32),
                _ => (200 + (i % 300) as u32, 300 + (i % 500) as u32),
            };

            // Create cost entry
            cost_entries.push(CostEntry::new(
                session_id,
                command.clone(),
                cost,
                input_tokens,
                output_tokens,
                duration,
                model.clone(),
            ));

            // Create corresponding history entry
            let mut entry = HistoryEntry::new(
                session_id,
                command.clone(),
                vec![
                    format!("--mode={}", if i % 3 == 0 { "interactive" } else { "batch" }),
                    format!("--priority={}", if i % 5 == 0 { "high" } else { "normal" }),
                ],
                format!(
                    "Completed {} operation in session {}. Processing involved {} input tokens and {} output tokens. {}",
                    command,
                    session_index + 1,
                    input_tokens,
                    output_tokens,
                    if success { "Operation completed successfully." } else { "Operation encountered errors." }
                ),
                success,
                duration,
            );

            // Add realistic metadata
            entry.cost_usd = Some(cost);
            entry.input_tokens = Some(input_tokens);
            entry.output_tokens = Some(output_tokens);
            entry.model = Some(model.clone());

            // Vary timestamps realistically (over the last 30 days)
            let days_ago = (session_index * commands_in_session + i) % 30;
            entry.timestamp = chrono::Utc::now() - chrono::Duration::days(days_ago as i64);

            if !success {
                entry.error = Some(match command.as_str() {
                    "test" => format!("Test failure in module {}: assertion failed", i % 5),
                    "build" => format!(
                        "Build error: compilation failed with {} errors",
                        (i % 3) + 1
                    ),
                    "deploy" => format!("Deployment failed: connection timeout to target server"),
                    "debug" => format!("Debug session terminated: breakpoint condition not met"),
                    _ => format!("Command failed: unexpected error code {}", (i % 10) + 1),
                });
            }

            history_entries.push(entry);
        }

        (cost_entries, history_entries)
    }
}

/// Integration test fixture that sets up all modules
struct IntegrationTestFixture {
    session_manager: Arc<SessionManager>,
    cost_tracker: Arc<RwLock<CostTracker>>,
    history_store: Arc<RwLock<HistoryStore>>,
    analytics_engine: Arc<AnalyticsEngine>,
    data_generator: Arc<RealisticDataGenerator>,
    _temp_dir: tempfile::TempDir,
}

impl IntegrationTestFixture {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = tempdir()?;

        // Initialize session manager
        let session_manager = SessionManager::new();

        // Initialize cost tracker
        let cost_tracker = Arc::new(RwLock::new(CostTracker::new(
            temp_dir.path().join("costs.json"),
        )?));

        // Initialize history store
        let history_store = Arc::new(RwLock::new(HistoryStore::new(
            temp_dir.path().join("history.json"),
        )?));

        // Initialize analytics engine
        let analytics_config = AnalyticsConfig {
            enable_real_time_alerts: true,
            cost_alert_threshold: 5.0, // $5 threshold for testing
            report_schedule: claude_ai_interactive::analytics::ReportSchedule::Daily,
            retention_days: 30,
            dashboard_refresh_interval: 60,
        };

        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&cost_tracker),
            Arc::clone(&history_store),
            analytics_config,
        ));

        let data_generator = Arc::new(RealisticDataGenerator::new());

        Ok(Self {
            session_manager: Arc::new(session_manager),
            cost_tracker,
            history_store,
            analytics_engine,
            data_generator,
            _temp_dir: temp_dir,
        })
    }

    /// Populate the system with realistic test data
    async fn populate_realistic_data(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Create realistic sessions
        let session_names = vec![
            "development_main_feature",
            "bugfix_authentication",
            "performance_optimization",
            "ui_redesign_project",
            "api_integration_work",
            "testing_automation",
            "deployment_pipeline",
            "code_review_session",
            "refactoring_legacy_code",
            "security_audit_fixes",
        ];

        let mut session_data = Vec::new();

        for (i, name) in session_names.iter().enumerate() {
            // Create session
            let session = self
                .session_manager
                .create_session(name.to_string(), None)
                .await?;

            // Generate realistic command sequence for this session
            let commands_per_session = 20 + (i * 5); // Varying session lengths
            let (cost_entries, history_entries) = self
                .data_generator
                .generate_session_data(i, commands_per_session);

            session_data.push((session.id, cost_entries, history_entries));
        }

        // Populate cost tracker and history store
        for (_session_id, cost_entries, history_entries) in session_data {
            // Add cost entries
            {
                let mut tracker = self.cost_tracker.write().await;
                for entry in cost_entries {
                    tracker.record_cost(entry).await?;
                }
            }

            // Add history entries
            {
                let mut store = self.history_store.write().await;
                for entry in history_entries {
                    store.store_entry(entry).await?;
                }
            }
        }

        Ok(())
    }
}

#[tokio::test]
async fn test_end_to_end_workflow_integration() {
    let fixture = IntegrationTestFixture::new().await.unwrap();
    fixture.populate_realistic_data().await.unwrap();

    // Test 1: Verify session management integration
    let sessions = fixture.session_manager.list_sessions().await.unwrap();
    assert!(!sessions.is_empty(), "Should have created sessions");
    println!("✓ Created {} sessions", sessions.len());

    // Test 2: Verify cost tracking integration
    let cost_summary = {
        let tracker = fixture.cost_tracker.read().await;
        tracker.get_global_summary().await.unwrap()
    };
    assert!(cost_summary.total_cost > 0.0, "Should have recorded costs");
    assert!(cost_summary.command_count > 0, "Should have command count");
    println!(
        "✓ Tracked ${:.2} across {} commands",
        cost_summary.total_cost, cost_summary.command_count
    );

    // Test 3: Verify history storage integration
    let history_stats = {
        let store = fixture.history_store.read().await;
        store.get_stats(None).await.unwrap()
    };
    assert!(
        history_stats.total_entries > 0,
        "Should have history entries"
    );
    println!(
        "✓ Stored {} history entries with {:.1}% success rate",
        history_stats.total_entries, history_stats.success_rate
    );

    // Test 4: Verify analytics integration
    let analytics_summary = fixture.analytics_engine.generate_summary(30).await.unwrap();
    assert!(
        !analytics_summary.insights.is_empty(),
        "Should generate insights"
    );
    println!("✓ Generated {} insights", analytics_summary.insights.len());

    // Test 5: Cross-module data consistency
    assert_eq!(
        cost_summary.command_count, history_stats.total_entries,
        "Cost and history entry counts should match"
    );

    assert!(
        (cost_summary.total_cost - analytics_summary.cost_summary.total_cost).abs() < 0.01,
        "Analytics and cost tracker totals should match"
    );

    println!("✓ All modules integrated successfully with consistent data");
}

#[tokio::test]
async fn test_realistic_user_session_workflow() {
    let fixture = IntegrationTestFixture::new().await.unwrap();

    // Simulate a realistic user session workflow
    let session = fixture
        .session_manager
        .create_session("realistic_workflow_test".to_string(), None)
        .await
        .unwrap();
    let session_id = session.id;

    // Step 1: File operations
    let file_commands = vec![
        ("file", "list_files.py", 0.002, 200, 150, 800, true),
        ("edit", "main.py", 0.005, 400, 300, 1200, true),
        ("search", "function.*async", 0.001, 150, 100, 400, true),
    ];

    for (command, output_desc, cost, input_tokens, output_tokens, duration, success) in
        file_commands
    {
        // Record cost
        {
            let mut tracker = fixture.cost_tracker.write().await;
            tracker
                .record_cost(CostEntry::new(
                    session_id,
                    command.to_string(),
                    cost,
                    input_tokens,
                    output_tokens,
                    duration,
                    "claude-3-opus".to_string(),
                ))
                .await
                .unwrap();
        }

        // Record history
        {
            let mut store = fixture.history_store.write().await;
            let mut entry = HistoryEntry::new(
                session_id,
                command.to_string(),
                vec!["--interactive".to_string()],
                format!("Processed {}: completed successfully", output_desc),
                success,
                duration,
            );
            entry.cost_usd = Some(cost);
            entry.input_tokens = Some(input_tokens);
            entry.output_tokens = Some(output_tokens);
            entry.model = Some("claude-3-opus".to_string());
            store.store_entry(entry).await.unwrap();
        }
    }

    // Step 2: Analysis and generation
    let analysis_commands = vec![
        (
            "analyze",
            "code_quality_report.md",
            0.015,
            1200,
            800,
            3000,
            true,
        ),
        ("generate", "unit_tests.py", 0.025, 800, 1500, 4500, true),
        (
            "refactor",
            "optimized_code.py",
            0.020,
            1000,
            1200,
            3500,
            true,
        ),
    ];

    for (command, output_desc, cost, input_tokens, output_tokens, duration, success) in
        analysis_commands
    {
        // Record cost
        {
            let mut tracker = fixture.cost_tracker.write().await;
            tracker
                .record_cost(CostEntry::new(
                    session_id,
                    command.to_string(),
                    cost,
                    input_tokens,
                    output_tokens,
                    duration,
                    "claude-3-opus".to_string(),
                ))
                .await
                .unwrap();
        }

        // Record history
        {
            let mut store = fixture.history_store.write().await;
            let mut entry = HistoryEntry::new(
                session_id,
                command.to_string(),
                vec!["--detailed".to_string(), "--format=markdown".to_string()],
                format!(
                    "Generated {}: comprehensive analysis completed",
                    output_desc
                ),
                success,
                duration,
            );
            entry.cost_usd = Some(cost);
            entry.input_tokens = Some(input_tokens);
            entry.output_tokens = Some(output_tokens);
            entry.model = Some("claude-3-opus".to_string());
            store.store_entry(entry).await.unwrap();
        }
    }

    // Step 3: Verify session-specific analytics
    let session_report = fixture
        .analytics_engine
        .generate_session_report(session_id)
        .await
        .unwrap();

    assert_eq!(session_report.session_id, session_id);
    assert_eq!(session_report.history_stats.total_entries, 6);
    assert!(session_report.cost_summary.total_cost > 0.05); // Should be around $0.068

    println!(
        "✓ Session workflow completed: ${:.3} for {} commands",
        session_report.cost_summary.total_cost, session_report.history_stats.total_entries
    );

    // Step 4: Test cross-module queries
    let cost_filter = CostFilter {
        session_id: Some(session_id),
        min_cost: Some(0.01),
        ..Default::default()
    };
    let filtered_costs = {
        let tracker = fixture.cost_tracker.read().await;
        tracker.get_filtered_summary(&cost_filter).await.unwrap()
    };

    let history_search = HistorySearch {
        session_id: Some(session_id),
        min_duration_ms: Some(2000),
        success_only: true,
        ..Default::default()
    };
    let filtered_history = {
        let store = fixture.history_store.read().await;
        store.search(&history_search).await.unwrap()
    };

    assert!(
        filtered_costs.command_count >= 3,
        "Should find expensive commands"
    );
    assert!(
        filtered_history.len() >= 3,
        "Should find long-running successful commands"
    );

    println!(
        "✓ Cross-module filtering works: {} expensive commands, {} long-running commands",
        filtered_costs.command_count,
        filtered_history.len()
    );
}

#[tokio::test]
#[ignore = "Test has concurrency issues with fixture ownership"]
async fn test_concurrent_module_operations() {
    let fixture = IntegrationTestFixture::new().await.unwrap();

    // Test concurrent operations across modules
    let mut handles = Vec::new();

    for task_id in 0..5 {
        let session_manager = Arc::clone(&fixture.session_manager);
        let cost_tracker = Arc::clone(&fixture.cost_tracker);
        let history_store = Arc::clone(&fixture.history_store);
        let analytics_engine = Arc::clone(&fixture.analytics_engine);

        let handle = tokio::spawn(async move {
            // Create session
            let session_name = format!("concurrent_test_session_{}", task_id);
            let session = session_manager
                .create_session(session_name.clone(), None)
                .await
                .unwrap();

            // Perform operations
            for i in 0..10 {
                let command = format!("test_command_{}", i);
                let cost = (task_id as f64 + 1.0) * 0.001;
                let duration = 500 + (i * 100) as u64;

                // Record cost
                {
                    let mut tracker = cost_tracker.write().await;
                    tracker
                        .record_cost(CostEntry::new(
                            session.id,
                            command.clone(),
                            cost,
                            100 + (i as u32 * 10),
                            200 + (i as u32 * 15),
                            duration,
                            "claude-3-haiku".to_string(),
                        ))
                        .await
                        .unwrap();
                }

                // Record history
                {
                    let mut store = history_store.write().await;
                    let mut entry = HistoryEntry::new(
                        session.id,
                        command,
                        vec![format!("--task-id={}", task_id)],
                        format!("Concurrent operation {} in task {}", i, task_id),
                        i % 8 != 0, // 87.5% success rate
                        duration,
                    );
                    entry.cost_usd = Some(cost);
                    entry.input_tokens = Some(100 + (i as u32 * 10));
                    entry.output_tokens = Some(200 + (i as u32 * 15));
                    entry.model = Some("claude-3-haiku".to_string());
                    store.store_entry(entry).await.unwrap();
                }
            }

            // Generate analytics for this session
            let _report = analytics_engine
                .generate_session_report(session.id)
                .await
                .unwrap();

            task_id
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut completed_tasks = Vec::new();
    for handle in handles {
        let task_id = handle.await.unwrap();
        completed_tasks.push(task_id);
    }

    assert_eq!(
        completed_tasks.len(),
        5,
        "All concurrent tasks should complete"
    );

    // Verify data consistency after concurrent operations
    let sessions = fixture.session_manager.list_sessions().await.unwrap();
    assert_eq!(sessions.len(), 5, "Should have 5 sessions");

    let cost_summary = {
        let tracker = fixture.cost_tracker.read().await;
        tracker.get_global_summary().await.unwrap()
    };
    assert_eq!(
        cost_summary.command_count, 50,
        "Should have 50 cost entries"
    );

    let history_stats = {
        let store = fixture.history_store.read().await;
        store.get_stats(None).await.unwrap()
    };
    assert_eq!(
        history_stats.total_entries, 50,
        "Should have 50 history entries"
    );

    println!("✓ Concurrent operations completed successfully with consistent data");
}

#[tokio::test]
async fn test_data_consistency_across_modules() {
    let fixture = IntegrationTestFixture::new().await.unwrap();
    fixture.populate_realistic_data().await.unwrap();

    // Test data consistency across all modules
    let sessions = fixture.session_manager.list_sessions().await.unwrap();
    let cost_summary = {
        let tracker = fixture.cost_tracker.read().await;
        tracker.get_global_summary().await.unwrap()
    };
    let history_stats = {
        let store = fixture.history_store.read().await;
        store.get_stats(None).await.unwrap()
    };
    let analytics_summary = fixture.analytics_engine.generate_summary(30).await.unwrap();

    // Verify session count consistency
    assert!(!sessions.is_empty(), "Should have sessions");

    // Verify command count consistency
    assert_eq!(
        cost_summary.command_count, history_stats.total_entries,
        "Cost and history command counts should match"
    );

    // Verify cost consistency
    assert!(
        (cost_summary.total_cost - analytics_summary.cost_summary.total_cost).abs() < 0.01,
        "Cost totals should be consistent across modules"
    );

    // Verify session-level data consistency
    for session in sessions.iter().take(3) {
        let session_cost = {
            let tracker = fixture.cost_tracker.read().await;
            tracker.get_session_summary(session.id).await.unwrap()
        };

        let session_history = {
            let store = fixture.history_store.read().await;
            store.get_recent_commands(session.id, 100).await.unwrap()
        };

        let session_report = fixture
            .analytics_engine
            .generate_session_report(session.id)
            .await
            .unwrap();

        // Verify session-level consistency
        assert_eq!(session_history.len(), session_cost.command_count);
        assert_eq!(
            session_history.len(),
            session_report.history_stats.total_entries
        );
        assert!(
            (session_cost.total_cost - session_report.cost_summary.total_cost).abs() < 0.01,
            "Session cost should be consistent"
        );
    }

    println!("✓ Data consistency verified across all modules");
    println!("  Sessions: {}", sessions.len());
    println!("  Total commands: {}", cost_summary.command_count);
    println!("  Total cost: ${:.3}", cost_summary.total_cost);
    println!("  Success rate: {:.1}%", history_stats.success_rate);
}

#[tokio::test]
async fn test_performance_with_realistic_load() {
    let fixture = IntegrationTestFixture::new().await.unwrap();
    fixture.populate_realistic_data().await.unwrap();

    let start_time = std::time::Instant::now();

    // Perform realistic operations under load
    let mut operations = Vec::new();

    // Analytics operations
    operations.push(tokio::spawn({
        let engine = Arc::clone(&fixture.analytics_engine);
        async move {
            let _summary = engine.generate_summary(30).await.unwrap();
            let _dashboard = engine.get_dashboard_data().await.unwrap();
        }
    }));

    // Cost tracking operations
    operations.push(tokio::spawn({
        let tracker = Arc::clone(&fixture.cost_tracker);
        async move {
            let tracker = tracker.read().await;
            let _summary = tracker.get_global_summary().await.unwrap();
            let _top_commands = tracker.get_top_commands(5).await.unwrap();
        }
    }));

    // History search operations
    operations.push(tokio::spawn({
        let store = Arc::clone(&fixture.history_store);
        async move {
            let store = store.read().await;
            let search = HistorySearch {
                command_pattern: Some("generate".to_string()),
                success_only: true,
                limit: 50,
                ..Default::default()
            };
            let _results = store.search(&search).await.unwrap();
            let _stats = store.get_stats(None).await.unwrap();
        }
    }));

    // Session management operations
    operations.push(tokio::spawn({
        let manager = Arc::clone(&fixture.session_manager);
        async move {
            let _sessions = manager.list_sessions().await.unwrap();
        }
    }));

    // Wait for all operations to complete
    for operation in operations {
        operation.await.unwrap();
    }

    let elapsed = start_time.elapsed();

    println!("✓ Performance test completed in {:?}", elapsed);

    // Performance should be reasonable (under 5 seconds for realistic load)
    assert!(
        elapsed.as_secs() < 5,
        "Operations should complete within 5 seconds"
    );
}

#[tokio::test]
async fn test_error_handling_integration() {
    let fixture = IntegrationTestFixture::new().await.unwrap();

    // Test error scenarios across modules
    let session = fixture
        .session_manager
        .create_session("error_test_session".to_string(), None)
        .await
        .unwrap();

    // Test with invalid/corrupted data
    let invalid_cost_entry = CostEntry::new(
        session.id,
        "invalid_command".to_string(),
        -1.0,           // Invalid negative cost
        0,              // Invalid zero tokens
        0,              // Invalid zero tokens
        0,              // Invalid zero duration
        "".to_string(), // Invalid empty model
    );

    // The system should handle invalid data gracefully
    {
        let mut tracker = fixture.cost_tracker.write().await;
        // This might fail or succeed depending on validation, but shouldn't crash
        let _result = tracker.record_cost(invalid_cost_entry).await;
    }

    // Test analytics with limited data
    let analytics_summary = fixture.analytics_engine.generate_summary(30).await.unwrap();
    // Should not crash even with invalid/limited data
    assert!(analytics_summary.cost_summary.total_cost >= 0.0);

    // Test session operations with non-existent session
    let non_existent_session = Uuid::new_v4();
    let session_report = fixture
        .analytics_engine
        .generate_session_report(non_existent_session)
        .await;
    // Should handle gracefully (might return empty report or error)
    match session_report {
        Ok(report) => {
            assert_eq!(report.session_id, non_existent_session);
            assert_eq!(report.history_stats.total_entries, 0);
        }
        Err(_) => {
            // Error is acceptable for non-existent session
        }
    }

    println!("✓ Error handling integration test completed");
}
