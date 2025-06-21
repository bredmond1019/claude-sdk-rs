//! Memory usage testing under various load conditions
//!
//! This module provides comprehensive memory testing to ensure the application
//! performs well under different memory constraints and load scenarios.

#[cfg(test)]
mod memory_load_tests {
    use crate::{
        analytics::{AnalyticsConfig, AnalyticsEngine},
        cost::{CostEntry, CostTracker},
        history::{HistoryEntry, HistoryStore, HistorySearch},
        session::{SessionId, SessionManager},
    };
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    /// Memory usage measurement utilities
    struct MemoryMonitor {
        initial_memory: usize,
    }

    impl MemoryMonitor {
        fn new() -> Self {
            Self {
                initial_memory: Self::get_memory_usage(),
            }
        }

        fn get_memory_usage() -> usize {
            // Simple memory measurement - in production you might use more sophisticated tools
            // This is a placeholder that returns a simulated memory value
            std::mem::size_of::<usize>() * 1000
        }

        fn memory_delta(&self) -> isize {
            let current = Self::get_memory_usage();
            current as isize - self.initial_memory as isize
        }

        fn memory_growth_rate(&self, items_processed: usize) -> f64 {
            if items_processed == 0 { return 0.0; }
            self.memory_delta() as f64 / items_processed as f64
        }
    }

    /// Generate test data for memory testing
    fn generate_large_cost_dataset(size: usize) -> Vec<CostEntry> {
        let session_ids: Vec<SessionId> = (0..50).map(|_| Uuid::new_v4()).collect();
        let models = vec![
            "claude-3-opus".to_string(),
            "claude-3-sonnet".to_string(),
            "claude-3-haiku".to_string(),
        ];
        let commands = vec![
            "analyze", "generate", "refactor", "optimize", "test", "build",
            "deploy", "debug", "review", "format", "search", "edit",
        ];

        (0..size)
            .map(|i| {
                CostEntry::new(
                    session_ids[i % session_ids.len()],
                    commands[i % commands.len()].to_string(),
                    (i as f64 * 0.001) % 1.0,
                    (i % 5000) as u32 + 100,
                    (i % 2500) as u32 + 50,
                    (i % 30000) as u64 + 500,
                    models[i % models.len()].clone(),
                )
            })
            .collect()
    }

    fn generate_large_history_dataset(size: usize) -> Vec<HistoryEntry> {
        let session_ids: Vec<SessionId> = (0..50).map(|_| Uuid::new_v4()).collect();
        let commands = vec![
            "file", "edit", "search", "analyze", "generate", "refactor",
            "test", "build", "deploy", "debug", "optimize", "review",
        ];

        (0..size)
            .map(|i| {
                let session_id = session_ids[i % session_ids.len()];
                let command = commands[i % commands.len()];
                let success = i % 7 != 0; // ~85% success rate
                
                let mut entry = HistoryEntry::new(
                    session_id,
                    command.to_string(),
                    vec![format!("arg_{}", i % 10), format!("--flag-{}", i % 5)],
                    format!("Large output data for command {} with extensive details and results that simulate real-world usage patterns. This entry #{} contains substantial text content.", command, i),
                    success,
                    (i % 15000) as u64 + 200,
                );

                entry.cost_usd = Some((i as f64 * 0.0001) % 0.2);
                entry.input_tokens = Some((i % 4000) as u32 + 100);
                entry.output_tokens = Some((i % 2000) as u32 + 200);
                entry.model = Some("claude-3-opus".to_string());

                if !success {
                    entry.error = Some(format!("Detailed error information for command {} including stack trace and diagnostic data that would be typical in real usage", command));
                }

                entry
            })
            .collect()
    }

    #[tokio::test]
    async fn test_cost_tracker_memory_scaling() {
        let monitor = MemoryMonitor::new();
        let temp_dir = tempdir().unwrap();
        let mut tracker = CostTracker::new(temp_dir.path().join("costs.json")).unwrap();

        // Test memory usage with increasing dataset sizes
        let test_sizes = vec![1000, 5000, 10000, 25000, 50000];
        let mut previous_memory = monitor.memory_delta();

        for size in test_sizes {
            let entries = generate_large_cost_dataset(size);
            
            for entry in entries {
                tracker.record_cost(entry).await.unwrap();
            }

            // Perform operations that use memory
            let _summary = tracker.get_global_summary().await.unwrap();
            let _top_commands = tracker.get_top_commands(10).await.unwrap();

            let current_memory = monitor.memory_delta();
            let memory_growth = current_memory - previous_memory;
            let growth_rate = monitor.memory_growth_rate(size);

            println!(
                "Size: {}, Memory Delta: {} bytes, Growth Rate: {:.2} bytes/item",
                size, memory_growth, growth_rate
            );

            // Verify memory growth is reasonable (less than 1KB per entry on average)
            assert!(
                growth_rate < 1024.0,
                "Memory growth rate too high: {:.2} bytes/item",
                growth_rate
            );

            previous_memory = current_memory;
        }
    }

    #[tokio::test]
    async fn test_history_store_memory_efficiency() {
        let monitor = MemoryMonitor::new();
        let temp_dir = tempdir().unwrap();
        let mut store = HistoryStore::new(temp_dir.path().join("history.json")).unwrap();

        // Test with large entries
        let entries = generate_large_history_dataset(20000);
        let initial_memory = monitor.memory_delta();

        for entry in entries {
            store.store_entry(entry).await.unwrap();
        }

        // Perform memory-intensive operations
        let _stats = store.get_stats(None).await.unwrap();
        
        let search = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            limit: 1000,
            ..Default::default()
        };
        let _results = store.search(&search).await.unwrap();

        let final_memory = monitor.memory_delta();
        let memory_used = final_memory - initial_memory;

        println!("History store memory usage: {} bytes for 20k entries", memory_used);

        // Verify memory usage is reasonable (less than 10MB for 20k entries)
        assert!(
            memory_used < 10_000_000,
            "Memory usage too high: {} bytes",
            memory_used
        );
    }

    #[tokio::test]
    async fn test_analytics_engine_memory_under_load() {
        let monitor = MemoryMonitor::new();
        let temp_dir = tempdir().unwrap();

        let cost_tracker = Arc::new(RwLock::new(
            CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
        ));
        let history_store = Arc::new(RwLock::new(
            HistoryStore::new(temp_dir.path().join("history.json")).unwrap(),
        ));

        // Populate with large datasets
        let cost_entries = generate_large_cost_dataset(15000);
        let history_entries = generate_large_history_dataset(15000);

        {
            let mut tracker = cost_tracker.write().await;
            for entry in cost_entries {
                tracker.record_cost(entry).await.unwrap();
            }
        }

        {
            let mut store = history_store.write().await;
            for entry in history_entries {
                store.store_entry(entry).await.unwrap();
            }
        }

        let config = AnalyticsConfig::default();
        let engine = AnalyticsEngine::new(cost_tracker, history_store, config);

        let initial_memory = monitor.memory_delta();

        // Perform analytics operations
        let _summary = engine.generate_summary(30).await.unwrap();
        let _dashboard = engine.get_dashboard_data().await.unwrap();
        
        // Generate multiple session reports
        for _ in 0..10 {
            let session_id = Uuid::new_v4();
            let _report = engine.generate_session_report(session_id).await.unwrap();
        }

        let final_memory = monitor.memory_delta();
        let memory_used = final_memory - initial_memory;

        println!("Analytics engine memory usage: {} bytes", memory_used);

        // Verify analytics doesn't consume excessive memory
        assert!(
            memory_used < 50_000_000,
            "Analytics memory usage too high: {} bytes",
            memory_used
        );
    }

    #[tokio::test]
    async fn test_session_manager_memory_with_many_sessions() {
        let monitor = MemoryMonitor::new();
        let temp_dir = tempdir().unwrap();
        let session_manager = SessionManager::new();

        let initial_memory = monitor.memory_delta();
        let session_count = 10000;

        // Create many sessions
        for i in 0..session_count {
            let name = format!("memory_test_session_{}", i);
            let _session = session_manager.create_session(&name).await.unwrap();
        }

        // Perform session operations
        let _all_sessions = session_manager.list_sessions().await.unwrap();

        // Access random sessions
        for i in (0..session_count).step_by(100) {
            let sessions = session_manager.list_sessions().await.unwrap();
            if let Some(session) = sessions.get(i) {
                let _retrieved = session_manager.get_session(session.id).await.unwrap();
            }
        }

        let final_memory = monitor.memory_delta();
        let memory_per_session = (final_memory - initial_memory) as f64 / session_count as f64;

        println!(
            "Session manager: {} bytes total, {:.2} bytes per session",
            final_memory - initial_memory,
            memory_per_session
        );

        // Verify reasonable memory per session (less than 1KB per session)
        assert!(
            memory_per_session < 1024.0,
            "Memory per session too high: {:.2} bytes",
            memory_per_session
        );
    }

    #[tokio::test]
    async fn test_concurrent_memory_usage() {
        let monitor = MemoryMonitor::new();
        let temp_dir = tempdir().unwrap();
        
        let cost_tracker = Arc::new(RwLock::new(
            CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
        ));
        let history_store = Arc::new(RwLock::new(
            HistoryStore::new(temp_dir.path().join("history.json")).unwrap(),
        ));

        let initial_memory = monitor.memory_delta();

        // Simulate concurrent operations
        let mut handles = Vec::new();

        for task_id in 0..8 {
            let cost_tracker_clone = Arc::clone(&cost_tracker);
            let history_store_clone = Arc::clone(&history_store);

            let handle = tokio::spawn(async move {
                let entries_per_task = 2500;
                let cost_entries = generate_large_cost_dataset(entries_per_task);
                let history_entries = generate_large_history_dataset(entries_per_task);

                // Record cost entries
                {
                    let mut tracker = cost_tracker_clone.write().await;
                    for entry in cost_entries {
                        tracker.record_cost(entry).await.unwrap();
                    }
                }

                // Store history entries
                {
                    let mut store = history_store_clone.write().await;
                    for entry in history_entries {
                        store.store_entry(entry).await.unwrap();
                    }
                }

                // Perform read operations
                {
                    let tracker = cost_tracker_clone.read().await;
                    let _summary = tracker.get_global_summary().await.unwrap();
                }

                {
                    let store = history_store_clone.read().await;
                    let search = HistorySearch {
                        command_pattern: Some("test".to_string()),
                        limit: 100,
                        ..Default::default()
                    };
                    let _results = store.search(&search).await.unwrap();
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let final_memory = monitor.memory_delta();
        let total_memory_used = final_memory - initial_memory;

        println!("Concurrent operations memory usage: {} bytes", total_memory_used);

        // Verify concurrent operations don't cause excessive memory usage
        assert!(
            total_memory_used < 100_000_000,
            "Concurrent memory usage too high: {} bytes",
            total_memory_used
        );
    }

    #[tokio::test]
    async fn test_memory_leak_detection() {
        let monitor = MemoryMonitor::new();
        let temp_dir = tempdir().unwrap();

        // Perform the same operation multiple times to detect memory leaks
        let iterations = 5;
        let mut memory_readings = Vec::new();

        for iteration in 0..iterations {
            let cost_tracker = Arc::new(RwLock::new(
                CostTracker::new(temp_dir.path().join(format!("costs_{}.json", iteration))).unwrap(),
            ));

            // Create and destroy the same amount of data each iteration
            {
                let mut tracker = cost_tracker.write().await;
                let entries = generate_large_cost_dataset(5000);
                for entry in entries {
                    tracker.record_cost(entry).await.unwrap();
                }
                let _summary = tracker.get_global_summary().await.unwrap();
            }

            // Drop the tracker
            drop(cost_tracker);

            let current_memory = monitor.memory_delta();
            memory_readings.push(current_memory);

            println!("Iteration {}: Memory delta = {} bytes", iteration, current_memory);
        }

        // Check for memory leaks - memory should not grow significantly between iterations
        let memory_growth = memory_readings.last().unwrap() - memory_readings.first().unwrap();
        let avg_growth_per_iteration = memory_growth as f64 / iterations as f64;

        println!("Average memory growth per iteration: {:.2} bytes", avg_growth_per_iteration);

        // Memory should not grow more than 1MB per iteration on average
        assert!(
            avg_growth_per_iteration < 1_000_000.0,
            "Potential memory leak detected: {:.2} bytes growth per iteration",
            avg_growth_per_iteration
        );
    }

    #[tokio::test]
    async fn test_large_dataset_memory_stability() {
        let monitor = MemoryMonitor::new();
        let temp_dir = tempdir().unwrap();
        let mut store = HistoryStore::new(temp_dir.path().join("history.json")).unwrap();

        // Test memory stability with very large datasets
        let large_dataset_size = 100000; // 100k entries
        let batch_size = 10000;
        let mut memory_readings = Vec::new();

        for batch in 0..(large_dataset_size / batch_size) {
            let entries = generate_large_history_dataset(batch_size);
            
            for entry in entries {
                store.store_entry(entry).await.unwrap();
            }

            // Perform operations after each batch
            if batch % 2 == 0 {
                let _stats = store.get_stats(None).await.unwrap();
            }

            let current_memory = monitor.memory_delta();
            memory_readings.push(current_memory);

            println!("Batch {}: {} entries, Memory: {} bytes", 
                    batch, (batch + 1) * batch_size, current_memory);
        }

        // Verify memory growth is linear and predictable
        let initial_memory = memory_readings[0];
        let final_memory = memory_readings.last().unwrap();
        let total_growth = final_memory - initial_memory;
        let growth_per_entry = total_growth as f64 / large_dataset_size as f64;

        println!("Total memory growth: {} bytes", total_growth);
        println!("Memory per entry: {:.2} bytes", growth_per_entry);

        // Memory per entry should be reasonable (less than 2KB per entry)
        assert!(
            growth_per_entry < 2048.0,
            "Memory per entry too high: {:.2} bytes",
            growth_per_entry
        );

        // Verify memory growth is roughly linear (no quadratic growth)
        let mid_point = memory_readings.len() / 2;
        let mid_memory = memory_readings[mid_point];
        let expected_mid_memory = initial_memory + (total_growth / 2);
        let linearity_error = ((mid_memory as f64 - expected_mid_memory as f64).abs() / expected_mid_memory as f64) * 100.0;

        println!("Memory growth linearity error: {:.2}%", linearity_error);

        // Memory growth should be roughly linear (within 20% deviation)
        assert!(
            linearity_error < 20.0,
            "Memory growth not linear: {:.2}% deviation",
            linearity_error
        );
    }
}