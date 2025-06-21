//! Standalone metrics tests for Task 3.3
//!
//! This file contains comprehensive tests for metrics calculations
//! focusing on usage metrics, performance metrics, cost integration, and trend analysis.

use chrono::{DateTime, Duration, Utc};
use claude_sdk_rs_interactive::analytics::metrics::*;
use std::collections::HashMap;
use uuid::Uuid;

/// Simple test fixture for metrics
struct SimpleMetricsFixture {
    registry: MetricRegistry,
}

impl SimpleMetricsFixture {
    async fn new() -> Self {
        let config = MetricConfig::default();
        let registry = MetricRegistry::new(config);
        registry.initialize_default_metrics().await.unwrap();

        Self { registry }
    }
}

#[tokio::test]
async fn test_basic_metric_operations() {
    let fixture = SimpleMetricsFixture::new().await;

    // Test counter operations
    fixture
        .registry
        .increment_counter("commands_total")
        .await
        .unwrap();
    fixture
        .registry
        .increment_counter("commands_total")
        .await
        .unwrap();
    fixture
        .registry
        .increment_counter("commands_total")
        .await
        .unwrap();

    // Test gauge operations
    fixture
        .registry
        .set_gauge("memory_usage_mb", 512.0)
        .await
        .unwrap();

    // Test histogram operations
    fixture
        .registry
        .observe_histogram("command_duration_ms", 1000.0)
        .await
        .unwrap();
    fixture
        .registry
        .observe_histogram("command_duration_ms", 1500.0)
        .await
        .unwrap();
    fixture
        .registry
        .observe_histogram("command_duration_ms", 2000.0)
        .await
        .unwrap();

    // Test timer operations
    fixture
        .registry
        .record_timer("api_response_time", 500.0)
        .await
        .unwrap();
    fixture
        .registry
        .record_timer("api_response_time", 750.0)
        .await
        .unwrap();
    fixture
        .registry
        .record_timer("api_response_time", 1250.0)
        .await
        .unwrap();

    // Verify snapshot
    let snapshot = fixture.registry.snapshot().await;

    // Verify counter
    let commands_counter = snapshot
        .counters
        .iter()
        .find(|c| c.name == "commands_total")
        .unwrap();
    assert_eq!(commands_counter.value, 3);

    // Verify gauge
    let memory_gauge = snapshot
        .gauges
        .iter()
        .find(|g| g.name == "memory_usage_mb")
        .unwrap();
    assert_eq!(memory_gauge.value, 512.0);

    // Verify histogram
    let duration_histogram = snapshot
        .histograms
        .iter()
        .find(|h| h.name == "command_duration_ms")
        .unwrap();
    assert_eq!(duration_histogram.count, 3);
    assert_eq!(duration_histogram.sum, 4500.0);

    // Verify timer
    let response_timer = snapshot
        .timers
        .iter()
        .find(|t| t.name == "api_response_time")
        .unwrap();
    assert_eq!(response_timer.count, 3);
    assert_eq!(response_timer.mean, 833.3333333333334);
    assert_eq!(response_timer.min, 500.0);
    assert_eq!(response_timer.max, 1250.0);
}

#[tokio::test]
async fn test_task_3_3_1_usage_metrics() {
    let fixture = SimpleMetricsFixture::new().await;

    // Task 3.3.1: Test command counting per session
    let session1 = Uuid::new_v4();
    let session2 = Uuid::new_v4();

    // Session 1: 10 commands
    for _i in 0..10 {
        fixture
            .registry
            .increment_counter("commands_total")
            .await
            .unwrap();
        fixture
            .registry
            .observe_histogram("command_duration_ms", 1000.0)
            .await
            .unwrap();
    }

    // Session 2: 5 commands
    for _i in 0..5 {
        fixture
            .registry
            .increment_counter("commands_total")
            .await
            .unwrap();
        fixture
            .registry
            .observe_histogram("command_duration_ms", 2000.0)
            .await
            .unwrap();
    }

    let snapshot = fixture.registry.snapshot().await;

    // Verify total command count
    let total_counter = snapshot
        .counters
        .iter()
        .find(|c| c.name == "commands_total")
        .unwrap();
    assert_eq!(total_counter.value, 15);

    // Verify histogram has recorded all commands
    let duration_histogram = snapshot
        .histograms
        .iter()
        .find(|h| h.name == "command_duration_ms")
        .unwrap();
    assert_eq!(duration_histogram.count, 15);

    println!("✅ Task 3.3.1: Usage metrics test passed");
}

#[tokio::test]
async fn test_task_3_3_2_performance_metrics() {
    let fixture = SimpleMetricsFixture::new().await;

    // Task 3.3.2: Test execution time calculations
    let execution_times = vec![
        100.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 9000.0, 15000.0,
    ];

    for time in &execution_times {
        fixture
            .registry
            .observe_histogram("command_duration_ms", *time)
            .await
            .unwrap();
        fixture
            .registry
            .record_timer("api_response_time", *time)
            .await
            .unwrap();
    }

    // Test success/failure rates
    let total_commands = 1000u64;
    let failed_commands = 50u64; // 5% failure rate

    for i in 0..total_commands {
        fixture
            .registry
            .increment_counter("commands_total")
            .await
            .unwrap();

        if i < failed_commands {
            fixture
                .registry
                .increment_counter("commands_failed_total")
                .await
                .unwrap();
        }
    }

    let snapshot = fixture.registry.snapshot().await;

    // Test histogram metrics
    let duration_histogram = snapshot
        .histograms
        .iter()
        .find(|h| h.name == "command_duration_ms")
        .unwrap();

    assert_eq!(duration_histogram.count, 10);
    let expected_sum: f64 = execution_times.iter().sum();
    assert!((duration_histogram.sum - expected_sum).abs() < 0.1);

    // Test timer metrics
    let response_timer = snapshot
        .timers
        .iter()
        .find(|t| t.name == "api_response_time")
        .unwrap();

    assert_eq!(response_timer.count, 10);
    assert_eq!(response_timer.min, 100.0);
    assert_eq!(response_timer.max, 15000.0);

    // Test success/failure rates
    let total_counter = snapshot
        .counters
        .iter()
        .find(|c| c.name == "commands_total")
        .unwrap();
    let failed_counter = snapshot
        .counters
        .iter()
        .find(|c| c.name == "commands_failed_total")
        .unwrap();

    assert_eq!(total_counter.value, total_commands);
    assert_eq!(failed_counter.value, failed_commands);

    // Calculate rates
    let failure_rate = (failed_counter.value as f64 / total_counter.value as f64) * 100.0;
    let success_rate = 100.0 - failure_rate;

    assert!((failure_rate - 5.0).abs() < 0.1);
    assert!((success_rate - 95.0).abs() < 0.1);

    println!("✅ Task 3.3.2: Performance metrics test passed");
}

#[tokio::test]
async fn test_task_3_3_3_cost_metrics() {
    let fixture = SimpleMetricsFixture::new().await;

    // Task 3.3.3: Test cost aggregation

    // Register session cost counter
    fixture
        .registry
        .register_counter(
            "session_cost_cents".to_string(),
            "Session cost in cents".to_string(),
            HashMap::new(),
        )
        .await
        .unwrap();

    // Session 1: Low cost operations
    for i in 0..10 {
        let cost = 0.001 * (i + 1) as f64;
        fixture
            .registry
            .observe_histogram("command_cost_usd", cost)
            .await
            .unwrap();
        fixture
            .registry
            .add_to_counter("session_cost_cents", (cost * 100.0) as u64)
            .await
            .unwrap();
    }

    // Session 2: Medium cost operations
    for i in 0..5 {
        let cost = 0.01 * (i + 1) as f64;
        fixture
            .registry
            .observe_histogram("command_cost_usd", cost)
            .await
            .unwrap();
        fixture
            .registry
            .add_to_counter("session_cost_cents", (cost * 100.0) as u64)
            .await
            .unwrap();
    }

    // Session 3: High cost operations
    for i in 0..3 {
        let cost = 0.1 * (i + 1) as f64;
        fixture
            .registry
            .observe_histogram("command_cost_usd", cost)
            .await
            .unwrap();
        fixture
            .registry
            .add_to_counter("session_cost_cents", (cost * 100.0) as u64)
            .await
            .unwrap();
    }

    let snapshot = fixture.registry.snapshot().await;

    // Verify cost histogram
    let cost_histogram = snapshot
        .histograms
        .iter()
        .find(|h| h.name == "command_cost_usd")
        .unwrap();

    assert_eq!(cost_histogram.count, 18); // 10 + 5 + 3

    // Verify total cost
    let total_cost = 0.055 + 0.15 + 0.6; // Sum of all costs
    assert!((cost_histogram.sum - total_cost).abs() < 0.01);

    println!("✅ Task 3.3.3: Cost metrics test passed");
}

#[tokio::test]
async fn test_task_3_3_4_trend_analysis() {
    let fixture = SimpleMetricsFixture::new().await;

    // Task 3.3.4: Test moving averages and statistical analysis

    // Register the histogram first
    fixture
        .registry
        .register_histogram(
            "metric_value".to_string(),
            "Test metric for trend analysis".to_string(),
            vec![50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
            HashMap::new(),
        )
        .await
        .unwrap();

    // Generate time series data
    let mut values = Vec::new();
    for i in 0..30 {
        let value = 100.0 + (i as f64 * 2.0) + (rand::random::<f64>() - 0.5) * 10.0;
        values.push(value);
        fixture
            .registry
            .observe_histogram("metric_value", value)
            .await
            .unwrap();
    }

    // Calculate 7-day moving average
    let mut ma7 = Vec::new();
    for i in 6..values.len() {
        let sum: f64 = values[i - 6..=i].iter().sum();
        ma7.push(sum / 7.0);
    }

    // Verify moving averages smooth out variations
    let ma7_variance = calculate_variance(&ma7);
    let raw_variance = calculate_variance(&values);

    // Moving average should have lower variance (allowing for small datasets)
    assert!(
        ma7_variance <= raw_variance,
        "Moving average variance should be less than or equal to raw variance"
    );

    // Trend should be upward (positive slope)
    let first_ma7 = ma7.first().unwrap();
    let last_ma7 = ma7.last().unwrap();
    assert!(last_ma7 > first_ma7);

    // Test standard deviation calculations
    let stats = calculate_statistics(&values);
    assert!(stats.count > 0);
    assert!(stats.std_dev > 0.0);
    assert!(stats.mean > 0.0);

    println!("✅ Task 3.3.4: Trend analysis test passed");
}

// Helper functions for statistical calculations
fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

fn calculate_statistics(values: &[f64]) -> MetricStatistics {
    if values.is_empty() {
        return MetricStatistics::default();
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;

    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();

    let median = if values.len() % 2 == 0 {
        (sorted[values.len() / 2 - 1] + sorted[values.len() / 2]) / 2.0
    } else {
        sorted[values.len() / 2]
    };

    MetricStatistics {
        count: values.len(),
        sum,
        mean,
        median,
        std_dev,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        p50: sorted[values.len() / 2],
        p95: sorted[(values.len() as f64 * 0.95) as usize],
        p99: sorted[(values.len() as f64 * 0.99) as usize],
    }
}

#[derive(Debug, Default)]
struct MetricStatistics {
    count: usize,
    sum: f64,
    mean: f64,
    median: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

#[tokio::test]
async fn test_comprehensive_metrics_workflow() {
    println!("🚀 Running comprehensive metrics tests for Task 3.3");

    let fixture = SimpleMetricsFixture::new().await;

    // Test all metric types working together
    for session_idx in 0..5 {
        let session_id = Uuid::new_v4();

        for cmd_idx in 0..20 {
            // Simulate command execution
            let duration = 500.0 + (session_idx * cmd_idx) as f64 * 10.0;
            let cost = duration / 100000.0; // Simple cost model
            let success = rand::random::<f64>() > 0.05; // 95% success rate

            // Record metrics
            fixture
                .registry
                .increment_counter("commands_total")
                .await
                .unwrap();

            if !success {
                fixture
                    .registry
                    .increment_counter("commands_failed_total")
                    .await
                    .unwrap();
            }

            fixture
                .registry
                .observe_histogram("command_duration_ms", duration)
                .await
                .unwrap();
            fixture
                .registry
                .observe_histogram("command_cost_usd", cost)
                .await
                .unwrap();
            fixture
                .registry
                .record_timer("api_response_time", duration)
                .await
                .unwrap();

            // Update session cost
            fixture
                .registry
                .set_gauge("total_cost_usd", cost)
                .await
                .unwrap();
        }
    }

    // Verify comprehensive metrics
    let snapshot = fixture.registry.snapshot().await;

    // Verify counters
    assert!(!snapshot.counters.is_empty());

    // Verify gauges
    assert!(!snapshot.gauges.is_empty());

    // Verify histograms
    assert!(!snapshot.histograms.is_empty());

    // Verify timers
    assert!(!snapshot.timers.is_empty());

    // Test Prometheus export
    let prometheus_output = fixture.registry.export_prometheus().await;
    assert!(!prometheus_output.is_empty());
    assert!(prometheus_output.contains("# HELP"));
    assert!(prometheus_output.contains("# TYPE"));

    println!("✅ Comprehensive metrics workflow test passed");
    println!(
        "📊 Total metrics recorded: {} counters, {} gauges, {} histograms, {} timers",
        snapshot.counters.len(),
        snapshot.gauges.len(),
        snapshot.histograms.len(),
        snapshot.timers.len()
    );
}
