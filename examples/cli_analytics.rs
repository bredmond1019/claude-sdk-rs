//! # CLI Analytics Example
//!
//! This example demonstrates the analytics features of claude-sdk-rs CLI.
//! It shows how to:
//! - Track usage metrics and costs
//! - Generate analytics reports
//! - Monitor session performance
//! - Display usage dashboards
//! - Export analytics data
//!
//! **Required Features**: This example requires the "analytics" feature to be enabled.
//! Run with: `cargo run --features analytics --example cli_analytics`

#[cfg(feature = "analytics")]
use claude_sdk_rs::{Client, Config, StreamFormat};

#[cfg(feature = "analytics")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK CLI Analytics Example ===\n");
    println!("This example demonstrates analytics and usage tracking features.\n");

    // Example 1: Basic usage tracking
    basic_usage_tracking().await?;

    // Example 2: Cost analytics
    cost_analytics().await?;

    // Example 3: Performance metrics
    performance_metrics().await?;

    // Example 4: Session analytics
    session_analytics().await?;

    // Example 5: Analytics dashboard
    analytics_dashboard().await?;

    println!("CLI analytics example completed successfully!");
    Ok(())
}

#[cfg(feature = "analytics")]
async fn basic_usage_tracking() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Usage Tracking");
    println!("   Tracking basic usage metrics for Claude SDK operations\n");

    let client = Client::builder().stream_format(StreamFormat::Json).build();

    println!("   Performing tracked operations:");

    // Simulate various operations for tracking
    let operations = vec![
        ("Simple query", "What is the capital of France?"),
        (
            "Code generation",
            "Write a function to sort an array in Rust",
        ),
        ("Explanation", "Explain how blockchain works"),
        ("Translation", "Translate 'Hello World' to Spanish"),
    ];

    let mut total_queries = 0;
    let mut total_tokens = 0;
    let mut total_cost = 0.0;
    let mut total_duration = std::time::Duration::ZERO;

    for (operation_type, query) in operations {
        println!("   Operation: {}", operation_type);
        let start_time = std::time::Instant::now();

        match client.query(query).send_full().await {
            Ok(response) => {
                let duration = start_time.elapsed();
                total_queries += 1;
                total_duration += duration;

                if let Some(metadata) = response.metadata {
                    if let Some(cost) = metadata.cost_usd {
                        total_cost += cost;
                        println!("   Cost: ${:.6}", cost);
                    }

                    if let Some(tokens) = metadata.tokens_used {
                        if let Some(total) = tokens.total_tokens {
                            total_tokens += total;
                            println!("   Tokens: {}", total);
                        }
                    }
                }

                println!("   Duration: {:?}", duration);
                println!(
                    "   Response length: {} characters\n",
                    response.content.len()
                );
            }
            Err(e) => {
                println!("   Error: {}\n", e);
            }
        }
    }

    println!("   Usage Summary:");
    println!("   ==============");
    println!("   Total queries: {}", total_queries);
    println!("   Total tokens: {}", total_tokens);
    println!("   Total cost: ${:.6}", total_cost);
    println!("   Total duration: {:?}", total_duration);
    println!(
        "   Average duration: {:?}\n",
        total_duration / total_queries.max(1) as u32
    );

    Ok(())
}

#[cfg(feature = "analytics")]
async fn cost_analytics() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Cost Analytics");
    println!("   Detailed cost tracking and analysis\n");

    let client = Client::builder().stream_format(StreamFormat::Json).build();

    println!("   Cost Tracking Demo:");

    // Different types of queries with varying costs
    let cost_scenarios = vec![
        ("Short query", "Hi"),
        ("Medium query", "Explain the basics of machine learning in a few sentences"),
        ("Long query", "Write a comprehensive guide about Rust programming including memory management, ownership, borrowing, and lifetimes with examples"),
    ];

    let mut cost_breakdown = Vec::new();

    for (scenario, query) in cost_scenarios {
        println!("   Scenario: {}", scenario);

        match client.query(query).send_full().await {
            Ok(response) => {
                if let Some(metadata) = response.metadata {
                    if let Some(cost) = metadata.cost_usd {
                        let tokens = metadata
                            .tokens_used
                            .and_then(|t| t.total_tokens)
                            .unwrap_or(0);

                        cost_breakdown.push((scenario, cost, tokens));

                        println!("   Cost: ${:.6}", cost);
                        println!("   Tokens: {}", tokens);
                        if tokens > 0 {
                            println!("   Cost per token: ${:.8}", cost / tokens as f64);
                        }
                        println!("   Response length: {} chars\n", response.content.len());
                    }
                }
            }
            Err(e) => {
                println!("   Error: {}\n", e);
            }
        }
    }

    // Cost analysis
    println!("   Cost Analysis:");
    println!("   ==============");
    let total_cost: f64 = cost_breakdown.iter().map(|(_, cost, _)| cost).sum();
    let total_tokens: u32 = cost_breakdown.iter().map(|(_, _, tokens)| tokens).sum();

    println!("   Total cost: ${:.6}", total_cost);
    println!("   Total tokens: {}", total_tokens);
    println!(
        "   Average cost per token: ${:.8}",
        total_cost / total_tokens as f64
    );

    println!("\n   Cost Breakdown:");
    for (scenario, cost, tokens) in cost_breakdown {
        let percentage = (cost / total_cost) * 100.0;
        println!(
            "   - {}: ${:.6} ({:.1}% of total)",
            scenario, cost, percentage
        );
    }
    println!();

    Ok(())
}

#[cfg(feature = "analytics")]
async fn performance_metrics() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Performance Metrics");
    println!("   Tracking and analyzing performance characteristics\n");

    let client = Client::builder().stream_format(StreamFormat::Json).build();

    println!("   Performance Benchmarking:");

    let mut performance_data = Vec::new();

    // Different query types to benchmark
    let benchmark_queries = vec![
        ("Math calculation", "What is 15 * 23?"),
        (
            "Code generation",
            "Write a Rust function to reverse a string",
        ),
        ("Creative writing", "Write a haiku about programming"),
        (
            "Data analysis",
            "What are the pros and cons of functional programming?",
        ),
    ];

    for (query_type, query) in benchmark_queries {
        println!("   Benchmarking: {}", query_type);

        let start = std::time::Instant::now();
        match client.query(query).send_full().await {
            Ok(response) => {
                let duration = start.elapsed();
                let response_length = response.content.len();

                performance_data.push((query_type, duration, response_length));

                println!("   Duration: {:?}", duration);
                println!("   Response length: {} chars", response_length);
                println!(
                    "   Throughput: {:.2} chars/sec",
                    response_length as f64 / duration.as_secs_f64()
                );

                if let Some(metadata) = response.metadata {
                    if let Some(tokens) = metadata.tokens_used {
                        if let Some(total) = tokens.total_tokens {
                            println!(
                                "   Token throughput: {:.2} tokens/sec",
                                total as f64 / duration.as_secs_f64()
                            );
                        }
                    }
                }
                println!();
            }
            Err(e) => {
                println!("   Error: {}\n", e);
            }
        }
    }

    // Performance analysis
    println!("   Performance Summary:");
    println!("   ===================");

    let total_duration: std::time::Duration = performance_data
        .iter()
        .map(|(_, duration, _)| *duration)
        .sum();
    let avg_duration = total_duration / performance_data.len() as u32;

    let total_chars: usize = performance_data.iter().map(|(_, _, length)| *length).sum();

    println!("   Average response time: {:?}", avg_duration);
    println!("   Total characters generated: {}", total_chars);
    println!(
        "   Average throughput: {:.2} chars/sec",
        total_chars as f64 / total_duration.as_secs_f64()
    );

    // Find fastest and slowest
    if let Some((fastest_type, fastest_duration, _)) = performance_data
        .iter()
        .min_by_key(|(_, duration, _)| duration)
    {
        println!(
            "   Fastest query type: {} ({:?})",
            fastest_type, fastest_duration
        );
    }

    if let Some((slowest_type, slowest_duration, _)) = performance_data
        .iter()
        .max_by_key(|(_, duration, _)| duration)
    {
        println!(
            "   Slowest query type: {} ({:?})",
            slowest_type, slowest_duration
        );
    }
    println!();

    Ok(())
}

#[cfg(feature = "analytics")]
async fn session_analytics() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Session Analytics");
    println!("   Tracking analytics across session lifecycle\n");

    let client = Client::builder()
        .stream_format(StreamFormat::Json)
        .system_prompt("You are a helpful assistant. Keep track of our conversation.")
        .build();

    println!("   Session Analytics Tracking:");

    let mut session_data = Vec::new();
    let session_start = std::time::Instant::now();

    // Simulate a session with multiple interactions
    let session_queries = vec![
        "Hello! I'm starting a new project.",
        "I need help with Rust error handling.",
        "Can you show me an example?",
        "Thanks! What about async programming?",
        "Perfect! Let me summarize what I learned.",
    ];

    for (i, query) in session_queries.iter().enumerate() {
        println!("   Query {}: {}", i + 1, query);

        let query_start = std::time::Instant::now();
        match client.query(query).send_full().await {
            Ok(response) => {
                let query_duration = query_start.elapsed();

                let mut cost = 0.0;
                let mut tokens = 0;
                let mut session_id = "unknown".to_string();

                if let Some(metadata) = response.metadata {
                    cost = metadata.cost_usd.unwrap_or(0.0);
                    tokens = metadata
                        .tokens_used
                        .and_then(|t| t.total_tokens)
                        .unwrap_or(0);
                    session_id = metadata.session_id;
                }

                session_data.push((i + 1, query_duration, cost, tokens, response.content.len()));

                println!("   Response length: {} chars", response.content.len());
                println!("   Duration: {:?}", query_duration);
                println!("   Session ID: {}", session_id);
                println!();
            }
            Err(e) => {
                println!("   Error: {}\n", e);
            }
        }
    }

    // Session analytics summary
    let session_duration = session_start.elapsed();

    println!("   Session Analytics Summary:");
    println!("   =========================");
    println!("   Session duration: {:?}", session_duration);
    println!("   Total queries: {}", session_data.len());

    let total_cost: f64 = session_data.iter().map(|(_, _, cost, _, _)| cost).sum();
    let total_tokens: u32 = session_data.iter().map(|(_, _, _, tokens, _)| tokens).sum();
    let total_chars: usize = session_data.iter().map(|(_, _, _, _, chars)| chars).sum();

    println!("   Total cost: ${:.6}", total_cost);
    println!("   Total tokens: {}", total_tokens);
    println!("   Total characters: {}", total_chars);

    if !session_data.is_empty() {
        let avg_query_time: std::time::Duration = session_data
            .iter()
            .map(|(_, duration, _, _, _)| *duration)
            .sum::<std::time::Duration>()
            / session_data.len() as u32;

        println!("   Average query time: {:?}", avg_query_time);
        println!(
            "   Average cost per query: ${:.6}",
            total_cost / session_data.len() as f64
        );
    }
    println!();

    Ok(())
}

#[cfg(feature = "analytics")]
async fn analytics_dashboard() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Analytics Dashboard");
    println!("   Displaying comprehensive usage dashboard\n");

    // Simulate dashboard data (in a real implementation, this would come from stored analytics)
    println!("   Claude SDK Analytics Dashboard");
    println!("   ==============================");
    println!("   📊 Usage Statistics (Last 24 Hours)");
    println!();

    println!("   📈 Query Volume:");
    println!("   ├─ Total queries: 42");
    println!("   ├─ Successful: 40 (95.2%)");
    println!("   ├─ Failed: 2 (4.8%)");
    println!("   └─ Average per hour: 1.75");
    println!();

    println!("   💰 Cost Analysis:");
    println!("   ├─ Total cost: $0.245600");
    println!("   ├─ Average per query: $0.006140");
    println!("   ├─ Most expensive query: $0.023400");
    println!("   └─ Projected monthly cost: $7.37");
    println!();

    println!("   🚀 Performance Metrics:");
    println!("   ├─ Average response time: 2.34s");
    println!("   ├─ Fastest response: 0.89s");
    println!("   ├─ Slowest response: 5.67s");
    println!("   └─ 95th percentile: 4.12s");
    println!();

    println!("   📝 Token Usage:");
    println!("   ├─ Total tokens: 15,248");
    println!("   ├─ Input tokens: 6,102 (40.0%)");
    println!("   ├─ Output tokens: 9,146 (60.0%)");
    println!("   └─ Average tokens per query: 362");
    println!();

    println!("   🔍 Query Categories:");
    println!("   ├─ Code generation: 15 queries (35.7%)");
    println!("   ├─ Explanations: 12 queries (28.6%)");
    println!("   ├─ Q&A: 10 queries (23.8%)");
    println!("   └─ Other: 5 queries (11.9%)");
    println!();

    println!("   ⚡ Top Performance Insights:");
    println!("   ├─ Peak usage time: 14:00-15:00");
    println!("   ├─ Most efficient query type: Simple Q&A");
    println!("   ├─ Cost optimization opportunity: Reduce code generation queries");
    println!("   └─ Reliability: 99.2% uptime");
    println!();

    // Simulate real-time metrics update
    println!("   🔴 Live Metrics (updating...):");
    for i in 1..=5 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        println!("   ├─ Active sessions: {}", i);
        println!(
            "   ├─ Queries in progress: {}",
            if i % 2 == 0 { 0 } else { 1 }
        );
        println!("   └─ Current load: {}%", 10 + (i * 5));

        if i < 5 {
            print!("\r"); // Clear line for next update
        }
    }
    println!();

    println!("   📋 Export Options:");
    println!("   ├─ JSON export: /analytics/export.json");
    println!("   ├─ CSV export: /analytics/export.csv");
    println!("   └─ PDF report: /analytics/report.pdf");

    Ok(())
}

// When analytics feature is not enabled, provide helpful message
#[cfg(not(feature = "analytics"))]
fn main() {
    println!("=== CLI Analytics Example ===\n");
    println!("This example requires the 'analytics' feature to be enabled.");
    println!("Please run with: cargo run --features analytics --example cli_analytics");
    println!("\nThe 'analytics' feature includes:");
    println!("- Usage tracking and metrics collection");
    println!("- Cost analysis and reporting");
    println!("- Performance monitoring");
    println!("- Session analytics");
    println!("- Interactive dashboards");
    println!("- Data export capabilities");
    println!("\nFor more information, see the documentation about feature flags.");
}

// Example output (when analytics feature is enabled):
/*
=== Claude SDK CLI Analytics Example ===

This example demonstrates analytics and usage tracking features.

1. Basic Usage Tracking
   Tracking basic usage metrics for Claude SDK operations

   Performing tracked operations:
   Operation: Simple query
   Cost: $0.000123
   Tokens: 25
   Duration: 1.234s
   Response length: 87 characters

   Operation: Code generation
   Cost: $0.001456
   Tokens: 156
   Duration: 3.567s
   Response length: 423 characters

   Usage Summary:
   ==============
   Total queries: 4
   Total tokens: 512
   Total cost: $0.003245
   Total duration: 12.345s
   Average duration: 3.086s

2. Cost Analytics
   Detailed cost tracking and analysis

   Cost Analysis:
   ==============
   Total cost: $0.002134
   Total tokens: 387
   Average cost per token: $0.00000551

   Cost Breakdown:
   - Short query: $0.000045 (2.1% of total)
   - Medium query: $0.000678 (31.8% of total)
   - Long query: $0.001411 (66.1% of total)

CLI analytics example completed successfully!
*/
