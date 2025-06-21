//! # When to Use Async/Await in Rust
//!
//! This guide demonstrates when async/await is beneficial vs when to use sync code.

use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Example 1: I/O-bound operations benefit from async
async fn fetch_multiple_urls() {
    println!("\n=== Async I/O Example ===");

    // Simulate fetching 3 URLs concurrently
    let start = Instant::now();

    let futures = vec![
        fetch_url("https://api1.example.com"),
        fetch_url("https://api2.example.com"),
        fetch_url("https://api3.example.com"),
    ];

    // All requests happen concurrently
    let results = futures::future::join_all(futures).await;

    println!(
        "Async: Fetched {} URLs in {:?}",
        results.len(),
        start.elapsed()
    );
    // Would take ~1s instead of 3s sequential
}

async fn fetch_url(url: &str) -> String {
    println!("Fetching {}", url);
    sleep(Duration::from_secs(1)).await; // Simulate network delay
    format!("Data from {}", url)
}

/// Example 2: CPU-bound operations DON'T benefit from async
fn cpu_intensive_sync() {
    println!("\n=== CPU-bound Example ===");

    let start = Instant::now();
    let result = calculate_primes(1_000_000);
    println!("Sync: Found {} primes in {:?}", result, start.elapsed());
}

fn calculate_primes(limit: u32) -> usize {
    (2..limit).filter(|&n| is_prime(n)).count()
}

fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    (2..=(n as f64).sqrt() as u32).all(|i| n % i != 0)
}

/// Example 3: Good async use case - concurrent file operations
async fn process_files_async() {
    use tokio::fs;

    println!("\n=== Async File Processing ===");

    let files = vec!["file1.txt", "file2.txt", "file3.txt"];
    let mut handles = vec![];

    for file in files {
        let handle = tokio::spawn(async move {
            // Simulate reading and processing
            sleep(Duration::from_millis(100)).await;
            format!("Processed {}", file)
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.await.unwrap();
        println!("{}", result);
    }
}

/// Example 4: Database connection pooling
async fn database_example() {
    println!("\n=== Async Database Example ===");

    // Simulate multiple concurrent database queries
    let queries = vec![
        query_database("SELECT * FROM users"),
        query_database("SELECT * FROM posts"),
        query_database("SELECT * FROM comments"),
    ];

    let results = futures::future::join_all(queries).await;
    println!("Executed {} queries concurrently", results.len());
}

async fn query_database(query: &str) -> String {
    println!("Executing: {}", query);
    sleep(Duration::from_millis(200)).await; // Simulate DB latency
    format!("Results for: {}", query)
}

/// Example 5: When NOT to use async - simple sequential operations
fn when_not_to_use_async() {
    println!("\n=== When NOT to Use Async ===");

    // Don't use async for:
    // 1. Simple calculations
    let sum = (1..100).sum::<i32>();
    println!("Sum: {}", sum);

    // 2. In-memory operations
    let mut vec = vec![1, 2, 3, 4, 5];
    vec.sort();
    println!("Sorted: {:?}", vec);

    // 3. Single blocking operations
    let data = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
    println!("Read {} bytes synchronously", data.len());
}

/// Example 6: Mixed async/sync with spawn_blocking
async fn mixed_workload() {
    println!("\n=== Mixed CPU + I/O Workload ===");

    // Handle CPU-intensive work in blocking thread
    let cpu_task = tokio::task::spawn_blocking(|| calculate_primes(100_000));

    // Handle I/O concurrently
    let io_task = fetch_url("https://api.example.com");

    let (cpu_result, io_result) = tokio::join!(cpu_task, io_task);

    println!("CPU task found {} primes", cpu_result.unwrap());
    println!("I/O task fetched: {}", io_result);
}

/// Decision guide for async/await
fn async_decision_guide() {
    println!("\n=== When to Use Async/Await ===");
    println!(
        "
✅ USE ASYNC when:
- Making network requests (HTTP, gRPC, WebSocket)
- File I/O operations (especially multiple files)
- Database queries
- Waiting for external events
- Coordinating multiple concurrent operations
- Building servers that handle many connections

❌ DON'T USE ASYNC when:
- Doing CPU-intensive calculations
- Simple in-memory operations
- Sequential operations with no waiting
- Small scripts with minimal I/O
- When sync code is simpler and sufficient

🔄 CONSIDER ASYNC when:
- You need to handle timeouts
- Building reactive/event-driven systems
- Integrating with async ecosystems
- Need cancellation support
"
    );
}

#[tokio::main]
async fn main() {
    println!("=== Async/Await in Rust: When to Use It ===");

    // Show decision guide
    async_decision_guide();

    // Run examples
    fetch_multiple_urls().await;
    cpu_intensive_sync();
    process_files_async().await;
    database_example().await;
    when_not_to_use_async();
    mixed_workload().await;

    println!("\n=== Summary ===");
    println!("Async shines for I/O-bound work and concurrent operations.");
    println!("For CPU-bound work, use threads or keep it synchronous.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_is_fine_for_cpu() {
        // Measure sync performance
        let start = Instant::now();
        let count = calculate_primes(10_000);
        let duration = start.elapsed();

        assert!(count > 0);
        println!("Sync calculation took {:?}", duration);
    }

    #[tokio::test]
    async fn test_async_benefits_io() {
        // Demonstrate concurrent execution
        let start = Instant::now();

        let (r1, r2) = tokio::join!(
            sleep(Duration::from_millis(100)),
            sleep(Duration::from_millis(100))
        );

        let duration = start.elapsed();

        // Should take ~100ms, not 200ms
        assert!(duration < Duration::from_millis(150));
    }
}
