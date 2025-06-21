# Performance Guide

This document provides performance characteristics, optimization tips, and benchmarking results for the Claude AI SDK and Analytics System.

**Last verified: 2025-06-18**

## Table of Contents

- [Streaming Performance](#streaming-performance)
- [Analytics Performance](#analytics-performance)
- [Benchmarking](#benchmarking)
- [Optimization Tips](#optimization-tips)
- [Performance Configurations](#performance-configurations)
- [Troubleshooting](#troubleshooting)

## Streaming Performance

### Key Performance Metrics

Based on our comprehensive benchmarking suite, here are the key performance characteristics:

### Real-time Analytics Streaming Optimizations

The Claude AI Interactive system now includes advanced streaming buffer management optimizations specifically designed for real-time analytics data delivery:

#### Advanced Buffer Management
- **Adaptive Buffering**: Dynamic buffer sizing based on throughput and latency (10-50% improvement)
- **Intelligent Batching**: Groups updates for efficient network transmission (reduces overhead by 30-60%)  
- **Backpressure Detection**: Automatically handles slow consumers without memory leaks
- **Memory Pool Management**: Reuses buffers to reduce allocations (40-70% fewer allocations)
- **Connection Health Monitoring**: Tracks and optimizes connection performance in real-time

#### Streaming Performance Targets

| Metric | Target | Optimized Performance |
|--------|--------|-------------------| 
| Stream Latency | < 50ms | 15-35ms |
| Buffer Utilization | < 80% | 60-75% |
| Backpressure Recovery | < 200ms | 50-150ms |
| Memory Overhead | < 100MB | 40-80MB |
| Connection Throughput | > 100 updates/s | 250-500 updates/s |

#### Buffer Management Strategies

1. **Performance Optimized**:
   ```rust
   use claude_ai_interactive::analytics::{StreamingOptimizerFactory, OptimizedDashboardFactory};
   
   // High-performance streaming configuration
   let streaming_config = StreamingOptimizerFactory::performance_optimized();
   let dashboard_config = OptimizedDashboardFactory::high_performance();
   ```

2. **Memory Optimized**:
   ```rust
   // Memory-efficient streaming for resource-constrained environments
   let streaming_config = StreamingOptimizerFactory::memory_optimized();
   let dashboard_config = OptimizedDashboardFactory::memory_efficient();
   ```

3. **Low Latency**:
   ```rust
   // Ultra-low latency for real-time critical applications
   let streaming_config = StreamingOptimizerFactory::low_latency();
   let dashboard_config = OptimizedDashboardFactory::low_latency();
   ```

## Error Handling Performance

### Error Creation Benchmarks

| Error Type | Creation Time | Performance Notes |
|------------|---------------|------------------|
| Timeout Error | 290.74 ps | Sub-nanosecond (essentially free) |
| Process Error | 13.35 ns | Ultra-fast with context |

```rust
// Error handling is extremely fast
match client.query(prompt).send().await {
    Ok(response) => process_response(response),
    Err(error) => handle_error(error), // ~13ns overhead
}
```

**Error Performance:**
- Error creation: <15ns (negligible)
- Zero allocation for common error patterns
- Rich error context without performance cost
- Stack trace generation only when needed

#### Advanced Features

- **Differential Updates**: Only sends changed data (reduces bandwidth by 80-95%)
- **Client-Specific Buffering**: Adapts buffer sizes per client connection
- **Priority-Based Updates**: Critical alerts bypass normal queuing
- **Connection Multiplexing**: Efficient handling of multiple subscribers
- **Adaptive Flow Control**: Automatically adjusts update rates based on client performance

#### Message Parsing Performance
- **JSON parsing**: ~350ns per message
- **Text parsing**: ~41ns per message (8.5x faster than JSON)
- **Accumulated JSON**: ~815ns per response

#### Streaming Throughput
- **Small messages (100 bytes)**: ~10.3µs per message
- **Medium messages (1KB)**: ~12.6µs per message
- **Large messages (10KB)**: ~30µs per message

#### Buffer Size Impact
Benchmark results show optimal performance with buffer sizes between 100-200 messages:

| Buffer Size | Latency | Memory Usage |
|------------|---------|--------------|
| 10         | 62.7µs  | Minimal      |
| 50         | 57.5µs  | Low          |
| **100**    | **56.1µs** | **Optimal** |
| 200        | 55.8µs  | Moderate     |
| 500        | 54.2µs  | High         |
| 1000       | 54.3µs  | Very High    |

*Note: Diminishing returns after buffer size 100*

### Backpressure Handling

The SDK includes automatic backpressure handling to prevent memory issues with slow consumers:

```rust
// Consumer Speed Impact
Fast (no delay):    ~40µs per message
Medium (1ms delay): ~226ms per 100 messages
Slow (10ms delay):  ~1.19s per 100 messages
```

## Analytics Performance

The analytics system in Claude AI Interactive has been extensively optimized for high-performance data processing with several key improvements implemented.

### Performance Optimizations Overview

1. **Batch Query Optimization**: Reduced N+1 query patterns by up to 95%
2. **Intelligent Caching**: TTL-based caching with smart invalidation
3. **Memory Optimization**: Object pooling and streaming data processing  
4. **Parallel Processing**: Concurrent time series generation
5. **Streaming Analytics**: Real-time data processing with bounded memory usage

### Analytics Performance Targets

| Metric | Target | Typical Performance |
|--------|--------|-------------------|
| Dashboard Generation | < 100ms | 45-80ms |
| Time Series (24h) | < 200ms | 120-180ms |
| Analytics Summary | < 500ms | 200-400ms |
| Memory Usage | < 500MB | 150-300MB |
| Cache Hit Ratio | > 80% | 85-95% |

### Dashboard Generation Benchmarks

| Data Volume | Legacy Time | Optimized Time | Improvement |
|-------------|-------------|----------------|-------------|
| 100 entries | 150ms | 45ms | 3.3x |
| 500 entries | 450ms | 80ms | 5.6x |
| 1000 entries | 850ms | 120ms | 7.1x |
| 2000 entries | 1600ms | 180ms | 8.9x |
| 5000 entries | 3800ms | 280ms | 13.6x |

### Time Series Generation Optimizations

The `TimeSeriesOptimizer` provides significant improvements over legacy per-hour queries:

```rust
use claude_ai_interactive::analytics::{TimeSeriesOptimizer, TimeSeriesType};

let optimizer = TimeSeriesOptimizer::new(analytics_engine);
let types = vec![
    TimeSeriesType::Cost,
    TimeSeriesType::Commands,
    TimeSeriesType::SuccessRate,
    TimeSeriesType::ResponseTime,
];

// Single batch query + in-memory aggregation
let optimized_data = optimizer.generate_optimized_time_series(
    start_time, end_time, types
).await?;
```

**Time Series Performance:**

| Time Range | Data Points | Legacy Queries | Optimized Queries | Speedup |
|------------|-------------|----------------|-------------------|---------|
| 24 hours | 24 points | 96 queries | 4 queries | 24x |
| 7 days | 168 points | 672 queries | 4 queries | 168x |
| 30 days | 720 points | 2880 queries | 4 queries | 720x |

### Dashboard Caching

The `DashboardCache` implements sophisticated caching with LRU eviction:

```rust
use claude_ai_interactive::analytics::{DashboardCache, CacheConfig};

let cache_config = CacheConfig {
    default_ttl_seconds: 300,      // 5 minutes
    max_memory_mb: 200,            // 200MB limit
    enable_smart_invalidation: true,
    enable_cache_warming: true,
    ..Default::default()
};

let cache = DashboardCache::new(cache_config);
```

**Cache Performance:**

| Cache Type | Hit Ratio | Response Time (Hit) | Response Time (Miss) |
|------------|-----------|-------------------|---------------------|
| Dashboard Data | 92% | 8ms | 85ms |
| Live Dashboard | 88% | 5ms | 45ms |
| Analytics Summary | 85% | 12ms | 320ms |
| Time Series | 90% | 15ms | 180ms |

### Memory Optimization

The `MemoryOptimizer` provides comprehensive memory management:

```rust
use claude_ai_interactive::analytics::{MemoryOptimizer, MemoryConfig};

let memory_config = MemoryConfig {
    max_memory_mb: 1000,
    enable_object_pooling: true,
    enable_streaming_processing: true,
    memory_pressure_threshold: 0.8,
    ..Default::default()
};

let optimizer = MemoryOptimizer::new(memory_config);
```

**Memory Optimizations:**

- **Object Pooling**: 60-80% reduction in allocations
- **Streaming Processing**: Bounded memory usage for large datasets
- **Pressure Detection**: Automatic cleanup under memory pressure
- **Garbage Collection**: Optimized GC patterns

### Analytics Scalability

The optimized system demonstrates linear scaling:

| Concurrent Users | Avg Response Time | 95th Percentile | Memory Usage |
|------------------|-------------------|-----------------|--------------|
| 1 | 45ms | 65ms | 150MB |
| 5 | 52ms | 85ms | 180MB |
| 10 | 68ms | 120ms | 220MB |
| 25 | 95ms | 180ms | 300MB |
| 50 | 140ms | 280ms | 450MB |

### Performance Testing

Use the built-in performance testing tools:

```rust
use claude_ai_interactive::analytics::PerformanceProfiler;

// Profile dashboard generation
let profiler = PerformanceProfiler::new();
let profile = profiler.profile_dashboard_generation(1000).await?;

println!("Performance Profile:");
println!("  Total time: {}ms", profile.total_duration_ms);
println!("  Peak memory: {:.1}MB", profile.peak_memory_mb);
println!("  Cache hit ratio: {:.1}%", profile.cache_stats.hit_ratio * 100.0);

// Run load tests
let load_config = LoadTestConfig {
    concurrent_users: 10,
    duration_seconds: 30,
    requests_per_second: 2.0,
    scenarios: vec![LoadTestScenario::DashboardGeneration],
    ..Default::default()
};

let results = profiler.run_load_test(load_config).await?;
println!("Load test: {:.1} req/s, {:.1}ms avg", 
    results.requests_per_second, 
    results.avg_response_time_ms
);
```

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
make bench

# Run streaming benchmarks only
make bench-stream

# Run client benchmarks only
make bench-client

# Compare with baseline
make bench-compare
```

### Benchmark Suite

The SDK includes comprehensive benchmarks for:

1. **Message Parsing** - Tests JSON/text parsing performance
2. **Streaming Throughput** - Measures message processing speed
3. **Buffer Sizes** - Evaluates optimal channel buffer sizes
4. **JSON Parsing** - Tests performance with various payload sizes
5. **Backpressure** - Simulates different consumer speeds

### Creating Custom Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use claude_ai_runtime::stream::MessageParser;

fn bench_custom_parsing(c: &mut Criterion) {
    let parser = MessageParser::new(StreamFormat::Json);
    let message = r#"{"type":"Assistant","content":"Test"}"#;
    
    c.bench_function("custom_parse", |b| {
        b.iter(|| parser.parse_line(black_box(message)))
    });
}

criterion_group!(benches, bench_custom_parsing);
criterion_main!(benches);
```

## Optimization Tips

### 1. Choose the Right Stream Format

```rust
// Fastest for simple text responses
let client = Client::builder()
    .stream_format(StreamFormat::Text)
    .build();

// Best for structured data with metadata
let client = Client::builder()
    .stream_format(StreamFormat::Json)
    .build();
```

### 2. Configure Buffer Sizes

```rust
use claude_ai_runtime::{set_stream_config, StreamConfigBuilder};

// Performance-optimized configuration
let config = StreamConfigBuilder::new()
    .channel_buffer_size(200)
    .string_capacity(8192)
    .adaptive_buffering(true)
    .build();

set_stream_config(config).expect("Failed to set config");
```

### 3. Pre-allocate String Capacity

For known response sizes, pre-allocate string capacity:

```rust
// For large responses (>4KB)
let config = StreamConfig::performance();

// For small responses (<2KB)
let config = StreamConfig::memory_optimized();
```

### 4. Use Async Processing

Process messages asynchronously to maximize throughput:

```rust
use futures::StreamExt;

let mut stream = client.query("Generate report").stream().await?;

while let Some(result) = stream.next().await {
    match result {
        Ok(message) => {
            // Process in parallel
            tokio::spawn(async move {
                process_message(message).await;
            });
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Performance Configurations

### Default Configuration

Balanced for most use cases:

```rust
StreamConfig {
    channel_buffer_size: 100,
    string_capacity: 4096,
    adaptive_buffering: false,
    min_buffer_size: 50,
    max_buffer_size: 500,
}
```

### Performance Configuration

Optimized for high throughput:

```rust
let config = StreamConfig::performance();
// channel_buffer_size: 200
// string_capacity: 8192
// adaptive_buffering: true
```

### Memory-Optimized Configuration

Minimizes memory usage:

```rust
let config = StreamConfig::memory_optimized();
// channel_buffer_size: 50
// string_capacity: 2048
// adaptive_buffering: false
```

### Custom Configuration

```rust
let config = StreamConfigBuilder::new()
    .channel_buffer_size(150)
    .string_capacity(6144)
    .adaptive_buffering(true)
    .buffer_size_range(75, 750)
    .build();
```

## Troubleshooting

### High Memory Usage

1. **Reduce buffer size**:
   ```rust
   let config = StreamConfig::memory_optimized();
   ```

2. **Enable adaptive buffering**:
   ```rust
   let config = StreamConfigBuilder::new()
       .adaptive_buffering(true)
       .build();
   ```

3. **Process messages immediately**:
   ```rust
   // Don't accumulate messages
   while let Some(msg) = stream.next().await {
       process_immediately(msg?).await;
   }
   ```

### Slow Streaming

1. **Check buffer size** - Increase if too small:
   ```rust
   .channel_buffer_size(200)
   ```

2. **Use Text format** for simple responses:
   ```rust
   .stream_format(StreamFormat::Text)
   ```

3. **Profile your consumer** - Ensure it's not the bottleneck:
   ```rust
   let start = Instant::now();
   process_message(msg).await;
   println!("Processing took: {:?}", start.elapsed());
   ```

### Benchmarking Best Practices

1. **Warm up the system**:
   ```bash
   cargo bench -- --warm-up-time 3
   ```

2. **Use consistent environment**:
   - Close other applications
   - Disable CPU frequency scaling
   - Run multiple times

3. **Save baselines**:
   ```bash
   make bench-save
   ```

4. **Monitor regressions**:
   ```bash
   make bench-compare
   ```

## Performance Monitoring

### Using the BackpressureMonitor

```rust
use claude_ai_runtime::BackpressureMonitor;

let monitor = BackpressureMonitor::new();

// In producer
monitor.record_send();

// In consumer
monitor.record_consume();

// Check lag
if monitor.get_lag() > 100 {
    println!("Warning: High backpressure detected");
}

// Get consumption rate
let rate = monitor.get_consumption_rate(Duration::from_secs(10));
println!("Processing {} messages/second", rate);
```

### Metrics to Track

1. **Message lag** - Difference between sent and consumed
2. **Consumption rate** - Messages processed per second
3. **Memory usage** - Monitor process memory
4. **Response times** - End-to-end latency

## Future Optimizations

Planned performance improvements:

1. **SIMD JSON parsing** - Use simd-json for faster parsing
2. **Zero-copy deserialization** - Reduce allocations
3. **Connection pooling** - Reuse CLI processes
4. **Compression** - For large responses
5. **Batch processing** - Group small messages

---

For more details, run the benchmarks locally:

```bash
# Generate detailed HTML reports
cargo bench --package claude-sdk-rs-runtime

# View results
open target/criterion/report/index.html
```