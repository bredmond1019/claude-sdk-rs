# Part 7: Advanced Usage Patterns

This tutorial covers advanced usage patterns, production deployment considerations, and sophisticated integration techniques for the claude-sdk-rs Rust SDK. These patterns are essential for building robust, scalable applications with Claude AI.

## Table of Contents

1. [Advanced Error Handling and Recovery](#advanced-error-handling-and-recovery)
2. [Performance Optimization Techniques](#performance-optimization-techniques)
3. [Concurrent Request Handling](#concurrent-request-handling)
4. [Custom Message Processing](#custom-message-processing)
5. [Web Framework Integration](#web-framework-integration)
6. [Testing Strategies and Mock Patterns](#testing-strategies-and-mock-patterns)
7. [Production Deployment Considerations](#production-deployment-considerations)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Advanced Error Handling and Recovery

### Comprehensive Error Recovery Patterns

```rust
use claude_sdk_rs::{Client, Config, Error, Result, StreamFormat};
use std::time::Duration;
use tokio::time::sleep;

/// Advanced error handling with retry logic and circuit breaker pattern
pub struct ResilientClaudeClient {
    client: Client,
    max_retries: usize,
    retry_delay: Duration,
    circuit_breaker: CircuitBreaker,
}

struct CircuitBreaker {
    failure_count: std::sync::atomic::AtomicUsize,
    failure_threshold: usize,
    reset_timeout: Duration,
    last_failure: std::sync::Mutex<Option<std::time::Instant>>,
}

impl ResilientClaudeClient {
    pub fn new(config: Config) -> Self {
        Self {
            client: Client::new(config),
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            circuit_breaker: CircuitBreaker::new(5, Duration::from_secs(60)),
        }
    }

    /// Send query with advanced error handling and retry logic
    pub async fn send_with_recovery(&self, query: &str) -> Result<String> {
        if self.circuit_breaker.is_open() {
            return Err(Error::ProcessError("Circuit breaker is open".to_string()));
        }

        let mut last_error = None;
        
        for attempt in 0..=self.max_retries {
            match self.client.query(query).send().await {
                Ok(response) => {
                    self.circuit_breaker.record_success();
                    return Ok(response);
                }
                Err(error) => {
                    last_error = Some(error.clone());
                    
                    // Determine if error is retryable
                    if !self.is_retryable_error(&error) {
                        self.circuit_breaker.record_failure();
                        return Err(error);
                    }

                    if attempt < self.max_retries {
                        log::warn!("Request failed (attempt {}), retrying: {:?}", attempt + 1, error);
                        sleep(self.retry_delay * 2_u32.pow(attempt as u32)).await;
                    }
                }
            }
        }

        self.circuit_breaker.record_failure();
        Err(last_error.unwrap())
    }

    /// Determine if an error should trigger a retry
    fn is_retryable_error(&self, error: &Error) -> bool {
        match error {
            Error::Timeout(_) => true,
            Error::ProcessError(msg) => {
                // Retry on network-related errors but not on authentication
                msg.contains("network") || msg.contains("connection")
            }
            Error::Io(_) => true,
            Error::BinaryNotFound => false,
            Error::PermissionDenied(_) => false,
            Error::ConfigError(_) => false,
            _ => false,
        }
    }
}

impl CircuitBreaker {
    fn new(failure_threshold: usize, reset_timeout: Duration) -> Self {
        Self {
            failure_count: std::sync::atomic::AtomicUsize::new(0),
            failure_threshold,
            reset_timeout,
            last_failure: std::sync::Mutex::new(None),
        }
    }

    fn is_open(&self) -> bool {
        let count = self.failure_count.load(std::sync::atomic::Ordering::Relaxed);
        if count >= self.failure_threshold {
            if let Ok(guard) = self.last_failure.lock() {
                if let Some(last_failure) = *guard {
                    return last_failure.elapsed() < self.reset_timeout;
                }
            }
        }
        false
    }

    fn record_success(&self) {
        self.failure_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_failure(&self) {
        self.failure_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut guard) = self.last_failure.lock() {
            *guard = Some(std::time::Instant::now());
        }
    }
}
```

### Graceful Degradation Patterns

```rust
use serde_json::Value;

/// Client with graceful degradation capabilities
pub struct DegradationClient {
    primary_client: Client,
    fallback_client: Option<Client>,
    cache: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, (String, std::time::Instant)>>>,
    cache_ttl: Duration,
}

impl DegradationClient {
    pub fn new(primary_config: Config, fallback_config: Option<Config>) -> Self {
        Self {
            primary_client: Client::new(primary_config),
            fallback_client: fallback_config.map(Client::new),
            cache: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            cache_ttl: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Send query with fallback and caching
    pub async fn send_with_fallback(&self, query: &str) -> Result<String> {
        // Check cache first
        if let Some(cached) = self.get_cached_response(query).await {
            log::info!("Returning cached response for query");
            return Ok(cached);
        }

        // Try primary client
        match self.primary_client.query(query).send().await {
            Ok(response) => {
                self.cache_response(query, &response).await;
                Ok(response)
            }
            Err(primary_error) => {
                log::warn!("Primary client failed: {:?}", primary_error);
                
                // Try fallback client
                if let Some(ref fallback) = self.fallback_client {
                    match fallback.query(query).send().await {
                        Ok(response) => {
                            log::info!("Fallback client succeeded");
                            self.cache_response(query, &response).await;
                            Ok(response)
                        }
                        Err(fallback_error) => {
                            log::error!("Both primary and fallback clients failed");
                            // Return a degraded response if possible
                            self.generate_degraded_response(query, primary_error, fallback_error)
                        }
                    }
                } else {
                    Err(primary_error)
                }
            }
        }
    }

    async fn get_cached_response(&self, query: &str) -> Option<String> {
        let cache = self.cache.read().await;
        if let Some((response, timestamp)) = cache.get(query) {
            if timestamp.elapsed() < self.cache_ttl {
                return Some(response.clone());
            }
        }
        None
    }

    async fn cache_response(&self, query: &str, response: &str) {
        let mut cache = self.cache.write().await;
        cache.insert(query.to_string(), (response.to_string(), std::time::Instant::now()));
        
        // Simple cache cleanup (remove expired entries)
        let now = std::time::Instant::now();
        cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.cache_ttl);
    }

    fn generate_degraded_response(&self, query: &str, primary_error: Error, fallback_error: Error) -> Result<String> {
        // Provide helpful error information or basic response
        let response = format!(
            "I'm currently experiencing technical difficulties. Please try again later.\n\
             Query: {}\n\
             Error details available in logs.",
            query
        );
        Ok(response)
    }
}
```

## Performance Optimization Techniques

### Connection Pooling and Client Reuse

```rust
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

/// High-performance client pool for managing multiple Claude clients
pub struct ClaudeClientPool {
    clients: Vec<Arc<Client>>,
    semaphore: Arc<Semaphore>,
    round_robin_counter: std::sync::atomic::AtomicUsize,
}

impl ClaudeClientPool {
    /// Create a new client pool with the specified number of clients
    pub fn new(config: Config, pool_size: usize) -> Self {
        let clients = (0..pool_size)
            .map(|_| Arc::new(Client::new(config.clone())))
            .collect();

        Self {
            clients,
            semaphore: Arc::new(Semaphore::new(pool_size)),
            round_robin_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Execute a query using round-robin client selection
    pub async fn execute<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Client) -> futures::future::BoxFuture<'_, Result<T>> + Send,
        T: Send,
    {
        // Acquire a permit (controls concurrency)
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            Error::ProcessError("Failed to acquire client pool permit".to_string())
        })?;

        // Select client using round-robin
        let index = self.round_robin_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.clients.len();
        let client = &self.clients[index];

        f(client).await
    }

    /// Execute query with automatic retry across different clients
    pub async fn execute_with_retry(&self, query: &str, max_retries: usize) -> Result<String> {
        let mut last_error = None;
        
        for attempt in 0..max_retries {
            let result = self.execute(|client| {
                Box::pin(async move { client.query(query).send().await })
            }).await;

            match result {
                Ok(response) => return Ok(response),
                Err(error) => {
                    last_error = Some(error);
                    if attempt < max_retries - 1 {
                        tokio::time::sleep(Duration::from_millis(100 * 2_u64.pow(attempt as u32))).await;
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }
}
```

### Batch Processing Optimization

```rust
use futures::stream::{self, StreamExt};
use std::collections::HashMap;

/// Batch processor for handling multiple queries efficiently
pub struct BatchProcessor {
    client_pool: ClaudeClientPool,
    batch_size: usize,
    batch_timeout: Duration,
}

impl BatchProcessor {
    pub fn new(client_pool: ClaudeClientPool, batch_size: usize, batch_timeout: Duration) -> Self {
        Self {
            client_pool,
            batch_size,
            batch_timeout,
        }
    }

    /// Process multiple queries in batches with automatic batching and timeouts
    pub async fn process_batch(&self, queries: Vec<String>) -> Vec<Result<String>> {
        let batches: Vec<_> = queries.chunks(self.batch_size).collect();
        let mut results = Vec::with_capacity(queries.len());

        for batch in batches {
            let batch_results = self.process_single_batch(batch.to_vec()).await;
            results.extend(batch_results);
        }

        results
    }

    async fn process_single_batch(&self, batch: Vec<String>) -> Vec<Result<String>> {
        // Process all queries in the batch concurrently
        let futures = batch.into_iter().map(|query| {
            self.client_pool.execute(move |client| {
                Box::pin(async move { client.query(&query).send().await })
            })
        });

        // Execute with timeout
        let timeout_future = tokio::time::timeout(
            self.batch_timeout,
            futures::future::join_all(futures)
        );

        match timeout_future.await {
            Ok(results) => results,
            Err(_) => {
                log::error!("Batch processing timed out");
                vec![Err(Error::Timeout(self.batch_timeout.as_secs())); batch.len()]
            }
        }
    }

    /// Stream-based processing for continuous query handling
    pub async fn process_stream<S>(&self, query_stream: S) -> impl futures::Stream<Item = Result<String>>
    where
        S: futures::Stream<Item = String> + Send + 'static,
    {
        query_stream
            .map(|query| {
                let pool = self.client_pool.clone();
                async move {
                    pool.execute(|client| {
                        Box::pin(async move { client.query(&query).send().await })
                    }).await
                }
            })
            .buffer_unordered(self.batch_size)
    }
}
```

## Concurrent Request Handling

### Advanced Concurrency Patterns

```rust
use tokio::sync::{mpsc, oneshot};
use std::collections::VecDeque;

/// Message types for the worker system
#[derive(Debug)]
enum WorkerMessage {
    Query {
        content: String,
        response_tx: oneshot::Sender<Result<String>>,
    },
    Shutdown,
}

/// High-throughput concurrent query processor
pub struct ConcurrentProcessor {
    worker_tx: mpsc::UnboundedSender<WorkerMessage>,
    _worker_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl ConcurrentProcessor {
    /// Create a new concurrent processor with the specified number of workers
    pub fn new(config: Config, num_workers: usize) -> Self {
        let (worker_tx, worker_rx) = mpsc::unbounded_channel();
        let worker_rx = Arc::new(tokio::sync::Mutex::new(worker_rx));

        let worker_handles = (0..num_workers)
            .map(|worker_id| {
                let client = Client::new(config.clone());
                let rx = Arc::clone(&worker_rx);
                
                tokio::spawn(async move {
                    Self::worker_loop(worker_id, client, rx).await;
                })
            })
            .collect();

        Self {
            worker_tx,
            _worker_handles: worker_handles,
        }
    }

    /// Submit a query for processing
    pub async fn query(&self, content: String) -> Result<String> {
        let (response_tx, response_rx) = oneshot::channel();
        
        self.worker_tx
            .send(WorkerMessage::Query { content, response_tx })
            .map_err(|_| Error::ProcessError("Worker channel closed".to_string()))?;

        response_rx.await
            .map_err(|_| Error::ProcessError("Worker response channel closed".to_string()))?
    }

    async fn worker_loop(
        worker_id: usize,
        client: Client,
        rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<WorkerMessage>>>,
    ) {
        log::info!("Worker {} started", worker_id);

        loop {
            let message = {
                let mut rx_guard = rx.lock().await;
                rx_guard.recv().await
            };

            match message {
                Some(WorkerMessage::Query { content, response_tx }) => {
                    log::debug!("Worker {} processing query", worker_id);
                    let result = client.query(&content).send().await;
                    
                    if let Err(_) = response_tx.send(result) {
                        log::warn!("Worker {} failed to send response", worker_id);
                    }
                }
                Some(WorkerMessage::Shutdown) | None => {
                    log::info!("Worker {} shutting down", worker_id);
                    break;
                }
            }
        }
    }

    /// Graceful shutdown of all workers
    pub async fn shutdown(&self) {
        for _ in 0..self._worker_handles.len() {
            let _ = self.worker_tx.send(WorkerMessage::Shutdown);
        }
    }
}

/// Rate-limited concurrent processor
pub struct RateLimitedProcessor {
    processor: ConcurrentProcessor,
    rate_limiter: Arc<tokio::sync::Semaphore>,
    rate_reset_interval: Duration,
}

impl RateLimitedProcessor {
    pub fn new(config: Config, num_workers: usize, requests_per_minute: usize) -> Self {
        let processor = ConcurrentProcessor::new(config, num_workers);
        let rate_limiter = Arc::new(tokio::sync::Semaphore::new(requests_per_minute));
        
        // Spawn rate limiter reset task
        let limiter_clone = Arc::clone(&rate_limiter);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                // Reset the semaphore permits
                let current_permits = limiter_clone.available_permits();
                if current_permits < requests_per_minute {
                    limiter_clone.add_permits(requests_per_minute - current_permits);
                }
            }
        });

        Self {
            processor,
            rate_limiter,
            rate_reset_interval: Duration::from_secs(60),
        }
    }

    /// Submit a rate-limited query
    pub async fn query(&self, content: String) -> Result<String> {
        // Acquire rate limit permit
        let _permit = self.rate_limiter.acquire().await
            .map_err(|_| Error::ProcessError("Rate limiter failed".to_string()))?;

        self.processor.query(content).await
    }
}
```

## Custom Message Processing

### Advanced Stream Processing

```rust
use claude_sdk_rs::{Message, MessageType, MessageStream};
use futures::stream::StreamExt;
use serde_json::Value;

/// Custom message processor with filtering and transformation capabilities
pub struct MessageProcessor {
    client: Client,
    filters: Vec<Box<dyn Fn(&Message) -> bool + Send + Sync>>,
    transformers: Vec<Box<dyn Fn(Message) -> Message + Send + Sync>>,
}

impl MessageProcessor {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            filters: Vec::new(),
            transformers: Vec::new(),
        }
    }

    /// Add a filter function to process only specific message types
    pub fn add_filter<F>(mut self, filter: F) -> Self
    where
        F: Fn(&Message) -> bool + Send + Sync + 'static,
    {
        self.filters.push(Box::new(filter));
        self
    }

    /// Add a transformer function to modify messages
    pub fn add_transformer<F>(mut self, transformer: F) -> Self
    where
        F: Fn(Message) -> Message + Send + Sync + 'static,
    {
        self.transformers.push(Box::new(transformer));
        self
    }

    /// Process a streaming query with custom filters and transformers
    pub async fn process_stream(&self, query: &str) -> Result<Vec<Message>> {
        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .build();
        let client = Client::new(config);

        let mut stream = client.query(query).stream().await?;
        let mut processed_messages = Vec::new();

        while let Some(message_result) = stream.next().await {
            let mut message = message_result?;

            // Apply filters
            let mut should_process = true;
            for filter in &self.filters {
                if !filter(&message) {
                    should_process = false;
                    break;
                }
            }

            if !should_process {
                continue;
            }

            // Apply transformers
            for transformer in &self.transformers {
                message = transformer(message);
            }

            processed_messages.push(message);
        }

        Ok(processed_messages)
    }

    /// Process with custom aggregation logic
    pub async fn process_with_aggregation<T, F, G>(
        &self,
        query: &str,
        initial: T,
        aggregator: F,
        finalizer: G,
    ) -> Result<T>
    where
        F: Fn(T, Message) -> T,
        G: Fn(T) -> T,
        T: Clone,
    {
        let messages = self.process_stream(query).await?;
        let aggregated = messages.into_iter().fold(initial, aggregator);
        Ok(finalizer(aggregated))
    }
}

/// Advanced message analysis and routing
pub struct MessageRouter {
    routes: HashMap<MessageType, Box<dyn Fn(Message) -> futures::future::BoxFuture<'static, Result<()>> + Send + Sync>>,
}

impl MessageRouter {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    /// Register a handler for a specific message type
    pub fn register_handler<F, Fut>(mut self, message_type: MessageType, handler: F) -> Self
    where
        F: Fn(Message) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        self.routes.insert(
            message_type,
            Box::new(move |message| Box::pin(handler(message))),
        );
        self
    }

    /// Route and process a stream of messages
    pub async fn route_stream(&self, mut stream: MessageStream) -> Result<()> {
        while let Some(message_result) = stream.next().await {
            let message = message_result?;
            let message_type = message.message_type.clone();

            if let Some(handler) = self.routes.get(&message_type) {
                if let Err(e) = handler(message).await {
                    log::error!("Handler for {:?} failed: {:?}", message_type, e);
                }
            }
        }
        Ok(())
    }
}
```

## Web Framework Integration

### Axum Integration

```rust
use axum::{
    extract::{Query as AxumQuery, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Application state containing Claude client
#[derive(Clone)]
pub struct AppState {
    claude_client: Arc<ClaudeClientPool>,
    rate_limiter: Arc<RateLimitedProcessor>,
}

/// Request/Response types
#[derive(Deserialize)]
pub struct ChatRequest {
    message: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    stream: bool,
}

#[derive(Serialize)]
pub struct ChatResponse {
    response: String,
    metadata: Option<ResponseMetadata>,
}

/// Chat endpoint handler
async fn handle_chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let response = state
        .rate_limiter
        .query(request.message)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ChatResponse {
        response,
        metadata: None, // Could extract from full response
    }))
}

/// Streaming chat endpoint
async fn handle_stream_chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<axum::response::Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>>>, (StatusCode, String)> {
    let config = Config::builder()
        .stream_format(StreamFormat::StreamJson)
        .build();
    let client = Client::new(config);

    let stream = client
        .query(&request.message)
        .stream()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let sse_stream = stream.map(|message_result| {
        match message_result {
            Ok(message) => {
                let json = serde_json::to_string(&message).unwrap_or_default();
                Ok(axum::response::sse::Event::default().data(json))
            }
            Err(e) => {
                let error_json = serde_json::json!({"error": e.to_string()});
                Ok(axum::response::sse::Event::default().data(error_json.to_string()))
            }
        }
    });

    Ok(axum::response::Sse::new(sse_stream))
}

/// Health check endpoint
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

/// Create Axum router with Claude integration
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/chat", post(handle_chat))
        .route("/chat/stream", post(handle_stream_chat))
        .with_state(state)
        .layer(
            tower::ServiceBuilder::new()
                .layer(tower_http::cors::CorsLayer::permissive())
                .layer(tower_http::trace::TraceLayer::new_for_http())
        )
}

/// Example server setup
pub async fn run_server() -> Result<()> {
    let config = Config::builder()
        .model("claude-sonnet-3.5")
        .stream_format(StreamFormat::Json)
        .timeout_secs(120)
        .build();

    let client_pool = ClaudeClientPool::new(config.clone(), 10);
    let rate_limiter = RateLimitedProcessor::new(config, 5, 60); // 60 requests per minute

    let state = AppState {
        claude_client: Arc::new(client_pool),
        rate_limiter: Arc::new(rate_limiter),
    };

    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    
    log::info!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

### WebSocket Integration

```rust
use axum::{
    extract::{ws::{WebSocket, WebSocketUpgrade}, State},
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use tokio::sync::mpsc;

/// WebSocket handler for real-time chat
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

async fn handle_websocket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // Spawn task to handle outgoing messages
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(axum::extract::ws::Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages
    let receive_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            if let Ok(msg) = msg {
                if let axum::extract::ws::Message::Text(text) = msg {
                    // Process message with Claude
                    match state.rate_limiter.query(text).await {
                        Ok(response) => {
                            let response_json = serde_json::json!({
                                "type": "response",
                                "content": response
                            });
                            let _ = tx.send(response_json.to_string());
                        }
                        Err(e) => {
                            let error_json = serde_json::json!({
                                "type": "error",
                                "message": e.to_string()
                            });
                            let _ = tx.send(error_json.to_string());
                        }
                    }
                }
            } else {
                break;
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {},
        _ = receive_task => {},
    }
}
```

## Testing Strategies and Mock Patterns

### Comprehensive Testing Framework

```rust
use claude_sdk_rs::{Client, Config, Result};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

/// Mock client for testing
pub struct MockClaudeClient {
    responses: Arc<Mutex<Vec<Result<String>>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockClaudeClient {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Add a mock response
    pub fn add_response(&self, response: Result<String>) {
        self.responses.lock().unwrap().push(response);
    }

    /// Get call count
    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    /// Mock query method
    pub async fn query(&self, _query: &str) -> Result<String> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            Ok("Default mock response".to_string())
        } else {
            responses.remove(0)
        }
    }
}

/// Test utilities
pub struct TestUtils;

impl TestUtils {
    /// Create a test configuration
    pub fn test_config() -> Config {
        Config::builder()
            .model("claude-haiku-3.5") // Fastest for tests
            .timeout_secs(10)
            .stream_format(StreamFormat::Text)
            .build()
    }

    /// Create a test client (requires Claude CLI to be installed)
    pub fn test_client() -> Client {
        Client::new(Self::test_config())
    }

    /// Integration test helper
    pub async fn integration_test<F, Fut>(test_fn: F) -> Result<()>
    where
        F: FnOnce(Client) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let client = Self::test_client();
        test_fn(client).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_mock_client() {
        let mock = MockClaudeClient::new();
        mock.add_response(Ok("Test response".to_string()));
        mock.add_response(Err(Error::Timeout(30)));

        let response1 = mock.query("Test query 1").await;
        assert!(response1.is_ok());
        assert_eq!(response1.unwrap(), "Test response");

        let response2 = mock.query("Test query 2").await;
        assert!(response2.is_err());

        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn test_resilient_client() {
        let config = TestUtils::test_config();
        let resilient = ResilientClaudeClient::new(config);

        // This would test with a real client in practice
        // For testing, you'd typically use dependency injection
        // or trait objects to inject the mock
    }

    #[tokio::test]
    async fn test_concurrent_processor() {
        let config = TestUtils::test_config();
        let processor = ConcurrentProcessor::new(config, 2);

        // Test concurrent processing
        let queries = vec![
            "What is 1+1?".to_string(),
            "What is 2+2?".to_string(),
            "What is 3+3?".to_string(),
        ];

        let futures = queries.into_iter().map(|q| processor.query(q));
        let results = futures::future::join_all(futures).await;

        // All queries should complete (assuming Claude CLI is available)
        assert_eq!(results.len(), 3);
    }
}

/// Property-based testing
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_config_builder_properties(
            timeout in 1u64..600,
            max_tokens in 1usize..8192
        ) {
            let config = Config::builder()
                .timeout_secs(timeout)
                .max_tokens(max_tokens)
                .build();

            assert_eq!(config.timeout_secs, Some(timeout));
            assert_eq!(config.max_tokens, Some(max_tokens));
        }

        #[test]
        fn test_tool_permission_format(
            server in "[a-zA-Z][a-zA-Z0-9_]*",
            tool in "[a-zA-Z][a-zA-Z0-9_]*"
        ) {
            let permission = ToolPermission::mcp(&server, &tool);
            let formatted = permission.to_cli_format();
            assert!(formatted.starts_with("mcp__"));
            assert!(formatted.contains(&server));
            assert!(formatted.contains(&tool));
        }
    }
}
```

## Production Deployment Considerations

### Configuration Management

```rust
use serde::{Deserialize, Serialize};
use std::fs;

/// Production configuration with environment variable support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    pub claude: ClaudeConfig,
    pub server: ServerConfig,
    pub logging: LoggingConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub model: String,
    pub max_tokens: Option<usize>,
    pub timeout_secs: u64,
    pub max_concurrent_requests: usize,
    pub rate_limit_per_minute: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub health_check_interval_secs: u64,
    pub prometheus_port: Option<u16>,
}

impl ProductionConfig {
    /// Load configuration from file with environment variable overrides
    pub fn load() -> Result<Self> {
        let config_path = std::env::var("CONFIG_PATH")
            .unwrap_or_else(|_| "config/production.toml".to_string());

        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| Error::ConfigError(format!("Failed to read config file {}: {}", config_path, e)))?;

        let mut config: ProductionConfig = toml::from_str(&config_str)
            .map_err(|e| Error::ConfigError(format!("Failed to parse config: {}", e)))?;

        // Override with environment variables
        if let Ok(model) = std::env::var("CLAUDE_MODEL") {
            config.claude.model = model;
        }
        if let Ok(timeout) = std::env::var("CLAUDE_TIMEOUT") {
            config.claude.timeout_secs = timeout.parse()
                .map_err(|e| Error::ConfigError(format!("Invalid CLAUDE_TIMEOUT: {}", e)))?;
        }
        if let Ok(port) = std::env::var("SERVER_PORT") {
            config.server.port = port.parse()
                .map_err(|e| Error::ConfigError(format!("Invalid SERVER_PORT: {}", e)))?;
        }

        Ok(config)
    }

    /// Convert to Claude SDK config
    pub fn to_claude_config(&self) -> Config {
        Config::builder()
            .model(&self.claude.model)
            .timeout_secs(self.claude.timeout_secs)
            .max_tokens(self.claude.max_tokens.unwrap_or(4096))
            .stream_format(StreamFormat::Json)
            .build()
    }
}
```

### Health Checks and Readiness

```rust
use std::time::{Duration, Instant};

/// Health check system for production deployment
pub struct HealthChecker {
    client: Client,
    last_check: std::sync::Arc<std::sync::RwLock<Option<Instant>>>,
    last_result: std::sync::Arc<std::sync::RwLock<bool>>,
    check_interval: Duration,
}

impl HealthChecker {
    pub fn new(client: Client, check_interval: Duration) -> Self {
        Self {
            client,
            last_check: std::sync::Arc::new(std::sync::RwLock::new(None)),
            last_result: std::sync::Arc::new(std::sync::RwLock::new(false)),
            check_interval,
        }
    }

    /// Start background health checking
    pub fn start_background_checks(&self) {
        let client = self.client.clone();
        let last_check = Arc::clone(&self.last_check);
        let last_result = Arc::clone(&self.last_result);
        let interval = self.check_interval;

        tokio::spawn(async move {
            let mut check_interval = tokio::time::interval(interval);
            loop {
                check_interval.tick().await;
                let is_healthy = Self::perform_health_check(&client).await;
                
                {
                    let mut check_guard = last_check.write().unwrap();
                    *check_guard = Some(Instant::now());
                }
                {
                    let mut result_guard = last_result.write().unwrap();
                    *result_guard = is_healthy;
                }

                if !is_healthy {
                    log::error!("Health check failed");
                } else {
                    log::debug!("Health check passed");
                }
            }
        });
    }

    async fn perform_health_check(client: &Client) -> bool {
        match tokio::time::timeout(
            Duration::from_secs(10),
            client.query("Health check: respond with 'OK'").send()
        ).await {
            Ok(Ok(response)) => {
                response.to_lowercase().contains("ok")
            }
            _ => false,
        }
    }

    /// Check if the service is healthy
    pub fn is_healthy(&self) -> bool {
        let last_result = self.last_result.read().unwrap();
        *last_result
    }

    /// Check if the service is ready (has performed at least one health check)
    pub fn is_ready(&self) -> bool {
        let last_check = self.last_check.read().unwrap();
        last_check.is_some()
    }
}
```

### Graceful Shutdown

```rust
use tokio::signal;
use std::sync::atomic::{AtomicBool, Ordering};

/// Graceful shutdown handler
pub struct ShutdownHandler {
    shutdown_signal: Arc<AtomicBool>,
    active_requests: Arc<std::sync::atomic::AtomicUsize>,
}

impl ShutdownHandler {
    pub fn new() -> Self {
        Self {
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            active_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Start listening for shutdown signals
    pub async fn listen_for_shutdown(&self) {
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let active_requests = Arc::clone(&self.active_requests);

        tokio::spawn(async move {
            // Listen for SIGTERM or SIGINT
            #[cfg(unix)]
            {
                let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                    .expect("Failed to register SIGTERM handler");
                let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())
                    .expect("Failed to register SIGINT handler");

                tokio::select! {
                    _ = sigterm.recv() => {
                        log::info!("Received SIGTERM, initiating graceful shutdown");
                    }
                    _ = sigint.recv() => {
                        log::info!("Received SIGINT, initiating graceful shutdown");
                    }
                }
            }

            #[cfg(not(unix))]
            {
                signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
                log::info!("Received Ctrl+C, initiating graceful shutdown");
            }

            shutdown_signal.store(true, Ordering::Relaxed);

            // Wait for active requests to complete
            let start = Instant::now();
            let max_wait = Duration::from_secs(30);

            while active_requests.load(Ordering::Relaxed) > 0 && start.elapsed() < max_wait {
                log::info!("Waiting for {} active requests to complete", 
                    active_requests.load(Ordering::Relaxed));
                tokio::time::sleep(Duration::from_secs(1)).await;
            }

            if active_requests.load(Ordering::Relaxed) > 0 {
                log::warn!("Forcefully shutting down with {} active requests", 
                    active_requests.load(Ordering::Relaxed));
            } else {
                log::info!("All requests completed, shutting down gracefully");
            }

            std::process::exit(0);
        });
    }

    /// Check if shutdown has been initiated
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_signal.load(Ordering::Relaxed)
    }

    /// Increment active request counter
    pub fn increment_requests(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active request counter
    pub fn decrement_requests(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}
```

## Monitoring and Observability

### Metrics Collection

```rust
use prometheus::{Counter, Histogram, IntGauge, Registry};
use std::time::Instant;

/// Metrics collector for Claude AI operations
pub struct MetricsCollector {
    registry: Registry,
    request_count: Counter,
    request_duration: Histogram,
    active_requests: IntGauge,
    error_count: Counter,
    token_usage: Counter,
}

impl MetricsCollector {
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        
        let request_count = Counter::new("claude_requests_total", "Total number of requests")?;
        let request_duration = Histogram::new("claude_request_duration_seconds", "Request duration")?;
        let active_requests = IntGauge::new("claude_active_requests", "Active requests")?;
        let error_count = Counter::new("claude_errors_total", "Total number of errors")?;
        let token_usage = Counter::new("claude_tokens_used_total", "Total tokens used")?;

        registry.register(Box::new(request_count.clone()))?;
        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(active_requests.clone()))?;
        registry.register(Box::new(error_count.clone()))?;
        registry.register(Box::new(token_usage.clone()))?;

        Ok(Self {
            registry,
            request_count,
            request_duration,
            active_requests,
            error_count,
            token_usage,
        })
    }

    /// Record a successful request
    pub fn record_request(&self, duration: Duration, tokens_used: Option<u64>) {
        self.request_count.inc();
        self.request_duration.observe(duration.as_secs_f64());
        
        if let Some(tokens) = tokens_used {
            self.token_usage.inc_by(tokens as f64);
        }
    }

    /// Record an error
    pub fn record_error(&self, error_type: &str) {
        self.error_count.inc();
    }

    /// Increment active requests
    pub fn increment_active(&self) {
        self.active_requests.inc();
    }

    /// Decrement active requests
    pub fn decrement_active(&self) {
        self.active_requests.dec();
    }

    /// Get registry for Prometheus endpoint
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

/// Instrumented Claude client with automatic metrics collection
pub struct InstrumentedClient {
    client: Client,
    metrics: Arc<MetricsCollector>,
}

impl InstrumentedClient {
    pub fn new(client: Client, metrics: Arc<MetricsCollector>) -> Self {
        Self { client, metrics }
    }

    /// Send query with automatic metrics collection
    pub async fn query(&self, query: &str) -> Result<String> {
        let start = Instant::now();
        self.metrics.increment_active();

        let result = self.client.query(query).send().await;
        let duration = start.elapsed();

        self.metrics.decrement_active();

        match &result {
            Ok(_) => {
                self.metrics.record_request(duration, None);
            }
            Err(error) => {
                self.metrics.record_error(&format!("{:?}", error));
            }
        }

        result
    }

    /// Send query with full response and token tracking
    pub async fn query_full(&self, query: &str) -> Result<ClaudeResponse> {
        let start = Instant::now();
        self.metrics.increment_active();

        let config = Config::builder()
            .stream_format(StreamFormat::Json)
            .build();
        let client = Client::new(config);

        let result = client.query(query).send_full().await;
        let duration = start.elapsed();

        self.metrics.decrement_active();

        match &result {
            Ok(response) => {
                let tokens_used = response.metadata
                    .as_ref()
                    .and_then(|m| m.tokens_used.as_ref())
                    .and_then(|t| t.input_tokens.map(|input| input + t.output_tokens.unwrap_or(0)));
                
                self.metrics.record_request(duration, tokens_used);
            }
            Err(error) => {
                self.metrics.record_error(&format!("{:?}", error));
            }
        }

        result
    }
}
```

### Structured Logging

```rust
use tracing::{info, warn, error, debug, span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize structured logging for production
pub fn init_logging(config: &LoggingConfig) -> Result<()> {
    let level = config.level.parse::<Level>()
        .map_err(|e| Error::ConfigError(format!("Invalid log level: {}", e)))?;

    let mut layers = Vec::new();

    // Console output
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_level(true)
        .with_thread_ids(true);

    if config.format == "json" {
        layers.push(fmt_layer.json().boxed());
    } else {
        layers.push(fmt_layer.boxed());
    }

    // File output
    if let Some(log_file) = &config.file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file)
            .map_err(|e| Error::ConfigError(format!("Failed to open log file: {}", e)))?;

        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(file)
            .json();
        layers.push(file_layer.boxed());
    }

    tracing_subscriber::registry()
        .with(layers)
        .with(level)
        .init();

    Ok(())
}

/// Request tracing middleware
pub async fn trace_request<F, T>(
    operation: &str,
    query: &str,
    f: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let span = span!(Level::INFO, "claude_request", operation = operation, query_hash = format!("{:x}", std::collections::hash_map::DefaultHasher::new().finish()));
    
    async move {
        let start = Instant::now();
        debug!("Starting Claude operation: {}", operation);

        let result = f.await;
        let duration = start.elapsed();

        match &result {
            Ok(_) => {
                info!(
                    duration_ms = duration.as_millis(),
                    "Claude operation completed successfully"
                );
            }
            Err(error) => {
                error!(
                    duration_ms = duration.as_millis(),
                    error = %error,
                    "Claude operation failed"
                );
            }
        }

        result
    }
    .instrument(span)
    .await
}
```

## Troubleshooting Common Issues

### Diagnostic Tools

```rust
/// Diagnostic utilities for troubleshooting Claude AI issues
pub struct DiagnosticTools;

impl DiagnosticTools {
    /// Check if Claude CLI is properly installed and accessible
    pub async fn check_claude_cli() -> Result<String> {
        use tokio::process::Command;

        let output = Command::new("claude")
            .arg("--version")
            .output()
            .await
            .map_err(|e| Error::BinaryNotFound)?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            Ok(format!("Claude CLI version: {}", version.trim()))
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(Error::ProcessError(format!("Claude CLI error: {}", error)))
        }
    }

    /// Test basic connectivity and authentication
    pub async fn test_connectivity() -> Result<String> {
        let client = Client::new(Config::builder()
            .timeout_secs(10)
            .build());

        client.query("Test connectivity - respond with 'Connection OK'")
            .send()
            .await
    }

    /// Run comprehensive diagnostics
    pub async fn run_diagnostics() -> Vec<(String, Result<String>)> {
        let mut results = Vec::new();

        // Check CLI installation
        results.push(("CLI Installation".to_string(), Self::check_claude_cli().await));

        // Test connectivity
        results.push(("Connectivity Test".to_string(), Self::test_connectivity().await));

        // Check system resources
        results.push(("System Resources".to_string(), Self::check_system_resources()));

        // Test different configurations
        let test_configs = vec![
            ("Default Config", Config::default()),
            ("JSON Format", Config::builder().stream_format(StreamFormat::Json).build()),
            ("Short Timeout", Config::builder().timeout_secs(5).build()),
        ];

        for (name, config) in test_configs {
            let client = Client::new(config);
            let result = client.query("Simple test").send().await;
            results.push((format!("Config Test: {}", name), result));
        }

        results
    }

    fn check_system_resources() -> Result<String> {
        use sysinfo::{System, SystemExt};
        
        let mut sys = System::new_all();
        sys.refresh_all();

        let memory_usage = (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0;
        let cpu_usage = sys.processors().iter()
            .map(|p| p.cpu_usage())
            .sum::<f32>() / sys.processors().len() as f32;

        Ok(format!(
            "Memory: {:.1}% used, CPU: {:.1}% average",
            memory_usage, cpu_usage
        ))
    }

    /// Generate diagnostic report
    pub async fn generate_report() -> String {
        let diagnostics = Self::run_diagnostics().await;
        let mut report = String::new();
        
        report.push_str("=== Claude AI SDK Diagnostic Report ===\n\n");
        report.push_str(&format!("Timestamp: {}\n", chrono::Utc::now().to_rfc3339()));
        report.push_str(&format!("SDK Version: {}\n", env!("CARGO_PKG_VERSION")));
        report.push_str("\n");

        for (test_name, result) in diagnostics {
            report.push_str(&format!("## {}\n", test_name));
            match result {
                Ok(message) => report.push_str(&format!("✅ PASS: {}\n", message)),
                Err(error) => report.push_str(&format!("❌ FAIL: {:?}\n", error)),
            }
            report.push_str("\n");
        }

        report
    }
}

/// Error analysis and suggestions
pub struct ErrorAnalyzer;

impl ErrorAnalyzer {
    /// Analyze error and provide actionable suggestions
    pub fn analyze_error(error: &Error) -> String {
        match error {
            Error::BinaryNotFound => {
                "The Claude CLI is not installed or not in your PATH. \
                 Please install it from https://claude.ai/cli and ensure it's accessible."
            }
            Error::Timeout(seconds) => {
                &format!(
                    "Request timed out after {} seconds. Consider:\n\
                     - Increasing the timeout for complex queries\n\
                     - Breaking down large requests into smaller parts\n\
                     - Checking your network connection",
                    seconds
                )
            }
            Error::ProcessError(msg) if msg.contains("authentication") => {
                "Authentication failed. Please run 'claude auth login' to authenticate."
            }
            Error::ProcessError(msg) if msg.contains("rate limit") => {
                "Rate limit exceeded. Please wait before making more requests or \
                 implement rate limiting in your application."
            }
            Error::PermissionDenied(tool) => {
                &format!(
                    "Tool '{}' is not allowed. Check your allowed_tools configuration \
                     or grant necessary permissions.",
                    tool
                )
            }
            Error::SerializationError(_) => {
                "JSON parsing failed. This might indicate:\n\
                 - Unexpected CLI output format\n\
                 - Corrupted response\n\
                 - Version mismatch between SDK and CLI"
            }
            Error::Io(io_error) => {
                &format!(
                    "I/O error occurred: {}. This might be due to:\n\
                     - File system permissions\n\
                     - Network connectivity issues\n\
                     - Disk space problems",
                    io_error
                )
            }
            _ => "An unexpected error occurred. Check logs for more details."
        }.to_string()
    }

    /// Get troubleshooting steps for common issues
    pub fn get_troubleshooting_steps(error: &Error) -> Vec<String> {
        match error {
            Error::BinaryNotFound => vec![
                "1. Install Claude CLI: curl -sSL https://claude.ai/install.sh | sh".to_string(),
                "2. Add to PATH: export PATH=$PATH:~/.local/bin".to_string(),
                "3. Verify installation: claude --version".to_string(),
                "4. Authenticate: claude auth login".to_string(),
            ],
            Error::Timeout(_) => vec![
                "1. Increase timeout in configuration".to_string(),
                "2. Simplify or break down complex queries".to_string(),
                "3. Check network connectivity".to_string(),
                "4. Monitor system resources".to_string(),
            ],
            Error::ProcessError(_) => vec![
                "1. Check Claude CLI logs: claude --verbose".to_string(),
                "2. Verify authentication: claude auth status".to_string(),
                "3. Test with simple query: claude 'Hello'".to_string(),
                "4. Check system permissions".to_string(),
            ],
            _ => vec![
                "1. Run diagnostic report".to_string(),
                "2. Check application logs".to_string(),
                "3. Verify configuration".to_string(),
                "4. Contact support with error details".to_string(),
            ],
        }
    }
}
```

## Best Practices Summary

### Production Checklist

1. **Error Handling**
   - Implement retry logic with exponential backoff
   - Use circuit breakers for external dependencies
   - Provide graceful degradation
   - Log errors with appropriate context

2. **Performance**
   - Use connection pooling for high throughput
   - Implement rate limiting
   - Cache responses when appropriate
   - Monitor resource usage

3. **Security**
   - Restrict tool permissions
   - Validate input parameters
   - Use secure configuration management
   - Implement proper authentication

4. **Monitoring**
   - Collect metrics on requests, errors, and performance
   - Implement health checks
   - Use structured logging
   - Set up alerting for critical issues

5. **Deployment**
   - Use environment-specific configurations
   - Implement graceful shutdown
   - Plan for scaling and load balancing
   - Test deployment procedures

6. **Testing**
   - Write comprehensive unit tests
   - Use integration tests with real Claude CLI
   - Implement load testing
   - Test error scenarios

This advanced usage guide provides the foundation for building production-ready applications with the claude-sdk-rs SDK. Adapt these patterns to your specific use case and requirements.