use async_trait::async_trait;
use serde_json;
use std::io;
use std::time::{Duration, Instant};
// use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
// use tokio::sync::Mutex;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream};

use crate::mcp::protocol::{MCPRequest, MCPResponse};
// MCPMessage unused for now

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportType {
    Stdio {
        command: String,
        args: Vec<String>,
        /// Restart the process if it exits unexpectedly
        auto_restart: bool,
        /// Maximum number of restart attempts
        max_restarts: u32,
    },
    WebSocket {
        url: String,
        /// Heartbeat interval for keep-alive
        heartbeat_interval: Option<Duration>,
        /// Reconnection configuration
        reconnect_config: ReconnectConfig,
    },
    Http {
        base_url: String,
        /// Connection pool configuration
        pool_config: HttpPoolConfig,
    },
}

/// WebSocket/HTTP reconnection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconnectConfig {
    /// Enable automatic reconnection
    pub enabled: bool,
    /// Maximum number of reconnection attempts
    pub max_attempts: u32,
    /// Initial delay between reconnection attempts
    pub initial_delay: Duration,
    /// Maximum delay between reconnection attempts
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 5,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

/// HTTP connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpPoolConfig {
    /// Maximum number of connections per host
    pub max_connections_per_host: usize,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,
}

impl Default for HttpPoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 10,
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(90),
        }
    }
}

#[async_trait]
pub trait MCPTransport: Send + Sync {
    async fn connect(&mut self) -> Result<(), TransportError>;
    async fn send(&mut self, message: MCPRequest) -> Result<(), TransportError>;
    async fn receive(&mut self) -> Result<MCPResponse, TransportError>;
    async fn disconnect(&mut self) -> Result<(), TransportError>;

    /// Check if the transport is currently connected
    fn is_connected(&self) -> bool;

    /// Get transport-specific health information
    async fn health_check(&mut self) -> Result<TransportHealth, TransportError>;

    /// Send a ping/keep-alive message
    async fn ping(&mut self) -> Result<Duration, TransportError>;

    /// Get transport metrics
    fn get_metrics(&self) -> TransportMetrics;

    /// Force reconnection (if supported)
    async fn reconnect(&mut self) -> Result<(), TransportError>;
}

/// Transport health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportHealth {
    pub is_connected: bool,
    pub last_ping: Option<Duration>,
    pub connection_age: Duration,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub last_error: Option<String>,
}

/// Transport performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportMetrics {
    pub total_connections: u64,
    pub successful_connections: u64,
    pub failed_connections: u64,
    pub reconnection_attempts: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub average_latency: Duration,
    pub uptime: Duration,
}

#[derive(Debug)]
pub enum TransportError {
    IoError(io::Error),
    SerializationError(serde_json::Error),
    WebSocketError(tokio_tungstenite::tungstenite::Error),
    HttpError(reqwest::Error),
    ConnectionError(String),
    ProtocolError(String),
}

impl From<io::Error> for TransportError {
    fn from(err: io::Error) -> Self {
        TransportError::IoError(err)
    }
}

impl From<serde_json::Error> for TransportError {
    fn from(err: serde_json::Error) -> Self {
        TransportError::SerializationError(err)
    }
}

impl From<tokio_tungstenite::tungstenite::Error> for TransportError {
    fn from(err: tokio_tungstenite::tungstenite::Error) -> Self {
        TransportError::WebSocketError(err)
    }
}

impl From<reqwest::Error> for TransportError {
    fn from(err: reqwest::Error) -> Self {
        TransportError::HttpError(err)
    }
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::IoError(e) => write!(f, "IO error: {}", e),
            TransportError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            TransportError::WebSocketError(e) => write!(f, "WebSocket error: {}", e),
            TransportError::HttpError(e) => write!(f, "HTTP error: {}", e),
            TransportError::ConnectionError(e) => write!(f, "Connection error: {}", e),
            TransportError::ProtocolError(e) => write!(f, "Protocol error: {}", e),
        }
    }
}

impl std::error::Error for TransportError {}

impl Default for TransportMetrics {
    fn default() -> Self {
        Self {
            total_connections: 0,
            successful_connections: 0,
            failed_connections: 0,
            reconnection_attempts: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            total_messages_sent: 0,
            total_messages_received: 0,
            average_latency: Duration::from_millis(0),
            uptime: Duration::from_secs(0),
        }
    }
}

pub struct StdioTransport {
    command: String,
    args: Vec<String>,
    auto_restart: bool,
    max_restarts: u32,
    process: Option<Child>,
    reader: Option<BufReader<tokio::process::ChildStdout>>,
    writer: Option<tokio::process::ChildStdin>,
    restart_count: u32,
    connected_at: Option<Instant>,
    metrics: TransportMetrics,
    last_error: Option<String>,
}

impl StdioTransport {
    pub fn new(command: String, args: Vec<String>) -> Self {
        Self {
            command,
            args,
            auto_restart: true,
            max_restarts: 3,
            process: None,
            reader: None,
            writer: None,
            restart_count: 0,
            connected_at: None,
            metrics: TransportMetrics::default(),
            last_error: None,
        }
    }

    pub fn with_restart_config(mut self, auto_restart: bool, max_restarts: u32) -> Self {
        self.auto_restart = auto_restart;
        self.max_restarts = max_restarts;
        self
    }

    async fn attempt_restart(&mut self) -> Result<(), TransportError> {
        if !self.auto_restart || self.restart_count >= self.max_restarts {
            return Err(TransportError::ConnectionError(
                "Process restart limit reached".to_string(),
            ));
        }

        tracing::warn!(
            "Attempting to restart stdio process (attempt {}/{})",
            self.restart_count + 1,
            self.max_restarts
        );

        self.restart_count += 1;
        self.metrics.reconnection_attempts += 1;

        // Wait a bit before restarting
        let delay = Duration::from_millis(500 * (self.restart_count as u64));
        sleep(delay).await;

        self.connect().await
    }
}

#[async_trait]
impl MCPTransport for StdioTransport {
    async fn connect(&mut self) -> Result<(), TransportError> {
        self.metrics.total_connections += 1;

        let result = async {
            let mut child = Command::new(&self.command)
                .args(&self.args)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;

            let stdout = child.stdout.take().ok_or_else(|| {
                TransportError::ConnectionError("Failed to get stdout".to_string())
            })?;
            let stdin = child.stdin.take().ok_or_else(|| {
                TransportError::ConnectionError("Failed to get stdin".to_string())
            })?;

            self.reader = Some(BufReader::new(stdout));
            self.writer = Some(stdin);
            self.process = Some(child);
            self.connected_at = Some(Instant::now());

            Ok::<(), TransportError>(())
        }
        .await;

        match result {
            Ok(_) => {
                self.metrics.successful_connections += 1;
                self.last_error = None;
                tracing::debug!("Successfully connected stdio transport to {}", self.command);
                Ok(())
            }
            Err(e) => {
                self.metrics.failed_connections += 1;
                self.last_error = Some(e.to_string());
                Err(e)
            }
        }
    }

    async fn send(&mut self, message: MCPRequest) -> Result<(), TransportError> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionError("Not connected".to_string()))?;

        let json = serde_json::to_string(&message)?;
        let data = format!("{}\n", json);

        writer.write_all(data.as_bytes()).await?;
        writer.flush().await?;

        self.metrics.total_messages_sent += 1;
        self.metrics.total_bytes_sent += data.len() as u64;

        Ok(())
    }

    async fn receive(&mut self) -> Result<MCPResponse, TransportError> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionError("Not connected".to_string()))?;

        let mut line = String::new();
        reader.read_line(&mut line).await?;

        if line.is_empty() {
            return Err(TransportError::ConnectionError(
                "Connection closed".to_string(),
            ));
        }

        let response: MCPResponse = serde_json::from_str(&line)?;
        Ok(response)
    }

    async fn disconnect(&mut self) -> Result<(), TransportError> {
        if let Some(mut process) = self.process.take() {
            process.kill().await?;
        }
        self.reader = None;
        self.writer = None;
        self.connected_at = None;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.process.is_some() && self.reader.is_some() && self.writer.is_some()
    }

    async fn health_check(&mut self) -> Result<TransportHealth, TransportError> {
        let is_connected = self.is_connected();
        let connection_age = self
            .connected_at
            .map(|t| t.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        Ok(TransportHealth {
            is_connected,
            last_ping: None, // Stdio doesn't support ping
            connection_age,
            bytes_sent: self.metrics.total_bytes_sent,
            bytes_received: self.metrics.total_bytes_received,
            messages_sent: self.metrics.total_messages_sent,
            messages_received: self.metrics.total_messages_received,
            last_error: self.last_error.clone(),
        })
    }

    async fn ping(&mut self) -> Result<Duration, TransportError> {
        // Stdio transport doesn't support ping, return a synthetic response
        if !self.is_connected() {
            return Err(TransportError::ConnectionError("Not connected".to_string()));
        }
        Ok(Duration::from_millis(0))
    }

    fn get_metrics(&self) -> TransportMetrics {
        let mut metrics = self.metrics.clone();
        if let Some(connected_at) = self.connected_at {
            metrics.uptime = connected_at.elapsed();
        }
        metrics
    }

    async fn reconnect(&mut self) -> Result<(), TransportError> {
        self.disconnect().await?;
        self.attempt_restart().await
    }
}

pub struct WebSocketTransport {
    url: String,
    stream: Option<WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>,
    reconnect_config: ReconnectConfig,
    heartbeat_interval: Option<Duration>,
    connected_at: Option<Instant>,
    metrics: TransportMetrics,
    last_error: Option<String>,
    reconnect_attempts: u32,
}

impl WebSocketTransport {
    pub fn new(url: String) -> Self {
        Self {
            url,
            stream: None,
            reconnect_config: ReconnectConfig::default(),
            heartbeat_interval: Some(Duration::from_secs(30)),
            connected_at: None,
            metrics: TransportMetrics::default(),
            last_error: None,
            reconnect_attempts: 0,
        }
    }

    pub fn with_reconnect_config(mut self, config: ReconnectConfig) -> Self {
        self.reconnect_config = config;
        self
    }

    pub fn with_heartbeat_interval(mut self, interval: Option<Duration>) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    async fn attempt_reconnect(&mut self) -> Result<(), TransportError> {
        if !self.reconnect_config.enabled
            || self.reconnect_attempts >= self.reconnect_config.max_attempts
        {
            return Err(TransportError::ConnectionError(
                "Reconnection limit reached".to_string(),
            ));
        }

        tracing::warn!(
            "Attempting to reconnect WebSocket (attempt {}/{})",
            self.reconnect_attempts + 1,
            self.reconnect_config.max_attempts
        );

        self.reconnect_attempts += 1;
        self.metrics.reconnection_attempts += 1;

        // Calculate delay with exponential backoff
        let delay_ms = self.reconnect_config.initial_delay.as_millis() as f64
            * self
                .reconnect_config
                .backoff_multiplier
                .powi(self.reconnect_attempts as i32);
        let delay = Duration::from_millis(
            delay_ms.min(self.reconnect_config.max_delay.as_millis() as f64) as u64,
        );

        sleep(delay).await;

        self.connect().await
    }
}

#[async_trait]
impl MCPTransport for WebSocketTransport {
    async fn connect(&mut self) -> Result<(), TransportError> {
        self.metrics.total_connections += 1;

        let result = connect_async(&self.url).await;
        match result {
            Ok((ws_stream, _)) => {
                self.stream = Some(ws_stream);
                self.connected_at = Some(Instant::now());
                self.reconnect_attempts = 0; // Reset on successful connection
                self.metrics.successful_connections += 1;
                self.last_error = None;
                tracing::debug!("Successfully connected WebSocket to {}", self.url);
                Ok(())
            }
            Err(e) => {
                self.metrics.failed_connections += 1;
                let error = TransportError::WebSocketError(e);
                self.last_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    async fn send(&mut self, message: MCPRequest) -> Result<(), TransportError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionError("Not connected".to_string()))?;

        let json = serde_json::to_string(&message)?;
        stream.send(Message::Text(json.clone())).await?;

        self.metrics.total_messages_sent += 1;
        self.metrics.total_bytes_sent += json.len() as u64;

        Ok(())
    }

    async fn receive(&mut self) -> Result<MCPResponse, TransportError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionError("Not connected".to_string()))?;

        if let Some(msg) = stream.next().await {
            let msg = msg?;
            match msg {
                Message::Text(text) => {
                    self.metrics.total_messages_received += 1;
                    self.metrics.total_bytes_received += text.len() as u64;

                    let response: MCPResponse = serde_json::from_str(&text)?;
                    Ok(response)
                }
                Message::Close(_) => Err(TransportError::ConnectionError(
                    "Connection closed".to_string(),
                )),
                Message::Ping(_) => {
                    // Handle ping frames automatically
                    self.receive().await
                }
                Message::Pong(_) => {
                    // Handle pong frames automatically
                    self.receive().await
                }
                _ => Err(TransportError::ProtocolError(
                    "Unexpected message type".to_string(),
                )),
            }
        } else {
            Err(TransportError::ConnectionError(
                "Connection closed".to_string(),
            ))
        }
    }

    async fn disconnect(&mut self) -> Result<(), TransportError> {
        if let Some(mut stream) = self.stream.take() {
            stream.close(None).await?;
        }
        self.connected_at = None;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    async fn health_check(&mut self) -> Result<TransportHealth, TransportError> {
        let is_connected = self.is_connected();
        let connection_age = self
            .connected_at
            .map(|t| t.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        // Try to ping to get actual latency
        let last_ping = if is_connected {
            self.ping().await.ok()
        } else {
            None
        };

        Ok(TransportHealth {
            is_connected,
            last_ping,
            connection_age,
            bytes_sent: self.metrics.total_bytes_sent,
            bytes_received: self.metrics.total_bytes_received,
            messages_sent: self.metrics.total_messages_sent,
            messages_received: self.metrics.total_messages_received,
            last_error: self.last_error.clone(),
        })
    }

    async fn ping(&mut self) -> Result<Duration, TransportError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionError("Not connected".to_string()))?;

        let start = Instant::now();
        stream.send(Message::Ping(vec![])).await?;

        // Wait for pong response (with timeout)
        let result = timeout(Duration::from_secs(5), async {
            while let Some(msg) = stream.next().await {
                match msg? {
                    Message::Pong(_) => return Ok::<(), TransportError>(()),
                    Message::Text(text) => {
                        // Got a regular message, handle it
                        self.metrics.total_messages_received += 1;
                        self.metrics.total_bytes_received += text.len() as u64;
                    }
                    Message::Close(_) => {
                        return Err(TransportError::ConnectionError(
                            "Connection closed".to_string(),
                        ));
                    }
                    _ => {}
                }
            }
            Err(TransportError::ConnectionError(
                "Connection closed during ping".to_string(),
            ))
        })
        .await;

        match result {
            Ok(_) => Ok(start.elapsed()),
            Err(_) => Err(TransportError::ConnectionError("Ping timeout".to_string())),
        }
    }

    fn get_metrics(&self) -> TransportMetrics {
        let mut metrics = self.metrics.clone();
        if let Some(connected_at) = self.connected_at {
            metrics.uptime = connected_at.elapsed();
        }
        metrics
    }

    async fn reconnect(&mut self) -> Result<(), TransportError> {
        self.disconnect().await?;
        self.attempt_reconnect().await
    }
}

#[derive(Debug)]
pub struct HttpTransport {
    base_url: String,
    client: reqwest::Client,
    auth_token: Option<String>,
}

impl HttpTransport {
    pub fn new(base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30)) // 30-second default timeout
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            base_url,
            client,
            auth_token: None,
        }
    }

    pub fn with_auth_token(mut self, token: String) -> Self {
        self.auth_token = Some(token);
        self
    }

    pub fn set_auth_token(&mut self, token: Option<String>) {
        self.auth_token = token;
    }

    /// Send a request and wait for response (for HTTP-based MCP communication)
    pub async fn send_request(&self, request: MCPRequest) -> Result<MCPResponse, TransportError> {
        let mut request_builder = self
            .client
            .post(&format!("{}/mcp", self.base_url))
            .json(&request);

        // Add authentication header if available
        if let Some(ref token) = self.auth_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
        }

        let response = request_builder.send().await?;

        if !response.status().is_success() {
            return Err(TransportError::ConnectionError(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let mcp_response: MCPResponse = response.json().await?;

        Ok(mcp_response)
    }
}

#[async_trait]
impl MCPTransport for HttpTransport {
    async fn connect(&mut self) -> Result<(), TransportError> {
        Ok(())
    }

    async fn send(&mut self, message: MCPRequest) -> Result<(), TransportError> {
        let mut request_builder = self
            .client
            .post(&format!("{}/mcp", self.base_url))
            .json(&message);

        // Add authentication header if available
        if let Some(ref token) = self.auth_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
        }

        let response = request_builder.send().await?;

        if !response.status().is_success() {
            return Err(TransportError::ConnectionError(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        Ok(())
    }

    async fn receive(&mut self) -> Result<MCPResponse, TransportError> {
        Err(TransportError::ProtocolError(
            "HTTP transport does not support receive - use request/response pattern".to_string(),
        ))
    }

    async fn disconnect(&mut self) -> Result<(), TransportError> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        true // HTTP client is always "connected"
    }

    async fn health_check(&mut self) -> Result<TransportHealth, TransportError> {
        Ok(TransportHealth {
            is_connected: true,
            last_ping: None,
            connection_age: Duration::from_secs(0),
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            last_error: None,
        })
    }

    async fn ping(&mut self) -> Result<Duration, TransportError> {
        let start = Instant::now();
        // Perform a simple HTTP GET to check connectivity
        let response = self
            .client
            .get(&format!("{}/health", self.base_url))
            .send()
            .await;
        match response {
            Ok(_) => Ok(start.elapsed()),
            Err(e) => Err(TransportError::HttpError(e)),
        }
    }

    fn get_metrics(&self) -> TransportMetrics {
        TransportMetrics::default()
    }

    async fn reconnect(&mut self) -> Result<(), TransportError> {
        // HTTP doesn't need reconnection
        Ok(())
    }
}
