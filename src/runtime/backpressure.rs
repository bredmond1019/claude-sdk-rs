use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// Monitors streaming performance and adapts buffer sizes
///
/// `BackpressureMonitor` tracks the flow of messages through a streaming pipeline
/// and provides metrics to help prevent buffer overflow and optimize performance.
/// It's particularly useful for adjusting buffer sizes based on consumption patterns.
///
/// # Examples
///
/// ```rust
/// # use claude_sdk_rs::runtime::BackpressureMonitor;
/// # use std::time::Duration;
/// # use tokio::time::sleep;
/// # #[tokio::main]
/// # async fn main() {
/// let monitor = BackpressureMonitor::new();
///
/// // In producer
/// monitor.record_send();
///
/// // In consumer
/// monitor.record_consume();
///
/// // Check if backpressure is building up
/// let lag = monitor.get_lag();
/// if lag > 100 {
///     println!("Warning: {} messages pending", lag);
///     // Consider slowing down production
/// }
///
/// // Monitor consumption rate
/// sleep(Duration::from_secs(1)).await;
/// let rate = monitor.get_consumption_rate(Duration::from_secs(1));
/// println!("Processing {} messages/second", rate);
/// # }
/// ```
///
/// # Use Cases
///
/// - Monitoring streaming response processing
/// - Detecting slow consumers
/// - Implementing adaptive buffering
/// - Performance diagnostics
#[derive(Clone)]
pub struct BackpressureMonitor {
    /// Number of messages sent
    sent_count: Arc<AtomicU64>,
    /// Number of messages consumed
    consumed_count: Arc<AtomicU64>,
    /// Whether monitoring is active
    active: Arc<AtomicBool>,
    /// Last adjustment time
    last_adjustment: Arc<parking_lot::Mutex<Instant>>,
}

impl BackpressureMonitor {
    /// Creates a new backpressure monitor with default settings.
    pub fn new() -> Self {
        Self {
            sent_count: Arc::new(AtomicU64::new(0)),
            consumed_count: Arc::new(AtomicU64::new(0)),
            active: Arc::new(AtomicBool::new(true)),
            last_adjustment: Arc::new(parking_lot::Mutex::new(Instant::now())),
        }
    }

    /// Record a message being sent
    pub fn record_send(&self) {
        self.sent_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a message being consumed
    pub fn record_consume(&self) {
        self.consumed_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current lag (messages sent but not consumed)
    pub fn get_lag(&self) -> u64 {
        let sent = self.sent_count.load(Ordering::Relaxed);
        let consumed = self.consumed_count.load(Ordering::Relaxed);
        sent.saturating_sub(consumed)
    }

    /// Calculate the consumption rate (messages per second)
    pub fn get_consumption_rate(&self, duration: Duration) -> f64 {
        let consumed = self.consumed_count.load(Ordering::Relaxed);
        consumed as f64 / duration.as_secs_f64()
    }

    /// Stop monitoring
    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if we should apply backpressure
    pub fn should_apply_backpressure(&self, threshold: u64) -> bool {
        self.get_lag() > threshold
    }

    /// Get recommended buffer size based on current performance
    pub fn recommend_buffer_size(
        &self,
        config: &crate::runtime::stream_config::StreamConfig,
    ) -> usize {
        if !config.adaptive_buffering {
            return config.channel_buffer_size;
        }

        let lag = self.get_lag();
        let current_size = config.channel_buffer_size;

        // Check if enough time has passed since last adjustment
        let mut last_adj = self.last_adjustment.lock();
        let now = Instant::now();
        if now.duration_since(*last_adj) < Duration::from_secs(1) {
            return current_size;
        }

        let new_size = if lag > current_size as u64 * 2 {
            // High backpressure - increase buffer
            debug!(
                "High backpressure detected (lag: {}), increasing buffer size",
                lag
            );
            (current_size * 2).min(config.max_buffer_size)
        } else if lag < current_size as u64 / 4 {
            // Low utilization - decrease buffer
            debug!(
                "Low buffer utilization (lag: {}), decreasing buffer size",
                lag
            );
            (current_size / 2).max(config.min_buffer_size)
        } else {
            // Keep current size
            current_size
        };

        if new_size != current_size {
            *last_adj = now;
            warn!(
                "Adjusting buffer size from {} to {} (lag: {})",
                current_size, new_size, lag
            );
        }

        new_size
    }
}

/// Wrapper around mpsc sender that applies backpressure
///
/// `BackpressureSender` wraps a standard Tokio mpsc sender and automatically
/// applies backpressure when the consumer can't keep up with the producer.
/// It uses exponential backoff to slow down message production when needed.
///
/// # Examples
///
/// ```rust
/// # use claude_sdk_rs::runtime::{BackpressureMonitor, BackpressureSender};
/// # use tokio::sync::mpsc;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (tx, mut rx) = mpsc::channel(100);
/// let monitor = BackpressureMonitor::new();
///
/// // Create a sender that applies backpressure when lag exceeds 50 messages
/// let sender = BackpressureSender::new(tx, monitor.clone(), 50);
///
/// // Producer task
/// tokio::spawn(async move {
///     for i in 0..1000 {
///         // Automatically slows down if consumer is lagging
///         if let Err(e) = sender.send_with_backpressure(i).await {
///             eprintln!("Send failed: {}", e);
///             break;
///         }
///     }
/// });
///
/// // Consumer task
/// while let Some(msg) = rx.recv().await {
///     monitor.record_consume();
///     // Simulate slow processing
///     tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Backpressure Strategy
///
/// The sender monitors the lag (sent - consumed messages) and when it exceeds
/// the threshold, applies a delay before sending. The delay is calculated as:
/// - `delay = min(lag * 0.1ms, 100ms)`
///
/// This provides gentle slowdown for small lags and caps the maximum delay
/// to prevent excessive blocking.
pub struct BackpressureSender<T> {
    inner: mpsc::Sender<T>,
    monitor: BackpressureMonitor,
    threshold: u64,
}

impl<T> BackpressureSender<T> {
    /// Creates a new backpressure-aware sender with the specified threshold.
    pub fn new(inner: mpsc::Sender<T>, monitor: BackpressureMonitor, threshold: u64) -> Self {
        Self {
            inner,
            monitor,
            threshold,
        }
    }

    /// Send with backpressure handling
    pub async fn send_with_backpressure(&self, value: T) -> Result<(), mpsc::error::SendError<T>> {
        // Check if we need to apply backpressure
        if self.monitor.should_apply_backpressure(self.threshold) {
            let lag = self.monitor.get_lag();
            debug!("Applying backpressure (lag: {}), waiting before send", lag);

            // Simple exponential backoff based on lag
            let delay = Duration::from_millis((lag as f64 * 0.1).min(100.0) as u64);
            tokio::time::sleep(delay).await;
        }

        self.monitor.record_send();
        self.inner.send(value).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backpressure_monitor() {
        let monitor = BackpressureMonitor::new();

        // Simulate sending 10 messages
        for _ in 0..10 {
            monitor.record_send();
        }

        assert_eq!(monitor.get_lag(), 10);

        // Simulate consuming 5 messages
        for _ in 0..5 {
            monitor.record_consume();
        }

        assert_eq!(monitor.get_lag(), 5);

        // Test backpressure threshold
        assert!(monitor.should_apply_backpressure(3));
        assert!(!monitor.should_apply_backpressure(10));
    }

    #[tokio::test]
    async fn test_consumption_rate() {
        let monitor = BackpressureMonitor::new();

        // Simulate consuming 100 messages
        for _ in 0..100 {
            monitor.record_consume();
        }

        let rate = monitor.get_consumption_rate(Duration::from_secs(10));
        assert_eq!(rate, 10.0); // 100 messages / 10 seconds = 10 msg/s
    }
}
