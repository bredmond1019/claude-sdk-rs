#[cfg(test)]
mod tests {
    use claude_sdk_rs_mcp::transport::*;
    use std::time::Duration;

    #[test]
    fn test_transport_type_variants() {
        let stdio = TransportType::Stdio {
            command: "mcp-server".to_string(),
            args: vec!["--port".to_string(), "8080".to_string()],
            auto_restart: true,
            max_restarts: 3,
        };

        match stdio {
            TransportType::Stdio {
                command,
                args,
                auto_restart,
                max_restarts,
            } => {
                assert_eq!(command, "mcp-server");
                assert_eq!(args.len(), 2);
                assert!(auto_restart);
                assert_eq!(max_restarts, 3);
            }
            _ => panic!("Expected Stdio transport"),
        }

        let websocket = TransportType::WebSocket {
            url: "ws://localhost:8080".to_string(),
            heartbeat_interval: Some(Duration::from_secs(30)),
            reconnect_config: ReconnectConfig {
                enabled: true,
                max_attempts: 5,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 2.0,
            },
        };

        match websocket {
            TransportType::WebSocket {
                url,
                heartbeat_interval,
                reconnect_config,
            } => {
                assert_eq!(url, "ws://localhost:8080");
                assert!(heartbeat_interval.is_some());
                assert!(reconnect_config.enabled);
            }
            _ => panic!("Expected WebSocket transport"),
        }
    }

    #[test]
    fn test_transport_error() {
        let error = TransportError::ConnectionError("Connection refused".to_string());
        assert_eq!(error.to_string(), "Connection error: Connection refused");

        let error = TransportError::ProtocolError("Invalid message".to_string());
        assert_eq!(error.to_string(), "Protocol error: Invalid message");

        let error = TransportError::IoError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "EOF",
        ));
        assert!(error.to_string().contains("IO error"));
    }

    #[test]
    fn test_reconnect_config() {
        let config = ReconnectConfig {
            enabled: true,
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        };

        assert!(config.enabled);
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(10));
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_http_pool_config() {
        let config = HttpPoolConfig {
            max_connections_per_host: 10,
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(90),
        };

        assert_eq!(config.max_connections_per_host, 10);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.request_timeout, Duration::from_secs(30));
        assert_eq!(config.keep_alive_timeout, Duration::from_secs(90));
    }
}
