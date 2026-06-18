use crate::core::{Config, Error, Result, StreamFormat};
use crate::runtime::error_handling::{log_error_with_context, ErrorContext, ProcessErrorDetails};
use crate::runtime::telemetry;
use std::collections::HashMap;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration, Instant};
use tracing::debug;

/// Execute a one-shot Claude command with timeout
pub async fn execute_claude(config: &Config, query: &str) -> Result<String> {
    let context = ErrorContext::new("execute_claude")
        .with_debug_info("query_length", query.len().to_string())
        .with_debug_info("stream_format", format!("{:?}", config.stream_format))
        .with_debug_info(
            "timeout_secs",
            config.timeout_secs.unwrap_or(30).to_string(),
        );

    // Check for custom claude binary path first (for nvm compatibility)
    let claude_binary = if let Ok(custom_path) = std::env::var("CLAUDE_BINARY") {
        std::path::PathBuf::from(custom_path)
    } else {
        which::which("claude").map_err(|e| {
            let enhanced_context = context
                .clone()
                .with_error_chain(format!("Binary search failed: {}", e))
                .with_debug_info("search_error", e.to_string());
            let error = Error::BinaryNotFound;
            log_error_with_context(&error, &enhanced_context);

            // Record to telemetry
            let mut telemetry_context = HashMap::new();
            telemetry_context.insert("search_error".to_string(), e.to_string());
            telemetry_context.insert(
                "path_env".to_string(),
                std::env::var("PATH").unwrap_or_default(),
            );
            let error_clone = error.clone();
            tokio::spawn(async move {
                telemetry::record_error(&error_clone, "execute_claude", telemetry_context).await;
            });

            error
        })?
    };

    let mut cmd = Command::new(claude_binary);

    // Always use non-interactive mode for SDK
    cmd.arg("-p");

    // Add format flag
    match config.stream_format {
        StreamFormat::Json => {
            cmd.arg("--output-format").arg("json");
        }
        StreamFormat::StreamJson => {
            cmd.arg("--output-format").arg("stream-json");
            // stream-json requires verbose flag
            cmd.arg("--verbose");
        }
        StreamFormat::Text => {
            // Text is default, no need to specify
        }
    }

    // Add verbose flag if configured (and not already added for stream-json)
    if config.verbose && config.stream_format != StreamFormat::StreamJson {
        cmd.arg("--verbose");
    }

    // Add optional flags
    if let Some(system_prompt) = &config.system_prompt {
        cmd.arg("--system-prompt").arg(system_prompt);
    }

    if let Some(model) = &config.model {
        cmd.arg("--model").arg(model);
    }

    if let Some(mcp_config_path) = &config.mcp_config_path {
        cmd.arg("--mcp-config").arg(mcp_config_path);
    }

    if let Some(allowed_tools) = &config.allowed_tools {
        for tool in allowed_tools {
            cmd.arg("--allowedTools").arg(tool);
        }
        debug!("Added {} allowed tools", allowed_tools.len());
    }

    if let Some(max_tokens) = &config.max_tokens {
        cmd.arg("--max-tokens").arg(max_tokens.to_string());
    }

    // Add session management flags
    if config.continue_session {
        cmd.arg("--continue");
    }

    if let Some(session_id) = &config.resume_session_id {
        cmd.arg("--resume").arg(session_id);
    }

    // Add disallowed tools flags
    if let Some(disallowed_tools) = &config.disallowed_tools {
        for tool in disallowed_tools {
            cmd.arg("--disallowedTools").arg(tool);
        }
        debug!("Added {} disallowed tools", disallowed_tools.len());
    }

    // Add default skip permissions flag (enabled by default)
    if config.skip_permissions {
        cmd.arg("--dangerously-skip-permissions");
    }

    // Add append system prompt flag
    if let Some(append_prompt) = &config.append_system_prompt {
        cmd.arg("--append-system-prompt").arg(append_prompt);
    }

    // Add max turns flag
    if let Some(max_turns) = &config.max_turns {
        cmd.arg("--max-turns").arg(max_turns.to_string());
    }

    // Determine if we should use stdin or command argument
    let use_stdin =
        config.allowed_tools.is_some() && !config.allowed_tools.as_ref().unwrap().is_empty();

    if use_stdin {
        // When tools are present, use stdin for the query
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Executing Claude command with stdin: {:?}", cmd);

        let timeout_duration = Duration::from_secs(config.timeout_secs.unwrap_or(30));
        let mut child = cmd.spawn().map_err(|e| {
            let cmd_line = format!("{:?}", cmd);
            let error_details = ProcessErrorDetails::new(
                format!("Failed to spawn Claude process: {}", e),
                "claude",
                vec![],
            )
            .with_stderr(e.to_string());

            let enhanced_context = context
                .clone()
                .with_error_chain(format!("Process spawn failed: {}", e))
                .with_debug_info("command_line", cmd_line)
                .with_debug_info("spawn_error", e.to_string());

            let process_error = error_details.to_error();
            log_error_with_context(&process_error, &enhanced_context);
            process_error
        })?;

        // Write the query to stdin
        if let Some(stdin) = child.stdin.take() {
            let mut stdin = stdin;
            stdin
                .write_all(query.as_bytes())
                .await
                .map_err(|e| Error::ProcessError(format!("Failed to write to stdin: {}", e)))?;
            stdin
                .flush()
                .await
                .map_err(|e| Error::ProcessError(format!("Failed to flush stdin: {}", e)))?;
            drop(stdin); // Close stdin
        }

        // Wait for the process to complete
        let _start_time = Instant::now();
        let output = timeout(timeout_duration, child.wait_with_output())
            .await
            .map_err(|_| {
                let timeout_secs = config.timeout_secs.unwrap_or(30);
                let error = Error::Timeout(timeout_secs);

                // Record timeout to telemetry
                let mut telemetry_context = HashMap::new();
                telemetry_context.insert("timeout_duration".to_string(), timeout_secs.to_string());
                telemetry_context.insert("query_length".to_string(), query.len().to_string());
                telemetry_context.insert(
                    "stream_format".to_string(),
                    format!("{:?}", config.stream_format),
                );
                let error_clone = error.clone();
                tokio::spawn(async move {
                    telemetry::record_error(&error_clone, "execute_claude", telemetry_context)
                        .await;
                });

                error
            })??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);

            let error_details = ProcessErrorDetails::new(
                "Claude command execution failed",
                "claude",
                vec![], // Args were added to cmd already
            )
            .with_exit_code(output.status.code().unwrap_or(-1))
            .with_stderr(stderr.to_string())
            .with_stdout_preview(stdout.to_string());

            let enhanced_context = context
                .clone()
                .with_error_chain("Process completed with non-zero exit code".to_string())
                .with_debug_info("exit_code", output.status.code().unwrap_or(-1).to_string())
                .with_debug_info("stderr_length", stderr.len().to_string())
                .with_debug_info("stdout_length", stdout.len().to_string());

            let process_error = error_details.to_error();
            log_error_with_context(&process_error, &enhanced_context);
            return Err(process_error);
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout)
    } else {
        // Traditional approach - add query as command argument
        cmd.arg(query);

        debug!("Executing Claude command: {:?}", cmd);

        // Execute the command with timeout
        let timeout_duration = Duration::from_secs(config.timeout_secs.unwrap_or(30));
        let output = timeout(timeout_duration, cmd.output())
            .await
            .map_err(|_| Error::Timeout(config.timeout_secs.unwrap_or(30)))??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);

            let error_details = ProcessErrorDetails::new(
                "Claude command execution failed (traditional mode)",
                "claude",
                vec![], // Args were added to cmd already
            )
            .with_exit_code(output.status.code().unwrap_or(-1))
            .with_stderr(stderr.to_string())
            .with_stdout_preview(stdout.to_string());

            let enhanced_context = context
                .clone()
                .with_error_chain(
                    "Process completed with non-zero exit code (traditional mode)".to_string(),
                )
                .with_debug_info("execution_mode", "traditional")
                .with_debug_info("exit_code", output.status.code().unwrap_or(-1).to_string())
                .with_debug_info("stderr_length", stderr.len().to_string());

            let process_error = error_details.to_error();
            log_error_with_context(&process_error, &enhanced_context);
            return Err(process_error);
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout)
    }
}

/// Execute Claude command with streaming output
///
/// This function spawns a Claude CLI process and returns a stream of output lines.
/// Unlike `execute_claude`, this function provides real-time streaming of the output
/// as it's generated by the CLI process.
///
/// # Streaming Behavior
///
/// The streaming implementation reads Claude CLI output line-by-line and forwards
/// each line as it arrives. This provides true real-time streaming for:
///
/// - **Text format**: Each line is sent as a separate message chunk
/// - **JSON format**: Full response is accumulated and sent once complete
/// - **StreamJson format**: Each JSON line is parsed and sent as individual messages
///
/// # Limitations
///
/// - **Process cleanup**: Child processes are cleaned up when receivers are dropped,
///   but very long-running streams should be manually cancelled
/// - **Buffering**: Output is line-buffered, so partial lines won't be streamed
/// - **Error handling**: Process errors are sent through the stream, but some
///   errors (like authentication failures) may only appear in stderr
/// - **Timeout behavior**: Timeouts apply per-line, not to the entire response
///
/// # Arguments
///
/// * `config` - Configuration for the Claude CLI execution
/// * `query` - The query to send to Claude
///
/// # Returns
///
/// Returns a `Result` containing a receiver that yields `String` lines as they are
/// output by the Claude CLI process.
///
/// # Examples
///
/// ```rust,no_run
/// use claude_sdk_rs::core::Config;
/// use claude_sdk_rs::runtime::process::execute_claude_streaming;
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = Config::default();
///     let mut stream = execute_claude_streaming(&config, "Tell me a story").await?;
///     
///     while let Some(line_result) = stream.recv().await {
///         match line_result {
///             Ok(line) => println!("Received: {}", line),
///             Err(e) => eprintln!("Error: {}", e),
///         }
///     }
///     
///     Ok(())
/// }
/// ```
pub async fn execute_claude_streaming(
    config: &Config,
    query: &str,
) -> Result<mpsc::Receiver<Result<String>>> {
    // Check for custom claude binary path first (for nvm compatibility)
    let claude_binary = if let Ok(custom_path) = std::env::var("CLAUDE_BINARY") {
        std::path::PathBuf::from(custom_path)
    } else {
        which::which("claude").map_err(|_| Error::BinaryNotFound)?
    };

    let mut cmd = Command::new(claude_binary);

    // Always use non-interactive mode for SDK
    cmd.arg("-p");

    // Add format flag
    match config.stream_format {
        StreamFormat::Json => {
            cmd.arg("--output-format").arg("json");
        }
        StreamFormat::StreamJson => {
            cmd.arg("--output-format").arg("stream-json");
            // stream-json requires verbose flag
            cmd.arg("--verbose");
        }
        StreamFormat::Text => {
            // Text is default, no need to specify
        }
    }

    // Add verbose flag if configured (and not already added for stream-json)
    if config.verbose && config.stream_format != StreamFormat::StreamJson {
        cmd.arg("--verbose");
    }

    // Add optional flags
    if let Some(system_prompt) = &config.system_prompt {
        cmd.arg("--system-prompt").arg(system_prompt);
    }

    if let Some(model) = &config.model {
        cmd.arg("--model").arg(model);
    }

    if let Some(mcp_config_path) = &config.mcp_config_path {
        cmd.arg("--mcp-config").arg(mcp_config_path);
    }

    if let Some(allowed_tools) = &config.allowed_tools {
        for tool in allowed_tools {
            cmd.arg("--allowedTools").arg(tool);
        }
        debug!("Added {} allowed tools", allowed_tools.len());
    }

    if let Some(max_tokens) = &config.max_tokens {
        cmd.arg("--max-tokens").arg(max_tokens.to_string());
    }

    // Add session management flags
    if config.continue_session {
        cmd.arg("--continue");
    }

    if let Some(session_id) = &config.resume_session_id {
        cmd.arg("--resume").arg(session_id);
    }

    // Add disallowed tools flags
    if let Some(disallowed_tools) = &config.disallowed_tools {
        for tool in disallowed_tools {
            cmd.arg("--disallowedTools").arg(tool);
        }
        debug!("Added {} disallowed tools", disallowed_tools.len());
    }

    // Add default skip permissions flag (enabled by default)
    if config.skip_permissions {
        cmd.arg("--dangerously-skip-permissions");
    }

    // Add append system prompt flag
    if let Some(append_prompt) = &config.append_system_prompt {
        cmd.arg("--append-system-prompt").arg(append_prompt);
    }

    // Add max turns flag
    if let Some(max_turns) = &config.max_turns {
        cmd.arg("--max-turns").arg(max_turns.to_string());
    }

    // Set up stdio for streaming
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    debug!("Executing Claude command for streaming: {:?}", cmd);

    let mut child = cmd
        .spawn()
        .map_err(|e| Error::ProcessError(format!("Failed to spawn process: {}", e)))?;

    // Write the query to stdin and close it
    if let Some(stdin) = child.stdin.take() {
        let mut stdin = stdin;
        let query_owned = query.to_string();
        tokio::spawn(async move {
            if let Err(e) = stdin.write_all(query_owned.as_bytes()).await {
                debug!("Failed to write to stdin: {}", e);
                return;
            }
            if let Err(e) = stdin.flush().await {
                debug!("Failed to flush stdin: {}", e);
            }
            // stdin is automatically closed when dropped
        });
    }

    // Create channel for streaming output
    let stream_config = crate::runtime::stream_config::get_stream_config();
    let (tx, rx) = mpsc::channel::<Result<String>>(stream_config.channel_buffer_size);

    // Spawn task to read stdout line by line
    if let Some(stdout) = child.stdout.take() {
        let tx_clone = tx.clone();
        let timeout_duration = Duration::from_secs(config.timeout_secs.unwrap_or(30));

        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            loop {
                match timeout(timeout_duration, lines.next_line()).await {
                    Ok(Ok(Some(line))) => {
                        if tx_clone.send(Ok(line)).await.is_err() {
                            debug!("Receiver dropped, stopping stdout reading");
                            break;
                        }
                    }
                    Ok(Ok(None)) => {
                        // EOF reached
                        debug!("Reached EOF on stdout");
                        break;
                    }
                    Ok(Err(e)) => {
                        let _ = tx_clone
                            .send(Err(Error::ProcessError(format!(
                                "Failed to read line: {}",
                                e
                            ))))
                            .await;
                        break;
                    }
                    Err(_) => {
                        let _ = tx_clone.send(Err(Error::Timeout(30))).await;
                        break;
                    }
                }
            }
        });
    }

    // Spawn task to monitor process completion and handle errors
    let tx_error = tx.clone();
    tokio::spawn(async move {
        match child.wait().await {
            Ok(status) if !status.success() => {
                let _ = tx_error
                    .send(Err(Error::ProcessError(
                        "Claude command failed".to_string(),
                    )))
                    .await;
            }
            Err(e) => {
                let _ = tx_error
                    .send(Err(Error::ProcessError(format!("Process error: {}", e))))
                    .await;
            }
            Ok(_) => {
                // Process completed successfully, stdout task should handle EOF
                debug!("Claude process completed successfully");
            }
        }
    });

    Ok(rx)
}
