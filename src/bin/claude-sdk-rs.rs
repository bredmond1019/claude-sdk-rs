use anyhow::Result;
use clap::Parser;
use claude_sdk_rs::cli::cli::Cli;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing with environment-based filtering
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    info!("Starting claude-sdk-rs CLI");

    // Parse CLI arguments
    let cli = Cli::parse();

    // Execute command with user-friendly error handling
    if let Err(e) = cli.execute().await {
        // Log the full error for debugging
        error!("Command execution failed: {:?}", e);

        // Display user-friendly error message
        eprintln!("Error: {}", e.user_message());

        // Exit with error code
        std::process::exit(1);
    }

    info!("Command completed successfully");
    Ok(())
}
