use crate::{cli::cli::commands::*, cli::error::Result};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use std::path::PathBuf;

/// Interactive CLI for managing multiple Claude sessions and agents
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress output (overrides verbose)
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Timeout for operations in seconds
    #[arg(long, global = true, default_value_t = crate::cli::DEFAULT_TIMEOUT_SECS)]
    pub timeout: u64,

    /// Custom data directory path
    #[arg(long, global = true)]
    pub data_dir: Option<std::path::PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// List available Claude commands
    List(ListCommand),

    /// Manage Claude sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// Run a Claude command
    Run(RunCommand),

    /// View cost information
    Cost(CostCommand),

    /// Search command history
    History(HistoryCommand),

    /// Generate shell completion scripts
    Completion {
        /// Shell to generate completion for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Manage configuration settings
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

impl Cli {
    /// Execute the CLI command
    pub async fn execute(self) -> Result<()> {
        // Load configuration from file, then merge with CLI args
        let config = crate::cli::config::Config::load_default()
            .unwrap_or_else(|_| crate::cli::config::Config::default())
            .merge_with_cli_args(&self);

        // Set up data directory (prefer CLI arg, then config, then default)
        let data_dir: PathBuf = if let Some(dir) = self.data_dir {
            if !dir.exists() {
                std::fs::create_dir_all(&dir)?;
            }
            dir
        } else if let Some(dir) = config.data_dir.clone() {
            if !dir.exists() {
                std::fs::create_dir_all(&dir)?;
            }
            dir
        } else {
            crate::cli::ensure_data_dir()?
        };

        // Execute the specific command
        match self.command {
            Commands::List(cmd) => cmd.execute(&data_dir).await,
            Commands::Session { action } => action.execute(&data_dir).await,
            Commands::Run(cmd) => cmd.execute(&data_dir).await,
            Commands::Cost(cmd) => cmd.execute(&data_dir).await,
            Commands::History(cmd) => cmd.execute(&data_dir).await,
            Commands::Completion { shell } => {
                generate_completion(shell);
                Ok(())
            }
            Commands::Config { action } => action.execute(&data_dir, &config).await,
        }
    }
}

/// Generate shell completion script
fn generate_completion(shell: Shell) {
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    generate(shell, &mut cmd, name, &mut std::io::stdout());
}
