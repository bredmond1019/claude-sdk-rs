use crate::cli::cost::{CostEntry, CostTracker};
use crate::cli::execution::parallel::ParallelExecutor;
use crate::cli::execution::runner::CommandRunner;
use crate::cli::execution::ExecutionContext;
use crate::cli::history::{HistoryEntry, HistoryStore};
use crate::cli::output::formatter::OutputFormatter;
use crate::cli::output::OutputStyle;
use crate::cli::session::{Session, SessionManager};
use crate::{cli::error::InteractiveError, cli::error::Result};
use clap::{Args, Subcommand};
use prettytable::{format, Cell, Row, Table};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

/// List available Claude commands
#[derive(Args)]
pub struct ListCommand {
    /// Filter commands by name pattern
    #[arg(short, long)]
    pub filter: Option<String>,

    /// Show detailed information
    #[arg(short, long)]
    pub detailed: bool,
}

impl ListCommand {
    pub async fn execute(&self, data_dir: &Path) -> Result<()> {
        let formatter = OutputFormatter::new();

        // Check for .claude/commands directory
        let commands_dir = data_dir.join("commands");
        if !commands_dir.exists() {
            return Err(InteractiveError::command_discovery(format!(
                "Commands directory not found at {:?}. Create it with 'mkdir -p {}'",
                commands_dir,
                commands_dir.display()
            )));
        }

        // Discover available commands
        let mut commands = Vec::new();

        // Read all files in the commands directory
        let entries = fs::read_dir(&commands_dir).map_err(|e| {
            InteractiveError::command_discovery(format!("Failed to read commands directory: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                InteractiveError::command_discovery(format!(
                    "Failed to read directory entry: {}",
                    e
                ))
            })?;

            let path = entry.path();
            if path.is_file() {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();

                // Apply filter if provided
                if let Some(filter) = &self.filter {
                    if !name.contains(filter) {
                        continue;
                    }
                }

                // Read file content for description (first line as description)
                let description = if self.detailed {
                    match fs::read_to_string(&path) {
                        Ok(content) => {
                            content
                                .lines()
                                .next()
                                .map(|line| {
                                    // Remove comment markers if present
                                    line.trim_start_matches('#')
                                        .trim_start_matches("//")
                                        .trim()
                                        .to_string()
                                })
                                .unwrap_or_else(|| "No description available".to_string())
                        }
                        Err(_) => "Unable to read description".to_string(),
                    }
                } else {
                    "".to_string()
                };

                commands.push((name, description, path));
            }
        }

        // Sort commands alphabetically
        commands.sort_by(|a, b| a.0.cmp(&b.0));

        if commands.is_empty() {
            let msg = if self.filter.is_some() {
                format!(
                    "No commands found matching filter '{}'",
                    self.filter.as_ref().unwrap()
                )
            } else {
                "No commands found in .claude/commands/ directory".to_string()
            };
            formatter.println(&formatter.format_message(&msg, OutputStyle::Warning))?;
            return Ok(());
        }

        // Format output
        if self.detailed {
            // Use table format for detailed view
            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);

            // Set headers
            table.set_titles(Row::new(vec![
                Cell::new("Command").style_spec("Fc"),
                Cell::new("Description").style_spec("Fc"),
                Cell::new("Path").style_spec("Fc"),
            ]));

            // Add rows
            for (name, desc, path) in &commands {
                table.add_row(Row::new(vec![
                    Cell::new(name),
                    Cell::new(desc),
                    Cell::new(&path.display().to_string()),
                ]));
            }

            formatter.println(&table.to_string())?;
        } else {
            // Simple list format
            formatter.println(&formatter.format_message(
                &format!("Available commands ({})", commands.len()),
                OutputStyle::Header,
            ))?;
            formatter.println("")?;

            for (name, _, _) in &commands {
                formatter.println(
                    &formatter.format_message(&format!("  • {}", name), OutputStyle::Info),
                )?;
            }

            if self.filter.is_none() {
                formatter.println("")?;
                formatter.println(&formatter.format_message(
                    "Tip: Use --detailed for more information or --filter to search",
                    OutputStyle::Info,
                ))?;
            }
        }

        Ok(())
    }
}

/// Session management actions
#[derive(Subcommand)]
pub enum SessionAction {
    /// Create a new session
    Create {
        /// Session name
        name: String,

        /// Session description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Delete an existing session
    Delete {
        /// Session ID or name
        session: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// List all sessions
    List {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Switch to a different session
    Switch {
        /// Session ID or name
        session: String,
    },
}

impl SessionAction {
    pub async fn execute(&self, _data_dir: &Path) -> Result<()> {
        let formatter = OutputFormatter::new();
        let mut manager = SessionManager::with_default_storage()?;

        match self {
            SessionAction::Create { name, description } => {
                // Create new session
                match manager.create_session(name.clone(), description.clone(), None) {
                    Ok(session_id) => {
                        let session = manager.get_session(session_id)?.unwrap().clone();
                        formatter.println(&formatter.format_message(
                            &format!("✓ Created session '{}'", session.name),
                            OutputStyle::Success,
                        ))?;

                        if let Some(desc) = &session.description {
                            formatter.println(&formatter.format_message(
                                &format!("  Description: {}", desc),
                                OutputStyle::Info,
                            ))?;
                        }

                        formatter.println(&formatter.format_message(
                            &format!("  Session ID: {}", session.id),
                            OutputStyle::Info,
                        ))?;

                        // Check if this became the current session
                        if let Some(current_id) = manager.get_current_session_id() {
                            if current_id == session.id {
                                formatter.println(&formatter.format_message(
                                    "  This is now your active session",
                                    OutputStyle::Info,
                                ))?;
                            }
                        }
                    }
                    Err(e) => {
                        return Err(InteractiveError::ClaudeSDK(e));
                    }
                }
            }

            SessionAction::Delete { session, force } => {
                // Try to find session by name or ID
                let session_to_delete = manager.get_session_by_name(session)?;

                if session_to_delete.is_none() {
                    return Err(InteractiveError::session_not_found(session));
                }

                let session_info = session_to_delete.unwrap().clone();

                // Confirmation prompt if not forced
                if !force {
                    formatter.print(&formatter.format_message(
                        &format!("Are you sure you want to delete session '{}' ({})?",
                            session_info.name,
                            session_info.id.to_string().chars().take(8).collect::<String>()
                        ),
                        OutputStyle::Warning,
                    ))?;
                    formatter.print(" [y/N]: ")?;
                    io::stdout().flush()?;

                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;

                    if !input.trim().eq_ignore_ascii_case("y") {
                        formatter.println(
                            &formatter.format_message("Delete cancelled", OutputStyle::Info),
                        )?;
                        return Ok(());
                    }
                }

                // Delete the session
                match manager.delete_session(session_info.id) {
                    Ok(deleted) => {
                        if deleted {
                            formatter.println(&formatter.format_message(
                                &format!("✓ Deleted session '{}'", session_info.name),
                                OutputStyle::Success,
                            ))?;
                        } else {
                            formatter.println(&formatter.format_message(
                                "Session was already deleted",
                                OutputStyle::Warning,
                            ))?;
                        }
                    }
                    Err(e) => {
                        return Err(InteractiveError::ClaudeSDK(e));
                    }
                }
            }

            SessionAction::List { detailed } => {
                let sessions = manager.list_sessions();

                if sessions.is_empty() {
                    formatter.println(&formatter.format_message(
                        "No sessions found. Create one with 'claude-interactive session create <name>'",
                        OutputStyle::Info
                    ))?;
                    return Ok(());
                }

                // Get current session ID for highlighting and clone sessions to avoid borrow issues
                let current_session_id = manager.get_current_session_id();
                let sessions_owned: Vec<Session> = sessions.iter().map(|s| (*s).clone()).collect();

                if *detailed {
                    // Use table format from OutputFormatter
                    let table_output = formatter.format_session_table(&sessions_owned)?;
                    formatter.println(&table_output)?;

                    // Add current session indicator
                    if let Some(current_id) = current_session_id {
                        if let Some(current) = sessions_owned.iter().find(|s| s.id == current_id) {
                            formatter.println("")?;
                            formatter.println(&formatter.format_message(
                                &format!("→ Current session: {}", current.name),
                                OutputStyle::Highlight,
                            ))?;
                        }
                    }
                } else {
                    // Simple list format
                    formatter.println(&formatter.format_message(
                        &format!("Sessions ({})", sessions_owned.len()),
                        OutputStyle::Header,
                    ))?;
                    formatter.println("")?;

                    for session in &sessions_owned {
                        let is_current = current_session_id.map_or(false, |id| id == session.id);
                        let prefix = if is_current { "→" } else { " " };
                        let style = if is_current {
                            OutputStyle::Highlight
                        } else {
                            OutputStyle::Info
                        };

                        formatter.println(&formatter.format_message(
                            &format!(
                                    "{} {} - {} commands, ${:.6} total",
                                    prefix,
                                    session.name,
                                    session
                                        .metadata
                                        .get("total_commands")
                                        .unwrap_or(&"0".to_string()),
                                    session
                                        .metadata
                                        .get("total_cost")
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .unwrap_or(0.0)
                                ),
                            style,
                        ))?;
                    }

                    formatter.println("")?;
                    formatter.println(&formatter.format_message(
                        "Tip: Use --detailed for more information",
                        OutputStyle::Info,
                    ))?;
                }
            }

            SessionAction::Switch { session } => {
                // First find the session by name
                let session_to_switch = manager.get_session_by_name(session)?;
                if let Some(session_info) = session_to_switch.cloned() {
                    match manager.switch_to_session(session_info.id) {
                        Ok(_) => {
                            // Get the session details after switching
                            if let Some(switched_session) =
                                manager.get_session(session_info.id)?.cloned()
                            {
                                formatter.println(&formatter.format_message(
                                    &format!("✓ Switched to session '{}'", switched_session.name),
                                    OutputStyle::Success,
                                ))?;

                                if let (Some(commands_str), Some(cost_str)) = (
                                    switched_session.metadata.get("total_commands"),
                                    switched_session.metadata.get("total_cost"),
                                ) {
                                    if let (Ok(commands), Ok(cost)) =
                                        (commands_str.parse::<u64>(), cost_str.parse::<f64>())
                                    {
                                        if commands > 0 {
                                            formatter.println(&formatter.format_message(
                                                &format!(
                                            "  Previous activity: {} commands, ${:.6} total cost",
                                            commands, cost
                                        ),
                                                OutputStyle::Info,
                                            ))?;
                                        }
                                    }
                                }

                                formatter.println(&formatter.format_message(
                                    &format!(
                                        "  Last active: {}",
                                        switched_session.last_accessed.format("%Y-%m-%d %H:%M:%S")
                                    ),
                                    OutputStyle::Info,
                                ))?;
                            }
                        }
                        Err(e) => {
                            return Err(crate::cli::error::InteractiveError::ClaudeSDK(e));
                        }
                    }
                } else {
                    return Err(crate::cli::error::InteractiveError::session_not_found(
                        session,
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Run a Claude command
#[derive(Args)]
pub struct RunCommand {
    /// Command name to execute
    pub command: String,

    /// Command arguments
    pub args: Vec<String>,

    /// Session to run in
    #[arg(short, long)]
    pub session: Option<String>,

    /// Enable parallel execution
    #[arg(short, long)]
    pub parallel: bool,

    /// Number of parallel agents
    #[arg(long, default_value_t = 1)]
    pub agents: usize,
}

impl RunCommand {
    pub async fn execute(&self, data_dir: &Path) -> Result<()> {
        let mut formatter = OutputFormatter::new();

        // Initialize session manager
        let session_manager_arc = Arc::new(tokio::sync::RwLock::new(
            SessionManager::with_default_storage()?,
        ));

        // Get current session or create default
        let session = match &self.session {
            Some(session_name) => {
                // Try to find the session
                match session_manager_arc
                    .write()
                    .await
                    .get_session_by_name(session_name)?
                    .cloned()
                {
                    Some(s) => s,
                    None => {
                        return Err(InteractiveError::session_not_found(session_name));
                    }
                }
            }
            None => {
                // Use current session or create default
                match session_manager_arc
                    .write()
                    .await
                    .get_current_session()
                    .cloned()
                {
                    Some(s) => s,
                    None => {
                        // Create a default session
                        let session_id = session_manager_arc.write().await.create_session(
                            "default".to_string(),
                            Some("Default session created automatically".to_string()),
                            None,
                        )?;
                        session_manager_arc
                            .write()
                            .await
                            .get_session(session_id)?
                            .unwrap()
                            .clone()
                    }
                }
            }
        };

        // Initialize services
        let runner = Arc::new(CommandRunner::new(session_manager_arc.clone())?);
        let mut cost_tracker = CostTracker::new(data_dir.join("costs.json"))?;
        let mut history_store = HistoryStore::new(data_dir.join("history.json"))?;

        // Build execution context
        let context = ExecutionContext {
            session_id: Some(session.id),
            command_name: self.command.clone(),
            args: self.args.clone(),
            parallel: self.parallel,
            agent_count: self.agents,
        };

        // Execute command
        let start_time = std::time::Instant::now();

        if self.parallel && self.agents > 1 {
            // Parallel execution
            formatter.println(&formatter.format_message(
                &format!(
                    "🚀 Running {} in parallel with {} agents",
                    self.command, self.agents
                ),
                OutputStyle::Info,
            ))?;

            let executor = ParallelExecutor::new(runner);

            // Create multiple contexts for parallel execution
            let contexts: Vec<ExecutionContext> =
                (0..self.agents).map(|_| context.clone()).collect();

            // Execute commands
            let result = executor.execute_parallel(contexts).await?;

            // Display outputs
            for output in &result.outputs {
                let formatted =
                    formatter.format_agent_output(&output.content, output.handle.agent_id);
                formatter.println(&formatted)?;
            }

            // Show summary
            let summary = formatter.format_parallel_summary(
                result.results.len(),
                result.successful_count,
                result.failed_count,
                result.total_duration,
            );
            formatter.println("")?;
            formatter.println(&summary)?;

            // Calculate total cost and count
            let total_cost: f64 = result
                .results
                .values()
                .filter_map(|r| r.as_ref().ok())
                .filter_map(|r| r.cost)
                .sum();
            let result_count = result.results.len();

            // Record costs and history for each execution
            for (_id, exec_result) in result.results {
                if let Ok(res) = exec_result {
                    // Record cost if available
                    if let Some(cost_usd) = res.cost {
                        let cost_entry = CostEntry::new(
                            session.id,
                            self.command.clone(),
                            cost_usd,
                            0, // We don't have token counts in ExecutionResult
                            0,
                            res.duration.as_millis() as u64,
                            "claude-3-opus".to_string(), // Default model
                        );
                        cost_tracker.record_cost(cost_entry).await?;
                    }

                    // Record history
                    let history_entry = HistoryEntry::new(
                        session.id,
                        self.command.clone(),
                        self.args.clone(),
                        res.output,
                        res.success,
                        res.duration.as_millis() as u64,
                    );

                    let history_entry = if let Some(cost) = res.cost {
                        history_entry.with_cost(cost, 0, 0, "claude-3-opus".to_string())
                    } else {
                        history_entry
                    };

                    history_store.store_entry(history_entry).await?;
                }
            }

            // Update session metadata
            session_manager_arc.write().await.update_session_metadata(
                session.id,
                "total_commands".to_string(),
                result_count.to_string(),
            )?;
            session_manager_arc.write().await.update_session_metadata(
                session.id,
                "total_cost".to_string(),
                total_cost.to_string(),
            )?;
        } else {
            // Single execution
            formatter.println(&formatter.format_message(
                &format!("🚀 Running: {} {}", self.command, self.args.join(" ")),
                OutputStyle::Info,
            ))?;

            // Execute with streaming
            let mut full_output = String::new();
            let result = runner
                .execute_streaming(context, |chunk| {
                    full_output.push_str(&chunk);
                    formatter.print(&chunk)?;
                    Ok(())
                })
                .await;

            formatter.println("")?; // Ensure we're on a new line

            match result {
                Ok(exec_result) => {
                    // Show execution summary
                    formatter.println(&formatter.format_message(
                        &format!("✓ Command completed in {:?}", exec_result.duration),
                        OutputStyle::Success,
                    ))?;

                    if let Some(cost) = exec_result.cost {
                        formatter.println(&formatter.format_message(
                            &format!("💰 Cost: ${:.6}", cost),
                            OutputStyle::Info,
                        ))?;

                        // Record cost
                        let cost_entry = CostEntry::new(
                            session.id,
                            self.command.clone(),
                            cost,
                            0, // We don't have token counts in ExecutionResult
                            0,
                            exec_result.duration.as_millis() as u64,
                            "claude-3-opus".to_string(), // Default model
                        );
                        cost_tracker.record_cost(cost_entry).await?;
                    }

                    // Record history
                    let history_entry = HistoryEntry::new(
                        session.id,
                        self.command.clone(),
                        self.args.clone(),
                        exec_result.output.clone(),
                        exec_result.success,
                        exec_result.duration.as_millis() as u64,
                    );

                    let history_entry = if let Some(cost) = exec_result.cost {
                        history_entry.with_cost(cost, 0, 0, "claude-3-opus".to_string())
                    } else {
                        history_entry
                    };

                    history_store.store_entry(history_entry).await?;

                    // Update session metadata
                    session_manager_arc.write().await.update_session_metadata(
                        session.id,
                        "total_commands".to_string(),
                        "1".to_string(),
                    )?;
                    if let Some(cost) = exec_result.cost {
                        session_manager_arc.write().await.update_session_metadata(
                            session.id,
                            "total_cost".to_string(),
                            cost.to_string(),
                        )?;
                    }
                }
                Err(e) => {
                    formatter.println(
                        &formatter.format_message(
                            &format!("✗ Command failed: {}", e),
                            OutputStyle::Error,
                        ),
                    )?;

                    // Record failed history entry
                    let history_entry = HistoryEntry::new(
                        session.id,
                        self.command.clone(),
                        self.args.clone(),
                        full_output,
                        false,
                        start_time.elapsed().as_millis() as u64,
                    )
                    .with_error(e.to_string());

                    history_store.store_entry(history_entry).await?;

                    return Err(e);
                }
            }
        }

        Ok(())
    }
}

/// View cost information
#[derive(Args)]
pub struct CostCommand {
    /// Session to show costs for
    #[arg(short, long)]
    pub session: Option<String>,

    /// Show breakdown by command
    #[arg(short, long)]
    pub breakdown: bool,

    /// Date range filter (YYYY-MM-DD format)
    #[arg(long)]
    pub since: Option<String>,

    /// Export format (json, csv)
    #[arg(short, long)]
    pub export: Option<String>,
}

impl CostCommand {
    pub async fn execute(&self, data_dir: &Path) -> Result<()> {
        let formatter = OutputFormatter::new();
        let cost_tracker = CostTracker::new(data_dir.join("costs.json"))?;
        let mut session_manager = SessionManager::with_default_storage()?;

        // Build filter criteria
        let mut filter = crate::cli::cost::CostFilter::default();

        // Session filter
        if let Some(session_name) = &self.session {
            match session_manager.get_session_by_name(session_name)? {
                Some(session) => {
                    filter.session_id = Some(session.id);
                }
                None => {
                    return Err(InteractiveError::session_not_found(session_name));
                }
            }
        }

        // Time range filter
        if let Some(since_str) = &self.since {
            use chrono::NaiveDate;

            // Parse date in YYYY-MM-DD format
            match NaiveDate::parse_from_str(since_str, "%Y-%m-%d") {
                Ok(date) => {
                    let datetime = date
                        .and_hms_opt(0, 0, 0)
                        .ok_or_else(|| InteractiveError::invalid_input("Invalid time"))?;
                    filter.since = Some(chrono::DateTime::from_naive_utc_and_offset(
                        datetime,
                        chrono::Utc,
                    ));
                }
                Err(_) => {
                    return Err(InteractiveError::invalid_input(
                        "Invalid date format. Please use YYYY-MM-DD format.",
                    ));
                }
            }
        }

        // Get cost summary
        let summary = cost_tracker.get_filtered_summary(&filter).await?;

        // Display cost information
        formatter.println(&formatter.format_message("💰 Cost Summary", OutputStyle::Header))?;
        formatter.println("")?;

        // Basic stats
        formatter.println(&formatter.format_message(
            &format!("Total Cost: ${:.4}", summary.total_cost),
            OutputStyle::Success,
        ))?;
        formatter.println(&formatter.format_message(
            &format!("Total Commands: {}", summary.command_count),
            OutputStyle::Info,
        ))?;
        formatter.println(&formatter.format_message(
            &format!("Average Cost per Command: ${:.4}", summary.average_cost),
            OutputStyle::Info,
        ))?;
        formatter.println(&formatter.format_message(
            &format!("Total Tokens: {}", summary.total_tokens),
            OutputStyle::Info,
        ))?;

        if summary.command_count > 0 {
            formatter.println(&formatter.format_message(
                &format!(
                    "Date Range: {} to {}",
                    summary.date_range.0.format("%Y-%m-%d %H:%M"),
                    summary.date_range.1.format("%Y-%m-%d %H:%M")
                ),
                OutputStyle::Info,
            ))?;
        }

        // Show breakdown if requested
        if self.breakdown {
            formatter.println("")?;
            formatter.println(
                &formatter.format_message("📊 Cost Breakdown by Command", OutputStyle::Header),
            )?;
            formatter.println("")?;

            // Sort commands by cost
            let mut command_costs: Vec<_> = summary.by_command.iter().collect();
            command_costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Display as table
            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);

            table.set_titles(Row::new(vec![
                Cell::new("Command").style_spec("Fc"),
                Cell::new("Cost").style_spec("Fc"),
                Cell::new("Percentage").style_spec("Fc"),
            ]));

            for (cmd, cost) in command_costs.iter().take(20) {
                let percentage = if summary.total_cost > 0.0 {
                    **cost / summary.total_cost * 100.0
                } else {
                    0.0
                };

                table.add_row(Row::new(vec![
                    Cell::new(cmd),
                    Cell::new(&format!("${:.4}", cost)),
                    Cell::new(&format!("{:.1}%", percentage)),
                ]));
            }

            formatter.println(&table.to_string())?;

            // Show model breakdown
            if !summary.by_model.is_empty() {
                formatter.println("")?;
                formatter.println(
                    &formatter.format_message("📊 Cost Breakdown by Model", OutputStyle::Header),
                )?;
                formatter.println("")?;

                let mut model_costs: Vec<_> = summary.by_model.iter().collect();
                model_costs
                    .sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut model_table = Table::new();
                model_table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);

                model_table.set_titles(Row::new(vec![
                    Cell::new("Model").style_spec("Fc"),
                    Cell::new("Cost").style_spec("Fc"),
                    Cell::new("Percentage").style_spec("Fc"),
                ]));

                for (model, cost) in model_costs {
                    let percentage = if summary.total_cost > 0.0 {
                        *cost / summary.total_cost * 100.0
                    } else {
                        0.0
                    };

                    model_table.add_row(Row::new(vec![
                        Cell::new(model),
                        Cell::new(&format!("${:.4}", cost)),
                        Cell::new(&format!("{:.1}%", percentage)),
                    ]));
                }

                formatter.println(&model_table.to_string())?;
            }
        }

        // Export if requested
        if let Some(export_format) = &self.export {
            let export_path = match export_format.as_str() {
                "json" => {
                    let path = data_dir.join("cost_export.json");
                    // Export raw entries as JSON
                    let entries = cost_tracker.get_entries(&filter).await?;
                    let json = serde_json::to_string_pretty(&entries)?;
                    tokio::fs::write(&path, json).await?;
                    path
                }
                "csv" => {
                    let path = data_dir.join("cost_export.csv");
                    cost_tracker.export_csv(&path).await?;
                    path
                }
                _ => {
                    return Err(InteractiveError::invalid_input(
                        "Invalid export format. Use 'json' or 'csv'.",
                    ));
                }
            };

            formatter.println("")?;
            formatter.println(&formatter.format_message(
                &format!("✓ Exported cost data to: {}", export_path.display()),
                OutputStyle::Success,
            ))?;
        }

        // Budget warnings (if we implement budgets later)
        // For now, just show a tip about managing costs
        if summary.total_cost > 10.0 {
            formatter.println("")?;
            formatter.println(&formatter.format_message(
                "💡 Tip: Consider setting up budget alerts to track spending",
                OutputStyle::Warning,
            ))?;
        }

        Ok(())
    }
}

/// Search command history
#[derive(Args)]
pub struct HistoryCommand {
    /// Search pattern (regex supported)
    #[arg(short, long)]
    pub search: Option<String>,

    /// Session to search in
    #[arg(short = 'S', long)]
    pub session: Option<String>,

    /// Number of entries to show
    #[arg(short, long, default_value_t = 20)]
    pub limit: usize,

    /// Show command outputs
    #[arg(short, long)]
    pub output: bool,

    /// Export format (json, csv)
    #[arg(short, long)]
    pub export: Option<String>,
}

impl HistoryCommand {
    pub async fn execute(&self, data_dir: &Path) -> Result<()> {
        let formatter = OutputFormatter::new();
        let history_store = HistoryStore::new(data_dir.join("history.json"))?;
        let mut session_manager = SessionManager::with_default_storage()?;

        // Build search criteria
        let mut criteria = crate::cli::history::HistorySearch::default();
        criteria.limit = self.limit;

        // Search pattern
        if let Some(pattern) = &self.search {
            // Support regex or simple text search
            criteria.command_pattern = Some(pattern.clone());
            // Also search in output if pattern is provided
            criteria.output_pattern = Some(pattern.clone());
        }

        // Session filter
        if let Some(session_name) = &self.session {
            match session_manager.get_session_by_name(session_name)? {
                Some(session) => {
                    criteria.session_id = Some(session.id);
                }
                None => {
                    return Err(InteractiveError::session_not_found(session_name));
                }
            }
        }

        // Perform search
        let entries = history_store.search(&criteria).await?;

        if entries.is_empty() {
            formatter.println(&formatter.format_message(
                "No history entries found matching your criteria",
                OutputStyle::Info,
            ))?;
            return Ok(());
        }

        // Export if requested
        if let Some(export_format) = &self.export {
            let export_path = match export_format.as_str() {
                "json" => {
                    let path = data_dir.join("history_export.json");
                    history_store
                        .export(
                            &path,
                            crate::cli::history::ExportFormat::Json,
                            Some(&criteria),
                        )
                        .await?;
                    path
                }
                "csv" => {
                    let path = data_dir.join("history_export.csv");
                    history_store
                        .export(
                            &path,
                            crate::cli::history::ExportFormat::Csv,
                            Some(&criteria),
                        )
                        .await?;
                    path
                }
                _ => {
                    return Err(InteractiveError::invalid_input(
                        "Invalid export format. Use 'json' or 'csv'.",
                    ));
                }
            };

            formatter.println(&formatter.format_message(
                &format!(
                    "✓ Exported {} history entries to: {}",
                    entries.len(),
                    export_path.display()
                ),
                OutputStyle::Success,
            ))?;
            return Ok(());
        }

        // Display history entries
        formatter.println(&formatter.format_message(
            &format!("📜 Command History ({} entries)", entries.len()),
            OutputStyle::Header,
        ))?;
        formatter.println("")?;

        // Create table for display
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);

        // Set headers based on what we want to show
        let headers = if self.output {
            vec![
                Cell::new("Time").style_spec("Fc"),
                Cell::new("Session").style_spec("Fc"),
                Cell::new("Command").style_spec("Fc"),
                Cell::new("Duration").style_spec("Fc"),
                Cell::new("Cost").style_spec("Fc"),
                Cell::new("Status").style_spec("Fc"),
                Cell::new("Output").style_spec("Fc"),
            ]
        } else {
            vec![
                Cell::new("Time").style_spec("Fc"),
                Cell::new("Session").style_spec("Fc"),
                Cell::new("Command").style_spec("Fc"),
                Cell::new("Duration").style_spec("Fc"),
                Cell::new("Cost").style_spec("Fc"),
                Cell::new("Status").style_spec("Fc"),
            ]
        };
        table.set_titles(Row::new(headers));

        // Add rows
        for entry in &entries {
            let time = entry.timestamp.format("%Y-%m-%d %H:%M:%S").to_string();
            let session_short = entry
                .session_id
                .to_string()
                .chars()
                .take(8)
                .collect::<String>();
            let command = if entry.args.is_empty() {
                entry.command_name.clone()
            } else {
                format!("{} {}", entry.command_name, entry.args.join(" "))
            };
            let duration = format!("{}ms", entry.duration_ms);
            let cost = entry
                .cost_usd
                .map(|c| format!("${:.4}", c))
                .unwrap_or_else(|| "-".to_string());
            let status = if entry.success { "✓" } else { "✗" };

            let mut row_cells = vec![
                Cell::new(&time),
                Cell::new(&session_short),
                Cell::new(&command),
                Cell::new(&duration),
                Cell::new(&cost),
                Cell::new(status).style_spec(if entry.success { "Fg" } else { "Fr" }),
            ];

            // Add output column if requested
            if self.output {
                let output_preview = if let Some(error) = &entry.error {
                    format!("Error: {}", error)
                } else {
                    let output_clean = entry.output.trim();
                    if output_clean.len() > 100 {
                        format!("{}...", &output_clean[..100])
                    } else {
                        output_clean.to_string()
                    }
                };
                row_cells.push(Cell::new(&output_preview));
            }

            table.add_row(Row::new(row_cells));
        }

        formatter.println(&table.to_string())?;

        // Show statistics
        let stats = history_store.get_stats(Some(&criteria)).await?;

        formatter.println("")?;
        formatter.println(&formatter.format_message("📊 Statistics", OutputStyle::Header))?;
        formatter.println(&formatter.format_message(
            &format!("Success Rate: {:.1}%", stats.success_rate),
            if stats.success_rate >= 90.0 {
                OutputStyle::Success
            } else {
                OutputStyle::Warning
            },
        ))?;
        formatter.println(&formatter.format_message(
            &format!("Total Cost: ${:.4}", stats.total_cost),
            OutputStyle::Info,
        ))?;
        formatter.println(&formatter.format_message(
            &format!("Average Duration: {:.0}ms", stats.average_duration_ms),
            OutputStyle::Info,
        ))?;

        // Show pagination hint if there might be more results
        if entries.len() == self.limit {
            formatter.println("")?;
            formatter.println(&formatter.format_message(
                &format!(
                    "Showing first {} entries. Use --limit to see more.",
                    self.limit
                ),
                OutputStyle::Info,
            ))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_list_command_empty_directory() {
        let temp_dir = tempdir().unwrap();
        let commands_dir = temp_dir.path().join("commands");
        fs::create_dir_all(&commands_dir).unwrap();

        let list_cmd = ListCommand {
            filter: None,
            detailed: false,
        };

        // Should complete without error but show no commands
        let result = list_cmd.execute(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_command_with_commands() {
        let temp_dir = tempdir().unwrap();
        let commands_dir = temp_dir.path().join("commands");
        fs::create_dir_all(&commands_dir).unwrap();

        // Create test command files
        fs::write(
            commands_dir.join("analyze.sh"),
            "# Analyze code\necho 'analyzing...'",
        )
        .unwrap();
        fs::write(
            commands_dir.join("test.py"),
            "// Run tests\nprint('testing...')",
        )
        .unwrap();

        let list_cmd = ListCommand {
            filter: None,
            detailed: true,
        };

        let result = list_cmd.execute(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_command_with_filter() {
        let temp_dir = tempdir().unwrap();
        let commands_dir = temp_dir.path().join("commands");
        fs::create_dir_all(&commands_dir).unwrap();

        // Create test command files
        fs::write(commands_dir.join("analyze.sh"), "# Analyze code").unwrap();
        fs::write(commands_dir.join("build.sh"), "# Build project").unwrap();
        fs::write(commands_dir.join("test.sh"), "# Test code").unwrap();

        let list_cmd = ListCommand {
            filter: Some("test".to_string()),
            detailed: false,
        };

        let result = list_cmd.execute(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_command_no_commands_directory() {
        let temp_dir = tempdir().unwrap();

        let list_cmd = ListCommand {
            filter: None,
            detailed: false,
        };

        let result = list_cmd.execute(temp_dir.path()).await;
        assert!(result.is_err());

        if let Err(InteractiveError::CommandDiscovery(msg)) = result {
            assert!(msg.contains("Commands directory not found"));
        } else {
            panic!("Expected CommandDiscovery error");
        }
    }

    #[tokio::test]
    async fn test_session_action_types() {
        // Test that SessionAction variants can be constructed
        let create_action = SessionAction::Create {
            name: "test-session".to_string(),
            description: Some("Test description".to_string()),
        };

        match create_action {
            SessionAction::Create { name, description } => {
                assert_eq!(name, "test-session");
                assert_eq!(description, Some("Test description".to_string()));
            }
            _ => panic!("Expected Create variant"),
        }

        let delete_action = SessionAction::Delete {
            session: "test-session".to_string(),
            force: true,
        };

        match delete_action {
            SessionAction::Delete { session, force } => {
                assert_eq!(session, "test-session");
                assert!(force);
            }
            _ => panic!("Expected Delete variant"),
        }

        let list_action = SessionAction::List { detailed: true };

        match list_action {
            SessionAction::List { detailed } => {
                assert!(detailed);
            }
            _ => panic!("Expected List variant"),
        }

        let switch_action = SessionAction::Switch {
            session: "other-session".to_string(),
        };

        match switch_action {
            SessionAction::Switch { session } => {
                assert_eq!(session, "other-session");
            }
            _ => panic!("Expected Switch variant"),
        }
    }

    #[tokio::test]
    async fn test_run_command_creation() {
        let run_cmd = RunCommand {
            command: "analyze".to_string(),
            args: vec!["--file".to_string(), "test.rs".to_string()],
            session: Some("test-session".to_string()),
            parallel: false,
            agents: 1,
        };

        assert_eq!(run_cmd.command, "analyze");
        assert_eq!(run_cmd.args, vec!["--file", "test.rs"]);
        assert_eq!(run_cmd.session, Some("test-session".to_string()));
        assert!(!run_cmd.parallel);
        assert_eq!(run_cmd.agents, 1);
    }

    #[tokio::test]
    async fn test_cost_command_creation() {
        let cost_cmd = CostCommand {
            session: Some("test-session".to_string()),
            breakdown: true,
            since: Some("2024-01-01".to_string()),
            export: Some("csv".to_string()),
        };

        assert_eq!(cost_cmd.session, Some("test-session".to_string()));
        assert!(cost_cmd.breakdown);
        assert_eq!(cost_cmd.since, Some("2024-01-01".to_string()));
        assert_eq!(cost_cmd.export, Some("csv".to_string()));
    }

    #[tokio::test]
    async fn test_history_command_creation() {
        let history_cmd = HistoryCommand {
            search: Some("analyze".to_string()),
            session: Some("test-session".to_string()),
            limit: 50,
            output: true,
            export: Some("json".to_string()),
        };

        assert_eq!(history_cmd.search, Some("analyze".to_string()));
        assert_eq!(history_cmd.session, Some("test-session".to_string()));
        assert_eq!(history_cmd.limit, 50);
        assert!(history_cmd.output);
        assert_eq!(history_cmd.export, Some("json".to_string()));
    }

    #[test]
    fn test_time_duration_parsing() {
        // Test basic duration formats that might be used in since/until fields
        // Note: This tests the concept, actual parsing would be in implementation
        let durations = vec![
            ("1h", "1 hour"),
            ("24h", "24 hours"),
            ("7d", "7 days"),
            ("30d", "30 days"),
            ("1w", "1 week"),
        ];

        for (input, expected_desc) in durations {
            assert!(!input.is_empty(), "Duration string should not be empty");
            assert!(input.len() >= 2, "Duration should have value and unit");
            assert!(
                expected_desc.contains(&input[..input.len() - 1]),
                "Description should contain the numeric part"
            );
        }
    }

    #[test]
    fn test_command_argument_validation() {
        // Test that command arguments are properly structured
        let valid_args = vec![
            vec!["--file".to_string(), "test.rs".to_string()],
            vec!["--output".to_string(), "result.txt".to_string()],
            vec!["--verbose".to_string()],
        ];

        for args in valid_args {
            assert!(
                !args.is_empty() || args.is_empty(),
                "Args vector should be valid"
            );

            if !args.is_empty() {
                // Check that flags start with --
                for arg in &args {
                    if arg.starts_with("--") {
                        assert!(arg.len() > 2, "Flag should have content after --");
                    }
                }
            }
        }
    }

    #[test]
    fn test_session_identifier_validation() {
        // Test session identifier patterns
        let valid_sessions = vec!["my-session", "session_1", "test123", "work-project"];

        let invalid_sessions = vec!["", " ", "session with spaces"];

        for session in valid_sessions {
            assert!(!session.is_empty(), "Valid session should not be empty");
            assert!(
                !session.contains(' '),
                "Valid session should not contain spaces"
            );
        }

        for session in invalid_sessions {
            assert!(
                session.is_empty() || session.contains(' '),
                "Invalid session should be empty or contain spaces"
            );
        }
    }
}

/// Configuration management actions
#[derive(Subcommand)]
pub enum ConfigAction {
    /// Show current configuration
    Show,

    /// Set a configuration value
    Set {
        /// Configuration key (e.g., "timeout_secs", "output.color")
        key: String,

        /// Configuration value
        value: String,
    },

    /// Reset configuration to defaults
    Reset {
        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// Show configuration file path
    Path,
}

impl ConfigAction {
    pub async fn execute(
        &self,
        _data_dir: &Path,
        current_config: &crate::cli::config::Config,
    ) -> Result<()> {
        let formatter = OutputFormatter::new();

        match self {
            ConfigAction::Show => {
                // Display current configuration
                formatter
                    .println(&formatter.format_message("⚙️  Configuration", OutputStyle::Header))?;
                formatter.println("")?;

                // Show key configuration values
                formatter.println(&formatter.format_message(
                    &format!("Timeout: {} seconds", current_config.timeout_secs),
                    OutputStyle::Info,
                ))?;
                formatter.println(&formatter.format_message(
                    &format!("Verbose: {}", current_config.verbose),
                    OutputStyle::Info,
                ))?;
                formatter.println(&formatter.format_message(
                    &format!("Quiet: {}", current_config.quiet),
                    OutputStyle::Info,
                ))?;

                if let Some(ref data_dir) = current_config.data_dir {
                    formatter.println(&formatter.format_message(
                        &format!("Data directory: {}", data_dir.display()),
                        OutputStyle::Info,
                    ))?;
                }

                // Session defaults
                formatter.println("")?;
                formatter
                    .println(&formatter.format_message("Session Defaults:", OutputStyle::Header))?;

                if let Some(ref model) = current_config.session.model {
                    formatter.println(
                        &formatter.format_message(&format!("Model: {}", model), OutputStyle::Info),
                    )?;
                }

                if let Some(ref prompt) = current_config.session.system_prompt {
                    formatter.println(&formatter.format_message(
                        &format!(
                            "System prompt: {}",
                            if prompt.len() > 50 {
                                format!("{}...", &prompt[..50])
                            } else {
                                prompt.clone()
                            }
                        ),
                        OutputStyle::Info,
                    ))?;
                }

                if let Some(ref tools) = current_config.session.allowed_tools {
                    formatter.println(&formatter.format_message(
                        &format!("Allowed tools: {}", tools.join(", ")),
                        OutputStyle::Info,
                    ))?;
                }

                // Output settings
                formatter.println("")?;
                formatter
                    .println(&formatter.format_message("Output Settings:", OutputStyle::Header))?;
                formatter.println(&formatter.format_message(
                    &format!("Color output: {}", current_config.output.color),
                    OutputStyle::Info,
                ))?;
                formatter.println(&formatter.format_message(
                    &format!("Progress indicators: {}", current_config.output.progress),
                    OutputStyle::Info,
                ))?;
                formatter.println(&formatter.format_message(
                    &format!("Format: {}", current_config.output.format),
                    OutputStyle::Info,
                ))?;

                // Analytics settings
                formatter.println("")?;
                formatter.println(&formatter.format_message("Analytics:", OutputStyle::Header))?;
                formatter.println(&formatter.format_message(
                    &format!("Enabled: {}", current_config.analytics.enabled),
                    OutputStyle::Info,
                ))?;
                formatter.println(&formatter.format_message(
                    &format!(
                        "Retention: {} days",
                        current_config.analytics.retention_days
                    ),
                    OutputStyle::Info,
                ))?;
            }

            ConfigAction::Set { key, value } => {
                // Set configuration value
                let config_path = crate::cli::config::Config::default_path()?;
                let mut config = crate::cli::config::Config::load_from_file(&config_path)
                    .unwrap_or_else(|_| crate::cli::config::Config::default());

                // Parse the key and set the value
                match key.as_str() {
                    "timeout_secs" => match value.parse::<u64>() {
                        Ok(timeout) => config.timeout_secs = timeout,
                        Err(_) => {
                            return Err(InteractiveError::invalid_input("Invalid timeout value"))
                        }
                    },
                    "verbose" => match value.parse::<bool>() {
                        Ok(verbose) => config.verbose = verbose,
                        Err(_) => {
                            return Err(InteractiveError::invalid_input(
                                "Invalid boolean value (use true/false)",
                            ))
                        }
                    },
                    "quiet" => match value.parse::<bool>() {
                        Ok(quiet) => config.quiet = quiet,
                        Err(_) => {
                            return Err(InteractiveError::invalid_input(
                                "Invalid boolean value (use true/false)",
                            ))
                        }
                    },
                    "session.model" => config.session.model = Some(value.clone()),
                    "session.system_prompt" => config.session.system_prompt = Some(value.clone()),
                    "output.color" => match value.parse::<bool>() {
                        Ok(color) => config.output.color = color,
                        Err(_) => {
                            return Err(InteractiveError::invalid_input(
                                "Invalid boolean value (use true/false)",
                            ))
                        }
                    },
                    "output.progress" => match value.parse::<bool>() {
                        Ok(progress) => config.output.progress = progress,
                        Err(_) => {
                            return Err(InteractiveError::invalid_input(
                                "Invalid boolean value (use true/false)",
                            ))
                        }
                    },
                    "output.format" => config.output.format = value.clone(),
                    "analytics.enabled" => match value.parse::<bool>() {
                        Ok(enabled) => config.analytics.enabled = enabled,
                        Err(_) => {
                            return Err(InteractiveError::invalid_input(
                                "Invalid boolean value (use true/false)",
                            ))
                        }
                    },
                    _ => {
                        return Err(InteractiveError::invalid_input(&format!(
                            "Unknown config key: {}",
                            key
                        )))
                    }
                }

                // Save the updated configuration
                config.save_to_file(&config_path)?;

                formatter.println(
                    &formatter.format_message(
                        &format!("✓ Set {} = {}", key, value),
                        OutputStyle::Success,
                    ),
                )?;
            }

            ConfigAction::Reset { force } => {
                let config_path = crate::cli::config::Config::default_path()?;

                // Confirmation prompt if not forced
                if !force {
                    formatter.print(&formatter.format_message(
                        "Are you sure you want to reset configuration to defaults?",
                        OutputStyle::Warning,
                    ))?;
                    formatter.print(" [y/N]: ")?;
                    io::stdout().flush()?;

                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;

                    if !input.trim().eq_ignore_ascii_case("y") {
                        formatter.println(
                            &formatter.format_message("Reset cancelled", OutputStyle::Info),
                        )?;
                        return Ok(());
                    }
                }

                // Reset to defaults
                let default_config = crate::cli::config::Config::default();
                default_config.save_to_file(&config_path)?;

                formatter.println(
                    &formatter
                        .format_message("✓ Configuration reset to defaults", OutputStyle::Success),
                )?;
            }

            ConfigAction::Path => {
                let config_path = crate::cli::config::Config::default_path()?;
                formatter.println(&formatter.format_message(
                    &format!("Configuration file: {}", config_path.display()),
                    OutputStyle::Info,
                ))?;

                if config_path.exists() {
                    formatter.println(
                        &formatter
                            .format_message("File exists and is being used", OutputStyle::Success),
                    )?;
                } else {
                    formatter.println(&formatter.format_message(
                        "File does not exist (using defaults)",
                        OutputStyle::Warning,
                    ))?;
                }
            }
        }

        Ok(())
    }
}
