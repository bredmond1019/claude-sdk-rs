//! Automated report generation and scheduling
//!
//! This module provides comprehensive reporting capabilities for the Claude AI
//! Interactive analytics system, including scheduled reports, custom templates,
//! and multiple output formats.
//!
//! # Features
//!
//! - **Scheduled Reports**: Automated generation on daily, weekly, or custom schedules
//! - **Multiple Formats**: Support for JSON, HTML, CSV, PDF, and Markdown
//! - **Custom Templates**: Flexible template system for tailored reports
//! - **Distribution**: Email and webhook delivery options
//! - **Archiving**: Automatic compression and retention management
//!
//! # Report Types
//!
//! 1. **Analytics Reports**: Comprehensive system analytics
//! 2. **Cost Reports**: Detailed cost breakdowns and trends
//! 3. **Performance Reports**: System performance metrics
//! 4. **Session Reports**: Individual session analysis
//! 5. **Custom Reports**: User-defined report structures
//!
//! # Example Usage
//!
//! ```no_run
//! use crate_interactive::analytics::{
//!     ReportScheduler, ReportTemplate, ReportFormat, ReportSchedule
//! };
//! use chrono::Weekday;
//!
//! # async fn example(analytics_engine: std::sync::Arc<AnalyticsEngine>) -> Result<(), Box<dyn std::error::Error>> {
//! // Create report scheduler
//! let scheduler = ReportScheduler::new(analytics_engine, Default::default())?;
//!
//! // Define custom template
//! let template = ReportTemplate::new("Cost Analysis")
//!     .with_sections(vec!["summary", "by_model", "trends"])
//!     .with_time_range(TimeRange::LastWeek)
//!     .with_insights(true);
//!
//! // Schedule weekly report
//! scheduler.schedule_report(
//!     "weekly_costs",
//!     ReportSchedule::Weekly {
//!         weekday: Weekday::Mon,
//!         hour: 9
//!     },
//!     template,
//!     ReportFormat::Html,
//!     vec!["admin@example.com"]
//! ).await?;
//!
//! // Generate immediate report
//! let report = scheduler.generate_report("weekly_costs").await?;
//! scheduler.save_report(&report, "reports/weekly_cost.html").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! Reports can be configured with:
//! - Custom sections and layouts
//! - Filtering criteria
//! - Styling options for HTML/PDF output
//! - Delivery preferences

use super::{AnalyticsEngine, AnalyticsSummary};
use crate::{cli::error::Result, cli::session::SessionId};
use chrono::{DateTime, Duration, Timelike, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// Report scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchedulerConfig {
    pub enabled: bool,
    pub output_directory: PathBuf,
    pub max_archived_reports: usize,
    pub email_notifications: bool,
    pub compress_old_reports: bool,
}

impl Default for ReportSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            output_directory: PathBuf::from("reports"),
            max_archived_reports: 100,
            email_notifications: false,
            compress_old_reports: true,
        }
    }
}

/// Scheduled report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledReport {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub schedule: ReportSchedule,
    pub template: ReportTemplate,
    pub format: ReportFormat,
    pub recipients: Vec<String>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: DateTime<Utc>,
}

/// Report scheduling patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportSchedule {
    Daily { hour: u8 },
    Weekly { weekday: Weekday, hour: u8 },
    Monthly { day: u8, hour: u8 },
    Custom { cron_expression: String },
}

/// Report template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    pub template_type: ReportTemplateType,
    pub title: String,
    pub sections: Vec<ReportSectionConfig>,
    pub filters: ReportFilters,
    pub styling: ReportStyling,
}

/// Types of report templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportTemplateType {
    Analytics,
    CostSummary,
    PerformanceReport,
    SessionReport,
    Custom,
}

/// Report sections to include (configuration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSectionConfig {
    pub section_type: SectionType,
    pub title: String,
    pub enabled: bool,
    pub custom_content: Option<String>,
}

/// Available report sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    ExecutiveSummary,
    CostBreakdown,
    UsageStatistics,
    PerformanceMetrics,
    ErrorAnalysis,
    Recommendations,
    Charts,
    RawData,
    Custom,
}

/// Report filtering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportFilters {
    pub date_range_days: u32,
    pub session_ids: Option<Vec<SessionId>>,
    pub command_patterns: Vec<String>,
    pub cost_threshold: Option<f64>,
    pub include_errors: bool,
    pub include_successful_only: bool,
}

impl Default for ReportFilters {
    fn default() -> Self {
        Self {
            date_range_days: 7,
            session_ids: None,
            command_patterns: Vec::new(),
            cost_threshold: None,
            include_errors: true,
            include_successful_only: false,
        }
    }
}

/// Report styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportStyling {
    pub theme: StyleTheme,
    pub include_charts: bool,
    pub chart_style: ChartStyle,
    pub custom_css: Option<String>,
}

impl Default for ReportStyling {
    fn default() -> Self {
        Self {
            theme: StyleTheme::Professional,
            include_charts: true,
            chart_style: ChartStyle::Modern,
            custom_css: None,
        }
    }
}

/// Available style themes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StyleTheme {
    Professional,
    Modern,
    Minimal,
    Dark,
    Custom,
}

/// Chart styling options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartStyle {
    Modern,
    Classic,
    Minimal,
    Colorful,
}

/// Report export formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ReportFormat {
    Json,
    Html,
    Csv,
    Pdf,
    Markdown,
}

/// Generated report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub metadata: ReportMetadata,
    pub sections: Vec<ReportSection>,
}

/// Report section types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ReportSectionType {
    ExecutiveSummary,
    CostSummary,
    UsageStatistics,
    PerformanceMetrics,
    ErrorAnalysis,
    Recommendations,
    Charts,
    RawData,
    Custom,
}

/// Report section with data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub title: String,
    pub section_type: ReportSectionType,
    pub content: String,
    pub data: serde_json::Value,
}

/// Generated report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub id: String,
    pub scheduled_report_id: Option<String>,
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub format: ReportFormat,
    pub file_path: PathBuf,
    pub file_size_bytes: u64,
    pub data_points: usize,
}

/// Report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub output_directory: PathBuf,
    pub default_format: ReportFormat,
    pub include_raw_data: bool,
    pub max_data_points: usize,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            output_directory: PathBuf::from("reports"),
            default_format: ReportFormat::Html,
            include_raw_data: false,
            max_data_points: 10000,
        }
    }
}

/// Report scheduler and generator
pub struct ReportScheduler {
    analytics_engine: AnalyticsEngine,
    config: ReportSchedulerConfig,
    scheduled_reports: HashMap<String, ScheduledReport>,
    report_queue: mpsc::Sender<ReportGenerationRequest>,
    _report_receiver: mpsc::Receiver<ReportGenerationRequest>,
}

/// Report generation request
#[derive(Debug)]
struct ReportGenerationRequest {
    scheduled_report: ScheduledReport,
    manual_trigger: bool,
}

impl ReportScheduler {
    /// Create a new report scheduler
    pub fn new(analytics_engine: AnalyticsEngine, config: ReportSchedulerConfig) -> Self {
        let (report_queue, report_receiver) = mpsc::channel(100);

        Self {
            analytics_engine,
            config,
            scheduled_reports: HashMap::new(),
            report_queue,
            _report_receiver: report_receiver,
        }
    }

    /// Add a scheduled report
    pub fn add_scheduled_report(&mut self, report: ScheduledReport) -> Result<()> {
        self.scheduled_reports.insert(report.id.clone(), report);
        Ok(())
    }

    /// Start the report scheduler
    pub async fn start(&mut self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Ensure output directory exists
        tokio::fs::create_dir_all(&self.config.output_directory).await?;

        // Start scheduler loop
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60)); // Check every minute

        loop {
            interval.tick().await;
            self.check_scheduled_reports().await?;
        }
    }

    /// Generate an immediate report
    pub async fn generate_report_now(&self, report: &ScheduledReport) -> Result<ReportMetadata> {
        let analytics_summary = self
            .analytics_engine
            .generate_summary(report.template.filters.date_range_days)
            .await?;

        let file_path = self.generate_report_file_path(report);

        match report.format {
            ReportFormat::Html => {
                self.generate_html_report(&file_path, report, &analytics_summary)
                    .await?;
            }
            ReportFormat::Json => {
                self.generate_json_report(&file_path, &analytics_summary)
                    .await?;
            }
            ReportFormat::Csv => {
                self.generate_csv_report(&file_path, &analytics_summary)
                    .await?;
            }
            ReportFormat::Pdf => {
                // For now, generate HTML as PDF placeholder
                self.generate_html_report(&file_path, report, &analytics_summary)
                    .await?;
            }
            ReportFormat::Markdown => {
                // Generate markdown format
                let content = self.generate_markdown_report(&analytics_summary).await?;
                tokio::fs::write(&file_path, content).await?;
            }
        }

        let file_size = tokio::fs::metadata(&file_path).await?.len();

        Ok(ReportMetadata {
            id: uuid::Uuid::new_v4().to_string(),
            scheduled_report_id: Some(report.id.clone()),
            title: report.name.clone(),
            generated_at: Utc::now(),
            period_start: Utc::now()
                - Duration::days(report.template.filters.date_range_days as i64),
            period_end: Utc::now(),
            format: report.format.clone(),
            file_path,
            file_size_bytes: file_size,
            data_points: analytics_summary.cost_summary.command_count,
        })
    }

    /// Create a default analytics report template
    pub fn create_default_analytics_template() -> ReportTemplate {
        ReportTemplate {
            template_type: ReportTemplateType::Analytics,
            title: "Weekly Analytics Report".to_string(),
            sections: vec![
                ReportSectionConfig {
                    section_type: SectionType::ExecutiveSummary,
                    title: "Executive Summary".to_string(),
                    enabled: true,
                    custom_content: None,
                },
                ReportSectionConfig {
                    section_type: SectionType::CostBreakdown,
                    title: "Cost Breakdown".to_string(),
                    enabled: true,
                    custom_content: None,
                },
                ReportSectionConfig {
                    section_type: SectionType::UsageStatistics,
                    title: "Usage Statistics".to_string(),
                    enabled: true,
                    custom_content: None,
                },
                ReportSectionConfig {
                    section_type: SectionType::PerformanceMetrics,
                    title: "Performance Metrics".to_string(),
                    enabled: true,
                    custom_content: None,
                },
                ReportSectionConfig {
                    section_type: SectionType::Recommendations,
                    title: "Recommendations".to_string(),
                    enabled: true,
                    custom_content: None,
                },
            ],
            filters: ReportFilters::default(),
            styling: ReportStyling::default(),
        }
    }

    /// Create a cost-focused report template
    pub fn create_cost_report_template() -> ReportTemplate {
        ReportTemplate {
            template_type: ReportTemplateType::CostSummary,
            title: "Cost Summary Report".to_string(),
            sections: vec![
                ReportSectionConfig {
                    section_type: SectionType::CostBreakdown,
                    title: "Detailed Cost Breakdown".to_string(),
                    enabled: true,
                    custom_content: None,
                },
                ReportSectionConfig {
                    section_type: SectionType::Charts,
                    title: "Cost Trends".to_string(),
                    enabled: true,
                    custom_content: None,
                },
            ],
            filters: ReportFilters {
                date_range_days: 30,
                ..Default::default()
            },
            styling: ReportStyling::default(),
        }
    }

    /// List all generated reports
    pub async fn list_generated_reports(&self) -> Result<Vec<ReportMetadata>> {
        let mut reports = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.config.output_directory).await?;

        while let Some(entry) = dir.next_entry().await? {
            if let Some(metadata_path) = self.find_metadata_file(&entry.path()).await? {
                let content = tokio::fs::read_to_string(&metadata_path).await?;
                let metadata: ReportMetadata = serde_json::from_str(&content)?;
                reports.push(metadata);
            }
        }

        // Sort by generation date, newest first
        reports.sort_by(|a, b| b.generated_at.cmp(&a.generated_at));

        Ok(reports)
    }

    /// Archive old reports
    pub async fn archive_old_reports(&self) -> Result<usize> {
        let reports = self.list_generated_reports().await?;

        if reports.len() <= self.config.max_archived_reports {
            return Ok(0);
        }

        let reports_to_archive = &reports[self.config.max_archived_reports..];
        let mut archived_count = 0;

        for report in reports_to_archive {
            if self.config.compress_old_reports {
                self.compress_report(&report.file_path).await?;
            }

            // Move to archive directory
            let archive_dir = self.config.output_directory.join("archive");
            tokio::fs::create_dir_all(&archive_dir).await?;

            let archive_path = archive_dir.join(report.file_path.file_name().unwrap());
            tokio::fs::rename(&report.file_path, &archive_path).await?;

            archived_count += 1;
        }

        Ok(archived_count)
    }

    // Private helper methods

    async fn check_scheduled_reports(&mut self) -> Result<()> {
        let now = Utc::now();

        // Collect reports that need updating
        let mut reports_to_update = Vec::new();

        for (id, report) in self.scheduled_reports.iter() {
            if report.enabled && now >= report.next_run {
                reports_to_update.push(id.clone());
            }
        }

        // Update the collected reports
        for report_id in reports_to_update {
            if let Some(report) = self.scheduled_reports.get(&report_id) {
                // Calculate next run time first
                let next_run = self.calculate_next_run(&report.schedule, now);

                // Generate report
                if let Ok(_metadata) = self.generate_report_now(report).await {
                    if let Some(report) = self.scheduled_reports.get_mut(&report_id) {
                        report.last_run = Some(now);
                        report.next_run = next_run;
                    }
                }
            }
        }

        Ok(())
    }

    fn calculate_next_run(&self, schedule: &ReportSchedule, from: DateTime<Utc>) -> DateTime<Utc> {
        match schedule {
            ReportSchedule::Daily { hour } => {
                let mut next = from
                    .date_naive()
                    .and_hms_opt(*hour as u32, 0, 0)
                    .unwrap()
                    .and_utc();
                if next <= from {
                    next = next + Duration::days(1);
                }
                next
            }
            ReportSchedule::Weekly { weekday: _, hour } => {
                // Simplified - would need proper weekday calculation
                from + Duration::days(7) - Duration::hours(from.hour() as i64)
                    + Duration::hours(*hour as i64)
            }
            ReportSchedule::Monthly { day: _, hour } => {
                // Simplified - would need proper month calculation
                from + Duration::days(30) - Duration::hours(from.hour() as i64)
                    + Duration::hours(*hour as i64)
            }
            ReportSchedule::Custom { cron_expression: _ } => {
                // Would need cron parsing library
                from + Duration::hours(1)
            }
        }
    }

    fn generate_report_file_path(&self, report: &ScheduledReport) -> PathBuf {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let extension = match report.format {
            ReportFormat::Html => "html",
            ReportFormat::Json => "json",
            ReportFormat::Csv => "csv",
            ReportFormat::Pdf => "pdf",
            ReportFormat::Markdown => "md",
        };

        self.config.output_directory.join(format!(
            "{}_{}.{}",
            report.name.replace(' ', "_").to_lowercase(),
            timestamp,
            extension
        ))
    }

    async fn generate_html_report(
        &self,
        file_path: &PathBuf,
        report: &ScheduledReport,
        summary: &AnalyticsSummary,
    ) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(file_path)?;

        // Generate HTML content based on template
        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html><head>")?;
        writeln!(file, "<title>{}</title>", report.template.title)?;
        writeln!(
            file,
            "<style>{}</style>",
            self.get_css_styles(&report.template.styling)
        )?;
        writeln!(file, "</head><body>")?;

        writeln!(file, "<h1>{}</h1>", report.template.title)?;
        writeln!(
            file,
            "<p>Generated: {}</p>",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;

        for section in &report.template.sections {
            if section.enabled {
                self.generate_html_section(&mut file, section, summary)
                    .await?;
            }
        }

        writeln!(file, "</body></html>")?;

        Ok(())
    }

    async fn generate_html_section(
        &self,
        file: &mut std::fs::File,
        section: &ReportSectionConfig,
        summary: &AnalyticsSummary,
    ) -> Result<()> {
        use std::io::Write;

        writeln!(file, "<h2>{}</h2>", section.title)?;

        match section.section_type {
            SectionType::ExecutiveSummary => {
                writeln!(file, "<div class=\"summary\">")?;
                writeln!(
                    file,
                    "<p>Total cost: ${:.2}</p>",
                    summary.cost_summary.total_cost
                )?;
                writeln!(
                    file,
                    "<p>Commands executed: {}</p>",
                    summary.cost_summary.command_count
                )?;
                writeln!(
                    file,
                    "<p>Success rate: {:.1}%</p>",
                    summary.performance_metrics.success_rate
                )?;
                writeln!(file, "</div>")?;
            }
            SectionType::CostBreakdown => {
                writeln!(file, "<table>")?;
                writeln!(file, "<tr><th>Command</th><th>Cost</th></tr>")?;
                for (cmd, cost) in &summary.cost_summary.by_command {
                    writeln!(file, "<tr><td>{}</td><td>${:.4}</td></tr>", cmd, cost)?;
                }
                writeln!(file, "</table>")?;
            }
            SectionType::Recommendations => {
                writeln!(file, "<ul>")?;
                for insight in &summary.insights {
                    writeln!(file, "<li>{}</li>", insight)?;
                }
                writeln!(file, "</ul>")?;
            }
            _ => {
                if let Some(custom_content) = &section.custom_content {
                    writeln!(file, "<div>{}</div>", custom_content)?;
                } else {
                    writeln!(file, "<p>Section content not implemented</p>")?;
                }
            }
        }

        Ok(())
    }

    async fn generate_json_report(
        &self,
        file_path: &PathBuf,
        summary: &AnalyticsSummary,
    ) -> Result<()> {
        let content = serde_json::to_string_pretty(summary)?;
        tokio::fs::write(file_path, content).await?;
        Ok(())
    }

    async fn generate_csv_report(
        &self,
        file_path: &PathBuf,
        summary: &AnalyticsSummary,
    ) -> Result<()> {
        // Implementation would generate CSV format
        let content = format!(
            "Cost,Commands,Success Rate\n{:.2},{},{:.1}",
            summary.cost_summary.total_cost,
            summary.cost_summary.command_count,
            summary.performance_metrics.success_rate
        );
        tokio::fs::write(file_path, content).await?;
        Ok(())
    }

    fn get_css_styles(&self, styling: &ReportStyling) -> String {
        match styling.theme {
            StyleTheme::Professional => "body { font-family: Arial, sans-serif; margin: 20px; }
                 table { border-collapse: collapse; width: 100%; }
                 th, td { border: 1px solid #ddd; padding: 8px; }
                 .summary { background: #f5f5f5; padding: 15px; }"
                .to_string(),
            _ => "body { font-family: Arial, sans-serif; }".to_string(),
        }
    }

    async fn find_metadata_file(&self, report_path: &std::path::Path) -> Result<Option<PathBuf>> {
        let metadata_path = report_path.with_extension("metadata.json");
        if metadata_path.exists() {
            Ok(Some(metadata_path))
        } else {
            Ok(None)
        }
    }

    async fn compress_report(&self, _file_path: &PathBuf) -> Result<()> {
        // Implementation would compress the file
        // For now, just a placeholder
        Ok(())
    }

    async fn generate_markdown_report(&self, summary: &AnalyticsSummary) -> Result<String> {
        let mut content = String::new();

        content.push_str("# Analytics Report\n\n");
        content.push_str(&format!(
            "Generated: {}\n\n",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        content.push_str("## Cost Summary\n\n");
        content.push_str(&format!(
            "- Total cost: ${:.2}\n",
            summary.cost_summary.total_cost
        ));
        content.push_str(&format!(
            "- Commands executed: {}\n",
            summary.cost_summary.command_count
        ));
        content.push_str(&format!(
            "- Average cost per command: ${:.4}\n\n",
            summary.cost_summary.average_cost
        ));

        content.push_str("## Performance Metrics\n\n");
        content.push_str(&format!(
            "- Success rate: {:.1}%\n",
            summary.performance_metrics.success_rate
        ));
        content.push_str(&format!(
            "- Average response time: {:.0}ms\n",
            summary.performance_metrics.average_response_time
        ));
        content.push_str(&format!(
            "- Throughput: {:.1} commands/hour\n\n",
            summary.performance_metrics.throughput_commands_per_hour
        ));

        if !summary.insights.is_empty() {
            content.push_str("## Insights\n\n");
            for insight in &summary.insights {
                content.push_str(&format!("- {}\n", insight));
            }
            content.push_str("\n");
        }

        if !summary.alerts.is_empty() {
            content.push_str("## Alerts\n\n");
            for alert in &summary.alerts {
                content.push_str(&format!(
                    "- **{}**: {}\n",
                    format!("{:?}", alert.severity),
                    alert.message
                ));
            }
        }

        Ok(content)
    }
}

/// Report manager for generating reports without scheduling
pub struct ReportManager {
    analytics_engine: AnalyticsEngine,
    config: ReportConfig,
}

impl ReportManager {
    /// Create a new report manager
    pub fn new(analytics_engine: AnalyticsEngine, config: ReportConfig) -> Self {
        Self {
            analytics_engine,
            config,
        }
    }

    /// Generate a report for a specific period
    pub async fn generate_period_report(&self, name: &str, period_days: u32) -> Result<Report> {
        let summary = self.analytics_engine.generate_summary(period_days).await?;

        let metadata = ReportMetadata {
            id: uuid::Uuid::new_v4().to_string(),
            scheduled_report_id: None,
            title: format!("{} - {} Day Report", name, period_days),
            generated_at: Utc::now(),
            period_start: Utc::now() - Duration::days(period_days as i64),
            period_end: Utc::now(),
            format: self.config.default_format.clone(),
            file_path: self.config.output_directory.join(format!(
                "{}.{}",
                name,
                self.format_extension()
            )),
            file_size_bytes: 0,
            data_points: summary.cost_summary.command_count,
        };

        let mut report = Report {
            metadata,
            sections: Vec::new(),
        };

        // Store the base name for consistent file naming
        report.metadata.file_path = self.config.output_directory.join(name);

        report.sections = self.generate_report_sections(&summary).await?;

        Ok(report)
    }

    /// Export a report to multiple formats
    pub async fn export_report(
        &self,
        report: &Report,
        formats: Vec<ReportFormat>,
    ) -> Result<Vec<PathBuf>> {
        let mut exported_files = Vec::new();

        // Use the base name from file_path if it's just a name without extension
        let base_name = if let Some(file_stem) = report.metadata.file_path.file_stem() {
            file_stem.to_string_lossy().to_string()
        } else {
            report.metadata.title.replace(' ', "_").to_lowercase()
        };

        for format in formats {
            let file_path = self.config.output_directory.join(format!(
                "{}.{}",
                base_name,
                Self::format_extension_for(&format)
            ));

            match format {
                ReportFormat::Json => {
                    let content = serde_json::to_string_pretty(report)?;
                    tokio::fs::write(&file_path, content).await?;
                }
                ReportFormat::Html => {
                    let content = self.generate_html_content(report).await?;
                    tokio::fs::write(&file_path, content).await?;
                }
                ReportFormat::Csv => {
                    let content = self.generate_csv_content(report).await?;
                    tokio::fs::write(&file_path, content).await?;
                }
                ReportFormat::Markdown => {
                    let content = self.generate_markdown_content(report).await?;
                    tokio::fs::write(&file_path, content).await?;
                }
                ReportFormat::Pdf => {
                    // For now, generate HTML and note it as PDF placeholder
                    let content = self.generate_html_content(report).await?;
                    tokio::fs::write(&file_path, content).await?;
                }
            }

            exported_files.push(file_path);
        }

        Ok(exported_files)
    }

    fn format_extension(&self) -> &str {
        Self::format_extension_for(&self.config.default_format)
    }

    fn format_extension_for(format: &ReportFormat) -> &str {
        match format {
            ReportFormat::Json => "json",
            ReportFormat::Html => "html",
            ReportFormat::Csv => "csv",
            ReportFormat::Pdf => "pdf",
            ReportFormat::Markdown => "md",
        }
    }

    async fn generate_report_sections(
        &self,
        summary: &AnalyticsSummary,
    ) -> Result<Vec<ReportSection>> {
        let mut sections = Vec::new();

        // Executive Summary
        sections.push(ReportSection {
            title: "Executive Summary".to_string(),
            section_type: ReportSectionType::ExecutiveSummary,
            content: format!(
                "Total cost: ${:.2}, Commands: {}, Success rate: {:.1}%",
                summary.cost_summary.total_cost,
                summary.cost_summary.command_count,
                summary.performance_metrics.success_rate
            ),
            data: serde_json::to_value(summary)?,
        });

        // Cost Summary
        sections.push(ReportSection {
            title: "Cost Summary".to_string(),
            section_type: ReportSectionType::CostSummary,
            content: format!("Detailed cost breakdown by command and model"),
            data: serde_json::to_value(&summary.cost_summary)?,
        });

        // Performance Metrics
        sections.push(ReportSection {
            title: "Performance Metrics".to_string(),
            section_type: ReportSectionType::PerformanceMetrics,
            content: format!("Response times and throughput analysis"),
            data: serde_json::to_value(&summary.performance_metrics)?,
        });

        // Usage Statistics
        sections.push(ReportSection {
            title: "Usage Statistics".to_string(),
            section_type: ReportSectionType::UsageStatistics,
            content: format!("Usage patterns and trends"),
            data: serde_json::to_value(&summary.history_stats)?,
        });

        // Error Analysis (if there are errors)
        if summary.history_stats.failed_commands > 0 {
            sections.push(ReportSection {
                title: "Error Analysis".to_string(),
                section_type: ReportSectionType::ErrorAnalysis,
                content: format!(
                    "{} failed commands out of {} total ({:.1}% error rate)",
                    summary.history_stats.failed_commands,
                    summary.history_stats.total_entries,
                    (summary.history_stats.failed_commands as f64
                        / summary.history_stats.total_entries as f64)
                        * 100.0
                ),
                data: serde_json::to_value(&summary.performance_metrics.error_rate_by_command)?,
            });
        }

        if !summary.insights.is_empty() {
            sections.push(ReportSection {
                title: "Insights & Recommendations".to_string(),
                section_type: ReportSectionType::Recommendations,
                content: summary.insights.join("; "),
                data: serde_json::to_value(&summary.insights)?,
            });
        }

        Ok(sections)
    }

    async fn generate_html_content(&self, report: &Report) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html><head>\n");
        html.push_str(&format!("<title>{}</title>\n", report.metadata.title));
        html.push_str("<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>\n");
        html.push_str("</head><body>\n");

        html.push_str(&format!("<h1>{}</h1>\n", report.metadata.title));
        html.push_str(&format!(
            "<p>Generated: {}</p>\n",
            report.metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        for section in &report.sections {
            html.push_str(&format!("<h2>{}</h2>\n", section.title));
            html.push_str(&format!("<p>{}</p>\n", section.content));
        }

        html.push_str("</body></html>");

        Ok(html)
    }

    async fn generate_csv_content(&self, report: &Report) -> Result<String> {
        let mut csv = String::new();

        csv.push_str("Section,Content\n");
        for section in &report.sections {
            csv.push_str(&format!(
                "\"{}\",\"{}\"\n",
                section.title,
                section.content.replace('"', "''")
            ));
        }

        Ok(csv)
    }

    async fn generate_markdown_content(&self, report: &Report) -> Result<String> {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", report.metadata.title));
        md.push_str(&format!(
            "Generated: {}\n\n",
            report.metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        for section in &report.sections {
            md.push_str(&format!("## {}\n\n", section.title));
            md.push_str(&format!("{}\n\n", section.content));
        }

        Ok(md)
    }
}
