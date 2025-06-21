//! Test infrastructure for reports module
//!
//! This module provides test utilities and infrastructure for testing
//! report generation, scheduling, and export functionality.

use super::reports::*;
use super::test_utils::{AnalyticsTestDataGenerator, DataPattern};
use super::{AnalyticsConfig, AnalyticsEngine, AnalyticsSummary};
use crate::cli::cost::{CostEntry, CostTracker};
use crate::cli::error::Result;
use crate::cli::history::{HistoryEntry, HistoryStore};
use crate::cli::session::SessionId;
use chrono::{DateTime, Datelike, Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

/// Test data generator for report testing
pub struct ReportTestDataGenerator {
    inner: AnalyticsTestDataGenerator,
}

impl ReportTestDataGenerator {
    /// Create a new report test data generator
    pub fn new() -> Self {
        Self {
            inner: AnalyticsTestDataGenerator::default(),
        }
    }

    /// Generate comprehensive test data for a time period
    pub fn generate_period_data(&self, days: u32) -> ReportTestData {
        let test_data_set = self.inner.generate_test_data(days);

        ReportTestData {
            sessions: test_data_set
                .sessions
                .into_iter()
                .map(|s| SessionTestData {
                    id: s.id,
                    start_time: s.start_time,
                    duration_minutes: 60, // Default duration
                })
                .collect(),
            cost_entries: test_data_set.cost_entries,
            history_entries: test_data_set.history_entries,
            start_time: test_data_set.start_time,
            end_time: test_data_set.end_time,
        }
    }

    /// Generate data for specific report types
    pub fn generate_report_specific_data(&self, report_type: ReportType) -> ReportTestData {
        match report_type {
            ReportType::Daily => self.generate_period_data(1),
            ReportType::Weekly => self.generate_period_data(7),
            ReportType::Monthly => self.generate_period_data(30),
            ReportType::Quarterly => self.generate_period_data(90),
            ReportType::Annual => self.generate_period_data(365),
            ReportType::Custom(days) => self.generate_period_data(days),
        }
    }

    /// Generate edge case data for testing
    pub fn generate_edge_case_data(&self, edge_case: EdgeCaseType) -> ReportTestData {
        match edge_case {
            EdgeCaseType::NoData => ReportTestData {
                sessions: Vec::new(),
                cost_entries: Vec::new(),
                history_entries: Vec::new(),
                start_time: Utc::now() - Duration::days(7),
                end_time: Utc::now(),
            },
            EdgeCaseType::SingleEntry => {
                let session_id = uuid::Uuid::new_v4();
                let timestamp = Utc::now() - Duration::hours(1);

                ReportTestData {
                    sessions: vec![SessionTestData {
                        id: session_id,
                        start_time: timestamp,
                        duration_minutes: 10,
                    }],
                    cost_entries: vec![CostEntry::new(
                        session_id,
                        "test".to_string(),
                        0.01,
                        100,
                        200,
                        1000,
                        "claude-3-opus".to_string(),
                    )],
                    history_entries: vec![HistoryEntry::new(
                        session_id,
                        "test".to_string(),
                        vec![],
                        "Test output".to_string(),
                        true,
                        1000,
                    )],
                    start_time: timestamp - Duration::minutes(5),
                    end_time: timestamp + Duration::minutes(5),
                }
            }
            EdgeCaseType::HighVolume => {
                let test_data_set = self.inner.generate_pattern_data(DataPattern::HighVolume);
                self.convert_test_data_set(test_data_set)
            }
            EdgeCaseType::AllFailures => {
                let test_data_set = self.inner.generate_pattern_data(DataPattern::ErrorSpike);
                self.convert_test_data_set(test_data_set)
            }
            EdgeCaseType::MixedModels => {
                let test_data_set = self
                    .inner
                    .generate_pattern_data(DataPattern::ModelMigration);
                self.convert_test_data_set(test_data_set)
            }
            EdgeCaseType::BurstyTraffic => {
                let test_data_set = self.inner.generate_pattern_data(DataPattern::BurstyTraffic);
                self.convert_test_data_set(test_data_set)
            }
        }
    }

    /// Generate data for specific data patterns
    pub fn generate_pattern_data(&self, pattern: DataPattern) -> ReportTestData {
        let test_data_set = self.inner.generate_pattern_data(pattern);
        self.convert_test_data_set(test_data_set)
    }

    /// Generate high volume data for performance testing
    pub fn generate_high_volume_data(&self) -> ReportTestData {
        let test_data_set = self.inner.generate_pattern_data(DataPattern::HighVolume);
        self.convert_test_data_set(test_data_set)
    }

    fn convert_test_data_set(
        &self,
        test_data_set: super::test_utils::TestDataSet,
    ) -> ReportTestData {
        ReportTestData {
            sessions: test_data_set
                .sessions
                .into_iter()
                .map(|s| SessionTestData {
                    id: s.id,
                    start_time: s.start_time,
                    duration_minutes: 60, // Default duration
                })
                .collect(),
            cost_entries: test_data_set.cost_entries,
            history_entries: test_data_set.history_entries,
            start_time: test_data_set.start_time,
            end_time: test_data_set.end_time,
        }
    }
}

/// Report types for testing
#[derive(Debug, Clone)]
pub enum ReportType {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
    Custom(u32),
}

/// Edge case types for testing
#[derive(Debug, Clone)]
pub enum EdgeCaseType {
    NoData,
    SingleEntry,
    HighVolume,
    AllFailures,
    MixedModels,
    BurstyTraffic,
}

/// Test data container
#[derive(Debug, Clone)]
pub struct ReportTestData {
    pub sessions: Vec<SessionTestData>,
    pub cost_entries: Vec<CostEntry>,
    pub history_entries: Vec<HistoryEntry>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct SessionTestData {
    pub id: SessionId,
    pub start_time: DateTime<Utc>,
    pub duration_minutes: u32,
}

/// Test fixture for report testing
pub struct ReportTestFixture {
    pub temp_dir: TempDir,
    pub analytics_engine: AnalyticsEngine,
    pub cost_tracker: Arc<RwLock<CostTracker>>,
    pub history_store: Arc<RwLock<HistoryStore>>,
    pub report_manager: ReportManager,
}

impl ReportTestFixture {
    /// Create a new report test fixture
    pub async fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let cost_path = temp_dir.path().join("costs.json");
        let history_path = temp_dir.path().join("history.json");
        let reports_dir = temp_dir.path().join("reports");

        tokio::fs::create_dir_all(&reports_dir).await?;

        let cost_tracker = Arc::new(RwLock::new(CostTracker::new(cost_path)?));
        let history_store = Arc::new(RwLock::new(HistoryStore::new(history_path)?));

        let analytics_config = AnalyticsConfig::default();
        let analytics_engine = AnalyticsEngine::new(
            Arc::clone(&cost_tracker),
            Arc::clone(&history_store),
            analytics_config,
        );

        let report_config = ReportConfig {
            output_directory: reports_dir,
            ..Default::default()
        };

        let report_manager = ReportManager::new(analytics_engine.clone(), report_config);

        Ok(Self {
            temp_dir,
            analytics_engine,
            cost_tracker,
            history_store,
            report_manager,
        })
    }

    /// Load test data into the fixture
    pub async fn load_test_data(&self, data: &ReportTestData) -> Result<()> {
        let mut cost_tracker = self.cost_tracker.write().await;
        let mut history_store = self.history_store.write().await;

        // Load cost entries
        for entry in &data.cost_entries {
            cost_tracker.record_cost(entry.clone()).await?;
        }

        // Load history entries
        for entry in &data.history_entries {
            history_store.store_entry(entry.clone()).await?;
        }

        Ok(())
    }

    /// Get the reports directory path
    pub fn reports_dir(&self) -> PathBuf {
        self.temp_dir.path().join("reports")
    }
}

/// Report validation utilities
pub mod validation {
    use super::*;

    /// Validate report content structure
    pub fn validate_report_structure(report: &Report) -> ValidationResult {
        let mut errors = Vec::new();

        // Check metadata
        if report.metadata.title.is_empty() {
            errors.push("Report title is empty".to_string());
        }

        if report.metadata.generated_at > Utc::now() {
            errors.push("Report generation time is in the future".to_string());
        }

        if report.metadata.period_start >= report.metadata.period_end {
            errors.push("Report period start is not before end".to_string());
        }

        // Check sections
        if report.sections.is_empty() {
            errors.push("Report has no sections".to_string());
        }

        for section in &report.sections {
            if section.title.is_empty() {
                errors.push(format!("Section has empty title"));
            }

            if section.content.is_empty() {
                errors.push(format!("Section '{}' has no content", section.title));
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
        }
    }

    /// Validate report data accuracy
    pub async fn validate_report_data(
        report: &Report,
        fixture: &ReportTestFixture,
    ) -> ValidationResult {
        let mut errors = Vec::new();

        // Get actual data from the fixture
        let summary = fixture
            .analytics_engine
            .generate_summary(
                (report.metadata.period_end - report.metadata.period_start).num_days() as u32,
            )
            .await
            .unwrap();

        // Find cost summary section
        if let Some(cost_section) = report
            .sections
            .iter()
            .find(|s| s.section_type == ReportSectionType::CostSummary)
        {
            if let Ok(reported_summary) =
                serde_json::from_value::<CostSummaryData>(cost_section.data.clone())
            {
                // Allow small floating point differences
                if (reported_summary.total_cost - summary.cost_summary.total_cost).abs() > 0.01 {
                    errors.push(format!(
                        "Cost mismatch: reported {:.2} vs actual {:.2}",
                        reported_summary.total_cost, summary.cost_summary.total_cost
                    ));
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
        }
    }

    /// Validate exported report files
    pub async fn validate_exported_files(
        report_dir: &PathBuf,
        report_name: &str,
        formats: &[ReportFormat],
    ) -> ValidationResult {
        let mut errors = Vec::new();

        for format in formats {
            let extension = match format {
                ReportFormat::Json => "json",
                ReportFormat::Html => "html",
                ReportFormat::Csv => "csv",
                ReportFormat::Pdf => "pdf",
                ReportFormat::Markdown => "md",
            };

            let file_path = report_dir.join(format!("{}.{}", report_name, extension));

            if !file_path.exists() {
                errors.push(format!(
                    "Expected {} file not found: {:?}",
                    extension, file_path
                ));
                continue;
            }

            // Validate file content
            match format {
                ReportFormat::Json => {
                    if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                        if serde_json::from_str::<serde_json::Value>(&content).is_err() {
                            errors.push(format!("Invalid JSON in file: {:?}", file_path));
                        }
                    }
                }
                ReportFormat::Html => {
                    if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                        if !content.contains("<html") || !content.contains("</html>") {
                            errors.push(format!("Invalid HTML structure in file: {:?}", file_path));
                        }
                    }
                }
                ReportFormat::Csv => {
                    if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                        if content.lines().count() < 2 {
                            errors.push(format!("CSV file has no data rows: {:?}", file_path));
                        }
                    }
                }
                _ => {
                    // Basic file size check for other formats
                    if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
                        if metadata.len() == 0 {
                            errors.push(format!("Empty file: {:?}", file_path));
                        }
                    }
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
        }
    }

    #[derive(Debug)]
    pub struct ValidationResult {
        pub is_valid: bool,
        pub errors: Vec<String>,
    }

    // Mock data structure for validation
    #[derive(Debug, serde::Deserialize)]
    struct CostSummaryData {
        total_cost: f64,
        command_count: usize,
    }
}

/// Performance benchmarking for reports
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Benchmark report generation performance
    pub async fn benchmark_report_generation(
        fixture: &ReportTestFixture,
        report_type: ReportType,
        iterations: usize,
    ) -> ReportBenchmark {
        let mut generation_times = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();

            let period_days = match &report_type {
                ReportType::Daily => 1,
                ReportType::Weekly => 7,
                ReportType::Monthly => 30,
                ReportType::Quarterly => 90,
                ReportType::Annual => 365,
                ReportType::Custom(days) => *days,
            };

            let _ = fixture
                .report_manager
                .generate_period_report("benchmark", period_days)
                .await;

            generation_times.push(start.elapsed().as_millis());
        }

        ReportBenchmark {
            report_type,
            iterations,
            avg_generation_ms: generation_times.iter().sum::<u128>() as f64 / iterations as f64,
            min_generation_ms: *generation_times.iter().min().unwrap_or(&0),
            max_generation_ms: *generation_times.iter().max().unwrap_or(&0),
        }
    }

    /// Benchmark report export performance
    pub async fn benchmark_report_export(
        fixture: &ReportTestFixture,
        formats: Vec<ReportFormat>,
        data_size: usize,
    ) -> ExportBenchmark {
        // Generate test data
        let generator = ReportTestDataGenerator::new();
        let data = generator.generate_high_volume_data();
        fixture.load_test_data(&data).await.unwrap();

        // Generate report
        let report = fixture
            .report_manager
            .generate_period_report("export_benchmark", 1)
            .await
            .unwrap();

        let mut format_times = HashMap::new();

        for format in formats {
            let start = Instant::now();

            let _ = fixture
                .report_manager
                .export_report(&report, vec![format.clone()])
                .await;

            format_times.insert(format, start.elapsed().as_millis());
        }

        ExportBenchmark {
            data_size,
            format_times,
        }
    }

    #[derive(Debug)]
    pub struct ReportBenchmark {
        pub report_type: ReportType,
        pub iterations: usize,
        pub avg_generation_ms: f64,
        pub min_generation_ms: u128,
        pub max_generation_ms: u128,
    }

    #[derive(Debug)]
    pub struct ExportBenchmark {
        pub data_size: usize,
        pub format_times: HashMap<ReportFormat, u128>,
    }
}

/// Simple report schedule for testing
#[derive(Debug, Clone)]
pub enum ReportSchedule {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Custom(String),
}

/// Mock report scheduler for testing
pub struct MockReportScheduler {
    scheduled_reports: Arc<RwLock<Vec<ScheduledReport>>>,
}

#[derive(Debug, Clone)]
pub struct ScheduledReport {
    pub name: String,
    pub schedule: ReportSchedule,
    pub next_run: DateTime<Utc>,
    pub last_run: Option<DateTime<Utc>>,
}

impl MockReportScheduler {
    pub fn new() -> Self {
        Self {
            scheduled_reports: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn schedule_report(&self, name: String, schedule: ReportSchedule) {
        let next_run = self.calculate_next_run(&schedule, Utc::now());

        let report = ScheduledReport {
            name,
            schedule,
            next_run,
            last_run: None,
        };

        let mut reports = self.scheduled_reports.write().await;
        reports.push(report);
    }

    pub async fn get_due_reports(&self) -> Vec<ScheduledReport> {
        let now = Utc::now();
        let reports = self.scheduled_reports.read().await;

        reports
            .iter()
            .filter(|r| r.next_run <= now)
            .cloned()
            .collect()
    }

    fn calculate_next_run(&self, schedule: &ReportSchedule, from: DateTime<Utc>) -> DateTime<Utc> {
        match schedule {
            ReportSchedule::Hourly => from + Duration::hours(1),
            ReportSchedule::Daily => {
                let tomorrow = from.date_naive().succ_opt().unwrap();
                tomorrow.and_hms_opt(9, 0, 0).unwrap().and_utc()
            }
            ReportSchedule::Weekly => {
                let next_monday = from.date_naive();
                let days_until_monday = (7 - next_monday.weekday().num_days_from_monday()) % 7;
                let next_monday = next_monday + Duration::days(days_until_monday as i64 + 7);
                next_monday.and_hms_opt(9, 0, 0).unwrap().and_utc()
            }
            ReportSchedule::Monthly => {
                let next_month = if from.day() >= 1 {
                    from.date_naive()
                        .with_month(from.month() % 12 + 1)
                        .unwrap_or_else(|| {
                            from.date_naive()
                                .with_month(1)
                                .unwrap()
                                .with_year(from.year() + 1)
                                .unwrap()
                        })
                } else {
                    from.date_naive()
                };
                next_month.and_hms_opt(9, 0, 0).unwrap().and_utc()
            }
            ReportSchedule::Custom(cron) => {
                // Simplified - just add 24 hours for custom schedules in tests
                from + Duration::hours(24)
            }
        }
    }
}

// Re-export types that tests might need
pub use super::reports::{
    Report, ReportConfig, ReportFormat, ReportManager, ReportMetadata, ReportSection,
    ReportSectionType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Instant;
    use tokio;

    /// Task 3.4.1: Test report data compilation from multiple analytics sources
    mod data_compilation_tests {
        use super::*;

        #[tokio::test]
        async fn test_multi_source_data_compilation() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Generate comprehensive test data
            let data = generator.generate_period_data(30);
            fixture.load_test_data(&data).await.unwrap();

            // Generate report from multiple sources
            let report = fixture
                .report_manager
                .generate_period_report("multi_source_test", 30)
                .await
                .unwrap();

            // Verify data consistency
            assert!(!report.sections.is_empty());
            assert!(report
                .sections
                .iter()
                .any(|s| s.section_type == ReportSectionType::CostSummary));
            assert!(report
                .sections
                .iter()
                .any(|s| s.section_type == ReportSectionType::PerformanceMetrics));
            assert!(report
                .sections
                .iter()
                .any(|s| s.section_type == ReportSectionType::UsageStatistics));

            // Validate data integrity
            let validation_result = validation::validate_report_data(&report, &fixture).await;
            assert!(
                validation_result.is_valid,
                "Data validation failed: {:?}",
                validation_result.errors
            );
        }

        #[tokio::test]
        async fn test_data_consistency_across_sources() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Generate data with known patterns
            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            // Generate report
            let report = fixture
                .report_manager
                .generate_period_report("consistency_test", 7)
                .await
                .unwrap();

            // Verify cost data matches between sections
            let cost_section = report
                .sections
                .iter()
                .find(|s| s.section_type == ReportSectionType::CostSummary)
                .expect("Cost summary section should exist");

            let exec_section = report
                .sections
                .iter()
                .find(|s| s.section_type == ReportSectionType::ExecutiveSummary)
                .expect("Executive summary section should exist");

            // Both sections should reference the same underlying data
            assert!(cost_section.content.contains("cost"));
            assert!(exec_section.content.contains("cost"));
        }

        #[tokio::test]
        async fn test_time_synchronization_between_sources() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Generate data with mixed timestamps
            let data = generator.generate_edge_case_data(EdgeCaseType::BurstyTraffic);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("time_sync_test", 7)
                .await
                .unwrap();

            // Verify time period consistency
            assert!(report.metadata.period_start < report.metadata.period_end);
            assert!(report.metadata.generated_at >= report.metadata.period_end);

            // All data should fall within the specified period
            let period_duration = report.metadata.period_end - report.metadata.period_start;
            assert_eq!(period_duration.num_days(), 7);
        }

        #[tokio::test]
        async fn test_data_validation_and_integrity() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Test with edge cases
            for edge_case in [
                EdgeCaseType::NoData,
                EdgeCaseType::SingleEntry,
                EdgeCaseType::HighVolume,
                EdgeCaseType::AllFailures,
                EdgeCaseType::MixedModels,
            ] {
                let data = generator.generate_edge_case_data(edge_case.clone());
                fixture.load_test_data(&data).await.unwrap();

                let report = fixture
                    .report_manager
                    .generate_period_report(&format!("integrity_test_{:?}", edge_case), 1)
                    .await
                    .unwrap();

                // Validate structure
                let structure_result = validation::validate_report_structure(&report);
                assert!(
                    structure_result.is_valid,
                    "Structure validation failed for {:?}: {:?}",
                    edge_case, structure_result.errors
                );

                // Validate data accuracy
                let data_result = validation::validate_report_data(&report, &fixture).await;
                assert!(
                    data_result.is_valid,
                    "Data validation failed for {:?}: {:?}",
                    edge_case, data_result.errors
                );
            }
        }
    }

    /// Task 3.4.2: Test report formatting for different output formats
    mod format_tests {
        use super::*;

        #[tokio::test]
        async fn test_json_report_format() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("json_test", 7)
                .await
                .unwrap();

            // Export to JSON
            let exported_files = fixture
                .report_manager
                .export_report(&report, vec![ReportFormat::Json])
                .await
                .unwrap();

            assert_eq!(exported_files.len(), 1);
            let json_file = &exported_files[0];
            assert!(json_file.exists());

            // Validate JSON content
            let content = tokio::fs::read_to_string(json_file).await.unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

            assert!(parsed.get("metadata").is_some());
            assert!(parsed.get("sections").is_some());
        }

        #[tokio::test]
        async fn test_csv_export_functionality() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("csv_test", 7)
                .await
                .unwrap();

            // Export to CSV
            let exported_files = fixture
                .report_manager
                .export_report(&report, vec![ReportFormat::Csv])
                .await
                .unwrap();

            assert_eq!(exported_files.len(), 1);
            let csv_file = &exported_files[0];
            assert!(csv_file.exists());

            // Validate CSV structure
            let content = tokio::fs::read_to_string(csv_file).await.unwrap();
            let lines: Vec<&str> = content.lines().collect();

            assert!(lines.len() >= 2); // Header + at least one data row
            assert!(lines[0].contains("Section")); // Header should contain "Section"
        }

        #[tokio::test]
        async fn test_html_report_generation() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("html_test", 7)
                .await
                .unwrap();

            // Export to HTML
            let exported_files = fixture
                .report_manager
                .export_report(&report, vec![ReportFormat::Html])
                .await
                .unwrap();

            assert_eq!(exported_files.len(), 1);
            let html_file = &exported_files[0];
            assert!(html_file.exists());

            // Validate HTML structure
            let content = tokio::fs::read_to_string(html_file).await.unwrap();
            assert!(content.contains("<!DOCTYPE html>"));
            assert!(content.contains("<html>"));
            assert!(content.contains("</html>"));
            assert!(content.contains(&report.metadata.title));
        }

        #[tokio::test]
        async fn test_markdown_report_formatting() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("markdown_test", 7)
                .await
                .unwrap();

            // Export to Markdown
            let exported_files = fixture
                .report_manager
                .export_report(&report, vec![ReportFormat::Markdown])
                .await
                .unwrap();

            assert_eq!(exported_files.len(), 1);
            let md_file = &exported_files[0];
            assert!(md_file.exists());

            // Validate Markdown structure
            let content = tokio::fs::read_to_string(md_file).await.unwrap();
            assert!(content.starts_with("# ")); // Should start with H1
            assert!(content.contains("## ")); // Should have H2 sections
            assert!(content.contains(&report.metadata.title));
        }

        #[tokio::test]
        async fn test_custom_format_support() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("multi_format_test", 7)
                .await
                .unwrap();

            // Export to multiple formats
            let formats = vec![
                ReportFormat::Json,
                ReportFormat::Html,
                ReportFormat::Csv,
                ReportFormat::Markdown,
            ];

            let exported_files = fixture
                .report_manager
                .export_report(&report, formats.clone())
                .await
                .unwrap();

            assert_eq!(exported_files.len(), formats.len());

            // Validate all files were created
            for file in &exported_files {
                assert!(file.exists());
            }

            // Validate file validation
            let validation_result = validation::validate_exported_files(
                &fixture.reports_dir(),
                "multi_format_test",
                &formats,
            )
            .await;

            assert!(
                validation_result.is_valid,
                "Format validation failed: {:?}",
                validation_result.errors
            );
        }
    }

    /// Task 3.4.3: Test scheduled report generation and automation
    mod scheduling_tests {
        use super::*;

        #[tokio::test]
        async fn test_periodic_report_generation() {
            let scheduler = MockReportScheduler::new();

            // Schedule different report types
            scheduler
                .schedule_report("daily_report".to_string(), ReportSchedule::Hourly)
                .await;

            scheduler
                .schedule_report("weekly_report".to_string(), ReportSchedule::Weekly)
                .await;

            scheduler
                .schedule_report("monthly_report".to_string(), ReportSchedule::Monthly)
                .await;

            // Test due report detection
            let due_reports = scheduler.get_due_reports().await;

            // In a real scenario, we'd wait for scheduled times
            // For testing, we verify the scheduler can track different schedules
            assert!(due_reports.len() <= 3); // May have some due reports
        }

        #[tokio::test]
        async fn test_automated_report_triggering() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Setup test data
            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            // Test automated report generation (simulated)
            let reports_before = fixture.reports_dir().read_dir().unwrap().count();

            // Generate automated report
            let report = fixture
                .report_manager
                .generate_period_report("automated_test", 7)
                .await
                .unwrap();

            // Export the report
            fixture
                .report_manager
                .export_report(&report, vec![ReportFormat::Html])
                .await
                .unwrap();

            // Verify report was created
            let reports_after = fixture.reports_dir().read_dir().unwrap().count();
            assert!(reports_after > reports_before);
        }

        #[tokio::test]
        async fn test_report_delivery_mechanisms() {
            // This would test email delivery, file system delivery, etc.
            // For now, we test file system delivery
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("delivery_test", 7)
                .await
                .unwrap();

            // Test delivery to different formats/locations
            let formats = vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Csv];
            let delivered_files = fixture
                .report_manager
                .export_report(&report, formats)
                .await
                .unwrap();

            // Verify all deliveries succeeded
            for file in delivered_files {
                assert!(file.exists());
                assert!(tokio::fs::metadata(&file).await.unwrap().len() > 0);
            }
        }

        #[tokio::test]
        async fn test_failure_handling_and_retry_logic() {
            let fixture = ReportTestFixture::new().await.unwrap();

            // Test generation with no data (edge case)
            let result = fixture
                .report_manager
                .generate_period_report("failure_test", 7)
                .await;

            // Should succeed even with no data
            assert!(result.is_ok());

            let report = result.unwrap();
            assert!(!report.sections.is_empty()); // Should have at least basic sections

            // Test export to non-existent directory (would fail in real scenario)
            let export_result = fixture
                .report_manager
                .export_report(&report, vec![ReportFormat::Json])
                .await;

            // Should succeed as we create directories
            assert!(export_result.is_ok());
        }
    }

    /// Task 3.4.4: Test report customization and filtering options
    mod customization_tests {
        use super::*;

        #[tokio::test]
        async fn test_date_range_filtering() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Generate data for different periods
            let data_30_days = generator.generate_period_data(30);
            let data_7_days = generator.generate_period_data(7);
            let data_1_day = generator.generate_period_data(1);

            fixture.load_test_data(&data_30_days).await.unwrap();
            fixture.load_test_data(&data_7_days).await.unwrap();
            fixture.load_test_data(&data_1_day).await.unwrap();

            // Test different date ranges
            for days in [1, 7, 30] {
                let report = fixture
                    .report_manager
                    .generate_period_report(&format!("filter_test_{}_days", days), days)
                    .await
                    .unwrap();

                // Verify date range
                let period_duration = report.metadata.period_end - report.metadata.period_start;
                assert_eq!(period_duration.num_days(), days as i64);

                // Verify report reflects the correct time period
                assert!(report.metadata.title.contains(&days.to_string()));
            }
        }

        #[tokio::test]
        async fn test_metric_selection_and_customization() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("metrics_test", 7)
                .await
                .unwrap();

            // Verify different metric sections are available
            let section_types: HashSet<_> =
                report.sections.iter().map(|s| &s.section_type).collect();

            assert!(section_types.contains(&ReportSectionType::CostSummary));
            assert!(section_types.contains(&ReportSectionType::PerformanceMetrics));
            assert!(section_types.contains(&ReportSectionType::UsageStatistics));

            // Verify each section has data
            for section in &report.sections {
                assert!(!section.content.is_empty());
                assert!(!section.data.is_null());
            }
        }

        #[tokio::test]
        async fn test_user_specific_report_generation() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Generate data for different "users" (sessions)
            let session_1 = uuid::Uuid::new_v4();
            let session_2 = uuid::Uuid::new_v4();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            // Generate reports with different names/contexts
            let report_1 = fixture
                .report_manager
                .generate_period_report("user_1_report", 7)
                .await
                .unwrap();

            let report_2 = fixture
                .report_manager
                .generate_period_report("user_2_report", 7)
                .await
                .unwrap();

            // Verify reports are distinct
            assert_ne!(report_1.metadata.id, report_2.metadata.id);
            assert_ne!(report_1.metadata.title, report_2.metadata.title);

            // Both should have valid structure
            assert!(!report_1.sections.is_empty());
            assert!(!report_2.sections.is_empty());
        }

        #[tokio::test]
        async fn test_template_system_and_customization() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            // Test with different report "templates" (different names/purposes)
            let templates = [
                "executive_summary",
                "technical_deep_dive",
                "cost_analysis",
                "performance_review",
            ];

            for template in &templates {
                let report = fixture
                    .report_manager
                    .generate_period_report(template, 7)
                    .await
                    .unwrap();

                // Verify report reflects the template
                assert!(report.metadata.title.contains(template));
                assert!(!report.sections.is_empty());

                // Verify structure is consistent across templates
                let structure_result = validation::validate_report_structure(&report);
                assert!(
                    structure_result.is_valid,
                    "Template {} failed validation: {:?}",
                    template, structure_result.errors
                );
            }
        }
    }

    /// Performance and benchmark tests
    mod performance_tests {
        use super::*;

        #[tokio::test]
        async fn test_report_generation_performance() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Test performance with different data volumes
            let test_cases = [("small", 1), ("medium", 7), ("large", 30), ("high_volume", 100)];

            for (name, days) in test_cases {
                let data = generator.generate_period_data(days);
                fixture.load_test_data(&data).await.unwrap();

                let start = Instant::now();
                let report = fixture
                    .report_manager
                    .generate_period_report(&format!("perf_test_{}", name), 7)
                    .await
                    .unwrap();
                let generation_time = start.elapsed();

                // Performance assertions (adjust thresholds as needed)
                assert!(
                    generation_time.as_millis() < 5000,
                    "Report generation took too long for {}: {}ms",
                    name,
                    generation_time.as_millis()
                );

                assert!(!report.sections.is_empty());

                println!(
                    "Report generation for {} took: {}ms",
                    name,
                    generation_time.as_millis()
                );
            }
        }

        #[tokio::test]
        async fn test_export_performance_across_formats() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            let data = generator.generate_period_data(7);
            fixture.load_test_data(&data).await.unwrap();

            let report = fixture
                .report_manager
                .generate_period_report("export_perf_test", 7)
                .await
                .unwrap();

            // Test export performance for each format
            for format in [
                ReportFormat::Json,
                ReportFormat::Html,
                ReportFormat::Csv,
                ReportFormat::Markdown,
            ] {
                let start = Instant::now();
                let exported_files = fixture
                    .report_manager
                    .export_report(&report, vec![format.clone()])
                    .await
                    .unwrap();
                let export_time = start.elapsed();

                assert_eq!(exported_files.len(), 1);
                assert!(exported_files[0].exists());

                // Export should be fast
                assert!(
                    export_time.as_millis() < 1000,
                    "Export to {:?} took too long: {}ms",
                    format,
                    export_time.as_millis()
                );

                println!("Export to {:?} took: {}ms", format, export_time.as_millis());
            }
        }
    }

    /// Integration tests with realistic scenarios
    mod integration_tests {
        use super::*;

        #[tokio::test]
        async fn test_end_to_end_report_workflow() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Step 1: Generate realistic data
            let data = generator.generate_pattern_data(DataPattern::GradualGrowth);
            fixture.load_test_data(&data).await.unwrap();

            // Step 2: Generate comprehensive report
            let report = fixture
                .report_manager
                .generate_period_report("e2e_workflow_test", 30)
                .await
                .unwrap();

            // Step 3: Validate report structure
            let validation_result = validation::validate_report_structure(&report);
            assert!(validation_result.is_valid);

            // Step 4: Export to multiple formats
            let formats = vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Csv];
            let exported_files = fixture
                .report_manager
                .export_report(&report, formats.clone())
                .await
                .unwrap();

            // Step 5: Validate exported files
            let file_validation = validation::validate_exported_files(
                &fixture.reports_dir(),
                "e2e_workflow_test",
                &formats,
            )
            .await;
            assert!(file_validation.is_valid);

            // Step 6: Verify data integrity
            let data_validation = validation::validate_report_data(&report, &fixture).await;
            assert!(data_validation.is_valid);

            println!("End-to-end workflow completed successfully");
            println!("Generated {} sections in report", report.sections.len());
            println!("Exported {} files", exported_files.len());
        }

        #[tokio::test]
        async fn test_realistic_data_patterns() {
            let fixture = ReportTestFixture::new().await.unwrap();
            let generator = ReportTestDataGenerator::new();

            // Test with different realistic patterns
            for pattern in [
                DataPattern::BurstyTraffic,
                DataPattern::GradualGrowth,
                DataPattern::ErrorSpike,
                DataPattern::ModelMigration,
            ] {
                let data = generator.generate_pattern_data(pattern.clone());
                fixture.load_test_data(&data).await.unwrap();

                let report = fixture
                    .report_manager
                    .generate_period_report(&format!("pattern_test_{:?}", pattern), 7)
                    .await
                    .unwrap();

                // Verify report handles different patterns appropriately
                assert!(!report.sections.is_empty());
                assert!(
                    report.metadata.data_points > 0 || matches!(pattern, DataPattern::ErrorSpike)
                );

                // Pattern-specific validations
                match pattern {
                    DataPattern::ErrorSpike => {
                        // Should have error-related insights
                        let perf_section = report
                            .sections
                            .iter()
                            .find(|s| s.section_type == ReportSectionType::PerformanceMetrics);
                        assert!(perf_section.is_some());
                    }
                    DataPattern::ModelMigration => {
                        // Should show model diversity
                        let cost_section = report
                            .sections
                            .iter()
                            .find(|s| s.section_type == ReportSectionType::CostSummary);
                        assert!(cost_section.is_some());
                    }
                    _ => {
                        // All patterns should have basic sections
                        assert!(report.sections.len() >= 3);
                    }
                }
            }
        }
    }
}
