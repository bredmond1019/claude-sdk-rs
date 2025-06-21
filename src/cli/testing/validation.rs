//! Validation utilities and data quality checks
//!
//! This module provides validation for:
//! - Configuration files and settings
//! - Data integrity checks
//! - Input sanitization and validation
//! - Schema validation for stored data

use crate::{
    cost::{CostEntry, CostSummary},
    history::{HistoryEntry, HistoryStats},
    session::Session,
};
use chrono::Utc;
use serde_json;
use std::collections::HashMap;
use uuid::Uuid;

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub error_type: ValidationErrorType,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Types of validation errors
#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    Required,
    Format,
    Range,
    Logic,
    Consistency,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, field: String, message: String, error_type: ValidationErrorType) {
        self.valid = false;
        self.errors.push(ValidationError {
            field,
            message,
            error_type,
        });
    }

    /// Add a warning
    pub fn add_warning(&mut self, field: String, message: String, suggestion: Option<String>) {
        self.warnings.push(ValidationWarning {
            field,
            message,
            suggestion,
        });
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.valid && self.errors.is_empty()
    }

    /// Get summary of validation results
    pub fn summary(&self) -> String {
        if self.is_valid() {
            if self.warnings.is_empty() {
                "Validation passed".to_string()
            } else {
                format!("Validation passed with {} warnings", self.warnings.len())
            }
        } else {
            format!("Validation failed with {} errors", self.errors.len())
        }
    }
}

/// Session validator
pub struct SessionValidator;

impl SessionValidator {
    /// Validate a session object
    pub fn validate_session(session: &Session) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate session ID
        if session.id == Uuid::nil() {
            result.add_error(
                "id".to_string(),
                "Session ID cannot be nil".to_string(),
                ValidationErrorType::Required,
            );
        }

        // Validate session name
        if session.name.is_empty() {
            result.add_error(
                "name".to_string(),
                "Session name cannot be empty".to_string(),
                ValidationErrorType::Required,
            );
        } else if session.name.len() > 100 {
            result.add_error(
                "name".to_string(),
                "Session name cannot exceed 100 characters".to_string(),
                ValidationErrorType::Range,
            );
        }

        // Validate description length
        if let Some(description) = &session.description {
            if description.len() > 500 {
                result.add_error(
                    "description".to_string(),
                    "Session description cannot exceed 500 characters".to_string(),
                    ValidationErrorType::Range,
                );
            }
        }

        // Validate timestamps
        if session.created_at > Utc::now() {
            result.add_error(
                "created_at".to_string(),
                "Session creation time cannot be in the future".to_string(),
                ValidationErrorType::Logic,
            );
        }

        if session.last_active > Utc::now() {
            result.add_error(
                "last_active".to_string(),
                "Session last active time cannot be in the future".to_string(),
                ValidationErrorType::Logic,
            );
        }

        if session.last_active < session.created_at {
            result.add_error(
                "last_active".to_string(),
                "Session last active time cannot be before creation time".to_string(),
                ValidationErrorType::Logic,
            );
        }

        // Validate metadata
        if session.metadata.total_cost < 0.0 {
            result.add_error(
                "metadata.total_cost".to_string(),
                "Total cost cannot be negative".to_string(),
                ValidationErrorType::Range,
            );
        }

        if session.metadata.total_cost > 1000.0 {
            result.add_warning(
                "metadata.total_cost".to_string(),
                "Total cost is unusually high".to_string(),
                Some("Review session usage to ensure accuracy".to_string()),
            );
        }

        // Validate tags
        for tag in &session.metadata.tags {
            if tag.is_empty() {
                result.add_error(
                    "metadata.tags".to_string(),
                    "Tags cannot be empty".to_string(),
                    ValidationErrorType::Format,
                );
            } else if tag.len() > 50 {
                result.add_error(
                    "metadata.tags".to_string(),
                    "Tag cannot exceed 50 characters".to_string(),
                    ValidationErrorType::Range,
                );
            }
        }

        result
    }

    /// Validate session name format
    pub fn validate_session_name(name: &str) -> ValidationResult {
        let mut result = ValidationResult::new();

        if name.is_empty() {
            result.add_error(
                "name".to_string(),
                "Session name cannot be empty".to_string(),
                ValidationErrorType::Required,
            );
            return result;
        }

        // Check for valid characters
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || "-_. ".contains(c))
        {
            result.add_error(
                "name".to_string(),
                "Session name can only contain alphanumeric characters, hyphens, underscores, dots, and spaces".to_string(),
                ValidationErrorType::Format,
            );
        }

        // Check length
        if name.len() > 100 {
            result.add_error(
                "name".to_string(),
                "Session name cannot exceed 100 characters".to_string(),
                ValidationErrorType::Range,
            );
        }

        // Check for reserved names
        let reserved_names = ["system", "admin", "root", "config", "temp", "tmp"];
        if reserved_names.contains(&name.to_lowercase().as_str()) {
            result.add_error(
                "name".to_string(),
                "Session name is reserved".to_string(),
                ValidationErrorType::Logic,
            );
        }

        result
    }
}

/// Cost entry validator
pub struct CostValidator;

impl CostValidator {
    /// Validate a cost entry
    pub fn validate_cost_entry(entry: &CostEntry) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate session ID
        if entry.session_id == Uuid::nil() {
            result.add_error(
                "session_id".to_string(),
                "Session ID cannot be nil".to_string(),
                ValidationErrorType::Required,
            );
        }

        // Validate command name
        if entry.command_name.is_empty() {
            result.add_error(
                "command_name".to_string(),
                "Command name cannot be empty".to_string(),
                ValidationErrorType::Required,
            );
        }

        // Validate cost
        if entry.cost_usd < 0.0 {
            result.add_error(
                "cost_usd".to_string(),
                "Cost cannot be negative".to_string(),
                ValidationErrorType::Range,
            );
        }

        if entry.cost_usd > 100.0 {
            result.add_warning(
                "cost_usd".to_string(),
                "Cost is unusually high for a single command".to_string(),
                Some("Verify cost calculation".to_string()),
            );
        }

        // Validate tokens
        if entry.input_tokens == 0 && entry.output_tokens == 0 {
            result.add_warning(
                "tokens".to_string(),
                "No tokens recorded".to_string(),
                Some("Ensure token counts are being tracked".to_string()),
            );
        }

        if entry.input_tokens > 200000 {
            result.add_warning(
                "input_tokens".to_string(),
                "Input token count is very high".to_string(),
                Some("Consider breaking down large inputs".to_string()),
            );
        }

        if entry.output_tokens > 100000 {
            result.add_warning(
                "output_tokens".to_string(),
                "Output token count is very high".to_string(),
                Some("Consider limiting output length".to_string()),
            );
        }

        // Validate duration
        if entry.duration_ms == 0 {
            result.add_warning(
                "duration_ms".to_string(),
                "No duration recorded".to_string(),
                Some("Ensure execution time is being tracked".to_string()),
            );
        }

        if entry.duration_ms > 300000 {
            // 5 minutes
            result.add_warning(
                "duration_ms".to_string(),
                "Command took unusually long to execute".to_string(),
                Some("Check for performance issues".to_string()),
            );
        }

        // Validate timestamp
        if entry.timestamp > Utc::now() {
            result.add_error(
                "timestamp".to_string(),
                "Timestamp cannot be in the future".to_string(),
                ValidationErrorType::Logic,
            );
        }

        // Validate model
        if entry.model.is_empty() {
            result.add_warning(
                "model".to_string(),
                "Model name not specified".to_string(),
                Some("Record model used for better tracking".to_string()),
            );
        }

        result
    }

    /// Validate cost summary
    pub fn validate_cost_summary(summary: &CostSummary) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate totals
        if summary.total_cost < 0.0 {
            result.add_error(
                "total_cost".to_string(),
                "Total cost cannot be negative".to_string(),
                ValidationErrorType::Range,
            );
        }

        if summary.command_count == 0 && summary.total_cost > 0.0 {
            result.add_error(
                "consistency".to_string(),
                "Total cost is positive but command count is zero".to_string(),
                ValidationErrorType::Consistency,
            );
        }

        // Validate average cost calculation
        if summary.command_count > 0 {
            let expected_average = summary.total_cost / summary.command_count as f64;
            let tolerance = 0.0001;

            if (summary.average_cost - expected_average).abs() > tolerance {
                result.add_error(
                    "average_cost".to_string(),
                    "Average cost calculation is inconsistent".to_string(),
                    ValidationErrorType::Consistency,
                );
            }
        }

        // Validate date range
        if summary.date_range.0 > summary.date_range.1 {
            result.add_error(
                "date_range".to_string(),
                "Start date cannot be after end date".to_string(),
                ValidationErrorType::Logic,
            );
        }

        // Validate breakdown totals
        let command_total: f64 = summary.by_command.values().sum();
        let model_total: f64 = summary.by_model.values().sum();

        let tolerance = 0.01; // Allow small floating point differences

        if (command_total - summary.total_cost).abs() > tolerance {
            result.add_warning(
                "by_command".to_string(),
                "Command breakdown total doesn't match total cost".to_string(),
                Some("Check cost aggregation logic".to_string()),
            );
        }

        if (model_total - summary.total_cost).abs() > tolerance {
            result.add_warning(
                "by_model".to_string(),
                "Model breakdown total doesn't match total cost".to_string(),
                Some("Check cost aggregation logic".to_string()),
            );
        }

        result
    }
}

/// History entry validator
pub struct HistoryValidator;

impl HistoryValidator {
    /// Validate a history entry
    pub fn validate_history_entry(entry: &HistoryEntry) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate ID
        if entry.id.is_empty() {
            result.add_error(
                "id".to_string(),
                "History entry ID cannot be empty".to_string(),
                ValidationErrorType::Required,
            );
        }

        // Validate session ID
        if entry.session_id == Uuid::nil() {
            result.add_error(
                "session_id".to_string(),
                "Session ID cannot be nil".to_string(),
                ValidationErrorType::Required,
            );
        }

        // Validate command name
        if entry.command_name.is_empty() {
            result.add_error(
                "command_name".to_string(),
                "Command name cannot be empty".to_string(),
                ValidationErrorType::Required,
            );
        }

        // Validate output
        if entry.output.is_empty() && entry.error.is_none() {
            result.add_warning(
                "output".to_string(),
                "No output or error recorded".to_string(),
                Some("Ensure command results are captured".to_string()),
            );
        }

        // Validate success/error consistency
        if !entry.success && entry.error.is_none() {
            result.add_warning(
                "error".to_string(),
                "Command marked as failed but no error message provided".to_string(),
                Some("Record error details for failed commands".to_string()),
            );
        }

        if entry.success && entry.error.is_some() {
            result.add_warning(
                "error".to_string(),
                "Command marked as successful but error message is present".to_string(),
                Some("Check success/error consistency".to_string()),
            );
        }

        // Validate cost consistency
        if let Some(cost) = entry.cost_usd {
            if cost < 0.0 {
                result.add_error(
                    "cost_usd".to_string(),
                    "Cost cannot be negative".to_string(),
                    ValidationErrorType::Range,
                );
            }

            if !entry.success && cost > 0.0 {
                result.add_warning(
                    "cost_usd".to_string(),
                    "Failed command has positive cost".to_string(),
                    Some("Verify cost calculation for failed commands".to_string()),
                );
            }
        }

        // Validate duration
        if entry.duration_ms == 0 {
            result.add_warning(
                "duration_ms".to_string(),
                "No duration recorded".to_string(),
                Some("Ensure execution time is tracked".to_string()),
            );
        }

        // Validate timestamp
        if entry.timestamp > Utc::now() {
            result.add_error(
                "timestamp".to_string(),
                "Timestamp cannot be in the future".to_string(),
                ValidationErrorType::Logic,
            );
        }

        // Validate tags
        for tag in &entry.tags {
            if tag.is_empty() {
                result.add_error(
                    "tags".to_string(),
                    "Tags cannot be empty".to_string(),
                    ValidationErrorType::Format,
                );
            }
        }

        result
    }

    /// Validate history statistics
    pub fn validate_history_stats(stats: &HistoryStats) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate totals
        if stats.successful_commands + stats.failed_commands != stats.total_entries {
            result.add_error(
                "totals".to_string(),
                "Sum of successful and failed commands doesn't match total entries".to_string(),
                ValidationErrorType::Consistency,
            );
        }

        // Validate success rate calculation
        if stats.total_entries > 0 {
            let expected_rate =
                (stats.successful_commands as f64 / stats.total_entries as f64) * 100.0;
            let tolerance = 0.1;

            if (stats.success_rate - expected_rate).abs() > tolerance {
                result.add_error(
                    "success_rate".to_string(),
                    "Success rate calculation is inconsistent".to_string(),
                    ValidationErrorType::Consistency,
                );
            }
        }

        // Validate costs
        if stats.total_cost < 0.0 {
            result.add_error(
                "total_cost".to_string(),
                "Total cost cannot be negative".to_string(),
                ValidationErrorType::Range,
            );
        }

        if stats.average_cost < 0.0 {
            result.add_error(
                "average_cost".to_string(),
                "Average cost cannot be negative".to_string(),
                ValidationErrorType::Range,
            );
        }

        // Validate duration
        if stats.total_duration_ms == 0 && stats.total_entries > 0 {
            result.add_warning(
                "total_duration_ms".to_string(),
                "No duration recorded for any entries".to_string(),
                Some("Ensure execution times are tracked".to_string()),
            );
        }

        if stats.average_duration_ms < 0.0 {
            result.add_error(
                "average_duration_ms".to_string(),
                "Average duration cannot be negative".to_string(),
                ValidationErrorType::Range,
            );
        }

        // Validate date range
        if stats.date_range.0 > stats.date_range.1 {
            result.add_error(
                "date_range".to_string(),
                "Start date cannot be after end date".to_string(),
                ValidationErrorType::Logic,
            );
        }

        result
    }
}

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate JSON configuration
    pub fn validate_json_config(json_str: &str) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Try to parse JSON
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(_) => {
                // JSON is valid
            }
            Err(e) => {
                result.add_error(
                    "json".to_string(),
                    format!("Invalid JSON: {}", e),
                    ValidationErrorType::Format,
                );
            }
        }

        result
    }

    /// Validate configuration file structure
    pub fn validate_config_structure(config: &serde_json::Value) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check for required fields
        let required_fields = ["version", "settings"];
        for field in required_fields {
            if !config.get(field).is_some() {
                result.add_error(
                    field.to_string(),
                    format!("Required field '{}' is missing", field),
                    ValidationErrorType::Required,
                );
            }
        }

        // Validate version format
        if let Some(version) = config.get("version").and_then(|v| v.as_str()) {
            if !version.chars().all(|c| c.is_numeric() || c == '.') {
                result.add_error(
                    "version".to_string(),
                    "Version must contain only numbers and dots".to_string(),
                    ValidationErrorType::Format,
                );
            }
        }

        result
    }
}

/// Data integrity checker
pub struct DataIntegrityChecker;

impl DataIntegrityChecker {
    /// Check data consistency between cost and history records
    pub fn check_cost_history_consistency(
        cost_entries: &[CostEntry],
        history_entries: &[HistoryEntry],
    ) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check that every cost entry has a corresponding history entry
        for cost_entry in cost_entries {
            let matching_history = history_entries.iter().find(|h| {
                h.session_id == cost_entry.session_id
                    && h.command_name == cost_entry.command_name
                    && h.timestamp
                        .timestamp_millis()
                        .abs_diff(cost_entry.timestamp.timestamp_millis())
                        < 1000 // Within 1 second
            });

            if matching_history.is_none() {
                result.add_warning(
                    "consistency".to_string(),
                    format!(
                        "Cost entry for command '{}' has no matching history entry",
                        cost_entry.command_name
                    ),
                    Some("Ensure cost and history are recorded together".to_string()),
                );
            }
        }

        // Check for history entries with cost data that don't have cost entries
        for history_entry in history_entries {
            if history_entry.cost_usd.is_some() {
                let matching_cost = cost_entries.iter().find(|c| {
                    c.session_id == history_entry.session_id
                        && c.command_name == history_entry.command_name
                        && c.timestamp
                            .timestamp_millis()
                            .abs_diff(history_entry.timestamp.timestamp_millis())
                            < 1000
                });

                if matching_cost.is_none() {
                    result.add_warning(
                        "consistency".to_string(),
                        format!("History entry for command '{}' has cost data but no matching cost entry", history_entry.command_name),
                        Some("Ensure cost tracking is consistent".to_string()),
                    );
                }
            }
        }

        result
    }

    /// Check for duplicate entries
    pub fn check_for_duplicates<T, F>(items: &[T], key_extractor: F) -> ValidationResult
    where
        F: Fn(&T) -> String,
    {
        let mut result = ValidationResult::new();
        let mut seen_keys = HashMap::new();

        for (index, item) in items.iter().enumerate() {
            let key = key_extractor(item);

            if let Some(previous_index) = seen_keys.get(&key) {
                result.add_error(
                    "duplicates".to_string(),
                    format!(
                        "Duplicate entry found at indices {} and {}",
                        previous_index, index
                    ),
                    ValidationErrorType::Consistency,
                );
            } else {
                seen_keys.insert(key, index);
            }
        }

        result
    }
}
