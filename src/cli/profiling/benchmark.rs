//! Benchmarking utilities for performance testing

use super::profiler::ApplicationProfiler;
use super::*;
use crate::cli::error::Result;

/// Benchmark suite for comparing different implementation approaches
pub struct BenchmarkSuite {
    benchmarks: Vec<Benchmark>,
    baseline_result: Option<ProfilingResult>,
}

/// Individual benchmark test
#[derive(Debug, Clone)]
pub struct Benchmark {
    pub name: String,
    pub description: String,
    pub config: ProfilingConfig,
    pub result: Option<ProfilingResult>,
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline_name: String,
    pub comparisons: Vec<PerformanceComparison>,
    pub summary: ComparisonSummary,
}

/// Performance comparison between two results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub name: String,
    pub ops_per_second_ratio: f64,
    pub memory_usage_ratio: f64,
    pub search_latency_ratio: f64,
    pub improvement_percentage: f64,
    pub verdict: ComparisonVerdict,
}

/// Summary of all comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub total_benchmarks: usize,
    pub improvements: usize,
    pub regressions: usize,
    pub neutral: usize,
    pub best_performer: String,
    pub worst_performer: String,
    pub average_improvement: f64,
}

/// Verdict of a performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonVerdict {
    Improvement,
    Regression,
    Neutral,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            baseline_result: None,
        }
    }

    /// Add a benchmark to the suite
    pub fn add_benchmark(&mut self, name: String, description: String, config: ProfilingConfig) {
        self.benchmarks.push(Benchmark {
            name,
            description,
            config,
            result: None,
        });
    }

    /// Set the baseline result for comparisons
    pub fn set_baseline(&mut self, result: ProfilingResult) {
        self.baseline_result = Some(result);
    }

    /// Run all benchmarks in the suite
    pub async fn run_all_benchmarks(&mut self) -> Result<Vec<ProfilingResult>> {
        let mut results = Vec::new();

        for benchmark in &mut self.benchmarks {
            println!("Running benchmark: {}", benchmark.name);
            println!("Description: {}", benchmark.description);

            let mut profiler = ApplicationProfiler::new()?;
            let mut benchmark_results = profiler
                .run_profiling_suite(benchmark.config.clone())
                .await?;

            // Take the first result as the benchmark result
            if let Some(result) = benchmark_results.pop() {
                benchmark.result = Some(result.clone());
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Compare all benchmark results against the baseline
    pub fn compare_against_baseline(&self) -> Option<BenchmarkComparison> {
        let baseline = self.baseline_result.as_ref()?;
        let mut comparisons = Vec::new();

        for benchmark in &self.benchmarks {
            if let Some(result) = &benchmark.result {
                let comparison = self.compare_results(baseline, result, &benchmark.name);
                comparisons.push(comparison);
            }
        }

        if comparisons.is_empty() {
            return None;
        }

        let summary = self.generate_comparison_summary(&comparisons);

        Some(BenchmarkComparison {
            baseline_name: baseline.scenario_name.clone(),
            comparisons,
            summary,
        })
    }

    /// Compare two profiling results
    fn compare_results(
        &self,
        baseline: &ProfilingResult,
        test: &ProfilingResult,
        name: &str,
    ) -> PerformanceComparison {
        let ops_ratio = if baseline.metrics.ops_per_second > 0.0 {
            test.metrics.ops_per_second / baseline.metrics.ops_per_second
        } else {
            1.0
        };

        let memory_ratio = if baseline.metrics.peak_memory_mb > 0.0 {
            test.metrics.peak_memory_mb / baseline.metrics.peak_memory_mb
        } else {
            1.0
        };

        let search_latency_ratio = if baseline
            .metrics
            .search_metrics
            .avg_search_latency
            .as_nanos()
            > 0
        {
            test.metrics.search_metrics.avg_search_latency.as_nanos() as f64
                / baseline
                    .metrics
                    .search_metrics
                    .avg_search_latency
                    .as_nanos() as f64
        } else {
            1.0
        };

        // Calculate overall improvement percentage
        let improvement = ((ops_ratio - 1.0) * 100.0)
            - ((memory_ratio - 1.0) * 50.0)
            - ((search_latency_ratio - 1.0) * 25.0);

        let verdict = if improvement > 5.0 {
            ComparisonVerdict::Improvement
        } else if improvement < -5.0 {
            ComparisonVerdict::Regression
        } else {
            ComparisonVerdict::Neutral
        };

        PerformanceComparison {
            name: name.to_string(),
            ops_per_second_ratio: ops_ratio,
            memory_usage_ratio: memory_ratio,
            search_latency_ratio: search_latency_ratio,
            improvement_percentage: improvement,
            verdict,
        }
    }

    /// Generate summary of all comparisons
    fn generate_comparison_summary(
        &self,
        comparisons: &[PerformanceComparison],
    ) -> ComparisonSummary {
        let total = comparisons.len();
        let mut improvements = 0;
        let mut regressions = 0;
        let mut neutral = 0;

        let mut best_improvement = f64::NEG_INFINITY;
        let mut worst_improvement = f64::INFINITY;
        let mut best_performer = String::new();
        let mut worst_performer = String::new();
        let mut total_improvement = 0.0;

        for comparison in comparisons {
            match comparison.verdict {
                ComparisonVerdict::Improvement => improvements += 1,
                ComparisonVerdict::Regression => regressions += 1,
                ComparisonVerdict::Neutral => neutral += 1,
            }

            if comparison.improvement_percentage > best_improvement {
                best_improvement = comparison.improvement_percentage;
                best_performer = comparison.name.clone();
            }

            if comparison.improvement_percentage < worst_improvement {
                worst_improvement = comparison.improvement_percentage;
                worst_performer = comparison.name.clone();
            }

            total_improvement += comparison.improvement_percentage;
        }

        let average_improvement = if total > 0 {
            total_improvement / total as f64
        } else {
            0.0
        };

        ComparisonSummary {
            total_benchmarks: total,
            improvements,
            regressions,
            neutral,
            best_performer,
            worst_performer,
            average_improvement,
        }
    }

    /// Generate a detailed benchmark report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Benchmark Report\n");
        report.push_str("================\n\n");

        if let Some(baseline) = &self.baseline_result {
            report.push_str(&format!("Baseline: {}\n", baseline.scenario_name));
            report.push_str(&format!(
                "Dataset Size: {} entries\n",
                baseline.dataset_size
            ));
            report.push_str(&format!(
                "Baseline Performance: {:.2} ops/sec\n\n",
                baseline.metrics.ops_per_second
            ));
        }

        for benchmark in &self.benchmarks {
            report.push_str(&format!("Benchmark: {}\n", benchmark.name));
            report.push_str(&format!("Description: {}\n", benchmark.description));

            if let Some(result) = &benchmark.result {
                report.push_str(&format!(
                    "Performance: {:.2} ops/sec\n",
                    result.metrics.ops_per_second
                ));
                report.push_str(&format!(
                    "Memory Usage: {:.2} MB\n",
                    result.metrics.peak_memory_mb
                ));
                report.push_str(&format!(
                    "Search Latency: {:?}\n",
                    result.metrics.search_metrics.avg_search_latency
                ));
            } else {
                report.push_str("Result: Not run\n");
            }
            report.push_str("\n");
        }

        if let Some(comparison) = self.compare_against_baseline() {
            report.push_str("Performance Comparison\n");
            report.push_str("=====================\n\n");

            for comp in &comparison.comparisons {
                report.push_str(&format!("{}: ", comp.name));
                match comp.verdict {
                    ComparisonVerdict::Improvement => report.push_str("✅ IMPROVEMENT"),
                    ComparisonVerdict::Regression => report.push_str("❌ REGRESSION"),
                    ComparisonVerdict::Neutral => report.push_str("➖ NEUTRAL"),
                }
                report.push_str(&format!(" ({:.1}%)\n", comp.improvement_percentage));
                report.push_str(&format!(
                    "  Ops/sec: {:.2}x, Memory: {:.2}x, Latency: {:.2}x\n\n",
                    comp.ops_per_second_ratio, comp.memory_usage_ratio, comp.search_latency_ratio
                ));
            }

            report.push_str("Summary:\n");
            report.push_str(&format!(
                "Improvements: {}\n",
                comparison.summary.improvements
            ));
            report.push_str(&format!(
                "Regressions: {}\n",
                comparison.summary.regressions
            ));
            report.push_str(&format!("Neutral: {}\n", comparison.summary.neutral));
            report.push_str(&format!(
                "Best Performer: {}\n",
                comparison.summary.best_performer
            ));
            report.push_str(&format!(
                "Average Improvement: {:.1}%\n",
                comparison.summary.average_improvement
            ));
        }

        report
    }

    /// Save benchmark results and report
    pub async fn save_results(&self, output_dir: &std::path::Path) -> Result<()> {
        tokio::fs::create_dir_all(output_dir).await?;

        // Save individual benchmark results
        for (i, benchmark) in self.benchmarks.iter().enumerate() {
            if let Some(result) = &benchmark.result {
                let filename = format!("benchmark_{}_{}.json", i, benchmark.name.replace(' ', "_"));
                let filepath = output_dir.join(filename);
                result.save_to_file(&filepath).await?;
            }
        }

        // Save baseline result
        if let Some(baseline) = &self.baseline_result {
            let baseline_path = output_dir.join("baseline_result.json");
            baseline.save_to_file(&baseline_path).await?;
        }

        // Save comparison results
        if let Some(comparison) = self.compare_against_baseline() {
            let comparison_path = output_dir.join("benchmark_comparison.json");
            let content = serde_json::to_string_pretty(&comparison)?;
            tokio::fs::write(&comparison_path, content).await?;
        }

        // Save text report
        let report_path = output_dir.join("benchmark_report.txt");
        let report = self.generate_report();
        tokio::fs::write(&report_path, report).await?;

        Ok(())
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a standard benchmark suite for common performance tests
pub fn create_standard_benchmark_suite() -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::new();

    // Baseline configuration
    let baseline_config = ProfilingConfig {
        operation_count: 1000,
        dataset_size: 5000,
        operation_types: vec![OperationType::HistoryStore, OperationType::HistorySearch],
        profile_memory: true,
        profile_storage: true,
        warmup: false,
        warmup_iterations: 0,
    };

    // Small dataset benchmark
    let small_config = ProfilingConfig {
        dataset_size: 1000,
        ..baseline_config.clone()
    };
    suite.add_benchmark(
        "Small Dataset".to_string(),
        "Performance with 1K entries".to_string(),
        small_config,
    );

    // Large dataset benchmark
    let large_config = ProfilingConfig {
        dataset_size: 20000,
        operation_count: 2000,
        ..baseline_config.clone()
    };
    suite.add_benchmark(
        "Large Dataset".to_string(),
        "Performance with 20K entries".to_string(),
        large_config,
    );

    // Memory-intensive benchmark
    let memory_config = ProfilingConfig {
        operation_count: 5000,
        dataset_size: 10000,
        ..baseline_config.clone()
    };
    suite.add_benchmark(
        "Memory Intensive".to_string(),
        "High memory usage scenario".to_string(),
        memory_config,
    );

    // Search-focused benchmark
    let search_config = ProfilingConfig {
        operation_count: 2000,
        operation_types: vec![OperationType::HistorySearch, OperationType::FullTextSearch],
        ..baseline_config.clone()
    };
    suite.add_benchmark(
        "Search Focused".to_string(),
        "Search-heavy workload".to_string(),
        search_config,
    );

    suite
}
