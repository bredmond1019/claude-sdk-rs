# Tutorial: Automated Test Generator with claude-sdk-rs

This tutorial demonstrates how to build an intelligent test generator that analyzes your code and automatically creates comprehensive unit tests, integration tests, and property-based tests using the claude-sdk-rs SDK.

## Table of Contents

1. [Overview](#overview)
2. [Project Setup](#project-setup)
3. [Code Analysis and Test Planning](#code-analysis-and-test-planning)
4. [Unit Test Generation](#unit-test-generation)
5. [Integration Test Generation](#integration-test-generation)
6. [Property-Based Test Generation](#property-based-test-generation)
7. [Test Quality Assessment](#test-quality-assessment)
8. [CLI and CI Integration](#cli-and-ci-integration)

## Overview

Our automated test generator will:

- **Analyze code structure** - Understand functions, types, and dependencies
- **Generate unit tests** - Create comprehensive test cases for individual functions
- **Create integration tests** - Test component interactions and workflows
- **Property-based testing** - Generate tests that verify invariants and properties
- **Mock generation** - Create mocks and test doubles automatically
- **Coverage analysis** - Ensure comprehensive test coverage
- **Quality assessment** - Evaluate and improve generated tests

## Project Setup

Create a new Rust project with comprehensive testing dependencies:

```toml
[package]
name = "test-generator"
version = "0.1.0"
edition = "2021"

[dependencies]
claude-sdk-rs = { version = "0.1", features = ["tools"] }
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.0", features = ["derive"] }
syn = { version = "2.0", features = ["full", "parsing", "extra-traits"] }
quote = "1.0"
proc-macro2 = "1.0"
walkdir = "2.0"
regex = "1.0"
anyhow = "1.0"
tempfile = "3.0"

# Testing framework dependencies for generated tests
proptest = "1.0"
mockall = "0.11"
mockito = "1.0"
wiremock = "0.5"
pretty_assertions = "1.0"
insta = "1.0"
criterion = "0.5"
```

## Code Analysis and Test Planning

Let's start by building a comprehensive code analyzer that understands what to test:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission, StreamFormat};
use serde::{Deserialize, Serialize};
use syn::{File, Item, ItemFn, ItemStruct, ItemEnum, Type, Visibility};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Serialize, Deserialize)]
pub struct TestPlan {
    pub target_file: PathBuf,
    pub functions_to_test: Vec<FunctionTestPlan>,
    pub types_to_test: Vec<TypeTestPlan>,
    pub integration_scenarios: Vec<IntegrationScenario>,
    pub property_tests: Vec<PropertyTest>,
    pub mock_requirements: Vec<MockRequirement>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionTestPlan {
    pub function_name: String,
    pub signature: String,
    pub visibility: String,
    pub test_cases: Vec<TestCase>,
    pub error_cases: Vec<ErrorCase>,
    pub edge_cases: Vec<EdgeCase>,
    pub performance_tests: Vec<PerformanceTest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub description: String,
    pub inputs: Vec<TestInput>,
    pub expected_output: String,
    pub setup: Option<String>,
    pub teardown: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestInput {
    pub name: String,
    pub value: String,
    pub input_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorCase {
    pub name: String,
    pub description: String,
    pub inputs: Vec<TestInput>,
    pub expected_error: String,
    pub error_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeCase {
    pub name: String,
    pub description: String,
    pub scenario: String,
    pub expected_behavior: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceTest {
    pub name: String,
    pub description: String,
    pub benchmark_type: BenchmarkType,
    pub expected_complexity: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum BenchmarkType {
    Throughput,
    Latency,
    Memory,
    Complexity,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TypeTestPlan {
    pub type_name: String,
    pub type_kind: TypeKind,
    pub construction_tests: Vec<TestCase>,
    pub serialization_tests: Vec<TestCase>,
    pub invariant_tests: Vec<InvariantTest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TypeKind {
    Struct,
    Enum,
    Trait,
    NewType,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InvariantTest {
    pub name: String,
    pub description: String,
    pub invariant: String,
    pub property: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrationScenario {
    pub name: String,
    pub description: String,
    pub components: Vec<String>,
    pub workflow: Vec<WorkflowStep>,
    pub expected_outcome: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub action: String,
    pub component: String,
    pub input: Option<String>,
    pub expected_result: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PropertyTest {
    pub name: String,
    pub description: String,
    pub property: String,
    pub generators: Vec<PropertyGenerator>,
    pub shrinking_strategy: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PropertyGenerator {
    pub parameter: String,
    pub generator_type: String,
    pub constraints: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MockRequirement {
    pub trait_name: String,
    pub methods: Vec<String>,
    pub mock_behavior: Vec<MockBehavior>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MockBehavior {
    pub method: String,
    pub scenario: String,
    pub return_value: String,
}

pub struct TestAnalyzer {
    claude_client: Client,
    project_root: PathBuf,
}

impl TestAnalyzer {
    pub fn new(project_root: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Json)
            .system_prompt(include_str!("../prompts/test_generation_system.txt"))
            .timeout_secs(180)
            .allowed_tools(vec![
                ToolPermission::mcp("filesystem", "read").to_cli_format(),
                ToolPermission::bash("find").to_cli_format(),
                ToolPermission::bash("grep").to_cli_format(),
            ])
            .build();

        let claude_client = Client::new(config);

        Ok(Self {
            claude_client,
            project_root,
        })
    }

    pub async fn analyze_and_plan_tests(&self, target_file: &Path) -> Result<TestPlan, Box<dyn std::error::Error>> {
        println!("🔍 Analyzing {} for test generation...", target_file.display());

        // Parse the Rust code
        let ast = self.parse_rust_file(target_file)?;
        
        // Extract functions and types
        let (functions, types) = self.extract_items(&ast);

        // Analyze each function with Claude
        let functions_to_test = self.analyze_functions_for_testing(&functions, target_file).await?;
        
        // Analyze types for testing
        let types_to_test = self.analyze_types_for_testing(&types, target_file).await?;

        // Plan integration tests
        let integration_scenarios = self.plan_integration_tests(target_file).await?;

        // Generate property-based tests
        let property_tests = self.generate_property_tests(&functions, &types).await?;

        // Identify mock requirements
        let mock_requirements = self.identify_mock_requirements(&functions, target_file).await?;

        Ok(TestPlan {
            target_file: target_file.to_path_buf(),
            functions_to_test,
            types_to_test,
            integration_scenarios,
            property_tests,
            mock_requirements,
        })
    }

    fn parse_rust_file(&self, file_path: &Path) -> Result<File, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;
        let ast = syn::parse_file(&content)?;
        Ok(ast)
    }

    fn extract_items(&self, ast: &File) -> (Vec<&ItemFn>, Vec<&Item>) {
        let mut functions = Vec::new();
        let mut types = Vec::new();

        for item in &ast.items {
            match item {
                Item::Fn(item_fn) => {
                    // Only include public functions or functions with pub(crate)
                    if matches!(item_fn.vis, Visibility::Public(_) | Visibility::Restricted(_)) {
                        functions.push(item_fn);
                    }
                }
                Item::Struct(_) | Item::Enum(_) | Item::Trait(_) => {
                    types.push(item);
                }
                _ => {}
            }
        }

        (functions, types)
    }

    async fn analyze_functions_for_testing(
        &self,
        functions: &[&ItemFn],
        file_path: &Path,
    ) -> Result<Vec<FunctionTestPlan>, Box<dyn std::error::Error>> {
        let mut function_plans = Vec::new();

        for function in functions {
            let function_code = quote::quote!(#function).to_string();
            let plan = self.analyze_single_function(&function.sig.ident.to_string(), &function_code, file_path).await?;
            function_plans.push(plan);
        }

        Ok(function_plans)
    }

    async fn analyze_single_function(
        &self,
        function_name: &str,
        function_code: &str,
        file_path: &Path,
    ) -> Result<FunctionTestPlan, Box<dyn std::error::Error>> {
        let file_content = std::fs::read_to_string(file_path)?;
        
        let prompt = format!(
            "Analyze this Rust function and create a comprehensive test plan:\n\n\
             Function: {}\n\n\
             ```rust\n{}\n```\n\n\
             Context (full file):\n\
             ```rust\n{}\n```\n\n\
             Create a test plan that includes:\n\
             1. Normal test cases with various valid inputs\n\
             2. Error cases and edge cases\n\
             3. Performance tests if applicable\n\
             4. Boundary conditions and corner cases\n\n\
             Consider:\n\
             - Function parameters and their types\n\
             - Return type and possible values\n\
             - Error conditions and panics\n\
             - Side effects and state changes\n\
             - Performance characteristics\n\n\
             Format as JSON matching the FunctionTestPlan structure.",
            function_name,
            function_code,
            file_content
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_function_test_plan(&response.content)
    }

    async fn analyze_types_for_testing(
        &self,
        types: &[&Item],
        file_path: &Path,
    ) -> Result<Vec<TypeTestPlan>, Box<dyn std::error::Error>> {
        let mut type_plans = Vec::new();

        for type_item in types {
            let type_code = quote::quote!(#type_item).to_string();
            let plan = self.analyze_single_type(&type_code, file_path).await?;
            type_plans.push(plan);
        }

        Ok(type_plans)
    }

    async fn analyze_single_type(
        &self,
        type_code: &str,
        file_path: &Path,
    ) -> Result<TypeTestPlan, Box<dyn std::error::Error>> {
        let prompt = format!(
            "Analyze this Rust type and create tests for it:\n\n\
             ```rust\n{}\n```\n\n\
             Create tests for:\n\
             1. Construction and initialization\n\
             2. Serialization/deserialization if applicable\n\
             3. Type invariants and constraints\n\
             4. Method behavior if it's a struct with methods\n\
             5. Enum variant handling if it's an enum\n\n\
             Format as JSON matching the TypeTestPlan structure.",
            type_code
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_type_test_plan(&response.content)
    }

    async fn plan_integration_tests(&self, file_path: &Path) -> Result<Vec<IntegrationScenario>, Box<dyn std::error::Error>> {
        let file_content = std::fs::read_to_string(file_path)?;

        let prompt = format!(
            "Analyze this code and identify integration test scenarios:\n\n\
             ```rust\n{}\n```\n\n\
             Look for:\n\
             1. Component interactions\n\
             2. External dependencies\n\
             3. API workflows\n\
             4. State management patterns\n\
             5. Error propagation paths\n\n\
             Create integration test scenarios that verify these components work together correctly.\n\
             Format as JSON array of IntegrationScenario objects.",
            file_content
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_integration_scenarios(&response.content)
    }

    async fn generate_property_tests(
        &self,
        functions: &[&ItemFn],
        types: &[&Item],
    ) -> Result<Vec<PropertyTest>, Box<dyn std::error::Error>> {
        let functions_summary = functions.iter()
            .map(|f| format!("- {}", f.sig.ident))
            .collect::<Vec<_>>()
            .join("\n");

        let types_summary = types.iter()
            .map(|t| match t {
                Item::Struct(s) => format!("- struct {}", s.ident),
                Item::Enum(e) => format!("- enum {}", e.ident),
                Item::Trait(t) => format!("- trait {}", t.ident),
                _ => "- other".to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Generate property-based tests for these functions and types:\n\n\
             Functions:\n{}\n\n\
             Types:\n{}\n\n\
             Create property tests that verify:\n\
             1. Mathematical properties (associativity, commutativity, etc.)\n\
             2. Invariants that should always hold\n\
             3. Roundtrip properties (serialize/deserialize, encode/decode)\n\
             4. Idempotency where applicable\n\
             5. Monotonicity and ordering properties\n\n\
             Use proptest-style generators and shrinking strategies.\n\
             Format as JSON array of PropertyTest objects.",
            functions_summary,
            types_summary
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_property_tests(&response.content)
    }

    async fn identify_mock_requirements(
        &self,
        functions: &[&ItemFn],
        file_path: &Path,
    ) -> Result<Vec<MockRequirement>, Box<dyn std::error::Error>> {
        let file_content = std::fs::read_to_string(file_path)?;

        let prompt = format!(
            "Analyze this code and identify what needs to be mocked for testing:\n\n\
             ```rust\n{}\n```\n\n\
             Look for:\n\
             1. Trait dependencies that should be mocked\n\
             2. External service calls\n\
             3. File system operations\n\
             4. Network requests\n\
             5. Database operations\n\
             6. Time-dependent code\n\n\
             Create mock requirements with appropriate behaviors for different test scenarios.\n\
             Format as JSON array of MockRequirement objects.",
            file_content
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_mock_requirements(&response.content)
    }

    // Parsing methods for Claude responses
    fn parse_function_test_plan(&self, response: &str) -> Result<FunctionTestPlan, Box<dyn std::error::Error>> {
        let json_start = response.find('{').ok_or("No JSON found")?;
        let json_end = response.rfind('}').ok_or("No JSON found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let plan: FunctionTestPlan = serde_json::from_str(json_str)?;
        Ok(plan)
    }

    fn parse_type_test_plan(&self, response: &str) -> Result<TypeTestPlan, Box<dyn std::error::Error>> {
        let json_start = response.find('{').ok_or("No JSON found")?;
        let json_end = response.rfind('}').ok_or("No JSON found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let plan: TypeTestPlan = serde_json::from_str(json_str)?;
        Ok(plan)
    }

    fn parse_integration_scenarios(&self, response: &str) -> Result<Vec<IntegrationScenario>, Box<dyn std::error::Error>> {
        let json_start = response.find('[').ok_or("No JSON array found")?;
        let json_end = response.rfind(']').ok_or("No JSON array found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let scenarios: Vec<IntegrationScenario> = serde_json::from_str(json_str)?;
        Ok(scenarios)
    }

    fn parse_property_tests(&self, response: &str) -> Result<Vec<PropertyTest>, Box<dyn std::error::Error>> {
        let json_start = response.find('[').ok_or("No JSON array found")?;
        let json_end = response.rfind(']').ok_or("No JSON array found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let tests: Vec<PropertyTest> = serde_json::from_str(json_str)?;
        Ok(tests)
    }

    fn parse_mock_requirements(&self, response: &str) -> Result<Vec<MockRequirement>, Box<dyn std::error::Error>> {
        let json_start = response.find('[').ok_or("No JSON array found")?;
        let json_end = response.rfind(']').ok_or("No JSON array found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let requirements: Vec<MockRequirement> = serde_json::from_str(json_str)?;
        Ok(requirements)
    }
}
```

## Unit Test Generation

Now let's create the unit test generator:

```rust
use proc_macro2::TokenStream;
use quote::quote;
use std::fs;

pub struct UnitTestGenerator {
    claude_client: Client,
}

impl UnitTestGenerator {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Text)
            .system_prompt("You are an expert Rust developer creating comprehensive unit tests. Generate idiomatic, well-documented test code.")
            .timeout_secs(120)
            .build();

        let claude_client = Client::new(config);

        Ok(Self { claude_client })
    }

    pub async fn generate_unit_tests(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        println!("🧪 Generating unit tests...");

        let mut test_code = String::new();

        // Generate file header
        test_code.push_str(&self.generate_test_file_header(test_plan)?);

        // Generate tests for each function
        for function_plan in &test_plan.functions_to_test {
            let function_tests = self.generate_function_tests(function_plan).await?;
            test_code.push_str(&function_tests);
            test_code.push('\n');
        }

        // Generate tests for types
        for type_plan in &test_plan.types_to_test {
            let type_tests = self.generate_type_tests(type_plan).await?;
            test_code.push_str(&type_tests);
            test_code.push('\n');
        }

        Ok(test_code)
    }

    fn generate_test_file_header(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        let target_module = test_plan.target_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("target");

        let header = format!(
            r#"//! Unit tests for {}
//! 
//! This file was automatically generated using claude-sdk-rs test generator.
//! 
//! To run these tests:
//! ```bash
//! cargo test
//! ```

use super::*;
use pretty_assertions::assert_eq;
use proptest::prelude::*;
use mockall::predicate::*;

"#,
            target_module
        );

        Ok(header)
    }

    async fn generate_function_tests(&self, function_plan: &FunctionTestPlan) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = format!(
            "Generate comprehensive Rust unit tests for this function:\n\n\
             Function: {}\n\
             Signature: {}\n\n\
             Test Cases:\n{}\n\n\
             Error Cases:\n{}\n\n\
             Edge Cases:\n{}\n\n\
             Generate Rust test functions using the standard #[test] attribute.\n\
             Include:\n\
             1. Clear test names and documentation\n\
             2. Proper setup and teardown\n\
             3. Assertions with descriptive messages\n\
             4. Error testing with #[should_panic] or Result<(), Error>\n\
             5. Helper functions if needed\n\n\
             Use idiomatic Rust testing patterns and include comments explaining the test logic.",
            function_plan.function_name,
            function_plan.signature,
            function_plan.test_cases.iter()
                .map(|tc| format!("- {}: {}", tc.name, tc.description))
                .collect::<Vec<_>>()
                .join("\n"),
            function_plan.error_cases.iter()
                .map(|ec| format!("- {}: {}", ec.name, ec.description))
                .collect::<Vec<_>>()
                .join("\n"),
            function_plan.edge_cases.iter()
                .map(|ec| format!("- {}: {}", ec.name, ec.description))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let test_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(format!("// Tests for {}\n{}\n", function_plan.function_name, test_code))
    }

    async fn generate_type_tests(&self, type_plan: &TypeTestPlan) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = format!(
            "Generate Rust unit tests for this type:\n\n\
             Type: {} ({:?})\n\n\
             Construction Tests:\n{}\n\n\
             Serialization Tests:\n{}\n\n\
             Invariant Tests:\n{}\n\n\
             Generate comprehensive tests including:\n\
             1. Constructor tests and field validation\n\
             2. Serialization/deserialization roundtrips\n\
             3. Type invariant verification\n\
             4. Clone, Debug, PartialEq implementations if applicable\n\
             5. Builder pattern tests if applicable\n\n\
             Use appropriate test attributes and assertions.",
            type_plan.type_name,
            type_plan.type_kind,
            type_plan.construction_tests.iter()
                .map(|ct| format!("- {}: {}", ct.name, ct.description))
                .collect::<Vec<_>>()
                .join("\n"),
            type_plan.serialization_tests.iter()
                .map(|st| format!("- {}: {}", st.name, st.description))
                .collect::<Vec<_>>()
                .join("\n"),
            type_plan.invariant_tests.iter()
                .map(|it| format!("- {}: {}", it.name, it.description))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let test_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(format!("// Tests for {} type\n{}\n", type_plan.type_name, test_code))
    }

    pub async fn generate_benchmark_tests(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        println!("⚡ Generating benchmark tests...");

        let performance_tests: Vec<_> = test_plan.functions_to_test.iter()
            .flat_map(|f| &f.performance_tests)
            .collect();

        if performance_tests.is_empty() {
            return Ok("// No performance tests identified\n".to_string());
        }

        let prompt = format!(
            "Generate Criterion.rs benchmark tests for these performance scenarios:\n\n\
             {}\n\n\
             Create comprehensive benchmarks that:\n\
             1. Use criterion::black_box to prevent optimization\n\
             2. Test with various input sizes\n\
             3. Include parameterized benchmarks\n\
             4. Measure appropriate metrics (throughput, latency, etc.)\n\
             5. Include baseline comparisons where relevant\n\n\
             Generate a complete benches/benchmarks.rs file.",
            performance_tests.iter()
                .map(|pt| format!("- {}: {} ({:?})", pt.name, pt.description, pt.benchmark_type))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let benchmark_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(benchmark_code)
    }
}
```

## Integration Test Generation

Create integration tests that verify component interactions:

```rust
pub struct IntegrationTestGenerator {
    claude_client: Client,
}

impl IntegrationTestGenerator {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Text)
            .system_prompt("You are an expert at creating integration tests that verify component interactions and end-to-end workflows.")
            .timeout_secs(180)
            .build();

        let claude_client = Client::new(config);

        Ok(Self { claude_client })
    }

    pub async fn generate_integration_tests(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        println!("🔗 Generating integration tests...");

        let mut test_code = String::new();

        // Generate test file header
        test_code.push_str(&self.generate_integration_header(test_plan)?);

        // Generate mock setup
        test_code.push_str(&self.generate_mock_setup(&test_plan.mock_requirements).await?);

        // Generate integration test scenarios
        for scenario in &test_plan.integration_scenarios {
            let scenario_test = self.generate_integration_scenario(scenario).await?;
            test_code.push_str(&scenario_test);
            test_code.push('\n');
        }

        Ok(test_code)
    }

    fn generate_integration_header(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!(
            r#"//! Integration tests for {}
//! 
//! These tests verify that components work correctly together
//! and that end-to-end workflows function as expected.

use tokio_test;
use wiremock::{{MockServer, Mock, ResponseTemplate}};
use tempfile::TempDir;
use std::sync::Arc;

"#,
            test_plan.target_file.display()
        ))
    }

    async fn generate_mock_setup(&self, mock_requirements: &[MockRequirement]) -> Result<String, Box<dyn std::error::Error>> {
        if mock_requirements.is_empty() {
            return Ok(String::new());
        }

        let prompt = format!(
            "Generate mock setup code for these requirements:\n\n\
             {}\n\n\
             Create:\n\
             1. Mock trait implementations using mockall\n\
             2. Mock server setups using wiremock where applicable\n\
             3. Test fixtures and builders\n\
             4. Helper functions for common mock scenarios\n\n\
             Use Rust's mockall crate and include proper mock expectations.",
            mock_requirements.iter()
                .map(|mr| format!("Trait: {}\nMethods: {}\nBehaviors: {}",
                    mr.trait_name,
                    mr.methods.join(", "),
                    mr.mock_behavior.iter()
                        .map(|mb| format!("{}: {}", mb.method, mb.scenario))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
                .collect::<Vec<_>>()
                .join("\n\n")
        );

        let mock_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(format!("// Mock setup\n{}\n", mock_code))
    }

    async fn generate_integration_scenario(&self, scenario: &IntegrationScenario) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = format!(
            "Generate an integration test for this scenario:\n\n\
             Scenario: {}\n\
             Description: {}\n\
             Components: {}\n\n\
             Workflow:\n{}\n\n\
             Expected Outcome: {}\n\n\
             Create a comprehensive integration test that:\n\
             1. Sets up all necessary components\n\
             2. Executes the workflow step by step\n\
             3. Verifies intermediate states\n\
             4. Asserts the final outcome\n\
             5. Includes proper error handling\n\
             6. Cleans up resources\n\n\
             Use async test functions where appropriate and include detailed assertions.",
            scenario.name,
            scenario.description,
            scenario.components.join(", "),
            scenario.workflow.iter()
                .map(|step| format!("  {}: {} -> {}", step.action, step.component, step.expected_result))
                .collect::<Vec<_>>()
                .join("\n"),
            scenario.expected_outcome
        );

        let test_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(format!("// Integration test: {}\n{}\n", scenario.name, test_code))
    }

    pub async fn generate_end_to_end_tests(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        println!("🌐 Generating end-to-end tests...");

        // Look for API endpoints or main application flows
        let endpoints_summary = test_plan.integration_scenarios.iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Generate end-to-end tests for this application:\n\n\
             Integration Scenarios:\n{}\n\n\
             Create comprehensive E2E tests that:\n\
             1. Test complete user workflows\n\
             2. Use realistic test data\n\
             3. Verify system behavior under load\n\
             4. Test error scenarios and recovery\n\
             5. Include setup and teardown for test environments\n\n\
             Use appropriate testing frameworks and include database seeding,\n\
             API client setup, and other infrastructure components.",
            endpoints_summary
        );

        let e2e_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(e2e_code)
    }
}
```

## Property-Based Test Generation

Generate property-based tests using proptest:

```rust
pub struct PropertyTestGenerator {
    claude_client: Client,
}

impl PropertyTestGenerator {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Text)
            .system_prompt("You are an expert at creating property-based tests using proptest. Generate tests that verify mathematical properties and invariants.")
            .timeout_secs(120)
            .build();

        let claude_client = Client::new(config);

        Ok(Self { claude_client })
    }

    pub async fn generate_property_tests(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        println!("🎲 Generating property-based tests...");

        if test_plan.property_tests.is_empty() {
            return Ok("// No property-based tests identified\n".to_string());
        }

        let mut test_code = String::new();

        // Generate header
        test_code.push_str(&self.generate_property_header()?);

        // Generate each property test
        for property_test in &test_plan.property_tests {
            let test = self.generate_single_property_test(property_test).await?;
            test_code.push_str(&test);
            test_code.push('\n');
        }

        Ok(test_code)
    }

    fn generate_property_header(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(r#"//! Property-based tests
//! 
//! These tests use proptest to verify properties that should hold
//! for all valid inputs, not just specific test cases.

use proptest::prelude::*;
use proptest::test_runner::Config;

"#.to_string())
    }

    async fn generate_single_property_test(&self, property_test: &PropertyTest) -> Result<String, Box<dyn std::error::Error>> {
        let generators_desc = property_test.generators.iter()
            .map(|g| format!("  {}: {} with constraints: {}", 
                g.parameter, 
                g.generator_type, 
                g.constraints.join(", ")))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Generate a proptest property-based test:\n\n\
             Property: {}\n\
             Description: {}\n\
             Property to verify: {}\n\n\
             Generators:\n{}\n\n\
             Shrinking strategy: {}\n\n\
             Create a complete proptest test that:\n\
             1. Defines appropriate strategies for input generation\n\
             2. Implements the property check\n\
             3. Uses proper proptest macros (proptest! or prop_assert!)\n\
             4. Includes edge case handling\n\
             5. Has clear failure messages\n\n\
             Generate idiomatic Rust code using proptest patterns.",
            property_test.name,
            property_test.description,
            property_test.property,
            generators_desc,
            property_test.shrinking_strategy
        );

        let test_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(format!("// Property test: {}\n{}\n", property_test.name, test_code))
    }

    pub async fn generate_roundtrip_tests(&self, test_plan: &TestPlan) -> Result<String, Box<dyn std::error::Error>> {
        let types_with_serialization: Vec<_> = test_plan.types_to_test.iter()
            .filter(|t| !t.serialization_tests.is_empty())
            .collect();

        if types_with_serialization.is_empty() {
            return Ok("// No serialization roundtrip tests needed\n".to_string());
        }

        let prompt = format!(
            "Generate roundtrip property tests for these serializable types:\n\n\
             {}\n\n\
             Create proptest roundtrip tests that verify:\n\
             1. Serialize -> Deserialize -> Original\n\
             2. JSON roundtrips\n\
             3. Binary format roundtrips (if applicable)\n\
             4. Property preservation across serialization\n\n\
             Use proptest to generate random instances of each type.",
            types_with_serialization.iter()
                .map(|t| format!("- {} ({:?})", t.type_name, t.type_kind))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let roundtrip_code = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(format!("// Roundtrip property tests\n{}\n", roundtrip_code))
    }
}
```

## Test Quality Assessment

Create a quality assessor that evaluates and improves generated tests:

```rust
pub struct TestQualityAssessor {
    claude_client: Client,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestQualityReport {
    pub overall_score: u8,
    pub coverage_analysis: CoverageAnalysis,
    pub quality_metrics: QualityMetrics,
    pub recommendations: Vec<Recommendation>,
    pub missing_tests: Vec<MissingTest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    pub line_coverage: f64,
    pub branch_coverage: f64,
    pub function_coverage: f64,
    pub uncovered_lines: Vec<u32>,
    pub uncovered_branches: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub test_count: usize,
    pub assertion_count: usize,
    pub avg_test_complexity: f64,
    pub mock_usage_score: u8,
    pub documentation_score: u8,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub priority: Priority,
    pub description: String,
    pub example: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Coverage,
    EdgeCases,
    Performance,
    Documentation,
    Maintainability,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Priority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MissingTest {
    pub test_type: String,
    pub description: String,
    pub suggested_approach: String,
}

impl TestQualityAssessor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Json)
            .system_prompt("You are an expert test quality assessor. Analyze test code and provide detailed quality assessments and improvement recommendations.")
            .timeout_secs(120)
            .build();

        let claude_client = Client::new(config);

        Ok(Self { claude_client })
    }

    pub async fn assess_test_quality(
        &self,
        generated_tests: &str,
        original_code: &str,
    ) -> Result<TestQualityReport, Box<dyn std::error::Error>> {
        println!("📊 Assessing test quality...");

        let prompt = format!(
            "Analyze the quality of these generated tests:\n\n\
             Original Code:\n\
             ```rust\n{}\n```\n\n\
             Generated Tests:\n\
             ```rust\n{}\n```\n\n\
             Provide a comprehensive quality assessment including:\n\
             1. Test coverage analysis (estimated line, branch, function coverage)\n\
             2. Quality metrics (test count, complexity, documentation)\n\
             3. Specific recommendations for improvement\n\
             4. Missing test scenarios\n\
             5. Overall quality score (0-100)\n\n\
             Consider:\n\
             - Edge cases and error conditions\n\
             - Test maintainability and readability\n\
             - Mock usage appropriateness\n\
             - Performance test coverage\n\
             - Documentation quality\n\n\
             Format as JSON matching the TestQualityReport structure.",
            original_code,
            generated_tests
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_quality_report(&response.content)
    }

    pub async fn suggest_test_improvements(
        &self,
        quality_report: &TestQualityReport,
        original_tests: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        println!("💡 Generating test improvements...");

        let recommendations_summary = quality_report.recommendations.iter()
            .map(|r| format!("- {:?} ({}): {}", r.category, 
                match r.priority {
                    Priority::High => "HIGH",
                    Priority::Medium => "MED", 
                    Priority::Low => "LOW",
                }, r.description))
            .collect::<Vec<_>>()
            .join("\n");

        let missing_tests_summary = quality_report.missing_tests.iter()
            .map(|mt| format!("- {}: {}", mt.test_type, mt.description))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Improve these tests based on the quality assessment:\n\n\
             Current Tests:\n\
             ```rust\n{}\n```\n\n\
             Quality Score: {}/100\n\n\
             Recommendations:\n{}\n\n\
             Missing Tests:\n{}\n\n\
             Generate improved test code that addresses the highest priority recommendations.\n\
             Focus on:\n\
             1. Adding missing test cases\n\
             2. Improving test documentation\n\
             3. Enhancing edge case coverage\n\
             4. Better error testing\n\
             5. Performance test additions\n\n\
             Provide the complete improved test code, not just additions.",
            original_tests,
            quality_report.overall_score,
            recommendations_summary,
            missing_tests_summary
        );

        let improved_tests = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(improved_tests)
    }

    fn parse_quality_report(&self, response: &str) -> Result<TestQualityReport, Box<dyn std::error::Error>> {
        let json_start = response.find('{').ok_or("No JSON found")?;
        let json_end = response.rfind('}').ok_or("No JSON found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let report: TestQualityReport = serde_json::from_str(json_str)?;
        Ok(report)
    }
}
```

## CLI and CI Integration

Finally, let's create a comprehensive CLI tool:

```rust
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tempfile::TempDir;

#[derive(Parser)]
#[command(name = "test-generator")]
#[command(about = "AI-powered test generator for Rust projects")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate tests for a specific file
    Generate {
        /// Source file to generate tests for
        #[arg(short, long)]
        file: PathBuf,
        
        /// Output directory for generated tests
        #[arg(short, long, default_value = "tests")]
        output: PathBuf,
        
        /// Test types to generate
        #[arg(short, long, default_value = "unit,integration")]
        types: String,
        
        /// Run quality assessment
        #[arg(short, long)]
        assess_quality: bool,
    },
    
    /// Generate tests for entire project
    Project {
        /// Project root directory
        #[arg(short, long, default_value = ".")]
        root: PathBuf,
        
        /// Output directory for generated tests
        #[arg(short, long, default_value = "generated_tests")]
        output: PathBuf,
        
        /// Include benchmarks
        #[arg(short, long)]
        benchmarks: bool,
    },
    
    /// Assess quality of existing tests
    Assess {
        /// Test file to assess
        test_file: PathBuf,
        
        /// Corresponding source file
        source_file: PathBuf,
        
        /// Generate improvement suggestions
        #[arg(short, long)]
        improve: bool,
    },
    
    /// Watch files and regenerate tests
    Watch {
        /// Directory to watch
        #[arg(short, long, default_value = "src")]
        directory: PathBuf,
        
        /// Output directory
        #[arg(short, long, default_value = "tests")]
        output: PathBuf,
    },
}

pub struct TestGeneratorApp {
    analyzer: TestAnalyzer,
    unit_generator: UnitTestGenerator,
    integration_generator: IntegrationTestGenerator,
    property_generator: PropertyTestGenerator,
    quality_assessor: TestQualityAssessor,
}

impl TestGeneratorApp {
    pub fn new(project_root: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            analyzer: TestAnalyzer::new(project_root)?,
            unit_generator: UnitTestGenerator::new()?,
            integration_generator: IntegrationTestGenerator::new()?,
            property_generator: PropertyTestGenerator::new()?,
            quality_assessor: TestQualityAssessor::new()?,
        })
    }

    pub async fn generate_tests_for_file(
        &self,
        file_path: &Path,
        output_dir: &Path,
        test_types: &[&str],
        assess_quality: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Generating tests for {}", file_path.display());

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Analyze the file and create test plan
        let test_plan = self.analyzer.analyze_and_plan_tests(file_path).await?;

        // Generate different types of tests
        if test_types.contains(&"unit") {
            let unit_tests = self.unit_generator.generate_unit_tests(&test_plan).await?;
            let unit_test_file = output_dir.join(format!("{}_tests.rs", 
                file_path.file_stem().unwrap().to_str().unwrap()));
            std::fs::write(unit_test_file, unit_tests)?;
            println!("✅ Unit tests generated");
        }

        if test_types.contains(&"integration") {
            let integration_tests = self.integration_generator.generate_integration_tests(&test_plan).await?;
            let integration_test_file = output_dir.join(format!("{}_integration_tests.rs", 
                file_path.file_stem().unwrap().to_str().unwrap()));
            std::fs::write(integration_test_file, integration_tests)?;
            println!("✅ Integration tests generated");
        }

        if test_types.contains(&"property") {
            let property_tests = self.property_generator.generate_property_tests(&test_plan).await?;
            let property_test_file = output_dir.join(format!("{}_property_tests.rs", 
                file_path.file_stem().unwrap().to_str().unwrap()));
            std::fs::write(property_test_file, property_tests)?;
            println!("✅ Property tests generated");
        }

        // Quality assessment
        if assess_quality {
            let source_code = std::fs::read_to_string(file_path)?;
            let unit_test_file = output_dir.join(format!("{}_tests.rs", 
                file_path.file_stem().unwrap().to_str().unwrap()));
            
            if unit_test_file.exists() {
                let test_code = std::fs::read_to_string(&unit_test_file)?;
                let quality_report = self.quality_assessor.assess_test_quality(&test_code, &source_code).await?;
                
                println!("\n📊 Test Quality Report:");
                println!("  Overall Score: {}/100", quality_report.overall_score);
                println!("  Coverage: {:.1}% lines, {:.1}% branches", 
                    quality_report.coverage_analysis.line_coverage,
                    quality_report.coverage_analysis.branch_coverage);
                println!("  Total Tests: {}", quality_report.quality_metrics.test_count);
                
                if quality_report.overall_score < 80 {
                    println!("  💡 Generating improvements...");
                    let improved_tests = self.quality_assessor.suggest_test_improvements(&quality_report, &test_code).await?;
                    let improved_file = output_dir.join(format!("{}_tests_improved.rs", 
                        file_path.file_stem().unwrap().to_str().unwrap()));
                    std::fs::write(improved_file, improved_tests)?;
                    println!("  ✅ Improved tests generated");
                }
            }
        }

        Ok(())
    }

    pub async fn generate_project_tests(
        &self,
        project_root: &Path,
        output_dir: &Path,
        include_benchmarks: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🏗️  Generating tests for entire project...");

        // Find all Rust source files
        let source_files = self.find_rust_files(project_root)?;
        
        std::fs::create_dir_all(output_dir)?;

        for file_path in source_files {
            if file_path.to_str().unwrap_or("").contains("/tests/") {
                continue; // Skip existing test files
            }

            println!("Processing {}", file_path.display());
            
            match self.generate_tests_for_file(&file_path, output_dir, &["unit", "integration"], false).await {
                Ok(_) => println!("  ✅ Tests generated"),
                Err(e) => println!("  ❌ Failed: {}", e),
            }
        }

        // Generate benchmarks if requested
        if include_benchmarks {
            println!("⚡ Generating benchmarks...");
            self.generate_project_benchmarks(project_root, output_dir).await?;
        }

        println!("🎉 Project test generation complete!");
        Ok(())
    }

    async fn generate_project_benchmarks(
        &self,
        project_root: &Path,
        output_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Find performance-critical files and generate benchmarks
        let source_files = self.find_rust_files(project_root)?;
        let mut all_benchmarks = String::new();

        all_benchmarks.push_str("use criterion::{criterion_group, criterion_main, Criterion};\n\n");

        for file_path in source_files.iter().take(5) { // Limit for demo
            let test_plan = self.analyzer.analyze_and_plan_tests(file_path).await?;
            let benchmarks = self.unit_generator.generate_benchmark_tests(&test_plan).await?;
            all_benchmarks.push_str(&benchmarks);
        }

        all_benchmarks.push_str("\ncriterion_group!(benches, benchmark_functions);\ncriterion_main!(benches);\n");

        let benchmark_file = output_dir.join("benchmarks.rs");
        std::fs::write(benchmark_file, all_benchmarks)?;

        Ok(())
    }

    fn find_rust_files(&self, dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut rust_files = Vec::new();
        
        for entry in WalkDir::new(dir) {
            let entry = entry?;
            if entry.file_type().is_file() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "rs" {
                        rust_files.push(entry.path().to_path_buf());
                    }
                }
            }
        }
        
        Ok(rust_files)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { file, output, types, assess_quality } => {
            let project_root = file.parent().unwrap_or(Path::new(".")).to_path_buf();
            let app = TestGeneratorApp::new(project_root)?;
            
            let test_types: Vec<&str> = types.split(',').collect();
            app.generate_tests_for_file(&file, &output, &test_types, assess_quality).await?;
        }
        
        Commands::Project { root, output, benchmarks } => {
            let app = TestGeneratorApp::new(root.clone())?;
            app.generate_project_tests(&root, &output, benchmarks).await?;
        }
        
        Commands::Assess { test_file, source_file, improve } => {
            let project_root = source_file.parent().unwrap_or(Path::new(".")).to_path_buf();
            let app = TestGeneratorApp::new(project_root)?;
            
            let test_code = std::fs::read_to_string(&test_file)?;
            let source_code = std::fs::read_to_string(&source_file)?;
            
            let quality_report = app.quality_assessor.assess_test_quality(&test_code, &source_code).await?;
            
            println!("📊 Test Quality Assessment:");
            println!("  Overall Score: {}/100", quality_report.overall_score);
            println!("  Test Count: {}", quality_report.quality_metrics.test_count);
            println!("  Coverage: {:.1}%", quality_report.coverage_analysis.line_coverage);
            
            for rec in &quality_report.recommendations {
                println!("  💡 {:?}: {}", rec.priority, rec.description);
            }
            
            if improve && quality_report.overall_score < 90 {
                let improved = app.quality_assessor.suggest_test_improvements(&quality_report, &test_code).await?;
                let improved_file = test_file.with_extension("improved.rs");
                std::fs::write(improved_file, improved)?;
                println!("✅ Improved tests written to {}", test_file.display());
            }
        }
        
        Commands::Watch { directory, output } => {
            println!("👀 Watching {} for changes...", directory.display());
            // Implementation for file watching would go here
            // Similar to the documentation generator's watch mode
        }
    }

    Ok(())
}
```

## Usage Examples

```bash
# Generate tests for a single file
test-generator generate -f src/lib.rs -o tests/ --assess-quality

# Generate tests for entire project
test-generator project -r . -o generated_tests/ --benchmarks

# Assess existing tests
test-generator assess tests/lib_tests.rs src/lib.rs --improve

# Watch for changes and regenerate
test-generator watch -d src/ -o tests/

# Generate only specific test types
test-generator generate -f src/main.rs -t "unit,property"
```

This automated test generator demonstrates the sophisticated capabilities possible with the claude-sdk-rs SDK, creating comprehensive test suites that improve code quality and reliability through AI-powered analysis and generation.