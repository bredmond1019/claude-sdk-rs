# Tutorial: AI-Powered Documentation Generator

This tutorial shows how to build an intelligent documentation generator that analyzes your codebase and creates comprehensive, up-to-date documentation using the claude-sdk-rs SDK.

## Table of Contents

1. [Overview](#overview)
2. [Project Setup](#project-setup)
3. [Code Analysis Engine](#code-analysis-engine)
4. [Documentation Generation](#documentation-generation)
5. [Multiple Output Formats](#multiple-output-formats)
6. [Watch Mode and Auto-Updates](#watch-mode-and-auto-updates)
7. [Integration with Build Systems](#integration-with-build-systems)

## Overview

Our documentation generator will:

- **Analyze code structure** - Parse Rust, Python, JavaScript, and other languages
- **Generate API documentation** - Create detailed API docs with examples
- **Write user guides** - Generate tutorials and getting started guides
- **Create diagrams** - Generate architecture and flow diagrams
- **Multiple formats** - Output to Markdown, HTML, PDF, and more
- **Live updates** - Watch for code changes and regenerate docs

## Project Setup

Create a new Rust project with the required dependencies:

```toml
[package]
name = "doc-generator"
version = "0.1.0"
edition = "2021"

[dependencies]
claude-sdk-rs = { version = "0.1", features = ["tools"] }
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.0", features = ["derive"] }
walkdir = "2.0"
notify = "6.0"
tera = "1.19"
pulldown-cmark = "0.9"
syntect = "5.0"
handlebars = "4.0"
regex = "1.0"
anyhow = "1.0"
syn = { version = "2.0", features = ["full", "parsing"] }
quote = "1.0"
```

## Code Analysis Engine

Let's start by building a comprehensive code analyzer:

```rust
use claude_sdk_rs::{Client, Config, ToolPermission, StreamFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectAnalysis {
    pub metadata: ProjectMetadata,
    pub modules: Vec<ModuleInfo>,
    pub api_endpoints: Vec<ApiEndpoint>,
    pub examples: Vec<CodeExample>,
    pub dependencies: Vec<Dependency>,
    pub architecture: ArchitectureInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub language: String,
    pub framework: Option<String>,
    pub license: Option<String>,
    pub repository_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub path: PathBuf,
    pub description: String,
    pub public_functions: Vec<FunctionInfo>,
    pub public_types: Vec<TypeInfo>,
    pub examples: Vec<String>,
    pub tests: Vec<TestInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub signature: String,
    pub description: String,
    pub parameters: Vec<ParameterInfo>,
    pub return_type: String,
    pub examples: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TypeInfo {
    pub name: String,
    pub kind: TypeKind,
    pub description: String,
    pub fields: Vec<FieldInfo>,
    pub methods: Vec<FunctionInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TypeKind {
    Struct,
    Enum,
    Trait,
    Type,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FieldInfo {
    pub name: String,
    pub type_name: String,
    pub description: String,
    pub optional: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub type_name: String,
    pub description: String,
    pub optional: bool,
    pub default_value: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiEndpoint {
    pub method: String,
    pub path: String,
    pub description: String,
    pub parameters: Vec<ParameterInfo>,
    pub response_type: String,
    pub examples: Vec<ApiExample>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiExample {
    pub request: String,
    pub response: String,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeExample {
    pub title: String,
    pub description: String,
    pub code: String,
    pub language: String,
    pub category: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestInfo {
    pub name: String,
    pub description: String,
    pub test_type: TestType,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TestType {
    Unit,
    Integration,
    Documentation,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub description: String,
    pub optional: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitectureInfo {
    pub overview: String,
    pub components: Vec<ComponentInfo>,
    pub data_flow: Vec<DataFlowStep>,
    pub design_patterns: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentInfo {
    pub name: String,
    pub description: String,
    pub responsibilities: Vec<String>,
    pub interfaces: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataFlowStep {
    pub from: String,
    pub to: String,
    pub description: String,
    pub data_type: String,
}

pub struct CodeAnalyzer {
    claude_client: Client,
    project_root: PathBuf,
}

impl CodeAnalyzer {
    pub fn new(project_root: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Json)
            .system_prompt(include_str!("../prompts/documentation_system.txt"))
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

    pub async fn analyze_project(&self) -> Result<ProjectAnalysis, Box<dyn std::error::Error>> {
        println!("🔍 Starting project analysis...");

        // Step 1: Analyze project metadata
        let metadata = self.analyze_project_metadata().await?;
        println!("✅ Project metadata analyzed");

        // Step 2: Discover and analyze modules
        let modules = self.analyze_modules().await?;
        println!("✅ {} modules analyzed", modules.len());

        // Step 3: Detect API endpoints (if applicable)
        let api_endpoints = self.analyze_api_endpoints().await?;
        println!("✅ {} API endpoints found", api_endpoints.len());

        // Step 4: Extract code examples
        let examples = self.extract_code_examples().await?;
        println!("✅ {} code examples extracted", examples.len());

        // Step 5: Analyze dependencies
        let dependencies = self.analyze_dependencies().await?;
        println!("✅ {} dependencies analyzed", dependencies.len());

        // Step 6: Generate architecture overview
        let architecture = self.analyze_architecture(&modules).await?;
        println!("✅ Architecture analysis completed");

        Ok(ProjectAnalysis {
            metadata,
            modules,
            api_endpoints,
            examples,
            dependencies,
            architecture,
        })
    }

    async fn analyze_project_metadata(&self) -> Result<ProjectMetadata, Box<dyn std::error::Error>> {
        // Look for common metadata files
        let metadata_files = [
            "Cargo.toml",
            "package.json", 
            "pyproject.toml",
            "setup.py",
            "README.md",
        ];

        let mut file_contents = HashMap::new();
        
        for file in &metadata_files {
            let file_path = self.project_root.join(file);
            if file_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&file_path) {
                    file_contents.insert(file.to_string(), content);
                }
            }
        }

        let prompt = format!(
            "Analyze these project files and extract metadata:\n\n{}",
            file_contents.iter()
                .map(|(name, content)| format!("=== {} ===\n{}\n", name, content))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        // Parse the response to extract metadata
        self.parse_metadata_response(&response.content)
    }

    async fn analyze_modules(&self) -> Result<Vec<ModuleInfo>, Box<dyn std::error::Error>> {
        let mut modules = Vec::new();

        // Find all source files
        let source_files = self.find_source_files()?;

        for file_path in source_files {
            if let Ok(module_info) = self.analyze_single_module(&file_path).await {
                modules.push(module_info);
            }
        }

        Ok(modules)
    }

    async fn analyze_single_module(&self, file_path: &Path) -> Result<ModuleInfo, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;
        let relative_path = file_path.strip_prefix(&self.project_root)
            .unwrap_or(file_path);

        let prompt = format!(
            "Analyze this {} module and extract detailed information:\n\n\
             File: {}\n\n\
             ```{}\n{}\n```\n\n\
             Please provide:\n\
             1. Module purpose and description\n\
             2. All public functions with signatures and descriptions\n\
             3. All public types (structs, enums, traits)\n\
             4. Usage examples if available\n\
             5. Tests found in the module\n\n\
             Format as JSON with this structure:\n\
             {{\n\
               \"name\": \"module_name\",\n\
               \"description\": \"Module description\",\n\
               \"public_functions\": [\n\
                 {{\n\
                   \"name\": \"function_name\",\n\
                   \"signature\": \"fn signature()\",\n\
                   \"description\": \"What it does\",\n\
                   \"parameters\": [],\n\
                   \"return_type\": \"ReturnType\",\n\
                   \"examples\": [],\n\
                   \"errors\": []\n\
                 }}\n\
               ],\n\
               \"public_types\": [],\n\
               \"examples\": [],\n\
               \"tests\": []\n\
             }}",
            self.detect_language(file_path),
            relative_path.display(),
            self.detect_language(file_path),
            content
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_module_response(&response.content, relative_path.to_path_buf())
    }

    async fn analyze_api_endpoints(&self) -> Result<Vec<ApiEndpoint>, Box<dyn std::error::Error>> {
        // Look for API route definitions
        let route_files = self.find_files_containing(&[
            "route", "endpoint", "api", "handler",
            "axum", "actix", "warp", "rocket"
        ])?;

        let mut endpoints = Vec::new();

        for file_path in route_files {
            if let Ok(file_endpoints) = self.extract_endpoints_from_file(&file_path).await {
                endpoints.extend(file_endpoints);
            }
        }

        Ok(endpoints)
    }

    async fn extract_endpoints_from_file(&self, file_path: &Path) -> Result<Vec<ApiEndpoint>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;

        let prompt = format!(
            "Analyze this file for API endpoints and routes:\n\n\
             File: {}\n\n\
             ```rust\n{}\n```\n\n\
             Extract all HTTP endpoints with:\n\
             1. HTTP method (GET, POST, PUT, DELETE, etc.)\n\
             2. Route path\n\
             3. Description of what it does\n\
             4. Parameters (query, path, body)\n\
             5. Response type\n\
             6. Usage examples if possible\n\n\
             Format as JSON array of endpoints.",
            file_path.display(),
            content
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_endpoints_response(&response.content)
    }

    async fn extract_code_examples(&self) -> Result<Vec<CodeExample>, Box<dyn std::error::Error>> {
        let mut examples = Vec::new();

        // Look in examples directory
        let examples_dir = self.project_root.join("examples");
        if examples_dir.exists() {
            for entry in WalkDir::new(examples_dir) {
                if let Ok(entry) = entry {
                    if entry.file_type().is_file() && self.is_source_file(entry.path()) {
                        if let Ok(example) = self.analyze_example_file(entry.path()).await {
                            examples.push(example);
                        }
                    }
                }
            }
        }

        // Look for examples in test files
        let test_examples = self.extract_examples_from_tests().await?;
        examples.extend(test_examples);

        // Look for examples in documentation
        let doc_examples = self.extract_examples_from_docs().await?;
        examples.extend(doc_examples);

        Ok(examples)
    }

    async fn analyze_example_file(&self, file_path: &Path) -> Result<CodeExample, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;
        let relative_path = file_path.strip_prefix(&self.project_root)
            .unwrap_or(file_path);

        let prompt = format!(
            "Analyze this example file and create documentation:\n\n\
             File: {}\n\n\
             ```{}\n{}\n```\n\n\
             Provide:\n\
             1. A clear title for this example\n\
             2. Description of what it demonstrates\n\
             3. Category (basic, advanced, integration, etc.)\n\n\
             Format as JSON.",
            relative_path.display(),
            self.detect_language(file_path),
            content
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_example_response(&response.content, content, file_path)
    }

    async fn analyze_dependencies(&self) -> Result<Vec<Dependency>, Box<dyn std::error::Error>> {
        let mut dependencies = Vec::new();

        // Analyze Cargo.toml for Rust projects
        let cargo_toml = self.project_root.join("Cargo.toml");
        if cargo_toml.exists() {
            dependencies.extend(self.analyze_cargo_dependencies().await?);
        }

        // Analyze package.json for Node.js projects
        let package_json = self.project_root.join("package.json");
        if package_json.exists() {
            dependencies.extend(self.analyze_npm_dependencies().await?);
        }

        // Analyze requirements.txt for Python projects
        let requirements_txt = self.project_root.join("requirements.txt");
        if requirements_txt.exists() {
            dependencies.extend(self.analyze_python_dependencies().await?);
        }

        Ok(dependencies)
    }

    async fn analyze_cargo_dependencies(&self) -> Result<Vec<Dependency>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(self.project_root.join("Cargo.toml"))?;

        let prompt = format!(
            "Analyze this Cargo.toml and extract dependency information:\n\n\
             ```toml\n{}\n```\n\n\
             For each dependency, provide:\n\
             1. Name\n\
             2. Version\n\
             3. Brief description of what it's used for\n\
             4. Whether it's optional\n\n\
             Format as JSON array.",
            content
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_dependencies_response(&response.content)
    }

    async fn analyze_architecture(&self, modules: &[ModuleInfo]) -> Result<ArchitectureInfo, Box<dyn std::error::Error>> {
        let module_summary = modules.iter()
            .map(|m| format!("- {}: {}", m.name, m.description))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Analyze the overall architecture of this project based on its modules:\n\n\
             Modules:\n{}\n\n\
             Provide:\n\
             1. High-level architecture overview\n\
             2. Main components and their responsibilities\n\
             3. Data flow between components\n\
             4. Design patterns used\n\n\
             Consider the module structure and dependencies to understand how the system is organized.",
            module_summary
        );

        let response = self.claude_client
            .query(&prompt)
            .send_full()
            .await?;

        self.parse_architecture_response(&response.content)
    }

    // Helper methods for parsing responses
    fn parse_metadata_response(&self, response: &str) -> Result<ProjectMetadata, Box<dyn std::error::Error>> {
        // Extract JSON from response
        let json_start = response.find('{').ok_or("No JSON found")?;
        let json_end = response.rfind('}').ok_or("No JSON found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let metadata: ProjectMetadata = serde_json::from_str(json_str)?;
        Ok(metadata)
    }

    fn parse_module_response(&self, response: &str, path: PathBuf) -> Result<ModuleInfo, Box<dyn std::error::Error>> {
        let json_start = response.find('{').ok_or("No JSON found")?;
        let json_end = response.rfind('}').ok_or("No JSON found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let mut module: ModuleInfo = serde_json::from_str(json_str)?;
        module.path = path;
        Ok(module)
    }

    fn parse_endpoints_response(&self, response: &str) -> Result<Vec<ApiEndpoint>, Box<dyn std::error::Error>> {
        let json_start = response.find('[').ok_or("No JSON array found")?;
        let json_end = response.rfind(']').ok_or("No JSON array found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let endpoints: Vec<ApiEndpoint> = serde_json::from_str(json_str)?;
        Ok(endpoints)
    }

    fn parse_example_response(&self, response: &str, code: String, file_path: &Path) -> Result<CodeExample, Box<dyn std::error::Error>> {
        let json_start = response.find('{').ok_or("No JSON found")?;
        let json_end = response.rfind('}').ok_or("No JSON found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let mut example: CodeExample = serde_json::from_str(json_str)?;
        example.code = code;
        example.language = self.detect_language(file_path);
        Ok(example)
    }

    fn parse_dependencies_response(&self, response: &str) -> Result<Vec<Dependency>, Box<dyn std::error::Error>> {
        let json_start = response.find('[').ok_or("No JSON array found")?;
        let json_end = response.rfind(']').ok_or("No JSON array found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let dependencies: Vec<Dependency> = serde_json::from_str(json_str)?;
        Ok(dependencies)
    }

    fn parse_architecture_response(&self, response: &str) -> Result<ArchitectureInfo, Box<dyn std::error::Error>> {
        let json_start = response.find('{').ok_or("No JSON found")?;
        let json_end = response.rfind('}').ok_or("No JSON found")? + 1;
        let json_str = &response[json_start..json_end];
        
        let architecture: ArchitectureInfo = serde_json::from_str(json_str)?;
        Ok(architecture)
    }

    // Utility methods
    fn find_source_files(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut files = Vec::new();
        
        for entry in WalkDir::new(&self.project_root) {
            if let Ok(entry) = entry {
                if entry.file_type().is_file() && self.is_source_file(entry.path()) {
                    files.push(entry.path().to_path_buf());
                }
            }
        }
        
        Ok(files)
    }

    fn is_source_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            matches!(extension.to_str(), Some("rs" | "py" | "js" | "ts" | "java" | "cpp" | "c" | "h"))
        } else {
            false
        }
    }

    fn detect_language(&self, path: &Path) -> String {
        if let Some(extension) = path.extension() {
            match extension.to_str() {
                Some("rs") => "rust".to_string(),
                Some("py") => "python".to_string(),
                Some("js") => "javascript".to_string(),
                Some("ts") => "typescript".to_string(),
                Some("java") => "java".to_string(),
                Some("cpp" | "cc" | "cxx") => "cpp".to_string(),
                Some("c") => "c".to_string(),
                _ => "text".to_string(),
            }
        } else {
            "text".to_string()
        }
    }

    fn find_files_containing(&self, patterns: &[&str]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut matching_files = Vec::new();
        
        for entry in WalkDir::new(&self.project_root) {
            if let Ok(entry) = entry {
                if entry.file_type().is_file() && self.is_source_file(entry.path()) {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        let content_lower = content.to_lowercase();
                        if patterns.iter().any(|pattern| content_lower.contains(&pattern.to_lowercase())) {
                            matching_files.push(entry.path().to_path_buf());
                        }
                    }
                }
            }
        }
        
        Ok(matching_files)
    }

    async fn extract_examples_from_tests(&self) -> Result<Vec<CodeExample>, Box<dyn std::error::Error>> {
        // Implementation for extracting examples from test files
        Ok(Vec::new()) // Placeholder
    }

    async fn extract_examples_from_docs(&self) -> Result<Vec<CodeExample>, Box<dyn std::error::Error>> {
        // Implementation for extracting examples from documentation
        Ok(Vec::new()) // Placeholder
    }

    async fn analyze_npm_dependencies(&self) -> Result<Vec<Dependency>, Box<dyn std::error::Error>> {
        // Implementation for Node.js dependencies
        Ok(Vec::new()) // Placeholder
    }

    async fn analyze_python_dependencies(&self) -> Result<Vec<Dependency>, Box<dyn std::error::Error>> {
        // Implementation for Python dependencies
        Ok(Vec::new()) // Placeholder
    }
}
```

## Documentation Generation

Now let's create the documentation generator that takes the analysis and produces beautiful docs:

```rust
use handlebars::Handlebars;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub struct DocumentationGenerator {
    claude_client: Client,
    template_engine: Handlebars<'static>,
    output_dir: PathBuf,
}

impl DocumentationGenerator {
    pub fn new(output_dir: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .model("claude-3-opus-20240229")
            .stream_format(StreamFormat::Text)
            .system_prompt("You are a technical writer creating clear, comprehensive documentation.")
            .timeout_secs(120)
            .build();

        let claude_client = Client::new(config);
        let mut template_engine = Handlebars::new();
        
        // Register built-in templates
        Self::register_templates(&mut template_engine)?;

        Ok(Self {
            claude_client,
            template_engine,
            output_dir,
        })
    }

    pub async fn generate_documentation(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("📝 Generating documentation...");

        // Create output directory structure
        self.create_output_structure()?;

        // Generate different types of documentation
        self.generate_readme(analysis).await?;
        self.generate_api_docs(analysis).await?;
        self.generate_user_guide(analysis).await?;
        self.generate_examples_docs(analysis).await?;
        self.generate_architecture_docs(analysis).await?;
        self.generate_changelog(analysis).await?;

        // Generate HTML version
        self.generate_html_docs(analysis).await?;

        println!("✅ Documentation generation complete!");
        Ok(())
    }

    async fn generate_readme(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("  📄 Generating README.md...");

        let prompt = format!(
            "Create a comprehensive README.md for this project:\n\n\
             Project: {}\n\
             Description: {}\n\
             Language: {}\n\
             Version: {}\n\n\
             The README should include:\n\
             1. Project title and description\n\
             2. Installation instructions\n\
             3. Quick start guide\n\
             4. Basic usage examples\n\
             5. API overview\n\
             6. Contributing guidelines\n\
             7. License information\n\n\
             Make it engaging and easy to follow for new users.",
            analysis.metadata.name,
            analysis.metadata.description,
            analysis.metadata.language,
            analysis.metadata.version
        );

        let readme_content = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        let readme_path = self.output_dir.join("README.md");
        fs::write(readme_path, readme_content)?;

        Ok(())
    }

    async fn generate_api_docs(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("  📚 Generating API documentation...");

        let api_dir = self.output_dir.join("docs").join("api");
        fs::create_dir_all(&api_dir)?;

        // Generate overview
        let api_overview = self.generate_api_overview(analysis).await?;
        fs::write(api_dir.join("README.md"), api_overview)?;

        // Generate documentation for each module
        for module in &analysis.modules {
            let module_docs = self.generate_module_docs(module).await?;
            let module_file = api_dir.join(format!("{}.md", module.name));
            fs::write(module_file, module_docs)?;
        }

        // Generate endpoint documentation if applicable
        if !analysis.api_endpoints.is_empty() {
            let endpoints_docs = self.generate_endpoints_docs(&analysis.api_endpoints).await?;
            fs::write(api_dir.join("endpoints.md"), endpoints_docs)?;
        }

        Ok(())
    }

    async fn generate_api_overview(&self, analysis: &ProjectAnalysis) -> Result<String, Box<dyn std::error::Error>> {
        let modules_list = analysis.modules.iter()
            .map(|m| format!("- **{}**: {}", m.name, m.description))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Create an API documentation overview for this project:\n\n\
             Project: {}\n\
             Description: {}\n\n\
             Modules:\n{}\n\n\
             Create a comprehensive overview that explains:\n\
             1. What the API does\n\
             2. How to get started\n\
             3. Main concepts and terminology\n\
             4. Module organization\n\
             5. Common patterns and examples",
            analysis.metadata.name,
            analysis.metadata.description,
            modules_list
        );

        let overview = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(overview)
    }

    async fn generate_module_docs(&self, module: &ModuleInfo) -> Result<String, Box<dyn std::error::Error>> {
        let functions_list = module.public_functions.iter()
            .map(|f| format!("- `{}`: {}", f.signature, f.description))
            .collect::<Vec<_>>()
            .join("\n");

        let types_list = module.public_types.iter()
            .map(|t| format!("- `{}` ({:?}): {}", t.name, t.kind, t.description))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Create detailed documentation for this module:\n\n\
             Module: {}\n\
             Description: {}\n\n\
             Public Functions:\n{}\n\n\
             Public Types:\n{}\n\n\
             Create comprehensive documentation including:\n\
             1. Module overview and purpose\n\
             2. Detailed function documentation with parameters and examples\n\
             3. Type documentation with field descriptions\n\
             4. Usage patterns and best practices\n\
             5. Error handling information",
            module.name,
            module.description,
            functions_list,
            types_list
        );

        let docs = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(docs)
    }

    async fn generate_user_guide(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("  📖 Generating user guide...");

        let guides_dir = self.output_dir.join("docs").join("guides");
        fs::create_dir_all(&guides_dir)?;

        // Generate getting started guide
        let getting_started = self.generate_getting_started_guide(analysis).await?;
        fs::write(guides_dir.join("getting-started.md"), getting_started)?;

        // Generate tutorials for different use cases
        let tutorials = self.generate_tutorials(analysis).await?;
        for (name, content) in tutorials {
            fs::write(guides_dir.join(format!("{}.md", name)), content)?;
        }

        // Generate FAQ
        let faq = self.generate_faq(analysis).await?;
        fs::write(guides_dir.join("faq.md"), faq)?;

        Ok(())
    }

    async fn generate_getting_started_guide(&self, analysis: &ProjectAnalysis) -> Result<String, Box<dyn std::error::Error>> {
        let examples_summary = analysis.examples.iter()
            .take(3) // Use first 3 examples
            .map(|e| format!("Example: {} - {}", e.title, e.description))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Create a comprehensive getting started guide for this project:\n\n\
             Project: {}\n\
             Description: {}\n\
             Language: {}\n\n\
             Available Examples:\n{}\n\n\
             The guide should include:\n\
             1. Prerequisites and system requirements\n\
             2. Installation steps\n\
             3. Basic configuration\n\
             4. Your first program/example\n\
             5. Common next steps\n\
             6. Where to find more help\n\n\
             Make it beginner-friendly with clear step-by-step instructions.",
            analysis.metadata.name,
            analysis.metadata.description,
            analysis.metadata.language,
            examples_summary
        );

        let guide = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        Ok(guide)
    }

    async fn generate_examples_docs(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("  💡 Generating examples documentation...");

        let examples_dir = self.output_dir.join("docs").join("examples");
        fs::create_dir_all(&examples_dir)?;

        // Generate examples overview
        let examples_overview = self.generate_examples_overview(&analysis.examples).await?;
        fs::write(examples_dir.join("README.md"), examples_overview)?;

        // Group examples by category
        let mut categories: HashMap<String, Vec<&CodeExample>> = HashMap::new();
        for example in &analysis.examples {
            categories.entry(example.category.clone())
                .or_insert_with(Vec::new)
                .push(example);
        }

        // Generate documentation for each category
        for (category, examples) in categories {
            let category_docs = self.generate_category_examples_docs(&category, &examples).await?;
            let category_file = examples_dir.join(format!("{}.md", category.replace(" ", "-").to_lowercase()));
            fs::write(category_file, category_docs)?;
        }

        Ok(())
    }

    async fn generate_architecture_docs(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("  🏗️  Generating architecture documentation...");

        let arch_dir = self.output_dir.join("docs").join("architecture");
        fs::create_dir_all(&arch_dir)?;

        let arch_docs = format!(
            "# Architecture Overview\n\n\
             {}\n\n\
             ## Components\n\n\
             {}\n\n\
             ## Data Flow\n\n\
             {}\n\n\
             ## Design Patterns\n\n\
             {}",
            analysis.architecture.overview,
            analysis.architecture.components.iter()
                .map(|c| format!("### {}\n\n{}\n\n**Responsibilities:**\n{}\n",
                    c.name,
                    c.description,
                    c.responsibilities.iter()
                        .map(|r| format!("- {}", r))
                        .collect::<Vec<_>>()
                        .join("\n")
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            analysis.architecture.data_flow.iter()
                .map(|step| format!("- **{} → {}**: {} ({})", step.from, step.to, step.description, step.data_type))
                .collect::<Vec<_>>()
                .join("\n"),
            analysis.architecture.design_patterns.iter()
                .map(|pattern| format!("- {}", pattern))
                .collect::<Vec<_>>()
                .join("\n")
        );

        fs::write(arch_dir.join("overview.md"), arch_docs)?;

        Ok(())
    }

    async fn generate_changelog(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("  📅 Generating changelog...");

        let prompt = format!(
            "Create a changelog template for this project:\n\n\
             Project: {}\n\
             Current Version: {}\n\n\
             Create a CHANGELOG.md with:\n\
             1. Proper changelog format\n\
             2. Current version entry\n\
             3. Guidelines for future entries\n\
             4. Semantic versioning explanation",
            analysis.metadata.name,
            analysis.metadata.version
        );

        let changelog = self.claude_client
            .query(&prompt)
            .send()
            .await?;

        fs::write(self.output_dir.join("CHANGELOG.md"), changelog)?;

        Ok(())
    }

    async fn generate_html_docs(&self, analysis: &ProjectAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        println!("  🌐 Generating HTML documentation...");

        let html_dir = self.output_dir.join("html");
        fs::create_dir_all(&html_dir)?;

        // Copy CSS and JavaScript assets
        self.copy_assets(&html_dir)?;

        // Generate HTML pages
        self.generate_html_index(analysis, &html_dir).await?;
        self.generate_html_api_docs(analysis, &html_dir).await?;

        Ok(())
    }

    // Helper methods for HTML generation and templates
    fn register_templates(handlebars: &mut Handlebars) -> Result<(), Box<dyn std::error::Error>> {
        // Register HTML templates for different page types
        handlebars.register_template_string("index", include_str!("../templates/index.hbs"))?;
        handlebars.register_template_string("api", include_str!("../templates/api.hbs"))?;
        handlebars.register_template_string("module", include_str!("../templates/module.hbs"))?;
        Ok(())
    }

    fn create_output_structure(&self) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(&self.output_dir)?;
        fs::create_dir_all(self.output_dir.join("docs"))?;
        fs::create_dir_all(self.output_dir.join("docs").join("api"))?;
        fs::create_dir_all(self.output_dir.join("docs").join("guides"))?;
        fs::create_dir_all(self.output_dir.join("docs").join("examples"))?;
        fs::create_dir_all(self.output_dir.join("docs").join("architecture"))?;
        Ok(())
    }

    // Placeholder implementations for other methods
    async fn generate_endpoints_docs(&self, endpoints: &[ApiEndpoint]) -> Result<String, Box<dyn std::error::Error>> {
        Ok("API Endpoints documentation".to_string()) // Placeholder
    }

    async fn generate_tutorials(&self, analysis: &ProjectAnalysis) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        Ok(HashMap::new()) // Placeholder
    }

    async fn generate_faq(&self, analysis: &ProjectAnalysis) -> Result<String, Box<dyn std::error::Error>> {
        Ok("FAQ content".to_string()) // Placeholder
    }

    async fn generate_examples_overview(&self, examples: &[CodeExample]) -> Result<String, Box<dyn std::error::Error>> {
        Ok("Examples overview".to_string()) // Placeholder
    }

    async fn generate_category_examples_docs(&self, category: &str, examples: &[&CodeExample]) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("Category: {}", category)) // Placeholder
    }

    async fn generate_html_index(&self, analysis: &ProjectAnalysis, html_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // Placeholder
    }

    async fn generate_html_api_docs(&self, analysis: &ProjectAnalysis, html_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // Placeholder
    }

    fn copy_assets(&self, html_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // Placeholder
    }
}
```

## Watch Mode and Auto-Updates

Add a file watcher to automatically regenerate docs when code changes:

```rust
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::mpsc;
use std::time::Duration;

pub struct DocumentationWatcher {
    analyzer: CodeAnalyzer,
    generator: DocumentationGenerator,
    project_root: PathBuf,
}

impl DocumentationWatcher {
    pub fn new(project_root: PathBuf, output_dir: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let analyzer = CodeAnalyzer::new(project_root.clone())?;
        let generator = DocumentationGenerator::new(output_dir)?;

        Ok(Self {
            analyzer,
            generator,
            project_root,
        })
    }

    pub async fn start_watching(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("👀 Starting documentation watch mode...");

        let (tx, rx) = mpsc::channel();

        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    if let EventKind::Modify(_) = event.kind {
                        // Check if it's a source file
                        for path in event.paths {
                            if self.is_source_file(&path) {
                                let _ = tx.send(path);
                                break;
                            }
                        }
                    }
                }
            },
            Config::default(),
        )?;

        watcher.watch(&self.project_root, RecursiveMode::Recursive)?;

        println!("✅ Watching for changes in {}", self.project_root.display());

        // Initial generation
        self.regenerate_docs().await?;

        // Watch for changes
        loop {
            match rx.recv_timeout(Duration::from_secs(5)) {
                Ok(changed_file) => {
                    println!("📝 File changed: {}, regenerating docs...", changed_file.display());
                    
                    // Debounce: wait a bit more to catch rapid changes
                    std::thread::sleep(Duration::from_millis(500));
                    
                    if let Err(e) = self.regenerate_docs().await {
                        eprintln!("❌ Failed to regenerate docs: {}", e);
                    } else {
                        println!("✅ Documentation updated");
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // No changes, continue watching
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }

        Ok(())
    }

    async fn regenerate_docs(&self) -> Result<(), Box<dyn std::error::Error>> {
        let analysis = self.analyzer.analyze_project().await?;
        self.generator.generate_documentation(&analysis).await?;
        Ok(())
    }

    fn is_source_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            matches!(extension.to_str(), Some("rs" | "py" | "js" | "ts" | "java" | "cpp" | "c" | "h" | "md" | "toml" | "json"))
        } else {
            false
        }
    }
}
```

## CLI Application

Finally, let's create a command-line interface:

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "doc-generator")]
#[command(about = "AI-powered documentation generator")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate documentation for a project
    Generate {
        /// Project root directory
        #[arg(short, long, default_value = ".")]
        project: PathBuf,
        
        /// Output directory
        #[arg(short, long, default_value = "./docs")]
        output: PathBuf,
        
        /// Output format
        #[arg(short, long, default_value = "markdown")]
        format: String,
    },
    
    /// Watch for changes and auto-regenerate
    Watch {
        /// Project root directory
        #[arg(short, long, default_value = ".")]
        project: PathBuf,
        
        /// Output directory
        #[arg(short, long, default_value = "./docs")]
        output: PathBuf,
    },
    
    /// Analyze project structure only
    Analyze {
        /// Project root directory
        #[arg(short, long, default_value = ".")]
        project: PathBuf,
        
        /// Output analysis to JSON file
        #[arg(short, long)]
        json: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { project, output, format } => {
            println!("🚀 Generating documentation for {}", project.display());
            
            let analyzer = CodeAnalyzer::new(project)?;
            let analysis = analyzer.analyze_project().await?;
            
            let generator = DocumentationGenerator::new(output)?;
            generator.generate_documentation(&analysis).await?;
            
            println!("✅ Documentation generated successfully!");
        }
        
        Commands::Watch { project, output } => {
            let watcher = DocumentationWatcher::new(project, output)?;
            watcher.start_watching().await?;
        }
        
        Commands::Analyze { project, json } => {
            println!("🔍 Analyzing project structure...");
            
            let analyzer = CodeAnalyzer::new(project)?;
            let analysis = analyzer.analyze_project().await?;
            
            if let Some(json_path) = json {
                let json_content = serde_json::to_string_pretty(&analysis)?;
                std::fs::write(json_path, json_content)?;
                println!("✅ Analysis saved to JSON");
            } else {
                println!("📊 Analysis Results:");
                println!("  - Modules: {}", analysis.modules.len());
                println!("  - API Endpoints: {}", analysis.api_endpoints.len());
                println!("  - Examples: {}", analysis.examples.len());
                println!("  - Dependencies: {}", analysis.dependencies.len());
            }
        }
    }

    Ok(())
}
```

## Usage Examples

```bash
# Generate documentation for current project
doc-generator generate

# Generate docs for specific project
doc-generator generate -p /path/to/project -o /path/to/output

# Watch for changes and auto-update
doc-generator watch -p /path/to/project

# Analyze project structure
doc-generator analyze -p /path/to/project --json analysis.json

# Generate HTML documentation
doc-generator generate --format html
```

This documentation generator showcases the power of combining the claude-sdk-rs SDK with traditional code analysis tools to create intelligent, comprehensive documentation that stays up-to-date with your codebase.