/// Analyze Knowledge Node - Determines if sufficient information was found
///
/// This node analyzes the search results from all knowledge sources (Notion, HelpScout, Slack)
/// to determine if enough relevant information was found to provide a comprehensive answer.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Analyzes search results to determine if sufficient information was found
///
/// Evaluates search results based on:
/// - Total number of results found across all sources
/// - Number of high-relevance results (relevance score >= 80)
/// - Coverage across different knowledge sources
#[derive(Debug, Clone)]
pub struct AnalyzeKnowledgeNode;
#[async_trait]
impl Node for AnalyzeKnowledgeNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        // Collect results from all search nodes
        let notion_results = input.get("notion_search_results");
        let helpscout_results = input.get("helpscout_search_results");
        let slack_results = input.get("slack_search_results");

        let mut total_results = 0;
        let mut high_relevance_count = 0;
        let mut all_sources = Vec::new();

        // Analyze Notion results
        if let Some(notion) = notion_results {
            if let Some(count) = notion.get("results_found").and_then(|v| v.as_u64()) {
                total_results += count;
            }
            if let Some(pages) = notion.get("pages").and_then(|v| v.as_array()) {
                for page in pages {
                    if let Some(relevance) = page.get("relevance").and_then(|v| v.as_u64()) {
                        if relevance >= 80 {
                            high_relevance_count += 1;
                        }
                    }
                }
            }
            all_sources.push("notion");
        }

        // Analyze HelpScout results
        if let Some(helpscout) = helpscout_results {
            if let Some(count) = helpscout.get("results_found").and_then(|v| v.as_u64()) {
                total_results += count;
            }
            // Check both articles and conversations for high relevance
            for result_type in ["articles", "conversations"] {
                if let Some(items) = helpscout.get(result_type).and_then(|v| v.as_array()) {
                    for item in items {
                        if let Some(relevance) = item.get("relevance").and_then(|v| v.as_u64()) {
                            if relevance >= 80 {
                                high_relevance_count += 1;
                            }
                        }
                    }
                }
            }
            all_sources.push("helpscout");
        }

        // Analyze Slack results
        if let Some(slack) = slack_results {
            if let Some(count) = slack.get("results_found").and_then(|v| v.as_u64()) {
                total_results += count;
            }
            if let Some(messages) = slack.get("messages").and_then(|v| v.as_array()) {
                for message in messages {
                    if let Some(relevance) = message.get("relevance").and_then(|v| v.as_u64()) {
                        if relevance >= 80 {
                            high_relevance_count += 1;
                        }
                    }
                }
            }
            all_sources.push("slack");
        }

        // Determine if we have enough information (at least 2 results with 1+ high relevance)
        let sufficient_info = total_results >= 2 && high_relevance_count >= 1;

        let analysis_message = if !sufficient_info {
            "Insufficient information found to provide a comprehensive answer".to_string()
        } else {
            format!(
                "Found {} results with {} high-relevance matches",
                total_results, high_relevance_count
            )
        };

        Ok(serde_json::json!({
            "total_results_found": total_results,
            "high_relevance_count": high_relevance_count,
            "sufficient_information": sufficient_info,
            "sources_searched": all_sources,
            "analysis_completed": true,
            "analysis_message": analysis_message,
            "event_data": input.get("event_data").cloned().unwrap_or(input.clone())
        }))
    }

    fn name(&self) -> &str {
        "AnalyzeKnowledgeNode"
    }
}
