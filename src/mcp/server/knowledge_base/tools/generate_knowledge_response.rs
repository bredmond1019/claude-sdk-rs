/// Generate Knowledge Response Node - Creates synthesized responses from search results
///
/// This node takes the analyzed search results from all knowledge sources and generates
/// a comprehensive, well-formatted response that includes relevant information and source links.
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::core::{error::WorkflowError, nodes::Node, task::TaskContext};

/// Generates comprehensive responses from knowledge base search results
///
/// Creates formatted responses that include:
/// - Summary of findings from all sources
/// - Organized sections for different source types
/// - Direct links to relevant documentation and conversations
/// - Fallback responses when insufficient information is found
#[derive(Debug, Clone)]
pub struct GenerateKnowledgeResponseNode;
#[async_trait]
impl Node for GenerateKnowledgeResponseNode {
    async fn execute(&self, input: Value, _context: &TaskContext) -> Result<Value, WorkflowError> {
        let sufficient_info = input
            .get("sufficient_information")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !sufficient_info {
            let response = "I apologize, but I couldn't find enough relevant information to provide a comprehensive answer to your question. You might want to try rephrasing your question or contacting support directly.".to_string();

            return Ok(serde_json::json!({
                "generated_response": response,
                "response_type": "insufficient_info",
                "response_generated": true,
                "event_data": input.get("event_data").cloned().unwrap_or(input.clone())
            }));
        }

        // Collect all search results
        let notion_results = input.get("notion_search_results");
        let helpscout_results = input.get("helpscout_search_results");
        let slack_results = input.get("slack_search_results");

        let mut response_parts = Vec::new();

        // Add introduction
        response_parts.push(
            "Based on my search across our knowledge base, here's what I found:\n".to_string(),
        );

        // Process Notion results
        if let Some(notion) = notion_results {
            if let Some(pages) = notion.get("pages").and_then(|v| v.as_array()) {
                if !pages.is_empty() {
                    response_parts.push("\n**Documentation & Pages:**".to_string());
                    for page in pages.iter().take(3) {
                        if let (Some(title), Some(url)) = (
                            page.get("title").and_then(|v| v.as_str()),
                            page.get("url").and_then(|v| v.as_str()),
                        ) {
                            response_parts.push(format!("- [{}]({})", title, url));
                        }
                    }
                }
            }
        }

        // Process HelpScout results
        if let Some(helpscout) = helpscout_results {
            let mut helpscout_items = Vec::new();

            if let Some(articles) = helpscout.get("articles").and_then(|v| v.as_array()) {
                for article in articles.iter().take(2) {
                    if let (Some(title), Some(url)) = (
                        article.get("title").and_then(|v| v.as_str()),
                        article.get("url").and_then(|v| v.as_str()),
                    ) {
                        helpscout_items.push(format!("- [{}]({})", title, url));
                    }
                }
            }

            if let Some(conversations) = helpscout.get("conversations").and_then(|v| v.as_array()) {
                for conv in conversations.iter().take(2) {
                    if let (Some(subject), Some(url)) = (
                        conv.get("subject").and_then(|v| v.as_str()),
                        conv.get("url").and_then(|v| v.as_str()),
                    ) {
                        helpscout_items.push(format!("- [{}]({})", subject, url));
                    }
                }
            }

            if !helpscout_items.is_empty() {
                response_parts.push("\n**Support Articles & Conversations:**".to_string());
                response_parts.extend(helpscout_items);
            }
        }

        // Process Slack results
        if let Some(slack) = slack_results {
            if let Some(messages) = slack.get("messages").and_then(|v| v.as_array()) {
                if !messages.is_empty() {
                    response_parts.push("\n**Recent Team Discussions:**".to_string());
                    for message in messages.iter().take(3) {
                        if let (Some(channel), Some(user), Some(text)) = (
                            message.get("channel").and_then(|v| v.as_str()),
                            message.get("user").and_then(|v| v.as_str()),
                            message.get("text").and_then(|v| v.as_str()),
                        ) {
                            let preview = if text.len() > 100 {
                                format!("{}...", &text[..100])
                            } else {
                                text.to_string()
                            };
                            response_parts
                                .push(format!("- {} in {}: \"{}\"", user, channel, preview));
                        }
                    }
                }
            }
        }

        // Add closing note
        response_parts.push("\n\nIf you need more specific information or if this doesn't fully answer your question, please let me know!".to_string());

        let generated_response = response_parts.join("\n");

        Ok(serde_json::json!({
            "generated_response": generated_response,
            "response_type": "knowledge_synthesis",
            "response_generated": true,
            "event_data": input.get("event_data").cloned().unwrap_or(input.clone())
        }))
    }

    fn name(&self) -> &str {
        "GenerateKnowledgeResponseNode"
    }
}
