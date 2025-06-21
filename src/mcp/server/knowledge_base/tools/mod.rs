mod analyze_knowledge;
mod filter_spam_query;
mod generate_knowledge_response;
mod helpscout_search;
mod helpscout_service;
mod notion_search;
mod notion_service;
/// Knowledge base tools module - contains all workflow nodes for knowledge base operations
///
/// This module provides a complete set of nodes for processing knowledge base queries:
/// - Query routing and validation
/// - Search across multiple sources (Notion, HelpScout, Slack)
/// - Analysis and response generation
/// - Final response delivery
mod query_router;
mod search_router;
mod send_knowledge_reply;
mod slack_search;
mod slack_service;
mod validate_query;
pub use analyze_knowledge::AnalyzeKnowledgeNode;
pub use filter_spam_query::FilterSpamQueryNode;
pub use generate_knowledge_response::GenerateKnowledgeResponseNode;
pub use helpscout_search::HelpscoutSearchNode;
pub use helpscout_service::HelpscoutServiceNode;
pub use notion_search::NotionSearchNode;
pub use notion_service::NotionServiceNode;
pub use query_router::QueryRouterNode;
pub use search_router::SearchRouterNode;
pub use send_knowledge_reply::SendKnowledgeReplyNode;
pub use slack_search::SlackSearchNode;
pub use slack_service::SlackServiceNode;
pub use validate_query::ValidateQueryNode;

/// Helper function to extract keywords from a query string
///
/// Removes common stop words and short words, returning meaningful keywords
/// for better search results across knowledge sources.
pub fn extract_keywords(query: &str) -> Vec<String> {
    // Simple keyword extraction - split on whitespace and remove common words
    let stop_words = [
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for",
        "of", "as", "by",
    ];
    query
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|word| !word.is_empty() && word.len() > 2 && !stop_words.contains(&word.as_str()))
        .collect()
}
