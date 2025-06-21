use serde::{Deserialize, Serialize};

/// Knowledge base event data structure for query processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseEventData {
    pub query_id: String,
    pub user_id: String,
    pub user_query: String,
    pub query_type: String,
    pub sources: Vec<String>,
}

pub mod tools;

pub use tools::*;
