use crate::mcp::core::error::WorkflowError;
use crate::mcp::server::MCPToolServer;

use super::tools::register_customer_support_tools;

/// Example MCP server that exposes customer support tools
pub struct CustomerSupportMCPServer {
    server: MCPToolServer,
}

impl CustomerSupportMCPServer {
    pub async fn new() -> Result<Self, WorkflowError> {
        let server = MCPToolServer::new("customer-support-server".to_string(), "1.0.0".to_string());

        let mut mcp_server = Self { server };
        register_customer_support_tools(&mut mcp_server).await?;
        Ok(mcp_server)
    }

    pub async fn get_tool_count(&self) -> usize {
        self.server.get_tool_count().await
    }

    pub async fn get_tool_names(&self) -> Vec<String> {
        self.server.get_tool_names().await
    }

    pub fn get_server(&self) -> &MCPToolServer {
        &self.server
    }
}
