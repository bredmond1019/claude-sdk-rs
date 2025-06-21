#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_claude_cli_response_with_optional_costs() {
        // Test 1: JSON with cost fields present
        let json_with_cost = r#"{
            "type": "assistant_response",
            "subtype": "completion",
            "cost_usd": 0.001234,
            "is_error": false,
            "duration_ms": 1500,
            "duration_api_ms": 1200,
            "num_turns": 1,
            "result": "Hello, world!",
            "total_cost": 0.001234,
            "session_id": "session_123"
        }"#;

        let response: ClaudeCliResponse = serde_json::from_str(json_with_cost).unwrap();
        assert_eq!(response.cost_usd, Some(0.001234));
        assert_eq!(response.total_cost, Some(0.001234));

        // Test 2: JSON without cost fields
        let json_without_cost = r#"{
            "type": "assistant_response",
            "subtype": "completion",
            "is_error": false,
            "duration_ms": 1500,
            "duration_api_ms": 1200,
            "num_turns": 1,
            "result": "Hello, world!",
            "session_id": "session_123"
        }"#;

        let response: ClaudeCliResponse = serde_json::from_str(json_without_cost).unwrap();
        assert_eq!(response.cost_usd, None);
        assert_eq!(response.total_cost, None);

        // Test 3: JSON with null cost values
        let json_null_cost = r#"{
            "type": "assistant_response",
            "subtype": "completion",
            "cost_usd": null,
            "is_error": false,
            "duration_ms": 1500,
            "num_turns": 1,
            "result": "Hello, world!",
            "total_cost": null,
            "session_id": "session_123"
        }"#;

        let response: ClaudeCliResponse = serde_json::from_str(json_null_cost).unwrap();
        assert_eq!(response.cost_usd, None);
        assert_eq!(response.total_cost, None);
    }
}