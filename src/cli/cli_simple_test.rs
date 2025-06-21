#[cfg(test)]
mod cli_simple_tests {
    use crate::cli::*;

    #[test]
    fn test_list_command_creation() {
        let cmd = ListCommand {
            filter: Some("test".to_string()),
            detailed: true,
        };

        assert_eq!(cmd.filter, Some("test".to_string()));
        assert!(cmd.detailed);
    }

    #[test]
    fn test_run_command_creation() {
        let cmd = RunCommand {
            command: "analyze".to_string(),
            args: vec!["--file".to_string(), "test.rs".to_string()],
            session: Some("my-session".to_string()),
            parallel: true,
            agents: 4,
        };

        assert_eq!(cmd.command, "analyze");
        assert_eq!(cmd.args.len(), 2);
        assert_eq!(cmd.session, Some("my-session".to_string()));
        assert!(cmd.parallel);
        assert_eq!(cmd.agents, 4);
    }

    #[test]
    fn test_session_action_variants() {
        // Test Create variant
        let create = SessionAction::Create {
            name: "test-session".to_string(),
            description: Some("Test description".to_string()),
        };

        match create {
            SessionAction::Create { name, description } => {
                assert_eq!(name, "test-session");
                assert_eq!(description, Some("Test description".to_string()));
            }
            _ => panic!("Wrong variant"),
        }

        // Test Delete variant
        let delete = SessionAction::Delete {
            session: "test-session".to_string(),
            force: true,
        };

        match delete {
            SessionAction::Delete { session, force } => {
                assert_eq!(session, "test-session");
                assert!(force);
            }
            _ => panic!("Wrong variant"),
        }

        // Test List variant
        let list = SessionAction::List { detailed: false };

        match list {
            SessionAction::List { detailed } => {
                assert!(!detailed);
            }
            _ => panic!("Wrong variant"),
        }

        // Test Switch variant
        let switch = SessionAction::Switch {
            session: "other-session".to_string(),
        };

        match switch {
            SessionAction::Switch { session } => {
                assert_eq!(session, "other-session");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_cost_command_creation() {
        let cmd = CostCommand {
            session: Some("work-session".to_string()),
            breakdown: true,
            since: Some("2024-01-01".to_string()),
            export: Some("json".to_string()),
        };

        assert_eq!(cmd.session, Some("work-session".to_string()));
        assert!(cmd.breakdown);
        assert_eq!(cmd.since, Some("2024-01-01".to_string()));
        assert_eq!(cmd.export, Some("json".to_string()));
    }

    #[test]
    fn test_history_command_creation() {
        let cmd = HistoryCommand {
            search: Some("analyze.*test".to_string()),
            session: Some("debug-session".to_string()),
            limit: 50,
            output: true,
            export: Some("csv".to_string()),
        };

        assert_eq!(cmd.search, Some("analyze.*test".to_string()));
        assert_eq!(cmd.session, Some("debug-session".to_string()));
        assert_eq!(cmd.limit, 50);
        assert!(cmd.output);
        assert_eq!(cmd.export, Some("csv".to_string()));
    }

    #[test]
    fn test_config_action_variants() {
        // Test Show variant
        let show = ConfigAction::Show;
        match show {
            ConfigAction::Show => {
                // Passes if variant matches
            }
            _ => panic!("Wrong variant"),
        }

        // Test Set variant
        let set = ConfigAction::Set {
            key: "timeout_secs".to_string(),
            value: "60".to_string(),
        };

        match set {
            ConfigAction::Set { key, value } => {
                assert_eq!(key, "timeout_secs");
                assert_eq!(value, "60");
            }
            _ => panic!("Wrong variant"),
        }

        // Test Reset variant
        let reset = ConfigAction::Reset { force: false };

        match reset {
            ConfigAction::Reset { force } => {
                assert!(!force);
            }
            _ => panic!("Wrong variant"),
        }

        // Test Path variant
        let path = ConfigAction::Path;
        match path {
            ConfigAction::Path => {
                // Passes if variant matches
            }
            _ => panic!("Wrong variant"),
        }
    }
}
