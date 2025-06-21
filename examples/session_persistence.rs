//! Session Persistence Example
//!
//! This example demonstrates the session persistence functionality across
//! different storage backends (memory, file, and SQLite).
//!
//! NOTE: This example requires internal crate access and is meant for internal testing only.
//! It cannot be run as a public example since it uses internal types from claude_sdk_rs_core
//! that are not publicly exported.

fn main() {
    eprintln!("This example requires internal crate access and cannot be run as a public example.");
    eprintln!("It uses types from claude_sdk_rs_core that are not publicly exported.");
    eprintln!("\nFor session management examples, see:");
    eprintln!("  - examples/session_management.rs");
    eprintln!("  - examples/02_sdk_sessions.rs");
    std::process::exit(1);
}
