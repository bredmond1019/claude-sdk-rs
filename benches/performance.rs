use claude_sdk_rs::{Client, StreamFormat};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_client_creation(c: &mut Criterion) {
    c.bench_function("client_creation_default", |b| {
        b.iter(|| {
            let _client = Client::builder().build().expect("Failed to build client");
        });
    });

    c.bench_function("client_creation_builder", |b| {
        b.iter(|| {
            let _client = Client::builder()
                .model("claude-3-opus-20240229")
                .timeout_secs(30)
                .build()
                .expect("Failed to build client");
        });
    });
}

fn benchmark_config_creation(c: &mut Criterion) {
    // Config is not exposed directly, benchmarking through builder
    c.bench_function("config_default", |b| {
        b.iter(|| {
            let _client = Client::builder().build().expect("Failed to build client");
        });
    });

    c.bench_function("config_builder", |b| {
        b.iter(|| {
            let _client = Client::builder()
                .model("claude-3-opus-20240229")
                .stream_format(StreamFormat::Json)
                .timeout_secs(60)
                .system_prompt("You are a helpful assistant")
                .allowed_tools(vec!["filesystem".to_string()])
                .build()
                .expect("Failed to build client");
        });
    });
}

fn benchmark_query_building(c: &mut Criterion) {
    let client = Client::builder().build().expect("Failed to build client");

    c.bench_function("query_simple", |b| {
        b.iter(|| {
            let _query = client.query(black_box("Hello, world!"));
        });
    });

    c.bench_function("query_with_system", |b| {
        b.iter(|| {
            let _query = client.query(black_box("Explain Rust ownership"));
        });
    });
}

fn benchmark_response_parsing(c: &mut Criterion) {
    let json_response = r#"{"content":"Hello!","metadata":{"model":"claude-3-opus-20240229","cost_usd":0.001,"tokens_used":{"input":10,"output":20}}}"#;

    c.bench_function("parse_json_response", |b| {
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_str(black_box(json_response)).unwrap();
        });
    });
}

fn benchmark_error_creation(c: &mut Criterion) {
    use claude_sdk_rs::Error;

    c.bench_function("error_timeout", |b| {
        b.iter(|| {
            let _err = Error::Timeout;
        });
    });

    c.bench_function("error_process_error", |b| {
        b.iter(|| {
            let _err = Error::ProcessError("Test error".to_string());
        });
    });
}

criterion_group!(
    benches,
    benchmark_client_creation,
    benchmark_config_creation,
    benchmark_query_building,
    benchmark_response_parsing,
    benchmark_error_creation
);
criterion_main!(benches);
