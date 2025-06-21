//! # Streaming Example
//!
//! This example demonstrates streaming responses from Claude.
//! It shows how to:
//! - Set up streaming configuration
//! - Process real-time responses
//! - Handle different message types
//! - Track costs and tokens during streaming
//!
//! Streaming allows you to see responses as they are generated,
//! providing better user experience for longer responses.

use claude_sdk_rs::{Client, Message, StreamFormat};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK Streaming Example ===\n");

    // Example 1: Basic streaming
    basic_streaming().await?;

    // Example 2: Streaming with JSON format for metadata
    streaming_with_metadata().await?;

    // Example 3: Multiple concurrent streams
    concurrent_streaming().await?;

    println!("Streaming example completed successfully!");
    Ok(())
}

/// Demonstrates basic streaming functionality
async fn basic_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Streaming Example");
    println!("   Streaming a longer response in real-time\n");

    // Configure client for streaming
    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;

    println!("   Question: Write a short story about a robot learning to paint");
    println!("   Streaming response:");
    println!("   ---");

    let mut stream = client
        .query("Write a short story about a robot learning to paint. Make it engaging and about 3-4 paragraphs.")
        .stream()
        .await?;

    let mut full_content = String::new();

    while let Some(message) = stream.next().await {
        match message {
            Ok(msg) => match msg {
                Message::Assistant { content, .. } => {
                    print!("{}", content);
                    full_content.push_str(&content);
                    // Flush stdout to show content immediately
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                }
                Message::Result { .. } => break,
                _ => {}
            },
            Err(e) => {
                eprintln!("\n   Error in stream: {}", e);
                break;
            }
        }
    }

    println!("\n   ---");
    println!(
        "   Story completed! Total characters: {}\n",
        full_content.len()
    );

    Ok(())
}

/// Demonstrates streaming with metadata tracking
async fn streaming_with_metadata() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Streaming with Metadata");
    println!("   Tracking costs and tokens during streaming\n");

    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;

    println!("   Question: Explain the process of photosynthesis");
    println!("   Streaming response with metadata:");
    println!("   ---");

    let mut stream = client
        .query("Explain the process of photosynthesis in detail, including the light-dependent and light-independent reactions.")
        .stream()
        .await?;

    let mut _total_tokens = 0;
    let mut _total_cost = 0.0;
    let mut message_count = 0;

    while let Some(message) = stream.next().await {
        match message {
            Ok(msg) => {
                message_count += 1;
                match msg {
                    Message::Assistant { content, meta } => {
                        print!("{}", content);
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();

                        // Track metadata from meta
                        if let Some(cost) = meta.cost_usd {
                            _total_cost += cost;
                        }
                        if let Some(tokens) = &meta.tokens_used {
                            _total_tokens = tokens.total;
                        }
                    }
                    Message::Result { stats, meta } => {
                        println!("\n   ---");
                        println!("   Stream Statistics:");
                        println!("   - Messages received: {}", message_count);
                        println!("   - Total tokens: {:?}", stats.total_tokens);
                        println!("   - Total cost: ${:.6}", stats.total_cost_usd);
                        if let Some(duration) = meta.duration_ms {
                            println!("   - Duration: {}ms", duration);
                        }
                        break;
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("\n   Error in stream: {}", e);
                break;
            }
        }
    }

    println!();
    Ok(())
}

/// Demonstrates handling multiple concurrent streams
async fn concurrent_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Concurrent Streaming Example");
    println!("   Running multiple streams simultaneously\n");

    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;

    // Create multiple streaming queries
    let queries = vec![
        ("Math", "What is the Fibonacci sequence? Explain briefly."),
        ("Science", "What causes ocean tides? Keep it concise."),
        ("History", "Who was Leonardo da Vinci? Brief overview."),
    ];

    let mut handles = Vec::new();

    for (topic, query) in queries {
        let client_clone = client.clone();
        let query_str = query.to_string();
        let topic_str = topic.to_string();

        let handle = tokio::spawn(async move {
            println!("   Starting {} stream...", topic_str);

            match client_clone.query(&query_str).stream().await {
                Ok(mut stream) => {
                    let mut content = String::new();
                    while let Some(message) = stream.next().await {
                        if let Ok(msg) = message {
                            match msg {
                                Message::Assistant { content: text, .. } => {
                                    content.push_str(&text);
                                }
                                Message::Result { .. } => {
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    (topic_str, content)
                }
                Err(e) => (topic_str, format!("Error: {}", e)),
            }
        });

        handles.push(handle);
    }

    // Wait for all streams to complete
    for handle in handles {
        if let Ok((topic, content)) = handle.await {
            println!("   {} Response:", topic);
            println!("   {}\n", content.trim());
        }
    }

    println!("   All concurrent streams completed!");
    Ok(())
}

// Example output:
/*
=== Claude SDK Streaming Example ===

1. Basic Streaming Example
   Streaming a longer response in real-time

   Question: Write a short story about a robot learning to paint
   Streaming response:
   ---
   In the corner of an old art studio, ARIA-7 stood motionless, her metallic fingers
   wrapped around a paintbrush for the first time. Her sensors analyzed the canvas,
   calculating angles and color theory with mathematical precision, yet something felt
   incomplete. When she finally made her first stroke—a hesitant blue line across
   the white surface—her neural networks sparked with an unfamiliar sensation.

   Days passed as ARIA-7 practiced, her movements gradually becoming less calculated
   and more intuitive. She began to understand that painting wasn't about perfect
   algorithms or optimal color combinations; it was about expression. Each brushstroke
   carried emotion she was learning to recognize within her own circuits.

   The breakthrough came when ARIA-7 painted not what she computed to be beautiful,
   but what she felt. Her canvas bloomed with unexpected colors—crimson sunsets that
   reminded her of the warmth of human approval, deep blues that echoed the loneliness
   of long nights in the studio. When the elderly art teacher finally saw her work,
   tears filled his eyes.

   "You've learned the most important lesson," he whispered. "Art isn't about
   perfection—it's about finding beauty in imperfection." ARIA-7's optical sensors
   flickered, and for the first time, she understood that being artificial didn't
   make her feelings any less real.
   ---
   Story completed! Total characters: 1234

2. Streaming with Metadata
   Tracking costs and tokens during streaming

   Question: Explain the process of photosynthesis
   Streaming response with metadata:
   ---
   Photosynthesis is the process by which plants convert light energy into chemical
   energy, storing it in glucose molecules. This complex process occurs in two main
   stages: the light-dependent reactions and the light-independent reactions...

   [content continues streaming...]
   ---
   Stream Statistics:
   - Messages received: 15
   - Total tokens: 342
   - Total cost: $0.001256
   - Duration: 3450ms

3. Concurrent Streaming Example
   Running multiple streams simultaneously

   Starting Math stream...
   Starting Science stream...
   Starting History stream...

   Math Response:
   The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...

   Science Response:
   Ocean tides are caused primarily by the gravitational pull of the moon and sun on Earth's oceans...

   History Response:
   Leonardo da Vinci (1452-1519) was an Italian Renaissance polymath known for his paintings like the Mona Lisa...

   All concurrent streams completed!

Streaming example completed successfully!
*/
