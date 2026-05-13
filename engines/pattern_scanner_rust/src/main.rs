//! pattern-scanner-rust — JSON-RPC sidecar speaking the Wave 13.1.5 protocol.
//!
//! Protocol (newline-framed, one JSON object per line):
//!   IN:  {"jsonrpc":"2.0","id":<int>,"method":"initialize"}
//!   IN:  {"jsonrpc":"2.0","id":<int>,"method":"on_phase","params":{"event":{...},"context":{...}}}
//!   IN:  {"jsonrpc":"2.0","method":"shutdown"}    (notification — no id)
//!   OUT: {"jsonrpc":"2.0","id":<int>,"result":{...}}
//!
//! Exit on stdin EOF.
//!
//! Wave 14.1 trust contract: we do NOT emit derived events in v0. Every
//! response sets derived_events=[].

mod patterns;

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

use patterns::{Match, Scanner};

/// Per-message stdin cap matches enchanter.transport.stdio (8 MiB).
const PER_MESSAGE_MAX: usize = 8 * 1024 * 1024;

fn main() -> io::Result<()> {
    let scanner = Scanner::new();

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut line = String::new();
    loop {
        line.clear();
        let n = match stdin.lock().read_line(&mut line) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("pattern-scanner-rust: stdin read error: {e}");
                return Err(e);
            }
        };
        if n == 0 {
            // EOF — graceful shutdown.
            return Ok(());
        }
        if line.len() > PER_MESSAGE_MAX {
            eprintln!(
                "pattern-scanner-rust: oversized message {} bytes (cap {})",
                line.len(),
                PER_MESSAGE_MAX
            );
            // Drop the connection — parent will restart us.
            return Ok(());
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("pattern-scanner-rust: malformed JSON: {e}");
                continue;
            }
        };

        let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        match method {
            "initialize" => {
                let result = initialize_result();
                write_response(&mut out, id, result)?;
            }
            "on_phase" => {
                let result = handle_on_phase(&scanner, &msg);
                write_response(&mut out, id, result)?;
            }
            "shutdown" => {
                // Notification — no reply expected (parent may also drop stdin).
                return Ok(());
            }
            other => {
                // Reply with a JSON-RPC error so the parent's protocol layer
                // surfaces a descriptive veto reason rather than crash-restart.
                let err = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("method not found: {other}")
                    }
                });
                writeln!(out, "{}", err)?;
                out.flush()?;
            }
        }
    }
}

fn initialize_result() -> Value {
    json!({
        "name": "pattern-scanner-rust",
        "phases": ["trust-gate", "post-response"],
        "required": false,
        "budget_tier": "always",
        "topics": {
            "subscribes": ["mcp.tool.call.requested", "mcp.tool.result.received"],
            "emits": ["pattern-scanner.matched"]
        }
    })
}

/// Build the on_phase result given a parsed incoming envelope.
fn handle_on_phase(scanner: &Scanner, msg: &Value) -> Value {
    let event = msg.pointer("/params/event").cloned().unwrap_or(Value::Null);
    let phase = event.get("phase").and_then(|v| v.as_str()).unwrap_or("");
    let payload = event.get("payload").cloned().unwrap_or(Value::Null);

    let corpus = pick_corpus(phase, &payload);
    if corpus.is_empty() {
        return ack_clean();
    }

    let matches = scanner.scan(&corpus);
    classify(matches, phase)
}

/// Aggregate matches into an ack/veto verdict.
///
/// Veto rule: at `trust-gate`, severity >= 7 → veto (fail-closed on critical
/// CVE patterns BEFORE the tool fires). At `post-response`, even severity-9
/// matches degrade-only — the result already exists; veto'ing would be
/// security theatre and Python `secret-mask` redacts.
fn classify(matches: Vec<Match>, phase: &str) -> Value {
    if matches.is_empty() {
        return ack_clean();
    }
    let max_sev = matches.iter().map(|m| m.severity).max().unwrap_or(0);
    let first = &matches[0];

    if phase == "trust-gate" && max_sev >= 7 {
        return json!({
            "status": "veto",
            "reason": format!(
                "pattern-scanner: {} (severity {})",
                first.pattern_id, first.severity
            ),
            "derived_events": [],
            "degraded": false
        });
    }

    // Advisory match (post-response, or sub-critical at trust-gate).
    json!({
        "status": "ack",
        "reason": format!(
            "pattern-scanner: matched {} pattern(s); first={}",
            matches.len(), first.pattern_id
        ),
        "derived_events": [],
        "degraded": true
    })
}

fn ack_clean() -> Value {
    json!({
        "status": "ack",
        "reason": Value::Null,
        "derived_events": [],
        "degraded": false
    })
}

/// Pick which string field(s) of the event payload to scan based on phase.
///
/// trust-gate (mcp.tool.call.requested): args + prompt_summary — anything the
///   caller is about to send.
/// post-response (mcp.tool.result.received): result — anything coming back.
///
/// Any string value found at one of these keys is concatenated with a
/// newline separator. Non-string values are JSON-stringified so we still
/// catch embedded secrets in nested dicts.
fn pick_corpus(phase: &str, payload: &Value) -> String {
    let keys: &[&str] = match phase {
        "trust-gate" => &["args", "prompt_summary", "tool_name"],
        "post-response" => &["result", "text", "output"],
        _ => &["args", "prompt_summary", "result", "text", "output"],
    };
    let mut buf = String::new();
    if let Some(obj) = payload.as_object() {
        for k in keys {
            if let Some(v) = obj.get(*k) {
                append_value(&mut buf, v);
            }
        }
    } else if payload.is_string() {
        append_value(&mut buf, payload);
    }
    buf
}

fn append_value(buf: &mut String, v: &Value) {
    match v {
        Value::String(s) => {
            buf.push_str(s);
            buf.push('\n');
        }
        Value::Null => {}
        other => {
            buf.push_str(&other.to_string());
            buf.push('\n');
        }
    }
}

fn write_response<W: Write>(out: &mut W, id: Option<Value>, result: Value) -> io::Result<()> {
    let envelope = json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "result": result
    });
    // serde_json::to_string is guaranteed not to emit embedded newlines for
    // any Value (it escapes \n inside strings).
    let line = serde_json::to_string(&envelope).expect("serde_json::to_string on Value cannot fail");
    writeln!(out, "{}", line)?;
    out.flush()
}
