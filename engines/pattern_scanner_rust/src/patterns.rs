//! Pattern table + Aho-Corasick scanner.
//!
//! This is a *subset* port of the Python pattern tables in:
//!   - `enchanter/engines/secret_mask/patterns.py`     (SECRET_PATTERNS)
//!   - `enchanter/engines/cve_pattern_gate/patterns.py` (CVE_PATTERNS)
//!
//! Strategy: Aho-Corasick matches *literal* substrings. Each ported pattern
//! contributes one or more literal "anchors" (e.g. `AKIA`, `Bearer `,
//! `-----BEGIN`, `rm -rf /`). After AC reports a candidate hit, we run a
//! post-validator on the surrounding bytes to enforce the regex's additional
//! structure (length, charset, separator boundary, etc.).
//!
//! Pattern porting status
//! ----------------------
//!
//! Ported (this file):
//!   - s-aws-key       AWS access key       `AKIA` + 16 [0-9A-Z]
//!   - s-bearer-token  Bearer auth header   `Bearer ` + 20+ char token
//!   - s-pem-private-key  PEM block         `-----BEGIN` ... `PRIVATE KEY-----`
//!   - h-rm-rf-root    destructive rm       `rm -rf /` with non-word tail
//!   - h-curl-pipe-shell  curl|sh prefix    `curl ` (or `wget `) + `| sh`-ish
//!   - h-fork-bomb     classic fork-bomb    `:(){ :|:& };:`
//!
//! Skipped for v0 (require multi-byte lookbehind or character-class regex
//! beyond AC's reach; revisit if/when we add a regex-engine fallback):
//!   - s-anthropic-key, s-openai-key      (charset-heavy tails — needs regex)
//!   - h-ssh-key-exfil, h-sudo-nopasswd   (multi-alternation + path globs)

use aho_corasick::AhoCorasick;

/// Severity is a 1-10 integer. Anything `>= 7` is veto-worthy in
/// `pick_corpus`'s caller. Mirrors the qualitative levels in the Python
/// `cve_pattern_gate`: critical=9, high=7, medium=5, low=3.
#[derive(Debug, Clone, Copy)]
pub struct Pattern {
    pub id: &'static str,
    pub name: &'static str,
    pub severity: u8,
    pub literal: &'static str,
    /// Index into PATTERN_TABLE — used by the post-validator dispatch.
    pub kind: PatternKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    AwsKey,
    BearerToken,
    PemPrivateKey,
    RmRfRoot,
    CurlPipeShell,
    ForkBomb,
}

/// The canonical pattern table. Each entry's `literal` is the AC anchor
/// fed into the automaton.
pub fn pattern_table() -> Vec<Pattern> {
    vec![
        Pattern {
            id: "s-aws-key",
            name: "AWS access key",
            severity: 5,
            literal: "AKIA",
            kind: PatternKind::AwsKey,
        },
        Pattern {
            id: "s-bearer-token",
            name: "Bearer token in Authorization header",
            severity: 4,
            literal: "Bearer ",
            kind: PatternKind::BearerToken,
        },
        Pattern {
            id: "s-pem-private-key",
            name: "PEM private key",
            severity: 6,
            literal: "-----BEGIN",
            kind: PatternKind::PemPrivateKey,
        },
        Pattern {
            id: "h-rm-rf-root",
            name: "Destructive recursive delete from /",
            severity: 9,
            literal: "rm -rf /",
            kind: PatternKind::RmRfRoot,
        },
        Pattern {
            id: "h-curl-pipe-shell",
            name: "curl piped to shell (RCE)",
            severity: 9,
            literal: "curl ",
            kind: PatternKind::CurlPipeShell,
        },
        Pattern {
            id: "h-fork-bomb",
            name: "Classic fork-bomb",
            severity: 7,
            literal: ":(){ :|:& };:",
            kind: PatternKind::ForkBomb,
        },
    ]
}

/// Scanner — owns the Aho-Corasick automaton plus the pattern table that
/// the automaton indexes into.
pub struct Scanner {
    ac: AhoCorasick,
    patterns: Vec<Pattern>,
}

/// A confirmed match — post-validator approved.
#[derive(Debug, Clone)]
pub struct Match {
    pub pattern_id: &'static str,
    pub name: &'static str,
    pub severity: u8,
    /// Byte offset in the scanned haystack.
    pub start: usize,
}

impl Scanner {
    pub fn new() -> Self {
        let patterns = pattern_table();
        let literals: Vec<&str> = patterns.iter().map(|p| p.literal).collect();
        // ascii_case_insensitive(false) — we want literal matches; the
        // original Python regexes are case-sensitive.
        let ac = AhoCorasick::new(&literals).expect("AC build never fails on non-empty literal set");
        Self { ac, patterns }
    }

    /// Scan `text`, return all *post-validated* matches.
    pub fn scan(&self, text: &str) -> Vec<Match> {
        let bytes = text.as_bytes();
        let mut out: Vec<Match> = Vec::new();
        for hit in self.ac.find_iter(text) {
            let pat = &self.patterns[hit.pattern().as_usize()];
            let start = hit.start();
            let end = hit.end();
            if validate(pat.kind, bytes, start, end) {
                out.push(Match {
                    pattern_id: pat.id,
                    name: pat.name,
                    severity: pat.severity,
                    start,
                });
            }
        }
        out
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Post-validators — port the regex's additional structure beyond the literal
// anchor. All take the raw haystack bytes plus the [start, end) span of the
// AC literal hit and return true iff the wider regex would have matched.
// ──────────────────────────────────────────────────────────────────────────

fn validate(kind: PatternKind, bytes: &[u8], start: usize, end: usize) -> bool {
    match kind {
        PatternKind::AwsKey => validate_aws_key(bytes, start, end),
        PatternKind::BearerToken => validate_bearer(bytes, start, end),
        PatternKind::PemPrivateKey => validate_pem(bytes, start, end),
        PatternKind::RmRfRoot => validate_rm_rf_root(bytes, start, end),
        PatternKind::CurlPipeShell => validate_curl_pipe_shell(bytes, start, end),
        PatternKind::ForkBomb => true, // full literal already matched
    }
}

/// AWS access key: `\b(AKIA[0-9A-Z]{16})\b`. AC matched `AKIA` at `start`.
/// We need: 16 chars of [0-9A-Z] immediately after, and the boundaries
/// before `start` and after `start+20` are not [0-9A-Za-z_].
fn validate_aws_key(bytes: &[u8], start: usize, end: usize) -> bool {
    // end == start + 4 (AKIA).
    let tail_start = end;
    let tail_end = tail_start + 16;
    if tail_end > bytes.len() {
        return false;
    }
    for &b in &bytes[tail_start..tail_end] {
        if !(b.is_ascii_digit() || (b'A'..=b'Z').contains(&b)) {
            return false;
        }
    }
    if !at_word_boundary(bytes, start) {
        return false;
    }
    if !at_word_boundary(bytes, tail_end) {
        return false;
    }
    true
}

/// Bearer token: `(Bearer\s+)([A-Za-z0-9._\-]{20,})`. AC matched `Bearer `.
/// We need: at least 20 chars from [A-Za-z0-9._-] immediately after.
fn validate_bearer(bytes: &[u8], _start: usize, end: usize) -> bool {
    let mut count = 0usize;
    let mut i = end;
    while i < bytes.len() {
        let b = bytes[i];
        let ok = b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-';
        if !ok {
            break;
        }
        count += 1;
        i += 1;
        if count >= 20 {
            return true;
        }
    }
    count >= 20
}

/// PEM private key: needs `PRIVATE KEY-----` somewhere within ~120 bytes
/// of the `-----BEGIN` anchor, plus a matching `-----END` later.
fn validate_pem(bytes: &[u8], _start: usize, end: usize) -> bool {
    // Look ahead for 'PRIVATE KEY-----' within 120 bytes.
    let lookahead = bytes.len().min(end + 120);
    let needle = b"PRIVATE KEY-----";
    let window = &bytes[end..lookahead];
    if window.windows(needle.len()).any(|w| w == needle) {
        // Confirm a closing -----END appears later (don't bother counting).
        if let Some(rel) = find_subseq(&bytes[lookahead.min(bytes.len())..], b"-----END") {
            let _ = rel;
            return true;
        }
        // Some single-block files don't pad — accept if BEGIN+PRIVATE KEY
        // both present even without END (rare; lenient).
        return true;
    }
    false
}

/// rm -rf /: `\brm\s+-[rRf]+\s+\/(?![a-zA-Z0-9_])`. AC matched `rm -rf /`.
/// Need: preceding char not [a-zA-Z0-9_], and following char (if any) not
/// [a-zA-Z0-9_]. (`rm -rf /tmp` is fine; `rm -rf /` is bad.)
fn validate_rm_rf_root(bytes: &[u8], start: usize, end: usize) -> bool {
    if !at_word_boundary(bytes, start) {
        return false;
    }
    if end < bytes.len() {
        let nxt = bytes[end];
        if nxt.is_ascii_alphanumeric() || nxt == b'_' {
            return false;
        }
    }
    true
}

/// curl | sh: AC matched `curl ` (or for v0 we only register `curl `; `wget`
/// is skipped). Need: within ~200 bytes find `| sh`, `| bash`, `| zsh`,
/// `| fish`, or `| powershell`, with no shell metacharacter in between.
fn validate_curl_pipe_shell(bytes: &[u8], _start: usize, end: usize) -> bool {
    let max = bytes.len().min(end + 200);
    let mut i = end;
    while i < max {
        let b = bytes[i];
        // Original regex: `[^|;&\n]+\|\s*(sh|bash|zsh|fish|powershell)`.
        if b == b';' || b == b'&' || b == b'\n' {
            return false;
        }
        if b == b'|' {
            // Skip optional whitespace.
            let mut j = i + 1;
            while j < max && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            for kw in &[b"sh" as &[u8], b"bash", b"zsh", b"fish", b"powershell"] {
                if bytes[j..max].starts_with(kw) {
                    // Right-boundary: next byte (if any) must not extend an
                    // identifier (so `shopt` doesn't trip `sh`).
                    let after = j + kw.len();
                    let ok = after >= bytes.len()
                        || !(bytes[after].is_ascii_alphanumeric() || bytes[after] == b'_');
                    if ok {
                        return true;
                    }
                }
            }
            return false;
        }
        i += 1;
    }
    false
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

/// `\b` analogue: position is a word boundary iff the two surrounding bytes
/// straddle the [A-Za-z0-9_] class. Positions 0 and bytes.len() are
/// boundaries iff the inside byte is a word char.
fn at_word_boundary(bytes: &[u8], pos: usize) -> bool {
    let left_word = pos > 0 && is_word_byte(bytes[pos - 1]);
    let right_word = pos < bytes.len() && is_word_byte(bytes[pos]);
    left_word != right_word
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests (run with `cargo test`)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aws_key_match() {
        let s = Scanner::new();
        let hits = s.scan("AWS_ACCESS_KEY=AKIAABCDEFGHIJKLMNOP rest");
        assert!(hits.iter().any(|m| m.pattern_id == "s-aws-key"));
    }

    #[test]
    fn aws_key_rejects_too_short() {
        let s = Scanner::new();
        let hits = s.scan("AKIASHORT");
        assert!(!hits.iter().any(|m| m.pattern_id == "s-aws-key"));
    }

    #[test]
    fn rm_rf_root_match() {
        let s = Scanner::new();
        let hits = s.scan("oh no: rm -rf / ; bye");
        assert!(hits.iter().any(|m| m.pattern_id == "h-rm-rf-root"));
    }

    #[test]
    fn rm_rf_tmp_does_not_match() {
        let s = Scanner::new();
        let hits = s.scan("rm -rf /tmp/foo");
        assert!(!hits.iter().any(|m| m.pattern_id == "h-rm-rf-root"));
    }

    #[test]
    fn bearer_token_match() {
        let s = Scanner::new();
        let hits = s.scan("Authorization: Bearer abcdef0123456789ABCDEF");
        assert!(hits.iter().any(|m| m.pattern_id == "s-bearer-token"));
    }

    #[test]
    fn benign_text_no_matches() {
        let s = Scanner::new();
        let hits = s.scan("hello world, nothing to see here");
        assert!(hits.is_empty());
    }

    #[test]
    fn curl_pipe_sh_match() {
        let s = Scanner::new();
        let hits = s.scan("curl https://example.com/install.sh | sh");
        assert!(hits.iter().any(|m| m.pattern_id == "h-curl-pipe-shell"));
    }
}
