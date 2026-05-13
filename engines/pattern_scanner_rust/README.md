# pattern-scanner-rust

A polyglot proof-of-concept sidecar engine for the Enchanter runtime.
Aho-Corasick-based scanner for a subset of the patterns shipped by
`enchanter/engines/secret_mask` and `enchanter/engines/cve_pattern_gate`.

This crate validates the Wave 13.1.5 sidecar runtime contract end-to-end
with a non-Python binary. It is **opt-in** — Python `secret-mask` and
`cve-pattern-gate` remain authoritative. Use this sidecar as an advisory
parallel scanner; promote to required only after benchmarking proves a
material speedup.

## Layout convention — why this lives outside `enchanter/engines/`

`enchanter/engines/` is reserved for Python-runtime engines. The
`load_engine_registry` discovery walker imports each subpackage as a
Python module and resolves an `adapter` attribute via
`module.path:attribute` notation — that walk would choke on a directory
that has no `__init__.py` and no Python adapter.

Sidecar engines, by contrast, are **binary artifacts** registered via
their own `engine.toml` and spawned as subprocesses. They live alongside
the `enchanter/` package, one directory per crate / binary:

```
agent/
├── enchanter/             ← Python package; Python engines live in
│   └── engines/             enchanter/engines/<name>/
└── engines/               ← polyglot sidecars (this directory)
    └── pattern_scanner_rust/
        ├── Cargo.toml
        ├── engine.toml      ← runtime = "sidecar"
        └── src/
```

Operators register a sidecar by pointing the loader at its `engine.toml`
explicitly (or by extending discovery to also walk `engines/`); see
"Register" below.

## Build

From this directory:

```
cargo build --release
```

The binary is emitted to `target/release/pattern-scanner-rust` (or
`pattern-scanner-rust.exe` on Windows). `engine.toml` references it via
a relative `command = "./target/release/pattern-scanner-rust"` path,
resolved against the manifest directory at spawn time.

## Smoke (manual)

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
    | ./target/release/pattern-scanner-rust
```

Expected output (one JSON line):

```
{"jsonrpc":"2.0","id":1,"result":{"name":"pattern-scanner-rust","phases":["trust-gate","post-response"],"required":false,"budget_tier":"always","topics":{"subscribes":["mcp.tool.call.requested","mcp.tool.result.received"],"emits":["pattern-scanner.matched"]}}}
```

## Register

Two options:

1. **Manual registration (recommended for v0):** the operator builds the
   binary and constructs a `SidecarAdapter` from this crate's
   `engine.toml` directly via `enchanter.loader.parse_manifest` +
   `load_sidecar_adapter`.
2. **Discovery extension (future):** teach `load_engine_registry` to
   also walk an `engines/` directory at the repo root for `engine.toml`
   files where `runtime = "sidecar"`. Not implemented in this wave.

## Pattern coverage (v0)

Subset of Python parity — see `src/patterns.rs` doc header for the full
table:

| pattern_id          | source engine     | severity | AC literal anchor |
|---------------------|-------------------|----------|-------------------|
| s-aws-key           | secret-mask       | 5        | `AKIA`            |
| s-bearer-token      | secret-mask       | 4        | `Bearer ` (trailing space) |
| s-pem-private-key   | secret-mask       | 6        | `-----BEGIN`      |
| h-rm-rf-root        | cve-pattern-gate  | 9        | `rm -rf /`        |
| h-curl-pipe-shell   | cve-pattern-gate  | 9        | `curl `           |
| h-fork-bomb         | cve-pattern-gate  | 7        | `:(){ :|:& };:`   |

Skipped for v0 (require regex features beyond literal AC anchors;
revisit with a regex-engine fallback):
- s-anthropic-key, s-openai-key
- h-ssh-key-exfil, h-sudo-nopasswd

## Verdict rule

- `trust-gate` + severity ≥ 7 → **veto** (fail-closed on critical CVE
  patterns before tool fires)
- otherwise (`post-response`, or sub-critical) → **ack + degraded=true**

In v0 we do **not** emit `derived_events` — Wave 14.1's source-allowlist
and topic-allowlist validation would force every derived event to carry
`source="pattern-scanner-rust"` and `topic="pattern-scanner.matched"`,
which is straightforward but unnecessary for the proof-of-concept.

## Cross-compilation / toolchain pinning

The release binary is platform-specific. Operators distributing
prebuilt binaries should:

- Pin a toolchain via `rust-toolchain.toml` (not committed in v0; add
  when this crate moves beyond proof-of-concept).
- Cross-compile per target with `cargo build --release --target ...`
  and ship a per-platform asset.
- The crate's only runtime dependency is the platform libc — no
  dynamically-linked sidecar deps.

## Future work

1. **Pattern parity** — port the remaining secret and CVE regexes
   (anthropic/openai keys, ssh-key exfil, sudo NOPASSWD) with a small
   regex-engine fallback gated by an AC pre-filter for hot-path skip.
2. **Derived events** — emit `pattern-scanner.matched` with `source =
   "pattern-scanner-rust"` after Wave 14.1 contract is exercised in the
   integration test fixtures.
3. **Perf benchmarks** — `cargo bench` vs the Python regex tables on
   representative corpora; promote `required = true` once a measurable
   tail-latency improvement is confirmed.
4. **Discovery integration** — extend `load_engine_registry` to walk
   `engines/` for sidecar manifests so manual registration is no longer
   required.
