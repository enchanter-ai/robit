#!/usr/bin/env python3
"""Robit Inference Engine — evidence accumulation over an artifact stream.

Port of ``wixie/shared/scripts/inference-engine.py``.  Algorithms U1-U6
preserved verbatim; paths and env-var names adapted for robit.

Stdlib only. No external runtime deps.

Subcommands:
    emit <record.json|->            Append an artifact record to state/artifacts.jsonl
    reconcile                       Fingerprint artifacts (U1), run Wald SPRT (U2),
                                    update Beta-Binomial (U3), apply EMA decay (U5).
                                    Atomic catalog write.
    render-briefing <plugin>        Render state/briefings/<plugin>.md from catalog
    query <term>                    Search catalog by code, tag, or pattern_id.
                                    JSON to stdout.
    backfill <source.jsonl>         Replay an external JSONL (e.g. precedent.jsonl)
                                    through emit
    status                          Print catalog summary + last reconcile timestamp

Algorithms (all stdlib):
    U1 Pattern fingerprint          SHA-1 of (code, sorted(tags)) — deterministic id
                                    per semantic pattern
    U2 Wald SPRT                    log-likelihood ratio over recurrences; elevate at
                                    +2.89, retire at -2.25
    U3 Beta-Binomial posterior      alpha/beta per pattern; mean + 95% CI via beta
                                    quantile
    U5 EMA decay                    weight *= exp(-lambda * days_since_last_seen);
                                    lambda = ln(2)/30
    U6 Reservoir sampling           bounded retention of raw artifacts per pattern
                                    (K=50 Vitter)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Windows default console is cp1252 — reconfigure to UTF-8 so non-ASCII glyphs
# (em-dashes, arrows, CI brackets) in output don't crash the subcommand.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─── Opt-in gate ──────────────────────────────────────────────────────────────


def env_enabled() -> bool:
    """Opt-in gate — default off during rollout."""
    from robit._compat import get_env

    return get_env("ROBIT_INFERENCE_ENABLED", "0") == "1"


# ─── Path helpers ─────────────────────────────────────────────────────────────


def _state_dir() -> Path:
    """Return state dir, honouring ROBIT_INFERENCE_STATE override at call time.

    The legacy ENCHANTER_INFERENCE_STATE name is still honored (with a
    deprecation warning) via :mod:`robit._compat`.
    """
    from robit._compat import get_env
    from robit.inference.paths import DEFAULT_STATE_DIR

    override = get_env("ROBIT_INFERENCE_STATE")
    return Path(override) if override else DEFAULT_STATE_DIR


def artifacts_path(state_dir: Path | None = None) -> Path:
    # One master append-only log. Rotation deferred until volume warrants it.
    return (state_dir or _state_dir()) / "artifacts.jsonl"


def catalog_path(state_dir: Path | None = None) -> Path:
    return (state_dir or _state_dir()) / "catalog.json"


def briefings_dir(state_dir: Path | None = None) -> Path:
    return (state_dir or _state_dir()) / "briefings"


# ─── IO helpers ───────────────────────────────────────────────────────────────


def append_jsonl_locked(path: Path, line: str) -> None:
    """Append one JSON line, locked + flushed. Cross-platform.

    Closes F-015: parallel races and partial-line writes on artifacts.jsonl
    that biased the SPRT walk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not line.endswith("\n"):
        line += "\n"
    with path.open("a", encoding="utf-8") as f:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl

            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def load_catalog(state_dir: Path | None = None) -> dict:
    p = catalog_path(state_dir)
    if not p.exists():
        return {"version": 1, "last_reconciled": None, "patterns": {}}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_artifacts(state_dir: Path | None = None) -> list[dict]:
    """Load every artifact from artifacts.jsonl plus any legacy artifacts-*.jsonl files."""
    sd = state_dir or _state_dir()
    out: list[dict] = []
    master = sd / "artifacts.jsonl"
    paths = [master] if master.exists() else []
    paths.extend(sorted(sd.glob("artifacts-*.jsonl")))
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── U1: Pattern fingerprint ──────────────────────────────────────────────────


def fingerprint(record: dict) -> str:
    """SHA-1 of (code, sorted(tags)). Deterministic across sessions."""
    code = record.get("code", "")
    tags = sorted(record.get("tags") or [])
    key = f"{code}|{'|'.join(tags)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ─── U2: Wald Sequential Probability Ratio Test ───────────────────────────────
#
# Hypothesis test:
#   H0: pattern is noise          (p0 = 0.05 per independent observation)
#   H1: pattern is real recurrence (p1 = 0.30 per independent observation)
# Thresholds (Wald, alpha=0.05, beta=0.10):
#   A = ln((1-beta)/alpha) = ln(0.90/0.05) ≈ 2.890
#   B = ln(beta/(1-alpha)) = ln(0.10/0.95) ≈ -2.251

P0 = 0.05
P1 = 0.30
LLR_ELEVATE: float = math.log((1 - 0.10) / 0.05)   # ≈ +2.890
LLR_RETIRE: float = math.log(0.10 / (1 - 0.05))     # ≈ -2.251
LLR_POS: float = math.log(P1 / P0)                  # per positive observation
LLR_NEG: float = math.log((1 - P1) / (1 - P0))      # per non-recurrence (rarely used)


def sprt_update(prior_llr: float, positive_observations: int) -> float:
    """Add positive_observations to the running LLR. Capped at [-10, +10]."""
    new_llr = prior_llr + positive_observations * LLR_POS
    return max(-10.0, min(10.0, new_llr))


def sprt_verdict(llr: float) -> str:
    if llr >= LLR_ELEVATE:
        return "elevated"
    if llr <= LLR_RETIRE:
        return "retired"
    return "noise"


# ─── U3: Beta-Binomial posterior ──────────────────────────────────────────────


def beta_update(
    alpha: float, beta: float, successes: int, failures: int
) -> tuple[float, float]:
    return alpha + successes, beta + failures


def beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _beta_quantile(alpha: float, beta: float, q: float, iters: int = 60) -> float:
    """Bisection over the regularized incomplete beta via series. Stdlib-only.

    For small alpha/beta (common here), a coarse bisection is sufficient for
    95% CI reporting — we don't need 6 decimal places.
    """
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        cdf = _incomplete_beta(alpha, beta, mid)
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _incomplete_beta(a: float, b: float, x: float, steps: int = 256) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # I_x(a,b) via Simpson's rule on the integrand t^(a-1)*(1-t)^(b-1)/B(a,b).
    log_betaB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    h = x / steps

    def f(t: float) -> float:
        if t <= 0 or t >= 1:
            return 0.0
        return math.exp((a - 1) * math.log(t) + (b - 1) * math.log(1 - t) - log_betaB)

    s = f(0.0) + f(x)
    for i in range(1, steps):
        s += (4 if i % 2 else 2) * f(i * h)
    return max(0.0, min(1.0, s * h / 3.0))


def beta_ci(
    alpha: float, beta: float, level: float = 0.95
) -> tuple[float, float]:
    tail = (1 - level) / 2
    return (
        _beta_quantile(alpha, beta, tail),
        _beta_quantile(alpha, beta, 1 - tail),
    )


# ─── U5: EMA decay ────────────────────────────────────────────────────────────
#
# lambda = ln(2) / half_life_days.  half_life=30 days means a pattern unseen
# for 30 days weighs half what it did when last seen.

HALF_LIFE_DAYS: float = 30.0
LAMBDA: float = math.log(2.0) / HALF_LIFE_DAYS


def ema_weight(days_since_last_seen: float, base: float = 1.0) -> float:
    return base * math.exp(-LAMBDA * max(0.0, days_since_last_seen))


# ─── U6: Reservoir sampling (Vitter Algorithm R) ──────────────────────────────


def reservoir_add(
    reservoir: list, item: Any, k: int, rng: random.Random
) -> list:
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        idx = rng.randint(0, len(reservoir))
        if idx < k:
            reservoir[idx] = item
    return reservoir


# ─── Recurrence counting ──────────────────────────────────────────────────────


def recurrence_count(record: dict) -> int:
    """A single artifact may document multiple sub-session recurrences.

    Honest rule: one artifact = one observation, UNLESS evidence explicitly
    counts sub-session recurrences (iterations, user_rounds_of_pushback,
    occurrences). In that case, count each as an independent SPRT observation.
    """
    ev = record.get("evidence") or {}
    recurrences = 1
    for key in ("user_rounds_of_pushback", "iterations", "occurrences", "times_hit"):
        val = ev.get(key)
        if isinstance(val, int) and val > 1:
            recurrences = max(recurrences, val)
    return recurrences


# ─── Public API functions (importable, not just CLI) ──────────────────────────


def emit(record: dict, state_dir: Path | None = None) -> bool:
    """Append *record* to artifacts.jsonl.

    Returns True on success, False if the opt-in gate is off.
    Raises on IO errors (callers may catch and degrade gracefully).
    """
    if not env_enabled():
        return False
    record = dict(record)
    record.setdefault("ts", iso_now())
    record.setdefault("session_id", os.environ.get("CLAUDE_SESSION_ID", "unknown"))
    record.setdefault("plugin", os.environ.get("ENCHANTED_ATTRIBUTION_PLUGIN", "unknown"))
    path = artifacts_path(state_dir)
    append_jsonl_locked(path, json.dumps(record, ensure_ascii=False))
    return True


def emit_unconditional(record: dict, state_dir: Path | None = None) -> None:
    """Append *record* to artifacts.jsonl regardless of the opt-in gate.

    Used internally by the engine wrapper's post-session flush, which has
    already decided to emit.
    """
    record = dict(record)
    record.setdefault("ts", iso_now())
    record.setdefault("session_id", os.environ.get("CLAUDE_SESSION_ID", "unknown"))
    record.setdefault("plugin", os.environ.get("ENCHANTED_ATTRIBUTION_PLUGIN", "unknown"))
    path = artifacts_path(state_dir)
    append_jsonl_locked(path, json.dumps(record, ensure_ascii=False))


def reconcile(state_dir: Path | None = None) -> dict:
    """Run the full reconcile cycle and write catalog.json atomically.

    Returns the new catalog dict.
    """
    sd = state_dir or _state_dir()
    artifacts = iter_artifacts(sd)
    if not artifacts:
        return load_catalog(sd)

    prior_catalog = load_catalog(sd)
    prior_patterns: dict[str, dict] = prior_catalog.get("patterns", {})
    patterns: dict[str, dict] = {}
    rng = random.Random(42)
    now = datetime.now(timezone.utc)

    for art in artifacts:
        pid = fingerprint(art)
        pat = patterns.get(pid)
        obs = recurrence_count(art)
        art_ts = parse_ts(art.get("ts")) or now

        if pat is None:
            pat = {
                "pattern_id": pid,
                "code": art.get("code", ""),
                "title": art.get("title", ""),
                "category": art.get("category", ""),
                "tags": art.get("tags") or [],
                "first_seen": art.get("ts") or iso_now(),
                "last_seen": art.get("ts") or iso_now(),
                "sessions_seen": [],
                "observations": 0,
                "llr": 0.0,
                "alpha": 1.0,
                "beta": 1.0,
                "reservoir": [],
                "signal": art.get("signal", ""),
                "counter": art.get("counter", ""),
            }
            patterns[pid] = pat

        sid = art.get("session_id", "unknown")
        if sid not in pat["sessions_seen"]:
            pat["sessions_seen"].append(sid)

        pat["observations"] += obs
        pat["llr"] = sprt_update(pat["llr"], obs)
        pat["alpha"], pat["beta"] = beta_update(pat["alpha"], pat["beta"], obs, 0)
        pat["last_seen"] = art.get("ts") or pat["last_seen"]
        if art.get("signal") and not pat["signal"]:
            pat["signal"] = art["signal"]
        if art.get("counter") and not pat["counter"]:
            pat["counter"] = art["counter"]

        reservoir_add(
            pat["reservoir"],
            {"ts": art.get("ts"), "session": sid, "title": art.get("title", "")},
            50,
            rng,
        )

    for pid, pat in patterns.items():
        last = parse_ts(pat["last_seen"]) or now
        days = (now - last).total_seconds() / 86400.0
        pat["days_since_last_seen"] = round(days, 3)
        pat["weight"] = round(ema_weight(days), 4)
        pat["verdict"] = sprt_verdict(pat["llr"])
        pat["posterior_mean"] = round(beta_mean(pat["alpha"], pat["beta"]), 4)
        lo, hi = beta_ci(pat["alpha"], pat["beta"])
        pat["posterior_ci95"] = [round(lo, 4), round(hi, 4)]
        prior = prior_patterns.get(pid) or {}
        if pat["verdict"] == "elevated":
            pat["elevated_at"] = prior.get("elevated_at") or iso_now()
        if pat["verdict"] == "retired":
            pat["retired_at"] = prior.get("retired_at") or iso_now()

    new_catalog: dict[str, Any] = {
        "version": 1,
        "last_reconciled": iso_now(),
        "total_artifacts": len(artifacts),
        "total_patterns": len(patterns),
        "elevated_count": sum(1 for p in patterns.values() if p["verdict"] == "elevated"),
        "retired_count": sum(1 for p in patterns.values() if p["verdict"] == "retired"),
        "patterns": patterns,
    }
    atomic_write_json(catalog_path(sd), new_catalog)
    return new_catalog


def render_briefing(plugin: str, state_dir: Path | None = None) -> Path:
    """Render briefings/<plugin>.md from the catalog. Returns the output path."""
    sd = state_dir or _state_dir()
    catalog = load_catalog(sd)
    patterns = catalog.get("patterns", {})

    CROSS_CUTTING = {"meta", "review"}
    relevant: list[dict] = []
    for pat in patterns.values():
        if pat.get("verdict") != "elevated":
            continue
        tags = [t.lower() for t in pat.get("tags", [])]
        category = (pat.get("category") or "").lower()
        if (
            plugin.lower() == "all"
            or plugin.lower() in tags
            or category in CROSS_CUTTING
        ):
            relevant.append(pat)

    relevant.sort(key=lambda p: p.get("weight", 0), reverse=True)

    BRIEFING_CAP = 10
    total_matching = len(relevant)
    truncated = plugin.lower() == "all" and total_matching > BRIEFING_CAP
    if truncated:
        relevant = relevant[:BRIEFING_CAP]

    bd = briefings_dir(sd)
    bd.mkdir(parents=True, exist_ok=True)
    out = bd / f"{plugin}.md"
    lines = [
        f"# {plugin.title()} Briefing — elevated patterns",
        "",
        f"Rendered: {iso_now()}   ·   Catalog reconciled: {catalog.get('last_reconciled', 'never')}",
        "",
        "*This briefing is machine-generated by `robit/inference/engine.py`.*  ",
        "*Source of truth is `robit/state/inference/catalog.json`.*",
        "",
        "---",
        "",
    ]

    if not relevant:
        lines.append(
            "_No elevated patterns yet. Run reconcile after at least one cross-session recurrence._"
        )
    else:
        header = f"## {len(relevant)} elevated pattern(s)"
        if truncated:
            header = (
                f"## Top {len(relevant)} of {total_matching} elevated patterns "
                "(ranked by EMA-decayed weight)"
            )
        lines.append(header)
        if truncated:
            lines.append("")
            lines.append(
                f"_Truncated to top {BRIEFING_CAP} by weight. Full catalog at "
                "`state/inference/catalog.json`._"
            )
        lines.append("")
        for pat in relevant:
            lines += [
                f"### {pat.get('code', '?')} — {pat.get('title', 'untitled')}",
                "",
                f"- **Weight:** {pat.get('weight', 0):.3f}   "
                f"**Posterior:** {pat.get('posterior_mean', 0):.3f} "
                f"(95% CI {pat.get('posterior_ci95', [0, 0])})   "
                f"**LLR:** {pat.get('llr', 0):.2f}",
                f"- **Observations:** {pat.get('observations', 0)} across "
                f"{len(pat.get('sessions_seen', []))} session(s)   "
                f"**Last seen:** {pat.get('last_seen', '?')[:10]} "
                f"({pat.get('days_since_last_seen', 0):.0f}d ago)",
                f"- **Tags:** `{', '.join(pat.get('tags', []))}`",
                "",
                f"**Signal:** {pat.get('signal', '')}",
                "",
                f"**Counter:** {pat.get('counter', '')}",
                "",
                "---",
                "",
            ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def query(term: str, state_dir: Path | None = None) -> list[dict]:
    """Return catalog patterns matching *term* (code, tag, or pattern_id)."""
    catalog = load_catalog(state_dir)
    hits: list[dict] = []
    for pat in catalog.get("patterns", {}).values():
        if (
            term.lower() == pat.get("pattern_id", "").lower()
            or term.lower() == pat.get("code", "").lower()
            or term.lower() in [t.lower() for t in pat.get("tags", [])]
        ):
            hits.append(pat)
    return hits


def status(state_dir: Path | None = None) -> dict:
    """Return a status summary dict (same shape as CLI output)."""
    catalog = load_catalog(state_dir)
    patterns = catalog.get("patterns", {})
    verdicts: dict[str, int] = {"elevated": 0, "noise": 0, "retired": 0}
    for pat in patterns.values():
        v = pat.get("verdict", "noise")
        verdicts[v] = verdicts.get(v, 0) + 1
    return {
        "enabled": env_enabled(),
        "last_reconciled": catalog.get("last_reconciled"),
        "total_artifacts": catalog.get("total_artifacts", 0),
        "total_patterns": catalog.get("total_patterns", 0),
        "verdicts": verdicts,
        "state_dir": str(state_dir or _state_dir()),
    }


# ─── Subcommand wrappers (thin CLI shims over the public API functions) ───────


def cmd_emit(args: list[str]) -> int:
    if not env_enabled():
        sys.stderr.write(
            "[inference-engine] ROBIT_INFERENCE_ENABLED!=1 — emit skipped\n"
        )
        return 0
    if not args:
        sys.stderr.write("usage: inference-engine emit <record.json|->\n")
        return 2
    src = args[0]
    raw = sys.stdin.read() if src == "-" else Path(src).read_text(encoding="utf-8")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[inference-engine] invalid JSON: {e}\n")
        return 2

    record.setdefault("ts", iso_now())
    record.setdefault("session_id", os.environ.get("CLAUDE_SESSION_ID", "unknown"))
    record.setdefault("plugin", os.environ.get("ENCHANTED_ATTRIBUTION_PLUGIN", "unknown"))
    path = artifacts_path()
    append_jsonl_locked(path, json.dumps(record, ensure_ascii=False))
    print(f"emitted {record.get('code', '?')} -> {path.name}")
    return 0


def cmd_backfill(args: list[str]) -> int:
    if not args:
        sys.stderr.write("usage: inference-engine backfill <source.jsonl>\n")
        return 2
    src = Path(args[0])
    if not src.exists():
        sys.stderr.write(f"source not found: {src}\n")
        return 2
    count = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        record.setdefault("ts", record.get("date", iso_now()))
        record.setdefault("session_id", record.get("source_session", "backfill"))
        record.setdefault("plugin", record.get("scope", "unknown"))
        path = artifacts_path()
        append_jsonl_locked(path, json.dumps(record, ensure_ascii=False))
        count += 1
    print(f"backfilled {count} records from {src.name}")
    return 0


def cmd_reconcile(args: list[str]) -> int:
    cat = reconcile()
    if not cat.get("patterns"):
        sys.stderr.write("[inference-engine] no artifacts to reconcile\n")
        return 0
    print(
        f"reconciled {cat['total_artifacts']} artifacts -> "
        f"{cat['total_patterns']} patterns "
        f"({cat['elevated_count']} elevated, {cat['retired_count']} retired)"
    )
    return 0


def cmd_render_briefing(args: list[str]) -> int:
    if not args:
        sys.stderr.write("usage: inference-engine render-briefing <plugin>\n")
        return 2
    out = render_briefing(args[0])
    print(f"rendered {out}")
    return 0


def cmd_query(args: list[str]) -> int:
    if not args:
        sys.stderr.write("usage: inference-engine query <code|tag|pattern_id>\n")
        return 2
    hits = query(args[0])
    print(json.dumps(hits, indent=2, ensure_ascii=False))
    return 0 if hits else 1


def cmd_status(_args: list[str]) -> int:
    print(json.dumps(status(), indent=2))
    return 0


# ─── Dispatch ─────────────────────────────────────────────────────────────────

COMMANDS: dict[str, Any] = {
    "emit": cmd_emit,
    "reconcile": cmd_reconcile,
    "render-briefing": cmd_render_briefing,
    "query": cmd_query,
    "backfill": cmd_backfill,
    "status": cmd_status,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        sys.stderr.write(__doc__ or "")
        return 0 if len(argv) >= 2 else 2
    cmd = argv[1]
    if cmd not in COMMANDS:
        sys.stderr.write(f"unknown subcommand: {cmd}\n")
        sys.stderr.write(f"available: {', '.join(COMMANDS)}\n")
        return 2
    return COMMANDS[cmd](argv[2:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
