"""enchanter.insights_cli — command-line interface for the enchanter-agent runtime.

Entry point declared in pyproject.toml::

    [project.scripts]
    enchanter-insights = "enchanter.insights_cli:main"

The ``enchanter`` binary name is reserved for the 0.5.0 coding-agent CLI
(Wave 15.0+). This module owns the inspection surface only.

Subcommand tree::

    enchanter-insights version
    enchanter-insights status [--json]
    enchanter-insights engines list [--json]
    enchanter-insights engines show <name> [--json]
    enchanter-insights conduct list [--json]
    enchanter-insights conduct show <name>
    enchanter-insights inference status [--json]
    enchanter-insights inference reconcile
    enchanter-insights tier route <task_class>
    enchanter-insights serve [--stdio | --http HOST:PORT | --proxy HOST:PORT]
                             [--path PATH] [--accept LIST] [--no-conduct]

Exit codes:
    0  success
    1  user error (bad args, unknown engine/conduct module)
    2  runtime error (failed to load registry, unexpected exception)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import enchanter
from enchanter.insights_cli.format import (
    aggregate_status_json,
    aggregate_status_text,
    conduct_list_json,
    conduct_list_table,
    engine_show_json,
    engine_show_text,
    engines_list_json,
    engines_list_table,
    inference_status_json,
    inference_status_text,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write(text: str) -> None:
    """Write text to stdout, tolerating Windows cp1252 consoles."""
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))


def _err(text: str) -> None:
    try:
        sys.stderr.write(text + "\n")
    except UnicodeEncodeError:
        sys.stderr.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def _load_registry():
    """Load the engine registry, returning (registry_dict, None) or (None, exit_code)."""
    try:
        from enchanter.loader import load_engine_registry
        return load_engine_registry(), None
    except Exception as exc:  # noqa: BLE001
        _err(f"error: failed to load engine registry: {exc}")
        return None, 2


def _load_conduct_rules():
    """Load conduct rules, returning (rules, None) or (None, exit_code)."""
    try:
        from enchanter.conduct import load_conduct
        return load_conduct(), None
    except Exception as exc:  # noqa: BLE001
        _err(f"error: failed to load conduct modules: {exc}")
        return None, 2


def _manifest_to_dict(manifest) -> dict[str, Any]:
    """Convert an EngineManifest dataclass to a plain dict for serialisation."""
    topics = manifest.topics
    return {
        "name": manifest.name,
        "description": manifest.description,
        "version": manifest.version,
        "phases": list(manifest.phases),
        "required": manifest.required,
        "budget_tier": manifest.budget_tier,
        "adapter": manifest.adapter,
        "topics": {
            "subscribes": list(topics.subscribes),
            "emits": list(topics.emits),
        },
        "depends_on": list(manifest.depends_on),
        "tags": list(manifest.tags),
        "manifest_path": manifest.manifest_path,
    }


def _rule_to_dict(rule) -> dict[str, Any]:
    """Convert a ConductRule dataclass to a plain dict for serialisation."""
    return {
        "name": rule.name,
        "package": rule.package,
        "enforcement": rule.enforcement,
        "tags": list(rule.tags),
        "path": str(rule.path),
    }


# ─── Command handlers ─────────────────────────────────────────────────────────


def cmd_version(args: argparse.Namespace) -> int:
    _write(f"enchanter-agent {enchanter.__version__}\n")
    return 0


# -- engines --

def cmd_engines_list(args: argparse.Namespace) -> int:
    registry, err = _load_registry()
    if err is not None:
        return err

    # registry is dict[name, adapter]; we need the manifests for metadata.
    # Re-parse to get EngineManifest objects (they have the rich metadata).
    try:
        from enchanter.loader.discovery import _default_root, find_engine_manifests
        from enchanter.loader.manifest import parse_manifest

        root = _default_root()
        manifest_paths = find_engine_manifests(root)
        manifests = [parse_manifest(p) for p in manifest_paths]
    except Exception as exc:  # noqa: BLE001
        _err(f"error: failed to parse engine manifests: {exc}")
        return 2

    dicts = [_manifest_to_dict(m) for m in manifests]

    if args.json:
        _write(engines_list_json(dicts))
    else:
        _write(engines_list_table(dicts))
    return 0


def cmd_engines_show(args: argparse.Namespace) -> int:
    try:
        from enchanter.loader.discovery import _default_root, find_engine_manifests
        from enchanter.loader.manifest import parse_manifest

        root = _default_root()
        manifest_paths = find_engine_manifests(root)
        manifests = {parse_manifest(p).name: parse_manifest(p) for p in manifest_paths}
    except Exception as exc:  # noqa: BLE001
        _err(f"error: failed to parse engine manifests: {exc}")
        return 2

    name: str = args.name
    if name not in manifests:
        known = sorted(manifests.keys())
        _err(f"error: engine '{name}' not found. Known engines: {', '.join(known)}")
        return 1

    d = _manifest_to_dict(manifests[name])
    if args.json:
        _write(engine_show_json(d))
    else:
        _write(engine_show_text(d))
    return 0


# -- conduct --

def cmd_conduct_list(args: argparse.Namespace) -> int:
    rules, err = _load_conduct_rules()
    if err is not None:
        return err

    dicts = [_rule_to_dict(r) for r in rules]
    if args.json:
        _write(conduct_list_json(dicts))
    else:
        _write(conduct_list_table(dicts))
    return 0


def cmd_conduct_show(args: argparse.Namespace) -> int:
    rules, err = _load_conduct_rules()
    if err is not None:
        return err

    name: str = args.name
    matched = [r for r in rules if r.name == name]
    if not matched:
        known = sorted(r.name for r in rules)
        _err(f"error: conduct module '{name}' not found. Known modules: {', '.join(known)}")
        return 1

    rule = matched[0]
    _write(rule.body)
    if not rule.body.endswith("\n"):
        _write("\n")
    return 0


# -- inference --

def cmd_inference_status(args: argparse.Namespace) -> int:
    try:
        from enchanter.inference.engine import status
        s = status()
    except Exception as exc:  # noqa: BLE001
        _err(f"error: inference status failed: {exc}")
        return 2

    if args.json:
        _write(inference_status_json(s))
    else:
        _write(inference_status_text(s))
    return 0


def cmd_inference_reconcile(args: argparse.Namespace) -> int:
    try:
        from enchanter.inference.engine import reconcile
        cat = reconcile()
    except Exception as exc:  # noqa: BLE001
        _err(f"error: inference reconcile failed: {exc}")
        return 2

    total_arts = cat.get("total_artifacts", 0)
    total_pats = cat.get("total_patterns", 0)
    elevated = cat.get("elevated_count", 0)
    retired = cat.get("retired_count", 0)
    _write(
        f"reconciled {total_arts} artifacts -> {total_pats} patterns "
        f"({elevated} elevated, {retired} retired)\n"
    )
    return 0


# -- tier --

def cmd_tier_route(args: argparse.Namespace) -> int:
    task_class: str = args.task_class

    try:
        from enchanter.runtime.models_registry import ModelsRegistry
        from enchanter.runtime.tier_router import TierRouter, UnknownTaskClassError

        registry = ModelsRegistry.load()
        router = TierRouter(registry)
        model_id = router.route(task_class)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        # Distinguish user error (unknown task class) from runtime error.
        exc_type = type(exc).__name__
        if "UnknownTaskClass" in exc_type or "ValueError" in exc_type:
            _err(f"error: {exc}")
            return 1
        _err(f"error: tier routing failed: {exc}")
        return 2

    _write(f"{model_id}\n")
    return 0


# -- serve --


def cmd_serve(args: argparse.Namespace) -> int:
    import asyncio as _asyncio

    # --proxy is dispatched to a different runtime (the wire proxy, not MCP).
    if getattr(args, "proxy", None):
        host, _, port_s = args.proxy.rpartition(":")
        if not host or not port_s:
            _err(f"error: --proxy expects HOST:PORT, got {args.proxy!r}")
            return 1
        try:
            port = int(port_s)
        except ValueError:
            _err(f"error: invalid port in --proxy: {port_s!r}")
            return 1

        accept_raw = getattr(args, "accept", "anthropic,openai,gemini")
        accept_set = frozenset(
            tok.strip() for tok in accept_raw.split(",") if tok.strip()
        )
        valid_families = {"anthropic", "openai", "gemini"}
        invalid = accept_set - valid_families
        if invalid:
            _err(
                f"error: --accept contains unknown families: {','.join(sorted(invalid))}. "
                f"Valid: {','.join(sorted(valid_families))}"
            )
            return 1
        if not accept_set:
            _err("error: --accept must list at least one family")
            return 1

        conduct = not getattr(args, "no_conduct", False)

        try:
            from enchanter.proxy import serve_proxy
        except Exception as exc:  # noqa: BLE001
            _err(f"error: enchanter.proxy import failed: {exc}")
            return 2

        try:
            _asyncio.run(
                serve_proxy(
                    host=host,
                    port=port,
                    accept=accept_set,
                    conduct=conduct,
                )
            )
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001
            _err(f"error: serve_proxy failed: {exc}")
            return 2
        return 0

    try:
        from enchanter.mcp_server import MCPServer, serve_http, serve_stdio
    except Exception as exc:  # noqa: BLE001
        _err(f"error: mcp_server import failed: {exc}")
        return 2

    server = MCPServer()

    if args.http:
        host, _, port_s = args.http.rpartition(":")
        if not host or not port_s:
            _err(f"error: --http expects HOST:PORT, got {args.http!r}")
            return 1
        try:
            port = int(port_s)
        except ValueError:
            _err(f"error: invalid port in --http: {port_s!r}")
            return 1
        try:
            _asyncio.run(serve_http(server, host=host, port=port, path=args.path))
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001
            _err(f"error: serve_http failed: {exc}")
            return 2
        return 0

    # Default: stdio
    try:
        _asyncio.run(serve_stdio(server))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001
        _err(f"error: serve_stdio failed: {exc}")
        return 2
    return 0


# -- status --

def cmd_status(args: argparse.Namespace) -> int:
    version = enchanter.__version__

    # Engines count.
    try:
        from enchanter.loader import load_engine_registry
        registry = load_engine_registry()
        engine_count = len(registry)
    except Exception:  # noqa: BLE001
        engine_count = -1

    # Conduct count.
    try:
        from enchanter.conduct import load_conduct
        conduct_rules = load_conduct()
        conduct_count = len(conduct_rules)
    except Exception:  # noqa: BLE001
        conduct_count = -1

    # Inference status.
    try:
        from enchanter.inference.engine import status
        inf = status()
    except Exception:  # noqa: BLE001
        inf = {
            "enabled": False,
            "last_reconciled": None,
            "total_artifacts": 0,
            "total_patterns": 0,
            "verdicts": {},
        }

    # Tier defaults.
    tier_defaults: dict[str, str] = {}
    try:
        from enchanter.runtime.models_registry import ModelsRegistry
        from enchanter.runtime.tier_router import TierRouter, _VALID_TASK_CLASSES

        reg = ModelsRegistry.load()
        router = TierRouter(reg)
        for tc in sorted(_VALID_TASK_CLASSES):
            try:
                tier_defaults[tc] = router.route(tc)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                tier_defaults[tc] = "?"
    except Exception:  # noqa: BLE001
        tier_defaults = {}

    # Fast-path bypass status (proxy mode).
    try:
        from enchanter.proxy import fastpath
        fp_cfg = fastpath.load_config(force_reload=True)
        fastpath_status = {
            "enabled": fp_cfg.enabled,
            "allowed_keys": len(fp_cfg.allowed_key_hashes),
            "allowed_models": (
                sorted(fp_cfg.allowed_models) if fp_cfg.allowed_models is not None else None
            ),
        }
    except Exception:  # noqa: BLE001
        fastpath_status = {"enabled": False, "allowed_keys": 0, "allowed_models": None}

    data: dict[str, Any] = {
        "version": version,
        "engine_count": engine_count,
        "conduct_count": conduct_count,
        "inference": inf,
        "tier_defaults": tier_defaults,
        "fastpath": fastpath_status,
    }

    if args.json:
        _write(aggregate_status_json(data))
    else:
        _write(aggregate_status_text(data))
    return 0


# ─── Argument parser ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enchanter-insights",
        description="enchanter-agent runtime inspection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers(dest="noun", metavar="<noun>")

    # ── version ──────────────────────────────────────────────────────────────
    subparsers.add_parser("version", help="Print enchanter-agent version")

    # ── status ────────────────────────────────────────────────────────────────
    p_status = subparsers.add_parser(
        "status", help="Aggregate status: engines, conduct, inference, tier defaults"
    )
    p_status.add_argument("--json", action="store_true", help="Output as JSON")

    # ── engines ────────────────────────────────────────────────────────────────
    p_engines = subparsers.add_parser("engines", help="Engine registry commands")
    engines_sub = p_engines.add_subparsers(dest="verb", metavar="<verb>")
    p_engines.set_defaults(func=None)

    p_eng_list = engines_sub.add_parser("list", help="List all discovered engines")
    p_eng_list.add_argument("--json", action="store_true", help="Output as JSON")
    p_eng_list.set_defaults(func=cmd_engines_list)

    p_eng_show = engines_sub.add_parser("show", help="Show details for one engine")
    p_eng_show.add_argument("name", help="Engine name (e.g. destructive-op-gate)")
    p_eng_show.add_argument("--json", action="store_true", help="Output as JSON")
    p_eng_show.set_defaults(func=cmd_engines_show)

    # ── conduct ────────────────────────────────────────────────────────────────
    p_conduct = subparsers.add_parser("conduct", help="Conduct module commands")
    conduct_sub = p_conduct.add_subparsers(dest="verb", metavar="<verb>")
    p_conduct.set_defaults(func=None)

    p_con_list = conduct_sub.add_parser("list", help="List all conduct modules")
    p_con_list.add_argument("--json", action="store_true", help="Output as JSON")
    p_con_list.set_defaults(func=cmd_conduct_list)

    p_con_show = conduct_sub.add_parser("show", help="Print a conduct module's body")
    p_con_show.add_argument("name", help="Conduct module name (e.g. discipline)")
    p_con_show.set_defaults(func=cmd_conduct_show)

    # ── inference ──────────────────────────────────────────────────────────────
    p_infer = subparsers.add_parser("inference", help="Inference substrate commands")
    infer_sub = p_infer.add_subparsers(dest="verb", metavar="<verb>")
    p_infer.set_defaults(func=None)

    p_inf_status = infer_sub.add_parser("status", help="Show inference catalog summary")
    p_inf_status.add_argument("--json", action="store_true", help="Output as JSON")
    p_inf_status.set_defaults(func=cmd_inference_status)

    p_inf_rec = infer_sub.add_parser("reconcile", help="Run inference reconcile cycle")
    p_inf_rec.set_defaults(func=cmd_inference_reconcile)

    # ── serve ──────────────────────────────────────────────────────────────────
    p_serve = subparsers.add_parser(
        "serve",
        help="Run enchanter as an MCP server (stdio/HTTP) or as a wire proxy",
    )
    serve_mode = p_serve.add_mutually_exclusive_group()
    serve_mode.add_argument(
        "--stdio",
        action="store_true",
        help="Speak MCP over stdin/stdout (default).",
    )
    serve_mode.add_argument(
        "--http",
        metavar="HOST:PORT",
        help="Listen on HOST:PORT and serve Streamable-HTTP MCP.",
    )
    serve_mode.add_argument(
        "--proxy",
        metavar="HOST:PORT",
        help=(
            "Listen on HOST:PORT and serve the wire-format proxy "
            "(Anthropic/OpenAI/Gemini endpoints fronted by the enchanter "
            "engine bus)."
        ),
    )
    p_serve.add_argument(
        "--path",
        default="/mcp",
        help="HTTP path the MCP server binds to (default: /mcp). Ignored by --proxy.",
    )
    p_serve.add_argument(
        "--accept",
        default="anthropic,openai,gemini",
        help=(
            "Comma-separated proxy families to accept. Adapters whose "
            "family is omitted respond 404. (Default: "
            "anthropic,openai,gemini.) Only used with --proxy."
        ),
    )
    p_serve.add_argument(
        "--no-conduct",
        action="store_true",
        help=(
            "Disable enchanter-conduct injection on proxied requests. "
            "Only used with --proxy."
        ),
    )
    p_serve.set_defaults(func=cmd_serve)

    # ── tier ───────────────────────────────────────────────────────────────────
    p_tier = subparsers.add_parser("tier", help="Tier router commands")
    tier_sub = p_tier.add_subparsers(dest="verb", metavar="<verb>")
    p_tier.set_defaults(func=None)

    p_tier_route = tier_sub.add_parser(
        "route", help="Resolve model_id for a task class"
    )
    p_tier_route.add_argument(
        "task_class",
        help="One of: orchestrator, executor, validator, image, embed",
    )
    p_tier_route.set_defaults(func=cmd_tier_route)

    return parser


# ─── Dispatch helpers ─────────────────────────────────────────────────────────

# Map top-level nouns that are noun-only (no verb) to their handler.
_NOUN_ONLY_HANDLERS = {
    "version": cmd_version,
    "status": cmd_status,
    "serve": cmd_serve,
}

# Map (noun, verb) pairs to their handler (for noun+verb subcommands without
# func set on the namespace, as a fallback).
_NOUN_VERB_HANDLERS: dict[tuple[str, str], Any] = {
    ("engines", "list"): cmd_engines_list,
    ("engines", "show"): cmd_engines_show,
    ("conduct", "list"): cmd_conduct_list,
    ("conduct", "show"): cmd_conduct_show,
    ("inference", "status"): cmd_inference_status,
    ("inference", "reconcile"): cmd_inference_reconcile,
    ("tier", "route"): cmd_tier_route,
}


# ─── Entry point ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* and dispatch to the appropriate handler.

    Returns an integer exit code (0 success, 1 user error, 2 runtime error).
    When *argv* is None, ``sys.argv[1:]`` is used.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    noun: str | None = getattr(args, "noun", None)

    # No noun → print help and exit 1.
    if not noun:
        parser.print_help()
        return 1

    # Noun-only commands (version, status).
    if noun in _NOUN_ONLY_HANDLERS:
        return _NOUN_ONLY_HANDLERS[noun](args)

    # Noun+verb commands.
    verb: str | None = getattr(args, "verb", None)
    if not verb:
        # User typed `enchanter-insights engines` without a verb — print sub-help.
        # Find the subparser for this noun and print its help.
        # argparse doesn't expose sub-parsers easily; use a manual lookup.
        _print_subcommand_help(parser, noun)
        return 1

    # Try the func attribute first (set by set_defaults).
    handler = getattr(args, "func", None)
    if handler is not None:
        return handler(args)

    # Fallback via the lookup table.
    handler = _NOUN_VERB_HANDLERS.get((noun, verb))
    if handler is not None:
        return handler(args)

    _err(f"error: unknown subcommand '{noun} {verb}'")
    parser.print_help()
    return 1


def _print_subcommand_help(parser: argparse.ArgumentParser, noun: str) -> None:
    """Print help for a specific noun's subcommands."""
    # Walk the parser's subparsers to find the right one.
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub = action.choices.get(noun)
            if sub is not None:
                sub.print_help()
                return
    parser.print_help()
