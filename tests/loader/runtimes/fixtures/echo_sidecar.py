"""Mock sidecar — reads newline-framed JSON-RPC from stdin, writes canned responses.

Behavior is selected by argv flags so a single fixture covers every test case:

  --name <s>          : name returned from initialize (default "echo-sidecar")
  --phases <csv>      : phases returned from initialize (default "trust-gate")
  --required          : initialize.required = true (default false)
  --budget-tier <s>   : default "always"
  --subscribes <csv>  : default ""
  --emits <csv>       : default ""

  --mode ack          : on_phase → status=ack (default)
  --mode veto         : on_phase → status=veto, reason="echoed-veto"
  --mode derive       : on_phase → echoes the input event as a derived event
  --mode hang         : on_phase → never responds (forces timeout)
  --mode crash        : on_phase → exit(1) before responding
  --mode big          : on_phase → response body > 8 MiB (cap-trip)
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="echo-sidecar")
    p.add_argument("--phases", default="trust-gate")
    p.add_argument("--required", action="store_true")
    p.add_argument("--budget-tier", default="always")
    p.add_argument("--subscribes", default="")
    p.add_argument("--emits", default="")
    p.add_argument(
        "--mode",
        default="ack",
        choices=("ack", "veto", "derive", "hang", "crash", "big", "bad-json"),
    )
    return p


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _build_parser().parse_args()

    phases = [p for p in args.phases.split(",") if p]
    subs = [s for s in args.subscribes.split(",") if s]
    emits = [s for s in args.emits.split(",") if s]

    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "shutdown":
            # Notification — no response, just exit.
            return 0

        if method == "initialize":
            _write({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "name": args.name,
                    "phases": phases,
                    "required": args.required,
                    "budget_tier": args.budget_tier,
                    "topics": {"subscribes": subs, "emits": emits},
                },
            })
            continue

        if method == "on_phase":
            event = (msg.get("params") or {}).get("event") or {}

            if args.mode == "ack":
                _write({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "status": "ack",
                    "reason": None,
                    "derived_events": [],
                }})
            elif args.mode == "veto":
                _write({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "status": "veto",
                    "reason": "echoed-veto",
                    "derived_events": [],
                }})
            elif args.mode == "derive":
                # Echo input event as derived (with the source rewritten).
                derived = dict(event)
                derived["source"] = args.name
                derived["id"] = (event.get("id") or "evt") + "-derived"
                _write({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "status": "ack",
                    "reason": "echoed",
                    "derived_events": [derived],
                }})
            elif args.mode == "hang":
                # Block forever; parent will time out and SIGKILL us.
                time.sleep(60.0)
                return 0
            elif args.mode == "crash":
                # Die without responding.
                sys.stderr.write("crashing on purpose\n")
                sys.stderr.flush()
                return 1
            elif args.mode == "big":
                # Emit a response whose JSON body exceeds 8 MiB.
                blob = "x" * (9 * 1024 * 1024)
                _write({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "status": "ack", "reason": blob, "derived_events": [],
                }})
            elif args.mode == "bad-json":
                # Malformed line — not valid JSON.
                sys.stdout.write("not-json-{\n")
                sys.stdout.flush()
            continue

        # Unknown method — protocol error response.
        _write({"jsonrpc": "2.0", "id": msg_id, "error": {
            "code": -32601, "message": f"unknown method {method!r}",
        }})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
