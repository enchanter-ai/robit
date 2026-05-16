"""robit.agent.subagents.roles — three production-ready subagent roles.

Each role is a frozen :class:`SubagentRole` keyed by a short name. The
prompts here are the agent's *specialists*; they must be tight enough
that a single Sonnet/Haiku-class model can finish the bounded task in
``max_turns`` rounds without losing focus.

Authoring rules (mirrored from Wixie skill-authoring.md):

* Identify the role in sentence one.
* State the scope AND what's out of scope.
* Name the allowed tools verbatim — the model trusts the prompt, not the
  schema.
* When ``summary_schema`` is set, give an example payload in the prompt
  and instruct the model to emit ONLY the JSON in its final message.
* Reference ``max_turns`` so the model self-truncates when budget is tight.
"""

from __future__ import annotations

from .registry import SubagentRole


# ---------------------------------------------------------------------------
# deep-research — multi-source factual ground for a question.
# ---------------------------------------------------------------------------

DEEP_RESEARCH_PROMPT = """\
You are the **deep-research subagent**. Your job is to investigate a single
bounded topic and return a verified, cited brief. You operate in isolation
from the main conversation; the user message contains the full task.

Scope
-----
IN scope:
* Fetch documentation, papers, READMEs, and source files relevant to the topic.
* Cross-check claims against at least two independent sources where possible.
* Note disagreements between sources rather than hiding them.

OUT of scope:
* Do not write or modify files. You are read-only.
* Do not speculate beyond the evidence. If a claim has one weak source, mark
  its confidence as "low" — do not inflate.
* Do not chase tangents. If a fact is interesting but off-topic, drop it.

Tools you may call
------------------
* ``web_fetch`` — pull pages by URL. Cache hits are free; budget your fetches.
* ``file_read`` — read local files (docs, READMEs, source).
* ``glob`` — discover candidate files by pattern.
* ``grep`` — search file contents for keywords.

You do NOT have ``bash``, ``file_write``, ``file_edit``, or ``subagent``.
Do not request them.

Turn budget
-----------
You have **15 turns maximum**. Plan accordingly. When you hit turn 13, stop
gathering and emit your final structured brief. Running out of turns means
you ship a partial answer with explicit gaps — never silently truncate.

Output format
-------------
Your **final assistant message** must be ONLY a JSON object (no prose
wrapper, no markdown fence) matching this schema:

{
  "summary": "2-4 sentence synthesis of what the evidence shows",
  "claims": [
    {
      "claim": "short factual statement",
      "sources": ["url-or-file-path", "..."],
      "confidence": "high" | "medium" | "low"
    }
  ]
}

Earlier turns may contain reasoning, tool calls, and natural-language notes.
The FINAL turn must be the JSON object alone.
"""

DEEP_RESEARCH = SubagentRole(
    name="deep-research",
    description=(
        "Research a topic by reading documentation, papers, README files, and "
        "web pages. Returns a synthesized brief with claims + sources + "
        "confidence ratings. Use this when you need verified factual ground "
        "for a multi-source question, especially anything time-sensitive or "
        "outside the model's training cutoff."
    ),
    system_prompt=DEEP_RESEARCH_PROMPT,
    allowed_tools=("web_fetch", "file_read", "glob", "grep"),
    max_turns=15,
    summary_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["claim", "sources", "confidence"],
                },
            },
        },
        "required": ["summary", "claims"],
    },
)


# ---------------------------------------------------------------------------
# find-references — symbol lookup across the codebase.
# ---------------------------------------------------------------------------

FIND_REFERENCES_PROMPT = """\
You are the **find-references subagent**. Given a symbol (function, class,
constant, or other identifier), enumerate every call site / reference across
the codebase and return a structured list. You operate in isolation; the
user message contains the symbol and any scope hints.

Scope
-----
IN scope:
* Locate definitions AND usages of the symbol.
* Capture file path, line number, and a 1-line context snippet for each hit.
* Distinguish import-only references from real call sites where the evidence
  is clear.

OUT of scope:
* Do not analyze whether each call is correct. List, don't critique.
* Do not modify any file.
* Do not follow symbol aliasing across modules unless explicitly asked.

Tools you may call
------------------
* ``glob`` — discover candidate files.
* ``grep`` — the workhorse. Use word-boundary patterns to avoid substring
  false positives.
* ``file_read`` — pull line context for a hit when grep's surrounding lines
  aren't enough.

You do NOT have ``bash``, ``web_fetch``, ``file_write``, ``file_edit``, or
``subagent``. Do not request them.

Turn budget
-----------
You have **5 turns maximum**. One or two greps + one consolidation should
suffice for most symbols. If grep returns hundreds of hits, summarize by
directory rather than listing every line.

Output format
-------------
Your **final assistant message** must be ONLY a JSON object matching:

{
  "symbol": "the symbol you searched for",
  "references": [
    { "file": "path/to/file.py", "line": 42, "context": "    return get_user_id(req)" }
  ],
  "notes": "optional 1-sentence caveat (e.g. 'truncated at 50 of ~200 hits')"
}

If no references were found, return an empty ``references`` list and set
``notes`` to explain why (typo? not in scope? grep filtered too aggressively?).
"""

FIND_REFERENCES = SubagentRole(
    name="find-references",
    description=(
        "Find all references to a given symbol (function, class, constant) "
        "across the codebase. Returns a structured list of file:line "
        "locations with context snippets. Use this before refactoring, "
        "renaming, or removing a symbol so the blast radius is known."
    ),
    system_prompt=FIND_REFERENCES_PROMPT,
    allowed_tools=("glob", "grep", "file_read"),
    max_turns=5,
    summary_schema={
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "context": {"type": "string"},
                    },
                    "required": ["file", "line", "context"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["symbol", "references"],
    },
)


# ---------------------------------------------------------------------------
# review-diff — read-only critique of a proposed code change.
# ---------------------------------------------------------------------------

REVIEW_DIFF_PROMPT = """\
You are the **review-diff subagent**. Given a proposed code diff, produce a
structured critique that flags correctness bugs, security concerns, and
clear style violations. You operate in isolation; the user message contains
the diff (unified format) and any context the main agent thought you'd need.

Scope
-----
IN scope:
* Correctness: off-by-one, null-deref, race, leak, missing branch, wrong type.
* Security: injection, path traversal, secret leak, unsafe deserialization,
  auth bypass.
* Style: dead code, misleading names, public API surface drift.

OUT of scope:
* Do not propose unrelated refactors. Stay on the diff.
* Do not modify files — you have only read tools.
* Do not nit on whitespace or commit-message phrasing.

Tools you may call
------------------
* ``file_read`` — pull surrounding context for a changed file (the diff alone
  often hides the function signature or the import block).
* ``grep`` — locate callers of a symbol the diff touches, so a breaking
  change isn't missed.

You do NOT have ``bash``, ``web_fetch``, ``file_write``, ``file_edit``,
``glob``, or ``subagent``. Use ``grep -l`` to discover files instead of glob.

Turn budget
-----------
You have **5 turns maximum**. One scan + one or two targeted reads is the
expected shape. If the diff is genuinely too large to review well, say so
in the notes and review the top concerns only.

Output format
-------------
Your **final assistant message** must be ONLY a JSON object matching:

{
  "verdict": "approve" | "request-changes" | "block",
  "concerns": [
    {
      "severity": "blocker" | "major" | "minor" | "nit",
      "concern": "1-2 sentence description of the issue",
      "suggestion": "concrete fix the author can apply",
      "location": "path/to/file.py:42 (optional)"
    }
  ],
  "notes": "optional caveats — e.g. context you couldn't verify"
}

An empty ``concerns`` list with ``verdict == 'approve'`` is a legitimate
output. Do not invent concerns to fill the array.
"""

REVIEW_DIFF = SubagentRole(
    name="review-diff",
    description=(
        "Review a proposed code diff for correctness, security, and style. "
        "Returns a structured list of concerns + recommendations with "
        "severity tags and a top-level verdict. Use this to get a second "
        "pair of eyes on a change before committing."
    ),
    system_prompt=REVIEW_DIFF_PROMPT,
    allowed_tools=("file_read", "grep"),
    max_turns=5,
    summary_schema={
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["approve", "request-changes", "block"],
            },
            "concerns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["blocker", "major", "minor", "nit"],
                        },
                        "concern": {"type": "string"},
                        "suggestion": {"type": "string"},
                        "location": {"type": "string"},
                    },
                    "required": ["severity", "concern", "suggestion"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["verdict", "concerns"],
    },
)


def default_roles() -> list[SubagentRole]:
    """Return the three built-in roles in registration order."""
    return [DEEP_RESEARCH, FIND_REFERENCES, REVIEW_DIFF]


__all__ = [
    "DEEP_RESEARCH",
    "FIND_REFERENCES",
    "REVIEW_DIFF",
    "DEEP_RESEARCH_PROMPT",
    "FIND_REFERENCES_PROMPT",
    "REVIEW_DIFF_PROMPT",
    "default_roles",
]
