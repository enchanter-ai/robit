"""deep_research.artifacts — schemas and writers for claims.json, sources.jsonl, trace.json.

All writes are atomic (write-to-tmp then rename) to avoid partial-file reads
across the 6-phase pipeline.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema dataclasses (match SKILL.md Phase-5 spec exactly)
# ---------------------------------------------------------------------------


@dataclass
class SubQuestion:
    id: str
    question: str
    acceptance: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question, "acceptance": self.acceptance}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubQuestion":
        return cls(id=d["id"], question=d["question"], acceptance=d.get("acceptance", ""))


@dataclass
class Claim:
    id: str
    claim: str
    sq: str
    supporting: list[str]
    independent_count: int
    confidence: str  # "high" | "medium" | "low"
    contradicts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "sq": self.sq,
            "supporting": self.supporting,
            "independent_count": self.independent_count,
            "confidence": self.confidence,
            "contradicts": self.contradicts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        return cls(
            id=d["id"],
            claim=d["claim"],
            sq=d.get("sq", ""),
            supporting=d.get("supporting", []),
            independent_count=d.get("independent_count", 0),
            confidence=d.get("confidence", "low"),
            contradicts=d.get("contradicts"),
        )


@dataclass
class Source:
    id: str
    url: str
    date: str | None
    source_type: str
    findings: list[dict[str, str]]
    error: str | None = None
    sub_question_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "url": self.url,
        }
        if self.error:
            d["error"] = self.error
        else:
            d["date"] = self.date
            d["source_type"] = self.source_type
            d["findings"] = self.findings
        if self.sub_question_id:
            d["sub_question_id"] = self.sub_question_id
        return d


@dataclass
class ClaimsDoc:
    """The top-level claims.json structure per SKILL.md Phase 5."""

    topic: str
    generated: str
    freshness: str
    triangulation_score: float
    verdict: str  # "READY" | "PARTIAL" | "FAIL"
    source_count: int
    claims: list[Claim]
    unresolved_contradictions: list[dict[str, Any]]
    coverage_gaps: list[str]
    sub_questions: list[SubQuestion]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "generated": self.generated,
            "freshness": self.freshness,
            "triangulation_score": self.triangulation_score,
            "verdict": self.verdict,
            "source_count": self.source_count,
            "claims": [c.to_dict() for c in self.claims],
            "unresolved_contradictions": self.unresolved_contradictions,
            "coverage_gaps": self.coverage_gaps,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
        }


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically: write to a tmp file, then rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_jsonl(path: Path, lines: list[Any]) -> None:
    """Write JSONL atomically — each element on its own line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for item in lines:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_claims(path: Path, doc: ClaimsDoc) -> None:
    """Write claims.json atomically."""
    _atomic_write_json(path, doc.to_dict())


def write_sources(path: Path, sources: list[Source]) -> None:
    """Write sources.jsonl atomically — one source per line."""
    _atomic_write_jsonl(path, [s.to_dict() for s in sources])


def write_trace(path: Path, trace: dict[str, Any]) -> None:
    """Write trace.json atomically."""
    _atomic_write_json(path, trace)


# ---------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------


def read_claims(path: Path) -> ClaimsDoc:
    """Load claims.json into a ClaimsDoc."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ClaimsDoc(
        topic=data["topic"],
        generated=data["generated"],
        freshness=data["freshness"],
        triangulation_score=data["triangulation_score"],
        verdict=data["verdict"],
        source_count=data["source_count"],
        claims=[Claim.from_dict(c) for c in data.get("claims", [])],
        unresolved_contradictions=data.get("unresolved_contradictions", []),
        coverage_gaps=data.get("coverage_gaps", []),
        sub_questions=[SubQuestion.from_dict(sq) for sq in data.get("sub_questions", [])],
    )


def read_sources(path: Path) -> list[Source]:
    """Load sources.jsonl into a list of Source objects."""
    sources: list[Source] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sources.append(
                Source(
                    id=d.get("id", ""),
                    url=d.get("url", ""),
                    date=d.get("date"),
                    source_type=d.get("source_type", "other"),
                    findings=d.get("findings", []),
                    error=d.get("error"),
                    sub_question_id=d.get("sub_question_id"),
                )
            )
    return sources


def today_str() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()
