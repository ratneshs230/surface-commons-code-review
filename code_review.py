"""Code Review — Commons guild tool.

Replaces Codium AI / GitHub Copilot review tier. Three substantive endpoints:

  POST /api/v1/commons/code-review/diff          — git diff -> review JSON
  POST /api/v1/commons/code-review/file          — single file -> review JSON
  POST /api/v1/commons/code-review/explain-change — diff -> human prose

Plus GET /api/v1/commons/code-review/health (public).

All review work goes through route_think ("fast" tier, claude_cli substrate
as universal fallback). Auth + billing identical to other Commons routers.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.service_auth import require_service_key_and_agent
from app.models.agent import Agent
from app.services.ledger import (
    InsufficientFundsError,
    record_transaction,
)
from app.services.router_service import route_think


_log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/commons/code-review",
    tags=["commons-code-review"],
)


# ── Pricing & limits ──────────────────────────────────────────────────────

SC_COST = {
    "diff":            float(os.environ.get("CODE_REVIEW_SC_DIFF",    "15")),
    "file":            float(os.environ.get("CODE_REVIEW_SC_FILE",    "10")),
    "explain_change":  float(os.environ.get("CODE_REVIEW_SC_EXPLAIN", "8")),
}

MAX_DIFF_CHARS = int(os.environ.get("CODE_REVIEW_MAX_DIFF_CHARS", str(200_000)))
MAX_FILE_CHARS = int(os.environ.get("CODE_REVIEW_MAX_FILE_CHARS", str(120_000)))

LANGUAGES = ("python", "javascript", "typescript", "go", "rust")
FOCUS_AREAS = ("security", "perf", "style", "all")

SEVERITIES = ("critical", "high", "medium", "low", "info")
CATEGORIES = (
    "security", "performance", "style", "correctness",
    "maintainability", "documentation", "testing", "other",
)


# ── Audit + billing helpers ───────────────────────────────────────────────

def _json_dump(d: dict) -> str:
    return json.dumps(d, default=str)


async def _log_tool_call(
    db: AsyncSession,
    *,
    agent_id: str,
    tool_name: str,
    input_summary: dict,
    output_len: int,
    sc_cost: float,
    latency_ms: int,
    error: str | None = None,
) -> None:
    await db.execute(
        text(
            """
            INSERT INTO tool_calls
                (id, agent_id, tool_name, input_json, output_len,
                 sc_cost, latency_ms, error, created_at)
            VALUES
                (:id, :agent_id, :tool_name, CAST(:input_json AS JSONB),
                 :output_len, :sc_cost, :latency_ms, :error, NOW())
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "tool_name": tool_name,
            "input_json": _json_dump(input_summary),
            "output_len": output_len,
            "sc_cost": sc_cost,
            "latency_ms": latency_ms,
            "error": error,
        },
    )


async def _charge(
    db: AsyncSession,
    agent: Agent,
    *,
    tool: str,
    ref: str,
) -> float:
    cost = SC_COST[tool]
    if cost <= 0:
        return 0.0
    try:
        await record_transaction(
            db,
            agent_id=agent.id,
            delta=-cost,
            reason=f"commons.code_review.{tool}",
            ref_id=ref,
            note=f"CodeReview: {tool}",
        )
    except InsufficientFundsError as e:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_sc",
                "balance": e.balance,
                "needed": cost,
                "tool": f"commons.code_review.{tool}",
            },
        )
    return cost


# ── Substrate helpers ─────────────────────────────────────────────────────

def _extract_text_from_think(result: dict) -> str:
    parts = result.get("content")
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        chunks: list[str] = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text") or ""
                if t:
                    chunks.append(t)
            elif isinstance(p, str):
                chunks.append(p)
        if chunks:
            return "".join(chunks)
    if isinstance(result.get("text"), str):
        return result["text"]
    return ""


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|markdown|md)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


async def _think_json(
    *, agent_id: str, system: str, user_payload: str,
    max_tokens: int = 2048,
) -> dict:
    """Single LLM round-trip that must return a JSON object. Resilient to
    code fences and to substrates that drift to prose before/after JSON."""
    user_msg = (
        f"{system}\n\n"
        f"---\n"
        f"{user_payload}\n\n"
        f"---\n"
        f"Reply with the JSON object only — no prose, no markdown fences."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
    try:
        result = await route_think(
            agent_id, "fast", messages,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "think_substrate_failed", "msg": str(e)[:300]},
        )

    raw = _extract_text_from_think(result).strip()
    if not raw:
        raise HTTPException(
            status_code=502,
            detail={"error": "think_empty_response",
                    "raw_keys": list(result.keys())},
        )
    cleaned = _strip_code_fence(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise HTTPException(
                status_code=502,
                detail={"error": "think_response_not_json",
                        "raw": raw[:400]},
            )
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=502,
                detail={"error": "think_response_not_json",
                        "msg": str(e), "raw": raw[:400]},
            )


async def _think_text(
    *, agent_id: str, system: str, user_payload: str,
    max_tokens: int = 2048,
) -> str:
    """Single LLM round-trip that returns prose (no JSON parsing)."""
    user_msg = (
        f"{system}\n\n"
        f"---\n"
        f"{user_payload}\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
    try:
        result = await route_think(
            agent_id, "fast", messages,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "think_substrate_failed", "msg": str(e)[:300]},
        )

    raw = _extract_text_from_think(result).strip()
    if not raw:
        raise HTTPException(
            status_code=502,
            detail={"error": "think_empty_response",
                    "raw_keys": list(result.keys())},
        )
    return _strip_code_fence(raw)


# ── Result-shape normalisers ──────────────────────────────────────────────

def _coerce_int(v, default: int | None = None) -> int | None:
    if v is None:
        return default
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _normalize_severity(v) -> str:
    if not isinstance(v, str):
        return "medium"
    v = v.strip().lower()
    if v in SEVERITIES:
        return v
    aliases = {"err": "high", "error": "high", "warn": "medium",
               "warning": "medium", "note": "info", "hint": "info",
               "blocker": "critical", "major": "high", "minor": "low"}
    return aliases.get(v, "medium")


def _normalize_category(v) -> str:
    if not isinstance(v, str):
        return "other"
    v = v.strip().lower()
    if v in CATEGORIES:
        return v
    aliases = {"perf": "performance", "sec": "security",
               "bug": "correctness", "docs": "documentation",
               "doc": "documentation", "test": "testing",
               "tests": "testing", "maint": "maintainability",
               "format": "style", "lint": "style"}
    return aliases.get(v, "other")


def _normalize_issue(raw: dict, *, default_file: str | None = None) -> dict:
    if not isinstance(raw, dict):
        return {}
    file = raw.get("file") or raw.get("filename") or default_file
    line = _coerce_int(raw.get("line") or raw.get("line_number"))
    severity = _normalize_severity(raw.get("severity"))
    category = _normalize_category(raw.get("category"))
    msg = raw.get("msg") or raw.get("message") or raw.get("description") or ""
    if not isinstance(msg, str):
        msg = str(msg)
    suggestion = raw.get("suggestion") or raw.get("fix") or ""
    if not isinstance(suggestion, str):
        suggestion = str(suggestion)
    if not msg.strip():
        return {}
    return {
        "file": file if isinstance(file, str) and file else None,
        "line": line,
        "severity": severity,
        "category": category,
        "msg": msg.strip(),
        "suggestion": suggestion.strip() or None,
    }


def _compute_score(issues: list[dict]) -> int:
    """Heuristic 0-100 score: each issue subtracts weight by severity."""
    weights = {"critical": 25, "high": 12, "medium": 5, "low": 2, "info": 0}
    score = 100
    for it in issues:
        score -= weights.get(it.get("severity", "medium"), 5)
    return max(0, min(100, score))


def _summarize_counts(issues: list[dict]) -> dict[str, int]:
    counts = {s: 0 for s in SEVERITIES}
    for it in issues:
        counts[it.get("severity", "medium")] = (
            counts.get(it.get("severity", "medium"), 0) + 1
        )
    return counts


def _diff_files(diff_text: str) -> list[str]:
    """Pull file paths out of a unified diff for a quick context summary."""
    files: list[str] = []
    for line in diff_text.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
        if m:
            files.append(m.group(2))
            continue
        m = re.match(r"^\+\+\+ b/(.+?)$", line)
        if m and (not files or files[-1] != m.group(1)):
            files.append(m.group(1))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ── Health (public) ───────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    return {
        "service": "commons.code_review",
        "version": "1.0.0",
        "endpoints": ["diff", "file", "explain-change"],
        "sc_cost": SC_COST,
        "limits": {
            "max_diff_chars": MAX_DIFF_CHARS,
            "max_file_chars": MAX_FILE_CHARS,
        },
        "languages": list(LANGUAGES),
        "focus_areas": list(FOCUS_AREAS),
        "severities": list(SEVERITIES),
        "categories": list(CATEGORIES),
    }


# ── /diff ─────────────────────────────────────────────────────────────────

class DiffRequest(BaseModel):
    diff_text: str = Field(..., description="Raw git diff output.")
    language: Literal["python", "javascript", "typescript", "go", "rust"] | None = (
        Field(None, description="Optional primary language hint.")
    )
    focus: Literal["security", "perf", "style", "all"] = Field(
        "all", description="Where to focus the review.")


_DIFF_SYSTEM = (
    "You are a senior engineer performing a code review on a git diff. "
    "Be precise, specific, and actionable. Flag real issues only — do not "
    "invent problems. For each issue, cite the file and the line number "
    "from the diff (use the post-change line on the '+' side). Categorize "
    "each issue and grade severity. Cover the requested focus area: "
    "'security' = injection, auth, secrets, input validation, SSRF, RCE, "
    "crypto misuse; 'perf' = O(n^2) loops, N+1, sync I/O in hot path, "
    "unbounded memory; 'style' = naming, dead code, formatting, idioms; "
    "'all' = everything plus correctness, maintainability, docs, tests. "
    "Output strict JSON: "
    '{"issues": [{"file": str, "line": int|null, "severity": '
    '"critical"|"high"|"medium"|"low"|"info", "category": '
    '"security"|"performance"|"style"|"correctness"|"maintainability"'
    '|"documentation"|"testing"|"other", "msg": str, "suggestion": str}], '
    '"summary": str}. '
    "Keep summary 1–3 sentences. If the diff is clean, return empty issues "
    "and a short positive summary."
)


@router.post("/diff")
async def review_diff(
    req: DiffRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    diff = (req.diff_text or "").strip()
    if not diff:
        raise HTTPException(status_code=400, detail="diff_empty")
    if len(diff) > MAX_DIFF_CHARS:
        raise HTTPException(
            status_code=413,
            detail={"error": "diff_too_long",
                    "max_chars": MAX_DIFF_CHARS,
                    "got_chars": len(diff)},
        )

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    await _charge(db, agent, tool="diff", ref=ref)
    # Commit charge before route_think to avoid agent-row deadlock — same
    # rationale as commons/meeting_notes.py.
    await db.commit()

    files = _diff_files(diff)
    ctx_bits = [f"focus={req.focus}"]
    if req.language:
        ctx_bits.append(f"language={req.language}")
    if files:
        ctx_bits.append(f"files={','.join(files[:8])}")
    ctx = ", ".join(ctx_bits)

    payload = (
        f"Context: {ctx}\n\n"
        f"Git diff:\n\n```diff\n{diff}\n```\n"
    )

    parsed = await _think_json(
        agent_id=agent.id,
        system=_DIFF_SYSTEM,
        user_payload=payload,
        max_tokens=3072,
    )

    raw_issues = parsed.get("issues") if isinstance(parsed, dict) else None
    if not isinstance(raw_issues, list):
        raw_issues = []
    issues: list[dict] = []
    for it in raw_issues:
        norm = _normalize_issue(it)
        if norm:
            issues.append(norm)

    summary = parsed.get("summary") if isinstance(parsed, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        summary = (
            "No issues found." if not issues
            else f"Found {len(issues)} issue(s) across {len(files) or 1} file(s)."
        )

    score = _compute_score(issues)
    counts = _summarize_counts(issues)

    out = {
        "issues": issues,
        "summary": summary.strip(),
        "score": score,
        "counts": counts,
        "files": files,
        "focus": req.focus,
        "language": req.language,
        "sc_cost": SC_COST["diff"],
        "ref": ref,
    }

    await _log_tool_call(
        db, agent_id=agent.id, tool_name="commons.code_review.diff",
        input_summary={
            "diff_chars": len(diff),
            "focus": req.focus,
            "language": req.language,
            "file_count": len(files),
            "issue_count": len(issues),
            "score": score,
        },
        output_len=len(json.dumps(out)),
        sc_cost=SC_COST["diff"],
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    await db.commit()
    return out


# ── /file ─────────────────────────────────────────────────────────────────

class FileRequest(BaseModel):
    filename: str = Field(..., description="Path or filename for context (e.g. 'src/auth.py').")
    content: str = Field(..., description="Full file content.")
    language: Literal["python", "javascript", "typescript", "go", "rust"] | None = (
        Field(None, description="Optional language hint; inferred from filename when absent.")
    )
    focus: Literal["security", "perf", "style", "all"] = Field(
        "all", description="Where to focus the review.")


_FILE_SYSTEM = (
    "You are a senior engineer reviewing a single source file. Flag real "
    "issues only — do not invent problems. For each issue, cite the file "
    "and the line number (1-indexed) where the issue lives. Categorize and "
    "grade severity. Cover the requested focus area: 'security' = injection, "
    "auth, secrets, input validation, SSRF, RCE, crypto misuse; 'perf' = "
    "O(n^2) loops, N+1, sync I/O in hot path, unbounded memory; 'style' = "
    "naming, dead code, formatting, idioms; 'all' = everything plus "
    "correctness, maintainability, docs, tests. Output strict JSON: "
    '{"issues": [{"file": str, "line": int|null, "severity": '
    '"critical"|"high"|"medium"|"low"|"info", "category": '
    '"security"|"performance"|"style"|"correctness"|"maintainability"'
    '|"documentation"|"testing"|"other", "msg": str, "suggestion": str}], '
    '"summary": str}. '
    "Keep summary 1–3 sentences. If the file is clean, return empty issues "
    "and a short positive summary."
)

_LANG_BY_EXT = {
    ".py":  "python",
    ".js":  "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts":  "typescript", ".tsx": "typescript",
    ".go":  "go",
    ".rs":  "rust",
}


def _infer_language(filename: str) -> str | None:
    if not filename:
        return None
    fname = filename.lower()
    for ext, lang in _LANG_BY_EXT.items():
        if fname.endswith(ext):
            return lang
    return None


@router.post("/file")
async def review_file(
    req: FileRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    fname = (req.filename or "").strip()
    content = req.content or ""
    if not fname:
        raise HTTPException(status_code=400, detail="filename_empty")
    if not content.strip():
        raise HTTPException(status_code=400, detail="content_empty")
    if len(content) > MAX_FILE_CHARS:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_long",
                    "max_chars": MAX_FILE_CHARS,
                    "got_chars": len(content)},
        )

    language = req.language or _infer_language(fname)

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    await _charge(db, agent, tool="file", ref=ref)
    await db.commit()

    # Number lines so the LLM can produce accurate line citations.
    numbered_lines = [f"{i+1:>4}: {line}"
                      for i, line in enumerate(content.splitlines())]
    numbered = "\n".join(numbered_lines)

    ctx_bits = [f"focus={req.focus}", f"file={fname}"]
    if language:
        ctx_bits.append(f"language={language}")
    ctx = ", ".join(ctx_bits)

    payload = (
        f"Context: {ctx}\n\n"
        f"File contents (line numbers prefixed):\n\n```\n{numbered}\n```\n"
    )

    parsed = await _think_json(
        agent_id=agent.id,
        system=_FILE_SYSTEM,
        user_payload=payload,
        max_tokens=3072,
    )

    raw_issues = parsed.get("issues") if isinstance(parsed, dict) else None
    if not isinstance(raw_issues, list):
        raw_issues = []
    issues: list[dict] = []
    for it in raw_issues:
        norm = _normalize_issue(it, default_file=fname)
        if norm:
            # Force the file field to the user-supplied filename so the
            # LLM can't drift to a stub name.
            norm["file"] = fname
            issues.append(norm)

    summary = parsed.get("summary") if isinstance(parsed, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        summary = (
            "No issues found." if not issues
            else f"Found {len(issues)} issue(s) in {fname}."
        )

    score = _compute_score(issues)
    counts = _summarize_counts(issues)

    out = {
        "issues": issues,
        "summary": summary.strip(),
        "score": score,
        "counts": counts,
        "file": fname,
        "language": language,
        "focus": req.focus,
        "sc_cost": SC_COST["file"],
        "ref": ref,
    }

    await _log_tool_call(
        db, agent_id=agent.id, tool_name="commons.code_review.file",
        input_summary={
            "filename": fname,
            "content_chars": len(content),
            "focus": req.focus,
            "language": language,
            "issue_count": len(issues),
            "score": score,
        },
        output_len=len(json.dumps(out)),
        sc_cost=SC_COST["file"],
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    await db.commit()
    return out


# ── /explain-change ───────────────────────────────────────────────────────

class ExplainChangeRequest(BaseModel):
    diff_text: str = Field(..., description="Raw git diff output.")


_EXPLAIN_SYSTEM = (
    "You are a senior engineer explaining a code change to a teammate. "
    "Read the supplied git diff and produce a clear, human-readable "
    "explanation of WHAT the change does and WHY it might matter. Keep it "
    "factual — do not speculate beyond what the diff shows. Output a short "
    "markdown document with these sections: '## What changed' (plain-English "
    "bullets, one per logical change), '## Files touched' (bulleted list of "
    "file paths with a phrase about each), and '## Likely intent' (1–3 "
    "sentences inferring the goal, hedged appropriately). Do not wrap the "
    "answer in code fences. Do not invent context that's not in the diff."
)


@router.post("/explain-change")
async def explain_change(
    req: ExplainChangeRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    diff = (req.diff_text or "").strip()
    if not diff:
        raise HTTPException(status_code=400, detail="diff_empty")
    if len(diff) > MAX_DIFF_CHARS:
        raise HTTPException(
            status_code=413,
            detail={"error": "diff_too_long",
                    "max_chars": MAX_DIFF_CHARS,
                    "got_chars": len(diff)},
        )

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    await _charge(db, agent, tool="explain_change", ref=ref)
    await db.commit()

    files = _diff_files(diff)
    ctx_bits = []
    if files:
        ctx_bits.append(f"files={','.join(files[:8])}")
    ctx = ", ".join(ctx_bits) if ctx_bits else "no-context"

    payload = (
        f"Context: {ctx}\n\n"
        f"Git diff:\n\n```diff\n{diff}\n```\n"
    )

    explanation = await _think_text(
        agent_id=agent.id,
        system=_EXPLAIN_SYSTEM,
        user_payload=payload,
        max_tokens=2048,
    )

    if not explanation.strip():
        raise HTTPException(
            status_code=502,
            detail={"error": "think_empty_response"},
        )

    out = {
        "explanation": explanation.strip(),
        "files": files,
        "sc_cost": SC_COST["explain_change"],
        "ref": ref,
    }

    await _log_tool_call(
        db, agent_id=agent.id, tool_name="commons.code_review.explain_change",
        input_summary={
            "diff_chars": len(diff),
            "file_count": len(files),
            "out_chars": len(explanation),
        },
        output_len=len(explanation),
        sc_cost=SC_COST["explain_change"],
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    await db.commit()
    return out
