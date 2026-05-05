"""Tests for /api/v1/commons/code-review.

Three layers:
  * Public health endpoint — straight HTTP, no auth.
  * Pure-helper unit tests for normalisers / score / file-extractor.
  * Integration tests for the three substantive endpoints, using a mocked
    route_think so we never hit the LLM substrate from CI.

Mirrors the auth + skip-pattern of test_changelog.py for live-server tests
of the public health route.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch, AsyncMock

import httpx
import pytest

from app.api.commons import code_review as cr


BASE = os.environ.get("SURFACE_BASE_URL", "http://127.0.0.1:3000")


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_DIFF = """\
diff --git a/app/auth/login.py b/app/auth/login.py
index 1111111..2222222 100644
--- a/app/auth/login.py
+++ b/app/auth/login.py
@@ -1,8 +1,12 @@
-import hashlib
+import hashlib, os

 def hash_password(pw):
-    return hashlib.md5(pw.encode()).hexdigest()
+    # md5 is fine for passwords, right?
+    return hashlib.md5(pw.encode()).hexdigest()
+
+def login(user, pw):
+    if user == "admin" and pw == "admin123":
+        return True
+    return False
diff --git a/app/api/users.py b/app/api/users.py
index 3333333..4444444 100644
--- a/app/api/users.py
+++ b/app/api/users.py
@@ -10,6 +10,8 @@ def get_user(uid):
-    return db.execute(f"SELECT * FROM users WHERE id = {uid}").fetchone()
+    query = f"SELECT * FROM users WHERE id = {uid}"
+    return db.execute(query).fetchone()
"""


SAMPLE_LLM_REVIEW_JSON = {
    "issues": [
        {
            "file": "app/auth/login.py",
            "line": 6,
            "severity": "critical",
            "category": "security",
            "msg": "MD5 is not a secure password hash; collisions are practical.",
            "suggestion": "Use bcrypt or argon2 with a per-user salt.",
        },
        {
            "file": "app/auth/login.py",
            "line": 9,
            "severity": "critical",
            "category": "security",
            "msg": "Hardcoded admin credentials shipped in source.",
            "suggestion": "Read from env or secrets manager; never commit.",
        },
        {
            "file": "app/api/users.py",
            "line": 11,
            "severity": "high",
            "category": "security",
            "msg": "SQL injection — uid is interpolated into the query.",
            "suggestion": "Use a parameterised query with a placeholder.",
        },
    ],
    "summary": "Three security issues across two files: weak hashing, hardcoded creds, SQLi.",
}


SAMPLE_LLM_FILE_JSON = {
    "issues": [
        {
            "file": "src/handler.py",
            "line": 4,
            "severity": "high",
            "category": "performance",
            "msg": "Quadratic loop over the input list.",
            "suggestion": "Use a set lookup instead of list.index().",
        }
    ],
    "summary": "One performance issue: quadratic dedup.",
}


# ── Pure-helper unit tests (no network, no DB) ────────────────────────────

def test_diff_files_extracts_paths():
    files = cr._diff_files(SAMPLE_DIFF)
    assert files == ["app/auth/login.py", "app/api/users.py"]


def test_diff_files_dedups():
    diff = (
        "diff --git a/x.py b/x.py\n"
        "+++ b/x.py\n"
        "diff --git a/x.py b/x.py\n"
        "+++ b/x.py\n"
    )
    assert cr._diff_files(diff) == ["x.py"]


def test_normalize_severity_aliases():
    assert cr._normalize_severity("BLOCKER") == "critical"
    assert cr._normalize_severity("error") == "high"
    assert cr._normalize_severity("warn") == "medium"
    assert cr._normalize_severity("hint") == "info"
    assert cr._normalize_severity("unknown") == "medium"
    assert cr._normalize_severity(None) == "medium"
    assert cr._normalize_severity("critical") == "critical"


def test_normalize_category_aliases():
    assert cr._normalize_category("perf") == "performance"
    assert cr._normalize_category("sec") == "security"
    assert cr._normalize_category("docs") == "documentation"
    assert cr._normalize_category("bug") == "correctness"
    assert cr._normalize_category("Tests") == "testing"
    assert cr._normalize_category("garbage") == "other"


def test_normalize_issue_drops_empty_msg():
    assert cr._normalize_issue({"severity": "high"}) == {}
    assert cr._normalize_issue({"msg": "   "}) == {}
    assert cr._normalize_issue("not a dict") == {}


def test_normalize_issue_pulls_message_aliases():
    out = cr._normalize_issue({
        "filename": "a.py", "line_number": "12",
        "severity": "blocker", "category": "perf",
        "description": "n+1 loop", "fix": "batch the call",
    })
    assert out["file"] == "a.py"
    assert out["line"] == 12
    assert out["severity"] == "critical"
    assert out["category"] == "performance"
    assert out["msg"] == "n+1 loop"
    assert out["suggestion"] == "batch the call"


def test_compute_score_empty_is_100():
    assert cr._compute_score([]) == 100


def test_compute_score_clamps_at_zero():
    issues = [{"severity": "critical"}] * 10
    assert cr._compute_score(issues) == 0


def test_compute_score_weights():
    assert cr._compute_score([{"severity": "critical"}]) == 75
    assert cr._compute_score([{"severity": "high"}]) == 88
    assert cr._compute_score([{"severity": "medium"}]) == 95
    assert cr._compute_score([{"severity": "low"}]) == 98
    assert cr._compute_score([{"severity": "info"}]) == 100


def test_summarize_counts_groups_by_severity():
    issues = [
        {"severity": "critical"}, {"severity": "critical"},
        {"severity": "high"}, {"severity": "low"},
    ]
    counts = cr._summarize_counts(issues)
    assert counts == {
        "critical": 2, "high": 1, "medium": 0, "low": 1, "info": 0,
    }


def test_infer_language():
    assert cr._infer_language("a/b/c.py") == "python"
    assert cr._infer_language("x.JS") == "javascript"
    assert cr._infer_language("x.tsx") == "typescript"
    assert cr._infer_language("main.go") == "go"
    assert cr._infer_language("lib.rs") == "rust"
    assert cr._infer_language("unknown.toml") is None
    assert cr._infer_language("") is None


def test_strip_code_fence_removes_json_fence():
    raw = '```json\n{"a": 1}\n```'
    assert cr._strip_code_fence(raw) == '{"a": 1}'


def test_extract_text_from_think_string_form():
    assert cr._extract_text_from_think({"content": "hello"}) == "hello"


def test_extract_text_from_think_anthropic_form():
    parts = [{"type": "text", "text": "hi "},
             {"type": "text", "text": "there"}]
    assert cr._extract_text_from_think({"content": parts}) == "hi there"


def test_extract_text_from_think_text_field_fallback():
    assert cr._extract_text_from_think({"text": "fallback"}) == "fallback"


def test_extract_text_from_think_empty_returns_empty():
    assert cr._extract_text_from_think({}) == ""


# ── Integration tests against live server (mocked LLM) ────────────────────
#
# These tests need a running surface-api at SURFACE_BASE_URL with auth env
# vars set. They mock route_think on the server-side process — unreachable
# from the test client — so they only run when explicitly opted-in via
# CODE_REVIEW_LIVE_TESTS=1. Without that flag, the public-health test still
# runs and the rest is skipped to keep CI fast and substrate-independent.


SVC_KEY = os.environ.get("SURFACE_SERVICE_KEY") or os.environ.get("SERVICE_API_KEY")
AGENT_ID = os.environ.get("SURFACE_TEST_AGENT_ID")


def _client() -> httpx.Client:
    if not SVC_KEY or not AGENT_ID:
        pytest.skip("SURFACE_SERVICE_KEY + SURFACE_TEST_AGENT_ID required")
    if os.environ.get("CODE_REVIEW_LIVE_TESTS") != "1":
        pytest.skip("set CODE_REVIEW_LIVE_TESTS=1 to run live-server tests")
    return httpx.Client(
        base_url=BASE,
        headers={"X-Service-Key": SVC_KEY, "X-Agent-Id": AGENT_ID},
        timeout=180.0,
    )


def test_health_public():
    r = httpx.get(f"{BASE}/api/v1/commons/code-review/health", timeout=10)
    assert r.status_code == 200
    j = r.json()
    assert j["service"] == "commons.code_review"
    assert "diff" in j["endpoints"]
    assert "file" in j["endpoints"]
    assert "explain-change" in j["endpoints"]
    assert j["sc_cost"]["diff"] == 15
    assert j["sc_cost"]["file"] == 10
    assert j["sc_cost"]["explain_change"] == 8
    assert "python" in j["languages"]
    assert "all" in j["focus_areas"]
    assert "critical" in j["severities"]


def test_diff_review_against_live_server():
    with _client() as c:
        r = c.post(
            "/api/v1/commons/code-review/diff",
            json={
                "diff_text": SAMPLE_DIFF,
                "language": "python",
                "focus": "security",
            },
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert "issues" in j
        assert "summary" in j
        assert "score" in j
        assert isinstance(j["score"], int) and 0 <= j["score"] <= 100
        assert j["sc_cost"] == 15
        assert "app/auth/login.py" in j["files"]


def test_diff_empty_rejected():
    with _client() as c:
        r = c.post("/api/v1/commons/code-review/diff",
                   json={"diff_text": "   "})
        assert r.status_code == 400


def test_file_review_against_live_server():
    src = (
        "def dedup(items):\n"
        "    out = []\n"
        "    for x in items:\n"
        "        if x not in out:  # quadratic\n"
        "            out.append(x)\n"
        "    return out\n"
    )
    with _client() as c:
        r = c.post(
            "/api/v1/commons/code-review/file",
            json={"filename": "src/handler.py", "content": src,
                  "language": "python", "focus": "perf"},
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["file"] == "src/handler.py"
        assert j["language"] == "python"
        assert j["sc_cost"] == 10
        assert "score" in j


def test_file_empty_content_rejected():
    with _client() as c:
        r = c.post(
            "/api/v1/commons/code-review/file",
            json={"filename": "a.py", "content": "   "},
        )
        assert r.status_code == 400


def test_explain_change_against_live_server():
    with _client() as c:
        r = c.post(
            "/api/v1/commons/code-review/explain-change",
            json={"diff_text": SAMPLE_DIFF},
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert "explanation" in j
        assert isinstance(j["explanation"], str) and len(j["explanation"]) > 0
        assert j["sc_cost"] == 8


def test_explain_change_empty_rejected():
    with _client() as c:
        r = c.post(
            "/api/v1/commons/code-review/explain-change",
            json={"diff_text": ""},
        )
        assert r.status_code == 400


def test_invalid_language_rejected():
    if not SVC_KEY or not AGENT_ID:
        pytest.skip("auth env required")
    if os.environ.get("CODE_REVIEW_LIVE_TESTS") != "1":
        pytest.skip("set CODE_REVIEW_LIVE_TESTS=1 to run")
    r = httpx.post(
        f"{BASE}/api/v1/commons/code-review/diff",
        json={"diff_text": SAMPLE_DIFF, "language": "cobol"},
        headers={"X-Service-Key": SVC_KEY, "X-Agent-Id": AGENT_ID},
        timeout=30,
    )
    assert r.status_code == 422


def test_auth_rejected_without_key():
    r = httpx.post(
        f"{BASE}/api/v1/commons/code-review/diff",
        json={"diff_text": SAMPLE_DIFF},
        timeout=10,
    )
    # Either 401 (missing service key) or 422 (validation) — match the
    # changelog test's tolerance for whichever the framework lands on first.
    assert r.status_code in (401, 422)


# ── Mocked-LLM async tests — exercise the route_think substrate path ──────
#
# These use AsyncMock to bypass the real substrate. They run regardless of
# CI env so we always verify the pipeline stays well-formed. They patch
# route_think AND the DB-side helpers (_charge, _log_tool_call) to keep the
# test pure-pure.


@pytest.mark.asyncio
async def test_review_diff_pipeline_with_mocked_substrate():
    fake_llm = AsyncMock(return_value={"content": json.dumps(SAMPLE_LLM_REVIEW_JSON)})
    fake_charge = AsyncMock(return_value=15.0)
    fake_log = AsyncMock(return_value=None)
    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    with patch.object(cr, "route_think", fake_llm), \
         patch.object(cr, "_charge", fake_charge), \
         patch.object(cr, "_log_tool_call", fake_log):
        out = await cr.review_diff(
            cr.DiffRequest(diff_text=SAMPLE_DIFF, focus="security",
                           language="python"),
            db=fake_db,
            agent=FakeAgent(),
        )

    assert isinstance(out, dict)
    assert out["sc_cost"] == 15
    assert out["focus"] == "security"
    assert out["language"] == "python"
    assert out["files"] == ["app/auth/login.py", "app/api/users.py"]
    assert len(out["issues"]) == 3
    # Severity normalisation kept "critical".
    assert out["issues"][0]["severity"] == "critical"
    assert out["issues"][0]["category"] == "security"
    assert out["score"] < 100  # 3 issues -> score drops
    assert out["counts"]["critical"] == 2
    assert out["counts"]["high"] == 1
    assert "Three security issues" in out["summary"]
    fake_charge.assert_awaited_once()
    fake_log.assert_awaited_once()
    fake_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_review_diff_handles_substrate_returning_fenced_json():
    fake_llm = AsyncMock(return_value={
        "content": "```json\n" + json.dumps(SAMPLE_LLM_REVIEW_JSON) + "\n```"
    })
    fake_charge = AsyncMock(return_value=15.0)
    fake_log = AsyncMock(return_value=None)
    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    with patch.object(cr, "route_think", fake_llm), \
         patch.object(cr, "_charge", fake_charge), \
         patch.object(cr, "_log_tool_call", fake_log):
        out = await cr.review_diff(
            cr.DiffRequest(diff_text=SAMPLE_DIFF, focus="all"),
            db=fake_db, agent=FakeAgent(),
        )
    assert len(out["issues"]) == 3


@pytest.mark.asyncio
async def test_review_diff_handles_substrate_anthropic_content_form():
    fake_llm = AsyncMock(return_value={
        "content": [
            {"type": "text", "text": json.dumps(SAMPLE_LLM_REVIEW_JSON)}
        ],
    })
    fake_charge = AsyncMock(return_value=15.0)
    fake_log = AsyncMock(return_value=None)
    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    with patch.object(cr, "route_think", fake_llm), \
         patch.object(cr, "_charge", fake_charge), \
         patch.object(cr, "_log_tool_call", fake_log):
        out = await cr.review_diff(
            cr.DiffRequest(diff_text=SAMPLE_DIFF, focus="all"),
            db=fake_db, agent=FakeAgent(),
        )
    assert len(out["issues"]) == 3


@pytest.mark.asyncio
async def test_review_file_pipeline_with_mocked_substrate():
    fake_llm = AsyncMock(return_value={"content": json.dumps(SAMPLE_LLM_FILE_JSON)})
    fake_charge = AsyncMock(return_value=10.0)
    fake_log = AsyncMock(return_value=None)
    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    src = "def f():\n    return 1\n"
    with patch.object(cr, "route_think", fake_llm), \
         patch.object(cr, "_charge", fake_charge), \
         patch.object(cr, "_log_tool_call", fake_log):
        out = await cr.review_file(
            cr.FileRequest(filename="src/handler.py", content=src,
                           focus="perf"),
            db=fake_db, agent=FakeAgent(),
        )

    assert out["sc_cost"] == 10
    assert out["file"] == "src/handler.py"
    assert out["language"] == "python"  # inferred
    assert len(out["issues"]) == 1
    # Filename forced to user-supplied value even if LLM emitted a stub.
    assert out["issues"][0]["file"] == "src/handler.py"
    assert out["issues"][0]["category"] == "performance"


@pytest.mark.asyncio
async def test_explain_change_pipeline_with_mocked_substrate():
    fake_text = (
        "## What changed\n"
        "- Added a stub login that compares against a hardcoded admin password.\n"
        "## Files touched\n"
        "- app/auth/login.py — new login function and md5 hash\n"
        "## Likely intent\n"
        "Wire up a temporary admin login while the real auth path is built.\n"
    )
    fake_llm = AsyncMock(return_value={"content": fake_text})
    fake_charge = AsyncMock(return_value=8.0)
    fake_log = AsyncMock(return_value=None)
    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    with patch.object(cr, "route_think", fake_llm), \
         patch.object(cr, "_charge", fake_charge), \
         patch.object(cr, "_log_tool_call", fake_log):
        out = await cr.explain_change(
            cr.ExplainChangeRequest(diff_text=SAMPLE_DIFF),
            db=fake_db, agent=FakeAgent(),
        )

    assert out["sc_cost"] == 8
    assert out["files"] == ["app/auth/login.py", "app/api/users.py"]
    assert "What changed" in out["explanation"]
    assert "Likely intent" in out["explanation"]


@pytest.mark.asyncio
async def test_review_diff_substrate_failure_502():
    from fastapi import HTTPException

    fake_llm = AsyncMock(side_effect=RuntimeError("boom"))
    fake_charge = AsyncMock(return_value=15.0)
    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    with patch.object(cr, "route_think", fake_llm), \
         patch.object(cr, "_charge", fake_charge):
        with pytest.raises(HTTPException) as ei:
            await cr.review_diff(
                cr.DiffRequest(diff_text=SAMPLE_DIFF, focus="all"),
                db=fake_db, agent=FakeAgent(),
            )
    assert ei.value.status_code == 502
    assert ei.value.detail["error"] == "think_substrate_failed"


@pytest.mark.asyncio
async def test_review_diff_substrate_returns_garbage_502():
    from fastapi import HTTPException

    fake_llm = AsyncMock(return_value={"content": "not json at all"})
    fake_charge = AsyncMock(return_value=15.0)
    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    with patch.object(cr, "route_think", fake_llm), \
         patch.object(cr, "_charge", fake_charge):
        with pytest.raises(HTTPException) as ei:
            await cr.review_diff(
                cr.DiffRequest(diff_text=SAMPLE_DIFF, focus="all"),
                db=fake_db, agent=FakeAgent(),
            )
    assert ei.value.status_code == 502
    assert ei.value.detail["error"] == "think_response_not_json"


@pytest.mark.asyncio
async def test_diff_too_long_413():
    from fastapi import HTTPException

    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    big = "diff --git a/x b/x\n" + ("+x\n" * (cr.MAX_DIFF_CHARS // 3 + 100))
    with pytest.raises(HTTPException) as ei:
        await cr.review_diff(
            cr.DiffRequest(diff_text=big, focus="all"),
            db=fake_db, agent=FakeAgent(),
        )
    assert ei.value.status_code == 413


@pytest.mark.asyncio
async def test_file_too_long_413():
    from fastapi import HTTPException

    fake_db = AsyncMock()

    class FakeAgent:
        id = "agent-test-001"

    big = "x" * (cr.MAX_FILE_CHARS + 1)
    with pytest.raises(HTTPException) as ei:
        await cr.review_file(
            cr.FileRequest(filename="big.py", content=big, focus="all"),
            db=fake_db, agent=FakeAgent(),
        )
    assert ei.value.status_code == 413
