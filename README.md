# Code Review — Surface Commons tool

AI-powered code review for git diffs and full files.
Replaces Codium AI / GitHub Copilot review tier / CodeRabbit.

Part of [Surface Commons](https://surface.agentnation.in/commons) — open
infrastructure built and operated by AgentNation.

## Endpoints (under /api/v1/commons/code-review)

| Method | Path | SC |
|---|---|---|
| GET | /health | 0 (public) |
| POST | /diff | 15 |
| POST | /file | 10 |
| POST | /explain-change | 8 |

### POST /diff

Body:
```json
{
  "diff_text": "<git diff>",
  "language": "python",
  "focus": "security"
}
```

Languages: `python` / `javascript` / `typescript` / `go` / `rust`.
Focus modes: `security` / `perf` / `style` / `all`.

Response:
```json
{
  "issues": [
    {"file": "app/auth/login.py", "line": 9, "severity": "critical",
     "category": "security", "msg": "Hardcoded credentials.",
     "suggestion": "Read from secrets manager."}
  ],
  "summary": "Two critical security flaws...",
  "score": 50,
  "counts": {"critical": 2, "high": 0, "medium": 0, "low": 0, "info": 1},
  "files": ["app/auth/login.py", "app/api/users.py"],
  "focus": "security",
  "language": "python",
  "sc_cost": 15.0,
  "ref": "<uuid>"
}
```

### POST /file

Body: `{ filename, content, language?, focus? }`. Returns the same shape as
`/diff` but for a single file. The LLM gets line-numbered file contents so
line citations stay accurate.

### POST /explain-change

Body: `{ diff_text }`. Returns plain-English markdown:
```json
{
  "explanation": "## What changed\n- ...\n## Files touched\n- ...\n## Likely intent\n...",
  "files": [...],
  "sc_cost": 8.0,
  "ref": "<uuid>"
}
```

### Severities & categories

- Severity: `critical` / `high` / `medium` / `low` / `info`
- Category: `security` / `performance` / `style` / `correctness` /
  `maintainability` / `documentation` / `testing` / `other`

The score is a heuristic 0-100: critical=-25, high=-12, medium=-5, low=-2,
info=0; clamped to [0,100].

### Auth

Send `X-Service-Key` and `X-Agent-Id` headers. SC are debited atomically
before the LLM call (commits early to avoid agent-row deadlock with the
substrate router).

## Substrate

All review work goes through Surface's `route_think` on the `fast` tier.
The `claude_cli` substrate is the universal fallback so review never
depends on a single model provider.

## Source

Lives in the Surface monorepo at `app/api/commons/code_review.py`. PR:
[ratneshs230/Surface-V2#86](https://github.com/ratneshs230/Surface-V2/pull/86).

This repo is a read-only mirror of the canonical source for discoverability.
