# Contributing

## Getting a working checkout

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt
python -m playwright install chromium
python -m pytest -q
```

The suite is offline by design: no network, no API keys, no local models. If a test of yours
needs the open web it belongs behind the `live` marker, which nothing runs by default:

```bash
RUN_LIVE_WEB_TESTS=1 python -m pytest -m live
```

`tests/conftest.py` enforces the offline default — it disables the browser tier, proxies, the
external scrape API, rate-limit pacing and the search cache, and redirects the source cache
into `tmp_path`. If a change of yours starts touching `~/.footnote-mcp` during tests, that
fixture is where to fix it, not the test.

## What CI will check

`.github/workflows/tests.yml` runs the suite on `{ubuntu, macos, windows} x {3.10, 3.12}` with
`fail-fast: false`. Two things this catches that local runs do not:

- **Python 3.10.** The floor is `requires-python = ">=3.10"`, so no `match`, no `X | Y` in
  runtime type expressions unless the module has `from __future__ import annotations`.
- **Windows.** No `signal.SIGALRM`, no `setitimer`, and paths are not POSIX. Where a feature
  genuinely cannot exist there, degrade explicitly and mark the test
  `@pytest.mark.skipif` with a reason — see `test_hard_task_timeout_interrupts_wall_clock_work`.

## Adding a tool

Tools live in `src/footnote_mcp/tools_*.py` and `tools_data/`, and a new one is wired in three
places in `server.py`: the function import, an entry in `SYNC_TOOLS` (or `BROWSER_TOOLS`) with
its required arguments and defaults, and a `Tool(...)` definition in `list_tools()` whose
schema matches those defaults. `tests/test_server_dispatch.py` covers the dispatch path.

Descriptions in `list_tools()` are read by a model deciding whether to call the tool, so say
what it returns and when to reach for it rather than how it is implemented.

## House rules

**Claims in the README must be traceable to the repository.** No feature, flag, environment
variable, provider or number that is not in the code. Console output must come from a real
run, trimmed if long — never written by hand. A README that advertises something the code does
not do costs more trust than a missing section.

**Politeness is not optional.** Every outbound request goes through `fetch._get`, which applies
the per-domain rate limiter, circuit breaker and negative cache in `politeness.py`. A new code
path that calls an HTTP library directly bypasses all of it. If you need something `_get` does
not do, extend `_get`.

**Say what does not work.** Limitations in the README come from the code, the issue tracker or
a measurement. An honest edge stated up front deflects the issue that would otherwise be filed.

## Style

Type checking is `pyright`, configured in `pyrightconfig.json` and installed by
`requirements-dev.txt`; there is no formatter or linter config in the repo, so match the
surrounding code rather than reformatting files you are passing through. Commit messages
explain why the change was needed; the diff already shows what changed.
