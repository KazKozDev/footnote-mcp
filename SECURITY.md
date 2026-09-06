# Security policy

## Reporting a vulnerability

Report privately through GitHub, not in a public issue:
**[Report a vulnerability](https://github.com/KazKozDev/footnote-mcp/security/advisories/new)**.

Please include what an attacker gains, the steps to reproduce it, and the version
(`pip show footnote-mcp`) plus how the server is run — stdio through an MCP client, Docker, or
the hosted HTTP endpoint. Expect a first reply within a week. Please give a fix a reasonable
window before disclosing publicly.

Only the latest release is supported. Fixes land in a new version rather than as patches to
older ones.

## Where the risk actually is

This is not a passive library. It fetches arbitrary URLs, drives a browser, executes
model-generated code, and can be exposed over HTTP. Reports touching these are especially
welcome:

- **Generated extraction recipes.** `tool_code_run_sandboxed` executes model-written code in a
  subprocess. A static validator allows only `csv`, `datetime`, `html`, `json`, `math`, `re`
  and `statistics`, and rejects `__import__`, `eval`, `exec`, `compile`, `open`, `getattr`,
  `setattr`, `globals`, `locals`, `vars`, `input` and `breakpoint`. Anything that reaches the
  filesystem, the network or the parent process through that validator is a vulnerability.
- **The hosted HTTP server.** Authentication, the per-user API keys in `FOOTNOTE_MCP_API_KEYS`,
  the rate limiter, and the public-host and `Origin` checks that guard against DNS rebinding.
- **Fetching and the browser tier.** Anything that turns a fetched page into code execution,
  a local file read, or a request to an address the caller did not ask for — SSRF into a
  private network included.
- **Cached data.** The source cache under `~/.footnote-mcp/` holds fetched pages, and the
  browser tier can carry cookies supplied to `web_fetch_authenticated`.

## Known boundaries, not vulnerabilities

These are documented design limits. Reports that restate them will be closed as such, though
a concrete bypass of one is a real finding:

- The recipe validator is an import allowlist and a timed subprocess, not a hardened sandbox.
  Treat the models feeding it as you would any untrusted code source.
- The hosted server keeps rate limits in memory; they reset when the instance restarts.
- Search results and page content come from the open web. Nothing verifies that a site is
  honest — that is what `evidence_entailment` and `corroborate_claim` are for, and they judge
  whether a source supports a claim, not whether the source is telling the truth.
