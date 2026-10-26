# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | yes |

This is an alpha. Everything before 0.1.0 is unsupported and should be treated as
untrusted if you find it.

## Reporting a vulnerability

Use the repository's private security advisory form. Please do not open a public
issue for a vulnerability.

Include:

- What an attacker can achieve, not just what breaks.
- Steps to reproduce, ideally against `docker compose` with the seeded fixture.
- Whether real credentials or a real database are reachable in the scenario.

You can expect an acknowledgement within a few days. Fix timing depends on
severity; we will tell you what we are doing.

## Threat model

SchemaPulse reads one `SELECT` from a database you point it at, and writes
`CREATE INDEX` / `DROP INDEX` **only** to a shadow database. The security
properties it tries to hold:

### Secrets

- `GEMINI_API_KEY` is held as a `SecretStr`, has no default, and is never logged.
- Redacted output shows `***set***` when a key is present and `<unset>` when it is
  not — an absent key must not render as something that reads like a value.
- Reports and JSON results do not include credentials. Check before sharing
  `.schemapulse/last-result.json`; database URLs in your environment may still
  contain passwords.

### The model is untrusted input

Model output is treated as an attacker-controlled string.

- The analysed query must parse, via SQLGlot, as a single `SELECT`. Anything else
  is refused.
- Candidate DDL is re-parsed and **rebuilt from the AST**. The model's text is
  never concatenated into a statement, so injection through a column name or a
  predicate does not reach the database as SQL.
- Only `CREATE INDEX` is executable, on the shadow connection only.
- Identifiers are quoted and validated; `CONCURRENTLY`, `IF NOT EXISTS`, and
  `INCLUDE` are set by the tool, not by the model.
- Candidates are grounded against the live catalog, so an invented table or
  column is rejected before it is ever created.

### Target vs shadow

`assert_production_safe` aborts startup when the target and shadow URLs are
identical. This is the single most important invariant in the project: it is
what keeps model-proposed DDL away from a database you care about.

If you use SchemaPulse in production-adjacent infrastructure, point the shadow at
a genuinely disposable copy. A shadow that shares storage with production is not
a shadow.

### Known non-goals

- No authentication or authorisation. SchemaPulse trusts whoever can run it and
  reach the database URLs. Do not expose it as a service.
- No sandboxing of the model. A prompt injection in your schema comments or query
  text could influence the hypothesis, but validation is catalog-based and
  execution is allow-listed, so it cannot escalate to arbitrary SQL.
- SQL objects referenced by your query must be readable by the configured role.
  SchemaPulse does not elevate privileges.

## Hardening notes for operators

- Run it as a role with read access to the target and `CREATE INDEX` on the
  shadow, and nothing more.
- Keep the shadow on separate storage; `CREATE INDEX` there consumes real disk.
- Treat `.schemapulse/` as potentially sensitive: it contains your query text and
  schema details.
- Pin dependencies when deploying. `requirements` are lower bounds, not locks.