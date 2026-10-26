# Roadmap

Alpha, 0.1.0. This is what is deliberately *not* built yet, and why.

## Now

- Single-statement `SELECT` analysis, index candidates only.
- One shadow database on one machine.
- Providers: Gemini, Ollama, and a deterministic fixture.

## Next

**More candidate types.** Statistics tuning (`ALTER TABLE ... SET STATISTICS`,
`ANALYZE`) and `extended_statistics` are the obvious additions. Both are cheap to
validate with the machinery already here: parse, allow-list, execute on the
shadow, measure. The blocker is that the current safety guard only recognises
`CREATE INDEX`.

**Non-PostgreSQL engines.** The measurement loop is engine-agnostic. The catalog
inspection, `pg_stats` access, and SQL dialect are not. Supporting another engine
means a dialect layer, not a new agent.

**Better noise handling.** The current guard is relative spread on medians. A
paired statistical test would detect "the baseline was slow on run 3" instead of
treating it as global noise.

**Cost modelling for writes.** An index is a write cost, and SchemaPulse only
measures reads. Reporting `INSERT`/`UPDATE` cost on the shadow for both arms is
the missing half of the argument.

**Multi-statement analysis.** Refusing anything that is not a single `SELECT` is
the right MVP scope, but it is a real limitation for `INSERT ... SELECT` and
CTE-heavy queries.

## Not planned

- **Auto-applying changes.** Even a verified index needs a deployment decision.
  `CREATE INDEX` locks writes; `CONCURRENTLY` has its own failure modes. The
  recommendation stays a recommendation.
- **Optimising a live production database directly.** That is what the shadow
  database exists to prevent.
- **Reporting a speedup without the raw timings.** If the per-run numbers are not
  shown, the median is not evidence.

## Known limitations of 0.1.0

- Single-statement `SELECT` only.
- Index candidates only.
- One shadow on one machine, so the measured numbers are a smoke test rather than
  a capacity plan.
- The live model adapters are exercised end-to-end only when a real key or a
  local Ollama model is available; the fixture provider is the tested default.
- Timings are machine-specific and are deliberately absent from the golden
  fixture file.