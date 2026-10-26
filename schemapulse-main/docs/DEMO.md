# Demo runbook

The shortest reliable path from a cold machine to the full evidence chain.
Every command and timing below was rehearsed from a torn-down Docker environment
on Python 3.12.11 / PostgreSQL 16.15 / Docker 29.6.1.

**Gemma proposes. PostgreSQL proves.**

## Timing at a glance

Budget for this before you present. The two slow steps are the ones people get
caught by.

| Step | Command | Cold | Warm |
| --- | --- | --- | --- |
| Docker | `docker compose up -d --wait` | ~10 s | ~2 s |
| Seed | `schemapulse demo --seed-only` | **~5 min** | ~2 s |
| Doctor | `schemapulse doctor` | ~5 s | ~5 s |
| **Golden demo** | `schemapulse demo --provider fixture` | **~41 s** | ~41 s |
| Gemini | `... --provider gemini` | ~70 s | ~70 s |
| Ollama | `... --provider ollama` | **~7 min** | ~7 min |
| Report | `schemapulse report --open` | ~2 s | ~2 s |

If you have ten minutes, run steps 1–5 and show the report. That is the whole
thesis and it needs no network.

### One command instead of five

```bash
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
```

Runs steps 1–5 in order, stops with a clear message if any of them fails, and
deliberately **skips** the live model so an accidental API call can never happen
on stage. Add `-Live` when you have decided to spend one.

---

## 1. After a reboot — do this first

This is the single most common way a demo dies. Docker Desktop does not always
come back up on its own, and **the demo data does not survive a restart**: both
containers mount `tmpfs`, so PostgreSQL starts empty every time.

```bash
docker info                 # if this errors, Docker Desktop is not running — start it
docker compose up -d --wait
schemapulse demo --seed-only
```

Expect `seeded. next: schemapulse inspect demo/slow_orders.sql`. If you skip the
seed, `doctor` will still say the databases are healthy — they genuinely are —
but every demo query returns nothing.

## 2. Prerequisites

| Need | Why | Check |
| --- | --- | --- |
| Python 3.11+ | runtime | `python -V` |
| Docker Desktop | the two demo PostgreSQL servers | `docker info` |
| Ollama *(optional)* | local Gemma fallback | `ollama list` |

No API key is needed for the golden demo.

```bash
pip install -e ".[dev]"
```

## 3. Docker startup

```bash
docker compose up -d --wait
```

Baseline on **:5432**, shadow on **:5433**. `--wait` blocks until both report
healthy, so you never race the seed.

## 4. Seeding

```bash
schemapulse demo --seed-only
```

1,000,000 orders and 20,000 customers, loaded into **both** servers so the two
arms are measured against identical data. Idempotent — safe to re-run.

- First run: `fixture ready · 1,000,000 orders · 20,000 customers · 19,794
  pending (1.98%) · 104 MB · 0 non-primary indexes on orders`
- Re-run: `baseline: already seeded, left untouched`

That last number matters: **0 non-primary indexes on `orders`** is what makes the
query slow, and it is what the demo is about.

## 5. Doctor

```bash
schemapulse doctor
```

Must be all green before you present. Checks Python, Docker, both databases, and
whether the configured model endpoint is reachable.

```
python           ok  3.12.11
docker           ok  server 29.6.1, compose yes
target database  ok  postgresql://postgres@localhost:5432/schemapulse · PostgreSQL 16.15
shadow database  ok  postgresql://postgres@localhost:5433/schemapulse · PostgreSQL 16.15
gemma provider   ok  gemini/gemma-4-26b-a4b-it — model endpoint reachable
demo fixture     ok  ...\demo\slow_orders.sql
ready. try: schemapulse inspect demo/slow_orders.sql
```

`ready.` on the last line is the go signal.

## 6. The fixture golden demo — the headline

```bash
schemapulse demo --provider fixture
```

One command: environment → seed → inspect → diagnose → report. Deterministic,
offline, no API key, no model call. Exit code **0**.

Expected output:

```
✓ passed validation CREATE INDEX "sp_orders_status_created_at_partial"
    ON "orders" ("created_at" DESC) WHERE "status" = 'pending'
baseline median 267.68 ms (spread 21%)
candidate median 2.48 ms
VERIFIED · 108.11x verified

rows examined: 1,000,045 -> 96
buffers touched: 53,413 -> 708
deterministic baseline (fixture) — no model was called
```

**The multiplier is machine-specific** — 60x–110x observed across rehearsals. Do
not read the number aloud as if it were fixed. Read the invariants instead,
because those hold everywhere:

- index name `sp_orders_status_created_at_partial`, **partial** on
  `(created_at DESC) WHERE status = 'pending'`
- `rows examined: 1,000,045 -> 96`
- `buffers touched: 53,413 -> 708`
- verdict `verified`, exit `0`

Longer version, if you want to narrate each stage:

```bash
schemapulse inspect  demo/slow_orders.sql                      # no model, no network
schemapulse diagnose demo/slow_orders.sql --provider fixture   # the loop
```

`inspect` shows a `Seq Scan on orders` reading 1,000,045 rows to return 50, with
`filter (o.status = 'pending')` discarding 980,205 of them, and no non-primary
index available.

Two details worth pointing at, because they are the whole thesis:

- The DDL is **rebuilt from the parsed AST**, never interpolated from model text.
- `cleaned up sp_orders_status_created_at_partial` happens **even on failure**.
  The only writes SchemaPulse ever performs are `CREATE INDEX` / `DROP INDEX`,
  and only on the shadow.

## 7. The Gemini live demo

```bash
schemapulse provider --probe                                  # optional reachability check
schemapulse diagnose demo/slow_orders.sql --provider gemini
```

Needs `GEMINI_API_KEY`. The key is never logged, never printed, and never written
to a report — verified by inspecting `.schemapulse/last-result.json`.

Expected output — note the model proposes something **different** from the
fixture, which is how you know it was a real model:

```
Gemma gemini/gemma-4-26b-a4b-it attempt 1
  → response 20988 ms · 3 candidate(s) · live
  ✓ passed validation CREATE INDEX idx_orders_status_created_at ON orders (status, created_at DESC);
VERIFIED · 65.07x verified
live model: gemini/gemma-4-26b-a4b-it
```

| | Fixture | Live Gemma 4 |
| --- | --- | --- |
| index | `sp_orders_status_created_at_partial` | `idx_orders_status_created_at` |
| shape | partial, `(created_at DESC) WHERE status='pending'` | composite `(status, created_at DESC)`, not partial |

Both reach `verified`. The difference is the proof the call was live.

If you want to show the AST rebuild on screen, compare the model's own text to
what was executed:

```
model wrote:  CREATE INDEX idx_orders_status_created_at ON orders (status, created_at DESC);
executed:     CREATE INDEX "idx_orders_status_created_at" ON "orders" ("status", "created_at" DESC)
```

Unquoted identifiers and the trailing `;` are gone. SchemaPulse re-emitted the
statement from the parse tree.

Occasionally attempt 1 comes back `finishReason=RECITATION` and is recorded as
`FAILED` before a retry succeeds. That is the retry path working; the failure
stays in the evidence table on purpose.

## 8. The Ollama rejection demo

```bash
schemapulse diagnose demo/slow_orders.sql --provider ollama
```

Needs `ollama serve` and `ollama pull gemma3:4b-it-q4_K_M` first. Expect
**exit code 2** and a non-green verdict. Both outcomes below are correct.

The local 4B model tends to propose indexes on the **wrong table** (`customers`)
for a query dominated by `orders`. SchemaPulse validates, measures, and reports
what it actually found:

```
REJECTED
  the planner did not select the new index, so the 1.54x median difference
  is not attributable to it
```

**What it refuses to do:** bank a 1.54x "win" when the plan shape is unchanged.

If the model names a *different* access method, the safety guard blocks every
candidate before anything is measured, and the verdict is `FAILED` instead:

```
✗ rejected safety guard rejected it: USING gin access method is not allowed; only btree is supported
```

That is also correct, and arguably the better demo. SchemaPulse does not get to
choose between `GIN` and `GIST` semantics, `TABLESPACE`, or storage parameters,
and the model does not get to write the statement that executes. Both paths exit
`2`.

An explicit `USING btree` is **accepted**, because `btree` is PostgreSQL's default
and naming it changes nothing: the validated DDL is still rebuilt from the parsed
AST, and the executed statement is the canonical form without the clause. Only
`btree` is accepted this way; every other method is refused. Writing `USING` after
the column list (`ON orders (status) USING btree`) is a genuine PostgreSQL syntax
error and is refused as unparseable.

**Say this out loud when you show it:** *"The model was wrong, the database said
so, and the tool did not pretend otherwise."* That sentence is the project.

## 9. Opening the report

```bash
schemapulse report --open
```

Renders `.schemapulse\report.html` from the most recent run — a self-contained
page with the plan before and after, per-run timings, the verdict, and the
`CREATE INDEX CONCURRENTLY` recommendation with its warnings. Also written as
`.schemapulse\last-result.json` for machine reading.

Re-run `diagnose` first if you want the report to match a specific provider.

## 10. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `docker info` errors | daemon not running | start Docker Desktop, wait for it to settle |
| databases healthy but query returns nothing | `tmpfs` — data gone after restart | `schemapulse demo --seed-only` |
| seed seems hung | it is not; ~5 min cold | wait, or watch `docker compose logs -f` |
| `baseline timing spread was 51%` | noisy machine, not a bad index | the noise guard already handles it; quieter machine if it matters |
| `finishReason=RECITATION` once | model declined that attempt | retried automatically; failure kept as evidence |
| `finishReason=RECITATION` every attempt | model will not answer this prompt | `--provider fixture` |
| `USING gin access method is not allowed` | model chose a non-default access method | expected; only `btree` is supported |
| `candidate SQL could not be fully parsed` | usually `USING` written after the column list | prompt now shows the exact shape; `ON t (c)`, never `ON t (c) USING btree` |
| `Ollama provider failed ... timed out` | 4B model on CPU is slow | raise `SCHEMAPULSE_OLLAMA_TIMEOUT_S`, or use fixture/gemini |
| target and shadow URLs identical | refused on purpose | they must differ; that refusal is the safety invariant |

## 11. Why `REJECTED` is a success

A tool that only ever prints ✅ has learned nothing. It is indistinguishable
from a tool that prints ✅ unconditionally, and you cannot tell the difference
without checking the work.

The entire value of *"Gemma proposes. PostgreSQL proves"* is that **the verdict
is computed by the deterministic system, not asserted by the model.** If the
system could be talked into a green tick, the slogan would be false.

So the demo is not "look, it made my query fast." It is:

1. a correct index, proposed and then **measured** → `VERIFIED`
2. a wrong index, proposed by a real local model → `REJECTED`
3. a hostile statement from a model → **blocked before execution**

Three outcomes, one rule: nothing is believed until PostgreSQL measures it. Exit
codes `0` / `2` / `1` make that rule enforceable in CI, not just in a demo.

**Do not** try to force a `VERIFIED` on demo day. If the fixture run does not
verify, something is genuinely wrong with the environment — run `doctor`, re-seed,
and investigate. Do not reach for `--min-speedup 0.1`.

## One honest caveat to state out loud

The numbers come from a synthetic shadow database with a single query's workload,
on one machine. It is a smoke test, not a capacity plan. SchemaPulse prints this
warning in the report itself — worth reading aloud rather than skipping, because
it is the difference between a demo and a tool.