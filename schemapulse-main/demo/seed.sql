-- ===========================================================================
-- SchemaPulse deterministic demo seed
--
-- Determinism matters more than realism here:
--   * every run produces byte-identical data, so a recorded demo replays
--   * the same row counts land on the baseline and the shadow database, so a
--     before/after measurement compares like with like
--   * `status` has an exact known distribution, so the golden query's expected
--     selectivity is known in advance
--
-- Row counts are supplied as psql variables so the test suite can use a smaller
-- dataset for fast runs:
--     psql -v orders=50000 -v customers=1000 -f demo/seed.sql
-- ===========================================================================

\set ON_ERROR_STOP on

-- Clean slate.  Re-seeding is always safe.
TRUNCATE orders, customers RESTART IDENTITY;

-- ---------------------------------------------------------------------------
-- customers: deterministic pseudo-random attributes
-- ---------------------------------------------------------------------------
INSERT INTO customers (id, email, country, signup_date, is_active)
SELECT
    g                                                                       AS id,
    'customer' || g || '@example.test'                                       AS email,
    (ARRAY['US','GB','DE','IN','BR','JP','CA','AU'])[
        1 + ((g::bigint * 2654435761 / 1000) % 8)                           ] AS country,
    TIMESTAMPTZ '2022-01-01 00:00:00+00'
        + ((g::bigint * 7919) % 126230400) * INTERVAL '1 second'            AS signup_date,
    (g % 11) <> 0                                                           AS is_active
FROM generate_series(1, :customers) AS g;

ANALYZE customers;

-- ---------------------------------------------------------------------------
-- orders: ~1M rows scattered across 24 months
--
-- status distribution (exact, by construction):
--     g % 97  =  0     -> 'cancelled'   ~1.03% of all rows
--     g % 50  =  0     -> 'pending'     2.0% of the remainder  <- the selective predicate
--     g % 50  =  1     -> 'paid'        2.0%
--     g % 50 <= 9      -> 'processing'  16.0%
--     otherwise        -> 'shipped'     80.0%
--
-- created_at is a deterministic *scatter* across the 24 month window rather
-- than a sorted run.  That is what makes `ORDER BY created_at DESC LIMIT 50`
-- genuinely require a sort in the absence of a usable index.
-- ---------------------------------------------------------------------------
INSERT INTO orders (id, customer_id, status, amount_cents, currency, created_at, updated_at)
SELECT
    g                                                                AS id,
    1 + mod(g::bigint * 2654435761, :customers)                      AS customer_id,
    CASE
        WHEN g % 97 = 0         THEN 'cancelled'
        WHEN g % 50 = 0         THEN 'pending'
        WHEN g % 50 = 1         THEN 'paid'
        WHEN g % 50 <= 9        THEN 'processing'
        ELSE 'shipped'
    END                                                               AS status,
    499 + ((g::bigint * 104729) % 499001)                            AS amount_cents,
    (ARRAY['USD','EUR','GBP','INR'])[1 + (g % 4)]                    AS currency,
    TIMESTAMPTZ '2024-01-01 00:00:00+00'
        + ((g::bigint * 7919) % 63072000) * INTERVAL '1 second'      AS created_at,
    TIMESTAMPTZ '2024-01-01 00:00:00+00'
        + ((g::bigint * 7919) % 63072000) * INTERVAL '1 second'
        + ((g % 86400) * INTERVAL '1 second')                        AS updated_at
FROM generate_series(1, :orders) AS g;

-- Statistics are part of the fixture, not an afterthought: a planner working
-- from default estimates would produce a different (and misleading) baseline.
ANALYZE orders;
ANALYZE customers;

-- ---------------------------------------------------------------------------
-- Report the fixture so a demo run can print exactly what it is measuring.
-- ---------------------------------------------------------------------------
SELECT
    (SELECT count(*) FROM customers)                              AS customers,
    (SELECT count(*) FROM orders)                                 AS orders,
    (SELECT count(*) FROM orders WHERE status = 'pending')       AS pending_orders,
    round(
        100.0 * (SELECT count(*) FROM orders WHERE status = 'pending')
        / NULLIF((SELECT count(*) FROM orders), 0), 3)           AS pending_pct,
    pg_size_pretty(pg_total_relation_size('orders'))             AS orders_size,
    (SELECT count(*) FROM pg_indexes
      WHERE schemaname = 'public' AND tablename = 'orders'
        AND indexname NOT LIKE '%_pkey')                         AS extra_order_indexes;
