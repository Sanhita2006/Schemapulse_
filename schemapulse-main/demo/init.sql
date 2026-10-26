-- ===========================================================================
-- SchemaPulse demo schema
--
-- Two tables is enough to make the optimization problem real:
--   customers : dimension, 20k rows, indexed by primary key
--   orders    : fact, ~1M rows, NO useful index beyond the primary key
--
-- `orders` is deliberately left under-indexed so that the golden query has to
-- fall back to a sequential scan.  The speedup SchemaPulse later measures comes
-- from a genuine planner decision changing, not from an artificial delay.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS customers (
    id           BIGINT PRIMARY KEY,
    email        TEXT        NOT NULL,
    country      TEXT        NOT NULL,
    signup_date  TIMESTAMPTZ NOT NULL,
    is_active    BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS orders (
    id           BIGINT PRIMARY KEY,
    customer_id  BIGINT       NOT NULL REFERENCES customers (id),
    status       TEXT         NOT NULL,
    amount_cents BIGINT       NOT NULL,
    currency     TEXT         NOT NULL DEFAULT 'USD',
    created_at   TIMESTAMPTZ  NOT NULL,
    updated_at   TIMESTAMPTZ  NOT NULL
);

-- A realistic e-commerce partial index would exist in a mature system.  We do
-- NOT create it: its absence is the problem SchemaPulse has to solve.
--
-- A secondary index on customer_id is intentionally omitted too, so the
-- golden query's cost is dominated by the orders scan.

COMMENT ON TABLE orders IS
    'Demo fact table for SchemaPulse. Deliberately missing a partial index on (status, created_at).';
COMMENT ON COLUMN orders.status IS
    'Low-selectivity column: ~2% pending. Textbook partial-index candidate.';
