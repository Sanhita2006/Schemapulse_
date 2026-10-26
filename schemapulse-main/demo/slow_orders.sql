-- ===========================================================================
-- SchemaPulse golden demo query
--
-- The "support inbox" query: show the 50 most recent orders that are still
-- pending, with the customer's email.
--
-- Why this query is a good optimization target:
--   * `status = 'pending'` matches ~2% of 1M rows, so a *partial* index can win
--     by an enormous margin while staying tiny.
--   * `ORDER BY created_at DESC LIMIT 50` needs the newest 50 matching rows
--     only, so an index on (created_at DESC) can satisfy the ordering without
--     a sort.
--   * The join to `customers` is a primary-key lookup on 50 rows — cheap, so the
--     measured cost is dominated by the scan of `orders`.
--
-- The fix is *not* obvious from "column X has no index". It requires reading the
-- plan, the statistics and the ORDER BY / LIMIT interaction together. That is
-- exactly the part a deterministic rule engine struggles with and a language
-- model is genuinely useful for.
-- ===========================================================================
SELECT
    o.id           AS order_id,
    o.created_at   AS created_at,
    o.amount_cents AS amount_cents,
    c.email        AS customer_email,
    c.country      AS customer_country
FROM orders o
JOIN customers c ON c.id = o.customer_id
WHERE o.status = 'pending'
ORDER BY o.created_at DESC
LIMIT 50;
