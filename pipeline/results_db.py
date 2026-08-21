"""Queryable results layer: a small SQLite table that feeds the report and,
later, the public leaderboard. One row per (run, breakdown_type, breakdown_key).
"""
import os
import sqlite3

RESULTS_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "results.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    breakdown_type TEXT NOT NULL,   -- 'overall' | 'language' | 'domain'
    breakdown_key TEXT NOT NULL,    -- 'all' | language name | domain name
    accuracy REAL,                  -- correct / (n_items - n_malformed); NULL if all malformed
    n_items INTEGER NOT NULL,
    n_malformed INTEGER NOT NULL,   -- refusals/format failures, excluded from accuracy (not scored wrong)
    malformed_rate REAL,
    run_timestamp TEXT NOT NULL,
    real_cost_usd REAL,             -- whole-run cost from measured API token usage; only set on the
                                     -- 'overall'/'all' row (cost isn't a per-language/domain quantity),
                                     -- NULL for local models (no cost tracking -- GPU-hours, not $, brief's
                                     -- other cost axis, not yet instrumented)
    PRIMARY KEY (run_id, breakdown_type, breakdown_key)
);
"""


def get_connection(db_path: str = RESULTS_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    # CREATE TABLE IF NOT EXISTS above doesn't add new columns to an already-existing
    # table -- migrate in place (ALTER TABLE, not drop/recreate) so real rows already
    # in results.db from earlier runs are never touched, just gain a NULL-default column.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    if "real_cost_usd" not in existing_cols:
        conn.execute("ALTER TABLE results ADD COLUMN real_cost_usd REAL")
        conn.commit()
    return conn


def insert_scored_run(conn: sqlite3.Connection, run_id: str, model_id: str, benchmark: str, run_timestamp: str, scores: dict,
                       real_cost_usd: float = None):
    rows = [("overall", "all", scores["overall"])]
    rows += [("language", lang, s) for lang, s in scores["by_language"].items()]
    rows += [("domain", dom, s) for dom, s in scores["by_domain"].items()]

    conn.executemany(
        """INSERT OR REPLACE INTO results
           (run_id, model_id, benchmark, breakdown_type, breakdown_key, accuracy, n_items, n_malformed, malformed_rate, run_timestamp, real_cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (run_id, model_id, benchmark, bt, bk, s["accuracy"], s["n_items"], s["n_malformed"], s["malformed_rate"], run_timestamp,
             real_cost_usd if bt == "overall" else None)
            for bt, bk, s in rows
        ],
    )
    conn.commit()


def query_all(db_path: str = RESULTS_DB_PATH):
    conn = get_connection(db_path)
    cur = conn.execute("SELECT * FROM results ORDER BY model_id, breakdown_type, breakdown_key")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
