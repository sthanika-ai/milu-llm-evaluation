"""Export results/results.db into diff-friendly CSV tables under results/tables/.

results.db (SQLite) is gitignored -- see .gitignore's rationale -- so these CSVs are the
form the results layer actually gets committed in. Read-only: never writes back to
results.db. Reuses pipeline/results_db.py's own query_all(), the same function
pipeline/run.py's orchestration and every ingest/retry script insert through, so this
script sees exactly the rows that are actually there, not a reimplementation of the schema.

Usage:
    python scripts/export_results_tables.py
    python scripts/export_results_tables.py --db-path /path/to/results.db --out-dir results/tables
"""
import argparse
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pipeline import results_db  # noqa: E402


def _write_csv(path: str, rows: list, fieldnames: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=results_db.RESULTS_DB_PATH)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "tables"))
    args = parser.parse_args()

    rows = results_db.query_all(args.db_path)
    if not rows:
        print(f"[export] no rows in {args.db_path} -- nothing to export.")
        return

    overall = [r for r in rows if r["breakdown_type"] == "overall"]
    by_language = [r for r in rows if r["breakdown_type"] == "language"]
    by_domain = [r for r in rows if r["breakdown_type"] == "domain"]

    league_fields = ["model_id", "run_id", "benchmark", "accuracy", "n_items", "n_malformed",
                      "malformed_rate", "real_cost_usd", "run_timestamp"]
    breakdown_fields = ["model_id", "run_id", "breakdown_key", "accuracy", "n_items",
                         "n_malformed", "malformed_rate", "run_timestamp"]

    overall_sorted = sorted(overall, key=lambda r: (r["accuracy"] is None, -(r["accuracy"] or 0)))
    _write_csv(os.path.join(args.out_dir, "league_table.csv"), overall_sorted, league_fields)
    _write_csv(os.path.join(args.out_dir, "by_language.csv"),
               sorted(by_language, key=lambda r: (r["model_id"], r["breakdown_key"])), breakdown_fields)
    _write_csv(os.path.join(args.out_dir, "by_domain.csv"),
               sorted(by_domain, key=lambda r: (r["model_id"], r["breakdown_key"])), breakdown_fields)

    print(f"[export] {len(overall)} overall rows -> league_table.csv")
    print(f"[export] {len(by_language)} language-breakdown rows -> by_language.csv")
    print(f"[export] {len(by_domain)} domain-breakdown rows -> by_domain.csv")
    print(f"[export] wrote tables to {args.out_dir}")


if __name__ == "__main__":
    main()
