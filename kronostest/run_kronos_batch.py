import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from kronostest import run_kronos_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_kronos_test for multiple run_ids with signal_key=all"
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help="List of run_id values to process in order",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output files already exist",
    )
    return parser.parse_args()


def _discover_run_ids() -> List[str]:
    base_dir = Path(__file__).resolve().parents[1] / "outputs" / "backtest_results"
    if not base_dir.exists():
        return []

    prefix = "finetune_backtest_demo_"
    run_ids: List[str] = []
    for path in sorted(base_dir.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            run_ids.append(path.name[len(prefix):])
    return run_ids


def _summaries_for_run(run_id: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for signal, df in results.items():
        if df is None or df.empty:
            summaries.append(
                {
                    "run_id": run_id,
                    "signal": signal,
                    "metric": "best",
                    "buythreshold": None,
                    "sellthreshold": None,
                    "finalmoney": None,
                    "maxhuiche": None,
                    "buyacc": None,
                }
            )
            continue

        df = df.reset_index(drop=True)
        best_row = df.sort_values("finalmoney", ascending=False).iloc[0]
        summaries.append(
            {
                "run_id": run_id,
                "signal": signal,
                "metric": "best",
                "buythreshold": best_row.get("buythreshold"),
                "sellthreshold": best_row.get("sellthreshold"),
                "finalmoney": best_row.get("finalmoney"),
                "maxhuiche": best_row.get("maxhuiche"),
                "buyacc": best_row.get("buyacc"),
            }
        )

        if "modelname" in df.columns and (df["modelname"] == "avg").any():
            avg_row = df[df["modelname"] == "avg"].iloc[0]
            summaries.append(
                {
                    "run_id": run_id,
                    "signal": signal,
                    "metric": "avg",
                    "buythreshold": avg_row.get("buythreshold"),
                    "sellthreshold": avg_row.get("sellthreshold"),
                    "finalmoney": avg_row.get("finalmoney"),
                    "maxhuiche": avg_row.get("maxhuiche"),
                    "buyacc": avg_row.get("buyacc"),
                }
            )

    return summaries


def main() -> None:
    args = parse_args()
    run_ids: List[str] = args.run_ids

    if "all" in run_ids:
        discovered = _discover_run_ids()
        if not discovered:
            print("No run_ids found under outputs/backtest_results with prefix finetune_backtest_demo_.")
            return
        run_ids = discovered
        print(f"Discovered run_ids from outputs/backtest_results: {run_ids}")

    all_rows: List[Dict[str, Any]] = []

    for run_id in run_ids:
        print(f"\n===== run_id {run_id} (signal_key=all) =====")
        try:
            results = run_kronos_test(run_id=run_id, signal_key="all", skip_existing=args.skip_existing, inter=10)
        except Exception as exc:  # continue other runs even if one fails
            print(f"run_id {run_id} failed: {exc}")
            continue

        rows = _summaries_for_run(run_id, results)
        all_rows.extend(rows)

        if rows:
            per_run_df = pd.DataFrame(rows)
            print(per_run_df.to_string(index=False))
        else:
            print("No rows returned for this run_id.")

    if not all_rows:
        print("\nNo successful runs to compare.")
        return

    comparison_df = pd.DataFrame(all_rows)

    best_rows = comparison_df[comparison_df["metric"] == "best"]
    if not best_rows.empty:
        print("\n===== Best rows per run_id & signal (sorted by finalmoney desc) =====")
        print(best_rows.sort_values(["signal", "finalmoney"], ascending=[True, False]).to_string(index=False))

    avg_rows = comparison_df[comparison_df["metric"] == "avg"]
    if not avg_rows.empty:
        print("\n===== Avg rows per run_id & signal =====")
        print(avg_rows.sort_values(["signal", "finalmoney"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
