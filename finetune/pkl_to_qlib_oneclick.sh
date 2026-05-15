#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
One-click pipeline:
  pkl -> per-symbol csv(with file column) -> qlib_data -> day_future

Usage:
  bash pkl_to_qlib_oneclick.sh \
    --pkl /path/input.pkl \
    --qlib-dir /path/output_qlib_dir \
    [--work-dir /tmp/pkl_to_qlib_work] \
    [--python-bin python] \
    [--dump-bin-script /path/to/dump_bin.py] \
    [--future-calendar-script /path/to/future_trading_date_collector.py] \
    [--future-calendar-ref /path/to/day_future.txt] \
    [--limit-symbols 0] \
    [--overwrite-qlib-dir]

Defaults:
  --work-dir               /tmp/pkl_to_qlib_work
  --python-bin             python
  --dump-bin-script        /home/fanjiahao/workspace/kronos/qlib-main/scripts/dump_bin.py
  --future-calendar-script /home/fanjiahao/workspace/kronos/qlib-main/scripts/data_collector/contrib/future_trading_date_collector/future_trading_date_collector.py
  --future-calendar-ref    /home/fanjiahao/workspace/kronos/qlib_data/A-data-20260410/calendars/day_future.txt
  --limit-symbols          0   (0 means all symbols)

Notes:
  1) The script expects a pickle object like: dict[str, pandas.DataFrame].
  2) It writes intermediate CSV files to: <work-dir>/stocks_with_id
  3) If future calendar script fails, it falls back to appending business days
     and generates calendars/day_future.txt.
USAGE
}

PKL_PATH=""
QLIB_DIR=""
WORK_DIR="/tmp/pkl_to_qlib_work"
PYTHON_BIN="python"
DUMP_BIN_SCRIPT="/home/fanjiahao/workspace/kronos/qlib-main/scripts/dump_bin.py"
FUTURE_CAL_SCRIPT="/home/fanjiahao/workspace/kronos/qlib-main/scripts/data_collector/contrib/future_trading_date_collector/future_trading_date_collector.py"
FUTURE_CAL_REF="/home/fanjiahao/workspace/kronos/qlib_data/A-data-20260410/calendars/day_future.txt"
LIMIT_SYMBOLS=0
OVERWRITE_QLIB_DIR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pkl)
      PKL_PATH="${2:-}"
      shift 2
      ;;
    --qlib-dir)
      QLIB_DIR="${2:-}"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --dump-bin-script)
      DUMP_BIN_SCRIPT="${2:-}"
      shift 2
      ;;
    --future-calendar-script)
      FUTURE_CAL_SCRIPT="${2:-}"
      shift 2
      ;;
    --future-calendar-ref)
      FUTURE_CAL_REF="${2:-}"
      shift 2
      ;;
    --limit-symbols)
      LIMIT_SYMBOLS="${2:-0}"
      shift 2
      ;;
    --overwrite-qlib-dir)
      OVERWRITE_QLIB_DIR=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PKL_PATH" || -z "$QLIB_DIR" ]]; then
  echo "ERROR: --pkl and --qlib-dir are required." >&2
  usage
  exit 1
fi

if [[ ! -f "$PKL_PATH" ]]; then
  echo "ERROR: pkl file not found: $PKL_PATH" >&2
  exit 1
fi

if [[ ! -f "$DUMP_BIN_SCRIPT" ]]; then
  echo "ERROR: dump_bin script not found: $DUMP_BIN_SCRIPT" >&2
  exit 1
fi

if ! [[ "$LIMIT_SYMBOLS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --limit-symbols must be an integer >= 0." >&2
  exit 1
fi

if [[ -d "$QLIB_DIR" ]] && [[ -n "$(ls -A "$QLIB_DIR" 2>/dev/null || true)" ]] && [[ "$OVERWRITE_QLIB_DIR" -ne 1 ]]; then
  echo "ERROR: qlib dir already exists and is not empty: $QLIB_DIR" >&2
  echo "       Use --overwrite-qlib-dir to recreate it." >&2
  exit 1
fi

if [[ "$OVERWRITE_QLIB_DIR" -eq 1 ]]; then
  rm -rf "$QLIB_DIR"
fi

mkdir -p "$WORK_DIR"
STOCKS_WITH_ID_DIR="$WORK_DIR/stocks_with_id"
rm -rf "$STOCKS_WITH_ID_DIR"
mkdir -p "$STOCKS_WITH_ID_DIR"
mkdir -p "$QLIB_DIR"

echo "[1/4] Convert pkl -> csv(with file column)"
"$PYTHON_BIN" - "$PKL_PATH" "$STOCKS_WITH_ID_DIR" "$LIMIT_SYMBOLS" <<'PY'
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pkl_path = Path(sys.argv[1]).expanduser().resolve()
out_dir = Path(sys.argv[2]).expanduser().resolve()
limit_symbols = int(sys.argv[3])

with pkl_path.open("rb") as f:
    data = pickle.load(f)

if not isinstance(data, dict):
    raise TypeError(f"Expected dict[str, DataFrame], got: {type(data)}")

symbols = sorted(data.keys())
if limit_symbols > 0:
    symbols = symbols[:limit_symbols]

written = 0
skipped_empty = 0
skipped_bad = 0

for sym in symbols:
    df = data.get(sym)
    if not isinstance(df, pd.DataFrame) or df.empty:
        skipped_empty += 1
        continue

    cur = df.copy()

    if "date" in cur.columns:
        pass
    elif "datetime" in cur.columns:
        cur = cur.rename(columns={"datetime": "date"})
    else:
        if isinstance(cur.index, pd.DatetimeIndex):
            cur = cur.reset_index()
            first_col = cur.columns[0]
            cur = cur.rename(columns={first_col: "date"})
        else:
            cur = cur.reset_index()
            first_col = cur.columns[0]
            parsed = pd.to_datetime(cur[first_col], errors="coerce")
            if parsed.notna().mean() < 0.8:
                skipped_bad += 1
                continue
            cur = cur.rename(columns={first_col: "date"})

    cur["date"] = pd.to_datetime(cur["date"], errors="coerce")
    cur = cur[cur["date"].notna()]
    if cur.empty:
        skipped_empty += 1
        continue

    cur["date"] = cur["date"].dt.strftime("%Y-%m-%d")
    cur = cur.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if cur.empty:
        skipped_empty += 1
        continue

    keep_cols = ["date"]
    for c in list(cur.columns):
        if c == "date":
            continue
        s = cur[c]
        if pd.api.types.is_numeric_dtype(s):
            keep_cols.append(c)
            continue
        converted = pd.to_numeric(s, errors="coerce")
        if converted.notna().sum() > 0:
            cur[c] = converted
            keep_cols.append(c)

    cur = cur[keep_cols]
    numeric_cols = [c for c in cur.columns if c != "date"]
    if numeric_cols:
        cur[numeric_cols] = cur[numeric_cols].replace([np.inf, -np.inf], np.nan)

    cur.insert(0, "file", str(sym))
    out_path = out_dir / f"{sym}.csv"
    cur.to_csv(out_path, index=False)
    written += 1

print(
    f"pkl={pkl_path}\n"
    f"symbols_in_pkl={len(data)}\n"
    f"symbols_selected={len(symbols)}\n"
    f"csv_written={written}\n"
    f"skipped_empty={skipped_empty}\n"
    f"skipped_bad={skipped_bad}\n"
    f"csv_dir={out_dir}"
)
PY

echo "[2/4] Convert csv -> qlib_data (dump_bin.py dump_all)"
"$PYTHON_BIN" "$DUMP_BIN_SCRIPT" dump_all \
  --data_path "$STOCKS_WITH_ID_DIR" \
  --qlib_dir "$QLIB_DIR" \
  --symbol_field_name file \
  --date_field_name date \
  --file_suffix .csv

echo "[3/4] Generate/copy day_future"
CAL_DIR="$QLIB_DIR/calendars"
mkdir -p "$CAL_DIR"

write_fallback_future_calendar() {
  "$PYTHON_BIN" - "$QLIB_DIR" <<'PY'
import sys
from pathlib import Path

import pandas as pd

qlib_dir = Path(sys.argv[1]).expanduser().resolve()
day_path = qlib_dir / "calendars" / "day.txt"
future_path = qlib_dir / "calendars" / "day_future.txt"

if not day_path.exists():
    raise FileNotFoundError(f"Missing day.txt: {day_path}")

day = pd.read_csv(day_path, header=None)[0].astype(str)
dates = pd.to_datetime(day, errors="coerce").dropna()
if dates.empty:
    raise RuntimeError("day.txt is empty or invalid.")

last_date = dates.max()
extra = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=400).strftime("%Y-%m-%d")
merged = sorted(set(day.tolist() + extra.tolist()))
pd.Series(merged).to_csv(future_path, index=False, header=False)
print(f"fallback day_future generated: {future_path} (rows={len(merged)})")
PY
}

if [[ -f "$FUTURE_CAL_REF" ]]; then
  cp "$FUTURE_CAL_REF" "$CAL_DIR/day_future.txt"
  echo "day_future copied from ref: $FUTURE_CAL_REF"
elif [[ -f "$FUTURE_CAL_SCRIPT" ]]; then
  set +e
  "$PYTHON_BIN" "$FUTURE_CAL_SCRIPT" --qlib_dir "$QLIB_DIR" --freq day
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "WARNING: future_calendar_collector failed, using fallback generator."
    write_fallback_future_calendar
  fi
else
  echo "WARNING: no ref and no future calendar script, using fallback generator."
  write_fallback_future_calendar
fi

echo "[4/4] Verify outputs"
"$PYTHON_BIN" - "$QLIB_DIR" <<'PY'
import sys
from pathlib import Path

qlib_dir = Path(sys.argv[1]).expanduser().resolve()
for rel in [
    "calendars/day.txt",
    "calendars/day_future.txt",
    "instruments/all.txt",
]:
    p = qlib_dir / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
print(f"qlib_dir ready: {qlib_dir}")
PY

echo "Done."
echo "  pkl:        $PKL_PATH"
echo "  csv_dir:    $STOCKS_WITH_ID_DIR"
echo "  qlib_dir:   $QLIB_DIR"
echo "  day_future: $QLIB_DIR/calendars/day_future.txt"
