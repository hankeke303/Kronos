# pyright: reportMissingImports=false
import json
import os
import re
import sqlite3
from datetime import datetime
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request
from plotly.subplots import make_subplots

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_ROOT = os.path.join(PROJECT_ROOT, "outputs", "backtest_results")

TRACE_PREFIXES = {
    "return_curve": "return_curve_",
    "holdings": "holdings_snapshot_",
    "actions": "rebalance_actions_",
    "summary": "rebalance_summary_",
    "periods": "holding_periods_",
    "indicator": "indicator_raw_",
    "trade": "trade_details_",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
STOCK_NAME_CACHE_PATH = os.path.join(CACHE_DIR, "stock_name_map.json")
ENABLE_ONLINE_STOCK_NAME_LOOKUP = os.getenv("KRONOS_STOCK_NAME_ONLINE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
ONLINE_LOOKUP_BATCH_SIZE = 150


def _is_valid_stock_name(name: str, code: str = "") -> bool:
    name_s = str(name).strip()
    if not name_s:
        return False
    if name_s.lower() in {"nan", "none", "null"}:
        return False
    code_n = normalize_stock_code(code)
    if code_n and name_s == code_n:
        return False
    if re.fullmatch(r"\d+", name_s):
        return False
    return True


def normalize_stock_code(raw_code: str) -> str:
    s = str(raw_code).strip().upper().replace(".0", "")
    if not s:
        return s
    if s.isdigit() and len(s) < 6:
        return s.zfill(6)
    if "." in s:
        left = s.split(".", 1)[0]
        if left.isdigit():
            return left.zfill(6)
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)
    return s


def load_stock_name_cache() -> dict[str, str]:
    if not os.path.exists(STOCK_NAME_CACHE_PATH):
        return {}
    try:
        with open(STOCK_NAME_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            normalized = {}
            for k, v in data.items():
                code_n = normalize_stock_code(k)
                name_s = str(v).strip()
                if code_n and _is_valid_stock_name(name_s, code_n):
                    normalized[code_n] = name_s
            return normalized
    except Exception:
        return {}
    return {}


def save_stock_name_cache(mapping: dict[str, str]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(STOCK_NAME_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(mapping.items())), f, ensure_ascii=False, indent=2)


def _sina_symbol_for_code(code: str) -> str | None:
    code_n = normalize_stock_code(code)
    if not re.fullmatch(r"\d{6}", code_n):
        return None
    if code_n.startswith(("92", "83", "87", "43", "4", "8")):
        return f"bj{code_n}"
    if code_n.startswith(("5", "6", "9")):
        return f"sh{code_n}"
    return f"sz{code_n}"


def _fetch_stock_names_sina(codes: list[str]) -> dict[str, str]:
    symbols = []
    symbol_to_code = {}
    for code in codes:
        code_n = normalize_stock_code(code)
        symbol = _sina_symbol_for_code(code_n)
        if not symbol:
            continue
        symbols.append(symbol)
        symbol_to_code[symbol] = code_n

    if not symbols:
        return {}

    query = urllib.parse.quote(",".join(symbols), safe=",")
    url = f"https://hq.sinajs.cn/list={query}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = resp.read()
    except Exception:
        return {}

    text = raw.decode("gbk", errors="ignore")
    if not text.strip():
        text = raw.decode("utf-8", errors="ignore")

    result = {}
    for symbol, payload in re.findall(r'var\s+hq_str_([a-z]{2}\d{6})="([^"]*)";', text):
        code_n = symbol_to_code.get(symbol)
        if not code_n:
            continue
        parts = payload.split(",") if payload else []
        name_s = parts[0].strip() if parts else ""
        if _is_valid_stock_name(name_s, code_n):
            result[code_n] = name_s

    return result


def _ensure_stock_names(codes: list[str]):
    if not ENABLE_ONLINE_STOCK_NAME_LOOKUP:
        return

    normalized = []
    for code in codes:
        code_n = normalize_stock_code(code)
        if re.fullmatch(r"\d{6}", code_n):
            normalized.append(code_n)

    if not normalized:
        return

    unknown = sorted({code for code in normalized if code not in STOCK_NAME_MAP})
    if not unknown:
        return

    fetched_all = {}
    for i in range(0, len(unknown), ONLINE_LOOKUP_BATCH_SIZE):
        batch = unknown[i:i + ONLINE_LOOKUP_BATCH_SIZE]
        fetched = _fetch_stock_names_sina(batch)
        if fetched:
            fetched_all.update(fetched)

    if fetched_all:
        STOCK_NAME_MAP.update(fetched_all)
        save_stock_name_cache(STOCK_NAME_MAP)


def refresh_stock_name_cache() -> dict[str, str]:
    mapping = load_stock_name_cache()

    # Source 1: filtered_out_*.csv generated by filter report export.
    if os.path.isdir(BACKTEST_ROOT):
        for run_name in os.listdir(BACKTEST_ROOT):
            run_dir = os.path.join(BACKTEST_ROOT, run_name)
            if not os.path.isdir(run_dir):
                continue
            for file_name in os.listdir(run_dir):
                if not (file_name.startswith("filtered_out_") and file_name.endswith(".csv")):
                    continue
                csv_path = os.path.join(run_dir, file_name)
                try:
                    df = pd.read_csv(csv_path, low_memory=False)
                except Exception:
                    continue
                if "code" not in df.columns or "stock_name" not in df.columns:
                    continue
                for code, stock_name in zip(df["code"], df["stock_name"]):
                    code_n = normalize_stock_code(code)
                    name_s = str(stock_name).strip()
                    if code_n and _is_valid_stock_name(name_s, code_n):
                        mapping[code_n] = name_s

    # Source 2 (optional): status DB via env var KRONOS_STATUS_DB
    status_db_path = os.getenv("KRONOS_STATUS_DB", "").strip()
    if status_db_path and os.path.exists(status_db_path):
        try:
            conn = sqlite3.connect(status_db_path)
            cur = conn.cursor()
            cur.execute("SELECT code, name FROM stocks")
            for code, name in cur.fetchall():
                code_n = normalize_stock_code(code)
                name_s = str(name).strip() if name is not None else ""
                if code_n and _is_valid_stock_name(name_s, code_n):
                    mapping[code_n] = name_s
            conn.close()
        except Exception:
            pass

    save_stock_name_cache(mapping)
    return mapping


STOCK_NAME_MAP = refresh_stock_name_cache()


def lookup_stock_name(code: str) -> str:
    return STOCK_NAME_MAP.get(normalize_stock_code(code), "")


def instrument_label(code: str) -> str:
    code_n = normalize_stock_code(code)
    name = lookup_stock_name(code_n)
    return f"{code_n} {name}" if name else code_n


def safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        if "instrument" in df.columns:
            raw_inst = df["instrument"]
            ser = raw_inst.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            ser = ser.where(raw_inst.notna(), "")
            df["instrument"] = ser.apply(normalize_stock_code)
            _ensure_stock_names(df["instrument"].dropna().astype(str).tolist())
            df["instrument_name"] = df["instrument"].apply(lookup_stock_name)
        return df
    except Exception:
        return pd.DataFrame()


def ensure_datetime(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.DataFrame, str | None]:
    for col in candidates:
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col], errors="coerce")
            return df, col
    return df, None


def parse_signals_from_output(run_dir: str) -> list[str]:
    output_path = os.path.join(run_dir, "output.txt")
    if not os.path.exists(output_path):
        return []

    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return []

    signals = re.findall(r"Backtesting signal:\s*([^\.\n]+)\.\.\.", text)
    deduped = []
    for sig in signals:
        sig = sig.strip()
        if sig and sig not in deduped:
            deduped.append(sig)
    return deduped


def build_signal_file_map(run_dir: str, include_output_signals: bool = True) -> dict[str, dict[str, str]]:
    signal_map: dict[str, dict[str, str]] = {}
    if not os.path.isdir(run_dir):
        return signal_map

    for file_name in os.listdir(run_dir):
        if not file_name.endswith(".csv"):
            continue

        for key, prefix in TRACE_PREFIXES.items():
            if not file_name.startswith(prefix):
                continue
            signal_name = file_name[len(prefix):-4]
            signal_map.setdefault(signal_name, {})[key] = os.path.join(run_dir, file_name)

    if include_output_signals:
        for signal_name in parse_signals_from_output(run_dir):
            signal_map.setdefault(signal_name, {})

    return signal_map


def scan_runs() -> list[dict]:
    if not os.path.isdir(BACKTEST_ROOT):
        return []

    runs = []
    for name in os.listdir(BACKTEST_ROOT):
        run_dir = os.path.join(BACKTEST_ROOT, name)
        if not os.path.isdir(run_dir):
            continue

        try:
            mtime = os.path.getmtime(run_dir)
        except OSError:
            continue

        signal_map = build_signal_file_map(run_dir, include_output_signals=True)
        runs.append(
            {
                "name": name,
                "mtime": datetime.fromtimestamp(mtime),
                "mtime_text": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "signal_count": len(signal_map),
                "has_trace": any(bool(v) for v in signal_map.values()),
            }
        )

    runs.sort(key=lambda x: x["mtime"], reverse=True)
    return runs


def extract_signal_log_block(run_dir: str, signal_name: str) -> str:
    output_path = os.path.join(run_dir, "output.txt")
    if not os.path.exists(output_path):
        return ""

    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return ""

    pattern = rf"Backtesting signal:\s*{re.escape(signal_name)}\.\.\.(.*?)(?=\nBacktesting signal:|\Z)"
    matched = re.search(pattern, text, flags=re.S)
    if not matched:
        return ""

    block = matched.group(0).strip()
    return block[:10000]


def empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": "#6b5f4a"},
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
    )
    return fig


def build_return_figure(return_df: pd.DataFrame) -> go.Figure:
    if return_df.empty:
        return empty_figure("总体收益趋势", "该回测目录下没有 return_curve 文件")

    cur_df = return_df.copy()
    if "datetime" not in cur_df.columns:
        if "Unnamed: 0" in cur_df.columns:
            cur_df = cur_df.rename(columns={"Unnamed: 0": "datetime"})
        elif "index" in cur_df.columns:
            cur_df = cur_df.rename(columns={"index": "datetime"})

    if "datetime" not in cur_df.columns:
        return empty_figure("总体收益趋势", "return_curve 缺少 datetime 列")

    cur_df["datetime"] = pd.to_datetime(cur_df["datetime"], errors="coerce")
    cur_df = cur_df.dropna(subset=["datetime"]).sort_values("datetime")
    if cur_df.empty:
        return empty_figure("总体收益趋势", "return_curve 中无有效时间数据")

    fig = go.Figure()
    if "cum_bench" in cur_df.columns:
        fig.add_trace(
            go.Scatter(
                x=cur_df["datetime"],
                y=pd.to_numeric(cur_df["cum_bench"], errors="coerce"),
                mode="lines",
                name="cum_bench",
                line={"color": "#4f6d7a", "width": 2},
            )
        )
    if "cum_return_w_cost" in cur_df.columns:
        fig.add_trace(
            go.Scatter(
                x=cur_df["datetime"],
                y=pd.to_numeric(cur_df["cum_return_w_cost"], errors="coerce"),
                mode="lines",
                name="cum_return_w_cost",
                line={"color": "#15616d", "width": 2.8},
            )
        )
    if "cum_ex_return_w_cost" in cur_df.columns:
        fig.add_trace(
            go.Scatter(
                x=cur_df["datetime"],
                y=pd.to_numeric(cur_df["cum_ex_return_w_cost"], errors="coerce"),
                mode="lines",
                name="cum_ex_return_w_cost",
                line={"color": "#e76f51", "width": 2.4},
            )
        )

    fig.update_layout(
        title="总体收益趋势（点击日期可联动明细）",
        template="plotly_white",
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        margin={"l": 50, "r": 20, "t": 70, "b": 50},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
        xaxis={"title": "Date"},
        yaxis={"title": "Cumulative Return"},
    )
    return fig


def build_turnover_figure(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df.empty:
        return empty_figure("持仓与换手诊断", "该回测目录下没有 rebalance_summary 文件")

    summary_df, dt_col = ensure_datetime(summary_df, ["datetime", "date"])
    if dt_col is None:
        return empty_figure("持仓与换手诊断", "summary 缺少时间列")

    df = summary_df.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
    if df.empty:
        return empty_figure("持仓与换手诊断", "summary 中无有效时间数据")

    holding = pd.to_numeric(df.get("holding_count", np.nan), errors="coerce")
    if holding.isna().all() and "holding_list" in df.columns:
        holding = df["holding_list"].fillna("").astype(str).apply(lambda s: 0 if not s else len([x for x in s.split(";") if x]))

    buy = pd.to_numeric(df.get("buy_count", np.nan), errors="coerce").fillna(0)
    sell = pd.to_numeric(df.get("sell_count", np.nan), errors="coerce").fillna(0)
    turnover = (buy + sell) / holding.replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("每日持仓数量", "换手代理与调仓数量"),
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=holding, mode="lines", name="holding_count", line={"color": "#205072", "width": 2.5}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=turnover, mode="lines", name="turnover_proxy", line={"color": "#f08a5d", "width": 2.2}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=buy, name="buy_count", marker={"color": "#6a994e"}, opacity=0.65),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=sell, name="sell_count", marker={"color": "#bc4749"}, opacity=0.6),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="持仓与换手诊断",
        template="plotly_white",
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        margin={"l": 50, "r": 20, "t": 80, "b": 50},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        barmode="overlay",
        xaxis2={"title": "Date"},
        yaxis={"title": "Count"},
        yaxis2={"title": "Turnover / Actions"},
    )
    return fig


def build_trade_figure(trade_df: pd.DataFrame) -> go.Figure:
    if trade_df.empty:
        return empty_figure("成交明细诊断", "该回测目录下没有 trade_details 文件")

    trade_df, dt_col = ensure_datetime(trade_df, ["datetime", "date"])
    if dt_col is None:
        return empty_figure("成交明细诊断", "trade_details 缺少时间列")

    df = trade_df.dropna(subset=[dt_col]).copy()
    if df.empty:
        return empty_figure("成交明细诊断", "trade_details 中无有效时间数据")

    value = pd.to_numeric(df.get("trade_value", np.nan), errors="coerce")
    if value.notna().sum() == 0:
        value = pd.to_numeric(df.get("deal_amount", np.nan), errors="coerce").abs()
    daily_value = value.groupby(df[dt_col]).sum(min_count=1)

    action_series = df.get("action", pd.Series(index=df.index, dtype="object")).astype(str).str.upper()
    buy_daily = (action_series == "BUY").groupby(df[dt_col]).sum()
    sell_daily = (action_series == "SELL").groupby(df[dt_col]).sum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=daily_value.index,
            y=daily_value.values,
            name="trade_value_or_amount",
            marker={"color": "#1f7a8c"},
            opacity=0.75,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=buy_daily.index,
            y=buy_daily.values,
            mode="lines",
            name="BUY actions",
            line={"color": "#6a994e", "width": 2},
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=sell_daily.index,
            y=sell_daily.values,
            mode="lines",
            name="SELL actions",
            line={"color": "#bc4749", "width": 2},
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="成交额(或成交量)与买卖动作",
        template="plotly_white",
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        margin={"l": 50, "r": 50, "t": 70, "b": 50},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title_text="Trade Value / Amount", secondary_y=False)
    fig.update_yaxes(title_text="Action Count", secondary_y=True)
    return fig


def build_period_figure(periods_df: pd.DataFrame) -> go.Figure:
    if periods_df.empty:
        return empty_figure("持仓区间诊断", "该回测目录下没有 holding_periods 文件")

    days = pd.to_numeric(periods_df.get("holding_days", np.nan), errors="coerce").dropna()
    if days.empty:
        return empty_figure("持仓区间诊断", "holding_periods 缺少 holding_days 数据")

    top_inst = (
        periods_df.assign(holding_days_num=pd.to_numeric(periods_df["holding_days"], errors="coerce"))
        .dropna(subset=["holding_days_num"])
        .groupby("instrument", as_index=False)["holding_days_num"]
        .mean()
        .sort_values("holding_days_num", ascending=False)
        .head(12)
    )
    top_inst["instrument_label"] = top_inst["instrument"].astype(str).apply(instrument_label)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("持仓区间长度分布", "平均持仓天数 Top12 标的"))
    fig.add_trace(
        go.Histogram(x=days, nbinsx=30, marker={"color": "#ef8354"}, name="holding_days_dist"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=top_inst["instrument_label"],
            y=top_inst["holding_days_num"],
            marker={"color": "#2d6a4f"},
            name="top_instruments",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="持仓区间诊断",
        template="plotly_white",
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        margin={"l": 50, "r": 20, "t": 70, "b": 80},
        showlegend=False,
    )
    fig.update_xaxes(title="Holding Days", row=1, col=1)
    fig.update_xaxes(title="Instrument", row=1, col=2, tickangle=-40)
    fig.update_yaxes(title="Count", row=1, col=1)
    fig.update_yaxes(title="Avg Holding Days", row=1, col=2)
    return fig


def build_holdings_figure(holdings_df: pd.DataFrame) -> go.Figure:
    if holdings_df.empty:
        return empty_figure("持仓结构诊断", "该回测目录下没有 holdings_snapshot 文件")

    df = holdings_df.copy()
    if "instrument" not in df.columns:
        return empty_figure("持仓结构诊断", "holdings_snapshot 缺少 instrument 列")

    instrument_days = (
        df.groupby("instrument", as_index=False)
        .size()
        .rename(columns={"size": "holding_snapshots"})
        .sort_values("holding_snapshots", ascending=False)
        .head(20)
    )
    instrument_days["instrument_label"] = instrument_days["instrument"].astype(str).apply(instrument_label)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=instrument_days["instrument_label"],
            y=instrument_days["holding_snapshots"],
            marker={"color": "#355070"},
            name="holding_snapshots",
        )
    )
    fig.update_layout(
        title="持仓快照次数 Top20 标的",
        template="plotly_white",
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        margin={"l": 50, "r": 20, "t": 70, "b": 100},
        xaxis={"title": "Instrument", "tickangle": -45},
        yaxis={"title": "Snapshot Count"},
    )
    return fig


def build_stat_cards(
    summary_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    periods_df: pd.DataFrame,
    return_df: pd.DataFrame,
) -> list[dict[str, str]]:
    cards = []

    if summary_df.empty:
        cards.append({"label": "交易日覆盖", "value": "N/A"})
        cards.append({"label": "平均持仓数", "value": "N/A"})
        cards.append({"label": "平均换手代理", "value": "N/A"})
    else:
        summary_df, dt_col = ensure_datetime(summary_df, ["datetime", "date"])
        valid = summary_df.dropna(subset=[dt_col]) if dt_col else pd.DataFrame()
        day_count = valid[dt_col].nunique() if dt_col else 0
        holding_count = pd.to_numeric(valid.get("holding_count", np.nan), errors="coerce")
        buy = pd.to_numeric(valid.get("buy_count", np.nan), errors="coerce").fillna(0)
        sell = pd.to_numeric(valid.get("sell_count", np.nan), errors="coerce").fillna(0)
        turnover = (buy + sell) / holding_count.replace(0, np.nan)

        cards.append({"label": "交易日覆盖", "value": str(int(day_count))})
        cards.append({"label": "平均持仓数", "value": f"{holding_count.mean():.2f}" if holding_count.notna().any() else "N/A"})
        cards.append({"label": "平均换手代理", "value": f"{turnover.mean():.4f}" if turnover.notna().any() else "N/A"})

    if trade_df.empty:
        cards.append({"label": "总成交额(或量)", "value": "N/A"})
    else:
        value = pd.to_numeric(trade_df.get("trade_value", np.nan), errors="coerce")
        if value.notna().sum() == 0:
            value = pd.to_numeric(trade_df.get("deal_amount", np.nan), errors="coerce").abs()
        cards.append({"label": "总成交额(或量)", "value": f"{value.sum(skipna=True):,.2f}" if value.notna().any() else "N/A"})

    period_count = int(len(periods_df)) if not periods_df.empty else 0
    cards.append({"label": "持仓区间记录数", "value": str(period_count)})

    if return_df.empty:
        cards.append({"label": "期末超额收益", "value": "N/A"})
    else:
        cur_df = return_df.copy()
        if "datetime" not in cur_df.columns:
            if "Unnamed: 0" in cur_df.columns:
                cur_df = cur_df.rename(columns={"Unnamed: 0": "datetime"})
            elif "index" in cur_df.columns:
                cur_df = cur_df.rename(columns={"index": "datetime"})
        if "cum_ex_return_w_cost" in cur_df.columns and len(cur_df) > 0:
            final_ex = pd.to_numeric(cur_df["cum_ex_return_w_cost"], errors="coerce").dropna()
            cards.append({"label": "期末超额收益", "value": f"{final_ex.iloc[-1]:.4f}" if len(final_ex) else "N/A"})
        else:
            cards.append({"label": "期末超额收益", "value": "N/A"})

    return cards


def load_signal_data(run_dir: str, signal_name: str, signal_map: dict[str, dict[str, str]]) -> dict:
    file_map = signal_map.get(signal_name, {})

    return_curve_df = safe_read_csv(file_map.get("return_curve", ""))
    summary_df = safe_read_csv(file_map.get("summary", ""))
    holdings_df = safe_read_csv(file_map.get("holdings", ""))
    actions_df = safe_read_csv(file_map.get("actions", ""))
    periods_df = safe_read_csv(file_map.get("periods", ""))
    trade_df = safe_read_csv(file_map.get("trade", ""))
    indicator_df = safe_read_csv(file_map.get("indicator", ""))
    log_block = extract_signal_log_block(run_dir, signal_name)

    return {
        "return_curve": return_curve_df,
        "summary": summary_df,
        "holdings": holdings_df,
        "actions": actions_df,
        "periods": periods_df,
        "trade": trade_df,
        "indicator": indicator_df,
        "log_block": log_block,
        "file_map": file_map,
    }


def _normalize_date_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    normalized_df, date_col = ensure_datetime(df, ["datetime", "date", "trade_date"])
    return normalized_df, date_col


def _nearest_date(target_date: pd.Timestamp | None, date_values: pd.Series) -> pd.Timestamp | None:
    cleaned = pd.to_datetime(date_values, errors="coerce").dropna().sort_values().unique()
    if len(cleaned) == 0:
        return None
    if target_date is None:
        return pd.Timestamp(cleaned[-1]).normalize()

    target = pd.Timestamp(target_date).normalize()
    distances = [abs(pd.Timestamp(d).normalize() - target) for d in cleaned]
    nearest_idx = int(np.argmin(distances))
    return pd.Timestamp(cleaned[nearest_idx]).normalize()


def _records_for_date(df: pd.DataFrame, date_col: str, nearest_day: pd.Timestamp, max_rows: int = 30) -> list[dict]:
    if df.empty or date_col is None or nearest_day is None:
        return []

    cur = df.copy()
    cur[date_col] = pd.to_datetime(cur[date_col], errors="coerce")
    cur = cur.dropna(subset=[date_col])
    cur = cur[cur[date_col].dt.normalize() == nearest_day]
    if cur.empty:
        return []

    if "instrument" in cur.columns:
        _ensure_stock_names(cur["instrument"].dropna().astype(str).tolist())
        if "instrument_name" not in cur.columns:
            cur["instrument_name"] = cur["instrument"].astype(str).apply(lookup_stock_name)

    if "weight" in cur.columns:
        cur = cur.assign(weight_num=pd.to_numeric(cur["weight"], errors="coerce")).sort_values(
            "weight_num", ascending=False
        )
        cur = cur.drop(columns=["weight_num"])
    elif "score" in cur.columns:
        cur = cur.assign(score_num=pd.to_numeric(cur["score"], errors="coerce")).sort_values(
            "score_num", ascending=False
        )
        cur = cur.drop(columns=["score_num"])

    cur = cur.head(max_rows)

    for col in cur.columns:
        if str(cur[col].dtype).startswith("datetime"):
            has_time = (
                (cur[col].dt.hour.fillna(0) != 0)
                | (cur[col].dt.minute.fillna(0) != 0)
                | (cur[col].dt.second.fillna(0) != 0)
            ).any()
            fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
            cur[col] = cur[col].dt.strftime(fmt)
    cur = cur.replace({np.nan: None})
    return cur.to_dict("records")


def _augment_holdings_for_detail(holdings_df: pd.DataFrame) -> pd.DataFrame:
    if holdings_df.empty:
        return holdings_df

    df, date_col = _normalize_date_col(holdings_df)
    if date_col is None or "instrument" not in df.columns:
        return holdings_df

    df = df.copy()
    df["date_norm"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date_norm", "instrument"])
    if df.empty:
        return holdings_df

    _ensure_stock_names(df["instrument"].dropna().astype(str).tolist())
    if "instrument_name" not in df.columns:
        df["instrument_name"] = df["instrument"].astype(str).apply(lookup_stock_name)

    if "market_value" in df.columns:
        df["market_value_num"] = pd.to_numeric(df["market_value"], errors="coerce")
    else:
        amount = pd.to_numeric(df.get("amount", np.nan), errors="coerce")
        price = pd.to_numeric(df.get("price", np.nan), errors="coerce")
        df["market_value_num"] = amount * price

    all_days = sorted(df["date_norm"].unique())
    day_order = {d: i for i, d in enumerate(all_days)}

    df = df.sort_values(["instrument", "date_norm"]).copy()
    df["_ord"] = df["date_norm"].map(day_order)
    df["_seg"] = (
        df.groupby("instrument")["_ord"].diff().fillna(1).ne(1)
    ).groupby(df["instrument"]).cumsum()

    df["holding_duration_days"] = df.groupby(["instrument", "_seg"]).cumcount() + 1
    df["holding_start_date"] = df.groupby(["instrument", "_seg"])["date_norm"].transform("first")
    start_mv = df.groupby(["instrument", "_seg"])["market_value_num"].transform("first")
    df["holding_pnl"] = df["market_value_num"] - start_mv

    df = df.drop(columns=["_ord", "_seg"])
    return df


def _enrich_actions_with_position_changes(actions_df: pd.DataFrame, holdings_df: pd.DataFrame) -> pd.DataFrame:
    if actions_df.empty or holdings_df.empty:
        return actions_df
    if "instrument" not in actions_df.columns or "instrument" not in holdings_df.columns:
        return actions_df

    act_df, act_date_col = _normalize_date_col(actions_df)
    hold_df, hold_date_col = _normalize_date_col(holdings_df)
    if act_date_col is None or hold_date_col is None:
        return actions_df

    act_df = act_df.copy()
    hold_df = hold_df.copy()

    act_df["date_norm"] = pd.to_datetime(act_df[act_date_col], errors="coerce").dt.normalize()
    hold_df["date_norm"] = pd.to_datetime(hold_df[hold_date_col], errors="coerce").dt.normalize()
    hold_df["amount_num"] = pd.to_numeric(hold_df.get("amount", np.nan), errors="coerce")
    hold_df["price_num"] = pd.to_numeric(hold_df.get("price", np.nan), errors="coerce")
    hold_df = hold_df.dropna(subset=["date_norm", "instrument"])
    if hold_df.empty:
        return actions_df

    hold_base = (
        hold_df[["date_norm", "instrument", "amount_num", "price_num"]]
        .sort_values(["date_norm", "instrument"])
        .drop_duplicates(subset=["date_norm", "instrument"], keep="last")
    )

    day_df = pd.DataFrame({"date_norm": sorted(pd.Timestamp(d).normalize() for d in hold_base["date_norm"].unique())})
    day_df["prev_date_norm"] = day_df["date_norm"].shift(1)
    act_df = act_df.merge(day_df, on="date_norm", how="left")

    curr_hold = hold_base.rename(columns={"amount_num": "curr_amount", "price_num": "curr_price"})
    prev_hold = hold_base.rename(
        columns={
            "date_norm": "prev_date_norm",
            "amount_num": "prev_amount",
            "price_num": "prev_price",
        }
    )

    act_df = act_df.merge(
        curr_hold[["date_norm", "instrument", "curr_amount", "curr_price"]],
        on=["date_norm", "instrument"],
        how="left",
    )
    act_df = act_df.merge(
        prev_hold[["prev_date_norm", "instrument", "prev_amount", "prev_price"]],
        on=["prev_date_norm", "instrument"],
        how="left",
    )

    act_df["prev_amount"] = pd.to_numeric(act_df.get("prev_amount", np.nan), errors="coerce").fillna(0.0)
    act_df["curr_amount"] = pd.to_numeric(act_df.get("curr_amount", np.nan), errors="coerce").fillna(0.0)
    act_df["delta_amount"] = act_df["curr_amount"] - act_df["prev_amount"]

    trade_price = pd.to_numeric(act_df.get("curr_price", np.nan), errors="coerce")
    trade_price = trade_price.fillna(pd.to_numeric(act_df.get("prev_price", np.nan), errors="coerce"))
    act_df["trade_shares"] = act_df["delta_amount"].abs()
    act_df["trade_amount"] = act_df["trade_shares"] * trade_price

    action_upper = act_df.get("action", pd.Series(index=act_df.index, dtype="object")).astype(str).str.upper()
    abs_trade_amount = pd.to_numeric(act_df.get("trade_amount", np.nan), errors="coerce").abs()
    act_df["cash_flow"] = np.where(
        action_upper == "BUY",
        -abs_trade_amount,
        np.where(action_upper == "SELL", abs_trade_amount, np.sign(act_df["delta_amount"]) * abs_trade_amount),
    )

    if "instrument" in act_df.columns and "instrument_name" not in act_df.columns:
        _ensure_stock_names(act_df["instrument"].dropna().astype(str).tolist())
        act_df["instrument_name"] = act_df["instrument"].astype(str).apply(lookup_stock_name)

    act_df = act_df.drop(columns=["prev_date_norm", "curr_price", "prev_price"], errors="ignore")
    return act_df


def _stock_contribution_ranking(
    holdings_df: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict]:
    columns = ["instrument", "contribution", "contribution_rate", "active_days"]
    if holdings_df.empty:
        return pd.DataFrame(columns=columns), {"message": "no holdings data"}

    df, date_col = _normalize_date_col(holdings_df)
    if date_col is None:
        return pd.DataFrame(columns=columns), {"message": "missing datetime column"}

    if "instrument" not in df.columns:
        return pd.DataFrame(columns=columns), {"message": "missing instrument column"}

    if "amount" not in df.columns or "price" not in df.columns:
        return pd.DataFrame(columns=columns), {"message": "missing amount/price columns"}

    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["amount_num"] = pd.to_numeric(df["amount"], errors="coerce")
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
    if "market_value" in df.columns:
        df["market_value_num"] = pd.to_numeric(df["market_value"], errors="coerce")
    else:
        df["market_value_num"] = df["amount_num"] * df["price_num"]

    df = df.dropna(subset=["date", "instrument", "amount_num", "price_num"])
    if df.empty:
        return pd.DataFrame(columns=columns), {"message": "no valid amount/price rows"}

    df = df.sort_values(["instrument", "date"])
    df["prev_amount"] = df.groupby("instrument")["amount_num"].shift(1)
    df["prev_price"] = df.groupby("instrument")["price_num"].shift(1)

    # Approximate per-stock daily contribution: previous day position * price change.
    df["daily_contribution"] = df["prev_amount"] * (df["price_num"] - df["prev_price"])
    df["daily_contribution"] = pd.to_numeric(df["daily_contribution"], errors="coerce").fillna(0.0)

    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= df["date"] >= pd.Timestamp(start_date).normalize()
    if end_date is not None:
        mask &= df["date"] <= pd.Timestamp(end_date).normalize()
    scoped = df[mask].copy()

    if scoped.empty:
        return pd.DataFrame(columns=columns), {"message": "no rows in selected date range"}

    agg = (
        scoped.groupby("instrument", as_index=False)
        .agg(
            contribution=("daily_contribution", "sum"),
            active_days=("date", "nunique"),
            avg_market_value=("market_value_num", "mean"),
        )
        .sort_values("contribution", ascending=False)
    )
    agg["contribution_rate"] = agg["contribution"] / agg["avg_market_value"].replace(0, np.nan)
    agg = agg[["instrument", "contribution", "contribution_rate", "active_days"]]

    meta = {
        "start_date": scoped["date"].min().strftime("%Y-%m-%d"),
        "end_date": scoped["date"].max().strftime("%Y-%m-%d"),
        "days": int(scoped["date"].nunique()),
        "total_contribution": float(agg["contribution"].sum()),
        "instrument_count": int(len(agg)),
    }
    return agg, meta


@app.route("/")
def index():
    runs = scan_runs()
    selected_run = request.args.get("run", "")

    run_names = [r["name"] for r in runs]
    if selected_run not in run_names:
        selected_run = run_names[0] if run_names else ""

    signal_map = {}
    signals = []
    selected_signal = request.args.get("signal", "")
    data = {
        "return_curve": pd.DataFrame(),
        "summary": pd.DataFrame(),
        "holdings": pd.DataFrame(),
        "actions": pd.DataFrame(),
        "periods": pd.DataFrame(),
        "trade": pd.DataFrame(),
        "indicator": pd.DataFrame(),
        "log_block": "",
        "file_map": {},
    }

    if selected_run:
        run_dir = os.path.join(BACKTEST_ROOT, selected_run)
        signal_map = build_signal_file_map(run_dir, include_output_signals=True)
        signals = sorted(signal_map.keys())

        if selected_signal not in signals:
            selected_signal = signals[0] if signals else ""

        if selected_signal:
            data = load_signal_data(run_dir, selected_signal, signal_map)

    return_curve_df = data["return_curve"]
    summary_df = data["summary"]
    holdings_df = data["holdings"]
    periods_df = data["periods"]
    trade_df = data["trade"]

    fig_return = build_return_figure(return_curve_df)
    fig_turnover = build_turnover_figure(summary_df)
    fig_trade = build_trade_figure(trade_df)
    fig_period = build_period_figure(periods_df)
    fig_holdings = build_holdings_figure(holdings_df)

    chart_config = {"displayModeBar": True, "responsive": True}
    charts = {
        "return_trend": fig_return.to_html(full_html=False, include_plotlyjs=False, config=chart_config),
        "turnover": fig_turnover.to_html(full_html=False, include_plotlyjs=False, config=chart_config),
        "trade": fig_trade.to_html(full_html=False, include_plotlyjs=False, config=chart_config),
        "period": fig_period.to_html(full_html=False, include_plotlyjs=False, config=chart_config),
        "holdings": fig_holdings.to_html(full_html=False, include_plotlyjs=False, config=chart_config),
    }

    stat_cards = build_stat_cards(summary_df, trade_df, periods_df, return_curve_df)

    return render_template(
        "index.html",
        runs=runs,
        selected_run=selected_run,
        signals=signals,
        selected_signal=selected_signal,
        selected_run_safe=selected_run,
        selected_signal_safe=selected_signal,
        charts=charts,
        stat_cards=stat_cards,
        log_block=data["log_block"],
        trace_files=data["file_map"],
        backtest_root=BACKTEST_ROOT,
    )


@app.route("/api/date-details")
def api_date_details():
    run_name = request.args.get("run", "")
    signal_name = request.args.get("signal", "")
    date_text = request.args.get("date", "")

    runs = scan_runs()
    valid_run_names = {r["name"] for r in runs}
    if run_name not in valid_run_names:
        return {"ok": False, "error": "invalid run"}, 400

    run_dir = os.path.join(BACKTEST_ROOT, run_name)
    signal_map = build_signal_file_map(run_dir, include_output_signals=True)
    if signal_name not in signal_map:
        return {"ok": False, "error": "invalid signal"}, 400

    data = load_signal_data(run_dir, signal_name, signal_map)
    return_df = data["return_curve"]
    summary_df = data["summary"]
    raw_holdings_df = data["holdings"]
    holdings_df = _augment_holdings_for_detail(raw_holdings_df)
    actions_df = _enrich_actions_with_position_changes(data["actions"], raw_holdings_df)

    if "instrument" in actions_df.columns and "instrument_name" not in actions_df.columns:
        actions_df = actions_df.copy()
        actions_df["instrument_name"] = actions_df["instrument"].astype(str).apply(lookup_stock_name)

    target_date = pd.to_datetime(date_text, errors="coerce") if date_text else None

    ref_df = return_df if not return_df.empty else summary_df
    ref_df, ref_date_col = _normalize_date_col(ref_df)
    nearest = _nearest_date(target_date, ref_df[ref_date_col]) if ref_date_col else None
    if nearest is None:
        return {"ok": False, "error": "no available dates"}, 404

    summary_df, summary_date_col = _normalize_date_col(summary_df)
    holdings_df, holdings_date_col = _normalize_date_col(holdings_df)
    actions_df, actions_date_col = _normalize_date_col(actions_df)

    summary_records = _records_for_date(summary_df, summary_date_col, nearest, max_rows=1)
    holdings_records = _records_for_date(holdings_df, holdings_date_col, nearest, max_rows=30)
    actions_records = _records_for_date(actions_df, actions_date_col, nearest, max_rows=40)

    return {
        "ok": True,
        "requested_date": date_text,
        "nearest_date": nearest.strftime("%Y-%m-%d"),
        "summary": summary_records[0] if summary_records else {},
        "holdings": holdings_records,
        "actions": actions_records,
    }


@app.route("/api/stock-contrib")
def api_stock_contrib():
    run_name = request.args.get("run", "")
    signal_name = request.args.get("signal", "")
    start_text = request.args.get("start", "")
    end_text = request.args.get("end", "")

    runs = scan_runs()
    valid_run_names = {r["name"] for r in runs}
    if run_name not in valid_run_names:
        return {"ok": False, "error": "invalid run"}, 400

    run_dir = os.path.join(BACKTEST_ROOT, run_name)
    signal_map = build_signal_file_map(run_dir, include_output_signals=True)
    if signal_name not in signal_map:
        return {"ok": False, "error": "invalid signal"}, 400

    data = load_signal_data(run_dir, signal_name, signal_map)
    holdings_df = data["holdings"]

    start_dt = pd.to_datetime(start_text, errors="coerce") if start_text else None
    end_dt = pd.to_datetime(end_text, errors="coerce") if end_text else None
    if start_dt is not None and pd.isna(start_dt):
        start_dt = None
    if end_dt is not None and pd.isna(end_dt):
        end_dt = None

    ranking_df, meta = _stock_contribution_ranking(holdings_df, start_dt, end_dt)
    if ranking_df.empty:
        return {"ok": False, "error": meta.get("message", "no ranking data")}, 404

    top_positive = ranking_df.head(30).copy()
    top_negative = ranking_df.sort_values("contribution", ascending=True).head(30).copy()

    top_positive["instrument_name"] = top_positive["instrument"].astype(str).apply(lookup_stock_name)
    top_negative["instrument_name"] = top_negative["instrument"].astype(str).apply(lookup_stock_name)

    for part_df in (top_positive, top_negative):
        part_df["contribution"] = pd.to_numeric(part_df["contribution"], errors="coerce").round(4)
        part_df["contribution_rate"] = pd.to_numeric(part_df["contribution_rate"], errors="coerce").round(6)
        part_df["active_days"] = pd.to_numeric(part_df["active_days"], errors="coerce").fillna(0).astype(int)

    return {
        "ok": True,
        "meta": meta,
        "top_positive": top_positive.replace({np.nan: None}).to_dict("records"),
        "top_negative": top_negative.replace({np.nan: None}).to_dict("records"),
    }


@app.route("/health")
def health():
    return {"status": "ok", "backtest_root": BACKTEST_ROOT, "exists": os.path.isdir(BACKTEST_ROOT)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8780, debug=True)
