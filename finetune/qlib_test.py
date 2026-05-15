import os
import sys
import argparse
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import trange, tqdm
from matplotlib import pyplot as plt

import qlib
from qlib.config import REG_CN
from qlib.backtest import backtest, executor, CommonInfrastructure
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils import flatten_dict
from qlib.utils.time import Freq

# Ensure project root is in the Python path
sys.path.append("../")
from config import Config
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference
from utils.training_utils import setup_ddp, cleanup_ddp, set_seed

from finetune.dataset import calc_extra_features

# =================================================================================
# 1. Data Loading and Processing for Inference
# =================================================================================

class QlibTestDataset(Dataset):
    """
    PyTorch Dataset for handling Qlib test data, specifically for inference.

    This dataset iterates through all possible sliding windows sequentially. It also
    yields metadata like symbol and timestamp, which are crucial for mapping
    predictions back to the original time series.
    """

    def __init__(self, data: dict, config: Config):
        self.data = data
        self.config = config
        self.window_size = config.lookback_window + config.predict_window
        self.symbols = list(self.data.keys())
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list
        self.indices = []
        self.backtest_start = pd.Timestamp(config.backtest_time_range[0])
        self.backtest_end = pd.Timestamp(config.backtest_time_range[1])
        # 可选项：允许在序列最开始时使用“短上下文”。
        # 关闭时：必须至少有 lookback_window 个历史点才会生成该时点信号。
        # 开启时：允许上下文从该股票数据起点开始，一直到当前锚点。
        self.allow_partial_context = getattr(config, "backtest_allow_partial_context", False)
        
        self.data = calc_extra_features(self.data)

        print("Preprocessing and building indices for test dataset...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            # Generate time features on-the-fly
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df  # Store preprocessed dataframe

            # anchor_idx 表示“当前用于产出信号的时点”在该股票序列中的位置。
            # 为了保证右侧未来窗口完整，需要满足：anchor_idx + predict_window < len(df)
            # 因此可取到的最大锚点下标是 len(df) - predict_window - 1。
            max_anchor_idx = len(df) - self.config.predict_window - 1
            if max_anchor_idx < 0:
                continue

            # 最小锚点下标取值规则：
            # 1) 短上下文开启：从 0 开始，表示最早时点也允许产生信号；
            # 2) 短上下文关闭：从 lookback_window-1 开始，保持原先“必须凑满 lookback”的行为。
            min_anchor_idx = 0 if self.allow_partial_context else self.config.lookback_window - 1
            if min_anchor_idx > max_anchor_idx:
                continue

            # 遍历本股票所有可用于回测的锚点：
            # 每个 anchor_idx 会映射成一个样本 (context -> future predict_window)。
            for anchor_idx in range(min_anchor_idx, max_anchor_idx + 1):
                timestamp = df.iloc[anchor_idx]['datetime']
                if self.backtest_start <= pd.Timestamp(timestamp) <= self.backtest_end:
                    # context_end 采用右开区间写法，因此需要 +1 才包含 anchor_idx 对应时点。
                    context_end = anchor_idx + 1
                    # context_start 是上下文左边界：
                    # - 开启短上下文：左边界最多退到序列起点 0；
                    # - 关闭短上下文：严格固定为 context_end - lookback_window。
                    if self.allow_partial_context:
                        context_start = max(0, context_end - self.config.lookback_window)
                    else:
                        context_start = context_end - self.config.lookback_window
                    self.indices.append((symbol, context_start, context_end, timestamp))

        print(f"Filtered inference windows by backtest range [{self.backtest_start.date()} - {self.backtest_end.date()}], total samples: {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        symbol, context_start, context_end, timestamp = self.indices[idx]
        df = self.data[symbol]

        predict_end = context_end + self.config.predict_window

        context_df = df.iloc[context_start:context_end]
        predict_df = df.iloc[context_end:predict_end]

        x = context_df[self.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.time_feature_list].values.astype(np.float32)
        y_stamp = predict_df[self.time_feature_list].values.astype(np.float32)

        # Instance-level normalization, consistent with training
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        context_len = context_end - context_start
        return torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp), symbol, timestamp, context_len


# =================================================================================
# 2. Backtesting Logic
# =================================================================================

class QlibBacktest:
    """
    A wrapper class for conducting backtesting experiments using Qlib.
    """

    def __init__(self, config: Config):
        self.config = config
        self.backtest_output_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "outputs",
                "backtest_results",
                self.config.backtest_save_folder_name,
            )
        )
        os.makedirs(self.backtest_output_dir, exist_ok=True)
        self.output_txt_path = os.path.join(self.backtest_output_dir, "output.txt")
        self.initialize_qlib()

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib for backtesting...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def _append_output_log(self, text: str):
        with open(self.output_txt_path, 'a') as f:
            f.write(text)

    @staticmethod
    def _safe_call(obj, method_name: str, *args):
        fn = getattr(obj, method_name, None)
        if not callable(fn):
            return None
        try:
            return fn(*args)
        except Exception:
            return None

    @staticmethod
    def _sanitize_signal_name(signal_name: str) -> str:
        keep_chars = []
        for ch in signal_name:
            if ch.isalnum() or ch in {"_", "-"}:
                keep_chars.append(ch)
            else:
                keep_chars.append("_")
        return "".join(keep_chars) or "signal"

    @staticmethod
    def _build_signal_rank_df(signal_series: pd.Series) -> pd.DataFrame:
        columns = ["datetime", "instrument", "score", "score_rank", "universe_size"]
        if signal_series is None or signal_series.empty:
            return pd.DataFrame(columns=columns)

        signal_df = signal_series.rename("score").reset_index()
        if signal_df.shape[1] < 3:
            return pd.DataFrame(columns=columns)

        signal_df.columns = ["instrument", "datetime", "score"]
        signal_df["datetime"] = pd.to_datetime(signal_df["datetime"], errors="coerce")
        signal_df = signal_df.dropna(subset=["datetime"])

        signal_df["score_rank"] = signal_df.groupby("datetime")["score"].rank(method="first", ascending=False)
        signal_df["universe_size"] = signal_df.groupby("datetime")["instrument"].transform("size")
        return signal_df[columns]

    def _build_holdings_df(
        self,
        signal_name: str,
        positions,
        signal_rank_df: pd.DataFrame,
    ) -> pd.DataFrame:
        columns = [
            "signal_name",
            "datetime",
            "instrument",
            "amount",
            "price",
            "weight",
            "market_value",
            "hold_days",
            "cash",
            "account_value",
            "score",
            "score_rank",
            "universe_size",
        ]

        if not isinstance(positions, dict):
            return pd.DataFrame(columns=columns)

        rows = []
        reserved = {
            "cash",
            "cash_delay",
            "today_account_value",
            "now_account_value",
            "account_value",
            "accum_info",
        }

        for raw_dt, pos in sorted(positions.items(), key=lambda kv: pd.Timestamp(kv[0])):
            dt = pd.Timestamp(raw_dt)

            raw_position = getattr(pos, "position", None)
            if raw_position is None and isinstance(pos, dict):
                raw_position = pos.get("position", pos)
            if not isinstance(raw_position, dict):
                raw_position = {}

            stock_meta = {}
            for key, value in raw_position.items():
                if not isinstance(key, str) or key in reserved:
                    continue
                if isinstance(value, dict):
                    stock_meta[key] = value
                else:
                    stock_meta[key] = {"amount": value}

            stock_list = self._safe_call(pos, "get_stock_list")
            if stock_list is None:
                stock_list = list(stock_meta.keys())

            cash = self._safe_call(pos, "get_cash")
            if cash is None:
                cash = raw_position.get("cash")

            account_value = self._safe_call(pos, "calculate_value")
            if account_value is None:
                account_value = raw_position.get("today_account_value", raw_position.get("now_account_value"))

            weight_dict = self._safe_call(pos, "get_stock_weight_dict")
            if not isinstance(weight_dict, dict):
                weight_dict = {}

            for instrument in stock_list:
                instrument = str(instrument)
                meta = stock_meta.get(instrument, {})

                amount = self._safe_call(pos, "get_stock_amount", instrument)
                if amount is None:
                    amount = meta.get("amount")

                price = self._safe_call(pos, "get_stock_price", instrument)
                if price is None:
                    price = meta.get("price")

                hold_days = self._safe_call(pos, "get_stock_count", instrument)
                if hold_days is None:
                    hold_days = meta.get("count")

                weight = weight_dict.get(instrument)
                if weight is None:
                    weight = meta.get("weight")

                market_value = meta.get("value")
                if market_value is None and amount is not None and price is not None:
                    try:
                        market_value = float(amount) * float(price)
                    except (TypeError, ValueError):
                        market_value = None

                rows.append(
                    {
                        "signal_name": signal_name,
                        "datetime": dt,
                        "instrument": instrument,
                        "amount": amount,
                        "price": price,
                        "weight": weight,
                        "market_value": market_value,
                        "hold_days": hold_days,
                        "cash": cash,
                        "account_value": account_value,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=columns)

        holdings_df = pd.DataFrame(rows)
        if not signal_rank_df.empty:
            holdings_df = holdings_df.merge(
                signal_rank_df,
                on=["datetime", "instrument"],
                how="left",
            )
        else:
            holdings_df["score"] = np.nan
            holdings_df["score_rank"] = np.nan
            holdings_df["universe_size"] = np.nan

        return holdings_df[columns].sort_values(["datetime", "instrument"]).reset_index(drop=True)

    def _build_rebalance_frames(
        self,
        signal_name: str,
        holdings_df: pd.DataFrame,
        signal_rank_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        action_columns = [
            "signal_name",
            "datetime",
            "instrument",
            "action",
            "score",
            "score_rank",
            "universe_size",
            "prev_hold_days",
            "curr_hold_days",
            "reason",
        ]
        summary_columns = [
            "signal_name",
            "datetime",
            "holding_count",
            "buy_count",
            "sell_count",
            "buy_list",
            "sell_list",
            "holding_list",
        ]

        if holdings_df.empty:
            return pd.DataFrame(columns=action_columns), pd.DataFrame(columns=summary_columns)

        holdings_df = holdings_df.copy()
        holdings_df["datetime"] = pd.to_datetime(holdings_df["datetime"], errors="coerce")
        holdings_df = holdings_df.dropna(subset=["datetime"])

        signal_lookup = {}
        if not signal_rank_df.empty:
            score_ref = signal_rank_df.copy()
            score_ref["datetime"] = pd.to_datetime(score_ref["datetime"], errors="coerce")
            score_ref = score_ref.dropna(subset=["datetime"])
            signal_lookup = score_ref.set_index(["datetime", "instrument"])[
                ["score", "score_rank", "universe_size"]
            ].to_dict("index")

        hold_days_lookup = holdings_df.set_index(["datetime", "instrument"])["hold_days"].to_dict()

        date_to_holdings = {
            pd.Timestamp(dt): set(df_dt["instrument"].astype(str).tolist())
            for dt, df_dt in holdings_df.groupby("datetime")
        }
        sorted_dates = sorted(date_to_holdings.keys())

        action_rows = []
        summary_rows = []
        prev_holdings = set()
        prev_date = None

        def score_info(cur_dt: pd.Timestamp, inst: str):
            data = signal_lookup.get((cur_dt, inst), None)
            if data is None:
                return np.nan, np.nan, np.nan
            return data.get("score", np.nan), data.get("score_rank", np.nan), data.get("universe_size", np.nan)

        for dt in sorted_dates:
            current_holdings = date_to_holdings[dt]
            buy_list = sorted(current_holdings - prev_holdings)
            sell_list = sorted(prev_holdings - current_holdings)

            summary_rows.append(
                {
                    "signal_name": signal_name,
                    "datetime": dt,
                    "holding_count": len(current_holdings),
                    "buy_count": len(buy_list),
                    "sell_count": len(sell_list),
                    "buy_list": ";".join(buy_list),
                    "sell_list": ";".join(sell_list),
                    "holding_list": ";".join(sorted(current_holdings)),
                }
            )

            for inst in buy_list:
                score, rank, universe_size = score_info(dt, inst)
                reason = "entered holdings"
                if pd.notna(rank):
                    rank_i = int(rank)
                    universe_i = int(universe_size) if pd.notna(universe_size) else -1
                    if rank_i <= self.config.backtest_n_symbol_hold:
                        reason = (
                            f"score rank {rank_i}/{universe_i} in topk={self.config.backtest_n_symbol_hold}, entered holdings"
                        )
                    else:
                        reason = f"entered by turnover/hold constraint, rank {rank_i}/{universe_i}"

                action_rows.append(
                    {
                        "signal_name": signal_name,
                        "datetime": dt,
                        "instrument": inst,
                        "action": "BUY",
                        "score": score,
                        "score_rank": rank,
                        "universe_size": universe_size,
                        "prev_hold_days": np.nan,
                        "curr_hold_days": hold_days_lookup.get((dt, inst), np.nan),
                        "reason": reason,
                    }
                )

            for inst in sell_list:
                score, rank, universe_size = score_info(dt, inst)
                prev_hold_days = hold_days_lookup.get((prev_date, inst), np.nan) if prev_date is not None else np.nan
                reason = "removed from holdings"
                if pd.notna(rank):
                    rank_i = int(rank)
                    universe_i = int(universe_size) if pd.notna(universe_size) else -1
                    if rank_i > self.config.backtest_n_symbol_hold:
                        reason = (
                            f"score rank {rank_i}/{universe_i} out of topk={self.config.backtest_n_symbol_hold}, removed"
                        )
                    else:
                        reason = f"removed by n_drop/turnover control, rank {rank_i}/{universe_i}"
                elif pd.notna(prev_hold_days) and prev_hold_days < self.config.backtest_hold_thresh:
                    reason = (
                        f"removed though hold_days={int(prev_hold_days)} < hold_thresh={self.config.backtest_hold_thresh}"
                    )

                action_rows.append(
                    {
                        "signal_name": signal_name,
                        "datetime": dt,
                        "instrument": inst,
                        "action": "SELL",
                        "score": score,
                        "score_rank": rank,
                        "universe_size": universe_size,
                        "prev_hold_days": prev_hold_days,
                        "curr_hold_days": np.nan,
                        "reason": reason,
                    }
                )

            prev_holdings = current_holdings
            prev_date = dt

        actions_df = pd.DataFrame(action_rows, columns=action_columns).sort_values(
            ["datetime", "action", "instrument"]
        )
        summary_df = pd.DataFrame(summary_rows, columns=summary_columns).sort_values("datetime")
        return actions_df, summary_df

    def _build_holding_periods_df(
        self,
        signal_name: str,
        holdings_df: pd.DataFrame,
        signal_rank_df: pd.DataFrame,
    ) -> pd.DataFrame:
        columns = [
            "signal_name",
            "instrument",
            "start_datetime",
            "end_datetime",
            "holding_days",
            "entry_score",
            "entry_rank",
            "exit_score",
            "exit_rank",
            "avg_weight",
            "avg_market_value",
            "max_hold_days",
        ]

        if holdings_df.empty:
            return pd.DataFrame(columns=columns)

        signal_lookup = {}
        if not signal_rank_df.empty:
            score_ref = signal_rank_df.copy()
            score_ref["datetime"] = pd.to_datetime(score_ref["datetime"], errors="coerce")
            score_ref = score_ref.dropna(subset=["datetime"])
            signal_lookup = score_ref.set_index(["datetime", "instrument"])[["score", "score_rank"]].to_dict("index")

        df = holdings_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        all_dates = sorted(df["datetime"].unique())
        date_order = {pd.Timestamp(dt): idx for idx, dt in enumerate(all_dates)}

        rows = []
        for instrument, inst_df in df.groupby("instrument"):
            inst_df = inst_df.sort_values("datetime").copy()
            inst_df["date_order"] = inst_df["datetime"].map(date_order)
            inst_df["segment"] = (inst_df["date_order"].diff().fillna(1) != 1).cumsum()

            for _, seg_df in inst_df.groupby("segment"):
                start_dt = pd.Timestamp(seg_df["datetime"].iloc[0])
                end_dt = pd.Timestamp(seg_df["datetime"].iloc[-1])
                holding_days = int(seg_df.shape[0])

                entry_info = signal_lookup.get((start_dt, instrument), {})
                exit_info = signal_lookup.get((end_dt, instrument), {})

                rows.append(
                    {
                        "signal_name": signal_name,
                        "instrument": instrument,
                        "start_datetime": start_dt,
                        "end_datetime": end_dt,
                        "holding_days": holding_days,
                        "entry_score": entry_info.get("score", np.nan),
                        "entry_rank": entry_info.get("score_rank", np.nan),
                        "exit_score": exit_info.get("score", np.nan),
                        "exit_rank": exit_info.get("score_rank", np.nan),
                        "avg_weight": (
                            pd.to_numeric(seg_df["weight"], errors="coerce").mean() if "weight" in seg_df else np.nan
                        ),
                        "avg_market_value": (
                            pd.to_numeric(seg_df["market_value"], errors="coerce").mean()
                            if "market_value" in seg_df
                            else np.nan
                        ),
                        "max_hold_days": (
                            pd.to_numeric(seg_df["hold_days"], errors="coerce").max()
                            if "hold_days" in seg_df
                            else np.nan
                        ),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns).sort_values(["instrument", "start_datetime"])

    @staticmethod
    def _object_to_dataframe(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if isinstance(obj, pd.Series):
            return obj.to_frame(name=obj.name or "value").reset_index()

        for method_name in ("to_dataframe", "to_frame", "to_df"):
            fn = getattr(obj, method_name, None)
            if not callable(fn):
                continue
            try:
                converted = fn()
            except Exception:
                continue
            if isinstance(converted, pd.DataFrame):
                return converted.copy()
            if isinstance(converted, pd.Series):
                return converted.to_frame(name=converted.name or "value").reset_index()

        if isinstance(obj, dict):
            try:
                if obj and all(
                    not isinstance(v, (dict, list, tuple, set, pd.DataFrame, pd.Series))
                    for v in obj.values()
                ):
                    return pd.DataFrame([obj])
            except Exception:
                return None
        return None

    def _collect_indicator_frames(self, obj, source: str, frames: list, visited: set, depth: int = 0):
        if obj is None or depth > 5:
            return

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        df = self._object_to_dataframe(obj)
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append((source, df))
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                self._collect_indicator_frames(value, f"{source}.{key}", frames, visited, depth + 1)
            return

        if isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                self._collect_indicator_frames(value, f"{source}.{idx}", frames, visited, depth + 1)
            return

        for attr in (
            "order_indicator_his",
            "trade_indicator_his",
            "order_history",
            "trade_history",
            "history",
            "records",
            "indicator",
            "indicators",
        ):
            if not hasattr(obj, attr):
                continue
            try:
                value = getattr(obj, attr)
            except Exception:
                continue
            self._collect_indicator_frames(value, f"{source}.{attr}", frames, visited, depth + 1)

    def _build_indicator_raw_df(self, signal_name: str, indicator_data) -> pd.DataFrame:
        frames = []
        self._collect_indicator_frames(indicator_data, "indicator", frames, visited=set(), depth=0)

        if not frames:
            return pd.DataFrame(columns=["signal_name", "source", "datetime", "instrument"])

        normalized = []
        for source, df in frames:
            cur_df = df.copy()
            if not isinstance(cur_df.index, pd.RangeIndex):
                cur_df = cur_df.reset_index()

            for old_name in ("date", "trade_date", "time", "index"):
                if old_name in cur_df.columns and "datetime" not in cur_df.columns:
                    cur_df = cur_df.rename(columns={old_name: "datetime"})
            for old_name in ("stock_id", "symbol", "ticker", "code"):
                if old_name in cur_df.columns and "instrument" not in cur_df.columns:
                    cur_df = cur_df.rename(columns={old_name: "instrument"})

            if "datetime" in cur_df.columns:
                cur_df["datetime"] = pd.to_datetime(cur_df["datetime"], errors="coerce")

            cur_df["signal_name"] = signal_name
            cur_df["source"] = source
            normalized.append(cur_df)

        if not normalized:
            return pd.DataFrame(columns=["signal_name", "source", "datetime", "instrument"])
        return pd.concat(normalized, axis=0, ignore_index=True, sort=False)

    @staticmethod
    def _normalize_direction(value):
        if pd.isna(value):
            return np.nan
        value_str = str(value).strip().lower()
        if value_str in {"buy", "b", "long", "1", "true"}:
            return "BUY"
        if value_str in {"sell", "s", "short", "-1", "false"}:
            return "SELL"
        return np.nan

    def _build_trade_detail_df(
        self,
        signal_name: str,
        indicator_raw_df: pd.DataFrame,
        actions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        columns = [
            "signal_name",
            "datetime",
            "instrument",
            "action",
            "deal_amount",
            "trade_price",
            "trade_value",
            "cost",
            "ffr",
            "pa",
            "trade_reason",
            "trade_source",
        ]

        rename_alias = {
            "deal_amount": ["deal_amount", "amount", "volume", "trade_amount", "filled_amount"],
            "trade_price": ["trade_price", "price", "deal_price", "avg_price"],
            "trade_value": ["trade_value", "value", "deal_value", "turnover"],
            "cost": ["cost", "transaction_cost", "fee", "commission"],
            "ffr": ["ffr", "fulfill_rate", "fill_rate"],
            "pa": ["pa", "price_advantage"],
            "direction": ["direction", "side", "order_dir", "buy_or_sell"],
            "reason": ["reason", "message", "msg", "note", "desc"],
        }

        trade_frames = []
        if not indicator_raw_df.empty:
            for source_name, df_src in indicator_raw_df.groupby("source"):
                cur_df = df_src.copy()

                for target_name, alias_list in rename_alias.items():
                    if target_name in cur_df.columns:
                        continue
                    for alias in alias_list:
                        if alias in cur_df.columns:
                            cur_df = cur_df.rename(columns={alias: target_name})
                            break

                if "datetime" in cur_df.columns:
                    cur_df["datetime"] = pd.to_datetime(cur_df["datetime"], errors="coerce")

                has_inst = "instrument" in cur_df.columns
                has_trade_info = any(col in cur_df.columns for col in ["deal_amount", "trade_value", "trade_price"]) 
                if not (has_inst and has_trade_info):
                    continue

                direction = cur_df["direction"].apply(self._normalize_direction) if "direction" in cur_df else np.nan
                if isinstance(direction, pd.Series):
                    action = direction
                else:
                    action = pd.Series(np.nan, index=cur_df.index)

                if "deal_amount" in cur_df.columns:
                    deal_amount_num = pd.to_numeric(cur_df["deal_amount"], errors="coerce")
                    action = action.where(action.notna(), np.where(deal_amount_num >= 0, "BUY", "SELL"))

                frame = pd.DataFrame(
                    {
                        "signal_name": signal_name,
                        "datetime": cur_df["datetime"] if "datetime" in cur_df else pd.NaT,
                        "instrument": cur_df["instrument"].astype(str),
                        "action": action,
                        "deal_amount": cur_df["deal_amount"] if "deal_amount" in cur_df else np.nan,
                        "trade_price": cur_df["trade_price"] if "trade_price" in cur_df else np.nan,
                        "trade_value": cur_df["trade_value"] if "trade_value" in cur_df else np.nan,
                        "cost": cur_df["cost"] if "cost" in cur_df else np.nan,
                        "ffr": cur_df["ffr"] if "ffr" in cur_df else np.nan,
                        "pa": cur_df["pa"] if "pa" in cur_df else np.nan,
                        "trade_reason": cur_df["reason"] if "reason" in cur_df else np.nan,
                        "trade_source": source_name,
                    }
                )
                trade_frames.append(frame)

        if trade_frames:
            trade_df = pd.concat(trade_frames, axis=0, ignore_index=True, sort=False)
            trade_df = trade_df.dropna(subset=["datetime", "instrument"], how="any")
            trade_df = trade_df.sort_values(["datetime", "instrument", "action"]).reset_index(drop=True)
            return trade_df[columns]

        # Fallback: no indicator trade details extracted, use position-diff actions.
        if actions_df.empty:
            return pd.DataFrame(columns=columns)

        fallback_df = actions_df.copy()
        fallback_df["signal_name"] = signal_name
        fallback_df["deal_amount"] = np.nan
        fallback_df["trade_price"] = np.nan
        fallback_df["trade_value"] = np.nan
        fallback_df["cost"] = np.nan
        fallback_df["ffr"] = np.nan
        fallback_df["pa"] = np.nan
        fallback_df["trade_reason"] = fallback_df.get("reason", np.nan)
        fallback_df["trade_source"] = "position_diff_fallback"
        return fallback_df[columns].sort_values(["datetime", "instrument", "action"]).reset_index(drop=True)

    def _export_trace_files(
        self,
        signal_name: str,
        signal_series: pd.Series,
        report_df: pd.DataFrame,
        positions,
        indicator_data=None,
    ) -> dict[str, str]:
        signal_rank_df = self._build_signal_rank_df(signal_series)
        holdings_df = self._build_holdings_df(signal_name, positions, signal_rank_df)
        actions_df, summary_df = self._build_rebalance_frames(signal_name, holdings_df, signal_rank_df)
        periods_df = self._build_holding_periods_df(signal_name, holdings_df, signal_rank_df)
        indicator_raw_df = self._build_indicator_raw_df(signal_name, indicator_data)
        trade_detail_df = self._build_trade_detail_df(signal_name, indicator_raw_df, actions_df)

        safe_signal_name = self._sanitize_signal_name(signal_name)
        holdings_path = os.path.join(self.backtest_output_dir, f"holdings_snapshot_{safe_signal_name}.csv")
        actions_path = os.path.join(self.backtest_output_dir, f"rebalance_actions_{safe_signal_name}.csv")
        summary_path = os.path.join(self.backtest_output_dir, f"rebalance_summary_{safe_signal_name}.csv")
        periods_path = os.path.join(self.backtest_output_dir, f"holding_periods_{safe_signal_name}.csv")
        indicator_raw_path = os.path.join(self.backtest_output_dir, f"indicator_raw_{safe_signal_name}.csv")
        trade_detail_path = os.path.join(self.backtest_output_dir, f"trade_details_{safe_signal_name}.csv")
        return_curve_path = os.path.join(self.backtest_output_dir, f"return_curve_{safe_signal_name}.csv")

        holdings_df.to_csv(holdings_path, index=False)
        actions_df.to_csv(actions_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        periods_df.to_csv(periods_path, index=False)
        indicator_raw_df.to_csv(indicator_raw_path, index=False)
        trade_detail_df.to_csv(trade_detail_path, index=False)
        report_df.to_csv(return_curve_path, index=True)

        return {
            "holdings": holdings_path,
            "actions": actions_path,
            "summary": summary_path,
            "periods": periods_path,
            "indicator_raw": indicator_raw_path,
            "trade_detail": trade_detail_path,
            "return_curve": return_curve_path,
        }

    def run_single_backtest(self, signal_series: pd.Series, signal_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Runs a single backtest for a given prediction signal.

        Args:
            signal_series (pd.Series): A pandas Series with a MultiIndex
                                       (instrument, datetime) and prediction scores.
        Returns:
            tuple[pd.DataFrame, dict[str, str]]:
                A cumulative return DataFrame and exported trace file paths.
        """
        strategy = TopkDropoutStrategy(
            topk=self.config.backtest_n_symbol_hold,
            n_drop=self.config.backtest_n_symbol_drop,
            hold_thresh=self.config.backtest_hold_thresh,
            signal=signal_series,
        )
        executor_config = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "delay_execution": True,
        }
        backtest_config = {
            "start_time": self.config.backtest_time_range[0],
            "end_time": self.config.backtest_time_range[1],
            "account": 100_000_000,
            "benchmark": self.config.backtest_benchmark,
            "exchange_kwargs": {
                "freq": "day", "limit_threshold": 0.095, "deal_price": "open",
                "open_cost": 0.001, "close_cost": 0.0015, "min_cost": 5,
            },
            "executor": executor.SimulatorExecutor(**executor_config),
        }

        portfolio_metric_dict, indicator_dict = backtest(strategy=strategy, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse("day"))
        report, positions = portfolio_metric_dict.get(analysis_freq)
        indicator_data = indicator_dict.get(analysis_freq) if isinstance(indicator_dict, dict) else indicator_dict

        report_df = pd.DataFrame({
            "cum_bench": report["bench"].cumsum(),
            "cum_return_w_cost": (report["return"] - report["cost"]).cumsum(),
            "cum_ex_return_w_cost": (report["return"] - report["bench"] - report["cost"]).cumsum(),
        })

        trace_file_paths = self._export_trace_files(
            signal_name,
            signal_series,
            report_df,
            positions,
            indicator_data=indicator_data,
        )

        # --- Analysis and Reporting ---
        analysis = {
            "excess_return_without_cost": risk_analysis(report["return"] - report["bench"], freq=analysis_freq),
            "excess_return_with_cost": risk_analysis(report["return"] - report["bench"] - report["cost"], freq=analysis_freq),
        }
        print("\n--- Backtest Analysis ---")
        print("Benchmark Return:", risk_analysis(report["bench"], freq=analysis_freq), sep='\n')
        print("\nExcess Return (w/o cost):", analysis["excess_return_without_cost"], sep='\n')
        print("\nExcess Return (w/ cost):", analysis["excess_return_with_cost"], sep='\n')

        self._append_output_log("\n--- Backtest Analysis ---\n")
        self._append_output_log("Benchmark Return:\n")
        self._append_output_log(str(risk_analysis(report["bench"], freq=analysis_freq)))
        self._append_output_log("\n\nExcess Return (w/o cost):\n")
        self._append_output_log(str(analysis["excess_return_without_cost"]))
        self._append_output_log("\n\nExcess Return (w/ cost):\n")
        self._append_output_log(str(analysis["excess_return_with_cost"]))
        self._append_output_log(f"\n\nreport_df: \n{report_df}\n")
        self._append_output_log(
            "\nTrace files:\n"
            f"holdings snapshot: {trace_file_paths['holdings']}\n"
            f"rebalance actions: {trace_file_paths['actions']}\n"
            f"rebalance summary: {trace_file_paths['summary']}\n"
            f"holding periods: {trace_file_paths['periods']}\n"
            f"indicator raw: {trace_file_paths['indicator_raw']}\n"
            f"trade details: {trace_file_paths['trade_detail']}\n"
            f"return curve: {trace_file_paths['return_curve']}\n"
        )
        return report_df, trace_file_paths

    def _plot_rebalance_diagnostics(
        self,
        summary_by_signal: dict[str, pd.DataFrame],
        trade_by_signal: dict[str, pd.DataFrame],
        save_path: str,
    ):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        has_data = False

        for signal_name, summary_df in summary_by_signal.items():
            if summary_df is None or summary_df.empty:
                continue

            df = summary_df.copy()
            if "datetime" not in df.columns:
                continue
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
            if df.empty:
                continue

            holding_count = pd.to_numeric(df.get("holding_count", np.nan), errors="coerce")
            buy_count = pd.to_numeric(df.get("buy_count", np.nan), errors="coerce")
            sell_count = pd.to_numeric(df.get("sell_count", np.nan), errors="coerce")
            turnover_proxy = (buy_count + sell_count) / holding_count.replace(0, np.nan)

            axes[0].plot(df.index, holding_count, label=signal_name)
            axes[1].plot(df.index, turnover_proxy, label=signal_name)
            axes[2].plot(df.index, buy_count, label=f"{signal_name}_buy", alpha=0.8)
            axes[2].plot(df.index, sell_count, label=f"{signal_name}_sell", linestyle="--", alpha=0.8)
            has_data = True

        for signal_name, trade_df in trade_by_signal.items():
            if trade_df is None or trade_df.empty:
                continue
            if "datetime" not in trade_df.columns:
                continue

            df = trade_df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            if df.empty:
                continue

            trade_value = pd.to_numeric(df.get("trade_value", np.nan), errors="coerce")
            if trade_value.notna().sum() == 0:
                trade_value = pd.to_numeric(df.get("deal_amount", np.nan), errors="coerce").abs()
            daily_trade_value = trade_value.groupby(df["datetime"]).sum(min_count=1)
            axes[2].plot(daily_trade_value.index, daily_trade_value.values, label=f"{signal_name}_trade_value")
            has_data = True

        if not has_data:
            plt.close(fig)
            return

        axes[0].set_title("Holding Count by Date")
        axes[0].set_ylabel("Count")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_title("Turnover Proxy by Date")
        axes[1].set_ylabel("(buy + sell) / holding")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].set_title("Rebalance Actions and Trade Value")
        axes[2].set_ylabel("Actions / Value")
        axes[2].set_xlabel("Date")
        axes[2].grid(True)
        axes[2].legend(ncol=2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.show()

        self._append_output_log(f"\nRebalance diagnostics figure: {save_path}\n")

    def run_and_plot_results(self, signals: dict[str, pd.DataFrame], save_path: str):
        """
        Runs backtests for multiple signals and plots the cumulative return curves.

        Args:
            signals (dict[str, pd.DataFrame]): A dictionary where keys are signal names
                                               and values are prediction DataFrames.
        """
        return_df, ex_return_df, bench_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        summary_by_signal, trade_by_signal = {}, {}

        for signal_name, pred_df in signals.items():
            print(f"\nBacktesting signal: {signal_name}...")
            pred_series = pred_df.stack()
            pred_series.index.names = ['datetime', 'instrument']
            pred_series = pred_series.swaplevel().sort_index()
            # 现在的 pred_series 的格式应该是左边有两列，第一列是不同的 instrument，每个 instrument 下面是不同的 datetime 列出，内容只有一栏 score

            self._append_output_log(f"\nBacktesting signal: {signal_name}...\n")

            report_df, trace_file_paths = self.run_single_backtest(pred_series, signal_name)

            return_df[signal_name] = report_df['cum_return_w_cost']
            ex_return_df[signal_name] = report_df['cum_ex_return_w_cost']
            if 'return' not in bench_df:
                bench_df['return'] = report_df['cum_bench']

            try:
                summary_by_signal[signal_name] = pd.read_csv(trace_file_paths["summary"])
            except Exception:
                summary_by_signal[signal_name] = pd.DataFrame()
            try:
                trade_by_signal[signal_name] = pd.read_csv(trace_file_paths["trade_detail"])
            except Exception:
                trade_by_signal[signal_name] = pd.DataFrame()

        # Plotting results
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        return_df.plot(ax=axes[0], title='Cumulative Return with Cost', grid=True)
        axes[0].plot(bench_df['return'], label=self.config.instrument.upper(), color='black', linestyle='--')
        axes[0].legend()
        axes[0].set_ylabel("Cumulative Return")

        ex_return_df.plot(ax=axes[1], title='Cumulative Excess Return with Cost', grid=True)
        axes[1].legend()
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Cumulative Excess Return")

        plt.tight_layout()
        # img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "backtest_result_untrained.png"))
        img_path = save_path
        plt.savefig(img_path, dpi=200)
        plt.show()

        diag_path = os.path.abspath(
            os.path.join(
                os.path.dirname(save_path),
                f"rebalance_diagnostics_{self.config.backtest_save_folder_name}.png",
            )
        )
        self._plot_rebalance_diagnostics(summary_by_signal, trade_by_signal, diag_path)


# =================================================================================
# 3. Inference Logic
# =================================================================================

def load_models(config: dict, device: torch.device, rank: int) -> tuple[KronosTokenizer, Kronos]:
    """Loads the fine-tuned tokenizer and predictor model."""
    print(f"[Rank {rank}] Loading models onto device: {device}...")
    tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_path']).to(device).eval()
    model = Kronos.from_pretrained(config['model_path']).to(device).eval()
    return tokenizer, model


def collate_fn_for_inference(batch):
    """
    Custom collate function to handle batches containing Tensors, strings, and Timestamps.

    Args:
        batch (list): A list of samples, where each sample is the tuple returned by
                      QlibTestDataset.__getitem__.

    Returns:
        A single tuple containing padded tensors and metadata.
    """
    # Unzip the list of samples into separate lists for each data type
    x, x_stamp, y_stamp, symbols, timestamps, context_lens = zip(*batch)

    # 采用左侧 padding 对齐变长上下文，保持最后一个时间步始终是“当前锚点”。
    max_context_len = max(int(v) for v in context_lens)

    def left_pad_to_len(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = target_len - tensor.size(0)
        if pad_len <= 0:
            return tensor
        pad = torch.zeros((pad_len, tensor.size(1)), dtype=tensor.dtype)
        return torch.cat([pad, tensor], dim=0)

    x_batch = torch.stack([left_pad_to_len(v, max_context_len) for v in x], dim=0)
    x_stamp_batch = torch.stack([left_pad_to_len(v, max_context_len) for v in x_stamp], dim=0)
    y_stamp_batch = torch.stack(y_stamp, dim=0)
    context_lens_tensor = torch.tensor(context_lens, dtype=torch.long)

    return x_batch, x_stamp_batch, y_stamp_batch, list(symbols), list(timestamps), context_lens_tensor


def generate_predictions(
    config: dict,
    test_data: dict,
    data_config: Config,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, pd.DataFrame] | None:
    """
    Runs inference on the test dataset to generate prediction signals.
    When used in a distributed setting, each rank processes a disjoint
    subset of the data and results are gathered on rank 0.

    Args:
        config (dict): A dictionary containing inference parameters.
        test_data (dict): The raw test data loaded from a pickle file.
        device (torch.device): Device assigned to the current process.
        rank (int): Global rank of the current process.
        world_size (int): Total number of processes involved.

    Returns:
        dict[str, pd.DataFrame] | None: A dictionary mapping signal names to DataFrames on
        rank 0; None on all other ranks.
    """

    tokenizer, model = load_models(config, device, rank)

    # Use the Dataset and DataLoader for efficient batching and processing
    dataset = QlibTestDataset(data=test_data, config=data_config)
    if world_size > 1:
        indices = list(range(rank, len(dataset), world_size))
        data_source = Subset(dataset, indices)
    else:
        data_source = dataset

    loader = DataLoader(
        data_source,
        batch_size=config['batch_size'] // config['sample_count'],
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn_for_inference,
        pin_memory=True,
    )

    results = defaultdict(list)
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps, context_lens in tqdm(loader, desc="Inference", disable=(rank != 0)):
            # 这里直接走“padding + mask”路径，不再按长度拆分子批次。
            preds = auto_regressive_inference(
                tokenizer,
                model,
                x.to(device),
                x_stamp.to(device),
                y_stamp.to(device),
                max_context=config['max_context'],
                pred_len=config['pred_len'],
                clip=config['clip'],
                T=config['T'],
                top_k=config['top_k'],
                top_p=config['top_p'],
                sample_count=config['sample_count'],
                context_lens=context_lens,
                use_kv_cache=config.get('use_kv_cache', True),
            )
            # You can try commenting on this line to keep the history data
            preds = preds[:, -config['pred_len']:, :]

            # The 'close' price is at index 3 in `feature_list`
            if config['backtest_pred'] == 'close':
                # 左侧 padding 后，最后一个时间步仍对应真实 anchor 时点。
                last_day_close = x[:, -1, 3].numpy()
                signals = {
                    'last': preds[:, -1, 3] - last_day_close,
                    'mean': np.mean(preds[:, :, 3], axis=1) - last_day_close,
                    'max': np.max(preds[:, :, 3], axis=1) - last_day_close,
                    'min': np.min(preds[:, :, 3], axis=1) - last_day_close,
                }
            elif config['backtest_pred'] == 'close_return':
                cum_close_return = preds[:, :, 9].cumsum(axis=1)
                signals = {
                    'last': cum_close_return[:, -1],
                    'mean': np.mean(cum_close_return, axis=1),
                    'max': np.max(cum_close_return, axis=1),
                    'min': np.min(cum_close_return, axis=1),
                }
            else:
                raise ValueError(f"Unsupported backtest_pred: {config['backtest_pred']}")

            for i in range(len(symbols)):
                for sig_type, sig_values in signals.items():
                    results[sig_type].append((timestamps[i], symbols[i], sig_values[i]))

    if world_size > 1 and dist.is_initialized():
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, results)
    else:
        gathered_results = [results]

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if rank != 0:
        return None

    merged = defaultdict(list)
    for partial in gathered_results:
        if not partial:
            continue
        for sig_type, records in partial.items():
            merged[sig_type].extend(records)

    print("Post-processing predictions into DataFrames...")
    prediction_dfs = {}
    for sig_type, records in merged.items():
        if not records:
            continue
        df = pd.DataFrame(records, columns=['datetime', 'instrument', 'score'])
        pivot_df = df.pivot_table(index='datetime', columns='instrument', values='score')
        # 现在的 pivot_df 的格式应该是，左边的 index 是一栏不同的 datetime，所有值的标签是 score，每种值（这里只有 score）下面都按照不同的 instrument 分列
        prediction_dfs[sig_type] = pivot_df.sort_index()

    return prediction_dfs


# =================================================================================
# 4. Main Execution
# =================================================================================

def main():
    """Main function to set up config, run inference, and execute backtesting."""
    parser = argparse.ArgumentParser(description="Run Kronos Inference and Backtesting")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for inference (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()

    # --- 1. Configuration Setup ---
    base_config = Config()

    ddp_enabled = "WORLD_SIZE" in os.environ
    if ddp_enabled:
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device(args.device)
        print(f"Running in single-process mode on device: {device}")

    set_seed(base_config.seed, rank)

    # Create a dedicated dictionary for this run's configuration
    run_config = {
        'data_path': base_config.dataset_path,
        'result_save_path': base_config.backtest_result_path,
        'result_name': base_config.backtest_save_folder_name,
        'tokenizer_path': base_config.finetuned_tokenizer_path,
        'model_path': base_config.finetuned_predictor_path,
        'max_context': base_config.max_context,
        'pred_len': base_config.predict_window,
        'clip': base_config.clip,
        'T': base_config.inference_T,
        'top_k': base_config.inference_top_k,
        'top_p': base_config.inference_top_p,
        'sample_count': base_config.inference_sample_count,
        'use_kv_cache': getattr(base_config, 'inference_use_kv_cache', True),
        'batch_size': base_config.backtest_batch_size,
        # =================================================================
        # New added by ZMJ
        # =================================================================
        'backtest_pred': base_config.backtest_pred,
        'allow_partial_context': getattr(base_config, 'backtest_allow_partial_context', False),
    }

    if rank == 0:
        print("--- Running with Configuration ---")
        for key, val in run_config.items():
            print(f"{key:>20}: {val}")
        print(f"{'device':>20}: {device}")
        print("-" * 35)

    # --- 2. Load Data ---
    split_paths = [
        # ("val", os.path.join(run_config['data_path'], "val_data.pkl")),
        # ("test", os.path.join(run_config['data_path'], "test_data.pkl")),
        # ("test", "/home/fanjiahao/workspace/kronos/20260409/kronos_12d_runtime_backtest_real_20240701_20251107.pkl"),
        ("test", "/home/fanjiahao/quant-resource/20260414/kronos_12d_runtime_backtest_real_20240102_20260414.pkl"),
    ]
    split_data = {}
    for split_name, split_path in split_paths:
        if rank == 0:
            print(f"Loading {split_name} data from {split_path}...")
        with open(split_path, 'rb') as f:
            split_data[split_name] = pickle.load(f)

    combined_data = {}
    symbols = set()
    for data_dict in split_data.values():
        symbols.update(data_dict.keys())
    symbols = sorted(symbols)

    for symbol in symbols:
        frames = []
        for split_name, _ in split_paths:
            df = split_data.get(split_name, {}).get(symbol)
            if df is not None and not df.empty:
                frames.append(df)
        if frames:
            # Keep validation rows ahead of test rows so the series stays continuous.
            combined_data[symbol] = pd.concat(frames)
    if rank == 0 and len(split_paths) > 1:
        print("Data merged")

    test_data = combined_data
    # if rank == 0:
    #     print(test_data)

    # --- 3. Generate Predictions ---
    model_preds = generate_predictions(run_config, test_data, base_config, device, rank, world_size)

    if ddp_enabled and dist.is_initialized():
        dist.barrier()
        
    if rank != 0:
        return

    # --- 4. Save Predictions ---
    if rank == 0 and model_preds:
        save_dir = os.path.join(run_config['result_save_path'], run_config['result_name'])
        os.makedirs(save_dir, exist_ok=True)
        predictions_file = os.path.join(save_dir, "predictions.pkl")
        print(f"Saving prediction signals to {predictions_file}...")
        with open(predictions_file, 'wb') as f:
            pickle.dump(model_preds, f)

    # --- 5. Run Backtesting ---
    save_dir = os.path.join(run_config['result_save_path'], run_config['result_name'])
    predictions_file = os.path.join(save_dir, "predictions.pkl")
    with open(predictions_file, 'rb') as f:
        model_preds = pickle.load(f)
        
    backtester = QlibBacktest(base_config)
    
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", f"backtest_result_{base_config.backtest_save_folder_name}.png"))
    backtester.run_and_plot_results(model_preds, save_path)


if __name__ == '__main__':
    main()


