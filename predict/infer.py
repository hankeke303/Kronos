import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Please install it with: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kronos import (  # noqa: E402
    Kronos,
    KronosTokenizer,
    auto_regressive_inference,
    calc_time_stamps,
)

FEATURE_TEMPLATES = {
    "template_6": ["open", "high", "low", "close", "vol", "amt"],
    "template_12": [
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amt",
        "flag",
        "circulating_market_cap",
        "open_return",
        "close_return",
        "hy_open_return",
        "hy_close_return",
    ],
}

DEFAULT_CONFIG = {
    "model": {
        "tokenizer_path": "",
        "predictor_path": "",
    },
    "input": {
        "csv_path": "",
        "time_col": "timestamps",
        "timestamp_mode": "auto_extrapolate",
        "future_time_csv_path": "",
        "future_time_col": "",
        "feature_template": "template_12",
        "feature_cols": [],
        "lookback_window": 90,
        "sort_ascending": True,
        "nan_policy": "error",
        "fallback_freq": "1D",
        "compute_extra_features": False,
    },
    "inference": {
        "pred_len": 10,
        "max_context": 512,
        "clip": 5.0,
        "T": 0.6,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 5,
        "device": "auto",
        "verbose": False,
    },
    "output": {
        "save_dir": "./outputs/predictions",
        "prefix": "infer",
        "float_precision": 8,
    },
    "signal": {
        "enabled": False,
        "backtest_pred": "close_return",
        "close_col": "close",
        "close_return_col": "close_return",
        "save_series": True,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    return cfg


def resolve_feature_cols(input_cfg: Dict[str, Any], templates: Dict[str, List[str]]) -> List[str]:
    feature_cols = input_cfg.get("feature_cols") or []
    if feature_cols:
        return feature_cols
    template_name = input_cfg.get("feature_template", "template_12")
    if template_name not in templates:
        raise ValueError(
            f"Unknown feature_template: {template_name}. "
            f"Available: {list(templates.keys())}"
        )
    return templates[template_name]


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    return device


def infer_frequency_delta(timestamps: pd.Series, fallback_freq: str) -> pd.Timedelta:
    if len(timestamps) < 2:
        return pd.to_timedelta(fallback_freq)

    diffs = timestamps.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.to_timedelta(fallback_freq)

    median_ns = int(np.median(diffs.astype("timedelta64[ns]").astype(np.int64)))
    if median_ns <= 0:
        return pd.to_timedelta(fallback_freq)
    return pd.to_timedelta(median_ns, unit="ns")


def build_future_timestamps(
    timestamps: pd.Series,
    pred_len: int,
    fallback_freq: str,
) -> pd.DatetimeIndex:
    last_ts = timestamps.iloc[-1]
    delta = infer_frequency_delta(timestamps, fallback_freq)
    return pd.date_range(start=last_ts + delta, periods=pred_len, freq=delta)


def resolve_context_and_timestamps(cfg: Dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, str]:
    input_cfg = cfg["input"]
    infer_cfg = cfg["inference"]

    time_col = input_cfg["time_col"]
    lookback_window = int(input_cfg["lookback_window"])
    pred_len = int(infer_cfg["pred_len"])
    fallback_freq = str(input_cfg["fallback_freq"])
    timestamp_mode = str(input_cfg.get("timestamp_mode", "auto_extrapolate")).strip().lower()

    if timestamp_mode == "auto_extrapolate":
        context_df = df.tail(lookback_window).copy()
        x_timestamp = pd.to_datetime(context_df[time_col])
        y_timestamp = build_future_timestamps(
            timestamps=x_timestamp,
            pred_len=pred_len,
            fallback_freq=fallback_freq,
        )
        freq_delta = infer_frequency_delta(x_timestamp, fallback_freq)
        return context_df, x_timestamp, y_timestamp, str(freq_delta)

    if timestamp_mode != "future_window":
        raise ValueError("input.timestamp_mode must be one of: auto_extrapolate, future_window")

    future_time_csv_path = str(input_cfg.get("future_time_csv_path", "") or "").strip()
    future_time_col = str(input_cfg.get("future_time_col", "") or time_col)

    if future_time_csv_path:
        if not os.path.exists(future_time_csv_path):
            raise FileNotFoundError(f"Future time CSV not found: {future_time_csv_path}")
        future_df = pd.read_csv(future_time_csv_path)
        if future_time_col not in future_df.columns:
            raise ValueError(f"Future time column '{future_time_col}' not found in {future_time_csv_path}")
        future_df[future_time_col] = pd.to_datetime(future_df[future_time_col], errors="raise")
        future_df = future_df.sort_values(future_time_col, ascending=True).reset_index(drop=True)
        if len(future_df) < pred_len:
            raise ValueError(
                f"Future time CSV rows ({len(future_df)}) are less than pred_len ({pred_len})."
            )

        context_df = df.tail(lookback_window).copy()
        x_timestamp = pd.to_datetime(context_df[time_col])
        y_timestamp = pd.DatetimeIndex(future_df[future_time_col].iloc[:pred_len].values)
        freq_delta = infer_frequency_delta(pd.Series(y_timestamp), fallback_freq)
        return context_df, x_timestamp, y_timestamp, str(freq_delta)

    required = lookback_window + pred_len
    if len(df) < required:
        raise ValueError(
            f"timestamp_mode=future_window requires at least {required} rows in input CSV, got {len(df)}."
        )

    window_df = df.tail(required).copy()
    context_df = window_df.iloc[:lookback_window].copy()
    future_df = window_df.iloc[lookback_window:lookback_window + pred_len].copy()
    x_timestamp = pd.to_datetime(context_df[time_col])
    y_timestamp = pd.DatetimeIndex(pd.to_datetime(future_df[time_col]).values)
    freq_delta = infer_frequency_delta(pd.Series(y_timestamp), fallback_freq)
    return context_df, x_timestamp, y_timestamp, str(freq_delta)


def apply_nan_policy(df: pd.DataFrame, columns: List[str], policy: str) -> pd.DataFrame:
    if policy not in {"error", "drop", "ffill"}:
        raise ValueError("nan_policy must be one of: error, drop, ffill")

    out = df.copy()
    if policy == "drop":
        out = out.dropna(subset=columns)
    elif policy == "ffill":
        out[columns] = out[columns].ffill()
    else:
        nan_columns = [col for col in columns if out[col].isnull().any()]
        if nan_columns:
            raise ValueError(f"NaN found in required columns: {nan_columns}")
    return out


def maybe_calc_extra_features(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df

    required_cols = ["open", "close", "hy_open", "hy_close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "compute_extra_features=true but required columns are missing: "
            f"{missing}"
        )

    out = df.copy()
    out["open_return"] = out["open"].pct_change()
    out["close_return"] = out["close"].pct_change()
    out["hy_open_return"] = out["hy_open"].pct_change()
    out["hy_close_return"] = out["hy_close"].pct_change()
    out = out.iloc[1:].copy()
    return out


def load_and_prepare_csv(cfg: Dict[str, Any], feature_cols: List[str]) -> pd.DataFrame:
    input_cfg = cfg["input"]
    csv_path = input_cfg["csv_path"]
    time_col = input_cfg["time_col"]

    if not csv_path:
        raise ValueError("input.csv_path is required")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in CSV")

    df[time_col] = pd.to_datetime(df[time_col], errors="raise")
    df = df.sort_values(time_col, ascending=input_cfg.get("sort_ascending", True)).reset_index(drop=True)

    df = maybe_calc_extra_features(df, input_cfg.get("compute_extra_features", False))
    if time_col not in df.columns:
        raise ValueError(
            "Time column disappeared after feature engineering. "
            "Please keep the original time column in CSV."
        )

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")

    required_cols = [time_col] + feature_cols
    df = apply_nan_policy(df, required_cols, input_cfg.get("nan_policy", "error"))

    lookback_window = int(input_cfg["lookback_window"])
    if len(df) < lookback_window:
        raise ValueError(
            f"CSV rows ({len(df)}) is less than lookback_window ({lookback_window})."
        )

    return df


def run_autoregressive_api(
    tokenizer: KronosTokenizer,
    predictor_model: Kronos,
    df_context: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.DatetimeIndex,
    feature_cols: List[str],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    infer_cfg = cfg["inference"]
    device = resolve_device(infer_cfg["device"])

    tokenizer = tokenizer.to(device)
    predictor_model = predictor_model.to(device)

    model_d_in = int(getattr(tokenizer, "d_in"))
    if model_d_in != len(feature_cols):
        raise ValueError(
            f"Tokenizer d_in ({model_d_in}) does not match feature count ({len(feature_cols)})."
        )

    x = df_context[feature_cols].values.astype(np.float32)
    x_stamp = calc_time_stamps(pd.to_datetime(x_timestamp)).values.astype(np.float32)
    y_stamp = calc_time_stamps(pd.to_datetime(pd.Series(y_timestamp))).values.astype(np.float32)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -float(infer_cfg["clip"]), float(infer_cfg["clip"]))

    x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :]).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :]).to(device)

    preds = auto_regressive_inference(
        tokenizer=tokenizer,
        model=predictor_model,
        x=x_tensor,
        x_stamp=x_stamp_tensor,
        y_stamp=y_stamp_tensor,
        max_context=int(infer_cfg["max_context"]),
        pred_len=int(infer_cfg["pred_len"]),
        clip=float(infer_cfg["clip"]),
        T=float(infer_cfg["T"]),
        top_k=int(infer_cfg["top_k"]),
        top_p=float(infer_cfg["top_p"]),
        sample_count=int(infer_cfg["sample_count"]),
        verbose=bool(infer_cfg["verbose"]),
    )

    preds = preds[:, -int(infer_cfg["pred_len"]):, :].squeeze(0)
    preds = preds * (x_std + 1e-5) + x_mean
    pred_df = pd.DataFrame(preds, columns=feature_cols, index=y_timestamp)
    return pred_df


def compute_signal_results(
    pred_df: pd.DataFrame,
    context_df: pd.DataFrame,
    cfg: Dict[str, Any],
    time_col: str,
) -> tuple[Dict[str, Any] | None, pd.DataFrame | None]:
    signal_cfg = cfg.get("signal", {})
    if not bool(signal_cfg.get("enabled", False)):
        return None, None

    backtest_pred = str(signal_cfg.get("backtest_pred", "close_return")).strip().lower()
    close_col = str(signal_cfg.get("close_col", "close"))
    close_return_col = str(signal_cfg.get("close_return_col", "close_return"))

    if backtest_pred == "close":
        if close_col not in context_df.columns:
            raise ValueError(f"signal.backtest_pred=close requires '{close_col}' in context data")
        if close_col not in pred_df.columns:
            raise ValueError(f"signal.backtest_pred=close requires '{close_col}' in prediction output")
        last_close = float(context_df[close_col].iloc[-1])
        signal_series = pred_df[close_col].astype(float).to_numpy() - last_close
        source_col = close_col
    elif backtest_pred == "close_return":
        if close_return_col not in pred_df.columns:
            raise ValueError(
                f"signal.backtest_pred=close_return requires '{close_return_col}' in prediction output"
            )
        signal_series = pred_df[close_return_col].astype(float).to_numpy().cumsum()
        source_col = close_return_col
    else:
        raise ValueError("signal.backtest_pred must be one of: close, close_return")

    signal_dict = {
        "mode": backtest_pred,
        "source_column": source_col,
        "last": float(signal_series[-1]),
        "mean": float(np.mean(signal_series)),
        "max": float(np.max(signal_series)),
        "min": float(np.min(signal_series)),
    }

    signal_series_df = pd.DataFrame(
        {
            time_col: pd.DatetimeIndex(pred_df.index),
            "step": np.arange(1, len(pred_df) + 1),
            "signal": signal_series,
        }
    )
    return signal_dict, signal_series_df


def save_outputs(
    pred_df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_cols: List[str],
    y_timestamp: pd.DatetimeIndex,
    resolved_mode: str,
    resolved_freq: str,
    signal_dict: Dict[str, Any] | None,
    signal_series_df: pd.DataFrame | None,
) -> Dict[str, str]:
    output_cfg = cfg["output"]
    input_cfg = cfg["input"]
    infer_cfg = cfg["inference"]
    model_cfg = cfg["model"]

    save_dir = Path(output_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_cfg.get("prefix", "infer")
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = save_dir / f"{prefix}_{now_str}_predictions.csv"
    json_path = save_dir / f"{prefix}_{now_str}_meta.json"

    result_df = pred_df.reset_index().rename(columns={"index": input_cfg["time_col"]})
    result_df.insert(1, "step", np.arange(1, len(result_df) + 1))
    result_df.to_csv(csv_path, index=False, float_format=f"%.{int(output_cfg['float_precision'])}f")

    meta = {
        "mode": resolved_mode,
        "time_col": input_cfg["time_col"],
        "feature_cols": feature_cols,
        "tokenizer_path": model_cfg["tokenizer_path"],
        "predictor_path": model_cfg["predictor_path"],
        "lookback_window": int(input_cfg["lookback_window"]),
        "pred_len": int(infer_cfg["pred_len"]),
        "freq": resolved_freq,
        "x_end_time": str(pd.to_datetime(y_timestamp[0]) - pd.to_timedelta(resolved_freq)),
        "y_start_time": str(y_timestamp[0]),
        "y_end_time": str(y_timestamp[-1]),
        "inference": {
            "max_context": int(infer_cfg["max_context"]),
            "clip": float(infer_cfg["clip"]),
            "T": float(infer_cfg["T"]),
            "top_k": int(infer_cfg["top_k"]),
            "top_p": float(infer_cfg["top_p"]),
            "sample_count": int(infer_cfg["sample_count"]),
            "device": str(resolve_device(infer_cfg["device"])),
        },
        "created_at": datetime.now().isoformat(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    saved_paths = {"csv": str(csv_path), "json": str(json_path)}

    if signal_dict is not None:
        signal_json_path = save_dir / f"{prefix}_{now_str}_signals.json"
        with open(signal_json_path, "w", encoding="utf-8") as f:
            json.dump(signal_dict, f, ensure_ascii=False, indent=2)
        saved_paths["signal_json"] = str(signal_json_path)

        signal_cfg = cfg.get("signal", {})
        if bool(signal_cfg.get("save_series", True)) and signal_series_df is not None:
            signal_csv_path = save_dir / f"{prefix}_{now_str}_signal_series.csv"
            signal_series_df.to_csv(signal_csv_path, index=False, float_format=f"%.{int(output_cfg['float_precision'])}f")
            saved_paths["signal_csv"] = str(signal_csv_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kronos inference from CSV with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    if not model_cfg.get("tokenizer_path") or not model_cfg.get("predictor_path"):
        raise ValueError("model.tokenizer_path and model.predictor_path are required")

    custom_templates = cfg.get("feature_templates") or {}
    templates = deep_update(FEATURE_TEMPLATES, custom_templates)
    feature_cols = resolve_feature_cols(cfg["input"], templates)

    df = load_and_prepare_csv(cfg, feature_cols)

    input_cfg = cfg["input"]
    infer_cfg = cfg["inference"]
    time_col = input_cfg["time_col"]

    context_df, x_timestamp, y_timestamp, resolved_freq = resolve_context_and_timestamps(cfg, df)

    tokenizer = KronosTokenizer.from_pretrained(model_cfg["tokenizer_path"])
    predictor_model = Kronos.from_pretrained(model_cfg["predictor_path"])
    tokenizer.eval()
    predictor_model.eval()

    pred_df = run_autoregressive_api(
        tokenizer=tokenizer,
        predictor_model=predictor_model,
        df_context=context_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        feature_cols=feature_cols,
        cfg=cfg,
    )

    signal_dict, signal_series_df = compute_signal_results(
        pred_df=pred_df,
        context_df=context_df,
        cfg=cfg,
        time_col=time_col,
    )

    paths = save_outputs(
        pred_df=pred_df,
        cfg=cfg,
        feature_cols=feature_cols,
        y_timestamp=y_timestamp,
        resolved_mode="unified",
        resolved_freq=resolved_freq,
        signal_dict=signal_dict,
        signal_series_df=signal_series_df,
    )

    print("Inference finished.")
    print(f"Prediction CSV saved to: {paths['csv']}")
    print(f"Metadata JSON saved to: {paths['json']}")
    if "signal_json" in paths:
        print(f"Signal JSON saved to: {paths['signal_json']}")
    if "signal_csv" in paths:
        print(f"Signal series CSV saved to: {paths['signal_csv']}")


if __name__ == "__main__":
    main()
