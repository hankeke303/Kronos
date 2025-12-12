# -*- coding: utf-8 -*-

"""
本脚本使用 Qlib 加载数据，并严格按照 Kronos 论文 (arXiv:2508.02739) 
附录 B 中详述的数据预处理和清洗流程来处理数据。

流程总结:
1.  按股票代码分组。
2.  阶段一：缺失值处理
    - 价格 (OHLC) 字段的 NaN 会导致序列被分割。
    - 成交量/额 (VA) 字段的 NaN 稍后在阶段二中填充为 0。
3.  阶段二：低质量数据段过滤
    - 按“结构性断点”（开盘价/昨收价跳跃）分割序列。
    - 识别并移除（通过分割）连续的“非流动”（成交量=0）和“价格停滞”（收盘价不变）时段。
    - 过滤掉所有长度小于频率特定“min_len”的片段。

输出:
    由于清洗过程会产生大量不连续的数据片段，脚本会将每个干净的片段
    保存为一个单独的 .feather 文件到指定的输出目录。

使用示例:
    python kronos_qlib_cleaner.py \
        --qlib_path "~/.qlib/qlib_data/cn_data" \
        --output_dir "./cleaned_data_1min" \
        --frequency "1min"

注意: 
    - 您需要安装 qlib, pandas, numpy, pyarrow。
    - `pip install qlib pandas numpy pyarrow`
"""

import os
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
import qlib
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.storage.file_storage import FileFeatureStorage
from tqdm import tqdm

# --- 日志配置 ---
logging.basicConfig(
    # filename="/shd/hkk/Kronos/temp/kronos_qlib_cleaner.log",
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Qlib 字段和价格列 ---
FIELDS_OHLCVA = [
    "open", "high", "low", "close",
    "high_limit", "low_limit",
    "volume_post", "amount", "volume_no",
    "ma_tt_5", "ma_tt_10", "ma_tt_20", "ma_tt_60", "ma_tt_120",
    "rsi_tt_3", "rsi_tt_6", "rsi_tt_12", "rsi_tt_14",
    "macd_tt_dif", "macd_tt_dea", "macd_tt_macd",
    "flag", "is_st"
]
FIELDS_OHLCVA = ['$' + f for f in FIELDS_OHLCVA]
PRICE_COLS = ["$open", "$high", "$low", "$close"]

def normalize_frequency_label(frequency: str) -> str:
    """将用户输入的频率转换为 Qlib 目录所需的 freq 名称。"""
    freq = frequency.strip().lower()
    mapping = {
        "1d": "day",
        "1day": "day",
        "day": "day",
        "daily": "day",
        "1min": "1min",
        "1m": "1min",
        "5m": "5min",
        "5min": "5min",
        "15m": "15min",
        "15min": "15min",
        "30m": "30min",
        "30min": "30min",
        "60m": "60min",
        "1h": "60min",
        "60min": "60min",
    }
    if freq in mapping:
        return mapping[freq]
    if freq.endswith("min") and freq[:-3].isdigit():
        return freq
    raise ValueError(f"暂不支持频率 '{frequency}' 的 Qlib 导出，请补充映射。")


def calendar_time_format(freq_label: str) -> str:
    """根据频率决定日历写入格式。"""
    return "%Y-%m-%d" if freq_label == "day" else "%Y-%m-%d %H:%M:%S"

def get_cleaning_thresholds(frequency: str) -> dict:
    """
    根据论文表4，返回指定频率的清洗阈值。
    """
    thresholds = {
        # 频率: (min_len, price_jump, illiquid, stagnant)
        "1min": (2048, 0.10, 15, 45),
        "5min": (1024, 0.15, 3, 10),
        "10min": (512, 0.15, 3, 6),
        "15min": (512, 0.15, 2, 5),
        "30min": (512, 0.20, 2, 3),
        "1h": (256, 0.20, 1, 3), # 60min in paper
        "1d": (128, 0.30, 1, 3), # Daily in paper
    }
    
    if frequency not in thresholds:
        logger.warning(f"频率 '{frequency}' 未在论文表4中明确定义。将使用 '1d' (日线) 的阈值。")
        freq_key = "1d"
    else:
        freq_key = frequency
        
    keys = ["min_len", "price_jump", "illiquid", "stagnant"]
    return dict(zip(keys, thresholds[freq_key]))

def split_on_price_nan(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    阶段一：在价格字段 (OHLC) 出现 NaN 的地方切分数据。
    """
    if df.empty:
        return []

    # 1. 找出任何价格字段为 NaN 的行
    nan_mask = df[PRICE_COLS].isna().any(axis=1)
    
    if not nan_mask.any():
        # 如果没有 NaN，原样返回（在列表中）
        return [df]

    # 2. 创建“状态块”ID：mask 变化时递增，确保相邻的有效/无效区间被区分
    state_ids = nan_mask.ne(nan_mask.shift(fill_value=False)).cumsum()

    # 3. 仅保留标记为有效 (nan_mask=False) 的分段
    segments = [
        block_df
        for group_id, block_df in df.groupby(state_ids)
        if not nan_mask.loc[block_df.index[0]]
    ]

    return segments

def split_on_structural_breaks(df: pd.DataFrame, jump_threshold: float) -> list[pd.DataFrame]:
    """
    阶段二 (a)：在“结构性断点”（价格跳跃）处切分数据。
    """
    if df.empty:
        return []
        
    # 1. 计算昨收价
    # .droplevel('instrument') 是为了 shift() 能正确跨日期
    prev_close = df["$close"].droplevel(level='instrument').shift(1)
    # 重新对齐索引
    prev_close.index = df.index
    
    # 2. 计算价格跳跃
    jump = (df["$open"] / prev_close - 1).abs()
    
    # 3. 找出断点 (跳跃 > 阈值)。iloc[0] 总是 NaN，填充为 False。
    break_mask = (jump > jump_threshold).fillna(False)
    
    if not break_mask.any():
        # 没有断点
        return [df]

    # 4. 获取断点行的索引位置
    break_indices = np.where(break_mask.values)[0]

    # 5. 手动在断点处切分 DataFrame（避免 numpy 调用 DataFrame.swapaxes 的弃用路径）
    segments = []
    start = 0
    for idx in break_indices:
        if idx > start:
            segments.append(df.iloc[start:idx])
        start = idx
    if start < len(df):
        segments.append(df.iloc[start:])

    return [s for s in segments if not s.empty]

def filter_invalid_periods(df: pd.DataFrame, thresholds: dict) -> list[pd.DataFrame]:
    """
    阶段二 (b) & (c)：填充V/A的NaN，移除“非流动”和“价格停滞”时段，并按min_len过滤。
    """
    if df.empty:
        return []

    # --- 阶段二 (b) 开始 ---
    # 1. 填充 Volume/Amount 的 NaN 为 0 (根据附录 B)
    df = df.copy()
    df["$volume_post"] = df["$volume_post"].fillna(0)
    df["$amount"] = df["$amount"].fillna(0)

    # 2. 识别“非流动” K线
    is_illiquid = (df["$volume_post"] == 0)
    
    # 3. 识别“价格停滞” K线
    # .droplevel() 是为了 shift() 能正确跨日期
    prev_close = df["$close"].droplevel(level='instrument').shift(1)
    prev_close.index = df.index
    is_stagnant = (df["$close"] == prev_close)

    # 4. 识别 *超过阈值* 的连续无效时段
    def get_invalid_mask(series: pd.Series, threshold: int) -> pd.Series:
        if threshold <= 0:
            return pd.Series(False, index=series.index, dtype=bool)
        
        # 创建连续组的ID
        groups = (series != series.shift()).cumsum()
        # 计算每个组的大小
        group_sizes = series.groupby(groups).transform('size')
        # 仅当该组为 True (即无效) 且其大小超过阈值时，才标记为无效
        return (group_sizes > threshold) & series

    invalid_illiquid = get_invalid_mask(is_illiquid, thresholds["illiquid"])
    invalid_stagnant = get_invalid_mask(is_stagnant, thresholds["stagnant"])
    
    # 算法 1, line 7: 合并所有无效掩码
    overall_invalid_mask = invalid_illiquid | invalid_stagnant

    # 5. 算法 1, line 8: 在无效时段的边界处再次切分
    if not overall_invalid_mask.any():
        # 如果没有无效时段
        segments = [df]
    else:
        # 使用与 split_on_price_nan 相同的逻辑
        valid_blocks = (~overall_invalid_mask).astype(int).groupby(overall_invalid_mask.cumsum()).cumsum()
        segments = [
            block_df
            for group_id, block_df in df[valid_blocks > 0].groupby(valid_blocks)
        ]

    # --- 阶段二 (c) 开始 ---
    # 6. 算法 1, line 10: 最终分段验证（按最小长度）
    min_len = thresholds["min_len"]
    final_segments = [s for s in segments if len(s) >= min_len]
    
    return final_segments

def _clean_single_instrument(payload: tuple[str, pd.DataFrame, dict]) -> list[pd.DataFrame]:
    """对单只股票执行完整清洗流程。"""
    instrument, instrument_df, thresholds = payload

    # 初始时，每只股票是一个分段
    current_segments = [instrument_df]

    # 1. 阶段一：按价格NaN切分
    segments_step1 = []
    for segment in current_segments:
        segments_step1.extend(split_on_price_nan(segment))

    # 2. 阶段二 (a)：按结构性断点切分
    segments_step2 = []
    for segment in segments_step1:
        segments_step2.extend(split_on_structural_breaks(segment, thresholds["price_jump"]))

    # 3. 阶段二 (b) & (c)：过滤无效时段并按长度筛选
    segments_step3 = []
    for segment in segments_step2:
        segments_step3.extend(filter_invalid_periods(segment, thresholds))

    return segments_step3


def run_cleaning_pipeline(
    raw_df: pd.DataFrame,
    thresholds: dict,
    num_workers: int = 1,
    show_progress: bool = True,
) -> list[pd.DataFrame]:
    """按附录 B 执行数据清洗，可选并行并展示进度条。"""

    logger.info("开始按 'instrument' 分组...")
    grouped = list(raw_df.groupby(level="instrument"))
    total_instruments = len(grouped)
    logger.info(f"共 {total_instruments} 只股票待清洗。")

    if total_instruments == 0:
        return []

    all_clean_segments: list[pd.DataFrame] = []
    progress_desc = "清洗进度"

    if num_workers is None or num_workers == 0:
        worker_count = os.cpu_count() or 1
    else:
        worker_count = max(1, num_workers)

    if worker_count == 1:
        iterator = grouped
        if show_progress:
            iterator = tqdm(iterator, total=total_instruments, desc=progress_desc, unit="stock")

        for instrument, instrument_df in iterator:
            all_clean_segments.extend(
                _clean_single_instrument((instrument, instrument_df, thresholds))
            )
    else:
        logger.info(f"使用 {worker_count} 个进程并行清洗。")
        tasks = [
            (instrument, instrument_df, thresholds)
            for instrument, instrument_df in grouped
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_clean_single_instrument, task) for task in tasks]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=progress_desc, unit="stock")

            for future in iterator:
                all_clean_segments.extend(future.result())

    return all_clean_segments

def load_qlib_data(qlib_provider_uri: str) -> pd.DataFrame:
    """
    初始化 Qlib 并加载所有 OHLCVA 数据。
    """
    try:
        qlib.init(provider_uri=qlib_provider_uri, expression_cache=None)
        logger.info(f"Qlib 初始化成功，URI: {qlib_provider_uri}")
    except Exception as e:
        logger.error(f"Qlib 初始化失败: {e}")
        logger.error("请确保 qlib_path 正确，并且 Qlib 已正确安装。")
        raise
        
    logger.info(f"正在加载原始 Qlib 数据... 字段: {FIELDS_OHLCVA}")
    # 与 qlib_data_preprocess.py 中一致，通过 QlibDataLoader 加载数据
    loader = QlibDataLoader(config=FIELDS_OHLCVA)
    raw_df = loader.load("all", None, None)

    # loader.load 通常返回 (datetime, instrument) 的 MultiIndex，这里统一为 (instrument, datetime)
    if not isinstance(raw_df.index, pd.MultiIndex) or raw_df.index.nlevels != 2:
        logger.error("Qlib 数据未返回包含 instrument/datetime 的 MultiIndex。")
        raise ValueError("数据索引格式不正确")

    desired_order = ["instrument", "datetime"]
    current_names = list(raw_df.index.names)
    if current_names != desired_order:
        try:
            raw_df = raw_df.reorder_levels(desired_order)
        except (KeyError, ValueError):
            try:
                raw_df = raw_df.reorder_levels([1, 0])
            except Exception as exc:
                logger.error("无法将数据索引转换为 (instrument, datetime)", exc_info=True)
                raise ValueError("数据索引格式不正确") from exc

    raw_df = raw_df.sort_index()
    raw_df.index = raw_df.index.set_names(desired_order)
    logger.info(f"原始数据加载完成。总行数: {len(raw_df)}")
    return raw_df

def save_cleaned_segments(segments: list[pd.DataFrame], output_dir: str, frequency: str):
    """将清洗后的数据导出为 Qlib 可直接加载的数据目录结构。"""
    if not segments:
        logger.warning("没有找到符合所有清洗条件的干净数据片段。")
        return

    clean_df = pd.concat(segments).sort_index()
    clean_df = clean_df[~clean_df.index.duplicated(keep="first")]

    freq_label = normalize_frequency_label(frequency)
    calendar_fmt = calendar_time_format(freq_label)
    provider_root = Path(output_dir).expanduser().resolve()
    calendars_dir = provider_root / "calendars"
    instruments_dir = provider_root / "instruments"
    features_dir = provider_root / "features"
    for folder in (calendars_dir, instruments_dir, features_dir):
        folder.mkdir(parents=True, exist_ok=True)

    calendar_values = [pd.Timestamp(ts) for ts in sorted(clean_df.index.get_level_values("datetime").unique())]
    calendar_strings = [ts.strftime(calendar_fmt) for ts in calendar_values]
    calendar_file = calendars_dir / f"{freq_label}.txt"
    future_calendar_file = calendars_dir / f"{freq_label}_future.txt"
    for target in (calendar_file, future_calendar_file):
        with target.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(calendar_strings))

    instrument_periods = []
    for instrument, inst_df in clean_df.groupby(level="instrument"):
        inst_times = inst_df.index.get_level_values("datetime")
        instrument_periods.append(
            (
                instrument,
                pd.Timestamp(inst_times.min()),
                pd.Timestamp(inst_times.max()),
            )
        )

    instrument_path = instruments_dir / "all.txt"
    with instrument_path.open("w", encoding="utf-8") as fp:
        for instrument, start, end in sorted(instrument_periods, key=lambda x: x[0]):
            fp.write(
                f"{instrument}\t{start.strftime('%Y-%m-%d %H:%M:%S')}\t{end.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

    calendar_index = {ts: idx for idx, ts in enumerate(calendar_values)}
    provider_uri = {freq_label: str(provider_root)}
    total_tasks = len(instrument_periods) * len(clean_df.columns)
    logger.info(
        "正在写入 Qlib 特征文件: %s, instruments=%d, fields=%d",
        provider_root,
        len(instrument_periods),
        len(clean_df.columns),
    )

    progress = tqdm(total=total_tasks, desc="写入特征", unit="file")
    for instrument, inst_df in clean_df.groupby(level="instrument"):
        inst_times = inst_df.index.get_level_values("datetime")
        positions = np.array([calendar_index[ts] for ts in inst_times], dtype=int)
        inst_df = inst_df.droplevel("instrument")
        for column in inst_df.columns:
            field_name = column.lstrip("$")
            storage = FileFeatureStorage(
                instrument=instrument,
                field=field_name,
                freq=freq_label,
                provider_uri=provider_uri,
            )
            storage.uri.parent.mkdir(parents=True, exist_ok=True)
            if storage.uri.exists():
                storage.uri.unlink()
            values = np.full(len(calendar_values), np.nan, dtype=np.float32)
            col_values = inst_df[column].astype(np.float32).to_numpy()
            values[positions] = col_values
            storage.write(values, index=0)
            progress.update(1)
    progress.close()

    logger.info("Qlib 数据导出完成。输出目录: %s", provider_root)

def main():
    parser = argparse.ArgumentParser(description="Kronos (Appendix B) 数据清洗脚本 for Qlib")
    parser.add_argument(
        "--qlib_path",
        type=str,
        required=True,
        help="Qlib 数据存储路径 (例如: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="清洗后数据片段的输出目录",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        required=True,
        help="数据频率 (例如: '1min', '1d', '5min')。必须与论文表4中的键匹配。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="并行清洗使用的进程数。设为0或负数表示自动使用全部 CPU。",
    )
    
    args = parser.parse_args()

    try:
        # 1. 获取阈值
        logger.info(f"数据频率 '{args.frequency}'。正在获取清洗阈值...")
        thresholds = get_cleaning_thresholds(args.frequency)
        logger.info(f"使用阈值: {thresholds}")

        # 2. 加载数据
        raw_df = load_qlib_data(args.qlib_path)

        # 3. 执行清洗
        logger.info("开始执行清洗流程...")
        clean_segments = run_cleaning_pipeline(
            raw_df,
            thresholds,
            num_workers=args.num_workers,
            show_progress=True,
        )
        logger.info(f"清洗完成。共生成 {len(clean_segments)} 个有效的干净数据片段。")

        # 4. 保存结果
        save_cleaned_segments(clean_segments, args.output_dir, args.frequency)

        logger.info("数据清洗流程全部完成。")

    except Exception as e:
        logger.error(f"处理过程中发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()