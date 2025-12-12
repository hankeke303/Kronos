import numpy as np
import pandas as pd
import pickle
import os


def generate_filtered_map_signals(
        pred_path='predictions.pkl',
        stock_pool_path='dtstockdict.npy',
        output_path='mapsignals_filtered.npy',
        signal_key='last',
        nan_threshold=0.5  # 阈值：超过 50% 为 NaN 则丢弃该日期
):
    print(f"1. Loading stock pool from {stock_pool_path}...")
    stock_pool = np.load(stock_pool_path, allow_pickle=True).item()

    print(f"2. Loading predictions from {pred_path}...")
    with open(pred_path, 'rb') as f:
        pred_data = pickle.load(f)

    if signal_key not in pred_data:
        raise ValueError(f"Key '{signal_key}' not found. Available: {pred_data.keys()}")

    df_pred = pred_data[signal_key]
    df_pred.index = pd.to_datetime(df_pred.index)

    print(f"3. Aligning and Filtering (Threshold: >{nan_threshold * 100:.0f}% NaNs)...")

    map_signals = {}
    dropped_dates_count = 0
    total_dates_processed = 0

    for date_key, stock_list in stock_pool.items():
        total_dates_processed += 1

        # --- A. 尝试获取当天预测数据 ---
        try:
            target_date = pd.Timestamp(date_key)
        except:
            print(f"   [Skip] Invalid date format: {date_key}")
            continue

        n_stocks = len(stock_list)

        # --- B. 数据对齐逻辑 ---
        if target_date in df_pred.index:
            daily_pred_series = df_pred.loc[target_date]
            # 强制对齐到当前股票池
            aligned_signals = daily_pred_series.reindex(stock_list).values.astype(np.float32)
        else:
            # 如果这天完全没预测，视为 100% NaN
            # 在新逻辑下，这种情况肯定会被过滤掉，所以直接标记为全 NaN
            aligned_signals = np.full(n_stocks, np.nan, dtype=np.float32)

        # --- C. 核心修改：NaN 比例检查 ---
        # 计算 NaN 的数量
        nan_count = np.isnan(aligned_signals).sum()
        nan_ratio = nan_count / n_stocks

        # 如果 NaN 比例超过阈值 (例如 0.5)，则跳过这一天，不存入字典
        if nan_ratio > nan_threshold:
            dropped_dates_count += 1
            # 可选：打印日志看看丢了哪些天
            # print(f"   Date {date_key} dropped: {nan_count}/{n_stocks} NaNs ({nan_ratio:.2%})")
            continue

            # --- D. 保存合格的数据 ---
        map_signals[date_key] = aligned_signals

    print("-" * 30)
    print(f"4. Saving result to {output_path}...")
    print(f"   Total dates in pool:   {total_dates_processed}")
    print(f"   Dates dropped (>50% NaN): {dropped_dates_count}")
    print(f"   Dates saved:           {len(map_signals)}")

    np.save(output_path, map_signals)
    print("Done.")


if __name__ == "__main__":
    generate_filtered_map_signals(
        pred_path='predictions-2.pkl',
        stock_pool_path='dtstockdict.npy',
        output_path='mapsignals.npy',  # 覆盖保存或者换个名字
        signal_key='last',
        nan_threshold=0.5  # 超过 50% nan 就丢弃
    )