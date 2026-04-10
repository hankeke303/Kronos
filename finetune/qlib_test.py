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
        self.initialize_qlib()

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib for backtesting...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def run_single_backtest(self, signal_series: pd.Series) -> pd.DataFrame:
        """
        Runs a single backtest for a given prediction signal.

        Args:
            signal_series (pd.Series): A pandas Series with a MultiIndex
                                       (instrument, datetime) and prediction scores.
        Returns:
            pd.DataFrame: A DataFrame containing the performance report.
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

        portfolio_metric_dict, _ = backtest(strategy=strategy, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse("day"))
        report, _ = portfolio_metric_dict.get(analysis_freq)

        # --- Analysis and Reporting ---
        analysis = {
            "excess_return_without_cost": risk_analysis(report["return"] - report["bench"], freq=analysis_freq),
            "excess_return_with_cost": risk_analysis(report["return"] - report["bench"] - report["cost"], freq=analysis_freq),
        }
        print("\n--- Backtest Analysis ---")
        print("Benchmark Return:", risk_analysis(report["bench"], freq=analysis_freq), sep='\n')
        print("\nExcess Return (w/o cost):", analysis["excess_return_without_cost"], sep='\n')
        print("\nExcess Return (w/ cost):", analysis["excess_return_with_cost"], sep='\n')

        report_df = pd.DataFrame({
            "cum_bench": report["bench"].cumsum(),
            "cum_return_w_cost": (report["return"] - report["cost"]).cumsum(),
            "cum_ex_return_w_cost": (report["return"] - report["bench"] - report["cost"]).cumsum(),
        })
        
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "backtest_results", self.config.backtest_save_folder_name, "output.txt")), 'a') as f:
            f.write("\n--- Backtest Analysis ---\n")
            f.write("Benchmark Return:\n")
            f.write(str(risk_analysis(report["bench"], freq=analysis_freq)))
            f.write("\n\nExcess Return (w/o cost):\n")
            f.write(str(analysis["excess_return_without_cost"]))
            f.write("\n\nExcess Return (w/ cost):\n")
            f.write(str(analysis["excess_return_with_cost"]))
            
            f.write(f"report_df: \n{report_df}\n")
        return report_df

    def run_and_plot_results(self, signals: dict[str, pd.DataFrame], save_path: str):
        """
        Runs backtests for multiple signals and plots the cumulative return curves.

        Args:
            signals (dict[str, pd.DataFrame]): A dictionary where keys are signal names
                                               and values are prediction DataFrames.
        """
        return_df, ex_return_df, bench_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for signal_name, pred_df in signals.items():
            print(f"\nBacktesting signal: {signal_name}...")
            pred_series = pred_df.stack()
            pred_series.index.names = ['datetime', 'instrument']
            pred_series = pred_series.swaplevel().sort_index()
            # 现在的 pred_series 的格式应该是左边有两列，第一列是不同的 instrument，每个 instrument 下面是不同的 datetime 列出，内容只有一栏 score
            
            with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "backtest_results", self.config.backtest_save_folder_name, "output.txt")), 'a') as f:
                f.write(f"\nBacktesting signal: {signal_name}...\n")
            
            report_df = self.run_single_backtest(pred_series)

            return_df[signal_name] = report_df['cum_return_w_cost']
            ex_return_df[signal_name] = report_df['cum_ex_return_w_cost']
            if 'return' not in bench_df:
                bench_df['return'] = report_df['cum_bench']

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
        ("test", "/home/fanjiahao/workspace/kronos/20260409/kronos_12d_runtime_backtest_real_20240701_20251107.pkl"),
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


