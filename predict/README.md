## 运行

```bash
python predict/infer.py --config predict/configs/infer.default.yaml
```

## 功能概览

- 支持 6 特征、12 特征或自定义特征（由 `feature_template` / `feature_cols` 控制）。
- 支持两种未来时间戳策略：自动外推、真实未来窗口。
- 可选生成与回测语义一致的 signal 统计（`last/mean/max/min`）。

## 配置说明

默认配置文件：`predict/configs/infer.default.yaml`

### 1) model

- `model.tokenizer_path`：Tokenizer checkpoint 路径（本地路径或 HuggingFace Hub ID）。
- `model.predictor_path`：Predictor checkpoint 路径（本地路径或 HuggingFace Hub ID）。

### 2) input

- `input.csv_path`：输入 CSV 路径。
- `input.time_col`：时间列名，默认 `timestamps`。
- `input.feature_template`：特征模板名（如 `template_6` / `template_12`）。
- `input.feature_cols`：显式特征列顺序；若非空，会覆盖 `feature_template`。
- `input.lookback_window`：历史上下文窗口长度。
- `input.nan_policy`：缺失值处理策略，支持 `error | drop | ffill`。
- `input.compute_extra_features`：是否自动计算 `open_return/close_return/hy_open_return/hy_close_return`。

### 3) inference

- `inference.pred_len`：未来预测步数。
- `inference.max_context`：模型最大上下文。
- `inference.clip`：归一化后截断阈值。
- `inference.T / top_k / top_p / sample_count`：采样参数。
- `inference.device`：`auto` / `cpu` / `cuda:0` 等。

### 4) output

- `output.save_dir`：输出目录。
- `output.prefix`：输出文件名前缀。
- `output.float_precision`：CSV 浮点保留位数。

### 5) signal（可选）

- `signal.enabled`：是否启用 signal 计算。
- `signal.backtest_pred`：`close` 或 `close_return`。
  - `close`：`pred_close - last_context_close`
  - `close_return`：对未来 `close_return` 做累计和后统计
- `signal.save_series`：是否保存逐步 signal 序列。

## 时间戳模式

### A. 自动外推（默认）

- `input.timestamp_mode: auto_extrapolate`
- 从历史时间间隔推断频率，自动生成未来 `pred_len` 个时间戳。
- 若无法可靠推断，使用 `input.fallback_freq`。

### B. 真实未来窗口

- `input.timestamp_mode: future_window`
- 两种方式：
  - 方式 1：提供 `input.future_time_csv_path`（读取前 `pred_len` 行时间列）。
  - 方式 2：不提供未来 CSV，直接从同一输入 CSV 末尾切分 `lookback_window + pred_len`。

## 输出文件

每次运行会在 `output.save_dir` 生成带时间戳的文件：

- `*_predictions.csv`：预测结果（时间列 + step + 特征列）。
- `*_meta.json`：本次运行元信息（模型路径、参数、频率、时间范围等）。
- `*_signals.json`：启用 `signal.enabled` 后生成，包含 `last/mean/max/min`。
- `*_signal_series.csv`：启用 `signal.enabled` 且 `signal.save_series=true` 后生成。

## 重要注意事项

- `feature_cols` 的数量和顺序必须与 tokenizer 的 `d_in` 及训练时一致。
- 若启用 `compute_extra_features=true`，输入 CSV 需包含 `open/close/hy_open/hy_close`。
- `signal.backtest_pred=close_return` 时，预测输出里必须存在 `close_return` 列。

