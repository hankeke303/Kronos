## 运行

```bash
python predict/infer.py --config predict/configs/infer.default.yaml
```

## 功能概览

- 支持 6 特征、12 特征或自定义特征（由 `feature_template` / `feature_cols` 控制）。
- 支持两种未来时间戳策略：自动外推、真实未来窗口。
- 可选生成与回测语义一致的 signal 统计（`last/mean/max/min`）。

## 程序执行流程（运行时会发生什么）

当你执行：

```bash
python predict/infer.py --config predict/configs/infer.default.yaml
```

程序会按以下顺序执行：

1. **读取并合并配置**
  - 先加载 `infer.default.yaml`。
  - 再和脚本内置默认配置合并（你没写的参数会自动补默认值）。

2. **校验模型路径**
  - 检查 `model.tokenizer_path`、`model.predictor_path` 是否已填写。
  - 如果为空或路径不存在，会直接报错并退出。

3. **确定特征列**
  - 优先使用 `input.feature_cols`；如果为空，则使用 `input.feature_template` 对应模板。
  - 这一步会决定模型输入维度，后续会和 tokenizer 的 `d_in` 做一致性检查。

4. **读取并预处理 CSV**
  - 解析时间列（`input.time_col`）为 datetime。
  - 按时间排序（`sort_ascending`）。
  - 按需计算衍生特征（`compute_extra_features`）。
  - 按 `nan_policy` 处理缺失值。
  - 检查是否有足够行数满足 `lookback_window`（或未来窗口模式下的 `lookback_window + pred_len`）。

5. **构造历史窗口与未来时间戳**
  - `auto_extrapolate`：从历史间隔推断频率，自动生成未来 `pred_len` 个时间戳。
  - `future_window`：使用你提供的未来时间 CSV，或从同一 CSV 尾部切分真实未来窗口。

6. **加载模型并执行推理**
  - 从 checkpoint 加载 `KronosTokenizer` 与 `Kronos`。
  - 将历史特征做“样本内标准化 + clip 截断”。
  - 调用自回归推理生成未来序列，再反归一化还原到原始量纲。

7. **可选计算 signal**
  - 当 `signal.enabled=true`：
    - `close` 模式：`pred_close - last_context_close`
    - `close_return` 模式：对预测 `close_return` 做累计和
  - 产出 `last/mean/max/min` 汇总，以及可选逐步 signal 序列。

8. **落盘并打印结果路径**
  - 保存预测 CSV、meta JSON。
  - 若启用 signal，再保存 `signals.json` 和可选 `signal_series.csv`。
  - 控制台会打印每个输出文件的完整路径。

### 失败时会怎样

- 任一步校验失败（列缺失、维度不匹配、路径错误、数据长度不足）会抛出明确错误并停止。
- 程序不会写入半成品模型文件；最多只会在输出目录留下已成功写出的结果文件。

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

