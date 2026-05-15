# Backtest View

一个用于浏览 Kronos 回测结果的网页服务。

- 自动扫描 `outputs/backtest_results/` 下所有子目录。
- 支持选择“回测目录 + 信号”查看可视化。
- 支持总体收益趋势图（来自 `return_curve_*.csv`）。
- 支持在收益趋势图点击日期后，查看当天或最近交易日的持仓/调仓明细。
- 支持“个股收益贡献排名”（全区间或自定义区间）。
- 内置股票代码到中文名缓存，并自动应用到图表和明细表。
- 优先展示你新增的 trace 文件：
  - `return_curve_*.csv`
  - `holdings_snapshot_*.csv`
  - `rebalance_actions_*.csv`
  - `rebalance_summary_*.csv`
  - `holding_periods_*.csv`
  - `indicator_raw_*.csv`
  - `trade_details_*.csv`
- 若 trace 文件缺失（旧回测目录），会自动回退到 `output.txt` 片段展示。

## 启动

在项目根目录执行：

```bash
cd backtest_view
pip install -r requirements.txt
python run.py
```

默认地址：

- `http://127.0.0.1:8787`

健康检查：

- `http://127.0.0.1:8787/health`

## 页面内容

- 持仓与换手诊断图
- 总体收益趋势图（可点击联动）
- 成交额(或成交量)与买卖动作图
- 持仓区间分布与 Top 标的图
- 持仓结构 Top20 图
- 点击日期联动的持仓/调仓明细表
- 日期联动明细包含持仓时长、本次持仓收益、调仓时间
- 个股收益贡献正负榜（可设置起止日期）
- 日志依据片段 (`output.txt`)

## 说明

- `trade_details_*.csv` 的 `trade_value` 缺失时，页面会用 `deal_amount` 的绝对值代替。
- 个股贡献额是近似估算值：`前一日持仓数量 × 当日价格变动`，用于横向排名与排查，不等同于精确成交归因。
- 股票中文名缓存文件在 `backtest_view/cache/stock_name_map.json`，会在服务启动时自动刷新并持久化。
- 对于历史结果目录，若无你新增的 CSV 文件，统计卡片与图表会显示为缺失提示，不会报错。
