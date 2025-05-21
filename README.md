# TALib 指标批量生成框架

本仓库基于 Binance 行情、TA-Lib 和 Python 多进程，实现大规模技术指标计算与参数网格搜索。所有行为由 `config.yaml` 驱动，无需改动代码。

---

## 1. 运行步骤

**环境要求：Python 3.11.11**

```bash
# 安装依赖（示例）
pip install -r requirements.txt  # ta-lib, pandas, numpy, aiohttp 等

# 执行
python main.py
```
程序会：
1. 从 `config.yaml` 读取配置
2. 通过 `BinanceManager` 拉取 K 线
3. 根据 generator 配置选择计算模式
4. 结果写入 `generator.result_path`（默认为 `./results.csv`）

---

## 2. config.yaml 结构

```yaml
logging:           # 日志级别 / 格式

default:           # 拉取行情的默认参数
  symbol: BTCUSDT
  interval: 1h
  start_time: ...
  end_time:   ...

generator:         # 生成器总控
  generator_mode: multiple   # single | multiple | all
  indicator: RSI            # single 时使用
  indicators: [MACD, EMA, ATR]  # multiple 时使用
  parameter_mode: multiple   # single | multiple
  result_path: ./results.csv
  verbose: true

  use_threading: false       # 预留字段（未启用）
  thread_count: 4

talib:              # 各指标配置（截断示例）
  EMA:
    default:       # 单参数模式下用
      timeperiod: 20
    params_range:  # 多参数模式网格
      timeperiod:
        start: 5
        end: 50
        step: 5
  MAMA:
    default: {fastlimit: 0.5, slowlimit: 0.05}
    params_range:
      fastlimit: {start: 0.1, end: 0.9, step: 0.1}
      slowlimit: {start: 0.01, end: 0.45, step: 0.05}
  ...
```

### 2.1 generator 段
| 字段 | 说明 |
|------|------|
| generator_mode | 指标选择方式：single / multiple / all |
| parameter_mode | 参数组合：single（默认 / 第一组）或 multiple（全部组合） |
| indicator | 当 generator_mode=single 时，指定唯一指标 |
| indicators | generator_mode=multiple 时的指标列表 |
| result_path | 输出 CSV 路径 |
| verbose | 是否打印进度与调试信息 |

### 2.2 talib 段
每个指标节点可包含：

| 字段 | 说明 |
|------|------|
| default | 单参数模式下使用的参数字典 |
| params | 显式列出固定参数列表 |
| params_range | 以 `start / end / step` 定义区间，框架自动生成笛卡尔积 |

> 如果既存在 `params` 又存在 `params_range` ，`params` 优先。

---

## 3. 运行模式举例

| generator_mode | parameter_mode | 效果 |
|----------------|----------------|-------|
| single | single | 只计算 `indicator` 指定指标的一组默认参数 |
| single | multiple | 计算 `indicator` 的所有参数组合 |
| multiple | single | 计算 `indicators` 列表中每个指标的默认参数 |
| multiple | multiple | **多进程** 计算 `indicators` 中每个指标的全部参数组合 |
| all | single | 计算 `talib` 中所有指标的默认参数 |
| all | multiple | **多进程** 计算所有指标 + 全参数组合 |

---

## 4. 多进程架构

```
┌──────────┐        ┌───────────┐
│  主进程  │─spawn─▶│ Worker N  │  (TA-Lib 计算)
│ (调度)  │        └───────────┘
│          │             ⋮
│          │        ┌───────────┐
│          │─spawn─▶│ Writer    │  (批量写 CSV)
└──────────┘        └───────────┘
```

• Worker 进程：按指标 & 参数组合计算结果，通过 `Queue` 发送 `