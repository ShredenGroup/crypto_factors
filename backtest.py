import backtrader as bt
import pandas as pd
import datetime
import os
import argparse
from config_manager import get_config_manager


class CryptoStrategy(bt.Strategy):
    """加密货币回测策略，使用生成的技术指标"""
    
    params = (
        ('ema_period', 20),           # 默认EMA周期
        ('rsi_period', 14),           # 默认RSI周期
        ('rsi_overbought', 70),       # RSI超买阈值
        ('rsi_oversold', 30),         # RSI超卖阈值
        ('macd_signal', 9),           # MACD信号线周期
        ('atr_period', 14),           # ATR周期
        ('hbfc_threshold', 1.0),      # 自定义HBFC阈值
        ('risk_percent', 2.0),        # 每笔交易风险比例
        ('trail_percent', 1.0),       # 追踪止损百分比
    )
    
    def __init__(self):
        """初始化策略，定义指标"""
        # 获取K线数据
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        
        # 定义订单相关变量
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.stop_loss = None
        
        # 加载数据源中的自定义指标
        self.hbfc_one = self.datas[0].hbfc_one
        self.taker_buy_ratio = self.datas[0].taker_buy_ratio
        
        # 加载数据源中的TA-Lib指标
        # EMA
        ema_name = f"EMA_timeperiod_{self.params.ema_period}_real"
        if hasattr(self.datas[0], ema_name):
            self.ema = getattr(self.datas[0], ema_name)
        else:
            self.ema = bt.indicators.EMA(self.dataclose, period=self.params.ema_period)
        
        # RSI
        rsi_name = f"RSI_timeperiod_{self.params.rsi_period}_real"
        if hasattr(self.datas[0], rsi_name):
            self.rsi = getattr(self.datas[0], rsi_name)
        else:
            self.rsi = bt.indicators.RSI(self.dataclose, period=self.params.rsi_period)
        
        # MACD - 尝试查找预计算的MACD
        macd_name_histogram = "MACD_fastperiod_12_slowperiod_26_signalperiod_9_macdhist"
        macd_name_signal = "MACD_fastperiod_12_slowperiod_26_signalperiod_9_macdsignal"
        macd_name_line = "MACD_fastperiod_12_slowperiod_26_signalperiod_9_macd"
        
        if all(hasattr(self.datas[0], n) for n in [macd_name_line, macd_name_signal, macd_name_histogram]):
            self.macd_line = getattr(self.datas[0], macd_name_line)
            self.macd_signal = getattr(self.datas[0], macd_name_signal)
            self.macd_histogram = getattr(self.datas[0], macd_name_histogram)
        else:
            self.macd = bt.indicators.MACD(
                self.dataclose,
                period_me1=12,
                period_me2=26,
                period_signal=self.params.macd_signal
            )
            self.macd_line = self.macd.macd
            self.macd_signal = self.macd.signal
            self.macd_histogram = self.macd.histogram
        
        # ATR - 尝试查找预计算的ATR
        atr_name = f"ATR_timeperiod_{self.params.atr_period}_real"
        if hasattr(self.datas[0], atr_name):
            self.atr = getattr(self.datas[0], atr_name)
        else:
            self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # 交叉指标
        self.macd_cross = bt.indicators.CrossOver(self.macd_line, self.macd_signal)
        
        # 打印策略启动信息
        print('策略启动完成，数据周期:', self.datas[0]._name)
    
    def log(self, txt, dt=None):
        """输出日志"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """订单状态更新通知"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交或接受，不做特殊处理
            return
        
        # 检查订单是否完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, '
                         f'成本={order.executed.value:.2f}, '
                         f'手续费={order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                
                # 设置止损
                stop_price = self.buyprice * (1.0 - self.params.trail_percent / 100.0)
                self.stop_loss = self.sell(exectype=bt.Order.StopTrail,
                                         trailpercent=self.params.trail_percent,
                                         price=stop_price)
                
            else:  # 卖出
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, '
                         f'成本={order.executed.value:.2f}, '
                         f'手续费={order.executed.comm:.2f}')
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易结果通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易利润: 总额={trade.pnl:.2f}, 净额={trade.pnlcomm:.2f}')
    
    def next(self):
        """策略核心逻辑 - 每个bar调用一次"""
        # 如果有未完成订单，不开新仓位
        if self.order:
            return
        
        # 如果已持仓，检查止损线
        if self.position:
            return
        
        # 生成买入信号 - 这里结合多个指标，可以自行调整
        buy_signal = False
        
        # 规则1: RSI超卖区域
        rsi_oversold = self.rsi < self.params.rsi_oversold
        
        # 规则2: MACD金叉
        macd_golden_cross = self.macd_cross > 0
        
        # 规则3: 市场在EMA之上（上升趋势）
        price_above_ema = self.dataclose > self.ema
        
        # 规则4: 自定义HBFC指标超过阈值（大量买盘与价格变动比值异常）
        hbfc_trigger = self.hbfc_one > self.params.hbfc_threshold
        
        # 综合判断买入信号
        # 当RSI超卖 且 (MACD金叉 或 HBFC触发) 且 价格在EMA之上
        if rsi_oversold and (macd_golden_cross or hbfc_trigger) and price_above_ema:
            buy_signal = True
            
        # 买入逻辑
        if buy_signal:
            # 计算仓位大小（基于风险）
            risk_amount = self.broker.getvalue() * self.params.risk_percent / 100
            stop_loss_price = self.dataclose * (1.0 - self.params.trail_percent / 100.0)
            risk_per_share = self.dataclose - stop_loss_price
            
            if risk_per_share > 0:  # 避免除零错误
                size = risk_amount / risk_per_share
                size = int(size)  # 整数股
                
                if size > 0:
                    self.log(f'买入信号! RSI={self.rsi[0]:.2f}, HBFC={self.hbfc_one[0]:.2f}')
                    self.order = self.buy(size=size)


def load_results_csv(filepath):
    """加载结果CSV文件，处理指标列名"""
    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到结果文件: {filepath}")
    
    # 读取CSV文件
    df = pd.read_csv(filepath)
    
    # 移除不必要的中间列，避免数据过多
    columns_to_drop = [col for col in df.columns if any(x in col for x in ['safe_returns', 'price_returns'])]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # 确保有时间列
    if 'timestamp' not in df.columns and 'datetime' not in df.columns:
        # 如果没有时间列，创建一个假的时间列用于回测
        df['datetime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1H')
    
    # 确保列名符合backtrader要求（小写，无特殊字符）
    rename_dict = {}
    for col in df.columns:
        new_name = col.lower().replace('_', '')
        # 避免重命名已有列
        if new_name != col and new_name not in rename_dict.values():
            rename_dict[col] = new_name
    
    # 重命名自定义指标，方便在策略中访问
    special_cols = {
        'HBFC_ONE': 'hbfc_one',
        'taker_buy_ratio': 'taker_buy_ratio'
    }
    
    for old, new in special_cols.items():
        if old in df.columns:
            rename_dict[old] = new
    
    df = df.rename(columns=rename_dict)
    
    # 确保有必要的OHLCV列
    essential_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in essential_cols:
        if col not in df.columns:
            raise ValueError(f"结果文件缺少必要列: {col}")
    
    return df


def run_backtest(csv_file=None, plot=True, cash=100000.0):
    """运行回测"""
    # 初始化backtrader环境
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(CryptoStrategy)
    
    # 如果未指定CSV文件，使用配置中的默认路径
    if csv_file is None:
        cfg = get_config_manager()
        csv_file = cfg.get_generator_config().get('result_path', './results.csv')
    
    # 加载结果数据
    df = load_results_csv(csv_file)
    
    # 设置日期为索引
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.set_index('datetime', inplace=True)
    
    # 打印数据集基本信息
    print(f"加载数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"数据周期: {df.index[1] - df.index[0]}")
    print(f"数据范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 创建Data Feed
    data = bt.feeds.PandasData(
        dataname=df,
        # 传递行情数据列
        open='open',
        high='high',
        low='low', 
        close='close',
        volume='volume',
        # 传递自定义指标列
        hbfc_one='hbfc_one',
        taker_buy_ratio='taker_buy_ratio',
        # 数据已按时间排序
        openinterest=None,
        plot=True
    )
    
    # 添加数据
    cerebro.adddata(data)
    
    # 设置初始资金
    cerebro.broker.setcash(cash)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=0.001)  # 0.1% 交易手续费
    
    # 设置滑点
    cerebro.broker.set_slippage_perc(0.001)  # 0.1% 滑点
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    print('初始资金: %.2f' % cerebro.broker.getvalue())
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]
    
    # 打印策略评估结果
    print('最终资金: %.2f' % cerebro.broker.getvalue())
    print('回报率: %.2f%%' % (strat.analyzers.returns.get_analysis()['rtot'] * 100.0))
    print('夏普率: %.2f' % strat.analyzers.sharpe.get_analysis()['sharperatio'])
    print('最大回撤: %.2f%%' % (strat.analyzers.drawdown.get_analysis()['max']['drawdown'] * 100.0))
    print('SQN: %.2f' % strat.analyzers.sqn.get_analysis()['sqn'])
    
    # 交易统计
    trade_analysis = strat.analyzers.trades.get_analysis()
    print('\n==== 交易统计 ====')
    print('总交易次数: %d' % trade_analysis['total']['total'])
    if trade_analysis['total']['total'] > 0:
        print('盈利交易: %d' % trade_analysis['won']['total'])
        print('亏损交易: %d' % trade_analysis['lost']['total'])
        if trade_analysis['won']['total'] > 0:
            print('平均盈利: %.2f' % trade_analysis['won']['pnl']['average'])
        if trade_analysis['lost']['total'] > 0:
            print('平均亏损: %.2f' % trade_analysis['lost']['pnl']['average'])
    
    # 绘制图表
    if plot:
        cerebro.plot(style='candle', barup='green', bardown='red',
                   plotdist=0.5, subplot=True, volume=True)
    
    return results


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='加密货币回测系统')
    parser.add_argument('--csv', type=str, help='结果CSV文件路径', default=None)
    parser.add_argument('--cash', type=float, help='初始资金', default=100000.0)
    parser.add_argument('--no-plot', action='store_true', help='不显示图表')
    
    args = parser.parse_args()
    
    # 运行回测
    run_backtest(csv_file=args.csv, plot=not args.no_plot, cash=args.cash)
