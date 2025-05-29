import pandas as pd
import numpy as np
from api_manager import BinanceManager
import asyncio
from config_manager import get_config_manager

class CustomIndicator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def HBFC_ONE(self):
        """
        计算taker购买量与价格涨跌幅比值指标
        需要data中含有: close, taker_buy_base, taker_buy_quote, volume
        """
        # 计算价格涨跌幅（百分比）
        self.data['price_returns'] = self.data['close'].pct_change() * 100  # 百分比表示
        
        # 处理零值和极小值（避免除零）
        epsilon = 1e-10  # 一个极小值
        
        # 创建safe_returns，处理除零情况
        self.data['safe_returns'] = self.data['price_returns'].apply(
            lambda x: x if abs(x) > epsilon else (epsilon if x >= 0 else -epsilon)
        )
        
        # 计算taker买入与价格涨跌幅的比值
        self.data['taker_buy_base_per_return'] = self.data['taker_buy_base'] / self.data['safe_returns']
        self.data['taker_buy_quote_per_return'] = self.data['taker_buy_quote'] / self.data['safe_returns']
        
        # 最终HBFC_ONE为归一化后的taker买入与涨跌幅比值
        # 使用滚动Z-score标准化，使指标更稳定
        window = 20  # 20周期滚动窗口
        
        # 滚动均值和标准差
        rolling_mean = self.data['taker_buy_base_per_return'].rolling(window=window).mean()
        rolling_std = self.data['taker_buy_base_per_return'].rolling(window=window).std()
        
        # Z-score标准化
        self.data['HBFC_ONE'] = self.data['taker_buy_base']*((self.data['close']+self.data['open'])/2)/self.data['price_returns']
        
        # 填充NaN值
        self.data['HBFC_ONE'] = self.data['HBFC_ONE'].fillna(0)
        
        return self.data['HBFC_ONE']
        
    def calculate_all(self, output_path=None):
        """
        计算所有自定义指标
        
        Args:
            output_path: 输出CSV文件路径，如果为None则使用配置中的默认路径
        
        Returns:
            计算完成的DataFrame
        """
        # 计算所有指标
        self.HBFC_ONE()
        
        # 添加价格变动信号: 如果价格变动率大于1%则为1，否则为0
        if 'price_returns' in self.data.columns:
            self.data['price_change_signal'] = (self.data['price_returns'].abs() > 0.7).astype(int)
            print(f"已添加价格变动信号列 price_change_signal: 变动率>0.7% = 1, 否则 = 0")
        
        # 决定输出路径
        if output_path is None:
            # 从配置中获取默认路径
            cfg = get_config_manager()
            output_path = cfg.get_generator_config().get('result_path', './results.csv')
        
        # 保存到CSV
        self.data.to_csv(output_path, index=True)  # 保留日期索引
        
        # 计算包含的K线基础列和指标列
        basic_cols = ['open', 'high', 'low', 'close', 'volume']
        custom_cols = ['HBFC_ONE', 'taker_buy_base_per_return', 'taker_buy_quote_per_return', 'taker_buy_ratio']
        
        # 检查哪些K线列实际存在
        included_basic_cols = [col for col in basic_cols if col in self.data.columns]
        
        print(f"已将数据保存至: {output_path}")
        print(f"包含K线数据列: {included_basic_cols}")
        print(f"包含自定义指标: {custom_cols}")
        print(f"总行数: {len(self.data)}，总列数: {len(self.data.columns)}")
        
        # 后续可添加其他自定义指标
        return self.data

if __name__ == "__main__":
    async def main():
        print("开始获取K线数据...")
        manager = BinanceManager()
        data = await manager.get_kline()
        
        print(f"获取到 {len(data)} 条K线数据，开始计算自定义指标...")
        
        # 确认包含K线基础数据列
        kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in kline_cols if col not in data.columns]
        if missing_cols:
            print(f"警告: 数据缺少以下K线列: {missing_cols}")
        else:
            print(f"K线数据完整，包含所有必要列")
        
        # 创建自定义指标计算器
        custom_indicator = CustomIndicator(data)
        
        # 计算并保存到CSV
        result = custom_indicator.calculate_all('./result.csv')
        
        print("\n计算完成，指标预览:")
        # 显示几个结果样本
        preview_cols = ['close', 'HBFC_ONE']
        print(result[preview_cols].tail(10))
        
    # 执行异步主函数
    asyncio.run(main())