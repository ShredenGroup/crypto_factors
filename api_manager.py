import asyncio
import pandas as pd
import aiohttp
import json
from datetime import datetime, timezone
import time
from typing import List, Dict
import concurrent.futures
import logging

# 导入配置管理器
from config_manager import get_config_manager

class BinanceManager():
    def __init__(self, config_path: str = None):
        # 使用配置管理器替代直接读取配置文件
        self.config_manager = get_config_manager(config_path) if config_path else get_config_manager()
        
        # 设置日志
        logging_config = self.config_manager.get_logging_config()
        logging.basicConfig(
            level=getattr(logging, logging_config.get('level', 'INFO')),
            format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger(__name__)
        
        # 基础配置
        self.base_url = 'https://fapi.binance.com'
        self.limit = 1500
        self.max_concurrent_requests = 5

    def _convert_to_timestamp(self, dt_str: str) -> int:
        """将UTC时间字符串转换为毫秒时间戳"""
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _convert_to_datetime(self, timestamp: int) -> str:
        """将毫秒时间戳转换为UTC时间字符串"""
        dt = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    async def _fetch_kline(self, session: aiohttp.ClientSession, params: Dict) -> List:
        """获取单次K线数据"""
        url = f'{self.base_url}/fapi/v1/klines'
        self.logger.info(f"正在请求: {url} 参数: {params}")
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                self.logger.info(f"成功获取数据，条数: {len(data)}")
                return data
            else:
                error_text = await response.text()
                self.logger.error(f"请求失败: {response.status} - {error_text}")
                raise Exception(f"API请求失败: {response.status} - {error_text}")

    def _generate_time_ranges(self, start_ts: int, end_ts: int, interval_ms: int) -> List[Dict]:
        """生成时间范围列表，用于并发请求"""
        ranges = []
        current_end = end_ts
        while current_end > start_ts:
            current_start = max(start_ts, current_end - (self.limit * interval_ms))
            ranges.append({
                'startTime': current_start,
                'endTime': current_end
            })
            current_end = current_start - 1
        return ranges

    async def get_kline(self, symbol: str = None, interval: str = None, start_time: str = None, end_time: str = None) -> pd.DataFrame:
        try:
            # 使用配置管理器获取默认值
            symbol = symbol or self.config_manager.get_symbol()
            interval = interval or self.config_manager.get_interval()
            if start_time is None or end_time is None:
                start_time_default, end_time_default = self.config_manager.get_time_range()
                start_time = start_time or start_time_default
                end_time = end_time or end_time_default

            self.logger.info(f"\n开始获取数据:")
            self.logger.info(f"交易对: {symbol}")
            self.logger.info(f"时间间隔: {interval}")
            self.logger.info(f"开始时间: {start_time}")
            self.logger.info(f"结束时间: {end_time}")

            # 转换时间格式
            start_ts = self._convert_to_timestamp(start_time) if start_time else None
            end_ts = self._convert_to_timestamp(end_time) if end_time else None
            
            self.logger.info(f"开始时间戳: {start_ts}")
            self.logger.info(f"结束时间戳: {end_ts}")

            # 如果没有指定时间范围，直接获取最新数据
            if not start_ts and not end_ts:
                self.logger.info("未指定时间范围，获取最新数据")
                async with aiohttp.ClientSession() as session:
                    data = await self._fetch_kline(session, {
                        'symbol': symbol,
                        'interval': interval,
                        'limit': self.limit
                    })
                    return self._process_data(data)

            # 计算时间间隔的毫秒数
            interval_ms = {
                '1m': 60 * 1000,
                '3m': 3 * 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '2h': 2 * 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '6h': 6 * 60 * 60 * 1000,
                '8h': 8 * 60 * 60 * 1000,
                '12h': 12 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
            }[interval]

            # 生成时间范围
            time_ranges = self._generate_time_ranges(start_ts, end_ts, interval_ms)
            self.logger.info(f"需要请求的时间范围数量: {len(time_ranges)}")

            # 并发请求
            all_data = []
            async with aiohttp.ClientSession() as session:
                # 将时间范围分成多个批次
                for i in range(0, len(time_ranges), self.max_concurrent_requests):
                    batch = time_ranges[i:i + self.max_concurrent_requests]
                    tasks = []
                    for time_range in batch:
                        params = {
                            'symbol': symbol,
                            'interval': interval,
                            'limit': self.limit,
                            'startTime': time_range['startTime'],
                            'endTime': time_range['endTime']
                        }
                        tasks.append(self._fetch_kline(session, params))
                    
                    # 并发执行当前批次的任务
                    self.logger.info(f"\n执行第 {i//self.max_concurrent_requests + 1} 批请求，共 {len(batch)} 个请求")
                    results = await asyncio.gather(*tasks)
                    
                    # 处理结果
                    for data in results:
                        if data:
                            all_data.extend(data)
                    
                    self.logger.info(f"当前总数据条数: {len(all_data)}")
                    
                    # 添加小延迟，避免请求过于频繁
                    await asyncio.sleep(0.1)

            self.logger.info(f"\n数据获取完成，开始处理数据")
            # 处理数据
            df = self._process_data(all_data)
            
            # 按时间排序
            df = df.sort_index()
            
            # 如果指定了时间范围，进行过滤
            if start_ts:
                start_dt = pd.Timestamp(start_ts, unit='ms', tz='UTC')
                df = df[df.index >= start_dt]
            if end_ts:
                end_dt = pd.Timestamp(end_ts, unit='ms', tz='UTC')
                df = df[df.index <= end_dt]

            self.logger.info(f"数据处理完成，最终数据条数: {len(df)}")
            return df

        except Exception as e:
            self.logger.error(f"获取K线数据时出错: {str(e)}")
            raise

    def _process_data(self, data: List) -> pd.DataFrame:
        """处理原始数据为DataFrame"""
        self.logger.info("开始处理数据...")
        # 只取前11列数据
        data = [row[:11] for row in data]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote'
        ])
        # 转换时间戳为UTC时间并设置为索引
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        # 转换数值类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        self.logger.info("数据处理完成")
        return df


async def main():
    try:
        manager = BinanceManager()
        # 使用配置文件中的默认值获取数据
        df = await manager.get_kline()
        print("\n最终结果:")
        print(f"数据条数: {len(df)}")
        print("\n前5条数据:")
        print(df.head())
        print("\n后5条数据:")
        print(df.tail())
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
