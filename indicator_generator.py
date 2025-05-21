from utils import convert_df_to_talib_pvformat
import numpy as np
import pandas as pd
from api_manager import BinanceManager
import asyncio
from talib.abstract import Function
from talib.abstract import *
import os
# 导入配置管理器
from config_manager import get_config_manager
import multiprocessing as mp

class TALibBatchProcessor:
    def __init__(self, data:dict, config_path:str=None, result_path:str='./result.csv'):
        """初始化TALib批处理器
        
        Args:
            data: 价格数据，格式需适配TALib
            config_path: 配置文件路径，如果为None则使用默认路径
            result_path: 结果保存路径
        """
        # 使用配置管理器替代直接读取配置文件
        self.config_manager = get_config_manager(config_path) if config_path else get_config_manager()
        self.data = data.copy()
        
        # 结果文件和日志控制
        generator_cfg = self.config_manager._config.get('generator', {})
        self.verbose = generator_cfg.get('verbose', True)
        self.result_path = result_path or generator_cfg.get('result_path', './result.csv')
        
        # 其他生成器相关字段（防止属性缺失）
        self.generator_mode = generator_cfg.get('generator_mode', 'all')
        self.parameter_mode = generator_cfg.get('parameter_mode', 'multiple')
        self.single_indicator = generator_cfg.get('indicator', '')
        self.multiple_indicators = generator_cfg.get('indicators', [])
        
        if self.verbose:
            print(f"TALibBatchProcessor 初始化完成: result_path={self.result_path}")
        
    def run_on_func_name(self, func_name):
        """运行指定技术指标函数，使用配置中定义的参数
        
        Args:
            func_name: 技术指标函数名称
        """
        print('运行技术指标函数: ', func_name)
        
        # 从配置管理器获取参数配置
        func_params = self.config_manager.get_indicator_params(func_name)
        print('函数参数列表: ', func_params)
        
        if not func_params:
            print(f"警告: {func_name} 没有参数配置")
            return
            
        func_object = Function(func_name)
        
        if len(func_params) > 0:
            for params in func_params:
                func_object.parameters = params
                print('函数对象信息: ', func_object.info)
                _results = func_object.run(self.data)
                feature_name = "{}_".format(func_name)
                for param in params.items():
                    feature_name += "{}_{}_".format(param[0], param[1])

                if isinstance(_results, list):
                    for (i, output_name) in enumerate(func_object.output_names):
                        _key_name = feature_name + output_name
                        self.data[_key_name] = _results[i]
                        # 保存每个指标结果到CSV
                        self.save_results_to_csv(_key_name, _results[i])
                else:
                    _key_name = feature_name + func_object.output_names[0]
                    self.data[_key_name] = _results
                    # 保存指标结果到CSV
                    self.save_results_to_csv(_key_name, _results)
        else:
            _results = func_object.run(self.data)
            feature_name = "{}_".format(func_name)
            if isinstance(_results, list):
                for (i, output_name) in enumerate(func_object.output_names):
                    _key_name = feature_name + output_name
                    self.data[_key_name] = _results[i]
                    # 保存每个指标结果到CSV
                    self.save_results_to_csv(_key_name, _results[i])
            else:
                _key_name = feature_name + func_object.output_names[0]
                self.data[_key_name] = _results
                # 保存指标结果到CSV
                self.save_results_to_csv(_key_name, _results)

    def run_on_func_all(self, func_name):
        """运行指定技术指标函数的所有可能参数组合
        
        使用配置中定义的参数范围，自动生成所有参数组合并计算指标
        
        Args:
            func_name: 技术指标函数名称
        """
        print(f'运行技术指标函数 {func_name} 的所有参数组合')
        
        # 从配置管理器获取所有参数组合
        all_params = self.config_manager.get_indicator_params(func_name)
        if not all_params:
            print(f"警告: {func_name} 没有参数配置或无法生成参数组合")
            return
            
        print(f"总共 {len(all_params)} 种参数组合")
        
        # 创建函数对象
        func_object = Function(func_name)
        
        # 遍历所有参数组合
        for i, params in enumerate(all_params):
            try:
                # 设置参数
                func_object.parameters = params
                
                # 生成特征名
                feature_name = "{}_".format(func_name)
                for param in params.items():
                    feature_name += "{}_{}_".format(param[0], param[1])
                
                # 运行指标函数
                _results = func_object.run(self.data)
                
                # 处理结果
                if isinstance(_results, list):
                    for (j, output_name) in enumerate(func_object.output_names):
                        _key_name = feature_name + output_name
                        self.data[_key_name] = _results[j]
                        # 保存每个指标结果到CSV
                        self.save_results_to_csv(_key_name, _results[j])
                else:
                    _key_name = feature_name + func_object.output_names[0]
                    self.data[_key_name] = _results
                    # 保存指标结果到CSV
                    self.save_results_to_csv(_key_name, _results)
                
                # 每处理10个参数组合输出一次进度
                if (i + 1) % 10 == 0 or i == len(all_params) - 1:
                    print(f"处理进度: {i + 1}/{len(all_params)}")
                    
            except Exception as e:
                print(f"计算参数组合失败 {params}: {str(e)}")
                
        print(f"{func_name} 所有参数组合处理完成")

    def save_results_to_csv(self, column_name, column_value):
        """将指标计算结果保存到CSV文件
        
        Args:
            column_name: 列名
            column_value: 列值（可以是单个值或数组）
        """
        # 检查文件是否存在
        file_exists = os.path.exists(self.result_path)
        
        if file_exists:
            # 如果文件存在，先读取现有文件
            existing_df = pd.read_csv(self.result_path)
            
            # 添加新列
            if isinstance(column_value, np.ndarray):
                # 确保数组长度与DataFrame行数匹配
                if len(existing_df) == len(column_value):
                    existing_df[column_name] = column_value
                else:
                    # 如果长度不匹配，可能需要调整
                    print(f"警告: 列 {column_name} 长度与现有数据不匹配，将只保存匹配的部分")
                    min_len = min(len(existing_df), len(column_value))
                    existing_df = existing_df.iloc[:min_len]
                    existing_df[column_name] = column_value[:min_len]
            else:
                # 单个值，扩展为与DataFrame同样行数的列
                existing_df[column_name] = column_value
            
            # 保存回文件
            existing_df.to_csv(self.result_path, index=False)
            print(f"结果列 '{column_name}' 已添加到文件: {self.result_path}")
        else:
            # 如果文件不存在，创建新文件
            df_result = pd.DataFrame()
            
            # 添加新列
            if isinstance(column_value, np.ndarray):
                df_result[column_name] = column_value
            else:
                # 单个值，创建只有一行的DataFrame
                df_result[column_name] = [column_value]
            
            # 保存到文件
            df_result.to_csv(self.result_path, index=False)
            print(f"结果列 '{column_name}' 已写入新文件: {self.result_path}")

    def run_on_all_func_names(self):
        """运行配置中所有技术指标函数"""
        all_indicators = self.config_manager.get_all_indicators()
        print(f"将处理 {len(all_indicators)} 个技术指标:")
        for func_name in all_indicators:
            print(f"处理指标: {func_name}")
            self.run_on_func_name(func_name)
            
    def run_on_func_default(self, func_name):
        """运行指定技术指标函数的默认参数配置"""
        if self.verbose:
            print(f"运行 {func_name} 的默认参数配置")
        # 获取默认参数
        talib_cfg = self.config_manager._config.get('talib', {})
        cfg = talib_cfg.get(func_name, {})
        default_params = cfg.get('default', {})
        # 如果没有default，回退为无参数调用
        params_list = [default_params] if default_params else [{}]
        func_object = Function(func_name)
        for params in params_list:
            func_object.parameters = params
            _results = func_object.run(self.data)
            feature_name = f"{func_name}_default_"
            for k, v in params.items():
                feature_name += f"{k}_{v}_"
            if isinstance(_results, list):
                for i, output_name in enumerate(func_object.output_names):
                    col = feature_name + output_name
                    self.data[col] = _results[i]
                    self.save_results_to_csv(col, _results[i])
            else:
                col = feature_name + func_object.output_names[0]
                self.data[col] = _results
                self.save_results_to_csv(col, _results)

    def run_on_multiple_funcs_default(self, indicators: list):
        """运行多个指标的默认参数配置"""
        for ind in indicators:
            self.run_on_func_default(ind)

    # --- 多进程加速全部指标全参数 ---
    def _worker_all_params(self, func_name, data_pickle, queue, verbose=False):
        """子进程工作函数，计算单个指标所有参数并发送结果到队列"""
        import pickle, numpy as np, pandas as pd, traceback
        from talib.abstract import Function
        data = pickle.loads(data_pickle)
        config = get_config_manager()
        params_list = config.get_indicator_params(func_name)
        func_object = Function(func_name)

        for params in params_list:
            # 特殊规则：MAMA 要求 fastlimit > slowlimit
            if func_name == "MAMA":
                fl = params.get("fastlimit", 0.5)
                sl = params.get("slowlimit", 0.05)
                if sl >= fl:
                    if verbose:
                        print(f"跳过无效 MAMA 参数 fastlimit={fl}, slowlimit={sl}")
                    continue

            try:
                func_object.parameters = params
                results = func_object.run(data)
            except Exception as e:
                if verbose:
                    print(f"{func_name} 参数 {params} 计算失败: {e}")
                    traceback.print_exc()
                continue  # 跳过错误参数组合

            base_name = f"{func_name}_"
            for k, v in params.items():
                base_name += f"{k}_{v}_"
            if isinstance(results, list):
                for i, out_name in enumerate(func_object.output_names):
                    col = base_name + out_name
                    queue.put((col, results[i]))
            else:
                col = base_name + func_object.output_names[0]
                queue.put((col, results))

        # 告知完成
        queue.put(("__DONE__", func_name))

    def _writer_process(self, queue, result_path):
        import pandas as pd, os, numpy as np, gc
        df_result = None
        batch_data = {}
        BATCH_SIZE = 50  # 收集多少列后批量写入

        def flush_batch():
            nonlocal df_result, batch_data
            if not batch_data:
                return
            batch_df = pd.DataFrame(batch_data)
            if df_result is None:
                df_result = batch_df
            else:
                # 对齐行数
                min_len = min(len(df_result), len(batch_df))
                df_result = df_result.iloc[:min_len]
                batch_df = batch_df.iloc[:min_len]
                df_result = pd.concat([df_result, batch_df], axis=1)
            batch_data.clear()
            # 主动触发垃圾回收，减小内存
            gc.collect()

        while True:
            item = queue.get()
            if item == "__STOP__":
                # 终止信号，写完剩余批次
                flush_batch()
                break
            col, values = item
            if col == "__DONE__":
                # 完成信号，不处理
                continue
            # 初始化 df_result 索引
            if df_result is None and not batch_data:
                if os.path.exists(result_path):
                    df_result = pd.read_csv(result_path)
                else:
                    if isinstance(values, np.ndarray):
                        df_result = pd.DataFrame(index=range(len(values)))
            # 截断长度一致
            if isinstance(values, np.ndarray) and df_result is not None:
                if len(df_result) != len(values):
                    min_len = min(len(df_result), len(values))
                    values = values[:min_len]
            batch_data[col] = values
            if len(batch_data) >= BATCH_SIZE:
                flush_batch()
                if df_result is not None:
                    df_result.to_csv(result_path, index=False)

        # 最终保存
        if df_result is not None:
            df_result.to_csv(result_path, index=False)

    def run_all_indicators_all_params(self):
        """使用多进程计算所有指标的所有参数组合，并通过消息队列写入结果"""
        all_indicators = self.config_manager.get_all_indicators()
        self.run_indicators_all_params(all_indicators)

    # 新增：支持指定子集指标
    def run_indicators_all_params(self, indicators:list):
        """多进程计算给定指标列表的所有参数组合"""
        if not indicators:
            print("指标列表为空，跳过计算")
            return
        queue = mp.Queue()
        import pickle
        data_pickle = pickle.dumps(self.data)
        writer = mp.Process(target=self._writer_process, args=(queue, self.result_path))
        writer.start()
        workers = []
        for ind in indicators:
            p = mp.Process(target=self._worker_all_params, args=(ind, data_pickle, queue, self.verbose))
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
        queue.put("__STOP__")
        writer.join()
        if self.verbose:
            print("指定指标多进程计算完成")

    # === 新增：多进程计算默认参数 ===
    def _worker_default(self, func_name, data_pickle, queue, verbose=False):
        """子进程工作函数，只计算单个指标的默认参数并发送结果到队列"""
        import pickle, numpy as np, traceback
        from talib.abstract import Function
        data = pickle.loads(data_pickle)
        config = get_config_manager()
        # 仅获取一组默认参数
        params_list = config.get_indicator_params(func_name, 'single') or [{}]
        func_object = Function(func_name)

        for params in params_list:
            try:
                func_object.parameters = params
                results = func_object.run(data)
            except Exception as e:
                if verbose:
                    print(f"{func_name} 默认参数计算失败: {e}")
                    traceback.print_exc()
                continue
            base_name = f"{func_name}_default_"
            for k, v in params.items():
                base_name += f"{k}_{v}_"
            if isinstance(results, list):
                for i, out_name in enumerate(func_object.output_names):
                    queue.put((base_name + out_name, results[i]))
            else:
                queue.put((base_name + func_object.output_names[0], results))
        queue.put(("__DONE__", func_name))

    def run_indicators_default_mp(self, indicators:list):
        """多进程计算多个指标的默认参数（单参数）"""
        if not indicators:
            print("指标列表为空，跳过计算")
            return
        queue = mp.Queue()
        import pickle
        data_pickle = pickle.dumps(self.data)
        writer = mp.Process(target=self._writer_process, args=(queue, self.result_path))
        writer.start()
        workers = []
        for ind in indicators:
            p = mp.Process(target=self._worker_default, args=(ind, data_pickle, queue, self.verbose))
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
        queue.put("__STOP__")
        writer.join()
        if self.verbose:
            print("默认参数多进程计算完成")

    def run_all_indicators_default_mp(self):
        """多进程计算所有指标的默认参数"""
        all_indicators = self.config_manager.get_all_indicators()
        self.run_indicators_default_mp(all_indicators)

if __name__ == "__main__":

    async def main():
        try:
            # 获取数据
            manager = BinanceManager()
            df = await manager.get_kline()
            new_df = convert_df_to_talib_pvformat(df)
            
            print(f"获取到 {len(df)} 条K线数据")
            
            # 简单测试某个指标
            print("简单测试ADOSC指标:")
            ad = ADOSC(new_df['high'], new_df['low'], new_df['close'], new_df['volume'], 3, 10)
            print(f"ADOSC结果长度: {len(ad)}")
            print(f"前5个值: {ad[:5]}")
            
            # 创建指标处理器
            print("\n创建指标处理器并执行批量计算...")
            processor = TALibBatchProcessor(new_df)
            indicator = "EMA"
            # 运行单个指标的所有参数组合
            #indicator = "EMA"  # 可以修改为任何配置的指标
            #processor.run_on_func_all(indicator)
            
            # 运行所有指标（可选，取消注释以运行）
            # processor.run_on_all_func_names()  # 每个指标使用配置的参数
            processor.run_all_indicators_all_params()  # 每个指标使用所有可能的参数组合
            
            print("\n处理完成")
            
        except Exception as e:
            print(f"错误: {str(e)}")
    
    asyncio.run(main())