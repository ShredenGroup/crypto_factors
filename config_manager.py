import yaml
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional
import itertools
import numpy as np


class ConfigManager:
    """配置管理器，负责加载和提供对配置的访问"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config = None
        self.load_config()
    
    def load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"无法加载配置文件 {self.config_path}: {str(e)}")
    
    def reload_config(self) -> None:
        """重新加载配置文件"""
        self.load_config()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置
        
        Returns:
            日志配置字典
        """
        return self._config.get('logging', {})
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认交易配置
        
        Returns:
            默认配置字典，包含symbol, interval, start_time, end_time等
        """
        return self._config.get('default', {})
    
    def get_symbol(self) -> str:
        """获取交易对符号
        
        Returns:
            交易对符号，如 'BTCUSDT'
        """
        return self.get_default_config().get('symbol', '')
    
    def get_interval(self) -> str:
        """获取时间间隔
        
        Returns:
            K线时间间隔，如 '1h'
        """
        return self.get_default_config().get('interval', '')
    
    def get_time_range(self) -> tuple:
        """获取时间范围
        
        Returns:
            (start_time, end_time) 元组
        """
        default = self.get_default_config()
        return (
            default.get('start_time', ''),
            default.get('end_time', '')
        )
    
    def get_talib_config(self) -> Dict[str, Any]:
        """获取技术指标配置
        
        Returns:
            技术指标配置字典
        """
        return self._config.get('talib', {})
    
    def _generate_param_values(self, param_range: Dict[str, Any]) -> List[Any]:
        """根据参数范围生成参数值列表
        
        Args:
            param_range: 包含start, end, step的参数范围字典
            
        Returns:
            参数值列表
        """
        start = param_range.get('start')
        end = param_range.get('end')
        step = param_range.get('step')
        
        if all(x is not None for x in [start, end, step]):
            # 为了确保包含end值，使用np.arange并进行适当处理
            values = np.arange(start, end + step / 2, step).tolist()
            # 对于整数参数，转换为整数
            if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
                values = [int(v) for v in values]
            return values
        return []
    
    def _generate_params_from_range(self, params_range: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据参数范围定义生成所有参数组合
        
        Args:
            params_range: 参数范围配置字典
            
        Returns:
            参数组合列表
        """
        # 存储每个参数的所有可能值
        param_values = {}
        
        for param_name, param_config in params_range.items():
            if isinstance(param_config, dict) and 'start' in param_config:
                # 范围类型参数
                param_values[param_name] = self._generate_param_values(param_config)
            elif isinstance(param_config, list):
                # 离散列表类型参数
                param_values[param_name] = param_config
            else:
                # 单值类型参数
                param_values[param_name] = [param_config]
        
        # 没有参数时返回空列表
        if not param_values:
            return []
        
        # 生成笛卡尔积获取所有参数组合
        param_names = list(param_values.keys())
        param_combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        # 转换为字典列表
        return [dict(zip(param_names, combo)) for combo in param_combinations]
    
    def get_all_indicators(self) -> List[str]:
        """获取所有配置的技术指标名称
        
        Returns:
            指标名称列表
        """
        return list(self.get_talib_config().keys())

    def get_param_combinations_count(self, indicator_name: str) -> int:
        """获取指标参数组合的数量
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            参数组合数量
        """
        return len(self.get_indicator_params(indicator_name))
    
    def get_total_combinations_count(self) -> Dict[str, int]:
        """获取所有指标的参数组合数量
        
        Returns:
            包含各指标参数组合数量的字典
        """
        result = {}
        for indicator in self.get_all_indicators():
            count = self.get_param_combinations_count(indicator)
            result[indicator] = count
        return result

    # ---------------- Generator helpers ----------------
    def get_generator_config(self) -> Dict[str, Any]:
        """返回 generator 配置块"""
        return self._config.get('generator', {})

    def get_generator_mode(self) -> str:
        return self.get_generator_config().get('generator_mode', 'all')

    def get_parameter_mode(self) -> str:
        return self.get_generator_config().get('parameter_mode', 'multiple')

    def get_selected_indicators(self) -> List[str]:
        """根据 generator_mode 返回需要计算的指标列表"""
        mode = self.get_generator_mode()
        gen_cfg = self.get_generator_config()
        if mode == 'single':
            ind = gen_cfg.get('indicator')
            return [ind] if ind else []
        elif mode == 'multiple':
            return gen_cfg.get('indicators', [])
        else:  # all
            return self.get_all_indicators()

    # ---------------- Indicator param helpers ----------------
    def get_indicator_params(self, indicator_name: str, parameter_mode: str = 'multiple') -> List[Dict[str, Any]]:
        """获取特定技术指标的参数列表，支持 single / multiple 模式。

        Args:
            indicator_name: 指标名称，如 'BBANDS', 'EMA' 等
            parameter_mode: 'single' 返回一条默认/第一条参数，'multiple' 返回全部组合
        """
        talib_config = self.get_talib_config()
        indicator_config = talib_config.get(indicator_name, {})

        # helper to pick first/ default
        def _first_param() -> List[Dict[str, Any]]:
            # 优先 default
            if 'default' in indicator_config:
                return [indicator_config['default']]
            if 'params' in indicator_config and indicator_config['params']:
                return [indicator_config['params'][0]]
            if 'params_range' in indicator_config:
                generated = self._generate_params_from_range(indicator_config['params_range'])
                return [generated[0]] if generated else []
            return [{}]

        # ---- multiple 模式 ----
        if parameter_mode == 'multiple':
            # 如果有直接定义的params，优先使用
            if 'params' in indicator_config:
                return indicator_config['params']
            # 否则检查是否有params_range定义，生成参数组合
            if 'params_range' in indicator_config:
                return self._generate_params_from_range(indicator_config['params_range'])
            # fallback to default / empty
            return _first_param()
        else:  # single
            return _first_param()


@lru_cache(maxsize=1)
def get_config_manager(config_path: str = 'config.yaml') -> ConfigManager:
    """获取配置管理器单例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    return ConfigManager(config_path)


# 使用示例
if __name__ == "__main__":
    # 获取配置管理器
    config = get_config_manager()
    
    # 打印日志配置
    print("日志配置:", config.get_logging_config())
    
    # 打印交易对和时间间隔
    print(f"交易对: {config.get_symbol()}, 时间间隔: {config.get_interval()}")
    
    # 打印时间范围
    start, end = config.get_time_range()
    print(f"时间范围: {start} 至 {end}")
    
    # 打印所有技术指标
    print("\n配置的技术指标:", config.get_all_indicators())
    
    # 打印每个指标的所有参数组合
    for indicator in config.get_all_indicators():
        params = config.get_indicator_params(indicator)
        print(f"\n{indicator} 参数组合 ({len(params)}个):")
        for i, param in enumerate(params[:5]):  # 只显示前5个组合
            print(f"  组合 {i+1}: {param}")
        if len(params) > 5:
            print(f"  ... 还有 {len(params)-5} 个组合")
    
    # 打印所有指标的参数组合总数
    print("\n各指标参数组合数量:")
    for indicator, count in config.get_total_combinations_count().items():
        print(f"  {indicator}: {count}个组合") 