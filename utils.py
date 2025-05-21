def convert_df_to_talib_format(df):
    """
    将 pandas DataFrame 的所有列转换为 numpy 数组格式。
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: 包含所有列的 numpy arrays 的字典
    """
    # 创建包含所有列的字典
    talib_data = {}
    for col in df.columns:
        # 将每一列转换为 numpy 数组
        talib_data[col] = df[col].values
    
    return talib_data

def convert_df_to_talib_pvformat(df):
    """
    将 pandas DataFrame 转换为 talib.abstract 可用的字典格式。
    
    Args:
        df: pandas DataFrame，包含 OHLCV 数据
        
    Returns:
        dict: 包含 numpy arrays 的字典，键为 'open', 'high', 'low', 'close', 'volume'
    """
    # 确保所有必要的列都存在
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 缺少必要的列: {col}")
    
    # 创建符合 talib.abstract 格式的字典
    talib_data = {}
    for col in required_columns:
        # 确保数据是 numpy 数组格式
        talib_data[col] = df[col].values
    
    return talib_data