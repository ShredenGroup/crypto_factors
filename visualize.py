import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import mplfinance as mpf
import seaborn as sns
import os
import argparse
from datetime import datetime, timedelta

def load_data(csv_path='result.csv'):
    """加载CSV数据并进行预处理"""
    print(f"加载数据: {csv_path}")
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return None
    
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 处理时间列
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"列名: {df.columns.tolist()}")
    
    return df

def plot_kline_with_indicator(df, days=7, show_volume=False):
    """绘制K线图和HBFC_ONE指标"""
    # 确保数据有必要的OHLC列
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        print("错误: 数据缺少必要的OHLC列")
        return
    
    # 限制显示时间范围
    if days > 0:
        end_date = df.index[-1]
        start_date = end_date - timedelta(days=days)
        df_plot = df.loc[start_date:end_date]
    else:
        df_plot = df
    
    print(f"绘制 {df_plot.index[0]} 到 {df_plot.index[-1]} 的数据，共 {len(df_plot)} 条记录")
    
    # 准备mplfinance可用的数据格式
    df_mpf = df_plot[['open', 'high', 'low', 'close']].copy()
    if 'volume' not in df_mpf.columns and show_volume:
        print("警告: 数据缺少volume列，将不显示成交量")
        show_volume = False
    
    # 添加HBFC_ONE到附加面板
    apds = []
    if 'HBFC_ONE' in df.columns:
        apds.append(mpf.make_addplot(df_plot['HBFC_ONE'], panel=1, color='blue', 
                                     title='HBFC_ONE', ylabel='Value'))
    
    # 添加price_change_signal标记 - 高亮显示价格变动超过阈值的K线
    if 'price_change_signal' in df_plot.columns:
        # 创建一个只在信号为1的位置有值的Series，其余位置为NaN
        signal_markers = pd.Series(index=df_plot.index, dtype=float)
        signal_markers.loc[df_plot['price_change_signal'] == 1] = df_plot.loc[df_plot['price_change_signal'] == 1, 'high'] * 1.01  # 略高于高点
        
        # 计算有多少个信号
        signal_count = (df_plot['price_change_signal'] == 1).sum()
        print(f"发现 {signal_count} 个价格变动信号点 (price_change_signal=1)")
        
        # 添加标记到主K线面板
        if signal_count > 0:
            apds.append(mpf.make_addplot(signal_markers, type='scatter', markersize=100,
                                       marker='^', color='red', alpha=0.7, panel=0))
            
            # 为了图例，我们需要在绘图后手动添加
            legend_marker = {"color": "red", "marker": "^", "markersize": 8}
            legend_text = "价格变动信号 (>0.7%)"
    
    # 创建样式
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick={'up':'green', 'down':'red'},
        volume={'up':'green', 'down':'red'},
    )
    
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridstyle=':',
        y_on_right=False,
        gridaxis='both',
    )
    
    # 创建图表
    fig, axes = mpf.plot(
        df_mpf,
        type='candle',
        addplot=apds,
        volume=False,  # 不显示成交量
        style=s,
        figsize=(12, 8),
        panel_ratios=(3, 1) if len(apds) > 0 else (1, 0),
        title=f'K线图与HBFC_ONE指标 ({df_plot.index[0].date()} 到 {df_plot.index[-1].date()})',
        returnfig=True
    )
    
    # 手动添加图例 - 如果有信号标记
    if 'price_change_signal' in df_plot.columns and signal_count > 0:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                                 markersize=10, label=f'价格变动信号 (>0.7%)')]
        axes[0].legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
def plot_hbfc_correlation(df):
    """分析HBFC_ONE与价格变动的相关性"""
    if 'HBFC_ONE' not in df.columns or 'price_returns' not in df.columns:
        print("错误: 数据缺少HBFC_ONE或price_returns列")
        return
    
    # 创建新的DataFrame进行相关性分析
    corr_df = df[['close', 'HBFC_ONE', 'price_returns']].copy()
    
    # 计算未来1、3、5期的价格变动百分比
    for i in [1, 3, 5]:
        corr_df[f'future_return_{i}'] = corr_df['close'].pct_change(periods=i).shift(-i) * 100
    
    # 删除NaN值
    corr_df = corr_df.dropna()
    
    # 计算相关性矩阵
    correlation = corr_df.corr()
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('HBFC_ONE与价格变动相关性分析')
    plt.tight_layout()
    plt.show()
    
    # 绘制散点图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, period in enumerate([1, 3, 5]):
        sns.regplot(x='HBFC_ONE', y=f'future_return_{period}', data=corr_df, ax=axes[i])
        axes[i].set_title(f'HBFC_ONE vs {period}期后收益率')
        axes[i].set_xlabel('HBFC_ONE')
        axes[i].set_ylabel(f'{period}期后收益率 (%)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return correlation

def plot_distribution(df):
    """分析指标分布情况"""
    # 检查必要列存在
    indicator_cols = ['HBFC_ONE', 'taker_buy_base_per_return', 'price_returns']
    available_cols = [col for col in indicator_cols if col in df.columns]
    
    if not available_cols:
        print("错误: 数据缺少所需的指标列")
        return
    
    # 创建子图
    fig, axes = plt.subplots(len(available_cols), 1, figsize=(12, 4*len(available_cols)))
    if len(available_cols) == 1:
        axes = [axes]
    
    # 遍历每个指标并绘制直方图和核密度估计
    for i, col in enumerate(available_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'{col} 分布')
        axes[i].grid(True, alpha=0.3)
        
        # 添加基本统计信息
        stats = df[col].describe()
        info_text = (f"均值: {stats['mean']:.4f}\n"
                    f"中位数: {stats['50%']:.4f}\n"
                    f"标准差: {stats['std']:.4f}\n"
                    f"最小值: {stats['min']:.4f}\n"
                    f"最大值: {stats['max']:.4f}")
        
        axes[i].text(0.02, 0.95, info_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='可视化分析结果CSV文件')
    parser.add_argument('--csv', type=str, default='result.csv', help='CSV文件路径')
    parser.add_argument('--days', type=int, default=7, help='显示的天数 (0 = 全部)')
    parser.add_argument('--distribution', action='store_true', help='显示指标分布')
    parser.add_argument('--correlation', action='store_true', help='显示相关性分析')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_data(args.csv)
    if df is None:
        return
    
    # 根据选项绘图
    if args.distribution:
        plot_distribution(df)
    elif args.correlation:
        plot_hbfc_correlation(df)
    else:
        plot_kline_with_indicator(df, args.days)

if __name__ == "__main__":
    main()
