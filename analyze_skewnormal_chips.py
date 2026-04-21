#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股筹码分布偏态正态分布分析脚本
功能：筛选筹码分布符合偏态正态分布的A股股票
作者：AI助手
日期：2026-04-19
"""

import tushare as ts
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import time
from datetime import datetime, timedelta
import sys
import json
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 可调整参数 ====================
# 1. 偏态正态分布拟合参数
SKEWNESS_THRESHOLD = 0.5  # 形状参数alpha的阈值，控制偏度容忍度
P_VALUE_THRESHOLD = 0.10  # 拟合优度检验的p值阈值（小于此值拒绝原假设）
SAMPLE_SIZE_MIN = 50      # 最小样本量要求

# 2. 数据获取参数
STOCK_LIMIT = 100         # 测试股票数量限制（设为None表示全部A股）
START_DATE = "20250101"   # 筹码数据开始日期
END_DATE = "20260419"     # 筹码数据结束日期

# 3. 输出控制参数
OUTPUT_TO_FILE = True     # 是否输出到文件
OUTPUT_FILENAME = "skewnormal_stocks_analysis.csv"  # 输出文件名
VERBOSE = True            # 是否显示详细进度
PLOT_DISTRIBUTIONS = False # 是否绘制分布图（设置为True会生成图表）

# ==================== tushare配置 ====================
# 需要设置您的tushare token，请从https://tushare.pro/注册获取
TUSHARE_TOKEN = "xxx"  # 请替换为您的token

def init_tushare():
    """初始化tushare"""
    if TUSHARE_TOKEN == "xxx":
        print("错误：请先设置TUSHARE_TOKEN")
        print("1. 访问 https://tushare.pro/ 注册账号")
        print("2. 在个人中心获取token")
        print("3. 替换脚本中的TUSHARE_TOKEN")
        sys.exit(1)
    
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()

def get_stock_list(pro):
    """获取A股股票列表"""
    print("获取A股股票列表...")
    
    try:
        # 获取上市状态为L（上市）的A股股票
        df = pro.stock_basic(
            exchange='',  # 空字符串表示所有交易所
            list_status='L',  # L上市，D退市，P暂停上市
            fields='ts_code,name,industry,list_date,market'
        )
        
        # 筛选沪深A股（排除北交所等）
        df = df[df['ts_code'].str.endswith(('.SH', '.SZ'))]
        
        if STOCK_LIMIT and len(df) > STOCK_LIMIT:
            print(f"限制测试股票数量为前 {STOCK_LIMIT} 只")
            df = df.head(STOCK_LIMIT)
        
        print(f"共获取 {len(df)} 只A股股票")
        return df
    
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        print("请检查网络连接和tushare token设置")
        sys.exit(1)

def get_price_data(pro, ts_code, start_date, end_date):
    """
    获取个股价格数据（作为筹码分布的替代指标）
    由于tushare没有直接的筹码分布接口，我们使用收盘价序列进行分析
    """
    try:
        # 获取日线行情数据
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,close,vol'
        )
        
        if len(df) < SAMPLE_SIZE_MIN:
            if VERBOSE:
                print(f"  {ts_code} 数据量不足: {len(df)} < {SAMPLE_SIZE_MIN}")
            return None
        
        # 返回收盘价序列
        return df['close'].values
    
    except Exception as e:
        if VERBOSE:
            print(f"  获取 {ts_code} 数据失败: {e}")
        return None

def test_skewnormal_fit(data):
    """
    检验数据是否符合偏态正态分布
    
    偏态正态分布参数：
    - loc: 位置参数（均值）
    - scale: 尺度参数（标准差）
    - a: 形状参数（偏度参数，alpha）
    
    返回：
    - is_skewnormal: 是否接受偏态正态分布假设
    - params: 拟合参数 (loc, scale, a)
    - p_value: 拟合优度检验p值
    - skewness: 实际偏度
    - kurtosis: 实际峰度
    """
    if len(data) < SAMPLE_SIZE_MIN:
        return False, None, None, None, None
    
    # 计算实际统计量
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    mean = np.mean(data)
    std = np.std(data)
    
    try:
        # 使用scipy的skewnorm分布进行拟合
        # skewnorm的参数：a=形状参数，loc=位置，scale=尺度
        a, loc, scale = stats.skewnorm.fit(data)
        
        # Kolmogorov-Smirnov检验：检验数据是否来自拟合的分布
        # 原假设H0：数据来自偏态正态分布
        ks_stat, p_value = stats.kstest(
            data, 
            'skewnorm', 
            args=(a, loc, scale)
        )
        
        # 判断条件：
        # 1. p值大于阈值（不拒绝原假设）
        # 2. 形状参数a的绝对值小于阈值（控制偏度程度）
        is_skewnormal = (
            p_value > P_VALUE_THRESHOLD and 
            abs(a) < SKEWNESS_THRESHOLD
        )
        
        return is_skewnormal, (loc, scale, a), p_value, skewness, kurtosis
        
    except Exception as e:
        if VERBOSE:
            print(f"  拟合失败: {e}")
        return False, None, None, skewness, kurtosis

def plot_distribution(ts_code, name, data, params, p_value, skewness, idx):
    """绘制分布图"""
    if not PLOT_DISTRIBUTIONS:
        return
    
    plt.figure(figsize=(12, 5))
    
    # 子图1：直方图和拟合曲线
    plt.subplot(1, 2, 1)
    
    # 绘制直方图
    plt.hist(data, bins=30, density=True, alpha=0.6, color='blue', label='实际数据')
    
    # 绘制拟合的偏态正态分布曲线
    if params:
        loc, scale, a = params
        x = np.linspace(np.min(data), np.max(data), 1000)
        y = stats.skewnorm.pdf(x, a, loc, scale)
        plt.plot(x, y, 'r-', lw=2, label=f'偏态正态拟合 (a={a:.3f})')
    
    plt.title(f'{ts_code} {name}\n偏度: {skewness:.3f}, p值: {p_value:.4f}')
    plt.xlabel('价格')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：Q-Q图
    plt.subplot(1, 2, 2)
    if params:
        loc, scale, a = params
        # 生成理论分位数
        theoretical = stats.skewnorm.ppf(np.linspace(0.01, 0.99, len(data)), a, loc, scale)
        plt.scatter(theoretical, np.sort(data), alpha=0.6)
        
        # 添加对角线
        min_val = min(np.min(data), np.min(theoretical))
        max_val = max(np.max(data), np.max(theoretical))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美拟合线')
        
        plt.title('Q-Q图')
        plt.xlabel('理论分位数')
        plt.ylabel('样本分位数')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'distribution_plot_{idx:03d}.png', dpi=150)
    plt.close()

def analyze_stocks(pro, stock_list):
    """分析股票列表"""
    results = []
    
    print(f"\n开始分析 {len(stock_list)} 只股票...")
    print("=" * 80)
    
    for idx, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        if VERBOSE:
            print(f"分析 {idx+1}/{len(stock_list)}: {ts_code} {name}")
        
        # 获取价格数据（作为筹码分布的代理）
        data = get_price_data(pro, ts_code, START_DATE, END_DATE)
        
        if data is None:
            continue
        
        # 进行偏态正态分布检验
        is_skewnormal, params, p_value, skewness, kurtosis = test_skewnormal_fit(data)
        
        # 绘制分布图（如果启用）
        if is_skewnormal and PLOT_DISTRIBUTIONS:
            plot_distribution(ts_code, name, data, params, p_value, skewness, idx)
        
        if is_skewnormal:
            loc, scale, a = params
            result = {
                'ts_code': ts_code,
                'name': name,
                'industry': row['industry'],
                'market': row['market'],
                'list_date': row['list_date'],
                'sample_size': len(data),
                'p_value': p_value,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shape_param_a': a,
                'loc_param': loc,
                'scale_param': scale,
                'mean_price': np.mean(data),
                'std_price': np.std(data),
                'min_price': np.min(data),
                'max_price': np.max(data),
                'is_skewnormal': True
            }
            results.append(result)
            
            if VERBOSE:
                print(f"  ✓ 符合偏态正态分布")
                print(f"     形状参数a: {a:.4f}, p值: {p_value:.4f}, 偏度: {skewness:.4f}")
        else:
            if VERBOSE and p_value is not None:
                reason = []
                if p_value <= P_VALUE_THRESHOLD:
                    reason.append(f"p值过低({p_value:.4f}<={P_VALUE_THRESHOLD})")
                if params and abs(params[2]) >= SKEWNESS_THRESHOLD:
                    reason.append(f"偏度过大(|{params[2]:.4f}|>={SKEWNESS_THRESHOLD})")
                if reason:
                    print(f"  ✗ 不符合: {', '.join(reason)}")
        
        # 避免请求过于频繁（tushare API限制）
        time.sleep(0.15)
    
    return results

def save_results(results, stock_list):
    """保存结果到文件"""
    if not results:
        print("没有找到符合条件的结果，不保存文件")
        return
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 按p值排序（p值越大，拟合越好）
    df_results = df_results.sort_values('p_value', ascending=False)
    
    # 显示结果摘要
    print("\n" + "=" * 80)
    print("分析结果摘要")
    print("=" * 80)
    
    # 显示前20只股票
    display_cols = ['ts_code', 'name', 'industry', 'p_value', 'skewness', 'shape_param_a']
    print(df_results[display_cols].head(20).to_string())
    
    # 输出到CSV文件
    if OUTPUT_TO_FILE:
        df_results.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: {OUTPUT_FILENAME}")
        
        # 保存参数配置
        config = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'SKEWNESS_THRESHOLD': SKEWNESS_THRESHOLD,
                'P_VALUE_THRESHOLD': P_VALUE_THRESHOLD,
                'SAMPLE_SIZE_MIN': SAMPLE_SIZE_MIN,
                'STOCK_LIMIT': STOCK_LIMIT,
                'START_DATE': START_DATE,
                'END_DATE': END_DATE,
                'OUTPUT_FILENAME': OUTPUT_FILENAME,
                'PLOT_DISTRIBUTIONS': PLOT_DISTRIBUTIONS
            },
            'total_stocks_analyzed': len(stock_list),
            'skewnormal_stocks_found': len(results),
            'success_rate': len(results) / len(stock_list) if len(stock_list) > 0 else 0
        }
        
        config_filename = 'analysis_config.json'
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"参数配置已保存到: {config_filename}")
    
    return df_results

def print_statistics(df_results):
    """打印统计信息"""
    if len(df_results) == 0:
        return
    
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    
    print(f"符合偏态正态分布的股票数量: {len(df_results)}")
    print(f"平均p值: {df_results['p_value'].mean():.4f}")
    print(f"平均偏度: {df_results['skewness'].mean():.4f}")
    print(f"平均峰度: {df_results['kurtosis'].mean():.4f}")
    print(f"平均形状参数a: {df_results['shape_param_a'].mean():.4f}")
    
    # 形状参数分布
    a_positive = len(df_results[df_results['shape_param_a'] > 0])
    a_negative = len(df_results[df_results['shape_param_a'] < 0])
    a_zeroish = len(df_results[np.abs(df_results['shape_param_a']) < 0.1])
    
    print(f"\n形状参数分布:")
    print(f"  正偏 (a>0): {a_positive} 只 ({a_positive/len(df_results)*100:.1f}%)")
    print(f"  负偏 (a<0): {a_negative} 只 ({a_negative/len(df_results)*100:.1f}%)")
    print(f"  接近对称 (|a|<0.1): {a_zeroish} 只 ({a_zeroish/len(df_results)*100:.1f}%)")
    
    # 行业分布
    print(f"\n行业分布 (前10):")
    industry_counts = df_results['industry'].value_counts().head(10)
    for industry, count in industry_counts.items():
        percentage = count / len(df_results) * 100
        print(f"  {industry}: {count} 只 ({percentage:.1f}%)")
    
    # 市场分布
    print(f"\n市场分布:")
    market_counts = df_results['market'].value_counts()
    for market, count in market_counts.items():
        percentage = count / len(df_results) * 100
        print(f"  {market}: {count} 只 ({percentage:.1f}%)")

def main():
    """主函数"""
    print("=" * 80)
    print("A股筹码分布偏态正态分布分析")
    print("=" * 80)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n参数配置:")
    print(f"  1. 偏态控制参数:")
    print(f"     - 形状参数阈值 (SKEWNESS_THRESHOLD): {SKEWNESS_THRESHOLD}")
    print(f"       (|a| < {SKEWNESS_THRESHOLD} 表示可接受的偏度)")
    print(f"  2. 拟合优度参数:")
    print(f"     - P值阈值 (P_VALUE_THRESHOLD): {P_VALUE_THRESHOLD}")
    print(f"       (p > {P_VALUE_THRESHOLD} 表示接受偏态正态分布假设)")
    print(f"  3. 数据质量参数:")
    print(f"     - 最小样本量 (SAMPLE_SIZE_MIN): {SAMPLE_SIZE_MIN}")
    print(f"  4. 范围控制参数:")
    print(f"     - 股票数量限制: {STOCK_LIMIT if STOCK_LIMIT else '全部A股'}")
    print(f"     - 数据时间范围: {START_DATE} 至 {END_DATE}")
    print(f"  5. 输出控制:")
    print(f"     - 输出文件: {OUTPUT_FILENAME}")
    print(f"     - 绘制分布图: {'是' if PLOT_DISTRIBUTIONS else '否'}")
    print("=" * 80)
    
    # 初始化tushare
    pro = init_tushare()
    
    # 获取股票列表
    stock_list = get_stock_list(pro)
    
    if len(stock_list) == 0:
        print("未获取到股票列表，程序退出")
        return
    
    # 分析股票
    results = analyze_stocks(pro, stock_list)
    
    if results:
        # 保存结果
        df_results = save_results(results, stock_list)
        
        # 打印统计信息
        print_statistics(df_results)
        
        # 提供参数调整建议
        print("\n" + "=" * 80)
        print("参数调整建议")
        print("=" * 80)
        print("如果想获得更多符合的股票，可以:")
        print("  1. 提高 SKEWNESS_THRESHOLD (如从 0.5 提高到 0.8)")
        print("  2. 提高 P_VALUE_THRESHOLD (如从 0.10 提高到 0.20)")
        print("  3. 降低 SAMPLE_SIZE_MIN (如从 50 降低到 30)")
        print("\n如果想获得更严格的筛选，可以:")
        print("  1. 降低 SKEWNESS_THRESHOLD (如从 0.5 降低到 0.3)")
        print("  2. 降低 P_VALUE_THRESHOLD (如从 0.10 降低到 0.05)")
        
    else:
        print("\n未找到符合偏态正态分布的股票")
        print("\n建议调整参数:")
        print("  1. 增加 SKEWNESS_THRESHOLD 以允许更大的偏度")
        print("  2. 增加 P_VALUE_THRESHOLD 以降低检验严格度")
        print("  3. 增加股票数量限制 STOCK_LIMIT")
        print("  4. 延长数据时间范围 (调整 START_DATE)")
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
