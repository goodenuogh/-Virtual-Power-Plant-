import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import weibull_min
import seaborn as sns
from datetime import datetime
import pandas as pd
import warnings
import os
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from dataclasses import dataclass
import matplotlib as mpl

# 过滤字体相关的警告
warnings.filterwarnings("ignore", category=UserWarning)

class WindSolarSampling:
    """
    风光出力概率分布采样分析类
    
    用于生成和分析风速和光照强度的概率分布样本，支持:
    1. 风速的Weibull分布采样
    2. 光照强度的Beta分布采样
    3. 拉丁超立方采样(LHS)和蒙特卡洛采样(MC)的对比
    4. 采样结果的统计分析和可视化
    """
    
    def __init__(self, wind_k=2.0, wind_c=6.0, solar_alpha=2.0, solar_beta=2.0):
        """
        初始化采样参数
        
        参数:
        ----------
        wind_k : float, optional (default=2.0)
            风速Weibull分布形状参数
        wind_c : float, optional (default=6.0)
            风速Weibull分布尺度参数
        solar_alpha : float, optional (default=2.0)
            光照强度Beta分布参数alpha
        solar_beta : float, optional (default=2.0)
            光照强度Beta分布参数beta
        """
        # 设置绘图样式
        sns.set_style("darkgrid")
        
        # 完全禁用Unicode减号
        mpl.rcParams['axes.unicode_minus'] = False
        
        # 使用简单的文本渲染
        mpl.rcParams['text.usetex'] = False
        
        # 使用更通用的字体
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['SimHei', 'Arial', 'DejaVu Sans'],
            'mathtext.default': 'regular',
            'savefig.dpi': 300,
            'figure.dpi': 300,
            'figure.figsize': [10, 6],
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.3
        })
        
        # 初始化参数
        self.wind_k = wind_k
        self.wind_c = wind_c
        self.solar_alpha = solar_alpha
        self.solar_beta = solar_beta
        
        # 创建数据点
        self.x_wind = np.linspace(0, 15, 1000)
        self.x_solar = np.linspace(0, 1, 1000)
        
        # 计算理论分布
        self.wind_pdf = weibull_min.pdf(self.x_wind, wind_k, loc=0, scale=wind_c)
        self.solar_pdf = stats.beta.pdf(self.x_solar, solar_alpha, solar_beta)

    def generate_wind_samples(self, N, method='lhs'):
        """生成风速样本"""
        if method == 'mc':
            # 蒙特卡洛采样
            return weibull_min.rvs(self.wind_k, loc=0, scale=self.wind_c, size=N)
        elif method == 'lhs':
            # 拉丁超立方采样
            # 1. 将[0,1]区间等分为N个子区间
            u = np.linspace(0, 1, N+1)
            # 2. 在每个子区间内随机采样一个点
            points = np.random.uniform(u[:-1], u[1:])
            # 3. 随机打乱顺序
            np.random.shuffle(points)
            # 4. 使用逆变换法获取服从Weibull分布的样本
            return weibull_min.ppf(points, self.wind_k, loc=0, scale=self.wind_c)

    def generate_solar_samples(self, N, method='lhs'):
        """生成光照强度样本"""
        if method == 'lhs':
            prob_intervals = np.linspace(0, 1, N+1)
            samples = []
            for i in range(N):
                y = np.random.uniform(prob_intervals[i], prob_intervals[i+1])
                x = stats.beta.ppf(y, self.solar_alpha, self.solar_beta)
                samples.append(x)
            return np.array(samples)
        else:  # 蒙特卡洛采样
            return stats.beta.rvs(self.solar_alpha, self.solar_beta, size=N)

    def plot_distributions(self, N=1000, save_path='figures/distribution_comparison.png'):
        """
        绘制风速和光照强度的分布对比图，比较蒙特卡洛采样和拉丁超立方采样
        
        参数:
        ----------
        N : int, optional (default=1000)
            采样点数量
        save_path : str, optional
            图表保存路径
            
        说明:
        ----------
        此函数生成四个子图，分别展示:
        1. 风速的蒙特卡洛采样与理论分布对比
        2. 风速的拉丁超立方采样与理论分布对比（包含等概率子区间）
        3. 光照强度的蒙特卡洛采样与理论分布对比
        4. 光照强度的拉丁超立方采样与理论分布对比（包含等概率子区间）
        
        拉丁超立方采样的核心思想是将概率空间划分为N个等概率子区间，
        在每个子区间中随机选取一个样本点，确保样本覆盖整个概率空间。
        """
        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 生成样本
        wind_mc = self.generate_wind_samples(N, method='mc')  # 蒙特卡洛风速样本
        wind_lhs = self.generate_wind_samples(N, method='lhs')  # 拉丁超立方风速样本
        solar_mc = self.generate_solar_samples(N, method='mc')  # 蒙特卡洛光照样本
        solar_lhs = self.generate_solar_samples(N, method='lhs')  # 拉丁超立方光照样本
        
        # 风速 Monte Carlo采样
        axs[0, 0].hist(wind_mc, bins=20, density=True, alpha=0.7, color='skyblue', label='MC采样')
        axs[0, 0].plot(self.x_wind, self.wind_pdf, 'r-', linewidth=2, label='理论分布')
        axs[0, 0].set_title('风速: Monte Carlo采样')
        axs[0, 0].set_xlabel('风速 (m/s)')
        axs[0, 0].set_ylabel('概率密度')
        axs[0, 0].legend()
        
        # 风速 Latin Hypercube采样
        axs[0, 1].hist(wind_lhs, bins=20, density=True, alpha=0.7, color='skyblue', label='LHS采样')
        axs[0, 1].plot(self.x_wind, self.wind_pdf, 'r-', linewidth=2, label='理论分布')
        axs[0, 1].set_title('风速: Latin Hypercube采样')
        axs[0, 1].set_xlabel('风速 (m/s)')
        axs[0, 1].set_ylabel('概率密度')
        
        # 添加LHS子区间分隔线 - 这是关键改进，直观展示LHS的等概率区间
        # 计算Weibull分布的分位数点
        quantiles = np.linspace(0, 1, 11)  # 10个等概率子区间
        wind_quantiles = weibull_min.ppf(quantiles, self.wind_k, loc=0, scale=self.wind_c)
        for q in wind_quantiles:
            if 0 < q < 15:  # 只在合理范围内绘制
                axs[0, 1].axvline(q, color='green', linestyle='--', alpha=0.5)
        
        axs[0, 1].legend()
        
        # 光照强度 Monte Carlo采样
        axs[1, 0].hist(solar_mc, bins=20, density=True, alpha=0.7, color='skyblue', label='MC采样')
        axs[1, 0].plot(self.x_solar, self.solar_pdf, 'r-', linewidth=2, label='理论分布')
        axs[1, 0].set_title('归一化光照强度: Monte Carlo采样')
        axs[1, 0].set_xlabel('归一化光照强度')
        axs[1, 0].set_ylabel('概率密度')
        axs[1, 0].legend()
        
        # 光照强度 Latin Hypercube采样
        axs[1, 1].hist(solar_lhs, bins=20, density=True, alpha=0.7, color='skyblue', label='LHS采样')
        axs[1, 1].plot(self.x_solar, self.solar_pdf, 'r-', linewidth=2, label='理论分布')
        axs[1, 1].set_title('归一化光照强度: Latin Hypercube采样')
        axs[1, 1].set_xlabel('归一化光照强度')
        axs[1, 1].set_ylabel('概率密度')
        
        # 添加LHS子区间分隔线
        # 计算Beta分布的分位数点
        solar_quantiles = stats.beta.ppf(quantiles, self.solar_alpha, self.solar_beta)
        for q in solar_quantiles:
            if 0 < q < 1:  # 只在[0,1]范围内绘制
                axs[1, 1].axvline(q, color='green', linestyle='--', alpha=0.5)
        
        axs[1, 1].legend()
        
        # 添加注释说明LHS子区间
        axs[0, 1].text(0.98, 0.98, '绿色虚线表示等概率子区间',
                      transform=axs[0, 1].transAxes,
                      ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布对比图已保存到: {save_path}")
        plt.close()

    def analyze_convergence(self, sample_sizes=[100, 500, 1000, 5000], 
                           n_trials=100, save_path='figures/convergence_analysis.png'):
        """分析收敛性"""
        wind_mc_errors = []
        wind_lhs_errors = []
        solar_mc_errors = []
        solar_lhs_errors = []
        
        for N in sample_sizes:
            wind_mc_trial_errors = []
            wind_lhs_trial_errors = []
            solar_mc_trial_errors = []
            solar_lhs_trial_errors = []
            
            for _ in range(n_trials):
                # 风速采样误差
                wind_mc = self.generate_wind_samples(N, 'mc')
                wind_lhs = self.generate_wind_samples(N, 'lhs')
                wind_mc_trial_errors.append(np.abs(np.mean(wind_mc) - 
                    weibull_min.mean(self.wind_k, loc=0, scale=self.wind_c)))
                wind_lhs_trial_errors.append(np.abs(np.mean(wind_lhs) - 
                    weibull_min.mean(self.wind_k, loc=0, scale=self.wind_c)))
                
                # 光照强度采样误差
                solar_mc = self.generate_solar_samples(N, 'mc')
                solar_lhs = self.generate_solar_samples(N, 'lhs')
                solar_mc_trial_errors.append(np.abs(np.mean(solar_mc) - 
                    stats.beta.mean(self.solar_alpha, self.solar_beta)))
                solar_lhs_trial_errors.append(np.abs(np.mean(solar_lhs) - 
                    stats.beta.mean(self.solar_alpha, self.solar_beta)))
            
            wind_mc_errors.append(np.mean(wind_mc_trial_errors))
            wind_lhs_errors.append(np.mean(wind_lhs_trial_errors))
            solar_mc_errors.append(np.mean(solar_mc_trial_errors))
            solar_lhs_errors.append(np.mean(solar_lhs_trial_errors))
        
        # 绘制收敛性分析图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(sample_sizes, wind_mc_errors, 'o-', label='MC')
        ax1.plot(sample_sizes, wind_lhs_errors, 's-', label='LHS')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('采样数量')
        ax1.set_ylabel('平均相对误差')
        ax1.set_title('风速采样收敛性分析')
        ax1.legend()
        
        ax2.plot(sample_sizes, solar_mc_errors, 'o-', label='MC')
        ax2.plot(sample_sizes, solar_lhs_errors, 's-', label='LHS')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('采样数量')
        ax2.set_ylabel('平均相对误差')
        ax2.set_title('光照强度采样收敛性分析')
        ax2.legend()
        
        plt.tight_layout()
        
        # 添加保存确认
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存图片
            plt.savefig(save_path)
            print(f"收敛性分析图已保存到: {os.path.abspath(save_path)}")
        except Exception as e:
            print(f"保存图片时出错: {str(e)}")
        finally:
            plt.close()

    def analyze_samples(self, N=1000):
        """
        分析采样结果并生成详细报告
        
        参数:
        ----------
        N : int, optional (default=1000)
            采样数量
        """
        # 生成样本
        wind_mc = self.generate_wind_samples(N, 'mc')
        wind_lhs = self.generate_wind_samples(N, 'lhs')
        solar_mc = self.generate_solar_samples(N, 'mc')
        solar_lhs = self.generate_solar_samples(N, 'lhs')
        
        # 计算理论值
        wind_theoretical_mean = weibull_min.mean(self.wind_k, loc=0, scale=self.wind_c)
        wind_theoretical_var = weibull_min.var(self.wind_k, loc=0, scale=self.wind_c)
        solar_theoretical_mean = stats.beta.mean(self.solar_alpha, self.solar_beta)
        solar_theoretical_var = stats.beta.var(self.solar_alpha, self.solar_beta)
        
        # 计算统计量
        results = {
            'wind': {
                'mc': {
                    'mean': np.mean(wind_mc),
                    'std': np.std(wind_mc),
                    'var': np.var(wind_mc),
                    'ci': stats.t.interval(0.95, len(wind_mc)-1, 
                                         loc=np.mean(wind_mc), 
                                         scale=stats.sem(wind_mc))
                },
                'lhs': {
                    'mean': np.mean(wind_lhs),
                    'std': np.std(wind_lhs),
                    'var': np.var(wind_lhs),
                    'ci': stats.t.interval(0.95, len(wind_lhs)-1, 
                                         loc=np.mean(wind_lhs), 
                                         scale=stats.sem(wind_lhs))
                },
                'theoretical': {
                    'mean': wind_theoretical_mean,
                    'var': wind_theoretical_var
                }
            },
            'solar': {
                'mc': {
                    'mean': np.mean(solar_mc),
                    'std': np.std(solar_mc),
                    'var': np.var(solar_mc),
                    'ci': stats.t.interval(0.95, len(solar_mc)-1, 
                                         loc=np.mean(solar_mc), 
                                         scale=stats.sem(solar_mc))
                },
                'lhs': {
                    'mean': np.mean(solar_lhs),
                    'std': np.std(solar_lhs),
                    'var': np.var(solar_lhs),
                    'ci': stats.t.interval(0.95, len(solar_lhs)-1, 
                                         loc=np.mean(solar_lhs), 
                                         scale=stats.sem(solar_lhs))
                },
                'theoretical': {
                    'mean': solar_theoretical_mean,
                    'var': solar_theoretical_var
                }
            }
        }
        
        return results

    def save_analysis_report(self, results, filename='sampling_results.txt'):
        """
        保存详细的分析报告
        
        参数:
        ----------
        results : dict
            分析结果字典
        filename : str, optional (default='sampling_results.txt')
            输出文件名
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("风光出力概率分布采样分析报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"分析时间: {datetime.now()}\n")
            f.write(f"采样参数:\n")
            f.write(f"  风速Weibull分布: k={self.wind_k}, c={self.wind_c}\n")
            f.write(f"  光照Beta分布: α={self.solar_alpha}, β={self.solar_beta}\n\n")
            
            # 风速分析结果
            f.write("1. 风速分析结果\n")
            f.write("-" * 30 + "\n")
            f.write("理论值:\n")
            f.write(f"  均值: {results['wind']['theoretical']['mean']:.4f}\n")
            f.write(f"  方差: {results['wind']['theoretical']['var']:.4f}\n\n")
            
            f.write("Monte Carlo采样结果:\n")
            f.write(f"  均值: {results['wind']['mc']['mean']:.4f}\n")
            f.write(f"  标准差: {results['wind']['mc']['std']:.4f}\n")
            f.write(f"  方差: {results['wind']['mc']['var']:.4f}\n")
            f.write(f"  95%置信区间: [{results['wind']['mc']['ci'][0]:.4f}, {results['wind']['mc']['ci'][1]:.4f}]\n\n")
            
            f.write("Latin Hypercube采样结果:\n")
            f.write(f"  均值: {results['wind']['lhs']['mean']:.4f}\n")
            f.write(f"  标准差: {results['wind']['lhs']['std']:.4f}\n")
            f.write(f"  方差: {results['wind']['lhs']['var']:.4f}\n")
            f.write(f"  95%置信区间: [{results['wind']['lhs']['ci'][0]:.4f}, {results['wind']['lhs']['ci'][1]:.4f}]\n\n")
            
            # 光照强度分析结果
            f.write("2. 光照强度分析结果\n")
            f.write("-" * 30 + "\n")
            f.write("理论值:\n")
            f.write(f"  均值: {results['solar']['theoretical']['mean']:.4f}\n")
            f.write(f"  方差: {results['solar']['theoretical']['var']:.4f}\n\n")
            
            f.write("Monte Carlo采样结果:\n")
            f.write(f"  均值: {results['solar']['mc']['mean']:.4f}\n")
            f.write(f"  标准差: {results['solar']['mc']['std']:.4f}\n")
            f.write(f"  方差: {results['solar']['mc']['var']:.4f}\n")
            f.write(f"  95%置信区间: [{results['solar']['mc']['ci'][0]:.4f}, {results['solar']['mc']['ci'][1]:.4f}]\n\n")
            
            f.write("Latin Hypercube采样结果:\n")
            f.write(f"  均值: {results['solar']['lhs']['mean']:.4f}\n")
            f.write(f"  标准差: {results['solar']['lhs']['std']:.4f}\n")
            f.write(f"  方差: {results['solar']['lhs']['var']:.4f}\n")
            f.write(f"  95%置信区间: [{results['solar']['lhs']['ci'][0]:.4f}, {results['solar']['lhs']['ci'][1]:.4f}]\n")

    def analyze_real_data(self, processed_data, save_path='figures/real_data_analysis.png'):
        """
        分析实际NASA POWER数据，生成全面的数据分析图表
        
        参数:
        ----------
        processed_data : dict
            处理后的NASA POWER数据，包含以下键:
            - wind: 包含'raw'(原始风速)和'power'(风电出力)的字典
            - solar: 包含'raw'(原始太阳辐射)和'power'(光伏出力)的字典
            - temp: 温度数据
            - days: 天数
            - time_periods: 每天的时间段数量
            
        save_path : str
            保存图表的路径
            
        返回:
        ----------
        distribution_params : dict
            拟合的分布参数，可用于后续模拟
            
        说明:
        ----------
        此函数使用NASA POWER的真实气象数据生成六个子图:
        1. 四川地区风速分布及Weibull拟合
        2. 四川地区太阳辐射分布
        3. 风电出力时间序列
        4. 光伏出力时间序列
        5. 风电典型日出力曲线
        6. 光伏典型日出力曲线
        
        同时计算并输出关键统计指标，包括均值、标准差、容量因数和相关性。
        """
        # 创建图形
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # 提取数据 - 这里使用的是NASA POWER的真实数据
        wind_raw = processed_data['wind']['raw']  # 原始风速数据
        wind_power = processed_data['wind']['power']  # 计算的风电出力
        solar_raw = processed_data['solar']['raw']  # 原始太阳辐射数据
        solar_power = processed_data['solar']['power']  # 计算的光伏出力
        temp = processed_data['temp']  # 温度数据
        
        # 计算每天的平均值
        wind_daily_avg = np.mean(wind_raw, axis=1)  # 日均风速
        solar_daily_avg = np.mean(solar_raw, axis=1)  # 日均太阳辐射
        wind_power_daily_avg = np.mean(wind_power, axis=1)  # 日均风电出力
        solar_power_daily_avg = np.mean(solar_power, axis=1)  # 日均光伏出力
        
        # 1. 绘制风速分布 - 使用真实数据
        sns.histplot(wind_raw.flatten(), kde=True, ax=axs[0, 0], color='skyblue')
        
        # 拟合Weibull分布 - 使用真实风速数据拟合Weibull分布参数
        wind_data = wind_raw.flatten()
        wind_data = wind_data[wind_data > 0]  # 移除零值
        try:
            # 拟合Weibull分布参数
            shape, loc, scale = weibull_min.fit(wind_data, loc=0)
            x = np.linspace(0, np.max(wind_data), 100)
            pdf = weibull_min.pdf(x, shape, loc=loc, scale=scale)
            axs[0, 0].plot(x, pdf, 'r-', linewidth=2, 
                          label=f'Weibull拟合 (k={shape:.2f}, c={scale:.2f})')
            axs[0, 0].legend()
        except:
            print("Weibull分布拟合失败")
        
        axs[0, 0].set_title('四川地区风速分布 (真实数据)')
        axs[0, 0].set_xlabel('风速 (m/s)')
        axs[0, 0].set_ylabel('频率')
        
        # 2. 绘制太阳辐射分布 - 使用真实数据
        sns.histplot(solar_raw.flatten(), kde=True, ax=axs[0, 1], color='orange')
        axs[0, 1].set_title('四川地区太阳辐射分布 (真实数据)')
        axs[0, 1].set_xlabel('太阳辐射 (W/m²)')
        axs[0, 1].set_ylabel('频率')
        
        # 3. 绘制风电出力时间序列
        days = np.arange(len(wind_power_daily_avg))
        axs[1, 0].plot(days, wind_power_daily_avg, 'b-', label='日均风电出力')
        axs[1, 0].set_title('风电出力时间序列')
        axs[1, 0].set_xlabel('天数')
        axs[1, 0].set_ylabel('归一化出力')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # 4. 绘制光伏出力时间序列
        axs[1, 1].plot(days, solar_power_daily_avg, 'r-', label='日均光伏出力')
        axs[1, 1].set_title('光伏出力时间序列')
        axs[1, 1].set_xlabel('天数')
        axs[1, 1].set_ylabel('归一化出力')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        # 5. 绘制典型日出力曲线 - 风电
        # 选择一个典型日
        typical_day_idx = np.argmin(np.abs(wind_power_daily_avg - np.mean(wind_power_daily_avg)))
        hours = np.arange(24)
        
        axs[2, 0].plot(hours, wind_power[typical_day_idx], 'b-o', label='风电典型日出力')
        axs[2, 0].set_title(f'风电典型日出力曲线 (第{typical_day_idx+1}天)')
        axs[2, 0].set_xlabel('小时')
        axs[2, 0].set_ylabel('归一化出力')
        axs[2, 0].set_xticks(np.arange(0, 24, 2))
        axs[2, 0].legend()
        axs[2, 0].grid(True)
        
        # 6. 绘制典型日出力曲线 - 光伏
        typical_day_idx = np.argmin(np.abs(solar_power_daily_avg - np.mean(solar_power_daily_avg)))
        
        axs[2, 1].plot(hours, solar_power[typical_day_idx], 'r-o', label='光伏典型日出力')
        axs[2, 1].set_title(f'光伏典型日出力曲线 (第{typical_day_idx+1}天)')
        axs[2, 1].set_xlabel('小时')
        axs[2, 1].set_ylabel('归一化出力')
        axs[2, 1].set_xticks(np.arange(0, 24, 2))
        axs[2, 1].legend()
        axs[2, 1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"实际数据分析图表已保存到: {save_path}")
        plt.close()
        
        # 计算统计指标
        print("\n实际数据统计分析:")
        print(f"风速: 均值={np.mean(wind_raw):.2f}m/s, 标准差={np.std(wind_raw):.2f}m/s")
        print(f"太阳辐射: 均值={np.mean(solar_raw):.2f}W/m², 标准差={np.std(solar_raw):.2f}W/m²")
        print(f"风电出力: 均值={np.mean(wind_power):.4f}, 容量因数={np.mean(wind_power):.4f}")
        print(f"光伏出力: 均值={np.mean(solar_power):.4f}, 容量因数={np.mean(solar_power):.4f}")
        
        # 计算相关性
        corr_wind_solar = np.corrcoef(wind_daily_avg, solar_daily_avg)[0, 1]
        corr_wind_power_solar_power = np.corrcoef(wind_power_daily_avg, solar_power_daily_avg)[0, 1]
        print(f"风速与太阳辐射相关性: {corr_wind_solar:.4f}")
        print(f"风电出力与光伏出力相关性: {corr_wind_power_solar_power:.4f}")
        
        # 返回拟合的分布参数，可用于后续模拟
        distribution_params = {}
        try:
            distribution_params['wind'] = {'k': shape, 'c': scale}
        except:
            distribution_params['wind'] = {'k': self.wind_k, 'c': self.wind_c}
        
        return distribution_params

def load_nasa_power_data(file_path):
    """
    加载NASA POWER数据
    
    参数:
    ----------
    file_path : str
        NASA POWER数据CSV文件路径，即'POWER_Parameter_Manager.csv'
        
    返回:
    ----------
    wind_data : numpy.ndarray
        风速数据 (m/s)，从WS2M列提取
    solar_data : numpy.ndarray
        太阳辐射数据 (W/m²)，从ALLSKY_SFC_SW_DWN列提取
    temp_data : numpy.ndarray
        温度数据 (°C)，从T2M列提取
    """
    # 读取CSV文件，跳过元数据头部
    try:
        # 尝试自动检测元数据行数
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('YEAR'):
                    header_rows = i
                    break
        
        # 读取数据
        data = pd.read_csv(file_path, skiprows=header_rows)
        print(f"成功加载NASA POWER数据，共{len(data)}行")
        
        # 提取需要的列 - 这里使用的是真实的NASA POWER数据
        wind_data = data['WS2M'].values  # 2米风速
        solar_data = data['ALLSKY_SFC_SW_DWN'].values  # 太阳辐射
        temp_data = data['T2M'].values  # 温度
        
        # 打印数据统计信息
        print(f"风速数据: 均值={np.mean(wind_data):.2f}m/s, 最大值={np.max(wind_data):.2f}m/s")
        print(f"太阳辐射数据: 均值={np.mean(solar_data):.2f}W/m², 最大值={np.max(solar_data):.2f}W/m²")
        print(f"温度数据: 均值={np.mean(temp_data):.2f}°C, 最大值={np.max(temp_data):.2f}°C")
        
        return wind_data, solar_data, temp_data
    
    except Exception as e:
        print(f"加载NASA POWER数据时出错: {str(e)}")
        return None, None, None

def process_power_data(wind_data, solar_data, temp_data):
    """
    处理NASA POWER数据，计算风电和光伏出力
    
    参数:
    ----------
    wind_data : numpy.ndarray
        风速数据 (m/s)
    solar_data : numpy.ndarray
        太阳辐射数据 (W/m²)
    temp_data : numpy.ndarray
        温度数据 (°C)
        
    返回:
    ----------
    processed_data : dict
        处理后的数据字典
    """
    # 重塑数据为(天数, 每天时间段)的形式
    # 假设数据是按小时记录的，每天24小时
    time_periods = 24
    days = len(wind_data) // time_periods
    
    # 将数据重塑为(天数, 时间段)形状
    wind_reshaped = wind_data[:days*time_periods].reshape(days, time_periods)
    solar_reshaped = solar_data[:days*time_periods].reshape(days, time_periods)
    temp_reshaped = temp_data[:days*time_periods].reshape(days, time_periods)
    
    # 风速转换为风电出力
    # 使用简化的风电机组功率曲线: P = 0 (v < v_ci), P = P_r*(v-v_ci)/(v_r-v_ci) (v_ci <= v <= v_r), P = P_r (v_r <= v <= v_co), P = 0 (v > v_co)
    v_ci = 3.0  # 切入风速
    v_r = 12.0  # 额定风速
    v_co = 25.0  # 切出风速
    P_r = 1.0  # 额定功率（归一化为1）
    
    wind_power = np.zeros_like(wind_reshaped)
    for i in range(days):
        for j in range(time_periods):
            v = wind_reshaped[i, j]
            if v < v_ci or v > v_co:
                wind_power[i, j] = 0
            elif v_ci <= v <= v_r:
                wind_power[i, j] = P_r * (v - v_ci) / (v_r - v_ci)
            else:  # v_r <= v <= v_co
                wind_power[i, j] = P_r
    
    # 太阳辐射转换为光伏出力
    # 使用简化的光伏功率模型: P = η * G * (1 - 0.005*(T-25))
    eta = 0.15  # 光电转换效率
    G_stc = 1000.0  # 标准测试条件下的辐照度 (W/m²)
    
    pv_power = np.zeros_like(solar_reshaped)
    for i in range(days):
        for j in range(time_periods):
            G = solar_reshaped[i, j]
            T = temp_reshaped[i, j]
            pv_power[i, j] = eta * (G / G_stc) * (1 - 0.005 * (T - 25))
            pv_power[i, j] = max(0, min(1, pv_power[i, j]))  # 限制在[0,1]范围内
    
    # 创建处理后的数据字典
    processed_data = {
        'wind': {
            'raw': wind_reshaped,
            'power': wind_power
        },
        'solar': {
            'raw': solar_reshaped,
            'power': pv_power
        },
        'temp': temp_reshaped,
        'days': days,
        'time_periods': time_periods
    }
    
    print(f"数据处理完成: {days}天, 每天{time_periods}个时间段")
    return processed_data

def main():
    """主程序"""
    # 创建采样分析器实例
    sampler = WindSolarSampling(wind_k=2.0, wind_c=6.0, 
                               solar_alpha=2.0, solar_beta=2.0)
    
    # 生成分布对比图
    sampler.plot_distributions(N=1000, save_path='figures/distribution_comparison.png')
    
    # 进行收敛性分析
    sampler.analyze_convergence(sample_sizes=[100, 500, 1000, 5000], 
                              n_trials=100, save_path='figures/convergence_analysis.png')
    
    # 生成详细分析报告
    results = sampler.analyze_samples(N=1000)
    sampler.save_analysis_report(results, 'sampling_results.txt')

if __name__ == '__main__':
    main()