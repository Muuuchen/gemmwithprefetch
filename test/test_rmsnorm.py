import time
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import json
from datetime import datetime
import os

import cutlass_gemm_with_prefetch

class CutlassPerformanceTester:
    """CUTLASS算子性能测试框架"""
    
    def __init__(self, device='cuda:0', warmup_iters=5, measure_iters=20):
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.results = []
        
        # 创建结果保存目录
        self.result_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.result_dir, exist_ok=True)
        
    def _create_tensors(self, M, K, N, dtype=torch.float8_e4m3fn):
        """创建测试用的张量"""
        A = torch.normal(0, 1, size=(M, K)).to(device=self.device).to(dtype=torch.float8_e4m3fn)
        B = torch.normal(0, 1, size=(K, N)).to(device=self.device).to(dtype=torch.float8_e5m2)
        C = torch.empty((M, N), device=self.device, dtype=torch.float8_e4m3fn)
        D = torch.normal(0, 1, size=(M, N)).to(device=self.device).to(dtype=torch.float8_e4m3fn)
        E = torch.empty((M, N), device=self.device, dtype=torch.float8_e4m3fn)
        F = torch.normal(0, 1, size=(M, N)).to(device=self.device).to(dtype=torch.float8_e4m3fn)
        
        return A, B, C, D, E, F
    
    def _warmup(self, func, *args, **kwargs):
        """预热GPU"""
        for _ in range(self.warmup_iters):
            func(*args, **kwargs)
        torch.cuda.synchronize()
    
    def _measure_time(self, func, *args, **kwargs):
        """精确测量函数执行时间"""
        # 创建CUDA事件用于精确计时
        start_events = []
        end_events = []
        
        # 多次测量
        times = []
        for _ in range(self.measure_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start.record()
            func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            
            times.append(start.elapsed_time(end))
        
        # 计算统计信息
        times = np.array(times)
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'raw_times': times.tolist()
        }
    
    def test_rmsnorm_prefetch(self, M, N, prefetch_ratios):
        """测试rmsnorm算子的prefetch_ratio参数影响"""
        print(f"\n{'='*60}")
        print(f"Testing RMSNorm with different prefetch_ratio values")
        print(f"Matrix size: M={M}, N={N}")
        print(f"{'='*60}")
        
        # 创建测试数据
        _, _, _, D, E, F = self._create_tensors(M, M, N)
        
        results = []
        
        for prefetch_ratio in prefetch_ratios:
            print(f"\nTesting prefetch_ratio={prefetch_ratio:.2f}")
            
            # 预热
            self._warmup(cutlass_gemm_with_prefetch.rmsnorm, E, D, F, prefetch_ratio)
            
            # 测量性能
            timing = self._measure_time(
                cutlass_gemm_with_prefetch.rmsnorm, 
                E, D, F, prefetch_ratio
            )
            
            result = {
                'op': 'rmsnorm',
                'M': M,
                'N': N,
                'prefetch_ratio': prefetch_ratio,
                **timing
            }
            
            results.append(result)
            print(f"  Mean time: {timing['mean']:.3f} ms (±{timing['std']:.3f})")
            
        return pd.DataFrame(results)
    
    def test_mm_overlap_prefetch(self, M, K, N, overlap_values, prefetch_values):
        """测试mm算子的overlap和prefetch参数影响"""
        print(f"\n{'='*60}")
        print(f"Testing MM with different overlap and prefetch values")
        print(f"Matrix size: M={M}, K={K}, N={N}")
        print(f"{'='*60}")
        
        # 创建测试数据
        A, B, C, D, _, F = self._create_tensors(M, K, N)
        
        results = []
        
        # 测试所有参数组合
        for overlap, prefetch in product(overlap_values, prefetch_values):
            print(f"\nTesting overlap={overlap:.2f}, prefetch={prefetch:.2f}")
            
            # 预热
            self._warmup(cutlass_gemm_with_prefetch.mm, A, B, C, F, overlap, prefetch)
            
            # 测量性能
            timing = self._measure_time(
                cutlass_gemm_with_prefetch.mm,
                A, B, C, F, overlap, prefetch
            )
            
            result = {
                'op': 'mm',
                'M': M,
                'K': K,
                'N': N,
                'overlap': overlap,
                'prefetch': prefetch,
                **timing
            }
            
            results.append(result)
            print(f"  Mean time: {timing['mean']:.3f} ms (±{timing['std']:.3f})")
            
        return pd.DataFrame(results)
    
    def test_combined_pipeline(self, M, K, N, prefetch_ratios, overlap_values, prefetch_values):
        """测试完整的流水线：mm -> rmsnorm -> mm"""
        print(f"\n{'='*60}")
        print(f"Testing Combined Pipeline: MM -> RMSNorm -> MM")
        print(f"Matrix size: M={M}, K={K}, N={N}")
        print(f"{'='*60}")
        
        # 创建测试数据
        A, B, C, D, E, F = self._create_tensors(M, K, N)
        
        results = []
        
        # 测试不同参数组合
        for prefetch_ratio, overlap, prefetch in product(prefetch_ratios, overlap_values, prefetch_values):
            print(f"\nTesting prefetch_ratio={prefetch_ratio:.2f}, overlap={overlap:.2f}, prefetch={prefetch:.2f}")
            
            def pipeline():
                C_temp = cutlass_gemm_with_prefetch.mm(A, B, C, D, 0.7, -1.0)
                E_temp = cutlass_gemm_with_prefetch.rmsnorm(E, C_temp, F, prefetch_ratio)
                C_out = cutlass_gemm_with_prefetch.mm(A, B, C_temp, D, overlap, prefetch)
                return C_out
            
            # 预热
            self._warmup(pipeline)
            
            # 测量性能
            timing = self._measure_time(pipeline)
            
            result = {
                'op': 'pipeline',
                'M': M,
                'K': K,
                'N': N,
                'prefetch_ratio': prefetch_ratio,
                'overlap': overlap,
                'prefetch': prefetch,
                **timing
            }
            
            results.append(result)
            print(f"  Mean time: {timing['mean']:.3f} ms (±{timing['std']:.3f})")
            
        return pd.DataFrame(results)
    
    def visualize_results(self, df, op_type):
        """可视化测试结果"""
        if op_type == 'rmsnorm':
            self._plot_rmsnorm_results(df)
        elif op_type == 'mm':
            self._plot_mm_results(df)
        elif op_type == 'pipeline':
            self._plot_pipeline_results(df)
    
    def _plot_rmsnorm_results(self, df):
        """绘制RMSNorm结果"""
        plt.figure(figsize=(10, 6))
        
        # 绘制平均时间和误差条
        plt.errorbar(df['prefetch_ratio'], df['mean'], yerr=df['std'], 
                    marker='o', capsize=5, capthick=2, markersize=8)
        
        plt.xlabel('Prefetch Ratio')
        plt.ylabel('Execution Time (ms)')
        plt.title('RMSNorm Performance vs Prefetch Ratio')
        plt.grid(True, alpha=0.3)
        
        # 找出最优参数
        best_idx = df['mean'].idxmin()
        best_ratio = df.loc[best_idx, 'prefetch_ratio']
        best_time = df.loc[best_idx, 'mean']
        plt.axvline(x=best_ratio, color='r', linestyle='--', alpha=0.5)
        plt.text(best_ratio, best_time, f'Best: {best_ratio:.2f}\n{best_time:.3f}ms', 
                ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/rmsnorm_performance.png', dpi=300)
        plt.close()
    
    def _plot_mm_results(self, df):
        """绘制MM结果"""
        # 创建热力图
        pivot = df.pivot(index='overlap', columns='prefetch', values='mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis_r', 
                   cbar_kws={'label': 'Execution Time (ms)'})
        plt.title('MM Performance Heatmap')
        plt.xlabel('Prefetch')
        plt.ylabel('Overlap')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/mm_performance_heatmap.png', dpi=300)
        plt.close()
        
        # 3D表面图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X = df['overlap'].values
        Y = df['prefetch'].values
        Z = df['mean'].values
        
        # 创建网格
        xi = np.linspace(X.min(), X.max(), 100)
        yi = np.linspace(Y.min(), Y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # 插值
        from scipy.interpolate import griddata
        Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')
        
        surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.8)
        ax.scatter(X, Y, Z, c='r', s=50)
        
        ax.set_xlabel('Overlap')
        ax.set_ylabel('Prefetch')
        ax.set_zlabel('Execution Time (ms)')
        ax.set_title('MM Performance Surface')
        
        fig.colorbar(surf)
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/mm_performance_3d.png', dpi=300)
        plt.close()
    
    def save_results(self, df, filename):
        """保存测试结果"""
        # 保存CSV
        df.to_csv(f'{self.result_dir}/{filename}.csv', index=False)
        
        # 保存JSON（包含原始数据）
        df.to_json(f'{self.result_dir}/{filename}.json', orient='records', indent=2)
        
        # 打印统计摘要
        print(f"\n{'='*60}")
        print(f"Summary for {filename}")
        print(f"{'='*60}")
        print(df[['mean', 'std', 'min', 'max']].describe())
        
        # 找出最优参数
        best_idx = df['mean'].idxmin()
        print(f"\nBest configuration:")
        print(df.loc[best_idx])

def main():
    """主测试函数"""
    # 创建测试器
    tester = CutlassPerformanceTester(warmup_iters=10, measure_iters=50)
    
    # 测试参数设置
    M = K = N = 512
    
    # RMSNorm参数
    prefetch_ratios = np.linspace(0.0, 1.0, 11)  # 0.0到1.0，步长0.1
    
    # MM参数
    overlap_values = np.linspace(0.0, 1.0, 6)    # 0.0到1.0，步长0.2
    prefetch_values = [-1.0, 0.0, 0.2, 0.5, 0.8, 1.0]
    
    # 1. 测试RMSNorm
    rmsnorm_results = tester.test_rmsnorm_prefetch(M, N, prefetch_ratios)
    tester.save_results(rmsnorm_results, 'rmsnorm_results')
    tester.visualize_results(rmsnorm_results, 'rmsnorm')
    
    # 2. 测试MM
    mm_results = tester.test_mm_overlap_prefetch(M, K, N, overlap_values, prefetch_values)
    tester.save_results(mm_results, 'mm_results')
    tester.visualize_results(mm_results, 'mm')
    
    # 3. 测试完整流水线（可选，参数组合较多时可能耗时较长）
    # pipeline_results = tester.test_combined_pipeline(
    #     M, K, N, 
    #     [0.3, 0.5, 0.7],  # 选择几个代表性的值
    #     [0.5, 0.7, 0.9], 
    #     [-1.0, 0.5, 1.0]
    # )
    # tester.save_results(pipeline_results, 'pipeline_results')
    # tester.visualize_results(pipeline_results, 'pipeline')
    
    print(f"\nAll results saved to: {tester.result_dir}")

if __name__ == "__main__":
    main()