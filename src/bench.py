import torch
import math
import numpy as np
import pybind11
import cutlass_gemm_with_prefetch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class cutlassGemmWithPrefetchBenchmark:
    def __init__(self):
        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
        
    def create_test_matrices(self, M, K, N):
        """创建测试矩阵"""
        A = torch.normal(0, 1, size=(M, K)).to(device=self.device).to(dtype=torch.float8_e4m3fn)
        B = torch.normal(0, 1, size=(K, N)).to(device=self.device).to(dtype=torch.float8_e5m2)
        C = torch.empty((M, N), device=self.device, dtype=torch.float8_e4m3fn)
        D = torch.normal(0, 1, size=(M, N)).to(device=self.device).to(dtype=torch.float8_e4m3fn)
        return A, B, C, D
    
    def warmup(self, A, B, C, D, overlap_ratio, prefetch_ratio, warmup_iterations=10):
        """GPU预热"""
        for _ in range(warmup_iterations):
            _ = cutlass_gemm_with_prefetch.mm(A, B, C, D, overlap_ratio, prefetch_ratio)
        torch.cuda.synchronize()
        
    def measure_single_run(self, A, B, C, D, overlap_ratio, prefetch_ratio):
        """测量单次运行时间"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = cutlass_gemm_with_prefetch.mm(A, B, C, D, overlap_ratio, prefetch_ratio)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        
        return elapsed_time, result

    def benchmark_single_size(self, M, K, N, overlap_ratio=0.5, prefetch_ratio=0.5, iterations=50, warmup_iterations=10):
        """对单个矩阵大小进行基准测试"""
        # 创建测试矩阵
        A, B, C, D = self.create_test_matrices(M, K, N)
        
        # 预热
        self.warmup(A, B, C, D, overlap_ratio, prefetch_ratio, warmup_iterations)
        
        # 性能测试
        times = []
        for i in range(iterations):
            elapsed_time, _ = self.measure_single_run(A, B, C, D, overlap_ratio, prefetch_ratio)
            times.append(elapsed_time)
        
        # 统计分析
        times = np.array(times)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        std_time = np.std(times)
        median_time = np.median(times)
        
        # 计算GFLOPS
        flops = 2 * M * N * K
        avg_gflops = (flops / (avg_time / 1000.0)) / 1e9
        peak_gflops = (flops / (min_time / 1000.0)) / 1e9
        
        result = {
            'matrix_size': (M, K, N),
            'overlap_ratio': overlap_ratio,
            'prefetch_ratio': prefetch_ratio,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'std_time_ms': std_time,
            'median_time_ms': median_time,
            'avg_gflops': avg_gflops,
            'peak_gflops': peak_gflops,
            'flops': flops,
            'all_times': times.tolist(),
            'success': True
        }
        
        # 清理内存
        del A, B, C, D
        torch.cuda.empty_cache()
        
        return result
    
    def benchmark_multiple_sizes(self, sizes, ratios, iterations=50):
        """批量测试"""
        results = []
        total_tests = len(sizes) * len(ratios)
        current_test = 0
        
        print(f"开始批量测试: {total_tests} 个测试")
        
        for M, K, N in sizes:
            for overlap_ratio, prefetch_ratio in ratios:
                current_test += 1
                print(f"[{current_test}/{total_tests}] {M}x{K}x{N}, o={overlap_ratio}, p={prefetch_ratio}", end="")
                
                try:
                    result = self.benchmark_single_size(M, K, N, overlap_ratio, prefetch_ratio, iterations)
                    results.append(result)
                    print(f" ✓ {result['avg_gflops']:.1f} GFLOPS")
                    
                except Exception as e:
                    print(f" ✗ Error: {str(e)}")
                    results.append({
                        'matrix_size': (M, K, N),
                        'overlap_ratio': overlap_ratio,
                        'prefetch_ratio': prefetch_ratio,
                        'error': str(e),
                        'success': False
                    })
        
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n测试完成! 成功 {successful}/{total_tests}")
        
        return results
    
    def create_comprehensive_analysis(self, results):
        """创建全面的分析报告和可视化"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            print("没有成功的测试结果")
            return
        
        # 创建DataFrame便于分析
        df_data = []
        for r in successful_results:
            M, K, N = r['matrix_size']
            df_data.append({
                'M': M, 'K': K, 'N': N,
                'shape_str': f"{M}x{K}x{N}",
                'overlap_ratio': r['overlap_ratio'],
                'prefetch_ratio': r['prefetch_ratio'],
                'ratio_str': f"o={r['overlap_ratio']}, p={r['prefetch_ratio']}",
                'avg_gflops': r['avg_gflops'],
                'peak_gflops': r['peak_gflops'],
                'avg_time_ms': r['avg_time_ms'],
                'std_time_ms': r['std_time_ms'],
                'flops': r['flops']
            })
        
        df = pd.DataFrame(df_data)
        
        # 生成所有分析图表
        self.plot_overview_dashboard(df)
        self.plot_shape_analysis(df)
        self.plot_ratio_analysis(df)
        self.plot_performance_trends(df)
        self.plot_detailed_heatmaps(df)
        self.plot_efficiency_analysis(df)
        
        return df
    
    def plot_overview_dashboard(self, df):
        """总览仪表盘"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 整体性能分布
        plt.subplot(2, 4, 1)
        plt.hist(df['avg_gflops'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(df['avg_gflops'].mean(), color='red', linestyle='--', label=f'Mean: {df["avg_gflops"].mean():.1f}')
        plt.xlabel('Average GFLOPS')
        plt.ylabel('Frequency')
        plt.title('性能分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 最佳配置Top10
        plt.subplot(2, 4, 2)
        top_configs = df.nlargest(10, 'avg_gflops')
        plt.barh(range(len(top_configs)), top_configs['avg_gflops'], color='lightgreen')
        plt.yticks(range(len(top_configs)), [f"{row['shape_str']}\n{row['ratio_str']}" for _, row in top_configs.iterrows()])
        plt.xlabel('GFLOPS')
        plt.title('Top 10 配置')
        plt.grid(True, alpha=0.3)
        
        # 3. Shape vs 平均性能
        plt.subplot(2, 4, 3)
        shape_perf = df.groupby('shape_str')['avg_gflops'].mean().sort_values(ascending=False)
        plt.bar(range(len(shape_perf)), shape_perf.values, color='orange', alpha=0.7)
        plt.xticks(range(len(shape_perf)), shape_perf.index, rotation=45)
        plt.ylabel('Average GFLOPS')
        plt.title('各Shape平均性能')
        plt.grid(True, alpha=0.3)
        
        # 4. Ratio vs 平均性能
        plt.subplot(2, 4, 4)
        ratio_perf = df.groupby('ratio_str')['avg_gflops'].mean().sort_values(ascending=False)
        plt.bar(range(len(ratio_perf)), ratio_perf.values, color='lightcoral', alpha=0.7)
        plt.xticks(range(len(ratio_perf)), ratio_perf.index, rotation=45)
        plt.ylabel('Average GFLOPS')
        plt.title('各Ratio平均性能')
        plt.grid(True, alpha=0.3)
        
        # 5. 性能 vs 矩阵大小散点图
        plt.subplot(2, 4, 5)
        sizes = df['M'] * df['K'] * df['N']
        plt.scatter(sizes, df['avg_gflops'], alpha=0.6, c=df['overlap_ratio'], cmap='viridis')
        plt.xlabel('Matrix Size (M*K*N)')
        plt.ylabel('GFLOPS')
        plt.title('性能 vs 矩阵大小')
        plt.colorbar(label='Overlap Ratio')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # 6. Overlap vs Prefetch 效果
        plt.subplot(2, 4, 6)
        pivot = df.groupby(['overlap_ratio', 'prefetch_ratio'])['avg_gflops'].mean().reset_index()
        plt.scatter(pivot['overlap_ratio'], pivot['prefetch_ratio'], 
                   s=pivot['avg_gflops']/5, c=pivot['avg_gflops'], 
                   cmap='plasma', alpha=0.7)
        plt.xlabel('Overlap Ratio')
        plt.ylabel('Prefetch Ratio')
        plt.title('Ratio组合效果 (大小=性能)')
        plt.colorbar(label='GFLOPS')
        plt.grid(True, alpha=0.3)
        
        # 7. 性能稳定性 (CV)
        plt.subplot(2, 4, 7)
        df['cv'] = df['std_time_ms'] / df['avg_time_ms'] * 100
        plt.scatter(df['avg_gflops'], df['cv'], alpha=0.6, c=df['prefetch_ratio'], cmap='coolwarm')
        plt.xlabel('Average GFLOPS')
        plt.ylabel('变异系数 (%)')
        plt.title('性能 vs 稳定性')
        plt.colorbar(label='Prefetch Ratio')
        plt.grid(True, alpha=0.3)
        
        # 8. 效率分析
        plt.subplot(2, 4, 8)
        theoretical_peak = 1000  # 假设的理论峰值GFLOPS
        df['efficiency'] = df['avg_gflops'] / theoretical_peak * 100
        plt.hist(df['efficiency'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Efficiency (%)')
        plt.ylabel('Frequency')
        plt.title('计算效率分布')
        plt.axvline(df['efficiency'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["efficiency"].mean():.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('CUTLASS GEMM 性能分析总览仪表盘', fontsize=16, y=0.98)
        plt.show()
    
    def plot_shape_analysis(self, df):
        """详细的Shape分析"""
        shapes = df['shape_str'].unique()
        n_shapes = len(shapes)
        
        fig, axes = plt.subplots(2, (n_shapes + 1) // 2, figsize=(6 * ((n_shapes + 1) // 2), 10))
        if n_shapes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, shape in enumerate(shapes):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            shape_data = df[df['shape_str'] == shape].copy()
            shape_data = shape_data.sort_values('avg_gflops', ascending=False)
            
            # 绘制性能柱状图
            bars = ax.bar(range(len(shape_data)), shape_data['avg_gflops'], 
                         color=plt.cm.Set3(np.linspace(0, 1, len(shape_data))), alpha=0.8)
            
            # 标注最佳配置
            bars[0].set_color('red')
            bars[0].set_alpha(1.0)
            
            # 设置标签
            labels = [f"o={row['overlap_ratio']}\np={row['prefetch_ratio']}" 
                     for _, row in shape_data.iterrows()]
            ax.set_xticks(range(len(shape_data)))
            ax.set_xticklabels(labels, rotation=45)
            
            # 计算性能提升
            best_perf = shape_data['avg_gflops'].iloc[0]
            worst_perf = shape_data['avg_gflops'].iloc[-1]
            improvement = (best_perf / worst_perf - 1) * 100
            
            ax.set_title(f'{shape}\n最大提升: {improvement:.1f}%')
            ax.set_ylabel('GFLOPS')
            ax.grid(True, alpha=0.3)
            
            # 在柱子上显示数值
            for bar, value in zip(bars, shape_data['avg_gflops']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + best_perf*0.01, 
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 隐藏多余的子图
        for idx in range(n_shapes, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('各Shape下不同Ratio配置的性能对比', fontsize=16, y=0.98)
        plt.show()
    
    def plot_ratio_analysis(self, df):
        """详细的Ratio分析"""
        ratios = df['ratio_str'].unique()
        n_ratios = len(ratios)
        
        fig, axes = plt.subplots(2, (n_ratios + 1) // 2, figsize=(8 * ((n_ratios + 1) // 2), 10))
        if n_ratios == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, ratio in enumerate(ratios):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            ratio_data = df[df['ratio_str'] == ratio].copy()
            ratio_data = ratio_data.sort_values('avg_gflops', ascending=False)
            
            # 绘制性能柱状图
            bars = ax.bar(range(len(ratio_data)), ratio_data['avg_gflops'], 
                         color=plt.cm.Set2(np.linspace(0, 1, len(ratio_data))), alpha=0.8)
            
            # 标注最佳shape
            bars[0].set_color('red')
            bars[0].set_alpha(1.0)
            
            # 设置标签
            ax.set_xticks(range(len(ratio_data)))
            ax.set_xticklabels(ratio_data['shape_str'], rotation=45)
            
            ax.set_title(f'{ratio}')
            ax.set_ylabel('GFLOPS')
            ax.grid(True, alpha=0.3)
            
            # 在柱子上显示数值
            best_perf = ratio_data['avg_gflops'].max()
            for bar, value in zip(bars, ratio_data['avg_gflops']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + best_perf*0.01, 
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 隐藏多余的子图
        for idx in range(n_ratios, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('各Ratio配置下不同Shape的性能对比', fontsize=16, y=0.98)
        plt.show()
    
    def plot_performance_trends(self, df):
        """性能趋势分析"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overlap Ratio 趋势
        axes[0, 0].set_title('Overlap Ratio 对性能的影响')
        for shape in df['shape_str'].unique():
            shape_data = df[df['shape_str'] == shape]
            overlap_trend = shape_data.groupby('overlap_ratio')['avg_gflops'].mean()
            axes[0, 0].plot(overlap_trend.index, overlap_trend.values, 'o-', label=shape, linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Overlap Ratio')
        axes[0, 0].set_ylabel('Average GFLOPS')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Prefetch Ratio 趋势
        axes[0, 1].set_title('Prefetch Ratio 对性能的影响')
        for shape in df['shape_str'].unique():
            shape_data = df[df['shape_str'] == shape]
            prefetch_trend = shape_data.groupby('prefetch_ratio')['avg_gflops'].mean()
            axes[0, 1].plot(prefetch_trend.index, prefetch_trend.values, 's-', label=shape, linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Prefetch Ratio')
        axes[0, 1].set_ylabel('Average GFLOPS')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 矩阵大小趋势
        axes[0, 2].set_title('矩阵大小对性能的影响')
        df['matrix_volume'] = df['M'] * df['K'] * df['N']
        for ratio in df['ratio_str'].unique()[:5]:  # 只显示前5个ratio
            ratio_data = df[df['ratio_str'] == ratio]
            axes[0, 2].scatter(ratio_data['matrix_volume'], ratio_data['avg_gflops'], 
                             label=ratio, alpha=0.7, s=50)
        axes[0, 2].set_xlabel('Matrix Volume (M*K*N)')
        axes[0, 2].set_ylabel('GFLOPS')
        axes[0, 2].set_xscale('log')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 箱线图 - Overlap Ratio
        axes[1, 0].set_title('Overlap Ratio 性能分布箱线图')
        overlap_groups = [df[df['overlap_ratio'] == ratio]['avg_gflops'].values 
                         for ratio in sorted(df['overlap_ratio'].unique())]
        overlap_labels = [f'{ratio:.2f}' for ratio in sorted(df['overlap_ratio'].unique())]
        box1 = axes[1, 0].boxplot(overlap_groups, labels=overlap_labels, patch_artist=True)
        for patch in box1['boxes']:
            patch.set_facecolor('lightblue')
        axes[1, 0].set_xlabel('Overlap Ratio')
        axes[1, 0].set_ylabel('GFLOPS')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 箱线图 - Prefetch Ratio
        axes[1, 1].set_title('Prefetch Ratio 性能分布箱线图')
        prefetch_groups = [df[df['prefetch_ratio'] == ratio]['avg_gflops'].values 
                          for ratio in sorted(df['prefetch_ratio'].unique())]
        prefetch_labels = [f'{ratio:.2f}' for ratio in sorted(df['prefetch_ratio'].unique())]
        box2 = axes[1, 1].boxplot(prefetch_groups, labels=prefetch_labels, patch_artist=True)
        for patch in box2['boxes']:
            patch.set_facecolor('lightgreen')
        axes[1, 1].set_xlabel('Prefetch Ratio')
        axes[1, 1].set_ylabel('GFLOPS')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 相关性分析
        axes[1, 2].set_title('性能相关性分析')
        correlation_data = df[['overlap_ratio', 'prefetch_ratio', 'avg_gflops', 'matrix_volume']].corr()
        im = axes[1, 2].imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(correlation_data.columns)))
        axes[1, 2].set_yticks(range(len(correlation_data.columns)))
        axes[1, 2].set_xticklabels(correlation_data.columns, rotation=45)
        axes[1, 2].set_yticklabels(correlation_data.columns)
        
        # 添加相关系数标注
        for i in range(len(correlation_data.columns)):
            for j in range(len(correlation_data.columns)):
                text = axes[1, 2].text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[1, 2], label='Correlation')
        
        plt.tight_layout()
        plt.suptitle('性能趋势和分布分析', fontsize=16, y=0.98)
        plt.show()
    
    def plot_detailed_heatmaps(self, df):
        """详细的热力图分析"""
        shapes = df['shape_str'].unique()
        n_shapes = len(shapes)
        
        # 创建综合热力图
        fig, axes = plt.subplots(1, n_shapes, figsize=(6 * n_shapes, 5))
        if n_shapes == 1:
            axes = [axes]
        
        for idx, shape in enumerate(shapes):
            ax = axes[idx]
            shape_data = df[df['shape_str'] == shape]
            
            # 创建数据透视表
            pivot_table = shape_data.pivot_table(
                values='avg_gflops', 
                index='prefetch_ratio', 
                columns='overlap_ratio', 
                aggfunc='mean'
            )
            
            # 绘制热力图
            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'GFLOPS'})
            ax.set_title(f'性能热力图: {shape}')
            ax.set_xlabel('Overlap Ratio')
            ax.set_ylabel('Prefetch Ratio')
        
        plt.tight_layout()
        plt.suptitle('各Shape的Ratio参数性能热力图', fontsize=16, y=1.02)
        plt.show()
        
        # 创建性能改善热力图
        self.plot_improvement_heatmap(df)
    
    def plot_improvement_heatmap(self, df):
        """性能改善热力图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 相对于baseline (0,0)的改善
        baseline_data = df[(df['overlap_ratio'] == 0.0) & (df['prefetch_ratio'] == 0.0)]
        
        if len(baseline_data) > 0:
            ax = axes[0]
            improvement_data = []
            
            for shape in df['shape_str'].unique():
                shape_baseline = baseline_data[baseline_data['shape_str'] == shape]
                if len(shape_baseline) > 0:
                    baseline_perf = shape_baseline['avg_gflops'].iloc[0]
                    shape_data = df[df['shape_str'] == shape]
                    
                    for _, row in shape_data.iterrows():
                        improvement = (row['avg_gflops'] / baseline_perf - 1) * 100
                        improvement_data.append({
                            'shape': shape,
                            'overlap_ratio': row['overlap_ratio'],
                            'prefetch_ratio': row['prefetch_ratio'],
                            'improvement': improvement
                        })
            
            if improvement_data:
                imp_df = pd.DataFrame(improvement_data)
                pivot_imp = imp_df.groupby(['overlap_ratio', 'prefetch_ratio'])['improvement'].mean().reset_index()
                pivot_table = pivot_imp.pivot(index='prefetch_ratio', columns='overlap_ratio', values='improvement')
                
                sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', 
                           center=0, ax=ax, cbar_kws={'label': 'Improvement (%)'})
                ax.set_title('相对于baseline(0,0)的性能改善 (%)')
                ax.set_xlabel('Overlap Ratio')
                ax.set_ylabel('Prefetch Ratio')
        
        # 2. 最佳配置分布
        ax = axes[1]
        best_configs = []
        for shape in df['shape_str'].unique():
            shape_data = df[df['shape_str'] == shape]
            best_config = shape_data.loc[shape_data['avg_gflops'].idxmax()]
            best_configs.append({
                'shape': shape,
                'overlap_ratio': best_config['overlap_ratio'],
                'prefetch_ratio': best_config['prefetch_ratio'],
                'performance': best_config['avg_gflops']
            })
        
        if best_configs:
            best_df = pd.DataFrame(best_configs)
            
            # 创建散点图显示最佳配置
            scatter = ax.scatter(best_df['overlap_ratio'], best_df['prefetch_ratio'], 
                               s=best_df['performance']/5, c=best_df['performance'], 
                               cmap='plasma', alpha=0.7, edgecolors='black')
            
            # 添加标注
            for _, row in best_df.iterrows():
                ax.annotate(row['shape'], 
                           (row['overlap_ratio'], row['prefetch_ratio']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_xlabel('Overlap Ratio')
            ax.set_ylabel('Prefetch Ratio')
            ax.set_title('各Shape的最佳Ratio配置\n(大小=性能, 颜色=性能)')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='GFLOPS')
        
        plt.tight_layout()
        plt.show()
    
    def plot_efficiency_analysis(self, df):
        """效率分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 计算效率 vs 矩阵大小
        theoretical_peak = 1000  # 假设理论峰值
        df['efficiency'] = df['avg_gflops'] / theoretical_peak * 100
        
        axes[0, 0].scatter(df['matrix_volume'], df['efficiency'], 
                          c=df['overlap_ratio'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Matrix Volume (M*K*N)')
        axes[0, 0].set_ylabel('Efficiency (%)')
        axes[0, 0].set_title('计算效率 vs 矩阵大小')
        axes[0, 0].set_xscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Overlap Ratio')
        
        # 2. 性能稳定性分析
        df['cv'] = df['std_time_ms'] / df['avg_time_ms'] * 100  # 变异系数
        axes[0, 1].scatter(df['avg_gflops'], df['cv'], 
                          c=df['prefetch_ratio'], cmap='coolwarm', alpha=0.6)
        axes[0, 1].set_xlabel('Average GFLOPS')
        axes[0, 1].set_ylabel('变异系数 (%)')
        axes[0, 1].set_title('性能 vs 稳定性')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Prefetch Ratio')
        
        # 3. Ratio组合效果分析
        ratio_performance = df.groupby(['overlap_ratio', 'prefetch_ratio']).agg({
            'avg_gflops': ['mean', 'std', 'count']
        }).round(2)
        ratio_performance.columns = ['mean_gflops', 'std_gflops', 'count']
        ratio_performance = ratio_performance.reset_index()
        
        scatter = axes[1, 0].scatter(ratio_performance['overlap_ratio'], 
                                   ratio_performance['prefetch_ratio'],
                                   s=ratio_performance['mean_gflops']/3,
                                   c=ratio_performance['std_gflops'],
                                   cmap='plasma', alpha=0.7, edgecolors='black')
        
        for _, row in ratio_performance.iterrows():
            axes[1, 0].annotate(f'{row["mean_gflops"]:.0f}', 
                              (row['overlap_ratio'], row['prefetch_ratio']),
                              ha='center', va='center', fontsize=8, color='white', weight='bold')
        
        axes[1, 0].set_xlabel('Overlap Ratio')
        axes[1, 0].set_ylabel('Prefetch Ratio')
        axes[1, 0].set_title('Ratio组合效果\n(大小=平均性能, 颜色=稳定性)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Std GFLOPS')
        
        # 4. 帕累托前沿分析 (性能 vs 稳定性)
        pareto_data = df.copy()
        pareto_data = pareto_data.sort_values('avg_gflops', ascending=False)
        
        # 计算帕累托前沿
        pareto_front = []
        min_cv_so_far = float('inf')
        
        for _, row in pareto_data.iterrows():
            if row['cv'] <= min_cv_so_far:
                pareto_front.append(row)
                min_cv_so_far = row['cv']
        
        pareto_df = pd.DataFrame(pareto_front)
        
        # 绘制所有点
        axes[1, 1].scatter(df['avg_gflops'], df['cv'], alpha=0.5, color='lightblue', 
                          label='所有配置')
        
        # 绘制帕累托前沿
        if len(pareto_df) > 0:
            axes[1, 1].scatter(pareto_df['avg_gflops'], pareto_df['cv'], 
                             color='red', s=100, alpha=0.8, label='帕累托前沿')
            axes[1, 1].plot(pareto_df['avg_gflops'], pareto_df['cv'], 
                           'r--', alpha=0.7, linewidth=2)
            
            # 标注帕累托前沿点
            for _, row in pareto_df.iterrows():
                axes[1, 1].annotate(f"{row['shape_str']}\no={row['overlap_ratio']}, p={row['prefetch_ratio']}", 
                                   (row['avg_gflops'], row['cv']),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
        
        axes[1, 1].set_xlabel('Average GFLOPS')
        axes[1, 1].set_ylabel('变异系数 (%)')
        axes[1, 1].set_title('帕累托前沿: 性能 vs 稳定性')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('效率和稳定性分析', fontsize=16, y=0.98)
        plt.show()
    
    def generate_performance_report(self, df):
        """生成性能报告"""
        print("\n" + "="*80)
        print("                    CUTLASS GEMM 性能分析报告")
        print("="*80)
        
        # 整体统计
        print(f"\n📊 整体统计:")
        print(f"   总测试数量: {len(df)}")
        print(f"   平均性能: {df['avg_gflops'].mean():.2f} ± {df['avg_gflops'].std():.2f} GFLOPS")
        print(f"   性能范围: {df['avg_gflops'].min():.2f} - {df['avg_gflops'].max():.2f} GFLOPS")
        print(f"   最大性能提升: {((df['avg_gflops'].max() / df['avg_gflops'].min()) - 1) * 100:.1f}%")
        
        # 最佳配置
        best_overall = df.loc[df['avg_gflops'].idxmax()]
        print(f"\n🏆 全局最佳配置:")
        print(f"   矩阵大小: {best_overall['shape_str']}")
        print(f"   参数: overlap={best_overall['overlap_ratio']}, prefetch={best_overall['prefetch_ratio']}")
        print(f"   性能: {best_overall['avg_gflops']:.2f} GFLOPS")
        print(f"   执行时间: {best_overall['avg_time_ms']:.3f} ± {best_overall['std_time_ms']:.3f} ms")
        
        # 各Shape最佳配置
        print(f"\n🎯 各Shape最佳配置:")
        for shape in df['shape_str'].unique():
            shape_data = df[df['shape_str'] == shape]
            best_shape = shape_data.loc[shape_data['avg_gflops'].idxmax()]
            worst_shape = shape_data.loc[shape_data['avg_gflops'].idxmin()]
            improvement = (best_shape['avg_gflops'] / worst_shape['avg_gflops'] - 1) * 100
            
            print(f"   {shape}:")
            print(f"     最佳: o={best_shape['overlap_ratio']}, p={best_shape['prefetch_ratio']} "
                  f"-> {best_shape['avg_gflops']:.2f} GFLOPS")
            print(f"     提升: {improvement:.1f}% (相对最差配置)")
        
        # 各Ratio最佳Shape
        print(f"\n⚙️ 各Ratio最佳Shape:")
        for ratio_str in df['ratio_str'].unique():
            ratio_data = df[df['ratio_str'] == ratio_str]
            best_ratio = ratio_data.loc[ratio_data['avg_gflops'].idxmax()]
            
            print(f"   {ratio_str}:")
            print(f"     最佳Shape: {best_ratio['shape_str']} -> {best_ratio['avg_gflops']:.2f} GFLOPS")
        
        # 参数效果分析
        print(f"\n🔍 参数效果分析:")
        overlap_effect = df.groupby('overlap_ratio')['avg_gflops'].mean()
        prefetch_effect = df.groupby('prefetch_ratio')['avg_gflops'].mean()
        
        best_overlap = overlap_effect.idxmax()
        best_prefetch = prefetch_effect.idxmax()
        
        print(f"   最优Overlap Ratio: {best_overlap} (平均 {overlap_effect[best_overlap]:.2f} GFLOPS)")
        print(f"   最优Prefetch Ratio: {best_prefetch} (平均 {prefetch_effect[best_prefetch]:.2f} GFLOPS)")
        
        # 稳定性分析
        df['cv'] = df['std_time_ms'] / df['avg_time_ms'] * 100
        most_stable = df.loc[df['cv'].idxmin()]
        print(f"\n📈 稳定性分析:")
        print(f"   最稳定配置: {most_stable['shape_str']}, o={most_stable['overlap_ratio']}, "
              f"p={most_stable['prefetch_ratio']}")
        print(f"   变异系数: {most_stable['cv']:.2f}%")
        print(f"   平均变异系数: {df['cv'].mean():.2f} ± {df['cv'].std():.2f}%")
        
        # 推荐配置
        print(f"\n💡 推荐配置:")
        
        # 按性能推荐
        top3_perf = df.nlargest(3, 'avg_gflops')
        print(f"   高性能推荐 (Top 3):")
        for i, (_, row) in enumerate(top3_perf.iterrows(), 1):
            print(f"     {i}. {row['shape_str']}, o={row['overlap_ratio']}, p={row['prefetch_ratio']} "
                  f"-> {row['avg_gflops']:.2f} GFLOPS")
        
        # 按稳定性推荐
        stable_configs = df[df['avg_gflops'] > df['avg_gflops'].quantile(0.8)]  # 性能前20%
        top3_stable = stable_configs.nsmallest(3, 'cv')
        print(f"   稳定性推荐 (高性能+低变异):")
        for i, (_, row) in enumerate(top3_stable.iterrows(), 1):
            print(f"     {i}. {row['shape_str']}, o={row['overlap_ratio']}, p={row['prefetch_ratio']} "
                  f"-> {row['avg_gflops']:.2f} GFLOPS (CV: {row['cv']:.2f}%)")
        
        print("="*80)

def generate_comprehensive_test_cases():
    """生成全面的测试样例"""
    
    # 1. 多样化的矩阵大小
    sizes = [
        # 小矩阵 (快速测试)
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        
        # 中等矩阵
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (3072, 3072, 3072),
        
        # 大矩阵
        (4096, 4096, 4096),
        (6144, 6144, 6144),
        
        # 非正方形矩阵
        (1024, 2048, 1024),  # M=K<N
        (2048, 1024, 2048),  # K<M=N
        (1024, 4096, 2048),  # M<N<K
        (4096, 1024, 2048),  # K<N<M
        
        # 特殊尺寸 (GPU友好的倍数)
        (1280, 1280, 1280),  # 5*256
        (1792, 1792, 1792),  # 7*256
        (2560, 2560, 2560),  # 10*256
    ]
    
    # 2. 全面的ratio组合
    ratios = [
        # 基础测试
        (0.0, 0.0),    # 无overlap, 无prefetch (baseline)
        (1.0, 1.0),    # 最大overlap和prefetch
        
        # 单变量测试 (只改变一个参数)
        (0.25, 0.0), (0.5, 0.0), (0.75, 0.0),  # 只有overlap
        (0.0, 0.25), (0.0, 0.5), (0.0, 0.75),  # 只有prefetch
        
        # 对称组合
        (0.25, 0.25), (0.5, 0.5), (0.75, 0.75),
        
        # 非对称组合
        (0.25, 0.75), (0.75, 0.25),  # 高prefetch-低overlap vs 低prefetch-高overlap
        (0.5, 0.25), (0.25, 0.5),    # 中等一个-低另一个
        (0.5, 0.75), (0.75, 0.5),    # 中等一个-高另一个
        
        # 精细测试
        (0.1, 0.1), (0.9, 0.9),      # 接近边界
        (0.33, 0.67), (0.67, 0.33),  # 1/3 vs 2/3
        (0.4, 0.6), (0.6, 0.4),      # 接近中等但不对称
    ]
    
    return sizes, ratios

def run_comprehensive_benchmark():
    """运行全面的基准测试"""
    print("🚀 开始CUTLASS GEMM综合性能测试")
    print("="*80)
    
    benchmark = cutlassGemmWithPrefetchBenchmark()
    
    # 生成测试样例
    sizes, ratios = generate_comprehensive_test_cases()
    
    print(f"📋 测试配置:")
    print(f"   矩阵大小: {len(sizes)} 种")
    print(f"   Ratio组合: {len(ratios)} 种") 
    print(f"   总测试数: {len(sizes) * len(ratios)}")
    print(f"   预计时间: ~{len(sizes) * len(ratios) * 0.5:.1f} 分钟")
    print("-"*80)
    
    # 运行测试
    results = benchmark.benchmark_multiple_sizes(sizes, ratios, iterations=30)
    
    # 创建全面分析
    print("\n🔍 开始分析结果...")
    df = benchmark.create_comprehensive_analysis(results)
    
    if df is not None and len(df) > 0:
        # 生成性能报告
        benchmark.generate_performance_report(df)
        
        # 保存结果
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            # 保存详细结果
            import json
            with open('cutlass_comprehensive_results.json', 'w') as f:
                json.dump(successful_results, f, indent=2)
            
            # 保存分析数据
            df.to_csv('cutlass_analysis_data.csv', index=False)
            
            print(f"\n💾 结果已保存:")
            print(f"   详细数据: cutlass_comprehensive_results.json")
            print(f"   分析数据: cutlass_analysis_data.csv")
            
            # 最终总结
            print(f"\n🎉 测试完成!")
            print(f"   成功测试: {len(successful_results)} / {len(results)}")
            print(f"   最高性能: {df['avg_gflops'].max():.2f} GFLOPS")
            print(f"   最佳配置: {df.loc[df['avg_gflops'].idxmax(), 'shape_str']}, "
                  f"o={df.loc[df['avg_gflops'].idxmax(), 'overlap_ratio']}, "
                  f"p={df.loc[df['avg_gflops'].idxmax(), 'prefetch_ratio']}")
    else:
        print("❌ 没有成功的测试结果")
    
    return results, df

if __name__ == "__main__":
    # 运行全面测试
    results, analysis_df = run_comprehensive_benchmark()
    
    print("\n" + "="*80)
    print("测试完成! 所有图表已显示，数据已保存。")
    print("测试说明：另一个主循环参数`args.mainloop.overlap_ratio`取值范围为[0.0, 1.0]，它决定了下一个内核（进行预取的内核）启动的时间提前量。该值越小，重叠度越高；值越大，重叠度越低。负值会完全禁用PDL，即不会有重叠，这将导致预取无效。 ")
    
    print("初始化后，预取线程束开始将 K 个 “A” 的瓦片加载到共享内存的未使用部分，并加载同一计算线程阵列（CTA）最终将加载的所有 K 个瓦片中的一半。\
            实际加载的 K 个瓦片的确切数量由args.mainloop.prefetch_ratio决定，该值范围在 [0.0, 1.0] 之间。值越小，预取越少；值越大，预取越多。\
        负值则导致 “尽力而为” 的预取，这意味着一旦激活直接内存访问（DMA）线程束开始加载（一旦接收到前一个内核已刷新其内存的信号），预取器就会停止发出权重加载请求。")
    print("此示例为 Hopper 架构实现了一个非持久化的、针对 warp 优化的通用矩阵乘法（GEMM）内核，并采用可编程相关启动（PDL），可将权重预取到二级缓存（L2 cache）中。\
有关相关启动的更多信息，请参考 CUDA 编程指南：\
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html\#programmatic-dependent-launch-and-synchronization \
在某些情况下，可编程相关启动（PDL）可能会导致出现一个时间窗口，在前一个内核未积极使用动态随机存取存储器（DRAM），而后一个内核处于空闲状态，直到前一个内核完成。在这个时间窗口内，后一个内核可以开始加载一个非相关操作数（例如，线性投影中的权重通常是静态的），并将其缓存到二级缓存（L2）中。")
    print("="*80)