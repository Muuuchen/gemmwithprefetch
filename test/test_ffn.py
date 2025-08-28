import torch
import torch.nn as nn
import cutlass_gemm_with_prefetch
import numpy as np
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class FFNConfig:
    """FFN配置参数"""
    gemm1_overlap: float
    gemm1_prefetch: float
    rmsnorm_prefetch: float
    rmsnorm_hierarchy: object
    gemm2_overlap: float
    gemm2_prefetch: float
    
    def to_dict(self):
        return {
            'gemm1_overlap': self.gemm1_overlap,
            'gemm1_prefetch': self.gemm1_prefetch,
            'rmsnorm_prefetch': self.rmsnorm_prefetch,
            'rmsnorm_hierarchy': 'SHAREDMEM' if self.rmsnorm_hierarchy == cutlass_gemm_with_prefetch.KernelOverlapHierarchy.SHAREDMEM else 'FREFTECH',
            'gemm2_overlap': self.gemm2_overlap,
            'gemm2_prefetch': self.gemm2_prefetch
        }

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    config: FFNConfig
    latency_ms: float
    tflops: float
    bandwidth_gbps: float

class FFNParameterExplorer:
    """FFN参数空间探索器"""
    
    def __init__(self, d_model: int = 4096, d_ff: int = 11008,
                 batch_size: int = 1, seq_len: int = 128):
        self.d_model = d_model
        self.d_ff = d_ff
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = 'cuda'
        
        # 初始化权重
        self._init_weights()
        
        # 定义参数空间
        self.mm_values = list(np.arange(0.0, 1.1, 0.1)) + [-1.0]
        self.rmsnorm_prefetch_values = list(np.arange(0.0, 1.1, 0.1))
        self.rmsnorm_hierarchy_values = [
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH,
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.SHAREDMEM
        ]
        
    def _init_weights(self):
        """初始化权重"""
        scale = 0.02
        self.W1 = (torch.randn(self.d_model, self.d_ff, device=self.device) * scale).to(torch.float8_e5m2)
        self.b1 = torch.zeros(self.d_ff, device=self.device).to(torch.float8_e4m3fn)
        self.W2 = (torch.randn(self.d_ff, self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b2 = torch.zeros(self.d_model, device=self.device).to(torch.float8_e4m3fn)
        self.norm_weight = torch.ones(self.d_ff, device=self.device).to(torch.float8_e4m3fn)
    
    def forward_pass(self, x: torch.Tensor, config: FFNConfig) -> torch.Tensor:
        """执行一次前向传播: GEMM1 → RMSNorm → GEMM2"""
        batch_size, seq_len, _ = x.shape
        residual = x.to(torch.float8_e4m3fn)
        
        # 1. First GEMM: x @ W1 + b1
        x_flat = x.reshape(-1, self.d_model).to(torch.float8_e4m3fn)
        hidden = torch.zeros(batch_size * seq_len, self.d_ff, 
                           device=x.device, dtype=torch.float8_e4m3fn)
        
        hidden = cutlass_gemm_with_prefetch.mm(
            x_flat,
            self.W1,
            hidden,
            self.b1.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.gemm1_overlap,
            config.gemm1_prefetch
        )
        
        # 2. RMSNorm on hidden layer
        norm_output = torch.zeros_like(hidden, dtype=torch.float8_e4m3fn)
        
        if config.rmsnorm_hierarchy == cutlass_gemm_with_prefetch.KernelOverlapHierarchy.SHAREDMEM:
            # SHAREDMEM模式不使用prefetch参数
            hidden_norm = cutlass_gemm_with_prefetch.rmsnorm(
                norm_output,
                hidden,
                self.norm_weight,
                0.0,  # prefetch参数在SHAREDMEM模式下被忽略
                config.rmsnorm_hierarchy
            )
        else:
            hidden_norm = cutlass_gemm_with_prefetch.rmsnorm(
                norm_output,
                hidden,
                self.norm_weight,
                config.rmsnorm_prefetch,
                config.rmsnorm_hierarchy
            )
        
        

        output = cutlass_gemm_with_prefetch.mm(
            hidden_norm,
            self.W2,
            residual.reshape(-1, self.d_model),  # 使用residual作为输出buffer
            self.b2.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.gemm2_overlap,
            config.gemm2_prefetch
        )
        
        return output.reshape(batch_size, seq_len, self.d_model)
    
    def benchmark_config(self, config: FFNConfig, warmup: int = 10, iterations: int = 50) -> BenchmarkResult:
        """基准测试单个配置"""
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, 
                       device=self.device, dtype=torch.float16)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.forward_pass(x, config)
        torch.cuda.synchronize()
        
        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            torch.cuda.synchronize()
            start_event.record()
            
            for _ in range(iterations):
                output = self.forward_pass(x, config)
                
            end_event.record()
            torch.cuda.synchronize()
        
        # 计算指标
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / iterations
        
        # FLOPS计算
        flops = self.batch_size * self.seq_len * (
            2 * self.d_model * self.d_ff +       # GEMM1
            self.d_ff * 6 +                       # RMSNorm
            self.d_ff * 4 +                       # GELU
            2 * self.d_ff * self.d_model         # GEMM2
        )
        tflops = (flops / 1e12) / (avg_time_ms / 1000)
        
        # 带宽计算
        bytes_accessed = self.batch_size * self.seq_len * (
            self.d_model * 2 +                    # x + residual读取
            self.d_model * self.d_ff +            # W1
            self.d_ff +                           # b1
            self.d_ff +                           # norm_weight
            self.d_ff * self.d_model +            # W2
            self.d_model +                        # b2
            self.d_ff + self.d_model              # 写入
        ) * 2  # FP8约2字节
        
        bandwidth_gbps = (bytes_accessed / 1e9) / (avg_time_ms / 1000)
        
        return BenchmarkResult(config, avg_time_ms, tflops, bandwidth_gbps)
    
    def generate_configs(self, sample_ratio: float = 1.0) -> List[FFNConfig]:
        """生成参数配置的笛卡尔积"""
        configs = []
        
        # 生成所有GEMM参数组合
        gemm_params = list(itertools.product(self.mm_values, self.mm_values))
        
        # 生成RMSNorm参数组合
        rmsnorm_params = []
        for hierarchy in self.rmsnorm_hierarchy_values:
            if hierarchy == cutlass_gemm_with_prefetch.KernelOverlapHierarchy.SHAREDMEM:
                # SHAREDMEM只需要一个配置
                rmsnorm_params.append((0.0, hierarchy))
            else:
                # FREFTECH需要测试所有prefetch值
                for prefetch in self.rmsnorm_prefetch_values:
                    rmsnorm_params.append((prefetch, hierarchy))
        
        # 生成完整的笛卡尔积
        for (g1_overlap, g1_prefetch), (rms_prefetch, rms_hierarchy), (g2_overlap, g2_prefetch) in \
            itertools.product(gemm_params, rmsnorm_params, gemm_params):
            
            configs.append(FFNConfig(
                gemm1_overlap=g1_overlap,
                gemm1_prefetch=g1_prefetch,
                rmsnorm_prefetch=rms_prefetch,
                rmsnorm_hierarchy=rms_hierarchy,
                gemm2_overlap=g2_overlap,
                gemm2_prefetch=g2_prefetch
            ))
        
        # 如果需要采样
        if sample_ratio < 1.0:
            n_samples = int(len(configs) * sample_ratio)
            configs = np.random.choice(configs, n_samples, replace=False).tolist()
        
        return configs
    
    def explore_parameter_space(self, sample_ratio: float = 0.1):
        """探索参数空间"""
        configs = self.generate_configs(sample_ratio)
        total_configs = len(configs)
        
        print(f"Total configurations to test: {total_configs}")
        if sample_ratio < 1.0:
            print(f"(Sampled {sample_ratio*100:.0f}% of full parameter space)")
        
        results = []
        
        # 测试每个配置
        for i, config in enumerate(configs):
            try:
                result = self.benchmark_config(config)
                results.append(result)
                
                if (i + 1) % 50 == 0:
                    print(f"Progress: {i+1}/{total_configs} configurations tested...")
                    
            except Exception as e:
                print(f"Failed config {i+1}: {e}")
        
        return results
    
    def analyze_results(self, results: List[BenchmarkResult]):
        """分析结果"""
        if not results:
            print("No results to analyze")
            return
        
        # 转换为DataFrame便于分析
        data = []
        for r in results:
            row = r.config.to_dict()
            row['latency_ms'] = r.latency_ms
            row['tflops'] = r.tflops
            row['bandwidth_gbps'] = r.bandwidth_gbps
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 找出最优和最差配置
        best_idx = df['latency_ms'].idxmin()
        worst_idx = df['latency_ms'].idxmax()
        
        print("\n" + "="*60)
        print("Performance Analysis")
        print("="*60)
        
        print(f"\nBest configuration (lowest latency):")
        print(df.iloc[best_idx])
        
        print(f"\nWorst configuration (highest latency):")
        print(df.iloc[worst_idx])
        
        speedup = df.iloc[worst_idx]['latency_ms'] / df.iloc[best_idx]['latency_ms']
        print(f"\nSpeedup from worst to best: {speedup:.2f}x")
        
        # 统计分析
        print("\nLatency statistics:")
        print(f"  Mean: {df['latency_ms'].mean():.3f} ms")
        print(f"  Std: {df['latency_ms'].std():.3f} ms")
        print(f"  Min: {df['latency_ms'].min():.3f} ms")
        print(f"  Max: {df['latency_ms'].max():.3f} ms")
        
        # 参数影响分析
        print("\nParameter impact on performance (correlation with latency):")
        numeric_cols = ['gemm1_overlap', 'gemm1_prefetch', 'rmsnorm_prefetch', 
                       'gemm2_overlap', 'gemm2_prefetch', 'latency_ms']
        correlations = df[numeric_cols].corr()['latency_ms'].drop('latency_ms')
        for param, corr in correlations.items():
            print(f"  {param}: {corr:.3f}")
        
        # RMSNorm hierarchy分析
        print("\nRMSNorm hierarchy comparison:")
        for hierarchy in ['FREFTECH', 'SHAREDMEM']:
            subset = df[df['rmsnorm_hierarchy'] == hierarchy]
            if not subset.empty:
                print(f"  {hierarchy}: mean latency = {subset['latency_ms'].mean():.3f} ms")
        
        return df
    
    def plot_analysis(self, df: pd.DataFrame):
        """可视化分析"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. GEMM1参数热力图
        pivot1 = df.pivot_table(values='latency_ms', 
                                index='gemm1_prefetch', 
                                columns='gemm1_overlap', 
                                aggfunc='mean')
        axes[0, 0].imshow(pivot1, cmap='RdYlBu_r', aspect='auto')
        axes[0, 0].set_title('GEMM1 Parameter Impact')
        axes[0, 0].set_xlabel('Overlap')
        axes[0, 0].set_ylabel('Prefetch')
        
        # 2. GEMM2参数热力图
        pivot2 = df.pivot_table(values='latency_ms', 
                                index='gemm2_prefetch', 
                                columns='gemm2_overlap', 
                                aggfunc='mean')
        axes[0, 1].imshow(pivot2, cmap='RdYlBu_r', aspect='auto')
        axes[0, 1].set_title('GEMM2 Parameter Impact')
        axes[0, 1].set_xlabel('Overlap')
        axes[0, 1].set_ylabel('Prefetch')
        
        # 3. RMSNorm影响
        df_freftech = df[df['rmsnorm_hierarchy'] == 'FREFTECH']
        if not df_freftech.empty:
            rmsnorm_impact = df_freftech.groupby('rmsnorm_prefetch')['latency_ms'].mean()
            axes[0, 2].plot(rmsnorm_impact.index, rmsnorm_impact.values, 'o-')
            axes[0, 2].set_title('RMSNorm Prefetch Impact (FREFTECH)')
            axes[0, 2].set_xlabel('Prefetch')
            axes[0, 2].set_ylabel('Latency (ms)')
        
        # 4. 延迟分布
        axes[1, 0].hist(df['latency_ms'], bins=50, edgecolor='black')
        axes[1, 0].set_title('Latency Distribution')
        axes[1, 0].set_xlabel('Latency (ms)')
        axes[1, 0].set_ylabel('Count')
        
        # 5. TFLOPS分布
        axes[1, 1].hist(df['tflops'], bins=50, edgecolor='black')
        axes[1, 1].set_title('TFLOPS Distribution')
        axes[1, 1].set_xlabel('TFLOPS')
        axes[1, 1].set_ylabel('Count')
        
        # 6. 性能散点图
        axes[1, 2].scatter(df['latency_ms'], df['tflops'], alpha=0.5)
        axes[1, 2].set_title('Latency vs TFLOPS')
        axes[1, 2].set_xlabel('Latency (ms)')
        axes[1, 2].set_ylabel('TFLOPS')
        
        plt.tight_layout()
        plt.savefig('ffn_parameter_analysis.png', dpi=150)
        plt.close()
        
        print("\nVisualization saved as ffn_parameter_analysis.png")


def main():
    """主函数"""
    # 创建探索器
    explorer = FFNParameterExplorer(
        d_model=4096,
        d_ff=4096,
        batch_size=8,
        seq_len=128
    )
    
    print("FFN Parameter Space Exploration")
    print(f"Network: GEMM1({explorer.d_model}x{explorer.d_ff}) → "
          f"RMSNorm({explorer.d_ff}) → "
          f"GEMM2({explorer.d_ff}x{explorer.d_model})")
    print(f"Batch size: {explorer.batch_size}, Sequence length: {explorer.seq_len}")
    
    # 计算总配置数
    n_gemm_configs = len(explorer.mm_values) ** 2
    n_rmsnorm_configs = len(explorer.rmsnorm_prefetch_values) + 1  # +1 for SHAREDMEM
    total_configs = n_gemm_configs * n_rmsnorm_configs * n_gemm_configs
    
    print(f"\nParameter space size:")
    print(f"  GEMM configurations: {n_gemm_configs} each")
    print(f"  RMSNorm configurations: {n_rmsnorm_configs}")
    print(f"  Total combinations: {total_configs}")
    
    # 决定采样率
    if total_configs > 1000:
        sample_ratio = min(1000 / total_configs, 0.2)
        print(f"\nSampling {sample_ratio*100:.1f}% of parameter space due to size")
    else:
        sample_ratio = 1.0
    
    # 运行探索
    results = explorer.explore_parameter_space(sample_ratio)
    
    # 分析结果
    df = explorer.analyze_results(results)
    
    # 可视化
    if df is not None:
        explorer.plot_analysis(df)
        
        # 保存结果
        df.to_csv('ffn_parameter_results.csv', index=False)
        print("\nResults saved to ffn_parameter_results.csv")


def comprehensive_shape_analysis():
    """对不同shape进行全面的参数分析和对比"""
    
    # 定义baseline配置
    baseline_config = FFNConfig(
        gemm1_overlap=-1.0,
        gemm1_prefetch=0.0,
        rmsnorm_prefetch=0.0,
        rmsnorm_hierarchy=cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH,
        gemm2_overlap=-1.0,
        gemm2_prefetch=0.0
    )
    
    # 定义要测试的shape配置
    shape_configs = [
        # (batch_size, seq_len, d_model, d_ff, name)
        (1, 128, 4096, 11008, "LLaMA-7B-like"),
        (1, 256, 4096, 11008, "LLaMA-7B-long"),
        (4, 128, 4096, 11008, "LLaMA-7B-batch4"),
        (8, 128, 4096, 11008, "LLaMA-7B-batch8"),
        (1, 128, 5120, 13824, "LLaMA-13B-like"),
        (1, 128, 8192, 22016, "LLaMA-30B-like"),
        (1, 128, 8192, 28672, "LLaMA-70B-like"),
        (1, 512, 4096, 11008, "LLaMA-7B-long-context"),
        (16, 64, 4096, 11008, "LLaMA-7B-batch16"),
        (1, 128, 4096, 4096, "Square-FFN"),
    ]
    
    all_results = {}
    
    for batch_size, seq_len, d_model, d_ff, name in shape_configs:
        print(f"\n{'='*80}")
        print(f"Testing shape: {name}")
        print(f"Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, d_ff={d_ff}")
        print(f"{'='*80}")
        
        try:
            # 创建探索器
            explorer = FFNParameterExplorer(
                d_model=d_model,
                d_ff=d_ff,
                batch_size=batch_size,
                seq_len=seq_len
            )
            
            # 计算参数空间大小
            n_gemm_configs = len(explorer.mm_values) ** 2
            n_rmsnorm_configs = len(explorer.rmsnorm_prefetch_values) + 1
            total_configs = n_gemm_configs * n_rmsnorm_configs * n_gemm_configs
            
            # 动态调整采样率
            if total_configs <= 500:
                sample_ratio = 1.0  # 完全测试
            elif total_configs <= 2000:
                sample_ratio = 0.5  # 测试50%
            elif total_configs <= 5000:
                sample_ratio = 0.3  # 测试30%
            elif total_configs <= 10000:
                sample_ratio = 0.2  # 测试20%
            else:
                sample_ratio = max(2000 / total_configs, 0.1)  # 至少测试10%或2000个配置
            
            print(f"Parameter space size: {total_configs}, sampling ratio: {sample_ratio:.1%}")
            
            # 首先测试baseline
            print("Testing baseline configuration...")
            baseline_result = explorer.benchmark_config(baseline_config, warmup=20, iterations=100)
            print(f"Baseline latency: {baseline_result.latency_ms:.3f} ms")
            
            # 运行参数探索
            results = explorer.explore_parameter_space(sample_ratio)
            
            # 转换为DataFrame
            data = []
            for r in results:
                row = r.config.to_dict()
                row['latency_ms'] = r.latency_ms
                row['tflops'] = r.tflops
                row['bandwidth_gbps'] = r.bandwidth_gbps
                row['speedup_vs_baseline'] = baseline_result.latency_ms / r.latency_ms
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # 找出top10配置
            top10_df = df.nsmallest(10, 'latency_ms').copy()
            top10_df['rank'] = range(1, 11)
            
            # 保存结果
            all_results[name] = {
                'shape': (batch_size, seq_len, d_model, d_ff),
                'baseline_latency': baseline_result.latency_ms,
                'baseline_tflops': baseline_result.tflops,
                'top10_configs': top10_df,
                'all_results': df,
                'sample_ratio': sample_ratio,
                'total_configs_tested': len(results)
            }
            
            # 清理GPU内存
            del explorer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to test shape {name}: {e}")
            continue
    
    # 生成综合报告
    generate_comprehensive_report(all_results)
    
    return all_results


def generate_comprehensive_report(all_results):
    """生成综合性能报告"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("="*100)
    
    # 创建一个大的表格来显示所有shape的top配置
    summary_data = []
    
    for shape_name, results in all_results.items():
        top1_config = results['top10_configs'].iloc[0]
        
        summary_data.append({
            'Shape': shape_name,
            'Dimensions': f"{results['shape'][0]}x{results['shape'][1]}x{results['shape'][2]}x{results['shape'][3]}",
            'Baseline Latency (ms)': f"{results['baseline_latency']:.3f}",
            'Best Latency (ms)': f"{top1_config['latency_ms']:.3f}",
            'Speedup': f"{top1_config['speedup_vs_baseline']:.2f}x",
            'Best TFLOPS': f"{top1_config['tflops']:.1f}",
            'GEMM1 Config': f"({top1_config['gemm1_overlap']:.1f}, {top1_config['gemm1_prefetch']:.1f})",
            'RMSNorm Config': f"{top1_config['rmsnorm_hierarchy']} ({top1_config['rmsnorm_prefetch']:.1f})",
            'GEMM2 Config': f"({top1_config['gemm2_overlap']:.1f}, {top1_config['gemm2_prefetch']:.1f})",
            'Configs Tested': results['total_configs_tested']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary of Best Configurations:")
    print(summary_df.to_string(index=False))
    
    # 为每个shape生成详细的top10报告
    for shape_name, results in all_results.items():
        print(f"\n\n{'='*80}")
        print(f"Detailed Top 10 Configurations for {shape_name}")
        print(f"Shape: {results['shape']}")
        print(f"Baseline Latency: {results['baseline_latency']:.3f} ms")
        print(f"{'='*80}")
        
        top10_df = results['top10_configs']
        
        # 打印top10配置
        for idx, row in top10_df.iterrows():
            print(f"\nRank {row['rank']}:")
            print(f"  Latency: {row['latency_ms']:.3f} ms (Speedup: {row['speedup_vs_baseline']:.2f}x)")
            print(f"  TFLOPS: {row['tflops']:.1f}")
            print(f"  GEMM1: overlap={row['gemm1_overlap']:.1f}, prefetch={row['gemm1_prefetch']:.1f}")
            print(f"  RMSNorm: {row['rmsnorm_hierarchy']}, prefetch={row['rmsnorm_prefetch']:.1f}")
            print(f"  GEMM2: overlap={row['gemm2_overlap']:.1f}, prefetch={row['gemm2_prefetch']:.1f}")
        
        # 统计参数分布
        print(f"\nParameter Distribution in Top 10:")
        print("GEMM1 overlap distribution:", top10_df['gemm1_overlap'].value_counts().to_dict())
        print("GEMM1 prefetch distribution:", top10_df['gemm1_prefetch'].value_counts().to_dict())
        print("RMSNorm hierarchy distribution:", top10_df['rmsnorm_hierarchy'].value_counts().to_dict())
        print("GEMM2 overlap distribution:", top10_df['gemm2_overlap'].value_counts().to_dict())
        print("GEMM2 prefetch distribution:", top10_df['gemm2_prefetch'].value_counts().to_dict())
    
    # 保存详细结果到文件
    save_detailed_results(all_results)
    
    # 生成可视化
    create_comparison_visualizations(all_results)


def save_detailed_results(all_results):
    """保存详细结果到文件"""
    
    # 保存每个shape的完整结果
    for shape_name, results in all_results.items():
        filename = f"ffn_results_{shape_name.replace(' ', '_').replace('-', '_')}.csv"
        results['all_results'].to_csv(filename, index=False)
        print(f"\nSaved full results for {shape_name} to {filename}")
    
    # 保存汇总的top10结果
    all_top10 = []
    for shape_name, results in all_results.items():
        top10 = results['top10_configs'].copy()
        top10['shape_name'] = shape_name
        top10['shape'] = str(results['shape'])
        all_top10.append(top10)
    
    combined_top10 = pd.concat(all_top10, ignore_index=True)
    combined_top10.to_csv('ffn_all_shapes_top10.csv', index=False)
    print("\nSaved combined top 10 results to ffn_all_shapes_top10.csv")


def create_comparison_visualizations(all_results):
    """创建对比可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 不同shape的加速比对比
    shapes = []
    speedups = []
    for shape_name, results in all_results.items():
        shapes.append(shape_name)
        speedups.append(results['top10_configs'].iloc[0]['speedup_vs_baseline'])
    
    axes[0, 0].bar(range(len(shapes)), speedups)
    axes[0, 0].set_xticks(range(len(shapes)))
    axes[0, 0].set_xticklabels(shapes, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Speedup vs Baseline')
    axes[0, 0].set_title('Best Speedup Achieved for Each Shape')
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    
    # 2. 延迟对比
    baseline_latencies = []
    best_latencies = []
    for shape_name, results in all_results.items():
        baseline_latencies.append(results['baseline_latency'])
        best_latencies.append(results['top10_configs'].iloc[0]['latency_ms'])
    
    x = np.arange(len(shapes))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, baseline_latencies, width, label='Baseline')
    axes[0, 1].bar(x + width/2, best_latencies, width, label='Best Config')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(shapes, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_title('Latency Comparison: Baseline vs Best')
    axes[0, 1].legend()
    
    # 3. TFLOPS对比
    baseline_tflops = []
    best_tflops = []
    for shape_name, results in all_results.items():
        baseline_tflops.append(results['baseline_tflops'])
        best_tflops.append(results['top10_configs'].iloc[0]['tflops'])
    
    axes[1, 0].scatter(baseline_tflops, best_tflops)
    for i, shape_name in enumerate(shapes):
        axes[1, 0].annotate(shape_name, (baseline_tflops[i], best_tflops[i]), 
                           fontsize=8, rotation=45)
    
    # 添加对角线
    max_tflops = max(max(baseline_tflops), max(best_tflops))
    axes[1, 0].plot([0, max_tflops], [0, max_tflops], 'r--', label='y=x')
    axes[1, 0].set_xlabel('Baseline TFLOPS')
    axes[1, 0].set_ylabel('Best Config TFLOPS')
    axes[1, 0].set_title('TFLOPS Improvement')
    axes[1, 0].legend()
    
    # 4. Top 10配置的加速比分布
    for shape_name, results in all_results.items():
        top10_speedups = results['top10_configs']['speedup_vs_baseline'].values
        axes[1, 1].plot(range(1, 11), top10_speedups, 'o-', label=shape_name, alpha=0.7)
    
    axes[1, 1].set_xlabel('Rank')
    axes[1, 1].set_ylabel('Speedup vs Baseline')
    axes[1, 1].set_title('Top 10 Configurations Speedup Distribution')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ffn_shape_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved as ffn_shape_comparison.png")


# 修改main函数以使用新的综合分析
def main_with_analysis():
    """运行综合shape分析"""
    
    print("Starting Comprehensive FFN Parameter Analysis")
    print("This will test multiple shapes and find optimal configurations")
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    # 运行综合分析
    all_results = comprehensive_shape_analysis()
    
    # 计算总时间
    total_time = time.time() - start_time
    print(f"\n\nTotal analysis time: {total_time/60:.1f} minutes")
    
    # 打印GPU信息
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Peak Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    return all_results


if __name__ == "__main__":

    main_with_analysis()
    # main()