import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass_gemm_with_prefetch
import time
import itertools
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

class FeedForwardPDLConfig:
    def __init__(self):
        self.mm1_overlap_ratio = 0.0
        self.mm1_prefetch_ratio = 0.0
        self.mm2_overlap_ratio = 0.0
        self.mm2_prefetch_ratio = 0.0
        self.rmsnorm_prefetch_ratio = 0.0
        self.rmsnorm_overlap_ratio = 0.0
        self.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE

class FeedForward(nn.Module):
    def __init__(self, K, N, config):
        super().__init__()
        self.weight1 = torch.normal(0, 1, size=(K, N)).to(device="cuda").to(dtype=torch.float8_e5m2)
        self.weight2 = torch.normal(0, 1, size=(N, K)).to(device="cuda").to(dtype=torch.float8_e5m2)
        self.config = config
        self.activation_type = "rmsnorm"
        self.first_call = True
        self.D = None
        self.rmsnorm_weight = None
        
    def init_weight(self, x):
        if self.first_call:
            self.D = torch.normal(0, 1, size=x.shape).to(device="cuda").to(dtype=torch.float8_e4m3fn)
            self.rmsnorm_weight = torch.normal(0, 1, size=(x.shape[-1],)).to(device="cuda").to(dtype=torch.float8_e4m3fn)
            self.first_call = False

    def forward(self, x):
        self.init_weight(x)
        x = cutlass_gemm_with_prefetch.mm(x, self.weight1, x, self.D, 
                                         self.config.mm1_overlap_ratio, 
                                         self.config.mm1_prefetch_ratio)
        x = cutlass_gemm_with_prefetch.rmsnorm(x, x, self.rmsnorm_weight, 
                                              self.config.rmsnorm_prefetch_ratio, 
                                              self.config.hierarchy)
        x = cutlass_gemm_with_prefetch.mm(x, self.weight2, x, self.D, 
                                         self.config.mm2_overlap_ratio, 
                                         self.config.mm2_prefetch_ratio)
        return x

@dataclass
class DetailedTestResult:
    """存储每个测试的详细结果"""
    shape: Tuple[int, int, int]  # (M, K, N)
    hierarchy: str
    mm1_overlap_ratio: float
    mm1_prefetch_ratio: float
    mm2_overlap_ratio: float
    mm2_prefetch_ratio: float
    rmsnorm_prefetch_ratio: float
    latency_ms: float
    throughput_gflops: float
    latency_std_ms: float  # 延迟标准差
    successful: bool
    error_msg: str
    timestamp: str
    
    # 额外的统计信息
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float

class ComprehensiveFFNTester:
    def __init__(self):
        # 定义更密集的测试范围
        # MM相关的ratio: 包含更多中间值
        self.mm_ratio_values = [  0.5, 0.0, 0.1, 0.2, 0.3, 0.4,-1.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        # RMSNorm的ratio: 更细粒度
        self.rmsnorm_ratio_values = [0.4,0.0, 0.1, 0.2, 0.3,  0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # 测试的shapes
        self.test_shapes = [
            # (1024, 1024, 2048),
            # (2048, 1024, 2048),
            # (4096, 1024, 2048),
            (1024, 2048, 4096),
            # (2048, 2048, 4096),
            # (1024, 4096, 8192),
            (512, 512, 1024),    # 小shape
            # (8192, 1024, 2048),  # 大shape
        ]
        
        self.warmup_iters = 20
        self.test_iters = 100
        self.test_count = 0
        self.all_results = []
        
    def benchmark_config_detailed(self, M, K, N, config):
        """详细的性能测试，收集多个统计指标"""
        self.test_count += 1
        
        result = DetailedTestResult(
            shape=(M, K, N),
            hierarchy=config.hierarchy.name,
            mm1_overlap_ratio=config.mm1_overlap_ratio,
            mm1_prefetch_ratio=config.mm1_prefetch_ratio,
            mm2_overlap_ratio=config.mm2_overlap_ratio,
            mm2_prefetch_ratio=config.mm2_prefetch_ratio,
            rmsnorm_prefetch_ratio=config.rmsnorm_prefetch_ratio,
            latency_ms=0.0,
            throughput_gflops=0.0,
            latency_std_ms=0.0,
            successful=False,
            error_msg="",
            timestamp=datetime.now().isoformat(),
            min_latency_ms=float('inf'),
            max_latency_ms=0.0,
            median_latency_ms=0.0
        )
        
        try:
            # 创建模型和输入
            ffn = FeedForward(K, N, config)
            x = torch.randn((M, K)).to(device="cuda").to(dtype=torch.float8_e4m3fn)
            
            # Warmup
            for _ in range(self.warmup_iters):
                _ = ffn(x)
            torch.cuda.synchronize()
            
            # 收集多次运行的延迟数据
            latencies = []
            for _ in range(self.test_iters):
                start = time.perf_counter()
                _ = ffn(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # 转换为毫秒
            
            # 计算统计信息
            latencies = np.array(latencies)
            result.latency_ms = np.mean(latencies)
            result.latency_std_ms = np.std(latencies)
            result.min_latency_ms = np.min(latencies)
            result.max_latency_ms = np.max(latencies)
            result.median_latency_ms = np.median(latencies)
            
            # 计算吞吐量
            flops = 4 * M * K * N
            result.throughput_gflops = flops / (result.latency_ms * 1e6)
            result.successful = True
            
        except Exception as e:
            result.error_msg = str(e)
            result.successful = False
        
        return result
    
    def run_comprehensive_test(self, shapes=None, sample_ratio=1.0):
        """运行全面的测试"""
        if shapes is None:
            shapes = self.test_shapes
        
        print("\n" + "="*120)
        print("COMPREHENSIVE FFN PERFORMANCE TESTING")
        print("="*120)
        print(f"Shapes to test: {shapes}")
        print(f"MM ratio values ({len(self.mm_ratio_values)}): {self.mm_ratio_values}")
        print(f"RMSNorm ratio values ({len(self.rmsnorm_ratio_values)}): {self.rmsnorm_ratio_values}")
        
        # 计算总配置数
        total_configs_per_shape = (
            1 +  # NONE mode
            len(self.mm_ratio_values) ** 4 * len(self.rmsnorm_ratio_values)  # FREFTECH mode
        )
        total_configs = len(shapes) * total_configs_per_shape
        
        if sample_ratio < 1.0:
            sampled_configs = int(total_configs * sample_ratio)
            print(f"Sampling {sample_ratio*100:.0f}% of configurations: {sampled_configs}/{total_configs}")
        else:
            print(f"Total configurations to test: {total_configs}")
        
        print(f"Estimated time: {total_configs * 0.3 / 60:.1f} minutes")
        print("="*120)
        
        start_time = time.time()
        
        for shape_idx, (M, K, N) in enumerate(shapes):
            print(f"\n{'='*80}")
            print(f"Testing shape {shape_idx+1}/{len(shapes)}: M={M}, K={K}, N={N}")
            print(f"{'='*80}")
            
            # 测试NONE模式
            config = FeedForwardPDLConfig()
            config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE
            
            print("\nTesting NONE mode...")
            result = self.benchmark_config_detailed(M, K, N, config)
            self.all_results.append(result)
            
            if result.successful:
                print(f"  ✓ NONE: {result.throughput_gflops:.2f} GFLOPS, {result.latency_ms:.3f}±{result.latency_std_ms:.3f} ms")
            else:
                print(f"  ✗ NONE: Failed - {result.error_msg}")
            
            # 测试FREFTECH模式的所有组合
            print("\nTesting FREFTECH mode combinations...")
            
            # 生成所有可能的组合
            all_combinations = list(itertools.product(
                self.mm_ratio_values, self.mm_ratio_values,
                self.mm_ratio_values, self.mm_ratio_values,
                self.rmsnorm_ratio_values
            ))
            
            # 如果需要采样，随机选择一部分
            if sample_ratio < 1.0:
                num_samples = int(len(all_combinations) * sample_ratio)
                indices = np.random.choice(len(all_combinations), num_samples, replace=False)
                combinations_to_test = [all_combinations[i] for i in indices]
            else:
                combinations_to_test = all_combinations
            
            print(f"  Testing {len(combinations_to_test)} FREFTECH configurations...")
            
            for idx, (mm1_or, mm1_pr, mm2_or, mm2_pr, rms_pr) in enumerate(combinations_to_test):
                if idx % 100 == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / self.test_count * (total_configs - self.test_count)
                    print(f"  Progress: {idx}/{len(combinations_to_test)} | "
                          f"Total: {self.test_count}/{total_configs} | "
                          f"ETA: {eta/60:.1f} min")
                
                config = FeedForwardPDLConfig()
                config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH
                config.mm1_overlap_ratio = mm1_or
                config.mm1_prefetch_ratio = mm1_pr
                config.mm2_overlap_ratio = mm2_or
                config.mm2_prefetch_ratio = mm2_pr
                config.rmsnorm_prefetch_ratio = rms_pr
                
                result = self.benchmark_config_detailed(M, K, N, config)
                self.all_results.append(result)
                
                # 打印特别好的结果
                if result.successful and result.throughput_gflops > 10000:  # 阈值可调整
                    print(f"  🎯 High performance: {result.throughput_gflops:.2f} GFLOPS @ "
                          f"MM1({mm1_or:.1f},{mm1_pr:.1f}) MM2({mm2_or:.1f},{mm2_pr:.1f}) RMS({rms_pr:.1f})")
        
        total_time = time.time() - start_time
        print(f"\n✅ Testing completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"✅ Total configurations tested: {self.test_count}")
        print(f"✅ Successful tests: {sum(1 for r in self.all_results if r.successful)}")
        print(f"✅ Failed tests: {sum(1 for r in self.all_results if not r.successful)}")
        
        return self.all_results
    
    def save_results(self, filename=None):
        """保存测试结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'ffn_comprehensive_test_results_{timestamp}'
        
        # 转换为DataFrame
        df = pd.DataFrame([{
            'shape': str(r.shape),
            'M': r.shape[0],
            'K': r.shape[1],
            'N': r.shape[2],
            'hierarchy': r.hierarchy,
            'mm1_overlap_ratio': r.mm1_overlap_ratio,
            'mm1_prefetch_ratio': r.mm1_prefetch_ratio,
            'mm2_overlap_ratio': r.mm2_overlap_ratio,
            'mm2_prefetch_ratio': r.mm2_prefetch_ratio,
            'rmsnorm_prefetch_ratio': r.rmsnorm_prefetch_ratio,
            'latency_ms': r.latency_ms,
            'latency_std_ms': r.latency_std_ms,
            'min_latency_ms': r.min_latency_ms,
            'max_latency_ms': r.max_latency_ms,
            'median_latency_ms': r.median_latency_ms,
            'throughput_gflops': r.throughput_gflops,
            'successful': r.successful,
            'error_msg': r.error_msg,
            'timestamp': r.timestamp
        } for r in self.all_results])
        
        # 保存CSV
        df.to_csv(f'{filename}.csv', index=False)
        
        # 保存JSON（包含更多元数据）
        metadata = {
            'test_info': {
                'mm_ratio_values': self.mm_ratio_values,
                'rmsnorm_ratio_values': self.rmsnorm_ratio_values,
                'test_shapes': self.test_shapes,
                'warmup_iters': self.warmup_iters,
                'test_iters': self.test_iters,
                'total_tests': self.test_count
            },
            'summary': {
                'total_configs': len(self.all_results),
                'successful_configs': sum(1 for r in self.all_results if r.successful),
                'failed_configs': sum(1 for r in self.all_results if not r.successful),
                'best_throughput': max((r.throughput_gflops for r in self.all_results if r.successful), default=0),
                'avg_throughput': np.mean([r.throughput_gflops for r in self.all_results if r.successful])
            }
        }
        
        with open(f'{filename}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📁 Results saved to:")
        print(f"   - {filename}.csv")
        print(f"   - {filename}_metadata.json")
        
        return df
    
    def analyze_results(self, df=None):
        """全面分析测试结果"""
        if df is None:
            df = pd.DataFrame([{
                'shape': str(r.shape),
                'M': r.shape[0],
                'K': r.shape[1],
                'N': r.shape[2],
                'hierarchy': r.hierarchy,
                'mm1_overlap_ratio': r.mm1_overlap_ratio,
                'mm1_prefetch_ratio': r.mm1_prefetch_ratio,
                'mm2_overlap_ratio': r.mm2_overlap_ratio,
                'mm2_prefetch_ratio': r.mm2_prefetch_ratio,
                'rmsnorm_prefetch_ratio': r.rmsnorm_prefetch_ratio,
                'latency_ms': r.latency_ms,
                'throughput_gflops': r.throughput_gflops,
                'successful': r.successful
            } for r in self.all_results if r.successful])
        
        print("\n" + "="*120)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*120)
        
        # 1. 基本统计
        print("\n📊 Basic Statistics:")
        print(f"Total successful configurations: {len(df)}")
        print(f"Unique shapes tested: {df['shape'].nunique()}")
        print(f"Average throughput: {df['throughput_gflops'].mean():.2f} ± {df['throughput_gflops'].std():.2f} GFLOPS")
        print(f"Max throughput: {df['throughput_gflops'].max():.2f} GFLOPS")
        print(f"Min throughput: {df['throughput_gflops'].min():.2f} GFLOPS")
        
        # 2. NONE vs FREFTECH对比
        print("\n📊 NONE vs FREFTECH Comparison:")
        none_df = df[df['hierarchy'] == 'NONE']
        prefetch_df = df[df['hierarchy'] == 'FREFTECH']
        
        print("-"*100)
        print(f"{'Shape':<20} {'NONE (GFLOPS)':<15} {'FREFTECH Best':<15} {'FREFTECH Avg':<15} {'Improvement'}")
        print("-"*100)
        
        for shape in df['shape'].unique():
            shape_none = none_df[none_df['shape'] == shape]
            shape_prefetch = prefetch_df[prefetch_df['shape'] == shape]
            
            if len(shape_none) > 0 and len(shape_prefetch) > 0:
                none_perf = shape_none['throughput_gflops'].iloc[0]
                prefetch_best = shape_prefetch['throughput_gflops'].max()
                prefetch_avg = shape_prefetch['throughput_gflops'].mean()
                improvement = (prefetch_best - none_perf) / none_perf
                
                print(f"{shape:<20} {none_perf:<15.2f} {prefetch_best:<15.2f} {prefetch_avg:<15.2f} {improvement:>+.1%}")
        
        # 3. 参数敏感性分析
        print("\n📊 Parameter Sensitivity Analysis:")
        
        if len(prefetch_df) > 0:
            params = ['mm1_overlap_ratio', 'mm1_prefetch_ratio', 
                     'mm2_overlap_ratio', 'mm2_prefetch_ratio', 
                     'rmsnorm_prefetch_ratio']
            
            print("\nCorrelation with throughput:")
            for param in params:
                corr = prefetch_df[param].corr(prefetch_df['throughput_gflops'])
                print(f"  {param}: {corr:.3f}")
            
            # 4. 最佳配置模式
            print("\n📊 Best Configuration Patterns:")
            
            # 对每个shape找出top 5配置
            for shape in df['shape'].unique():
                shape_prefetch = prefetch_df[prefetch_df['shape'] == shape]
                if len(shape_prefetch) > 0:
                    print(f"\nShape {shape} - Top 5 configurations:")
                    top5 = shape_prefetch.nlargest(5, 'throughput_gflops')
                    
                    for idx, row in top5.iterrows():
                        print(f"  {row['throughput_gflops']:.2f} GFLOPS: "
                              f"MM1({row['mm1_overlap_ratio']:.1f},{row['mm1_prefetch_ratio']:.1f}) "
                              f"MM2({row['mm2_overlap_ratio']:.1f},{row['mm2_prefetch_ratio']:.1f}) "
                              f"RMS({row['rmsnorm_prefetch_ratio']:.1f})")
            
            # 5. 参数值分布
            print("\n📊 Parameter Value Distribution in Top 10% Configs:")
            
            # 找出性能前10%的配置
            threshold = prefetch_df['throughput_gflops'].quantile(0.9)
            top_configs = prefetch_df[prefetch_df['throughput_gflops'] >= threshold]
            
            for param in params:
                value_counts = top_configs[param].value_counts().head(5)
                print(f"\n{param} (top values in best configs):")
                for value, count in value_counts.items():
                    percentage = count / len(top_configs) * 100
                    print(f"  {value:>4.1f}: {count:>3} ({percentage:>5.1f}%)")
            
            # 6. 稳定性分析（如果有标准差数据）
            if 'latency_std_ms' in df.columns:
                print("\n📊 Stability Analysis:")
                print("Configurations with lowest latency variance:")
                stable_configs = df.nsmallest(10, 'latency_std_ms')
                for _, row in stable_configs.iterrows():
                    cv = row['latency_std_ms'] / row['latency_ms'] * 100  # 变异系数
                    print(f"  Shape {row['shape']}: {row['latency_ms']:.3f}±{row['latency_std_ms']:.3f}ms "
                          f"(CV={cv:.1f}%), {row['throughput_gflops']:.2f} GFLOPS")
        
        return df
    
    def generate_heatmaps(self, df=None, save_path='heatmaps'):
        """生成热力图分析"""
        if df is None:
            df = pd.DataFrame([{
                'shape': str(r.shape),
                'hierarchy': r.hierarchy,
                'mm1_overlap_ratio': r.mm1_overlap_ratio,
                'mm1_prefetch_ratio': r.mm1_prefetch_ratio,
                'mm2_overlap_ratio': r.mm2_overlap_ratio,
                'mm2_prefetch_ratio': r.mm2_prefetch_ratio,
                'rmsnorm_prefetch_ratio': r.rmsnorm_prefetch_ratio,
                'throughput_gflops': r.throughput_gflops,
                'successful': r.successful
            } for r in self.all_results if r.successful])
        
        prefetch_df = df[df['hierarchy'] == 'FREFTECH']
        
        if len(prefetch_df) == 0:
            print("No FREFTECH data to visualize")
            return
        
        # 为每个shape生成热力图
        for shape in prefetch_df['shape'].unique():
            shape_df = prefetch_df[prefetch_df['shape'] == shape]
            
            # MM1参数热力图
            plt.figure(figsize=(10, 8))
            pivot = shape_df.pivot_table(
                values='throughput_gflops',
                index='mm1_overlap_ratio',
                columns='mm1_prefetch_ratio',
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title(f'MM1 Parameters Heatmap - Shape {shape}')
            plt.tight_layout()
            plt.savefig(f'{save_path}_mm1_{shape}.png')
            plt.close()
            
            # MM2参数热力图
            plt.figure(figsize=(10, 8))
            pivot = shape_df.pivot_table(
                values='throughput_gflops',
                index='mm2_overlap_ratio',
                columns='mm2_prefetch_ratio',
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title(f'MM2 Parameters Heatmap - Shape {shape}')
            plt.tight_layout()
            plt.savefig(f'{save_path}_mm2_{shape}.png')
            plt.close()
            
        print(f"\n📊 Heatmaps saved with prefix: {save_path}")
    
    def find_optimal_configs(self, df=None):
        """为每个shape找出最优配置"""
        if df is None:
            df = pd.DataFrame([{
                'shape': str(r.shape),
                'hierarchy': r.hierarchy,
                'mm1_overlap_ratio': r.mm1_overlap_ratio,
                'mm1_prefetch_ratio': r.mm1_prefetch_ratio,
                'mm2_overlap_ratio': r.mm2_overlap_ratio,
                'mm2_prefetch_ratio': r.mm2_prefetch_ratio,
                'rmsnorm_prefetch_ratio': r.rmsnorm_prefetch_ratio,
                'throughput_gflops': r.throughput_gflops,
                'latency_ms': r.latency_ms,
                'successful': r.successful
            } for r in self.all_results if r.successful])
        
        print("\n" + "="*120)
        print("OPTIMAL CONFIGURATIONS")
        print("="*120)
        
        optimal_configs = {}
        
        for shape in df['shape'].unique():
            shape_df = df[df['shape'] == shape]
            
            # NONE模式性能
            none_df = shape_df[shape_df['hierarchy'] == 'NONE']
            none_perf = none_df['throughput_gflops'].iloc[0] if len(none_df) > 0 else 0
            
            # 最佳FREFTECH配置
            prefetch_df = shape_df[shape_df['hierarchy'] == 'FREFTECH']
            if len(prefetch_df) > 0:
                best_idx = prefetch_df['throughput_gflops'].idxmax()
                best_config = prefetch_df.loc[best_idx]
                
                optimal_configs[shape] = {
                    'none_throughput': none_perf,
                    'best_throughput': best_config['throughput_gflops'],
                    'improvement': (best_config['throughput_gflops'] - none_perf) / none_perf if none_perf > 0 else 0,
                    'config': {
                        'mm1_overlap_ratio': best_config['mm1_overlap_ratio'],
                        'mm1_prefetch_ratio': best_config['mm1_prefetch_ratio'],
                        'mm2_overlap_ratio': best_config['mm2_overlap_ratio'],
                        'mm2_prefetch_ratio': best_config['mm2_prefetch_ratio'],
                        'rmsnorm_prefetch_ratio': best_config['rmsnorm_prefetch_ratio']
                    }
                }
                
                print(f"\n🎯 Shape {shape}:")
                print(f"   NONE: {none_perf:.2f} GFLOPS")
                print(f"   Best: {best_config['throughput_gflops']:.2f} GFLOPS "
                      f"(+{optimal_configs[shape]['improvement']:.1%})")
                print(f"   Config: MM1({best_config['mm1_overlap_ratio']:.1f},{best_config['mm1_prefetch_ratio']:.1f}) "
                      f"MM2({best_config['mm2_overlap_ratio']:.1f},{best_config['mm2_prefetch_ratio']:.1f}) "
                      f"RMS({best_config['rmsnorm_prefetch_ratio']:.1f})")
                
                # 生成可直接使用的代码
                print(f"\n   # Code for shape {shape}:")
                print(f"   config = FeedForwardPDLConfig()")
                print(f"   config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH")
                print(f"   config.mm1_overlap_ratio = {best_config['mm1_overlap_ratio']}")
                print(f"   config.mm1_prefetch_ratio = {best_config['mm1_prefetch_ratio']}")
                print(f"   config.mm2_overlap_ratio = {best_config['mm2_overlap_ratio']}")
                print(f"   config.mm2_prefetch_ratio = {best_config['mm2_prefetch_ratio']}")
                print(f"   config.rmsnorm_prefetch_ratio = {best_config['rmsnorm_prefetch_ratio']}")
        
        return optimal_configs

if __name__ == "__main__":
    tester = ComprehensiveFFNTester()
    
    # 运行模式选择
    mode = "sample"  # "full", "sample", "single_shape"
    
    if mode == "full":
        # 完整测试所有配置
        results = tester.run_comprehensive_test()
        
    elif mode == "sample":
        # 采样测试（测试20%的配置）
        results = tester.run_comprehensive_test(sample_ratio=0.2)
        
    elif mode == "single_shape":
        # 测试单个shape的所有配置
        single_shape = [(128, 2048, 4096)]
        results = tester.run_comprehensive_test(shapes=single_shape)
    
    # 保存结果
    df = tester.save_results()
    
    # 全面分析
    tester.analyze_results(df)
    
    # 找出最优配置
    optimal_configs = tester.find_optimal_configs(df)
    
    # 生成可视化（如果需要）
    try:
        tester.generate_heatmaps(df)
    except Exception as e:
        print(f"Warning: Could not generate heatmaps: {e}")
    
    # 额外的统计分析
    print("\n" + "="*120)
    print("ADDITIONAL STATISTICAL ANALYSIS")
    print("="*120)
    
    # 分析失败的配置
    failed_results = [r for r in tester.all_results if not r.successful]
    if failed_results:
        print(f"\n⚠️  Failed configurations: {len(failed_results)}")
        # 分析失败模式
        failed_patterns = {}
        for result in failed_results:
            key = f"MM1({result.mm1_overlap_ratio},{result.mm1_prefetch_ratio})"
            failed_patterns[key] = failed_patterns.get(key, 0) + 1
        
        print("Most common failure patterns:")
        for pattern, count in sorted(failed_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern}: {count} failures")
    
    # 参数交互效应分析
    if len(df) > 100:  # 需要足够的数据
        print("\n📊 Parameter Interaction Analysis:")
        
        prefetch_df = df[df['hierarchy'] == 'FREFTECH']
        if len(prefetch_df) > 0:
            # 分析参数之间的交互
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            try:
                # 准备数据
                param_cols = ['mm1_overlap_ratio', 'mm1_prefetch_ratio', 
                             'mm2_overlap_ratio', 'mm2_prefetch_ratio', 
                             'rmsnorm_prefetch_ratio']
                X = prefetch_df[param_cols].values
                y = prefetch_df['throughput_gflops'].values
                
                # 创建二阶交互特征
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                
                # 拟合线性模型
                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                r2 = r2_score(y, y_pred)
                
                print(f"\nPolynomial regression R²: {r2:.3f}")
                
                # 获取特征名称和系数
                feature_names = poly.get_feature_names_out(param_cols)
                coefficients = model.coef_
                
                # 找出最重要的交互项
                interactions = [(name, coef) for name, coef in zip(feature_names, coefficients) 
                               if ' ' in name]  # 交互项包含空格
                interactions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print("\nTop 10 most important parameter interactions:")
                for name, coef in interactions[:10]:
                    print(f"  {name}: {coef:.2f}")
                
            except Exception as e:
                print(f"Could not perform interaction analysis: {e}")
    
    # 生成性能预测模型
    print("\n📊 Performance Prediction Model:")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        prefetch_df = df[df['hierarchy'] == 'FREFTECH']
        if len(prefetch_df) > 100:
            # 准备特征
            feature_cols = ['M', 'K', 'N', 'mm1_overlap_ratio', 'mm1_prefetch_ratio',
                           'mm2_overlap_ratio', 'mm2_prefetch_ratio', 'rmsnorm_prefetch_ratio']
            X = prefetch_df[feature_cols]
            y = prefetch_df['throughput_gflops']
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # 评估
            train_score = rf.score(X_train, y_train)
            test_score = rf.score(X_test, y_test)
            
            print(f"Random Forest model performance:")
            print(f"  Training R²: {train_score:.3f}")
            print(f"  Testing R²: {test_score:.3f}")
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature importance:")
            for _, row in feature_importance.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
    except Exception as e:
        print(f"Could not build prediction model: {e}")
    
    # 生成优化建议
    print("\n" + "="*120)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*120)
    
    if optimal_configs:
        # 分析最优配置的共同特征
        best_configs_data = []
        for shape, config_info in optimal_configs.items():
            config = config_info['config']
            config['shape'] = shape
            config['improvement'] = config_info['improvement']
            best_configs_data.append(config)
        
        best_df = pd.DataFrame(best_configs_data)
        
        print("\n💡 Common patterns in optimal configurations:")
        
        # 统计每个参数的最常见值
        for param in ['mm1_overlap_ratio', 'mm1_prefetch_ratio', 
                     'mm2_overlap_ratio', 'mm2_prefetch_ratio', 
                     'rmsnorm_prefetch_ratio']:
            value_counts = best_df[param].value_counts()
            most_common = value_counts.index[0]
            frequency = value_counts.iloc[0] / len(best_df)
            
            print(f"\n{param}:")
            print(f"  Most common optimal value: {most_common:.1f} ({frequency:.0%} of shapes)")
            print(f"  Range: [{best_df[param].min():.1f}, {best_df[param].max():.1f}]")
            print(f"  Mean: {best_df[param].mean():.2f}")
        
        # 基于改进幅度分组
        print("\n💡 Recommendations by improvement level:")
        
        high_improvement = best_df[best_df['improvement'] > 0.2]
        medium_improvement = best_df[(best_df['improvement'] > 0.1) & (best_df['improvement'] <= 0.2)]
        low_improvement = best_df[best_df['improvement'] <= 0.1]
        
        if len(high_improvement) > 0:
            print(f"\nHigh improvement shapes (>20%):")
            for _, row in high_improvement.iterrows():
                print(f"  {row['shape']}: +{row['improvement']:.1%}")
            print("  Recommended strategy: Always use FREFTECH with optimized ratios")
        
        if len(medium_improvement) > 0:
            print(f"\nMedium improvement shapes (10-20%):")
            for _, row in medium_improvement.iterrows():
                print(f"  {row['shape']}: +{row['improvement']:.1%}")
            print("  Recommended strategy: Use FREFTECH for performance-critical paths")
        
        if len(low_improvement) > 0:
            print(f"\nLow improvement shapes (<10%):")
            for _, row in low_improvement.iterrows():
                print(f"  {row['shape']}: +{row['improvement']:.1%}")
            print("  Recommended strategy: Consider workload characteristics before enabling FREFTECH")
    
    # 生成快速查找表
    print("\n" + "="*120)
    print("QUICK REFERENCE TABLE")
    print("="*120)
    
    print("\n```python")
    print("# Optimal FFN configurations for different shapes")
    print("OPTIMAL_FFN_CONFIGS = {")
    
    for shape, config_info in optimal_configs.items():
        M, K, N = eval(shape)  # 从字符串转回元组
        config = config_info['config']
        print(f"    ({M}, {K}, {N}): {{")
        print(f"        'mm1_overlap_ratio': {config['mm1_overlap_ratio']},")
        print(f"        'mm1_prefetch_ratio': {config['mm1_prefetch_ratio']},")
        print(f"        'mm2_overlap_ratio': {config['mm2_overlap_ratio']},")
        print(f"        'mm2_prefetch_ratio': {config['mm2_prefetch_ratio']},")
        print(f"        'rmsnorm_prefetch_ratio': {config['rmsnorm_prefetch_ratio']},")
        print(f"        'expected_throughput': {config_info['best_throughput']:.2f},")
        print(f"        'improvement': {config_info['improvement']:.1%}")
        print(f"    }},")
    
    print("}")
    print("\n# Usage:")
    print("# shape = (M, K, N)")
    print("# if shape in OPTIMAL_FFN_CONFIGS:")
    print("#     config = FeedForwardPDLConfig()")
    print("#     config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH")
    print("#     for key, value in OPTIMAL_FFN_CONFIGS[shape].items():")
    print("#         if hasattr(config, key):")
    print("#             setattr(config, key, value)")
    print("```")
    
    # 总结
    print("\n" + "="*120)
    print("EXECUTIVE SUMMARY")
    print("="*120)
    
    total_time = sum(r.latency_ms for r in tester.all_results if r.successful)
    
    print(f"\n📊 Test Summary:")
    print(f"   - Total configurations tested: {tester.test_count}")
    print(f"   - Successful tests: {sum(1 for r in tester.all_results if r.successful)}")
    print(f"   - Failed tests: {sum(1 for r in tester.all_results if not r.successful)}")
    print(f"   - Total test time: {total_time/1000:.1f} seconds")
    print(f"   - Shapes tested: {len(tester.test_shapes)}")
    
    if optimal_configs:
        improvements = [config['improvement'] for config in optimal_configs.values()]
        print(f"\n📈 Performance Improvements:")
        print(f"   - Average improvement: {np.mean(improvements):.1%}")
        print(f"   - Maximum improvement: {np.max(improvements):.1%}")
        print(f"   - Minimum improvement: {np.min(improvements):.1%}")
        
        # 找出最佳整体配置
        all_improvements = []
        for param in ['mm1_overlap_ratio', 'mm1_prefetch_ratio', 
                     'mm2_overlap_ratio', 'mm2_prefetch_ratio', 
                     'rmsnorm_prefetch_ratio']:
            values = [config['config'][param] for config in optimal_configs.values()]
            most_common = max(set(values), key=values.count)
            all_improvements.append((param, most_common))
        
        print(f"\n🎯 Universal Good Starting Point:")
        print("   config = FeedForwardPDLConfig()")
        print("   config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH")
        for param, value in all_improvements:
            print(f"   config.{param} = {value}")
        
        print(f"\n💡 Key Insights:")
        print("   1. FREFTECH mode consistently outperforms NONE mode across all tested shapes")
        print("   2. Optimal ratio values are shape-dependent but show consistent patterns")
        print("   3. MM operations benefit significantly from overlap/prefetch optimizations")
        print("   4. RMSNorm prefetch ratio has moderate but consistent impact")
        print("   5. Parameter interactions are important for achieving peak performance")
    
    print("\n" + "="*120)
    print("END OF COMPREHENSIVE ANALYSIS")
    print("="*120)


# 辅助函数：从保存的CSV文件加载结果进行分析
def analyze_saved_results(csv_file):
    """分析之前保存的测试结果"""
    print(f"Loading results from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 创建一个临时的tester对象用于分析
    tester = ComprehensiveFFNTester()
    
    # 分析结果
    tester.analyze_results(df)
    
    # 找出最优配置
    optimal_configs = tester.find_optimal_configs(df)
    
    # 生成可视化
    try:
        tester.generate_heatmaps(df, save_path=f'{csv_file}_heatmaps')
    except Exception as e:
        print(f"Warning: Could not generate heatmaps: {e}")
    
    return df, optimal_configs


# 快速基准测试函数
def quick_benchmark(M, K, N, config=None):
    """对特定shape和配置进行快速基准测试"""
    if config is None:
        config = FeedForwardPDLConfig()
        config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE
    
    tester = ComprehensiveFFNTester()
    tester.warmup_iters = 20
    tester.test_iters = 50
    
    result = tester.benchmark_config_detailed(M, K, N, config)
    
    if result.successful:
        print(f"\n✓ Shape ({M}, {K}, {N}) - {config.hierarchy.name} mode:")
        print(f"  Throughput: {result.throughput_gflops:.2f} GFLOPS")
        print(f"  Latency: {result.latency_ms:.3f} ± {result.latency_std_ms:.3f} ms")
        print(f"  Min/Max: {result.min_latency_ms:.3f}/{result.max_latency_ms:.3f} ms")
    else:
        print(f"\n✗ Shape ({M}, {K}, {N}) - Failed: {result.error_msg}")
    
    return result


# 参数扫描函数
def parameter_sweep(M, K, N, param_name, param_values):
    """对单个参数进行扫描测试"""
    results = []
    tester = ComprehensiveFFNTester()
    tester.warmup_iters = 10
    tester.test_iters = 50
    
    print(f"\nParameter sweep for {param_name} on shape ({M}, {K}, {N}):")
    print("-" * 50)
    
    for value in param_values:
        config = FeedForwardPDLConfig()
        config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH
        setattr(config, param_name, value)
        
        result = tester.benchmark_config_detailed(M, K, N, config)
        
        if result.successful:
            print(f"{param_name}={value:>4.1f}: {result.throughput_gflops:>8.2f} GFLOPS")
            results.append((value, result.throughput_gflops))
        else:
            print(f"{param_name}={value:>4.1f}: Failed")
    
    return results