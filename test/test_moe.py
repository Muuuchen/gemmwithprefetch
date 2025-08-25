import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from pathlib import Path
import cutlass_gemm_with_prefetch
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple

# --- Multi-Shape MoE Fusion Benchmark Configuration ---
@dataclass
class ShapeConfig:
    """形状配置类"""
    name: str
    batch_size: int
    seq_len: int
    hidden_size: int
    expert_size: int
    
    @property
    def description(self):
        return f"{self.name}: B{self.batch_size}_S{self.seq_len}_H{self.hidden_size}_E{self.expert_size}"

# 定义多种测试尺寸配置
SHAPE_CONFIGS = [
    # 小规模测试 (适合快速验证)
    ShapeConfig("Tiny", 4, 512, 256, 1024),
    ShapeConfig("Small", 8, 1024, 512, 2048),
    
    # 中等规模测试 (常见应用场景)
    ShapeConfig("Medium", 16, 2048, 768, 3072),
    ShapeConfig("MediumLarge", 32, 2048, 1024, 4096),
    
    # 大规模测试 (高性能场景)
    ShapeConfig("Large", 64, 4096, 1536, 6144),
    ShapeConfig("XLarge", 128, 4096, 2048, 8192),
    
    # 特殊比例测试
    ShapeConfig("WideHidden", 16, 1024, 2048, 4096),  # 较大的hidden_size
    ShapeConfig("WideExpert", 16, 1024, 512, 4096),   # 较大的expert_size
    ShapeConfig("LongSeq", 8, 8192, 768, 3072),       # 长序列
    ShapeConfig("BatchHeavy", 256, 512, 512, 2048),   # 大batch
]

# 数据类型配置
DTYPE_A = torch.float8_e4m3fn
DTYPE_B = torch.float8_e5m2
DTYPE_C = torch.float8_e4m3fn
DEVICE = "cuda"

# 测试配置 - 可根据需要调整
WARMUP_ITER = 5
TIMING_ITER = 20

print(f"--- Multi-Shape MoE Fusion Benchmark ---")
print(f"Device: {torch.cuda.get_device_name(DEVICE)}")
print(f"Data Types: A={DTYPE_A}, B={DTYPE_B}, C={DTYPE_C}")
print(f"Warm-up Iterations: {WARMUP_ITER}")
print(f"Timing Iterations: {TIMING_ITER}")
print(f"Total Shape Configurations: {len(SHAPE_CONFIGS)}\n")

class MultiShapeMoEBenchmark:
    """多形状MoE融合操作性能测试"""
    
    def __init__(self):
        self.results = []
        self.device = DEVICE
        self.shape_results = {}  # 按shape组织的结果
        
        # 获取层次结构枚举
        self.hierarchies = [
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE,
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.PDL,
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH,
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.SHAREDMEM
        ]
        
        # 定义测试参数 (针对多shape测试，使用较少的参数组合)
        self.mm_overlap_values = [-1.0,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9,1.0]
        self.mm_prefetch_values = [-1.0,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9,1.0]
        self.rmsnorm_prefetch_base = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9,1.0]
    
    def create_moe_tensors(self, shape_config: ShapeConfig):
        """为指定的shape配置创建MoE测试张量"""
        batch_size = shape_config.batch_size
        seq_len = shape_config.seq_len
        hidden_size = shape_config.hidden_size
        expert_size = shape_config.expert_size
        
        try:
            # 第一个MM操作: input -> expert_up
            A1 = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.bfloat16, device=self.device).to(DTYPE_A)
            B1 = torch.randn(hidden_size, expert_size, dtype=torch.bfloat16, device=self.device).to(DTYPE_B)
            C1 = torch.empty(batch_size * seq_len, expert_size, dtype=DTYPE_C, device=self.device)
            D1 = torch.randn(batch_size * seq_len, expert_size, dtype=torch.bfloat16, device=self.device).to(DTYPE_C)
            
            # RMSNorm操作
            E = torch.empty(batch_size * seq_len, expert_size, dtype=DTYPE_C, device=self.device)
            F = torch.randn(expert_size, dtype=torch.bfloat16, device=self.device).to(DTYPE_C)
            
            # 第二个MM操作: expert_down -> output
            A2 = torch.empty(batch_size * seq_len, expert_size, dtype=DTYPE_A, device=self.device)
            B2 = torch.randn(expert_size, hidden_size, dtype=torch.bfloat16, device=self.device).to(DTYPE_B)
            C2 = torch.empty(batch_size * seq_len, hidden_size, dtype=DTYPE_C, device=self.device)
            D2 = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.bfloat16, device=self.device).to(DTYPE_C)
            
            tensors = {
                'mm1': {'A': A1, 'B': B1, 'C': C1, 'D': D1},
                'rmsnorm': {'E': E, 'F': F},
                'mm2': {'A': A2, 'B': B2, 'C': C2, 'D': D2}
            }
            
            # 计算内存使用量
            total_memory_mb = sum([
                t.numel() * t.element_size() for tensors_dict in tensors.values() 
                for t in tensors_dict.values()
            ]) / (1024 ** 2)
            
            print(f"  Created tensors for {shape_config.description}")
            print(f"  Memory usage: {total_memory_mb:.1f} MB")
            
            return tensors, total_memory_mb
            
        except Exception as e:
            print(f"  Failed to create tensors for {shape_config.description}: {e}")
            return None, 0
    
    def benchmark_moe_fusion_for_shape(self, shape_config: ShapeConfig, tensors, 
                                     mm1_overlap, mm1_prefetch, 
                                     rmsnorm_prefetch, rmsnorm_hierarchy, 
                                     mm2_overlap, mm2_prefetch):
        """对特定形状执行MoE融合测试"""
        
        def moe_fusion_forward():
            # 第一个矩阵乘法
            C1 = cutlass_gemm_with_prefetch.mm(
                tensors['mm1']['A'], tensors['mm1']['B'], 
                tensors['mm1']['C'].clone(), tensors['mm1']['D'], 
                mm1_overlap, mm1_prefetch
            )
            
            # RMSNorm
            E = cutlass_gemm_with_prefetch.rmsnorm(
                tensors['rmsnorm']['E'].clone(), C1, tensors['rmsnorm']['F'], 
                rmsnorm_prefetch, rmsnorm_hierarchy
            )
            
            # 第二个矩阵乘法
            C2 = cutlass_gemm_with_prefetch.mm(
                E, tensors['mm2']['B'], 
                tensors['mm2']['C'].clone(), tensors['mm2']['D'], 
                mm2_overlap, mm2_prefetch
            )
            
            return C1, E, C2
        
        # 预热
        try:
            for _ in range(WARMUP_ITER):
                _ = moe_fusion_forward()
            torch.cuda.synchronize()
        except Exception as e:
            return None, None, None, None
        
        # 精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        try:
            start_event.record()
            for _ in range(TIMING_ITER):
                _ = moe_fusion_forward()
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_time_ms = elapsed_time_ms / TIMING_ITER
            
            # 计算性能指标
            total_bytes = self.calculate_moe_throughput(tensors)
            throughput_gb_s = (total_bytes / (1024**3)) / (avg_time_ms / 1000)
            
            total_flops = self.calculate_moe_flops(shape_config)
            throughput_tflops = (total_flops / 1e12) / (avg_time_ms / 1000)
            
            return avg_time_ms, throughput_gb_s, throughput_tflops, total_flops
            
        except Exception as e:
            return None, None, None, None
    
    def calculate_moe_throughput(self, tensors):
        """计算总数据传输量"""
        total_bytes = 0
        for tensors_dict in tensors.values():
            for tensor in tensors_dict.values():
                total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes
    
    def calculate_moe_flops(self, shape_config: ShapeConfig):
        """计算总浮点运算量"""
        batch_tokens = shape_config.batch_size * shape_config.seq_len
        hidden_size = shape_config.hidden_size
        expert_size = shape_config.expert_size
        
        # MM1: (batch_tokens, hidden) x (hidden, expert)
        mm1_flops = 2 * batch_tokens * hidden_size * expert_size
        
        # RMSNorm: 大约每个元素5次操作
        rmsnorm_flops = 5 * batch_tokens * expert_size
        
        # MM2: (batch_tokens, expert) x (expert, hidden)
        mm2_flops = 2 * batch_tokens * expert_size * hidden_size
        
        return mm1_flops + rmsnorm_flops + mm2_flops
    
    def test_single_shape(self, shape_config: ShapeConfig):
        """测试单个形状配置"""
        print(f"\n{'='*70}")
        print(f"Testing Shape: {shape_config.description}")
        print(f"{'='*70}")
        
        # 创建张量
        tensors, memory_mb = self.create_moe_tensors(shape_config)
        if tensors is None:
            print(f"Skipping {shape_config.name} due to tensor creation failure")
            return []
        
        shape_results = []
        test_count = 0
        successful_tests = 0
        
        # 测试不同hierarchy
        for hierarchy in self.hierarchies:
            print(f"\n--- Testing Hierarchy: {hierarchy.name} ---")
            
            # 根据hierarchy调整rmsnorm prefetch测试范围
            if hierarchy == cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH:
                rmsnorm_test_values = [0.2, 0.5, 0.8]  # 减少测试点
            else:
                rmsnorm_test_values = self.rmsnorm_prefetch_base
            
            for rmsnorm_prefetch in rmsnorm_test_values:
                for mm1_overlap, mm1_prefetch in itertools.product(self.mm_overlap_values, self.mm_prefetch_values):
                    for mm2_overlap, mm2_prefetch in itertools.product(self.mm_overlap_values, self.mm_prefetch_values):
                        test_count += 1
                        
                        avg_time, throughput_gb, throughput_tflops, total_flops = self.benchmark_moe_fusion_for_shape(
                            shape_config, tensors, mm1_overlap, mm1_prefetch, 
                            rmsnorm_prefetch, hierarchy, mm2_overlap, mm2_prefetch
                        )
                        
                        if avg_time is not None:
                            successful_tests += 1
                            result = {
                                'shape_name': shape_config.name,
                                'batch_size': shape_config.batch_size,
                                'seq_len': shape_config.seq_len,
                                'hidden_size': shape_config.hidden_size,
                                'expert_size': shape_config.expert_size,
                                'total_tokens': shape_config.batch_size * shape_config.seq_len,
                                'memory_mb': memory_mb,
                                'total_flops': total_flops,
                                'hierarchy': hierarchy.name,
                                'mm1_overlap': mm1_overlap,
                                'mm1_prefetch': mm1_prefetch,
                                'rmsnorm_prefetch': rmsnorm_prefetch,
                                'mm2_overlap': mm2_overlap,
                                'mm2_prefetch': mm2_prefetch,
                                'avg_time_ms': avg_time,
                                'throughput_gb_s': throughput_gb,
                                'throughput_tflops': throughput_tflops,
                                # 额外的性能指标
                                'tokens_per_second': (shape_config.batch_size * shape_config.seq_len * 1000) / avg_time,
                                'flops_per_byte': total_flops / (memory_mb * 1024 * 1024),
                                'arithmetic_intensity': total_flops / (memory_mb * 1024 * 1024 * 8)  # 假设8字节数据
                            }
                            shape_results.append(result)
                            
                            if test_count % 10 == 0:  # 每10个测试打印一次进度
                                print(f"    Progress: {test_count} tests, {successful_tests} successful")
        
        print(f"\nShape {shape_config.name} complete: {successful_tests}/{test_count} tests successful")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        return shape_results
    
    def run_multi_shape_benchmark(self, selected_shapes=None):
        """运行多形状测试"""
        print("="*80)
        print("MULTI-SHAPE MoE FUSION BENCHMARK")
        print("="*80)
        
        # 选择要测试的shapes
        if selected_shapes is None:
            test_shapes = SHAPE_CONFIGS
        else:
            test_shapes = [cfg for cfg in SHAPE_CONFIGS if cfg.name in selected_shapes]
        
        print(f"Testing {len(test_shapes)} shape configurations:")
        for shape in test_shapes:
            print(f"  - {shape.description}")
        print()
        
        # GPU内存检查
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU memory: {total_memory:.1f} GB\n")
        
        all_results = []
        
        for i, shape_config in enumerate(test_shapes, 1):
            print(f"\n[{i}/{len(test_shapes)}] Starting shape: {shape_config.name}")
            
            try:
                shape_results = self.test_single_shape(shape_config)
                if shape_results:
                    all_results.extend(shape_results)
                    self.shape_results[shape_config.name] = shape_results
                    
                    # 打印该shape的最佳结果
                    best_result = min(shape_results, key=lambda x: x['avg_time_ms'])
                    print(f"  Best result for {shape_config.name}: {best_result['avg_time_ms']:.3f} ms, "
                          f"{best_result['throughput_tflops']:.2f} TFLOPS")
                else:
                    print(f"  No successful tests for {shape_config.name}")
                    
            except Exception as e:
                print(f"  Error testing {shape_config.name}: {e}")
                continue
        
        self.results = all_results
        print(f"\n{'='*80}")
        print(f"MULTI-SHAPE BENCHMARK COMPLETE")
        print(f"Total successful tests: {len(all_results)}")
        print(f"Shapes tested: {len(self.shape_results)}")
        print("="*80)
        
        return all_results

class MultiShapeResultAnalyzer:
    """多形状结果分析器"""
    
    def __init__(self, results, shape_results=None):
        self.results = results
        self.shape_results = shape_results or {}
        self.df = pd.DataFrame(results) if results else pd.DataFrame()
        
        # 创建结果目录
        self.output_dir = Path("multi_shape_moe_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_scaling_behavior(self):
        """分析扩展性行为"""
        if self.df.empty:
            return {}
        
        scaling_analysis = {}
        
        # 按shape分组分析
        shape_stats = self.df.groupby('shape_name').agg({
            'avg_time_ms': ['min', 'mean', 'max'],
            'throughput_tflops': ['min', 'mean', 'max'],
            'total_tokens': 'first',
            'total_flops': 'first',
            'memory_mb': 'first'
        }).round(4)
        
        scaling_analysis['shape_statistics'] = shape_stats
        
        # 分析随tokens数量的扩展性
        token_scaling = self.df.groupby('total_tokens')['avg_time_ms'].min().reset_index()
        token_scaling = token_scaling.sort_values('total_tokens')
        scaling_analysis['token_scaling'] = token_scaling
        
        # 分析计算密度
        self.df['compute_density'] = self.df['total_flops'] / self.df['avg_time_ms']
        density_by_shape = self.df.groupby('shape_name')['compute_density'].max()
        scaling_analysis['compute_density'] = density_by_shape
        
        return scaling_analysis
    
    def plot_comprehensive_analysis(self):
        """绘制综合分析图表"""
        if self.df.empty:
            print("No results to plot!")
            return
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(24, 18))
        
        # 1. 不同shape的性能对比 (2x2)
        ax1 = plt.subplot(3, 4, 1)
        shape_perf = self.df.groupby('shape_name')['avg_time_ms'].min()
        bars = ax1.bar(range(len(shape_perf)), shape_perf.values)
        ax1.set_xlabel('Shape Configuration')
        ax1.set_ylabel('Best Latency (ms)')
        ax1.set_title('Best Performance by Shape')
        ax1.set_xticks(range(len(shape_perf)))
        ax1.set_xticklabels(shape_perf.index, rotation=45, ha='right')
        
        # 为每个柱子添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 吞吐量对比
        ax2 = plt.subplot(3, 4, 2)
        shape_throughput = self.df.groupby('shape_name')['throughput_tflops'].max()
        bars2 = ax2.bar(range(len(shape_throughput)), shape_throughput.values, color='orange')
        ax2.set_xlabel('Shape Configuration')
        ax2.set_ylabel('Peak Throughput (TFLOPS)')
        ax2.set_title('Peak Throughput by Shape')
        ax2.set_xticks(range(len(shape_throughput)))
        ax2.set_xticklabels(shape_throughput.index, rotation=45, ha='right')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 扩展性分析 - tokens vs latency
        ax3 = plt.subplot(3, 4, 3)
        token_latency = self.df.groupby(['shape_name', 'total_tokens'])['avg_time_ms'].min().reset_index()
        for shape in token_latency['shape_name'].unique():
            shape_data = token_latency[token_latency['shape_name'] == shape]
            ax3.plot(shape_data['total_tokens'], shape_data['avg_time_ms'], 'o-', label=shape, alpha=0.7)
        ax3.set_xlabel('Total Tokens')
        ax3.set_ylabel('Best Latency (ms)')
        ax3.set_title('Scaling: Tokens vs Latency')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. 效率分析 - tokens per second
        ax4 = plt.subplot(3, 4, 4)
        efficiency = self.df.groupby('shape_name')['tokens_per_second'].max()
        bars4 = ax4.bar(range(len(efficiency)), efficiency.values, color='green')
        ax4.set_xlabel('Shape Configuration')
        ax4.set_ylabel('Peak Tokens/Second')
        ax4.set_title('Token Processing Efficiency')
        ax4.set_xticks(range(len(efficiency)))
        ax4.set_xticklabels(efficiency.index, rotation=45, ha='right')
        
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        # 5. 内存效率热力图
        ax5 = plt.subplot(3, 4, 5)
        memory_eff = self.df.pivot_table(
            values='throughput_tflops', 
            index='shape_name', 
            columns='hierarchy',
            aggfunc='max'
        )
        sns.heatmap(memory_eff, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5)
        ax5.set_title('Throughput by Shape and Hierarchy')
        ax5.set_xlabel('Hierarchy')
        ax5.set_ylabel('Shape')
        
        # 6. 算术强度分析
        ax6 = plt.subplot(3, 4, 6)
        arithmetic_intensity = self.df.groupby('shape_name')['arithmetic_intensity'].mean()
        bars6 = ax6.bar(range(len(arithmetic_intensity)), arithmetic_intensity.values, color='purple')
        ax6.set_xlabel('Shape Configuration')
        ax6.set_ylabel('Arithmetic Intensity (FLOP/Byte)')
        ax6.set_title('Compute vs Memory Intensity')
        ax6.set_xticks(range(len(arithmetic_intensity)))
        ax6.set_xticklabels(arithmetic_intensity.index, rotation=45, ha='right')
        
        # 7. 性能分布箱线图
        ax7 = plt.subplot(3, 4, 7)
        self.df.boxplot(column='avg_time_ms', by='shape_name', ax=ax7)
        ax7.set_xlabel('Shape Configuration')
        ax7.set_ylabel('Latency (ms)')
        ax7.set_title('Latency Distribution by Shape')
        ax7.tick_params(axis='x', rotation=45)
        plt.sca(ax7)
        plt.xticks(rotation=45, ha='right')
        
        # 8. 参数敏感性分析
        ax8 = plt.subplot(3, 4, 8)
        param_sensitivity = self.df.groupby(['hierarchy', 'rmsnorm_prefetch'])['avg_time_ms'].mean().reset_index()
        pivot_sensitivity = param_sensitivity.pivot(index='hierarchy', columns='rmsnorm_prefetch', values='avg_time_ms')
        sns.heatmap(pivot_sensitivity, annot=True, fmt='.2f', cmap='viridis_r', ax=ax8)
        ax8.set_title('Parameter Sensitivity Heatmap')
        ax8.set_xlabel('RMSNorm Prefetch')
        ax8.set_ylabel('Hierarchy')
        
        # 9. 最优配置分布
        ax9 = plt.subplot(3, 4, 9)
        best_configs = self.df.loc[self.df.groupby('shape_name')['avg_time_ms'].idxmin()]
        hierarchy_counts = best_configs['hierarchy'].value_counts()
        ax9.pie(hierarchy_counts.values, labels=hierarchy_counts.index, autopct='%1.1f%%')
        ax9.set_title('Best Hierarchy Distribution')
        
        # 10. 性能改进分析
        ax10 = plt.subplot(3, 4, 10)
        improvements = []
        for shape in self.df['shape_name'].unique():
            shape_data = self.df[self.df['shape_name'] == shape]
            if len(shape_data) > 1:
                best = shape_data['avg_time_ms'].min()
                worst = shape_data['avg_time_ms'].max()
                improvement = (worst - best) / worst * 100
                improvements.append((shape, improvement))
        
        if improvements:
            shapes, impr_values = zip(*improvements)
            bars10 = ax10.bar(range(len(shapes)), impr_values, color='red', alpha=0.7)
            ax10.set_xlabel('Shape Configuration')
            ax10.set_ylabel('Performance Improvement (%)')
            ax10.set_title('Max Performance Improvement')
            ax10.set_xticks(range(len(shapes)))
            ax10.set_xticklabels(shapes, rotation=45, ha='right')
        
        # 11. 内存使用vs性能
        ax11 = plt.subplot(3, 4, 11)
        scatter = ax11.scatter(self.df['memory_mb'], self.df['avg_time_ms'], 
                              c=self.df['total_tokens'], cmap='plasma', alpha=0.6)
        ax11.set_xlabel('Memory Usage (MB)')
        ax11.set_ylabel('Latency (ms)')
        ax11.set_title('Memory vs Performance')
        plt.colorbar(scatter, ax=ax11, label='Total Tokens')
        ax11.set_xscale('log')
        ax11.set_yscale('log')
        
        # 12. 综合性能排名
        ax12 = plt.subplot(3, 4, 12)
        # 计算综合得分：考虑延迟和吞吐量
        self.df['composite_score'] = (
            (1 / self.df['avg_time_ms']) * 0.5 + 
            (self.df['throughput_tflops'] / self.df['throughput_tflops'].max()) * 0.5
        )
        top_configs = self.df.nlargest(10, 'composite_score')[['shape_name', 'hierarchy', 'composite_score']]
        y_pos = np.arange(len(top_configs))
        bars12 = ax12.barh(y_pos, top_configs['composite_score'].values, color='gold')
        ax12.set_yticks(y_pos)
        ax12.set_yticklabels([f"{row['shape_name'][:6]}-{row['hierarchy'][:4]}" 
                             for _, row in top_configs.iterrows()], fontsize=8)
        ax12.set_xlabel('Composite Score')
        ax12.set_title('Top 10 Configurations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multi_shape_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_shape_comparison_report(self):
        """生成形状对比报告"""
        if self.df.empty:
            return
        
        # 按shape统计最佳性能
        shape_summary = self.df.groupby('shape_name').agg({
            'avg_time_ms': 'min',
            'throughput_tflops': 'max',
            'tokens_per_second': 'max',
                        'total_tokens': 'first',
            'memory_mb': 'first',
            'arithmetic_intensity': 'mean'
        }).round(4)
        
        shape_summary.columns = ['Best_Latency_ms', 'Peak_Throughput_TFLOPS', 
                               'Peak_Tokens_per_sec', 'Total_Tokens', 
                               'Memory_MB', 'Avg_Arithmetic_Intensity']
        
        # 添加效率指标
        shape_summary['Latency_per_Token_us'] = (shape_summary['Best_Latency_ms'] * 1000) / shape_summary['Total_Tokens']
        shape_summary['Memory_Efficiency_TFLOPS_per_GB'] = shape_summary['Peak_Throughput_TFLOPS'] / (shape_summary['Memory_MB'] / 1024)
        
        # 保存报告
        with open(self.output_dir / 'shape_comparison_report.txt', 'w') as f:
            f.write("Multi-Shape MoE Fusion Performance Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total configurations tested: {len(self.df)}\n")
            f.write(f"Number of shapes: {len(shape_summary)}\n")
            f.write(f"Best overall latency: {self.df['avg_time_ms'].min():.3f} ms\n")
            f.write(f"Best overall throughput: {self.df['throughput_tflops'].max():.2f} TFLOPS\n")
            f.write(f"Memory range: {self.df['memory_mb'].min():.1f} - {self.df['memory_mb'].max():.1f} MB\n\n")
            
            f.write("SHAPE-BY-SHAPE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for shape_name, row in shape_summary.iterrows():
                f.write(f"\n{shape_name.upper()}:\n")
                f.write(f"  Best Latency: {row['Best_Latency_ms']:.3f} ms\n")
                f.write(f"  Peak Throughput: {row['Peak_Throughput_TFLOPS']:.2f} TFLOPS\n")
                f.write(f"  Peak Token Rate: {row['Peak_Tokens_per_sec']:.0f} tokens/sec\n")
                f.write(f"  Total Tokens: {row['Total_Tokens']:,}\n")
                f.write(f"  Memory Usage: {row['Memory_MB']:.1f} MB\n")
                f.write(f"  Latency per Token: {row['Latency_per_Token_us']:.2f} μs/token\n")
                f.write(f"  Memory Efficiency: {row['Memory_Efficiency_TFLOPS_per_GB']:.2f} TFLOPS/GB\n")
            
            # 最佳配置
            f.write(f"\nBEST CONFIGURATIONS\n")
            f.write("-" * 30 + "\n")
            
            best_latency_config = self.df.loc[self.df['avg_time_ms'].idxmin()]
            f.write(f"\nBest Latency Configuration:\n")
            f.write(f"  Shape: {best_latency_config['shape_name']}\n")
            f.write(f"  Hierarchy: {best_latency_config['hierarchy']}\n")
            f.write(f"  MM1 Overlap: {best_latency_config['mm1_overlap']}\n")
            f.write(f"  MM1 Prefetch: {best_latency_config['mm1_prefetch']}\n")
            f.write(f"  RMSNorm Prefetch: {best_latency_config['rmsnorm_prefetch']}\n")
            f.write(f"  MM2 Overlap: {best_latency_config['mm2_overlap']}\n")
            f.write(f"  MM2 Prefetch: {best_latency_config['mm2_prefetch']}\n")
            f.write(f"  Result: {best_latency_config['avg_time_ms']:.3f} ms\n")
            
            best_throughput_config = self.df.loc[self.df['throughput_tflops'].idxmax()]
            f.write(f"\nBest Throughput Configuration:\n")
            f.write(f"  Shape: {best_throughput_config['shape_name']}\n")
            f.write(f"  Hierarchy: {best_throughput_config['hierarchy']}\n")
            f.write(f"  Result: {best_throughput_config['throughput_tflops']:.2f} TFLOPS\n")
            
            # 扩展性分析
            f.write(f"\nSCALING ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            # 按token数量排序
            token_analysis = shape_summary.sort_values('Total_Tokens')
            f.write("Scaling with Token Count:\n")
            for shape_name, row in token_analysis.iterrows():
                efficiency = row['Peak_Throughput_TFLOPS'] / (row['Total_Tokens'] / 1000)
                f.write(f"  {shape_name}: {row['Total_Tokens']:,} tokens -> "
                       f"{row['Best_Latency_ms']:.3f} ms "
                       f"(Efficiency: {efficiency:.4f} TFLOPS/K-tokens)\n")
        
        print(f"Shape comparison report saved to {self.output_dir / 'shape_comparison_report.txt'}")
        return shape_summary
    
    def save_results(self):
        """保存所有结果"""
        if not self.df.empty:
            # 保存完整结果CSV
            self.df.to_csv(self.output_dir / 'multi_shape_results.csv', index=False)
            
            # 保存每个shape的最佳结果
            best_results = self.df.loc[self.df.groupby('shape_name')['avg_time_ms'].idxmin()]
            best_results.to_csv(self.output_dir / 'best_results_by_shape.csv', index=False)
            
            # 保存统计摘要
            summary_stats = self.df.groupby('shape_name').agg({
                'avg_time_ms': ['min', 'mean', 'max', 'std'],
                'throughput_tflops': ['min', 'mean', 'max', 'std'],
                'total_tokens': 'first',
                'memory_mb': 'first'
            }).round(4)
            summary_stats.to_csv(self.output_dir / 'shape_statistics.csv')
            
            print(f"All results saved to {self.output_dir}")
        else:
            print("No results to save!")

def create_shape_selection_menu():
    """创建形状选择菜单"""
    print("\nAvailable Shape Configurations:")
    print("=" * 50)
    
    categories = {
        "Quick Test": ["Tiny", "Small"],
        "Standard Test": ["Medium", "MediumLarge"],
        "Performance Test": ["Large", "XLarge"],
        "Special Cases": ["WideHidden", "WideExpert", "LongSeq", "BatchHeavy"]
    }
    
    for category, shapes in categories.items():
        print(f"\n{category}:")
        for shape_name in shapes:
            config = next(cfg for cfg in SHAPE_CONFIGS if cfg.name == shape_name)
            print(f"  {shape_name}: {config.description}")
    
    print(f"\nOptions:")
    print("1. Quick Test (Tiny, Small)")
    print("2. Standard Test (Medium, MediumLarge)")  
    print("3. Performance Test (Large, XLarge)")
    print("4. Special Cases (WideHidden, WideExpert, LongSeq, BatchHeavy)")
    print("5. All Shapes")
    print("6. Custom Selection")
    
    while True:
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                return ["Tiny", "Small"]
            elif choice == "2":
                return ["Medium", "MediumLarge"]
            elif choice == "3":
                return ["Large", "XLarge"]
            elif choice == "4":
                return ["WideHidden", "WideExpert", "LongSeq", "BatchHeavy"]
            elif choice == "5":
                return None  # All shapes
            elif choice == "6":
                available_shapes = [cfg.name for cfg in SHAPE_CONFIGS]
                print(f"\nAvailable shapes: {', '.join(available_shapes)}")
                custom_input = input("Enter shape names (comma-separated): ").strip()
                custom_shapes = [s.strip() for s in custom_input.split(",")]
                valid_shapes = [s for s in custom_shapes if s in available_shapes]
                if valid_shapes:
                    return valid_shapes
                else:
                    print("No valid shapes selected!")
                    continue
            else:
                print("Invalid choice! Please enter 1-6.")
                continue
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return []

def estimate_test_time(selected_shapes):
    """估算测试时间"""
    if selected_shapes is None:
        selected_shapes = [cfg.name for cfg in SHAPE_CONFIGS]
    
    num_shapes = len(selected_shapes)
    
    # 估算参数：每个shape大约测试32个配置组合，每个配置预热+测试约2秒
    configs_per_shape = 32  # 4 hierarchies * 2 rmsnorm * 2 mm1_overlap * 2 mm2_overlap
    time_per_config = 2  # 秒
    
    total_time_seconds = num_shapes * configs_per_shape * time_per_config
    total_time_minutes = total_time_seconds / 60
    
    print(f"\nEstimated Test Time:")
    print(f"  Shapes to test: {num_shapes}")
    print(f"  Configs per shape: ~{configs_per_shape}")
    print(f"  Total configs: ~{num_shapes * configs_per_shape}")
    print(f"  Estimated time: ~{total_time_minutes:.1f} minutes ({total_time_seconds/3600:.1f} hours)")
    
    return total_time_minutes

def main():
    """主测试函数"""
    print("="*80)
    print("MULTI-SHAPE MoE FUSION BENCHMARK")
    print("="*80)
    
    # 检查GPU内存
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {total_memory_gb:.1f} GB")
        
        if total_memory_gb < 8:
            print("⚠️  Warning: Limited GPU memory detected. Consider using smaller shapes.")
    else:
        print("❌ CUDA not available!")
        return
    
    # 形状选择
    selected_shapes = create_shape_selection_menu()
    if not selected_shapes and selected_shapes != []:  # 空列表表示用户取消
        if selected_shapes is None:
            print("Testing all shapes...")
        else:
            print("No shapes selected. Exiting.")
            return
    elif selected_shapes == []:
        return
    
    # 估算测试时间
    estimated_time = estimate_test_time(selected_shapes)
    
    if estimated_time > 30:  # 超过30分钟
        confirm = input(f"\n⚠️  This test will take approximately {estimated_time:.1f} minutes. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Test cancelled.")
            return
    
    # 创建并运行基准测试
    print(f"\n{'='*80}")
    print("STARTING BENCHMARK")
    print("="*80)
    
    benchmark = MultiShapeMoEBenchmark()
    results = benchmark.run_multi_shape_benchmark(selected_shapes)
    
    if results:
        print(f"\n{'='*80}")
        print("ANALYZING RESULTS")
        print("="*80)
        
        # 分析结果
        analyzer = MultiShapeResultAnalyzer(results, benchmark.shape_results)
        
        # 生成综合分析
        print("Generating comprehensive analysis plots...")
        analyzer.plot_comprehensive_analysis()
        
        # 生成对比报告
        print("Generating shape comparison report...")
        shape_summary = analyzer.generate_shape_comparison_report()
        
        # 保存所有结果
        print("Saving results...")
        analyzer.save_results()
        
        # 扩展性分析
        scaling_analysis = analyzer.analyze_scaling_behavior()
        
        # 打印核心结果摘要
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print("="*80)
        
        df = pd.DataFrame(results)
        
        print(f"\n📊 PERFORMANCE STATISTICS:")
        print(f"  • Best Latency: {df['avg_time_ms'].min():.3f} ms")
        print(f"  • Best Throughput: {df['throughput_tflops'].max():.2f} TFLOPS")
        print(f"  • Memory Range: {df['memory_mb'].min():.0f} - {df['memory_mb'].max():.0f} MB")
        print(f"  • Token Range: {df['total_tokens'].min():,} - {df['total_tokens'].max():,}")
        
        print(f"\n🏆 BEST CONFIGURATIONS:")
        best_latency = df.loc[df['avg_time_ms'].idxmin()]
        best_throughput = df.loc[df['throughput_tflops'].idxmax()]
        
        print(f"  • Best Latency: {best_latency['shape_name']} with {best_latency['hierarchy']} hierarchy")
        print(f"    → {best_latency['avg_time_ms']:.3f} ms ({best_latency['throughput_tflops']:.2f} TFLOPS)")
        
        print(f"  • Best Throughput: {best_throughput['shape_name']} with {best_throughput['hierarchy']} hierarchy")
        print(f"    → {best_throughput['throughput_tflops']:.2f} TFLOPS ({best_throughput['avg_time_ms']:.3f} ms)")
        
        # 按shape显示最佳结果
        print(f"\n📈 BEST RESULT BY SHAPE:")
        shape_bests = df.loc[df.groupby('shape_name')['avg_time_ms'].idxmin()]
        for _, row in shape_bests.iterrows():
            tokens_per_ms = row['total_tokens'] / row['avg_time_ms']
            print(f"  • {row['shape_name']:<12}: {row['avg_time_ms']:>6.3f} ms | "
                  f"{row['throughput_tflops']:>5.2f} TFLOPS | "
                  f"{tokens_per_ms:>8.0f} tokens/ms")
        
        # 层次结构效果分析
        print(f"\n🔧 HIERARCHY EFFECTIVENESS:")
        hierarchy_stats = df.groupby('hierarchy').agg({
            'avg_time_ms': 'min',
            'throughput_tflops': 'max'
        }).round(3)
        
        for hierarchy, stats in hierarchy_stats.iterrows():
            print(f"  • {hierarchy:<12}: {stats['avg_time_ms']:>6.3f} ms (best) | "
                  f"{stats['throughput_tflops']:>5.2f} TFLOPS (peak)")
        
        print(f"\n💾 Results saved to: {analyzer.output_dir}")
        print(f"📈 Analysis plots generated")
        print(f"📋 Detailed report available")
        
    else:
        print("\n❌ No successful benchmark results obtained!")
        print("Please check:")
        print("  • GPU memory availability")
        print("  • CUDA kernel implementation")
        print("  • Input tensor compatibility")

if __name__ == "__main__":
    main()