import torch
import torch.nn as nn
import cutlass_gemm_with_prefetch
import numpy as np
import itertools
from dataclasses import dataclass
import pandas as pd
import os
from datetime import datetime
import random

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 结果保存目录
RESULTS_DIR = "decoder_layer_optimization_results"

def ensure_results_dir():
    """确保结果目录存在"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir)
    return run_dir

@dataclass
class DecoderLayerConfig:
    """Decoder Layer配置参数"""
    # Attention部分
    qkv_overlap: float
    qkv_prefetch: float
    output_overlap: float
    output_prefetch: float
    # FFN部分
    ffn1_overlap: float
    ffn1_prefetch: float
    ffn2_overlap: float
    ffn2_prefetch: float
    
    def to_dict(self):
        return {
            'qkv_overlap': self.qkv_overlap,
            'qkv_prefetch': self.qkv_prefetch,
            'output_overlap': self.output_overlap,
            'output_prefetch': self.output_prefetch,
            'ffn1_overlap': self.ffn1_overlap,
            'ffn1_prefetch': self.ffn1_prefetch,
            'ffn2_overlap': self.ffn2_overlap,
            'ffn2_prefetch': self.ffn2_prefetch,
        }

class DecoderLayerBenchmark:
    """Decoder Layer性能测试 - 简化版本，不使用RMSNorm"""
    
    def __init__(self, d_model: int = 768, n_heads: int = 12, d_ff: int = 3072,
                 batch_size: int = 1, seq_len: int = 128):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = 'cuda'
        
        # 初始化权重
        self._init_weights()
        
        # 定义参数空间
        self.mm_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -1.0]
        
    def _init_weights(self):
        """初始化权重"""
        scale = 0.02
        
        # Attention权重
        self.W_qkv = (torch.randn(self.d_model, 3 * self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_qkv = torch.zeros(3 * self.d_model, device=self.device).to(torch.float8_e4m3fn)
        self.W_o = (torch.randn(self.d_model, self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_o = torch.zeros(self.d_model, device=self.device).to(torch.float8_e4m3fn)
        
        # FFN权重
        self.W_ff1 = (torch.randn(self.d_model, self.d_ff, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_ff1 = torch.zeros(self.d_ff, device=self.device).to(torch.float8_e4m3fn)
        self.W_ff2 = (torch.randn(self.d_ff, self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_ff2 = torch.zeros(self.d_model, device=self.device).to(torch.float8_e4m3fn)
    
    def decoder_layer_forward(self, x: torch.Tensor, config: DecoderLayerConfig) -> torch.Tensor:
        """执行完整的decoder layer前向传播 - 不使用RMSNorm"""
        batch_size, seq_len, _ = x.shape
        
        # 保存第一个residual
        residual1 = x.to(torch.float8_e4m3fn)
        
        # 1. Attention部分
        # QKV Projection
        x_flat = residual1.reshape(-1, self.d_model)
        qkv_output = torch.zeros(batch_size * seq_len, 3 * self.d_model,
                                device=x.device, dtype=torch.float8_e4m3fn)
        
        qkv = cutlass_gemm_with_prefetch.mm(
            x_flat,
            self.W_qkv,
            qkv_output,
            self.b_qkv.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.qkv_overlap,
            config.qkv_prefetch
        )
        
        # Reshape for attention
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0, :, :].contiguous().to(torch.float16)
        k = qkv[:, :, 1, :, :].contiguous().to(torch.float16)
        v = qkv[:, :, 2, :, :].contiguous().to(torch.float16)
        
        # Flash Attention
        attn_output = cutlass_gemm_with_prefetch.fa(q, k, v)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model).to(torch.float8_e4m3fn)
        
        # Output Projection
        attn_flat = attn_output.reshape(-1, self.d_model)
        output = torch.zeros(batch_size * seq_len, self.d_model,
                           device=x.device, dtype=torch.float8_e4m3fn)
        output = cutlass_gemm_with_prefetch.mm(
            attn_flat,
            self.W_o,
            output,
            self.b_o.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.output_overlap,
            config.output_prefetch
        )
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        # First residual connection
        x = cutlass_gemm_with_prefetch.add_residual(output, residual1)
        
        # 保存第二个residual
        residual2 = x
        
        # 2. FFN部分
        # FFN第一层
        x_flat = x.reshape(-1, self.d_model)
        ff1_output = torch.zeros(batch_size * seq_len, self.d_ff,
                               device=x.device, dtype=torch.float8_e4m3fn)
        
        hidden = cutlass_gemm_with_prefetch.mm(
            x_flat,
            self.W_ff1,
            ff1_output,
            self.b_ff1.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.ffn1_overlap,
            config.ffn1_prefetch
        )
        
        # FFN第二层（直接连接，不使用激活函数）
        ff2_output = torch.zeros(batch_size * seq_len, self.d_model,
                               device=x.device, dtype=torch.float8_e4m3fn)
        output = cutlass_gemm_with_prefetch.mm(
            hidden,
            self.W_ff2,
            ff2_output,
            self.b_ff2.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.ffn2_overlap,
            config.ffn2_prefetch
        )
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        # Second residual connection
        x = cutlass_gemm_with_prefetch.add_residual(output, residual2)
        
        return x
    
    def benchmark_config(self, config: DecoderLayerConfig, warmup: int = 10, iterations: int = 50):
        """测试单个配置"""
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, 
                       device=self.device, dtype=torch.float16)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.decoder_layer_forward(x, config)
        torch.cuda.synchronize()
        
        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            torch.cuda.synchronize()
            start_event.record()
            
            for _ in range(iterations):
                output = self.decoder_layer_forward(x, config)
                
            end_event.record()
            torch.cuda.synchronize()
        
        # 计算延迟
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / iterations
        
        return avg_time_ms
    
    def generate_smart_configs(self, n_samples=20000):
        """生成智能采样的配置"""
        configs = []
        
        # 1. 添加关键配置（必须测试的）
        key_configs = [
            # Baseline
            DecoderLayerConfig(-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0),
            # All prefetch
            DecoderLayerConfig(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            # All overlap
            DecoderLayerConfig(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
            # Mixed
            DecoderLayerConfig(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        ]
        configs.extend(key_configs)
        
        # 2. 网格采样 - 使用更粗的网格
        coarse_values = [-1.0, 0.0, 0.5, 1.0]
        grid_configs = []
        
        # 为每个mm操作独立采样
        for qkv_vals in itertools.product(coarse_values, repeat=2):
            for out_vals in itertools.product(coarse_values, repeat=2):
                for ff1_vals in itertools.product(coarse_values, repeat=2):
                    for ff2_vals in itertools.product(coarse_values, repeat=2):
                        if random.random() < 0.1:  # 10%的概率选择
                            grid_configs.append(DecoderLayerConfig(
                                qkv_vals[0], qkv_vals[1],
                                out_vals[0], out_vals[1],
                                ff1_vals[0], ff1_vals[1],
                                ff2_vals[0], ff2_vals[1]
                            ))
        
        configs.extend(grid_configs[:500])  # 最多添加500个网格配置
        
        # 3. 随机采样
        n_random = n_samples - len(configs)
        for _ in range(n_random):
            config = DecoderLayerConfig(
                random.choice(self.mm_values),
                random.choice(self.mm_values),
                random.choice(self.mm_values),
                random.choice(self.mm_values),
                random.choice(self.mm_values),
                random.choice(self.mm_values),
                random.choice(self.mm_values),
                random.choice(self.mm_values)
            )
            configs.append(config)
        
        # 4. 基于经验的配置（某些模式可能更有效）
        experience_configs = []
        for _ in range(200):
            # 倾向于使用相同的overlap/prefetch策略
            strategy = random.choice(['overlap', 'prefetch', 'mixed', 'none'])
            if strategy == 'overlap':
                overlap_val = random.choice([0.5, 0.8, 1.0])
                config = DecoderLayerConfig(
                    overlap_val, 0.0, overlap_val, 0.0,
                    overlap_val, 0.0, overlap_val, 0.0
                )
            elif strategy == 'prefetch':
                prefetch_val = random.choice([0.5, 0.8, 1.0])
                config = DecoderLayerConfig(
                    0.0, prefetch_val, 0.0, prefetch_val,
                    0.0, prefetch_val, 0.0, prefetch_val
                )
            elif strategy == 'mixed':
                val = random.choice([0.2, 0.4, 0.6])
                config = DecoderLayerConfig(
                    val, val, val, val, val, val, val, val
                )
            else:
                config = DecoderLayerConfig(-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0)
            
            experience_configs.append(config)
        
        configs.extend(experience_configs)
        
        # 去重
        unique_configs = []
        seen = set()
        for config in configs:
            key = (config.qkv_overlap, config.qkv_prefetch,
                   config.output_overlap, config.output_prefetch,
                   config.ffn1_overlap, config.ffn1_prefetch,
                   config.ffn2_overlap, config.ffn2_prefetch)
            if key not in seen:
                seen.add(key)
                unique_configs.append(config)
        
        return unique_configs[:n_samples]
    
    def run_benchmark(self, n_samples=2000):
        """运行参数测试"""
        # Baseline配置
        baseline_config = DecoderLayerConfig(
            qkv_overlap=-1.0, qkv_prefetch=0.0,
            output_overlap=-1.0, output_prefetch=0.0,
            ffn1_overlap=-1.0, ffn1_prefetch=0.0,
            ffn2_overlap=-1.0, ffn2_prefetch=0.0,
        )
        
        print("Testing baseline configuration (all mm params: -1.0, 0.0)...")
        baseline_latency = self.benchmark_config(baseline_config, warmup=20, iterations=100)
        print(f"Baseline latency: {baseline_latency:.3f} ms")
        
        # 生成智能采样的配置
        all_configs = self.generate_smart_configs(n_samples)
        print(f"\nTotal configurations to test: {len(all_configs)}")
        
        # 测试所有配置
        results = []
        for i, config in enumerate(all_configs):
            try:
                latency = self.benchmark_config(config)
                speedup = baseline_latency / latency
                
                result = config.to_dict()
                result['latency_ms'] = latency
                result['speedup'] = speedup
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"Progress: {i+1}/{len(all_configs)}")
                    
            except Exception as e:
                print(f"Failed config {i+1}: {e}")
        
        return pd.DataFrame(results), baseline_latency


def test_multiple_shapes():
    """测试多个shape配置"""
    
    shapes = [
        # (batch_size, seq_len, d_model, n_heads, d_ff, name)
        (1,32,256,4,1024,"tiny"),
        (1, 64, 512, 8, 2048, "BERT-base")
        # (1, 128, 768, 12, 3072, "BERT-base"),
        # (1, 512, 768, 12, 3072, "BERT-base-long"),
        # (4, 128, 768, 12, 3072, "BERT-base-batch4"),
        # (1, 128, 1024, 16, 4096, "BERT-large"),
        # (1, 128, 4096, 32, 11008, "LLaMA-7B-like"),
        # (1, 256, 4096, 32, 11008, "LLaMA-7B-long"),
    ]
    
    run_dir = ensure_results_dir()
    summary_results = []
    
    for batch_size, seq_len, d_model, n_heads, d_ff, name in shapes:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"Shape: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
        print(f"{'='*70}")
        
        try:
            # 创建benchmark
            benchmark = DecoderLayerBenchmark(d_model, n_heads, d_ff, batch_size, seq_len)
            
            # 运行测试（使用采样）
            df, baseline_latency = benchmark.run_benchmark(n_samples=20000)
            
            if len(df) == 0:
                print(f"No valid results for {name}")
                continue
            
            # 找出top 10配置
            top10 = df.nsmallest(10, 'latency_ms')
            
            # 保存完整结果
            result_file = os.path.join(run_dir, f"{name.replace('-', '_')}_results.csv")
            df.to_csv(result_file, index=False)
            
            # 打印top 10
            print(f"\nTop 10 configurations for {name}:")
            print(f"Baseline latency: {baseline_latency:.3f} ms (all mm: -1.0, 0.0)")
            print("\nRank | Latency(ms) | Speedup | QKV(o,p) | Output(o,p) | FFN1(o,p) | FFN2(o,p)")
            print("-" * 80)
            
            for idx, (i, row) in enumerate(top10.iterrows()):
                print(f"{idx+1:4d} | {row['latency_ms']:11.3f} | {row['speedup']:7.2f}x | "
                      f"({row['qkv_overlap']:3.1f},{row['qkv_prefetch']:3.1f}) | "
                      f"({row['output_overlap']:3.1f},{row['output_prefetch']:3.1f}) | "
                      f"({row['ffn1_overlap']:3.1f},{row['ffn1_prefetch']:3.1f}) | "
                      f"({row['ffn2_overlap']:3.1f},{row['ffn2_prefetch']:3.1f})")
            
            # 分析最佳配置的模式
            print("\nBest configuration analysis:")
            best = top10.iloc[0]
            print(f"QKV strategy: overlap={best['qkv_overlap']:.1f}, prefetch={best['qkv_prefetch']:.1f}")
            print(f"Output strategy: overlap={best['output_overlap']:.1f}, prefetch={best['output_prefetch']:.1f}")
            print(f"FFN1 strategy: overlap={best['ffn1_overlap']:.1f}, prefetch={best['ffn1_prefetch']:.1f}")
            print(f"FFN2 strategy: overlap={best['ffn2_overlap']:.1f}, prefetch={best['ffn2_prefetch']:.1f}")
            
            # 保存到汇总
            summary_results.append({
                'Shape': name,
                'Dimensions': f"{batch_size}x{seq_len}x{d_model}x{n_heads}x{d_ff}",
                'Baseline(ms)': f"{baseline_latency:.3f}",
                'Best(ms)': f"{best['latency_ms']:.3f}",
                'Speedup': f"{best['speedup']:.2f}x",
                'Best_QKV': f"({best['qkv_overlap']:.1f},{best['qkv_prefetch']:.1f})",
                'Best_Output': f"({best['output_overlap']:.1f},{best['output_prefetch']:.1f})",
                'Best_FFN1': f"({best['ffn1_overlap']:.1f},{best['ffn1_prefetch']:.1f})",
                'Best_FFN2': f"({best['ffn2_overlap']:.1f},{best['ffn2_prefetch']:.1f})"
            })
            
            # 清理内存
            del benchmark
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to test {name}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
    
    # 保存汇总结果
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_file = os.path.join(run_dir, "summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print("\n" + "="*100)
        print("SUMMARY OF ALL SHAPES")
        print("="*100)
        print(summary_df.to_string(index=False))
    
    print(f"\nResults saved to: {run_dir}")
    
    # 打印GPU信息
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Peak Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    print("Starting Decoder Layer Parameter Optimization (Smart Sampling)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Baseline configuration: all mm parameters = (-1.0, 0.0)")
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)
    
    import time
    start_time = time.time()
    
    test_multiple_shapes()
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")