import torch
import torch.nn as nn
import cutlass_gemm_with_prefetch
import numpy as np
import itertools
from dataclasses import dataclass
import pandas as pd
import os
from datetime import datetime

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
    # RMSNorm
    rmsnorm_prefetch: float
    rmsnorm_hierarchy: int
    
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
            'rmsnorm_prefetch': self.rmsnorm_prefetch,
            'rmsnorm_hierarchy': self.rmsnorm_hierarchy
        }

class DecoderLayerBenchmark:
    """Decoder Layer性能测试"""
    
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
        
        # 定义参数空间（简化版本）
        self.mm_values = [0.0, 0.5, 1.0, -1.0]  # 减少参数空间
        self.rmsnorm_values = [0.0, 0.5, 1.0]
        self.hierarchy_values = [
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH,
            cutlass_gemm_with_prefetch.KernelOverlapHierarchy.SHAREDMEM
        ]
        
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
        
        # RMSNorm权重 - 3个RMSNorm层
        self.norm1_weight = torch.ones(self.d_model, device=self.device).to(torch.float8_e4m3fn)  # attention前
        self.norm2_weight = torch.ones(self.d_model, device=self.device).to(torch.float8_e4m3fn)  # FFN前
        self.norm3_weight = torch.ones(self.d_ff, device=self.device).to(torch.float8_e4m3fn)     # FFN中间
    
    def decoder_layer_forward(self, x: torch.Tensor, config: DecoderLayerConfig) -> torch.Tensor:
        """执行完整的decoder layer前向传播"""
        batch_size, seq_len, _ = x.shape
        
        # 保存第一个residual
        residual1 = x.to(torch.float8_e4m3fn)
        
        # 1. 第一个RMSNorm + Attention
        # RMSNorm1
        x_norm1 = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
        cutlass_gemm_with_prefetch.rmsnorm(
            residual1,
            self.norm1_weight,
            x_norm1,
            config.rmsnorm_prefetch,
            config.rmsnorm_hierarchy
        )
        
        # QKV Projection
        x_flat = x_norm1.reshape(-1, self.d_model)
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
        
        # 2. 第二个RMSNorm + FFN
        # RMSNorm2 (before FFN)
        x_norm2 = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
        cutlass_gemm_with_prefetch.rmsnorm(
            x,
            self.norm2_weight,
            x_norm2,
            config.rmsnorm_prefetch,
            config.rmsnorm_hierarchy
        )
        
        # FFN第一层
        x_flat = x_norm2.reshape(-1, self.d_model)
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
        
        # RMSNorm3 (in the middle of FFN)
        hidden_norm = torch.zeros_like(hidden, dtype=torch.float8_e4m3fn)
        cutlass_gemm_with_prefetch.rmsnorm(
            hidden,
            self.norm3_weight,
            hidden_norm,
            config.rmsnorm_prefetch,
            config.rmsnorm_hierarchy
        )
        
        # FFN第二层
        ff2_output = torch.zeros(batch_size * seq_len, self.d_model,
                               device=x.device, dtype=torch.float8_e4m3fn)
        output = cutlass_gemm_with_prefetch.mm(
            hidden_norm,
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
    
    def run_benchmark(self):
        """运行参数测试"""
        # Baseline配置 - 所有mm参数都是(-1.0, 0.0)
        baseline_config = DecoderLayerConfig(
            qkv_overlap=-1.0, qkv_prefetch=0.0,
            output_overlap=-1.0, output_prefetch=0.0,
            ffn1_overlap=-1.0, ffn1_prefetch=0.0,
            ffn2_overlap=-1.0, ffn2_prefetch=0.0,
            rmsnorm_prefetch=0.0,
            rmsnorm_hierarchy=cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH
        )
        
        print("Testing baseline configuration (all mm params: -1.0, 0.0)...")
        baseline_latency = self.benchmark_config(baseline_config, warmup=20, iterations=100)
        print(f"Baseline latency: {baseline_latency:.3f} ms")
        
        # 生成参数组合（简化版本，避免组合爆炸）
        all_configs = []
        
        # 只测试一些代表性的组合
        for qkv_params in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (-1.0, 0.0)]:
            for out_params in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (-1.0, 0.0)]:
                for ffn1_params in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (-1.0, 0.0)]:
                    for ffn2_params in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (-1.0, 0.0)]:
                        for rms_p in [0.0, 0.5]:
                            for hier in self.hierarchy_values:
                                config = DecoderLayerConfig(
                                    qkv_overlap=qkv_params[0], qkv_prefetch=qkv_params[1],
                                    output_overlap=out_params[0], output_prefetch=out_params[1],
                                    ffn1_overlap=ffn1_params[0], ffn1_prefetch=ffn1_params[1],
                                    ffn2_overlap=ffn2_params[0], ffn2_prefetch=ffn2_params[1],
                                    rmsnorm_prefetch=rms_p,
                                    rmsnorm_hierarchy=hier
                                )
                                all_configs.append(config)
        
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
        (1, 128, 768, 12, 3072, "BERT-base"),
        (1, 256, 768, 12, 3072, "BERT-base-med"),
        (1, 512, 768, 12, 3072, "BERT-base-long"),
        (4, 128, 768, 12, 3072, "BERT-base-batch4"),
        (1, 128, 1024, 16, 4096, "BERT-large"),
        (1, 128, 2048, 16, 8192, "GPT-2-medium"),
        (1, 128, 4096, 32, 11008, "LLaMA-7B-like"),
        (1, 128, 768, 8, 2048, "Small-model"),
    ]
    
    run_dir = ensure_results_dir()
    summary_results = []
    
    for batch_size, seq_len, d_model, n_heads, d_ff, name in shapes:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"Shape: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
        print(f"{'='*80}")
        
        try:
            # 创建benchmark
            benchmark = DecoderLayerBenchmark(d_model, n_heads, d_ff, batch_size, seq_len)
            
            # 运行测试
            df, baseline_latency = benchmark.run_benchmark()
            
            if len(df) == 0:
                print(f"No valid results for {name}")
                continue
            
            # 找出top 10配置
            top10 = df.nsmallest(10, 'latency_ms')
            
            # 保存完整结果
            result_file = os.path.join(run_dir, f"{name.replace('-', '_')}_results.csv")
            df.to_csv(result_file, index=False)
            
            # 打印top 5
            print(f"\nTop 5 configurations for {name}:")
            print(f"Baseline latency: {baseline_latency:.3f} ms (all mm: -1.0, 0.0)")
            print("\nRank | Latency(ms) | Speedup |   QKV   |  Output  |   FFN1   |   FFN2   | RMS")
            print("-" * 90)
            
            for idx, (i, row) in enumerate(top10.head(5).iterrows()):
                print(f"{idx+1:4d} | {row['latency_ms']:11.3f} | {row['speedup']:7.2f}x | "
                      f"({row['qkv_overlap']:3.1f},{row['qkv_prefetch']:3.1f}) | "
                      f"({row['output_overlap']:3.1f},{row['output_prefetch']:3.1f}) | "
                      f"({row['ffn1_overlap']:3.1f},{row['ffn1_prefetch']:3.1f}) | "
                      f"({row['ffn2_overlap']:3.1f},{row['ffn2_prefetch']:3.1f}) | "
                      f"{row['rmsnorm_prefetch']:.1f}")
            
            # 保存到汇总
            best_config = top10.iloc[0]
            summary_results.append({
                'Shape': name,
                'Dimensions': f"{batch_size}x{seq_len}x{d_model}x{n_heads}x{d_ff}",
                'Baseline(ms)': f"{baseline_latency:.3f}",
                'Best(ms)': f"{best_config['latency_ms']:.3f}",
                'Speedup': f"{best_config['speedup']:.2f}x",
                'Best_QKV': f"({best_config['qkv_overlap']:.1f},{best_config['qkv_prefetch']:.1f})",
                'Best_Output': f"({best_config['output_overlap']:.1f},{best_config['output_prefetch']:.1f})",
                'Best_FFN1': f"({best_config['ffn1_overlap']:.1f},{best_config['ffn1_prefetch']:.1f})",
                'Best_FFN2': f"({best_config['ffn2_overlap']:.1f},{best_config['ffn2_prefetch']:.1f})"
            })
            
            # 清理内存
            del benchmark
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to test {name}: {e}")
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
    print("Starting Decoder Layer Parameter Optimization")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Baseline configuration: all mm parameters = (-1.0, 0.0)")
    
    import time
    start_time = time.time()
    
    test_multiple_shapes()
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")