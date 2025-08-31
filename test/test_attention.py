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
RESULTS_DIR = "attention_optimization_results"

def ensure_results_dir():
    """确保结果目录存在"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir)
    return run_dir

@dataclass
class AttentionConfig:
    """Attention配置参数"""
    qkv_overlap: float
    qkv_prefetch: float
    output_overlap: float
    output_prefetch: float
    
    def to_dict(self):
        return {
            'qkv_overlap': self.qkv_overlap,
            'qkv_prefetch': self.qkv_prefetch,
            'output_overlap': self.output_overlap,
            'output_prefetch': self.output_prefetch
        }

class AttentionBenchmark:
    """Attention性能测试"""
    
    def __init__(self, d_model: int = 768, n_heads: int = 12,
                 batch_size: int = 1, seq_len: int = 128):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
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
        
        # QKV权重
        self.W_qkv = (torch.randn(self.d_model, 3 * self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_qkv = torch.zeros(3 * self.d_model, device=self.device).to(torch.float8_e4m3fn)
        
        # Output权重
        self.W_o = (torch.randn(self.d_model, self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_o = torch.zeros(self.d_model, device=self.device).to(torch.float8_e4m3fn)
    
    def attention_forward(self, x: torch.Tensor, config: AttentionConfig) -> torch.Tensor:
        """执行attention前向传播"""
        batch_size, seq_len, _ = x.shape
        
        # 保存residual
        residual = x.to(torch.float8_e4m3fn)
        
        # 1. QKV Projection
        x_flat = x.to(torch.float8_e4m3fn).reshape(-1, self.d_model)
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
        
        # 2. Reshape for attention
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0, :, :].contiguous().to(torch.float16)
        k = qkv[:, :, 1, :, :].contiguous().to(torch.float16)
        v = qkv[:, :, 2, :, :].contiguous().to(torch.float16)
        
        # 3. Flash Attention
        attn_output = cutlass_gemm_with_prefetch.fa(q, k, v)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model).to(torch.float8_e4m3fn)
        
        # 4. Output Projection
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
        
        # 5. Add residual
        output = cutlass_gemm_with_prefetch.add_residual(output.reshape(batch_size, seq_len, self.d_model) , residual)
        
        return output
    
    def benchmark_config(self, config: AttentionConfig, warmup: int = 10, iterations: int = 50):
        """测试单个配置"""
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, 
                       device=self.device, dtype=torch.float16)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.attention_forward(x, config)
        torch.cuda.synchronize()
        
        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            torch.cuda.synchronize()
            start_event.record()
            
            for _ in range(iterations):
                output = self.attention_forward(x, config)
                
            end_event.record()
            torch.cuda.synchronize()
        
        # 计算延迟
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / iterations
        
        return avg_time_ms
    
    def run_benchmark(self):
        """运行完整的参数测试"""
        # Baseline配置 - 所有mm参数都是(-1.0, 0.0)
        baseline_config = AttentionConfig(
            qkv_overlap=-1.0,
            qkv_prefetch=0.0,
            output_overlap=-1.0,
            output_prefetch=0.0
        )
        
        print("Testing baseline configuration (all mm params: -1.0, 0.0)...")
        baseline_latency = self.benchmark_config(baseline_config, warmup=20, iterations=100)
        print(f"Baseline latency: {baseline_latency:.3f} ms")
        
        # 生成所有参数组合
        all_configs = []
        for qkv_o, qkv_p, out_o, out_p in itertools.product(
            self.mm_values, self.mm_values, self.mm_values, self.mm_values
        ):
            all_configs.append(AttentionConfig(qkv_o, qkv_p, out_o, out_p))
        
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
                
                if (i + 1) % 50 == 0:
                    print(f"Progress: {i+1}/{len(all_configs)}")
                    
            except Exception as e:
                print(f"Failed config {i+1}: {e}")
        
        return pd.DataFrame(results), baseline_latency


def test_multiple_shapes():
    """测试多个shape配置"""
    
    shapes = [
        # (batch_size, seq_len, d_model, n_heads, name)
        # (1, 128, 768, 12, "BERT-base"),
        # (1, 512, 768, 12, "BERT-base-long"),
        # (4, 128, 768, 12, "BERT-base-batch4"),
        # (8, 128, 768, 12, "BERT-base-batch8"),
        # (1, 128, 1024, 16, "BERT-large"),
        # (1, 128, 4096, 32, "GPT-3-like"),
        # (1, 256, 4096, 32, "GPT-3-long"),
        # (1, 128, 768, 8, "Small-model"),
            (1,32,1024,4,"tiny"),



    ]
    
    run_dir = ensure_results_dir()
    summary_results = []
    
    for batch_size, seq_len, d_model, n_heads, name in shapes:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"Shape: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
        print(f"{'='*70}")
        
        try:
            # 创建benchmark
            benchmark = AttentionBenchmark(d_model, n_heads, batch_size, seq_len)
            
            # 运行测试
            df, baseline_latency = benchmark.run_benchmark()
            
            # 找出top 10配置
            top10 = df.nsmallest(10, 'latency_ms')
            
            # 保存完整结果
            result_file = os.path.join(run_dir, f"{name.replace('-', '_')}_results.csv")
            df.to_csv(result_file, index=False)
            
            # 打印top 10
            print(f"\nTop 10 configurations for {name}:")
            print(f"Baseline latency: {baseline_latency:.3f} ms (all mm: -1.0, 0.0)")
            print("\nRank | Latency(ms) | Speedup | QKV(overlap,prefetch) | Output(overlap,prefetch)")
            print("-" * 80)
            
            for idx, (i, row) in enumerate(top10.iterrows()):
                print(f"{idx+1:4d} | {row['latency_ms']:11.3f} | {row['speedup']:7.2f}x | "
                      f"({row['qkv_overlap']:4.1f},{row['qkv_prefetch']:4.1f}) | "
                      f"({row['output_overlap']:4.1f},{row['output_prefetch']:4.1f})")
            
            # 保存到汇总
            best_config = top10.iloc[0]
            summary_results.append({
                'Shape': name,
                'Dimensions': f"{batch_size}x{seq_len}x{d_model}x{n_heads}",
                'Baseline(ms)': f"{baseline_latency:.3f}",
                'Best(ms)': f"{best_config['latency_ms']:.3f}",
                'Speedup': f"{best_config['speedup']:.2f}x",
                'Best_QKV': f"({best_config['qkv_overlap']:.1f},{best_config['qkv_prefetch']:.1f})",
                'Best_Output': f"({best_config['output_overlap']:.1f},{best_config['output_prefetch']:.1f})"
            })
            
            # 清理内存
            del benchmark
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to test {name}: {e}")
    
    # 保存汇总结果
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
    print("Starting Attention Parameter Optimization")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Baseline configuration: all mm parameters = (-1.0, 0.0)")
    
    import time
    start_time = time.time()
    
    test_multiple_shapes()
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")