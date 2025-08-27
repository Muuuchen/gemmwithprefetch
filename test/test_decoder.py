import torch
import torch.nn as nn
import math
import cutlass_gemm_with_prefetch
import time
import numpy as np

class CUTLASSTransformerBlock(nn.Module):
    """
    使用CUTLASS加速的Transformer Block
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff
        
        # 初始化权重
        self._init_weights_fp8(device)
        
        # CUTLASS配置 - 使用合理的默认值
        self.mm_overlap = 0.2
        self.mm_prefetch = 0.4
        self.rmsnorm_prefetch = 0.2
        self.rmsnorm_hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH
        
    def _init_weights_fp8(self, device):
        """
        权重初始化
        - 权重矩阵(B): float8_e5m2
        - 偏置(D): float8_e4m3fn
        """
        scale = 0.02
        
        # QKV投影权重 - B矩阵使用e5m2
        self.W_qkv = nn.Parameter(
            (torch.randn(self.d_model, 3 * self.d_model, device=device) * scale)
            .to(torch.float8_e5m2)
        )
        self.b_qkv = nn.Parameter(
            torch.zeros(3 * self.d_model, device=device).to(torch.float8_e4m3fn)
        )
        
        # Output投影
        self.W_o = nn.Parameter(
            (torch.randn(self.d_model, self.d_model, device=device) * scale)
            .to(torch.float8_e5m2)
        )
        self.b_o = nn.Parameter(
            torch.zeros(self.d_model, device=device).to(torch.float8_e4m3fn)
        )
        
        # FFN权重
        self.W_ff1 = nn.Parameter(
            (torch.randn(self.d_model, self.d_ff, device=device) * scale)
            .to(torch.float8_e5m2)
        )
        self.b_ff1 = nn.Parameter(
            torch.zeros(self.d_ff, device=device).to(torch.float8_e4m3fn)
        )
        
        self.W_ff2 = nn.Parameter(
            (torch.randn(self.d_ff, self.d_model, device=device) * scale)
            .to(torch.float8_e5m2)
        )
        self.b_ff2 = nn.Parameter(
            torch.zeros(self.d_model, device=device).to(torch.float8_e4m3fn)
        )
        
        # RMSNorm权重
        self.norm1_weight = nn.Parameter(
            torch.ones(self.d_model, device=device).to(torch.float8_e4m3fn)
        )
        self.norm2_weight = nn.Parameter(
            torch.ones(self.d_model, device=device).to(torch.float8_e4m3fn)
        )
    
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-Head Attention
        """
        batch_size, seq_len, _ = x.shape
        
        # 确保输入是e4m3
        x = x.to(torch.float8_e4m3fn)
        
        # QKV projection
        # 创建输出buffer
        qkv_output = torch.zeros(batch_size, seq_len, 3 * self.d_model, 
                                device=x.device, dtype=torch.float8_e4m3fn)
        
        # MM操作: x @ W_qkv + b_qkv
        qkv = cutlass_gemm_with_prefetch.mm(
            x.reshape(-1, self.d_model),  # flatten batch and seq: [B*S, D]
            self.W_qkv,                    # [D, 3*D]
            qkv_output.reshape(-1, 3 * self.d_model),  # [B*S, 3*D]
            self.b_qkv.unsqueeze(0).expand(batch_size * seq_len, -1),  # [B*S, 3*D]
            self.mm_overlap,
            self.mm_prefetch
        )
        
        # Reshape回原始维度
        qkv = qkv.reshape(batch_size, seq_len, 3 * self.d_model)
        
        # Split QKV
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0, :, :]  # [B, S, H, D]
        k = qkv[:, :, 1, :, :]  # [B, S, H, D]
        v = qkv[:, :, 2, :, :]  # [B, S, H, D]
        
        # Flash Attention (需要FP16)
        q_fp16 = q.contiguous().to(torch.float16)
        k_fp16 = k.contiguous().to(torch.float16)
        v_fp16 = v.contiguous().to(torch.float16)
        
        # FA: [B, S, H, D] -> [B, S, H, D]
        attn_output = cutlass_gemm_with_prefetch.fa(q_fp16, k_fp16, v_fp16)
        
        # Reshape: [B, S, H, D] -> [B, S, D_model]
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # 转回e4m3
        attn_output = attn_output.to(torch.float8_e4m3fn)
        
        # Output projection
        o_output = torch.zeros(batch_size, seq_len, self.d_model,
                            device=x.device, dtype=torch.float8_e4m3fn)
        
        output = cutlass_gemm_with_prefetch.mm(
            attn_output.reshape(-1, self.d_model),  # [B*S, D]
            self.W_o,                                 # [D, D]
            o_output.reshape(-1, self.d_model),      # [B*S, D]
            self.b_o.unsqueeze(0).expand(batch_size * seq_len, -1),  # [B*S, D]
            self.mm_overlap,
            self.mm_prefetch
        )
        
        return output.reshape(batch_size, seq_len, self.d_model)
    
    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-Forward Network
        """
        batch_size, seq_len, _ = x.shape
        
        x = x.to(torch.float8_e4m3fn)
        
        # First layer
        ff1_output = torch.zeros(batch_size * seq_len, self.d_ff,
                                device=x.device, dtype=torch.float8_e4m3fn)
        
        hidden = cutlass_gemm_with_prefetch.mm(
            x.reshape(-1, self.d_model),     # [B*S, D]
            self.W_ff1,                       # [D, FF]
            ff1_output,                       # [B*S, FF]
            self.b_ff1.unsqueeze(0).expand(batch_size * seq_len, -1),  # [B*S, FF]
            self.mm_overlap,
            self.mm_prefetch
        )
        
        # GELU activation
        hidden = hidden.to(torch.float16)
        hidden = torch.nn.functional.gelu(hidden)
        hidden = hidden.to(torch.float8_e4m3fn)
        
        # Second layer
        ff2_output = torch.zeros(batch_size * seq_len, self.d_model,
                                device=x.device, dtype=torch.float8_e4m3fn)
        
        output = cutlass_gemm_with_prefetch.mm(
            hidden,                           # [B*S, FF]
            self.W_ff2,                       # [FF, D]
            ff2_output,                       # [B*S, D]
            self.b_ff2.unsqueeze(0).expand(batch_size * seq_len, -1),  # [B*S, D]
            self.mm_overlap,
            self.mm_prefetch
        )
        
        return output.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer Block前向传播
        """
        batch_size, seq_len, _ = x.shape
        
        # 确保输入是e4m3
        x = x.to(torch.float8_e4m3fn)
        
        # ========== Attention ==========
        residual = x
        
        # RMSNorm1 - flatten for norm
        x_flat = x.reshape(-1, self.d_model)  # [B*S, D]
        norm1_output = torch.zeros_like(x_flat, dtype=torch.float8_e4m3fn)
        
        x_norm = cutlass_gemm_with_prefetch.rmsnorm(
            norm1_output,
            x_flat,
            self.norm1_weight,
            self.rmsnorm_prefetch,
            self.rmsnorm_hierarchy
        )
        x_norm = x_norm.reshape(batch_size, seq_len, self.d_model)
        
        # Attention
        attn_out = self.attention(x_norm)
        
        # Residual (注释掉因为没有实现加法)
        # x = residual + attn_out
        x = attn_out
        
        # ========== FFN ==========
        residual = x
        
        # RMSNorm2
        x_flat = x.reshape(-1, self.d_model)
        norm2_output = torch.zeros_like(x_flat, dtype=torch.float8_e4m3fn)
        
        x_norm = cutlass_gemm_with_prefetch.rmsnorm(
            norm2_output,
            x_flat,
            self.norm2_weight,
            self.rmsnorm_prefetch,
            self.rmsnorm_hierarchy
        )
        x_norm = x_norm.reshape(batch_size, seq_len, self.d_model)
        
        # FFN
        ffn_out = self.ffn(x_norm)
        
        # Residual (注释掉因为没有实现加法)
        # output = residual + ffn_out
        output = ffn_out
        
        return output


def benchmark_transformer_block(model, batch_size, seq_len, d_model, warmup_iters=10, test_iters=100):
    """
    性能测试函数
    """
    try:
        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        # Warmup
        print(f"  Warmup ({warmup_iters} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(x)
        torch.cuda.synchronize()
        
        # 性能测试
        print(f"  Testing ({test_iters} iterations)...")
        
        # 使用CUDA events进行精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            torch.cuda.synchronize()
            start_event.record()
            
            for _ in range(test_iters):
                output = model(x)
            
            end_event.record()
            torch.cuda.synchronize()
        
        # 计算时间
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / test_iters
        
        # 计算吞吐量
        total_tokens = batch_size * seq_len
        tokens_per_second = (total_tokens * 1000) / avg_time_ms
        
        # 计算FLOPS (简化计算)
        flops_per_token = (
            3 * d_model * d_model +  # QKV projection
            seq_len * d_model +      # Attention computation
            d_model * d_model +      # Output projection  
            2 * d_model * model.d_ff # FFN layers
        )
        total_flops = flops_per_token * total_tokens * 2  # multiply by 2 for multiply-add
        tflops = (total_flops / 1e12) / (avg_time_ms / 1000)
        
        return {
            'avg_latency_ms': avg_time_ms,
            'tokens_per_second': tokens_per_second,
            'tflops': tflops,
        }
    
    except Exception as e:
        print(f"  Error during benchmark: {e}")
        return None


def test_transformer_block_with_performance():
    """
    完整的Transformer Block测试，包含性能测试
    """
    
    # 设置随机种子
    torch.manual_seed(0)
    
    # 测试配置
    batch_size = 1
    seq_len = 128
    n_heads = 32
    head_dim = 64
    d_model = n_heads * head_dim  # 2048
    d_ff = d_model * 2  # 4096
    
    print("="*60)
    print("CUTLASS Transformer Block Performance Test")
    print("="*60)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  FFN dimension: {d_ff}")
    print()
    
    # 创建模型
    print("Creating model...")
    model = CUTLASSTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff
    ).cuda()
    
    # 功能测试
    print("\n--- Functional Test ---")
    x = torch.randn(batch_size, seq_len, d_model, device='cuda')
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    
    try:
        with torch.no_grad():
            output = model(x)
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")
        
        if torch.isnan(output).any():
            print("⚠️  Warning: Output contains NaN values")
        else:
            print("✓ Output values are valid")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 端到端性能测试
    print("\n" + "="*60)
    print("End-to-End Performance Benchmark")
    print("="*60)
    
    # 测试不同配置
    test_configs = [
        (1, 64),    # batch_size, seq_length
        (1, 128),
        (4, 64),
        (4, 128),
        (8, 64),
        (8, 128),
    ]
    
    results = []
    print(f"\n{'Batch':<8} {'SeqLen':<8} {'Latency(ms)':<12} {'Tokens/sec':<12} {'TFLOPS':<10}")
    print("-" * 60)
    
    for bs, sl in test_configs:
        perf_results = benchmark_transformer_block(
            model, bs, sl, d_model, 
            warmup_iters=10, 
            test_iters=100
        )
        
        if perf_results:
            print(f"{bs:<8} {sl:<8} {perf_results['avg_latency_ms']:<12.3f} "
                  f"{perf_results['tokens_per_second']:<12.0f} "
                  f"{perf_results['tflops']:<10.2f}")
            
            results.append({
                'batch_size': bs,
                'seq_length': sl,
                **perf_results
            })
        else:
            print(f"{bs:<8} {sl:<8} {'Failed':<12} {'-':<12} {'-':<10}")
    
    # 性能总结
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    if results:
        # 找出最佳配置
        best_throughput = max(results, key=lambda x: x['tokens_per_second'])
        best_latency = min(results, key=lambda x: x['avg_latency_ms'])
        
        print(f"\nBest Throughput:")
        print(f"  Config: Batch={best_throughput['batch_size']}, SeqLen={best_throughput['seq_length']}")
        print(f"  Throughput: {best_throughput['tokens_per_second']:.0f} tokens/sec")
        print(f"  Latency: {best_throughput['avg_latency_ms']:.3f} ms")
        print(f"  Performance: {best_throughput['tflops']:.2f} TFLOPS")
        
        print(f"\nBest Latency:")
        print(f"  Config: Batch={best_latency['batch_size']}, SeqLen={best_latency['seq_length']}")
        print(f"  Latency: {best_latency['avg_latency_ms']:.3f} ms")
        print(f"  Throughput: {best_latency['tokens_per_second']:.0f} tokens/sec")
        print(f"  Performance: {best_latency['tflops']:.2f} TFLOPS")
        
        # 计算平均性能
        avg_latency = np.mean([r['avg_latency_ms'] for r in results])
        avg_throughput = np.mean([r['tokens_per_second'] for r in results])
        avg_tflops = np.mean([r['tflops'] for r in results])
        
        print(f"\nAverage Performance:")
        print(f"  Latency: {avg_latency:.3f} ms")
        print(f"  Throughput: {avg_throughput:.0f} tokens/sec")
        print(f"  TFLOPS: {avg_tflops:.2f}")
    
    # 模型信息
    print(f"\n" + "="*60)
    print("Model Information")
    print("="*60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 参数细节
    print("\nParameter details:")
    for name, param in model.named_parameters():
        param_count = param.numel()
        # print(f"  {name:<20} shape={list(param.shape):<25} dtype={str(param.dtype):<20} params={param_count:,}")
    
    # GPU信息
    print(f"\n" + "="*60)
    print("GPU Information")
    print("="*60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 内存使用
    allocated_mb = torch.cuda.memory_allocated() / 1e6
    reserved_mb = torch.cuda.memory_reserved() / 1e6
    print(f"Allocated Memory: {allocated_mb:.1f} MB")
    print(f"Reserved Memory: {reserved_mb:.1f} MB")


if __name__ == "__main__":
    test_transformer_block_with_performance()