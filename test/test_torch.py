import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np

# Flash Attention import
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    print("Warning: flash_attn not found, will use PyTorch SDPA instead")
    HAS_FLASH_ATTN = False

class PyTorchTransformerBlock(nn.Module):
    """
    使用PyTorch原生算子的Transformer Block（使用Flash Attention）
    保持与CUTLASS版本完全相同的结构
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
        
        # 初始化权重 - 与CUTLASS版本相同
        self._init_weights_fp8(device)
        
    def _init_weights_fp8(self, device):
        """
        权重初始化 - 与CUTLASS版本保持一致
        使用FP8格式（如果可用），否则使用FP16
        """
        scale = 0.02
        
        # QKV投影权重
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
    
    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm实现 - PyTorch版本
        """
        # 转换到FP32进行计算
        x_fp32 = x.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)
        
        # RMS normalization
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + 1e-6)
        x_norm = x_fp32 / rms
        
        # Apply weight
        output = x_norm * weight_fp32
        
        # 转回FP8
        return output.to(torch.float8_e4m3fn)
    
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-Head Attention - 使用Flash Attention或PyTorch SDPA
        保持与CUTLASS版本相同的计算流程
        """
        batch_size, seq_len, _ = x.shape
        
        # 确保输入是e4m3
        x = x.to(torch.float8_e4m3fn)
        
        # QKV projection - 使用PyTorch的矩阵乘法
        # 需要转换到支持的数据类型进行计算
        x_compute = x.to(torch.float16)
        W_qkv_compute = self.W_qkv.to(torch.float16)
        b_qkv_compute = self.b_qkv.to(torch.float16)
        
        # [B*S, D] @ [D, 3*D] + [3*D]
        qkv = torch.matmul(
            x_compute.reshape(-1, self.d_model),
            W_qkv_compute
        ) + b_qkv_compute
        
        # Reshape回原始维度
        qkv = qkv.reshape(batch_size, seq_len, 3 * self.d_model)
        
        # Split QKV - 保持与CUTLASS版本相同的形状
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0, :, :]  # [B, S, H, D]
        k = qkv[:, :, 1, :, :]  # [B, S, H, D]
        v = qkv[:, :, 2, :, :]  # [B, S, H, D]
        
        # 确保连续性（Flash Attention需要）
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        if HAS_FLASH_ATTN:
            # 使用Flash Attention
            # flash_attn期望输入形状为 [B, S, H, D]
            # 注意：flash_attn_func可能需要特定的数据类型（通常是fp16或bf16）
            attn_output = flash_attn_func(
                q,  # [B, S, H, D]
                k,  # [B, S, H, D]
                v,  # [B, S, H, D]
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.d_head),
                causal=False  # 对于encoder，使用非因果注意力
            )
            # attn_output: [B, S, H, D]
        else:
            # 使用PyTorch的scaled_dot_product_attention作为后备
            # 需要转换维度: [B, S, H, D] -> [B, H, S, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # 使用PyTorch 2.0的优化版本
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True
            ):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    scale=1.0 / math.sqrt(self.d_head)
                )
            # 转换回: [B, H, S, D] -> [B, S, H, D]
            attn_output = attn_output.transpose(1, 2)
        
        # Reshape: [B, S, H, D] -> [B, S, D_model]
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        W_o_compute = self.W_o.to(torch.float16)
        b_o_compute = self.b_o.to(torch.float16)
        
        output = torch.matmul(
            attn_output.reshape(-1, self.d_model),
            W_o_compute
        ) + b_o_compute
        
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        # 转回FP8
        return output.to(torch.float8_e4m3fn)
    
    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-Forward Network - PyTorch版本
        """
        batch_size, seq_len, _ = x.shape
        
        x = x.to(torch.float8_e4m3fn)
        
        # 转换到FP16进行计算
        x_compute = x.to(torch.float16)
        W_ff1_compute = self.W_ff1.to(torch.float16)
        b_ff1_compute = self.b_ff1.to(torch.float16)
        
        # First layer
        hidden = torch.matmul(
            x_compute.reshape(-1, self.d_model),
            W_ff1_compute
        ) + b_ff1_compute
        
        # GELU activation
        hidden = F.gelu(hidden)
        
        # Second layer
        W_ff2_compute = self.W_ff2.to(torch.float16)
        b_ff2_compute = self.b_ff2.to(torch.float16)
        
        output = torch.matmul(
            hidden,
            W_ff2_compute
        ) + b_ff2_compute
        
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        # 转回FP8
        return output.to(torch.float8_e4m3fn)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer Block前向传播 - 与CUTLASS版本完全相同的逻辑
        """
        batch_size, seq_len, _ = x.shape
        
        # 确保输入是e4m3
        x = x.to(torch.float8_e4m3fn)
        
        # ========== Attention ==========
        residual = x
        
        # RMSNorm1
        x_norm = self.rmsnorm(x.reshape(-1, self.d_model), self.norm1_weight)
        x_norm = x_norm.reshape(batch_size, seq_len, self.d_model)
        
        # Attention
        attn_out = self.attention(x_norm)
        
        # Residual (注释掉，与CUTLASS版本保持一致)
        # x = residual + attn_out
        x = attn_out
        
        # ========== FFN ==========
        residual = x
        
        # RMSNorm2
        x_norm = self.rmsnorm(x.reshape(-1, self.d_model), self.norm2_weight)
        x_norm = x_norm.reshape(batch_size, seq_len, self.d_model)
        
        # FFN
        ffn_out = self.ffn(x_norm)
        
        # Residual (注释掉，与CUTLASS版本保持一致)
        # output = residual + ffn_out
        output = ffn_out
        
        return output


def benchmark_pytorch_transformer(model, batch_size, seq_len, d_model, warmup_iters=10, test_iters=100):
    """
    性能测试函数 - PyTorch版本
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
        
        # 计算FLOPS
        flops_per_token = (
            3 * d_model * d_model +  # QKV projection
            seq_len * d_model +      # Attention computation
            d_model * d_model +      # Output projection  
            2 * d_model * model.d_ff # FFN layers
        )
        total_flops = flops_per_token * total_tokens * 2
        tflops = (total_flops / 1e12) / (avg_time_ms / 1000)
        
        return {
            'avg_latency_ms': avg_time_ms,
            'tokens_per_second': tokens_per_second,
            'tflops': tflops,
        }
    
    except Exception as e:
        print(f"  Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_cutlass():
    """
    对比PyTorch版本和CUTLASS版本的性能
    """
    import sys
    sys.path.append('.')
    
    # 尝试导入CUTLASS版本进行对比
    try:
        from test_decoder import CUTLASSTransformerBlock, benchmark_transformer_block
        has_cutlass = True
    except ImportError:
        print("Warning: Could not import CUTLASS version for comparison")
        has_cutlass = False
    
    # 设置随机种子
    torch.manual_seed(0)
    
    # 测试配置
    n_heads = 32
    head_dim = 64
    d_model = n_heads * head_dim  # 2048
    d_ff = d_model * 2  # 4096
    
    print("="*80)
    print("PyTorch (with Flash Attention) vs CUTLASS Transformer Block Performance")
    print("="*80)
    print(f"Configuration:")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  FFN dimension: {d_ff}")
    print(f"  Flash Attention: {'Available' if HAS_FLASH_ATTN else 'Not Available (using PyTorch SDPA)'}")
    print()
    
    # 创建PyTorch模型
    print("Creating PyTorch model...")
    pytorch_model = PyTorchTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff
    ).cuda()
    pytorch_model.eval()
    
    # 创建CUTLASS模型（如果可用）
    cutlass_model = None
    if has_cutlass:
        print("Creating CUTLASS model...")
        cutlass_model = CUTLASSTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff
        ).cuda()
        cutlass_model.eval()
    
    # 功能验证
    print("\n--- Functional Verification ---")
    test_input = torch.randn(1, 64, d_model, device='cuda')
    
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        print(f"PyTorch output shape: {pytorch_output.shape}, dtype: {pytorch_output.dtype}")
        
        if has_cutlass:
            cutlass_output = cutlass_model(test_input)
            print(f"CUTLASS output shape: {cutlass_output.shape}, dtype: {cutlass_output.dtype}")
    
    # 测试配置
    test_configs = [
        (1, 64),    # batch_size, seq_length
        (1, 128),
        (4, 64),
        (4, 128),
        (8, 64),
        (8, 128),
    ]
    
    # 结果存储
    pytorch_results = []
    cutlass_results = []
    
    print("\n" + "="*80)
    print("Performance Results")
    print("="*80)
    
    # 打印表头
    if has_cutlass:
        print(f"\n{'Config':<15} {'PyTorch (Flash)':<35} {'CUTLASS':<35} {'Speedup':<10}")
        print(f"{'='*15} {'='*35} {'='*35} {'='*10}")
        print(f"{'Batch x Seq':<15} {'Latency(ms) / Tokens/s / TFLOPS':<35} {'Latency(ms) / Tokens/s / TFLOPS':<35} {'Ratio':<10}")
        print("-"*95)
    else:
        print(f"\n{'Config':<15} {'PyTorch (Flash)':<35}")
        print(f"{'='*15} {'='*35}")
        print(f"{'Batch x Seq':<15} {'Latency(ms) / Tokens/s / TFLOPS':<35}")
        print("-"*50)
    
    for bs, sl in test_configs:
        config_str = f"{bs} x {sl}"
        print(f"\n{config_str:<15}", end="")
        
        # 测试PyTorch版本
        pytorch_perf = benchmark_pytorch_transformer(
            pytorch_model, bs, sl, d_model,
            warmup_iters=10,
            test_iters=100
        )
        
        if pytorch_perf:
            pytorch_results.append({
                'batch_size': bs,
                'seq_length': sl,
                **pytorch_perf
            })
            pytorch_str = f"{pytorch_perf['avg_latency_ms']:.3f}ms / {pytorch_perf['tokens_per_second']:.0f} / {pytorch_perf['tflops']:.1f}"
            print(f"{pytorch_str:<35}", end="")
        else:
            print(f"{'Failed':<35}", end="")
        
        # 测试CUTLASS版本（如果可用）
        if has_cutlass:
            cutlass_perf = benchmark_transformer_block(
                cutlass_model, bs, sl, d_model,
                warmup_iters=10,
                test_iters=100
            )
            
            if cutlass_perf:
                cutlass_results.append({
                    'batch_size': bs,
                    'seq_length': sl,
                    **cutlass_perf
                })
                cutlass_str = f"{cutlass_perf['avg_latency_ms']:.3f}ms / {cutlass_perf['tokens_per_second']:.0f} / {cutlass_perf['tflops']:.1f}"
                print(f"{cutlass_str:<35}", end="")
                
                # 计算加速比
                if pytorch_perf:
                    speedup = pytorch_perf['avg_latency_ms'] / cutlass_perf['avg_latency_ms']
                    print(f"{speedup:.2f}x")
                else:
                    print("-")
            else:
                print(f"{'Failed':<35} {'-':<10}")
        else:
            print()  # 换行
    
    # 性能总结
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    
    if pytorch_results:
        avg_pytorch_latency = np.mean([r['avg_latency_ms'] for r in pytorch_results])
        avg_pytorch_throughput = np.mean([r['tokens_per_second'] for r in pytorch_results])
        avg_pytorch_tflops = np.mean([r['tflops'] for r in pytorch_results])
        
        print(f"\nPyTorch Average Performance:")
        print(f"  Attention Implementation: {'Flash Attention' if HAS_FLASH_ATTN else 'PyTorch SDPA'}")
        print(f"  Latency: {avg_pytorch_latency:.3f} ms")
        print(f"  Throughput: {avg_pytorch_throughput:.0f} tokens/sec")
        print(f"  TFLOPS: {avg_pytorch_tflops:.2f}")
    
    if has_cutlass and cutlass_results:
        avg_cutlass_latency = np.mean([r['avg_latency_ms'] for r in cutlass_results])
        avg_cutlass_throughput = np.mean([r['tokens_per_second'] for r in cutlass_results])
        avg_cutlass_tflops = np.mean([r['tflops'] for r in cutlass_results])
        
        print(f"\nCUTLASS Average Performance:")
        print(f"  Latency: {avg_cutlass_latency:.3f} ms")
        print(f"  Throughput: {avg_cutlass_throughput:.0f} tokens/sec")
        print(f"  TFLOPS: {avg_cutlass_tflops:.2f}")
        
        if pytorch_results:
            print(f"\nAverage Speedup (CUTLASS vs PyTorch):")
            print(f"  Latency: {avg_pytorch_latency / avg_cutlass_latency:.2f}x faster")
            print(f"  Throughput: {avg_cutlass_throughput / avg_pytorch_throughput:.2f}x higher")
            print(f"  TFLOPS: {avg_cutlass_tflops / avg_pytorch_tflops:.2f}x higher")
    
    # 详细性能分解（如果两个版本都可用）
    if has_cutlass and pytorch_results and cutlass_results:
        print("\n" + "="*80)
        print("Detailed Performance Comparison")
        print("="*80)
        
        print(f"\n{'Metric':<20} {'PyTorch':<20} {'CUTLASS':<20} {'Improvement':<20}")
        print("-"*80)
        
        for pr, cr in zip(pytorch_results, cutlass_results):
            if pr['batch_size'] == cr['batch_size'] and pr['seq_length'] == cr['seq_length']:
                print(f"\nBatch={pr['batch_size']}, Seq={pr['seq_length']}:")
                print(f"  {'Latency (ms)':<18} {pr['avg_latency_ms']:<20.3f} {cr['avg_latency_ms']:<20.3f} {pr['avg_latency_ms']/cr['avg_latency_ms']:.2f}x")
                print(f"  {'Throughput (tok/s)':<18} {pr['tokens_per_second']:<20.0f} {cr['tokens_per_second']:<20.0f} {cr['tokens_per_second']/pr['tokens_per_second']:.2f}x")
                print(f"  {'TFLOPS':<18} {pr['tflops']:<20.2f} {cr['tflops']:<20.2f} {cr['tflops']/pr['tflops']:.2f}x")
    
    # GPU信息
    print(f"\n" + "="*80)
    print("System Information")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # 内存使用
    print(f"\nMemory Usage:")
    allocated_mb = torch.cuda.memory_allocated() / 1e6
    reserved_mb = torch.cuda.memory_reserved() / 1e6
    print(f"  Allocated: {allocated_mb:.1f} MB")
    print(f"  Reserved: {reserved_mb:.1f} MB")


if __name__ == "__main__":
    compare_with_cutlass()