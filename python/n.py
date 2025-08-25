import torch
import torch.nn as nn
import cutlass_gemm_with_prefetch
import time

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

def simple_benchmark(M, K, N, config, warmup=20, iterations=100):
    """简单的基准测试函数"""
    # 创建模型和输入
    ffn = FeedForward(K, N, config)
    x = torch.randn((M, K)).to(device="cuda").to(dtype=torch.float8_e4m3fn)
    
    # Warmup
    for _ in range(warmup):
        _ = ffn(x)
    torch.cuda.synchronize()
    
    # 实际测试
    start = time.perf_counter()
    for _ in range(iterations):
        _ = ffn(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    # 返回平均执行时间（毫秒）
    avg_time_ms = (end - start) * 1000 / iterations
    return avg_time_ms

def test_shape(M, K, N):
    """测试特定shape的不同配置"""
    print(f"\n{'='*60}")
    print(f"Testing shape: M={M}, K={K}, N={N}")
    print(f"{'='*60}")
    
    # 测试NONE模式
    config_none = FeedForwardPDLConfig()
    config_none.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE
    
    time_none = simple_benchmark(M, K, N, config_none)
    print(f"\nNONE mode: {time_none:.3f} ms")
    
    # 测试不同的FREFTECH配置
    test_configs = [
        # (mm1_overlap, mm1_prefetch, mm2_overlap, mm2_prefetch, rmsnorm_prefetch)
        # (0.0, 0.0, 0.0, 0.0, 0.0),
        # (-1.0, 0.0, -1.0, 0.0, 0.0),
        # (0.0, 0.5, 0.0, 0.5, 0.5),
        # (-1.0, 0.5, -1.0, 0.5, 0.5),
        # (0.0, 1.0, 0.0, 1.0, 1.0),
        # (-1.0, 1.0, -1.0, 1.0, 1.0),
        # (0.5, 0.5, 0.5, 0.5, 0.5),
        # (0.3, 0.7, 0.3, 0.7, 0.7),
        # (0.7, 0.3, 0.7, 0.3, 0.3),
        (0.2,0.2,0.2,0.3,0.3)
    ]
    
    print("\nFREFTECH mode configurations:")
    print(f"{'Config':<40} {'Time (ms)':<10} {'Speedup'}")
    print("-" * 60)
    
    for mm1_or, mm1_pr, mm2_or, mm2_pr, rms_pr in test_configs:
        config = FeedForwardPDLConfig()
        config.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH
        config.mm1_overlap_ratio = mm1_or
        config.mm1_prefetch_ratio = mm1_pr
        config.mm2_overlap_ratio = mm2_or
        config.mm2_prefetch_ratio = mm2_pr
        config.rmsnorm_prefetch_ratio = rms_pr
        
        try:
            time_prefetch = simple_benchmark(M, K, N, config)
            speedup = time_none / time_prefetch
            config_str = f"MM1({mm1_or},{mm1_pr}) MM2({mm2_or},{mm2_pr}) RMS({rms_pr})"
            print(f"{config_str:<40} {time_prefetch:<10.3f} {speedup:.2f}x")
        except Exception as e:
            config_str = f"MM1({mm1_or},{mm1_pr}) MM2({mm2_or},{mm2_pr}) RMS({rms_pr})"
            print(f"{config_str:<40} Failed: {str(e)[:20]}")

if __name__ == "__main__":
    # 测试一个shape
    M, K, N = 32, 1024, 7168
    test_shape(M, K, N)
    
    # 如果想测试多个shapes，可以这样：
    # shapes = [(1024, 1024, 2048), (2048, 2048, 4096), (4096, 1024, 2048)]
    # for M, K, N in shapes:
    #     test_shape(M, K, N)