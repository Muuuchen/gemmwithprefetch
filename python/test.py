import time
import torch
import math

import cutlass_gemm_with_prefetch

def benchmark_gemm_rmsnorm(M, K, N, alpha_values, beta_values, eps=1e-5, num_warmup=3, num_repeats=10):
    # 初始化数据
    cuda = torch.device('cuda:0')
    A = torch.normal(0, 1, size=(M, K)).to(device=cuda).to(dtype=torch.float8_e4m3fn)
    B = torch.normal(0, 1, size=(K, N)).to(device=cuda).to(dtype=torch.float8_e5m2)
    C = torch.empty((M, N), device=cuda, dtype=torch.float8_e4m3fn)
    D = torch.normal(0, 1, size=(M, N)).to(device=cuda).to(dtype=torch.float8_e4m3fn)
    E = torch.empty((M, N), device=cuda, dtype=torch.float8_e4m3fn)
    F = torch.normal(0, 1, size=(M, N)).to(device=cuda).to(dtype=torch.float8_e4m3fn)

    # 预热（避免首次运行的开销）
    for _ in range(num_warmup):
        C = cutlass_gemm_with_prefetch.mm(A, B, C, D, 0.7, -1.0)
        E = cutlass_gemm_with_prefetch.rmsnorm(E, D, F,0.5,cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE)
        # C = cutlass_gemm_with_prefetch.mm(A, B, C, D, 0.7, -1.0)

    # 测试不同参数组合
    results = []
    for alpha in alpha_values:
        for beta in beta_values:
            # 创建事件计时器
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # 开始计时
            torch.cuda.synchronize()
            start_event.record()

            # 执行 mm + rmsnorm + mm 操作
            for _ in range(num_repeats):
                # C = cutlass_gemm_with_prefetch.mm(A, B, C, D, 0.7, -1.0)
                E = cutlass_gemm_with_prefetch.rmsnorm(E, C, F,alpha,cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH)
                E = cutlass_gemm_with_prefetch.rmsnorm(E, C, F,alpha,cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH)
                E = cutlass_gemm_with_prefetch.rmsnorm(E, C, F,alpha,cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH)

                # C = cutlass_gemm_with_prefetch.mm(A, E, C, D , alpha, beta)

            # 结束计时
            end_event.record()
            torch.cuda.synchronize()

            # 计算平均时间（毫秒）
            elapsed_time = start_event.elapsed_time(end_event) / num_repeats
            results.append((alpha, beta, eps, elapsed_time))

            print(f"alpha={alpha}, beta={beta}, eps={eps} => {elapsed_time:.3f} ms")

    return results

if __name__ == "__main__":
    # 测试参数
    M = K = N = 2048
    alpha_values = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]  # 测试不同的 alpha 值
    beta_values = [-1.0 ]   

    # 运行基准测试
    results = benchmark_gemm_rmsnorm(M, K, N, alpha_values, beta_values)

    # 打印结果
    print("\nResults:")
    for alpha, beta, eps, time_ms in results:
        print(f"alpha={alpha:.1f}, beta={beta:.1f}, eps={eps} => {time_ms:.3f} ms")