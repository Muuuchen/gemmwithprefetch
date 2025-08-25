import torch
import numpy as np
import cutlass_gemm_with_prefetch  # 导入你编译的模块

# --- 1. 基准测试配置 ---
# 你可以根据你的 GPU 和测试需求调整这些参数
BATCH_SIZE = 16
SEQ_LEN = 4096
HIDDEN_SIZE = 512
DTYPE = torch.float8_e4m3fn 
DEVICE = "cuda"

WARMUP_ITER = 20
TIMING_ITER = 100

print(f"--- Benchmark Configuration ---")
print(f"Device: {torch.cuda.get_device_name(DEVICE)}")
print(f"Tensor Shape: ({BATCH_SIZE}, {SEQ_LEN}, {HIDDEN_SIZE})")
print(f"Dtype: {DTYPE}")
print(f"Warm-up Iterations: {WARMUP_ITER}")
print(f"Timing Iterations: {TIMING_ITER}\n")


def benchmark_rmsnorm(input_tensor, weight_tensor, out_tensor, ratio, hierarchy_mode):
    """
    一个用于对 RMSNorm 算子进行预热和计时的函数。
    
    返回:
        avg_time_ms (float): 平均执行时间（毫秒）
        throughput_gb_s (float): 吞吐量（GB/s）
    """
    # --- 预热 ---
    for _ in range(WARMUP_ITER):
        _ = cutlass_gemm_with_prefetch.rmsnorm(out_tensor, input_tensor, weight_tensor, ratio, hierarchy_mode)
    # 确保所有预热操作都已完成
    torch.cuda.synchronize()

    # --- 精确计时 ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(TIMING_ITER):
        _ = cutlass_gemm_with_prefetch.rmsnorm(out_tensor, input_tensor, weight_tensor, ratio, hierarchy_mode)
    end_event.record()

    # 等待所有计时操作完成
    torch.cuda.synchronize()

    # 计算总时间和平均时间
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / TIMING_ITER

    # 计算吞吐量
    # RMSNorm 需要读取 input 和 weight，并写入 output
    # total_bytes = (input_bytes + weight_bytes + output_bytes)
    bytes_per_iter = (input_tensor.numel() * input_tensor.element_size() +
                      weight_tensor.numel() * weight_tensor.element_size() +
                      out_tensor.numel() * out_tensor.element_size())
    
    total_bytes = bytes_per_iter * TIMING_ITER
    throughput_gb_s = (total_bytes / (1024**3)) / (elapsed_time_ms / 1000)

    return avg_time_ms, throughput_gb_s


def main():
    # --- 2. 准备输入数据 ---
    print("--- Preparing Tensors ---")
    # 根据你的算子绑定顺序，参数应该是 out, input, weight
    # D 对应 input, F 对应 weight, E 对应 out
    source_dtype = torch.bfloat16 

    input_tensor_D = torch.randn(
        BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=source_dtype, device=DEVICE
    ).to(DTYPE)

    weight_tensor_F = torch.randn(
        HIDDEN_SIZE, dtype=source_dtype, device=DEVICE
    ).to(DTYPE)

    # 输出张量 out_tensor_E 应该和输入张量 input_tensor_D 的形状和类型都一致
    # 所以这里不需要修改
    out_tensor_E = torch.empty_like(input_tensor_D)
    
    print(f"Input Tensor (D): {input_tensor_D.shape}, dtype={input_tensor_D.dtype}")
    print(f"Weight Tensor (F): {weight_tensor_F.shape}, dtype={weight_tensor_F.dtype}\n")

    # 获取枚举类，方便使用
    Hierarchy = [cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE,
                 cutlass_gemm_with_prefetch.KernelOverlapHierarchy.PDL,
                 cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH,
                 cutlass_gemm_with_prefetch.KernelOverlapHierarchy.SHAREDMEM]

    # --- 3. 循环测试所有场景 ---
    print("--- Starting Benchmark ---")
    
    # 遍历 KernelOverlapHierarchy 的所有成员
    for hierarchy_mode in Hierarchy:
        
        # 根据你的要求，确定需要测试的 ratio 值
        if hierarchy_mode == cutlass_gemm_with_prefetch.KernelOverlapHierarchy.FREFTECH:
            # 当模式为 FREFTECH 时，测试从 0.0 到 1.0 的 ratio
            # np.linspace(0.0, 1.0, 6) 会生成 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ratios_to_test = np.linspace(0.0, 1.0, 6)
        else:
            # 对于其他模式，使用一个固定的 ratio 值（例如 0.5）
            ratios_to_test = [0.5]
            
        for ratio in ratios_to_test:
            print(f"Testing Mode: {hierarchy_mode.name:<10} | Ratio: {ratio:.2f}")
            
            try:
                avg_time, throughput = benchmark_rmsnorm(
                    input_tensor=input_tensor_D,
                    weight_tensor=weight_tensor_F,
                    out_tensor=out_tensor_E,
                    ratio=ratio,
                    hierarchy_mode=hierarchy_mode
                )
                print(f"  -> Avg Time: {avg_time:.4f} ms | Throughput: {throughput:.2f} GB/s")
            except Exception as e:
                print(f"  -> ERROR during benchmark: {e}")
            print("-" * 40)

if __name__ == "__main__":
    main()