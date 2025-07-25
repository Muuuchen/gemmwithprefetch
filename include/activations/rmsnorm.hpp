#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

// Warp-level reduction
template <typename T, int NUM> __device__ void warpReduceSum(T *val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask);
    }
  }
}

// Block-level reduction
template <typename T, int NUM> __device__ void blockReduceSum(T *val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSum<T, NUM>(val);
}


// 原版本 - 使用prefetch
__global__ void rmsnorm_twoPassAlgo_e8(float4* output, const float4* input, const float4* weight,
                                       const int m, const int n, float epsilon)
{
    const int m_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_dim_x = blockDim.x;
    __shared__ float s_mean;
    float local_sums[1] = {0.0f};
    const int n_8 = n / 8;
    int offset = m_idx * n_8;

    input += offset;
    output += offset;

    // 预取weight数据
    if (tid == 0)
    {
        uint32_t weight_bytes = n_8 * sizeof(float4);
        // 确保是16的倍数且在32位范围内
        if (weight_bytes % 16 == 0 && weight_bytes <= 0xFFFFFFFF)
        {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                         :
                         : "l"(weight), "r"(weight_bytes)
                         : "memory");
        }
    }
    asm volatile("griddepcontrol.wait;");

    // 第一阶段：计算RMS（此时weight正在预取到cache）
    for (int index = tid; index < n_8; index += block_dim_x)
    {
        const float4 local_val = input[index];
        const half2* h1 = (half2*)&local_val.x;
        const half2* h2 = (half2*)&local_val.y;
        const half2* h3 = (half2*)&local_val.z;
        const half2* h4 = (half2*)&local_val.w;
        local_sums[0] += static_cast<float>(h1->x) * static_cast<float>(h1->x) +
                         static_cast<float>(h1->y) * static_cast<float>(h1->y) +
                         static_cast<float>(h2->x) * static_cast<float>(h2->x) +
                         static_cast<float>(h2->y) * static_cast<float>(h2->y) +
                         static_cast<float>(h3->x) * static_cast<float>(h3->x) +
                         static_cast<float>(h3->y) * static_cast<float>(h3->y) +
                         static_cast<float>(h4->x) * static_cast<float>(h4->x) +
                         static_cast<float>(h4->y) * static_cast<float>(h4->y);
    }

    if (blockDim.x < 32)
    {
        warpReduceSum<float, 1>(local_sums);
    }
    else
    {
        blockReduceSum<float, 1>(local_sums);
    }

    if (threadIdx.x == 0)
    {
        s_mean = rsqrtf(local_sums[0] / n + epsilon);
    }
    __syncthreads();

    asm volatile("griddepcontrol.launch_dependents;");

    // 第二阶段：应用归一化（使用预取的weight数据）
    for (int index = tid; index < n_8; index += block_dim_x)
    {
        const float4 local_val = input[index];
        const float4 weight_val = weight[index]; // 这里访问预取的数据

        const half2* l1 = (half2*)&local_val.x;
        const half2* l2 = (half2*)&local_val.y;
        const half2* l3 = (half2*)&local_val.z;
        const half2* l4 = (half2*)&local_val.w;

        const half2* w1 = (half2*)&weight_val.x;
        const half2* w2 = (half2*)&weight_val.y;
        const half2* w3 = (half2*)&weight_val.z;
        const half2* w4 = (half2*)&weight_val.w;

        float4 tmp;
        half2* h1 = (half2*)&tmp.x;
        half2* h2 = (half2*)&tmp.y;
        half2* h3 = (half2*)&tmp.z;
        half2* h4 = (half2*)&tmp.w;

        h1->x = half(static_cast<float>(l1->x) * s_mean * static_cast<float>(w1->x));
        h1->y = half(static_cast<float>(l1->y) * s_mean * static_cast<float>(w1->y));
        h2->x = half(static_cast<float>(l2->x) * s_mean * static_cast<float>(w2->x));
        h2->y = half(static_cast<float>(l2->y) * s_mean * static_cast<float>(w2->y));
        h3->x = half(static_cast<float>(l3->x) * s_mean * static_cast<float>(w3->x));
        h3->y = half(static_cast<float>(l3->y) * s_mean * static_cast<float>(w3->y));
        h4->x = half(static_cast<float>(l4->x) * s_mean * static_cast<float>(w4->x));
        h4->y = half(static_cast<float>(l4->y) * s_mean * static_cast<float>(w4->y));

        output[index] = tmp;
    }
}
