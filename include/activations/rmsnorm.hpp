
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
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

template <typename T>
__global__ void rmsnorm_twoPassAlgo_e1_fp8_none(T *output, const T *input,
                                           const T *weight, const int m,
                                           const int n, float ratio,float epsilon) {
  const int tid = threadIdx.x;
  const int m_idx = blockIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;

  for (int index = tid; index < n; index += bdimx) {
    float local_val = static_cast<float>(input[index]);
    local_sums[0] += local_val * local_val;
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }

  __syncthreads();

  for (int index = tid; index < n; index += bdimx) {
    const T weight_val = weight[index];
    const T local_val = input[index];
    output[index] = T(static_cast<float>(local_val) * s_mean *
                      static_cast<float>(weight_val));
  }
}

template <typename T>
__global__ void rmsnorm_twoPassAlgo_e1_fp8_pdl(T *output, const T *input,
                                           const T *weight, const int m,
                                           const int n, float ratio,float epsilon) {
  const int tid = threadIdx.x;
  const int m_idx = blockIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;
  asm volatile("griddepcontrol.wait;");
  for (int index = tid; index < n; index += bdimx) {
    float local_val = static_cast<float>(input[index]);
    local_sums[0] += local_val * local_val;
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  asm volatile("griddepcontrol.launch_dependents;");
  __syncthreads();

  for (int index = tid; index < n; index += bdimx) {
    const T weight_val = weight[index];
    const T local_val = input[index];
    output[index] = T(static_cast<float>(local_val) * s_mean *
                      static_cast<float>(weight_val));
  }
}


template <typename T>
__global__ void rmsnorm_twoPassAlgo_e1_fp8_shared_mem(
    T *output, const T *input, const T *weight, 
    const int m, const int n, float ratio, float epsilon) {
    
    extern __shared__ float shared_mem[];
    float* weight_s = shared_mem;
    
    const int tid = threadIdx.x;
    const int m_idx = blockIdx.x;
    const int bdimx = blockDim.x;
    

    for (int index = tid; index < n; index += bdimx) {
        weight_s[index] = static_cast<float>(weight[index]);
    }
    __syncthreads();
    
    __shared__ float s_mean;
    float local_sums[1] = {0.0f};
    int offset = m_idx * n;
    input += offset;
    output += offset;
    
    asm volatile("griddepcontrol.wait;");

    for (int index = tid; index < n; index += bdimx) {
        float local_val = static_cast<float>(input[index]);
        local_sums[0] += local_val * local_val;
    }
    

    if (blockDim.x <= 32) {
        warpReduceSum<float, 1>(local_sums);
    } else {
        blockReduceSum<float, 1>(local_sums);
    }
    
    if (threadIdx.x == 0) {
        s_mean = rsqrtf(local_sums[0] / n + epsilon);
    }
    
    asm volatile("griddepcontrol.launch_dependents;");
    __syncthreads();
    

    for (int index = tid; index < n; index += bdimx) {
        const T local_val = input[index];
        output[index] = T(static_cast<float>(local_val) * s_mean * weight_s[index]);
    }
}


template <typename T>
__global__ void rmsnorm_twoPassAlgo_e1_fp8_prefetch(T *output, const T *input,
                                           const T *weight, const int m,
                                           const int n, float ratio,float epsilon) {
  const int tid = threadIdx.x;
  
  if (tid == 0) {
    uint32_t weight_bytes = (uint32_t(m*n*ratio) * sizeof(T));

    if (weight_bytes % 16 == 0 && weight_bytes <= 0xFFFFFFFF) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                   :
                   : "l"(weight), "r"(weight_bytes)
                   : "memory");
    }
  }
  const int m_idx = blockIdx.x;
  const int bdimx = blockDim.x;

  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;
  asm volatile("griddepcontrol.wait;");

  for (int index = tid; index < n; index += bdimx) {
    float local_val = static_cast<float>(input[index]);
    local_sums[0] += local_val * local_val;
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  asm volatile("griddepcontrol.launch_dependents;");
  __syncthreads();

  for (int index = tid; index < n; index += bdimx) {
    const T weight_val = weight[index];
    const T local_val = input[index];
    output[index] = T(static_cast<float>(local_val) * s_mean *
                      static_cast<float>(weight_val));
  }
}
