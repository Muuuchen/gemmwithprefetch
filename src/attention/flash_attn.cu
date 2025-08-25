#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include "cute/tensor.hpp"
#include "flash_attention/flash_attention.hpp"



torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  int bs = q.size(0);
  int head_num = q.size(1);
  int q_len = q.size(2);
  int head_dim = q.size(3);
  int k_len = k.size(2);

  int head_stride = q.stride(1);

  auto out = torch::empty_like(q);

  float sm_scale = 1.0 / sqrt(head_dim) * M_LOG2E;

  // only for head_dim=64
  config::FlashConfig<cute::half_t> config;
  dim3 block = config.kThreadNum;
  dim3 grid((q_len + config.kBlockM - 1) / config.kBlockM, bs * head_num);
  int shm_size = config.kShmSize;
  auto partition_kernel = flash_forward<decltype(config)>;
  cudaFuncSetAttribute(partition_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  partition_kernel<<<grid, block, shm_size>>>(
      (void*)out.data_ptr(), (const void*)q.data_ptr(),
      (const void*)k.data_ptr(), (const void*)v.data_ptr(), head_stride, q_len,
      k_len, sm_scale);
  return out;
}