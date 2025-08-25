#ifndef LAUNCH_UTILS
#define LAUNCH_UTILS
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>


enum class KERNEL_OVERLAP_HIERARCHY {
  NONE = 0,
  PDL = 1,
  FREFTECH = 2,
  SHAREDMEM = 3
};
/*
处理空参数的问题：
如果可变参数部分为空（即宏调用时未传递可变参数），某些编译器（如
GCC）可能会在展开时残留逗号，导致编译错误。 此时可以用 ##__VA_ARGS__（##
是预处理器的 “连接符”），它会自动删除可变参数为空时的多余逗号。
*/
#define LAUNCH_KERNEL_WITH_PDL(kernel, grid_dim, block_dim, smem_size, stream, \
                               ...)                                            \
  do {                                                                         \
    int device_;                                                               \
    cudaGetDevice(&device_);                                                   \
    int arch_;                                                                 \
    cudaDeviceGetAttribute(                                                    \
        &arch_, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, device_);   \
    if (arch_ >= 9) {                                                          \
      cudaLaunchConfig_t config_;                                              \
      config_.gridDim = (grid_dim);                                            \
      config_.blockDim = (block_dim);                                          \
      config_.dynamicSmemBytes = (smem_size);                                  \
      config_.stream = (stream);                                               \
      cudaLaunchAttribute attrs_[1];                                           \
      attrs_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;       \
      attrs_[0].val.programmaticStreamSerializationAllowed = true;             \
      config_.numAttrs = 1;                                                    \
      config_.attrs = attrs_;                                                  \
      cudaLaunchKernelEx(&config_, &(kernel), __VA_ARGS__);                    \
    } else {                                                                   \
      (kernel)<<<(grid_dim), (block_dim), (smem_size), (stream)>>>(            \
          __VA_ARGS__);                                                        \
    }                                                                          \
  } while (0)

#endif