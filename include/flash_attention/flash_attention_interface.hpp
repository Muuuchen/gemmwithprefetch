
#include "flash_attention.hpp"
#include "launcher.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cute/tensor.hpp>
// torch::Tensor (torch::Tensor q, torch::Tensor k, torch::Tensor v);


