#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

#include "activations/rmsnorm_interface.hpp"
#include "helper.hpp"
#include "launcher.hpp"
template <>
void cutlass_rmsnorm_warpper(int m, int n, cutlass::float_e4m3_t *output,
                             cutlass::float_e4m3_t const *input,
                             cutlass::float_e4m3_t const *weight) {
  cudaStream_t stream = 0;
  const size_t shmem_size = 48;

  dim3 grid(m);
  dim3 block(
      cutlass::platform::min(1024, ((n + 31) / 32 + 31) / 32 * 32)); // LAUNCH
  // rmsnorm_twoPassAlgo_e1_fp8<<<grid, block, 0>>>(output, input, weight, m, n,
  //                                                1e-5);
  LAUNCH_KERNEL_WITH_PDL(rmsnorm_twoPassAlgo_e1_fp8<cutlass::float_e4m3_t>,
                         grid, block, shmem_size, stream, output, input, weight,
                         m, n, 1e-5);
}

void cutlass_rmsnorm_unpack(torch::Tensor output, torch::Tensor input,
                            torch::Tensor weight) {
  const int m = input.sizes()[0];
  const int n = input.sizes()[1];
  cutlass::float_e4m3_t const *ptrInput =
      reinterpret_cast<cutlass::float_e4m3_t *>(input.data_ptr());
  cutlass::float_e4m3_t const *ptrWeight =
      reinterpret_cast<cutlass::float_e4m3_t *>(weight.data_ptr());
  cutlass::float_e4m3_t *ptrOutput =
      reinterpret_cast<cutlass::float_e4m3_t *>(output.data_ptr());
  cutlass_rmsnorm_warpper(m, n, ptrOutput, ptrInput, ptrWeight);
}

void cutlass_rmsnorm_typecheck(torch::Tensor Output, torch::Tensor Input,
                               torch::Tensor Weight) {
  if (Input.dtype() != torch::kFloat8_e4m3fn ||
      Weight.dtype() != torch::kFloat8_e4m3fn ||
      Output.dtype() != torch::kFloat8_e4m3fn) {
    throw std::runtime_error("Unsupported data type for A");
  } else {
    cutlass_rmsnorm_unpack(Output, Input, Weight);
  }
}

torch::Tensor cutlass_rmsnorm(c10::optional<torch::Tensor> out,
                              torch::Tensor input, torch::Tensor weight) {
  torch::Tensor O;
  if (out.has_value()) {
    O = out.value();
  } else {
    const int m = input.sizes()[0];
    const int n = input.sizes()[1];

    auto o_options =
        torch::TensorOptions().device(torch::kCUDA).dtype(input.dtype());
    O = torch::empty({m, n}, o_options);
  }
  if (!(O.device().is_cuda() && input.device().is_cuda() &&
        weight.device().is_cuda())) {
    throw std::runtime_error("cutlass_rmsnorm only supports CUDA tensors");
  }
  torch::Tensor _input = input.contiguous();
  torch::Tensor _weight = weight.contiguous();
  torch::Tensor _output = O.contiguous();
  cutlass_rmsnorm_typecheck(_output, _input, _weight);
  if (!_output.is_contiguous()) {
    O.copy_(_output);
  }
  return O;
}
