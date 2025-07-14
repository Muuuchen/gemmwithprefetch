#include <torch/extension.h>

#include <cstdio>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include <pybind11/pybind11.h>

#include "include/gemm_with_prefetch_interface.hpp"

void cutlass_gemm_wrapper(int M, int N, int K, const int *ptrA, const int *ptrB,
                          int *ptrC, const int *ptrD);

void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                         torch::Tensor D) {
  const int M = A.sizes()[0];
  const int N = B.sizes()[1];
  const int K = A.sizes()[1];
  cutlass::float_e4m3_t const *ptrA =
      reinterpret_cast<cutlass::float_e4m3_t *>(A.data_ptr());
  cutlass::float_e5m2_t const *ptrB =
      reinterpret_cast<cutlass::float_e5m2_t *>(B.data_ptr());
  cutlass::float_e4m3_t *ptrC =
      reinterpret_cast<cutlass::float_e4m3_t *>(C.data_ptr());
  cutlass::float_e4m3_t const *ptrD =
      reinterpret_cast<cutlass::float_e4m3_t *>(D.data_ptr());
  cutlass_gemm_wrapper(M, N, K, ptrA, ptrB, ptrC, ptrD);
}

void cutlass_gemm_with_prefetch_type_check(torch::Tensor A, torch::Tensor B,
                                           torch::Tensor C, torch::Tensor D) {
  if (A.dtype() != torch::kFloat8_e4m3fn || B.dtype() != torch::kFloat8_e5m2 ||
      C.dtype() != torch::kFloat8_e4m3fn ||
      D.dtype() != torch::kFloat8_e4m3fn) {
    throw std::runtime_error("Unsupported data type for A");
  } else {
    cutlass_gemm_unpack(A, B, C, D);
  }
}

torch::Tensor cutlass_gemm_with_prefetch(torch::Tensor A, torch::Tensor B,
                                         c10::optional<torch::Tensor> out,
                                         torch::Tensor D) {
  torch::Tensor C;
  if (out.has_value()) { // Output tensor was provided. So we will use it.
    C = out.value();
  } else {
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];

    auto c_options =
        torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }
  // Check that all tensors are allocated on GPU device.
  if (!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device. Use "
                                ".to(device=torch.device('cuda'))");
  torch::Tensor _A = A.contiguous();
  torch::Tensor _B = B.contiguous();
  torch::Tensor _C = C.contiguous();
  torch::Tensor _D = D.contiguous();

  cutlass_gemm_with_prefetch_type_check(_A, _B, _C, _D);
  if (!C.is_contiguous())
    C.copy_(_C);

  // Return the Torch tensor back to PyTorch
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm",
        py::overload_cast<torch::Tensor, torch::Tensor,
                          c10::optional<torch::Tensor>, torch::Tensor>(
            &cutlass_gemm_with_prefetch),
        py::arg("A"), py::arg("B"), py::arg("out"), py::arg("D") = py::none());
}