#include <pybind11/pybind11.h>

#include <cstdio>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <iostream>

#include <torch/extension.h>

#include <activations/rmsnorm_interface.hpp>

#include <gemm_fp8/gemm_with_prefetch_interface.hpp>

torch::Tensor cutlass_gemm_with_prefetch(torch::Tensor A, torch::Tensor B,
                                         c10::optional<torch::Tensor> out,
                                         torch::Tensor D,
                                         float overlap_ratio ,
                                         float prefetch_ratio );

torch::Tensor cutlass_rmsnorm(c10::optional<torch::Tensor> out,
                              torch::Tensor input, torch::Tensor weight,float ratio);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm",
        py::overload_cast<torch::Tensor, torch::Tensor,
                          c10::optional<torch::Tensor>, torch::Tensor, float,
                          float>(&cutlass_gemm_with_prefetch),
        py::arg("A"), py::arg("B"), py::arg("out"), py::arg("D"),
        py::arg("overlap_ratio"), py::arg("prefetch_ratio") = py::none());
  m.def("rmsnorm",
        py::overload_cast<c10::optional<torch::Tensor>, torch::Tensor,
                          torch::Tensor,float>(&cutlass_rmsnorm),
        py::arg("out"), py::arg("input"), py::arg("weight"),py::arg("ratio") = py::none());
}