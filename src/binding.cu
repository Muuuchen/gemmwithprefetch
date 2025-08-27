#include <pybind11/pybind11.h>

#include <cstdio>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <iostream>

#include <torch/extension.h>

#include <activations/rmsnorm_interface.hpp>
#include <launcher.hpp>
#include <gemm_fp8/gemm_with_prefetch_interface.hpp>
#include <flash_attention/flash_attention.hpp>
#include <addResidual/add.hpp>

torch::Tensor add_residual(torch::Tensor input, torch::Tensor residual);
torch::Tensor cutlass_gemm_with_prefetch(torch::Tensor A, torch::Tensor B,
                                         c10::optional<torch::Tensor> out,
                                         torch::Tensor D,
                                         float overlap_ratio ,
                                         float prefetch_ratio );

torch::Tensor cutlass_rmsnorm(c10::optional<torch::Tensor> out,
                              torch::Tensor input, torch::Tensor weight,float ratio);

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) ;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

      py::enum_<KERNEL_OVERLAP_HIERARCHY>(m, "KernelOverlapHierarchy")
    .value("NONE", KERNEL_OVERLAP_HIERARCHY::NONE)
    .value("PDL", KERNEL_OVERLAP_HIERARCHY::PDL)
    .value("FREFTECH", KERNEL_OVERLAP_HIERARCHY::FREFTECH)
    .value("SHAREDMEM", KERNEL_OVERLAP_HIERARCHY::SHAREDMEM)
    .export_values();
    m.def("add_residual",
      py::overload_cast<torch::Tensor, torch::Tensor>(&add_residual),
        py::arg("input"), py::arg("residual"));
m.def("fa",
      py::overload_cast<torch::Tensor, torch::Tensor,
                          torch::Tensor>(&flash_attention),
        py::arg("Q"), py::arg("K"), py::arg("V")
      );
  m.def("mm",
        py::overload_cast<torch::Tensor, torch::Tensor,
                          c10::optional<torch::Tensor>, torch::Tensor, float,
                          float>(&cutlass_gemm_with_prefetch),
        py::arg("A"), py::arg("B"), py::arg("out"), py::arg("D"),
        py::arg("overlap_ratio"), py::arg("prefetch_ratio") = py::none());
  m.def("rmsnorm",
        py::overload_cast<c10::optional<torch::Tensor>, torch::Tensor,
                          torch::Tensor,float,KERNEL_OVERLAP_HIERARCHY>(&cutlass_rmsnorm),
        py::arg("out"), py::arg("input"), py::arg("weight"),py::arg("ratio"),py::arg("hierarchy"));
}