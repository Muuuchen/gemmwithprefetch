

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "collective/builder.hpp"
#include "collective/dispatch_policy_extra.hpp"
#include "kernel/sm90_gemm_tma_warpspecialized_with_prefetch.hpp"

#include "gemm_with_weight_prefetch_commandline.hpp"
#include "helper.hpp"

using namespace cute;

torch::Tensor cutlass_gemm_with_prefetch(torch::Tensor A, torch::Tensor B,
                                         c10::optional<torch::Tensor> out,
                                         torch::Tensor D,
                                         float overlap_ratio = 0.5f,
                                         float prefetch_ratio = 0.5f);
void cutlass_gemm_wrapper(int M, int N, int K,
                          cutlass::float_e4m3_t const *ptrA,
                          cutlass::float_e5m2_t const *ptrB,
                          cutlass::float_e4m3_t *ptrC,
                          cutlass::float_e4m3_t const *ptrD,
                          float overlap_ratio, float prefetch_ratio);

