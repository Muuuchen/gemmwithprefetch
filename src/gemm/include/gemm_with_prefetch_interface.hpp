
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
#include "helper.h"

using namespace cute;

void cutlass_gemm_wrapper(int M, int N, int K,
                          cutlass::float_e4m3_t const *ptrA,
                          cutlass::float_e5m2_t const *ptrB,
                          cutlass::float_e4m3_t *ptrC,
                          cutlass::float_e4m3_t const *ptrD) {

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // A matrix configuration
  using ElementA = cutlass::float_e4m3_t; // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor; // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 /
      cutlass::sizeof_bits<
          ElementA>::value; // Memory access granularity/alignment of A matrix
                            // in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::float_e5m2_t; // Element type for B matrix operand
  using LayoutB =
      cutlass::layout::ColumnMajor; // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 /
      cutlass::sizeof_bits<
          ElementB>::value; // Memory access granularity/alignment of B matrix
                            // in units of elements (up to 16 bytes)

  // C matrix configuration
  using ElementC =
      cutlass::float_e4m3_t; // Element type for C and D matrix operands
  using LayoutC =
      cutlass::layout::ColumnMajor; // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 /
      cutlass::sizeof_bits<
          ElementC>::value; // Memory access granularity/alignment of C matrix
                            // in units of elements (up to 16 bytes)

  // D matrix configuration
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  // Core kernel configurations
  using ElementAccumulator = float;    // Element type for internal accumulation
  using ElementCompute = float;        // Element type for epilogue computation
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
  using TileShape = Shape<_64, _64, _128>; // Threadblock-level tile size
  // Cluster_N > 1 is not supported yet.
  using ClusterShape =
      Shape<_1, _1, _1>; // Shape of the threadblocks in a cluster
  using KernelSchedule = cutlass::gemm::
      KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
          LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, // Indicates ProblemShape
      CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  Gemm gemm;

  // Extract information from Gemm kernel.
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
  using ElementScalar = typename EpilogueOutputOp::ElementScalar;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  // options l means batch size
  stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto a_coord = cutlass::make_Coord(M * 1, K);
  auto c_coord = cutlass::make_Coord(M * 1, N);
  auto b_coord = cutlass::make_Coord(K, N * 1);

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     {M, N, K, 1},
                                     {ptrA, stride_A, ptrB, stride_B},
                                     {{}, // epilogue.thread
                                      ptrC,
                                      stride_C,
                                      ptrD,
                                      stride_D}};

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.00f;
  fusion_args.beta = 0.00f;
  //   fusion_args.alpha_ptr = scalar_alpha.device_data();
  //   fusion_args.beta_ptr = scalar_beta.device_data();
  ///////////////////////////////////
  arguments.mainloop.overlap_ratio = 0.5;
  arguments.mainloop.prefetch_ratio = 0.5;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run(nullptr, nullptr, /* launch_with_pdl = */ true));
}