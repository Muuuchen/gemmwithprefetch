#pragma once

#include "rmsnorm.hpp"
#include "launcher.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

torch::Tensor cutlass_rmsnorm(c10::optional<torch::Tensor> out,
                              torch::Tensor input, torch::Tensor weight, float ratio,KERNEL_OVERLAP_HIERARCHY hierarchy = KERNEL_OVERLAP_HIERARCHY::NONE);

template <typename T>
void cutlass_rmsnorm_warpper(int m, int n, T *output, T const *input,
                             T const *weight, float ratio, KERNEL_OVERLAP_HIERARCHY hierarchy);
