#include <algorithm>
#include <cmath>
#include <cstddef>
#include<cuda.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include "cutlass/numeric_types.h"  
#include <future>
#include <torch/extension.h>

#include "launcher.hpp"
#include "helper.hpp"
#include "addResidual/add.hpp"


template< typename T>
void addResidual_wrapper(T* output,const T* input, const T* residual,const int m,
const int n){
    cudaStream_t stream = 0;
    int blocks_per_row = ceil((float)(n) / 1024);
    size_t shm_size = sizeof(T)*n;
    dim3 grid(m, blocks_per_row);
    dim3 block(std::min(n, 1024));
    LAUNCH_KERNEL_WITH_PDL(addBiasAttention_pdl<T>,  grid,block, shm_size, stream, output, input, residual, n);
}

torch::Tensor add_residual(torch::Tensor input, torch::Tensor residual){
    int m = input.size(0);
    int n = input.size(1);

    auto out = torch::empty_like(input);
    addResidual_wrapper<cutlass::float_e4m3_t>(
        reinterpret_cast<cutlass::float_e4m3_t*>(out.data_ptr()),
        reinterpret_cast<cutlass::float_e4m3_t*>(input.data_ptr()),
        reinterpret_cast<cutlass::float_e4m3_t*>(residual.data_ptr()),
        m, n
    );    return out;
}