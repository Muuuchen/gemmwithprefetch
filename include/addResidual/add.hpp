
#include <cuda.h>
#include <cuda_runtime.h>


// normally residula is done when last kernel is completed

template <typename T>
__global__ void addBiasAttention_none(T*  output, const T* input, const T* residual,const int n ){
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if(col_index < n){
        output[blockIdx.x *n + col_index] = input[blockIdx.x * n + col_index] + residual[blockIdx.x * n + col_index];
    }
}



template <typename T>
__global__ void addBiasAttention_pdl(T*  output, const T* input, const T* residual,const int n ){
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    asm volatile("griddepcontrol.wait;");
    if(col_index < n){
        output[blockIdx.x *n + col_index] = input[blockIdx.x * n + col_index] + residual[blockIdx.x * n + col_index];
    }
    asm volatile("griddepcontrol.launch_dependents;");
}


template <typename T>
__global__ void addBiasAttention_prefetch(T*  output, const T* input, const T* residual,const int n ){
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index == 0) {
        uint32_t weight_bytes = (uint32_t(n/2) * sizeof(T));
        if (weight_bytes % 16 == 0 && weight_bytes <= 0xFFFFFFFF) {
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                       :
                       : "l"(residual), "r"(weight_bytes)
                       : "memory");
        }
      }
    asm volatile("griddepcontrol.wait;");
    if(col_index < n){
        output[blockIdx.x *n + col_index] = input[blockIdx.x * n + col_index] + residual[blockIdx.x * n + col_index];
    }
    asm volatile("griddepcontrol.launch_dependents;");
}


template <typename T>
__global__ void addBiasAttention_shared_mem(T*  output, const T* input, const T* residual,const int n ){
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ T shared_mem[];
    T* weight_s = shared_mem;
    if(col_index < n){
        weight_s[threadIdx.x] = residual[blockIdx.x*n +col_index];
    }
    asm volatile("griddepcontrol.wait;");
    if(col_index < n){
        output[blockIdx.x *n + col_index] = input[blockIdx.x * n + col_index] + weight_s[threadIdx.x];
    }
    asm volatile("griddepcontrol.launch_dependents;");
}
