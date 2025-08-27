import os 
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
# from flashinfer import single_prefill_with_kv_cache
import cutlass_gemm_with_prefetch  # 导入你编译的模块


dim1 = 1024
dim2= 2048

input = torch.randn(dim1,dim2).cuda().to(torch.float8_e4m3fn)
residual = torch.randn(dim1,dim2).cuda().to(torch.float8_e4m3fn)

out = cutlass_gemm_with_prefetch.add_residual(input,residual)
print(out)