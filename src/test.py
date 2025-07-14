import torch
import math

import cutlass_gemm_with_prefetch

M = K = N = 4096
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, K)).to(device=cuda).to(dtype=torch.float8_e4m3fn)
B = torch.normal(0,1,size=(K, N)).to(device=cuda).to(dtype=torch.float8_e5m2)
C = torch.empty((M, N), device=cuda, dtype=torch.float8_e4m3fn)
D = torch.normal(0,1,size=(M, N)).to(device=cuda).to(dtype=torch.float8_e4m3fn)
C = cutlass_gemm_with_prefetch.mm(A, B,C,D)

