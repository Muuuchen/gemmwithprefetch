import time
import torch
import math

import cutlass_gemm_with_prefetch

M = K = N = 8192
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, K)).to(device=cuda).to(dtype=torch.float8_e4m3fn)
B = torch.normal(0,1,size=(K, N)).to(device=cuda).to(dtype=torch.float8_e5m2)
C = torch.empty((M, N), device=cuda, dtype=torch.float8_e4m3fn)
D = torch.normal(0,1,size=(M, N)).to(device=cuda).to(dtype=torch.float8_e4m3fn)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

C = cutlass_gemm_with_prefetch.mm(A, B, C, D,0.5,0.5)
C = cutlass_gemm_with_prefetch.mm(A, B, C, D,0.5,0.5)

end_event.record()

# 等待事件完成
torch.cuda.synchronize()

# 计算执行时间（毫秒）
elapsed_time = start_event.elapsed_time(end_event)



print(f"{elapsed_time} ms")
