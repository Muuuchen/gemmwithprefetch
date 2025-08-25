import os 
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
# from flashinfer import single_prefill_with_kv_cache
import cutlass_gemm_with_prefetch  # 导入你编译的模块


# Add a new environment variable  
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

REMOVE_NVCC_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]
for flag in REMOVE_NVCC_FLAGS:
    try:
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass


torch.manual_seed(0)


def manual_attn(q, k, v, attn_mask=None):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if attn_mask != None:
        att.masked_fill_(attn_mask, float('-inf'))  # Apply mask
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

batch_size = 1
n_head = 32
q_len = 1024 
kv_len = q_len
head_embd = 64 

q = torch.randn(batch_size, q_len, n_head, head_embd).cuda().half()
k = torch.randn(batch_size, kv_len, n_head, head_embd).cuda().half()
v = torch.randn(batch_size, kv_len, n_head, head_embd).cuda().half()
q1 = q.transpose(1, 2).contiguous()
k1 = k.transpose(1, 2).contiguous()
v1 = v.transpose(1, 2).contiguous()
q2 = q.reshape(batch_size * q_len, n_head, head_embd)
k2 = k.reshape(batch_size * kv_len, n_head, head_embd)
v2 = v.reshape(batch_size * kv_len, n_head, head_embd)

a = manual_attn(q1, k1, v1)
# c = single_prefill_with_kv_cache(q2, k2, v2)
d = cutlass_gemm_with_prefetch.fa(q, k, v)
print('attn values sanity check:', torch.allclose(a, d, rtol=1e-03, atol=1e-03))