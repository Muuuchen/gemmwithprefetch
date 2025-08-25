import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取目录信息
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")

# 获取 CUTLASS 目录
cutlass_dir = os.environ.get("CUTLASS_DIR", "/root/gemmwithprefetch/thirdparty/cutlass")
if not os.path.isdir(cutlass_dir):
    raise Exception("Environment variable CUTLASS_DIR must point to the CUTLASS installation") 

# 包含目录
_cutlass_include_dirs = ["tools/util/include", "include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]

project_include_dirs = [
    os.path.join(project_root, "include"),
]

include_dirs = cutlass_include_dirs + project_include_dirs + [
    "/usr/local/cuda/include",
]

print("Include directories:")
for inc in include_dirs:
    print(f"  {inc}")

# 最简单的编译标志
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--generate-code=arch=compute_90a,code=[sm_90a]",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

cxx_flags = [
    "-O3", 
    "-std=c++17",
]

# 源文件
source_files = [
    os.path.join(project_root, "src/gemm/gemm_with_prefetch.cu"),
    os.path.join(project_root, "src/activations/rmsnorm.cu"),
    os.path.join(project_root, "src/attention/flash_attn.cu"),
    os.path.join(project_root, "src/binding.cu"),
    
]

print(f"Source files: {source_files}")

# 验证文件存在
for src in source_files:
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file not found: {src}")

setup(
    name='cutlass_gemm_with_prefetch',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name="cutlass_gemm_with_prefetch",
            sources=source_files,
            include_dirs=include_dirs,
            extra_compile_args={
                'nvcc': nvcc_flags,
                'cxx': cxx_flags
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)