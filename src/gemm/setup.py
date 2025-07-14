import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取 CUTLASS 目录
cutlass_dir = os.environ.get("CUTLASS_DIR", "/root/gemmwithprefetch/thirdparty/cutlass")
if not os.path.isdir(cutlass_dir):
    raise Exception("Environment variable CUTLASS_DIR must point to the CUTLASS installation") 

# 设置包含目录
_cutlass_include_dirs = ["tools/util/include", "include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]
include_dirs = cutlass_include_dirs + ["include"]
print("@@@@@ Include directories:", include_dirs)

# NVCC 编译标志
nvcc_flags = [
    "-O3",
    "-DNDEBUG",
    "-std=c++17",
    "--generate-code=arch=compute_90a,code=[sm_90a]",  
    "-DCOMPILE_3X_HOPPER",  
    "-DCUTLASS_ENABLE_GDC_FOR_SM90=ON",
    "--expt-relaxed-constexpr",  # 允许更宽松的 constexpr
]

# C++ 编译标志
cxx_flags = [
    "-O3",
    "-DNDEBUG",
    "-std=c++17",
    "-fPIC",  # 位置无关代码
]

# 链接库和标志
library_dirs = [
    "/usr/local/cuda/lib64",
]

libraries = [
"cudart", "cuda"
]

setup(
    name='cutlass_gemm_with_prefetch',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name="cutlass_gemm_with_prefetch",
            sources=["gemm_with_prefetch.cu"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={
                'nvcc': nvcc_flags,
                'cxx': cxx_flags
            },
            extra_link_args=[
                f"-L{dir}" for dir in library_dirs  # 显式指定库路径
            ] + [
                f"-l{lib}" for lib in libraries  # 显式链接所有库
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(
            use_ninja=True,
            no_python_abi_suffix=False,  # 保留 Python ABI 后缀
        )
    },
    zip_safe=False,
)