# install bear for compile commands

## rmsnorm
结论；能够达到10%的收益
Tensor Shape: (16, 4096, 512)
Dtype: torch.float8_e4m3fn
Warm-up Iterations: 20
Timing Iterations: 100

--- Preparing Tensors ---
Input Tensor (D): torch.Size([16, 4096, 512]), dtype=torch.float8_e4m3fn
Weight Tensor (F): torch.Size([512]), dtype=torch.float8_e4m3fn

--- Starting Benchmark ---
Testing Mode: NONE       | Ratio: 0.50
  -> Avg Time: 0.0153 ms | Throughput: 4096.09 GB/s
----------------------------------------
Testing Mode: PDL        | Ratio: 0.50
  -> Avg Time: 0.0141 ms | Throughput: 4421.97 GB/s
----------------------------------------
Testing Mode: FREFTECH   | Ratio: 0.00
  -> Avg Time: 0.0141 ms | Throughput: 4434.52 GB/s
----------------------------------------
Testing Mode: FREFTECH   | Ratio: 0.20
  -> Avg Time: 0.0143 ms | Throughput: 4375.90 GB/s
----------------------------------------
Testing Mode: FREFTECH   | Ratio: 0.40
  -> Avg Time: 0.0143 ms | Throughput: 4376.88 GB/s
----------------------------------------
Testing Mode: FREFTECH   | Ratio: 0.60
  -> Avg Time: 0.0143 ms | Throughput: 4376.59 GB/s
----------------------------------------
Testing Mode: FREFTECH   | Ratio: 0.80
  -> Avg Time: 0.0143 ms | Throughput: 4374.73 GB/s
----------------------------------------
Testing Mode: FREFTECH   | Ratio: 1.00
  -> Avg Time: 0.0143 ms | Throughput: 4380.42 GB/s
----------------------------------------
Testing Mode: SHAREDMEM  | Ratio: 0.50
  -> Avg Time: 0.0136 ms | Throughput: 4595.84 GB/s


## MOE
// shape bucket

=- 【】 实验上证实， 理论上证明 
 【    】 【    】
    【      】 
   

//搜索空间， 贪心子结构

## TODO 
- [ ] shmem的ratio测试
- 不同的配置会有很大的影响，甚至有30%的收益
- 可以看到在gemm和rmsnorm的组合下， 采用shmem的策略会影响很大


- [] shape 下的性能曲线
- [] profile 具体的shmem 和具体带宽，看能不能总结出一些规律
- [] 扩展更多的模型

- [] 线上模型的测试，r1 ossm 固定规模下性能收益
- [] 代码修改路线 -[] 

-- 预期之外的行为， 怎么考虑

- jindu huanman 



➜  (workspace) gemmwithprefetch git:(main U:4 ?:3) ✗ python test/test_decoder.py
============================================================
CUTLASS Transformer Block Performance Test
============================================================
Configuration:
  Batch size: 1
  Sequence length: 128
  Model dimension: 2048
  Number of heads: 32
  Head dimension: 64
  FFN dimension: 4096

Creating model...

--- Functional Test ---
Input shape: torch.Size([1, 128, 2048]), dtype: torch.float32
✓ Forward pass successful!
Output shape: torch.Size([1, 128, 2048]), dtype: torch.float8_e4m3fn
✓ Output values are valid

============================================================
End-to-End Performance Benchmark
============================================================

Batch    SeqLen   Latency(ms)  Tokens/sec   TFLOPS    
------------------------------------------------------------
  Warmup (10 iterations)...
  Testing (100 iterations)...
1        64       0.271        235821       15.89     
  Warmup (10 iterations)...
  Testing (100 iterations)...
1        128      0.272        470736       31.84     
  Warmup (10 iterations)...
  Testing (100 iterations)...
4        64       0.267        959559       64.65     
  Warmup (10 iterations)...
  Testing (100 iterations)...
4        128      0.329        1555337      105.19    
  Warmup (10 iterations)...
  Testing (100 iterations)...
8        64       0.320        1599829      107.78    
  Warmup (10 iterations)...
  Testing (100 iterations)...
8        128      0.498        2055851      139.04    

============================================================
Performance Summary
============================================================

Best Throughput:
  Config: Batch=8, SeqLen=128
  Throughput: 2055851 tokens/sec
  Latency: 0.498 ms
  Performance: 139.04 TFLOPS

Best Latency:
  Config: Batch=4, SeqLen=64
  Latency: 0.267 ms
  Throughput: 959559 tokens/sec
  Performance: 64.65 TFLOPS

Average Performance:
  Latency: 0.326 ms
  Throughput: 1146189 tokens/sec
  TFLOPS: 77.40

============================================================
Model Information
============================================================
Total parameters: 33,572,864 (33.57M)

Parameter details:

============================================================
GPU Information
============================================================
Device: NVIDIA H100 PCIe
Compute Capability: (9, 0)
Total Memory: 85.0 GB
Allocated Memory: 34.9 MB
Reserved Memory: 111.1 MB
➜  (workspace) gemmwithprefetch git:(main U:4 ?:3) ✗ 
➜  (workspace) gemmwithprefetch git:(main U:4 ?:4) ✗ python test/te
test_attention.py  test_decoder.py    test_moe.py        test_rmsnorm.py    test_torch.py      
➜  (workspace) gemmwithprefetch git:(main U:4 ?:4) ✗ python test/test_torch.py 
================================================================================
PyTorch vs CUTLASS Transformer Block Performance Comparison
================================================================================
Configuration:
  Model dimension: 2048
  Number of heads: 32
  Head dimension: 64
  FFN dimension: 4096

Creating PyTorch model...
Creating CUTLASS model...

================================================================================
Performance Results
================================================================================

Config               PyTorch                        CUTLASS                        Speedup   
==================== ============================== ============================== ==========
Batch x SeqLen       Latency(ms) / Tokens/s         Latency(ms) / Tokens/s         Ratio     
------------------------------------------------------------------------------------------

1 x 64                Warmup (10 iterations)...
  Testing (100 iterations)...
0.470 ms / 136121               Warmup (10 iterations)...
  Testing (100 iterations)...
0.268 ms / 238751             1.75x

1 x 128               Warmup (10 iterations)...
  Testing (100 iterations)...
0.465 ms / 275500               Warmup (10 iterations)...
  Testing (100 iterations)...
0.268 ms / 478099             1.74x

4 x 64                Warmup (10 iterations)...
  Testing (100 iterations)...
0.490 ms / 522048               Warmup (10 iterations)...
  Testing (100 iterations)...
0.265 ms / 966201             1.85x

4 x 128               Warmup (10 iterations)...
  Testing (100 iterations)...
0.494 ms / 1035607              Warmup (10 iterations)...
  Testing (100 iterations)...
0.333 ms / 1539440            1.49x

8 x 64                Warmup (10 iterations)...
  Testing (100 iterations)...
0.488 ms / 1048315              Warmup (10 iterations)...
  Testing (100 iterations)...
0.335 ms / 1527980            1.46x

8 x 128               Warmup (10 iterations)...
  Testing (100 iterations)...
0.638 ms / 1604409              Warmup (10 iterations)...
  Testing (100 iterations)...
0.520 ms / 1967701            1.23x

================================================================================
Summary
================================================================================

PyTorch Average Performance:
  Latency: 0.508 ms
  Throughput: 770333 tokens/sec
  TFLOPS: 52.03

CUTLASS Average Performance:
  Latency: 0.331 ms
  Throughput: 1119696 tokens/sec
  TFLOPS: 75.61

Average Speedup (CUTLASS vs PyTorch):
  Latency: 1.53x faster
  Throughput: 1.45x higher
  TFLOPS: 1.45x higher

================================================================================
GPU Information
================================================================================
Device: NVIDIA H100 PCIe
Compute Capability: (9, 0)
Total Memory: 85.0 GB

Memory Usage:
  Allocated: 100.7 MB
  Reserved: 209.7 MB


    (workspace) gemmwithprefetch git:(main U:4 ?:6) ✗ python test/test_decoder.py
============================================================
CUTLASS Transformer Block Performance Test
============================================================
Configuration:
  Batch size: 1
  Sequence length: 128
  Model dimension: 2048
  Number of heads: 32
  Head dimension: 64
  FFN dimension: 4096

Creating model...

--- Functional Test ---
Input shape: torch.Size([1, 128, 2048]), dtype: torch.float32
✓ Forward pass successful!
Output shape: torch.Size([1, 128, 2048]), dtype: torch.float8_e4m3fn
✓ Output values are valid

============================================================
End-to-End Performance Benchmark
============================================================

Batch    SeqLen   Latency(ms)  Tokens/sec   TFLOPS    
------------------------------------------------------------
  Warmup (10 iterations)...
  Testing (100 iterations)...
1        64       0.267        239357       16.13     
  Warmup (10 iterations)...
  Testing (100 iterations)...
1        128      0.268        477431       32.29     
  Warmup (10 iterations)...
  Testing (100 iterations)...
4        64       0.264        969519       65.32     
  Warmup (10 iterations)...
  Testing (100 iterations)...
4        128      0.333        1538945      104.08    
  Warmup (10 iterations)...
  Testing (100 iterations)...
8        64       0.327        1563768      105.35    
  Warmup (10 iterations)...
  Testing (100 iterations)...
8        128      0.508        2014350      136.24    

============================================================
Performance Summary
============================================================

Best Throughput:
  Config: Batch=8, SeqLen=128
  Throughput: 2014350 tokens/sec
  Latency: 0.508 ms
  Performance: 136.24 TFLOPS

Best Latency:
  Config: Batch=4, SeqLen=64
  Latency: 0.264 ms
  Throughput: 969519 tokens/sec
  Performance: 65.32 TFLOPS

Average Performance:
  Latency: 0.328 ms
  Throughput: 1133895 tokens/sec
  TFLOPS: 76.57

============================================================
Model Information
============================================================
Total parameters: 33,572,864 (33.57M)

Parameter details:

============================================================
GPU Information
============================================================
Device: NVIDIA H100 PCIe
Compute Capability: (9, 0)
Total Memory: 85.0 GB
Allocated Memory: 34.9 MB
Reserved Memory: 111.1 MB
➜  (workspace) gemmwithprefetch git:(main U:4 ?:6) ✗ 


参数还没有调优的情况下 比不开pdl性能好（5%）（调优模块还得再改改bug
，代码改动比较多但是都实现了），但是比torch好50%




decoder layer
Top 10 configurations for tiny:
Baseline latency: 0.441 ms (all mm: -1.0, 0.0)

Rank | Latency(ms) | Speedup | QKV(o,p) | Output(o,p) | FFN1(o,p) | FFN2(o,p)
--------------------------------------------------------------------------------
   1 |       0.389 |    1.13x | (0.8,0.8) | (0.6,0.6) | (1.0,1.0) | (0.2,0.2)
   2 |       0.389 |    1.13x | (0.4,0.6) | (0.2,0.0) | (0.6,0.4) | (1.0,0.6)
   3 |       0.389 |    1.13x | (0.6,0.4) | (0.6,0.8) | (0.2,0.2) | (0.4,0.2)
   4 |       0.389 |    1.13x | (0.6,0.8) | (0.0,0.4) | (0.0,0.0) | (0.2,-1.0)
   5 |       0.389 |    1.13x | (0.2,0.4) | (0.0,0.8) | (0.0,0.0) | (0.0,1.0)
   6 |       0.389 |    1.13x | (0.8,0.2) | (0.8,0.2) | (-1.0,-1.0) | (0.8,0.2)
   7 |       0.389 |    1.13x | (0.6,0.4) | (1.0,0.8) | (0.0,0.2) | (1.0,0.6)
   8 |       0.389 |    1.13x | (1.0,0.4) | (-1.0,0.2) | (0.0,0.8) | (1.0,0.2)
   9 |       0.389 |    1.13x | (0.4,0.4) | (0.2,0.2) | (0.6,0.8) | (0.4,0.6)
  10 |       0.389 |    1.13x | (1.0,0.4) | (-1.0,0.4) | (0.4,0.8) | (-1.0,0.8)

Best configuration analysis:
QKV strategy: overlap=0.8, prefetch=0.8
Output strategy: overlap=0.6, prefetch=0.6
FFN1 strategy: overlap=1.0, prefetch=1.0
FFN2 strategy: overlap=0.2, prefetch=0.2





Top 10 configurations for BERT-base:
Baseline latency: 0.408 ms (all mm: -1.0, 0.0)

Rank | Latency(ms) | Speedup | QKV(o,p) | Output(o,p) | FFN1(o,p) | FFN2(o,p)
--------------------------------------------------------------------------------
   1 |       0.402 |    1.01x | (0.4,1.0) | (1.0,1.0) | (0.8,-1.0) | (1.0,0.4)
   2 |       0.402 |    1.01x | (0.6,0.8) | (0.8,0.6) | (0.2,0.0) | (0.8,-1.0)
   3 |       0.403 |    1.01x | (-1.0,0.6) | (0.8,0.6) | (0.2,0.8) | (0.8,1.0)
   4 |       0.403 |    1.01x | (0.8,0.4) | (0.8,0.8) | (0.6,-1.0) | (1.0,1.0)
   5 |       0.403 |    1.01x | (-1.0,1.0) | (0.0,0.4) | (0.0,0.6) | (1.0,-1.0)
   6 |       0.403 |    1.01x | (0.6,0.2) | (0.4,0.8) | (0.2,-1.0) | (0.8,0.4)
   7 |       0.403 |    1.01x | (1.0,0.2) | (1.0,0.4) | (1.0,1.0) | (0.0,0.2)
   8 |       0.403 |    1.01x | (1.0,1.0) | (0.0,1.0) | (-1.0,0.8) | (-1.0,0.8)
   9 |       0.403 |    1.01x | (0.6,0.4) | (-1.0,1.0) | (0.6,0.8) | (0.6,0.8)
  10 |       0.403 |    1.01x | (0.6,0.2) | (0.8,0.6) | (1.0,0.2) | (-1.0,0.2)

Best configuration analysis:
QKV strategy: overlap=0.4, prefetch=1.0
Output strategy: overlap=1.0, prefetch=1.0
FFN1 strategy: overlap=0.8, prefetch=-1.0
FFN2 strategy: overlap=1.0, prefetch=0.4



Top 10 configurations for BERT-large:
Baseline latency: 0.419 ms (all mm: -1.0, 0.0)

Rank | Latency(ms) | Speedup | QKV(o,p) | Output(o,p) | FFN1(o,p) | FFN2(o,p)
--------------------------------------------------------------------------------
   1 |       0.413 |    1.01x | (0.8,0.2) | (0.8,0.2) | (0.2,0.0) | (0.4,0.6)
   2 |       0.413 |    1.01x | (0.4,1.0) | (0.0,0.8) | (0.8,0.8) | (0.0,0.4)
   3 |       0.413 |    1.01x | (0.4,0.6) | (1.0,0.6) | (0.0,0.4) | (0.2,0.2)
   4 |       0.413 |    1.01x | (0.8,0.4) | (0.4,1.0) | (0.0,1.0) | (0.6,0.6)
   5 |       0.414 |    1.01x | (0.8,-1.0) | (1.0,0.6) | (0.2,1.0) | (0.4,0.0)
   6 |       0.414 |    1.01x | (-1.0,0.6) | (0.8,1.0) | (0.2,-1.0) | (0.8,0.8)
   7 |       0.414 |    1.01x | (0.0,0.4) | (0.0,0.4) | (0.4,0.0) | (-1.0,0.4)
   8 |       0.414 |    1.01x | (1.0,0.4) | (0.2,0.4) | (1.0,0.0) | (0.0,0.8)
   9 |       0.414 |    1.01x | (0.6,0.2) | (1.0,0.6) | (0.6,0.4) | (0.4,0.0)
  10 |       0.414 |    1.01x | (0.0,0.0) | (0.0,1.0) | (0.8,0.0) | (0.0,-1.0)

Best configuration analysis:
QKV strategy: overlap=0.8, prefetch=0.2
Output strategy: overlap=0.8, prefetch=0.2
FFN1 strategy: overlap=0.2, prefetch=0.0
FFN2 strategy: overlap=0.4, prefetch=0.6

====================================================================================================
SUMMARY OF ALL SHAPES
====================================================================================================
     Shape         Dimensions Baseline(ms) Best(ms) Speedup  Best_QKV Best_Output  Best_FFN1 Best_FFN2
      tiny    1x32x256x4x1024        0.441    0.389   1.13x (0.8,0.8)   (0.6,0.6)  (1.0,1.0) (0.2,0.2)
 BERT-base  1x128x768x12x3072        0.408    0.402   1.01x (0.4,1.0)   (1.0,1.0) (0.8,-1.0) (1.0,0.4)
BERT-large 1x128x1024x16x4096        0.419    0.413   1.01x (0.8,0.2)   (0.8,0.2)  (0.2,0.0) (0.4,0.6)

Results saved to: decoder_layer_optimization_results/run_20250828_111340





## attention

Top 10 configurations for tiny:
Baseline latency: 0.210 ms (all mm: -1.0, 0.0)

Rank | Latency(ms) | Speedup | QKV(overlap,prefetch) | Output(overlap,prefetch)
--------------------------------------------------------------------------------
   1 |       0.165 |    1.27x | ( 1.0, 0.2) | ( 0.0, 1.0)
   2 |       0.165 |    1.27x | ( 1.0, 0.2) | ( 0.0, 0.8)
   3 |       0.165 |    1.27x | ( 0.8,-1.0) | ( 0.6, 1.0)
   4 |       0.165 |    1.27x | ( 1.0, 0.2) | ( 0.0, 0.4)
   5 |       0.165 |    1.27x | ( 0.8,-1.0) | ( 0.6,-1.0)
   6 |       0.165 |    1.27x | ( 0.4, 0.8) | (-1.0, 0.8)
   7 |       0.165 |    1.27x | ( 0.4, 1.0) | ( 0.2, 0.0)
   8 |       0.165 |    1.27x | ( 0.4, 1.0) | ( 0.0, 0.8)
   9 |       0.165 |    1.27x | ( 0.4, 1.0) | ( 0.2, 0.6)
  10 |       0.165 |    1.27x | ( 0.8,-1.0) | ( 0.2, 0.6)

====================================================================================================
SUMMARY OF ALL SHAPES
====================================================================================================
Shape  Dimensions Baseline(ms) Best(ms) Speedup  Best_QKV Best_Output
 tiny 1x32x1024x4        0.210    0.165   1.27x (1.0,0.2)   (0.0,1.0)
