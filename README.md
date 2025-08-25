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
