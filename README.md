# install bear for compile commands


****
 uv pip install -e . --no-build-isolation

How to Add Cutlass Kernel to Torch/Sglang
背景
我们知道很多cutlass实现的kernel采用了相对比较多的优化手段和新的特性，那么我们在拿到cutlass的cuda源码的时候，该如何方便的用python来调用，这篇文章给出一个基本的方法和模板。
GemmWithPrefetch
结合之前所讲的PDL特性，我们可以将一个cuda kernel的Prologue部分，比如预取等操作，在前一个kernel还没有结束的时候就被调用，因此，Cutlass基于这个思路进一步实现了一个结合prefetch的代码。通过PDL提前launch当前GEMM的预取部分，将一部分数据提前预取到L2 Cache中，来优化后续Gemm的性能。
那么接下来我们来介绍如何把这个kernel集成到python中，并进一步在torch中使用。
原函数封装：
具体的原Kernel如何实现我们可以在Cutlass的Examples中找到很多优秀的示例。
在项目中我们首先需要定义在CUDA中真正执行计算的函数。
在CUDA中我们能传入的只有数据指针，我们在后面自己管理从torch tensor到cutlass Tensor的转变

