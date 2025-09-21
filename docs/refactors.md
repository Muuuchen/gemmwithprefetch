# how to unify interface 

ops: op(input, output ,weigth, prefetch_ratio, overlap_ratio, type:str)

dispatch 过程我来对不同的模板 来选定不同的下面编译的东西



kernel：template<> // ratio 值比较固定，那么可以编译出不同的模板


测试项： latency, throughput 



## Marco
我们需要用元编程技巧来定义一组算子实例

cuh中定义模板
hpp中定义模板元函数
interface 中定义接口launch和函数的声明声明，和 wrapper 的声明

cu中定义实现
binding 中定义