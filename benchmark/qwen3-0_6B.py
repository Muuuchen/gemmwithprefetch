import torch
import torch.nn as nn
import cutlass_gemm_with_prefetch
import numpy as np
from dataclasses import dataclass
import time

# 设置环境变量
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@dataclass
class DecoderLayerConfig:
    """Decoder Layer配置参数"""
    # Attention部分
    qkv_overlap: float
    qkv_prefetch: float
    output_overlap: float
    output_prefetch: float
    # FFN部分
    ffn1_overlap: float
    ffn1_prefetch: float
    ffn2_overlap: float
    ffn2_prefetch: float

    def to_dict(self):
        return {
            'qkv_overlap': self.qkv_overlap,
            'qkv_prefetch': self.qkv_prefetch,
            'output_overlap': self.output_overlap,
            'output_prefetch': self.output_prefetch,
            'ffn1_overlap': self.ffn1_overlap,
            'ffn1_prefetch': self.ffn1_prefetch,
            'ffn2_overlap': self.ffn2_overlap,
            'ffn2_prefetch': self.ffn2_prefetch,
        }

class Qwen3DecoderLayer:
    """Qwen3 Decoder Layer性能测试"""

    def __init__(self, batch_size: int = 1, seq_len: int = 128):
        # Qwen3模型参数
        self.d_model = 1024  # hidden_size
        self.n_heads = 16    # num_attention_heads
        self.n_kv_heads = 8  # num_key_value_heads (GQA)
        self.d_head = 128    # head_dim
        self.d_ff = 3072     # intermediate_size

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = 'cuda'

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        scale = 0.02

        # Attention权重 - 注意GQA的KV heads数量不同
        # Q: d_model -> n_heads * d_head
        # K,V: d_model -> n_kv_heads * d_head
        qkv_dim = self.n_heads * self.d_head + 2 * self.n_kv_heads * self.d_head

        self.W_qkv = (torch.randn(self.d_model, qkv_dim, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_qkv = torch.zeros(qkv_dim, device=self.device).to(torch.float8_e4m3fn)
        self.W_o = (torch.randn(self.d_model, self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_o = torch.zeros(self.d_model, device=self.device).to(torch.float8_e4m3fn)

        # FFN权重
        self.W_ff1 = (torch.randn(self.d_model, self.d_ff, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_ff1 = torch.zeros(self.d_ff, device=self.device).to(torch.float8_e4m3fn)
        self.W_ff2 = (torch.randn(self.d_ff, self.d_model, device=self.device) * scale).to(torch.float8_e5m2)
        self.b_ff2 = torch.zeros(self.d_model, device=self.device).to(torch.float8_e4m3fn)

    def decoder_layer_forward(self, x: torch.Tensor, config: DecoderLayerConfig) -> torch.Tensor:
        """执行完整的decoder layer前向传播"""
        batch_size, seq_len, _ = x.shape

        # 保存第一个residual
        residual1 = x.to(torch.float8_e4m3fn)

        # 1. Attention部分
        # QKV Projection
        x_flat = residual1.reshape(-1, self.d_model)
        qkv_dim = self.n_heads * self.d_head + 2 * self.n_kv_heads * self.d_head
        qkv_output = torch.zeros(batch_size * seq_len, qkv_dim,
                                device=x.device, dtype=torch.float8_e4m3fn)

        qkv = cutlass_gemm_with_prefetch.mm(
            x_flat,
            self.W_qkv,
            qkv_output,
            self.b_qkv.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.qkv_overlap,
            config.qkv_prefetch
        )

        # Split Q, K, V (处理GQA)
        qkv = qkv.reshape(batch_size, seq_len, -1)
        q_dim = self.n_heads * self.d_head
        kv_dim = self.n_kv_heads * self.d_head

        q = qkv[:, :, :q_dim].reshape(batch_size, seq_len, self.n_heads, self.d_head).to(torch.float16)
        k = qkv[:, :, q_dim:q_dim+kv_dim].reshape(batch_size, seq_len, self.n_kv_heads, self.d_head).to(torch.float16)
        v = qkv[:, :, q_dim+kv_dim:].reshape(batch_size, seq_len, self.n_kv_heads, self.d_head).to(torch.float16)

        # 扩展KV heads以匹配Q heads (GQA)
        repeat_factor = self.n_heads // self.n_kv_heads
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        # Flash Attention
        attn_output = cutlass_gemm_with_prefetch.fa(q, k, v)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model).to(torch.float8_e4m3fn)

        # Output Projection
        attn_flat = attn_output.reshape(-1, self.d_model)
        output = torch.zeros(batch_size * seq_len, self.d_model,
                           device=x.device, dtype=torch.float8_e4m3fn)
        output = cutlass_gemm_with_prefetch.mm(
            attn_flat,
            self.W_o,
            output,
            self.b_o.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.output_overlap,
            config.output_prefetch
        )
        output = output.reshape(batch_size, seq_len, self.d_model)

        # First residual connection
        x = cutlass_gemm_with_prefetch.add_residual(output, residual1)

        # 保存第二个residual
        residual2 = x

        # 2. FFN部分
        # FFN第一层
        x_flat = x.reshape(-1, self.d_model)
        ff1_output = torch.zeros(batch_size * seq_len, self.d_ff,
                               device=x.device, dtype=torch.float8_e4m3fn)

        hidden = cutlass_gemm_with_prefetch.mm(
            x_flat,
            self.W_ff1,
            ff1_output,
            self.b_ff1.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.ffn1_overlap,
            config.ffn1_prefetch
        )

        # FFN第二层
        ff2_output = torch.zeros(batch_size * seq_len, self.d_model,
                               device=x.device, dtype=torch.float8_e4m3fn)
        output = cutlass_gemm_with_prefetch.mm(
            hidden,
            self.W_ff2,
            ff2_output,
            self.b_ff2.unsqueeze(0).expand(batch_size * seq_len, -1),
            config.ffn2_overlap,
            config.ffn2_prefetch
        )
        output = output.reshape(batch_size, seq_len, self.d_model)

        # Second residual connection
        x = cutlass_gemm_with_prefetch.add_residual(output, residual2)

        return x

    def test_config(self, config: DecoderLayerConfig, warmup: int = 10, iterations: int = 50):
        """测试单个配置的性能"""
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.d_model,
                       device=self.device, dtype=torch.float16)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.decoder_layer_forward(x, config)
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_event.record()

            for _ in range(iterations):
                output = self.decoder_layer_forward(x, config)

            end_event.record()
            torch.cuda.synchronize()

        # 计算延迟
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / iterations

        return avg_time_ms


def test_three_ratios(batch_size=1, seq_len=128):
    """测试三个不同的ratio配置"""

    print(f"Testing Qwen3 Decoder Layer")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Model params: d_model=1024, n_heads=16, n_kv_heads=8, d_ff=3072")
    print("="*70)

    # 创建模型
    model = Qwen3DecoderLayer(batch_size=batch_size, seq_len=seq_len)

    # 定义三个ratio配置
    configs = [
        # Ratio 1: Baseline (no overlap/prefetch)
        ("Baseline", DecoderLayerConfig(
            qkv_overlap=-1.0, qkv_prefetch=0.0,
            output_overlap=-1.0, output_prefetch=0.0,
            ffn1_overlap=-1.0, ffn1_prefetch=0.0,
            ffn2_overlap=-1.0, ffn2_prefetch=0.0,
        )),

        # Ratio 2: Full prefetch
        ("Full Prefetch", DecoderLayerConfig(
            qkv_overlap=0.0, qkv_prefetch=1.0,
            output_overlap=0.0, output_prefetch=1.0,
            ffn1_overlap=0.0, ffn1_prefetch=1.0,
            ffn2_overlap=0.0, ffn2_prefetch=1.0,
        )),

        # Ratio 3: Full overlap
        ("Full Overlap", DecoderLayerConfig(
            qkv_overlap=1.0, qkv_prefetch=0.0,
            output_overlap=1.0, output_prefetch=0.0,
            ffn1_overlap=1.0, ffn1_prefetch=0.0,
            ffn2_overlap=1.0, ffn2_prefetch=0.0,
        )),
    ]

    results = []
    baseline_latency = None

    for name, config in configs:
        print(f"\nTesting {name}...")
        print(f"Config: {config.to_dict()}")

        try:
            latency = model.test_config(config, warmup=20, iterations=100)

            if baseline_latency is None:
                baseline_latency = latency
                speedup = 1.0
            else:
                speedup = baseline_latency / latency

            results.append({
                'Name': name,
                'Latency (ms)': latency,
                'Speedup': speedup
            })

            print(f"Latency: {latency:.3f} ms")
            print(f"Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"Failed to test {name}: {e}")
            import traceback
            traceback.print_exc()

    # 打印结果汇总
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for result in results:
        print(f"{result['Name']:15s} | Latency: {result['Latency (ms)']:8.3f} ms | Speedup: {result['Speedup']:5.2f}x")

    # 打印GPU信息
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Peak Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    print("Starting Qwen3 Decoder Layer Performance Test")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # 设置随机种子
    torch.manual_seed(42)

    # 测试不同的输入shape
    test_shapes = [
        (1, 128),   # batch=1, seq_len=128
        # (1, 256),   # batch=1, seq_len=256
        # (4, 128),   # batch=4, seq_len=128
    ]

    for batch_size, seq_len in test_shapes:
        print(f"\n{'='*70}")
        test_three_ratios(batch_size, seq_len)
        torch.cuda.empty_cache()
