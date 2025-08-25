
import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass_gemm_with_prefetch


class FeedForwardPDLConfig:
    def __init__(self):
        self.mm1_overlap_ratio= 0.0
        self.mm1_prefetch_ratio = 0.0
        self.mm2_overlap_ratio = 0.0
        self.mm2_prefetch_ratio = 0.0
        self.rmsnorm_prefetch_ratio = 0.0
        self.rmsnorm_overlap_ratio = 0.0
        self.hierarchy = cutlass_gemm_with_prefetch.KernelOverlapHierarchy.NONE

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, config):
        super().__init__()
        self.weight1 = torch.normal(0, 1, size=(d_model, d_ff)).to(device="cuda").to(dtype=torch.float8_e5m2)
        self.weight2 =  torch.normal(0, 1, size=( d_ff,d_model)).to(device="cuda").to(dtype=torch.float8_e5m2)
        self.config = config
        self.activation_type = "rmsnorm"
        self.first_call = True
        self.D = None
        self.rmsnorm_weight = None
    def init_weight(self,x):
        if self.first_call:
            self.D = torch.normal(0, 1, size=x.shape).to(device="cuda").to(dtype=torch.float8_e4m3fn)
            self.rmsnorm_weight = torch.normal(0, 1, size=x.shape).to(device="cuda").to(dtype=torch.float8_e4m3fn)

    def forward(self, x):

        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        MM:
            (left,right,output,weight)
        """
        self.init_weight(x)
        x = cutlass_gemm_with_prefetch.mm(x,self.weight1,x,self.D,self.config.mm1_overlap_ratio, self.config.mm1_prefetch_ratio)
        x = cutlass_gemm_with_prefetch.rmsnorm(x,x,self.rmsnorm_weight, self.config.rmsnorm_prefetch_ratio, self.config.hierarchy)
        x = cutlass_gemm_with_prefetch.mm(x,self.weight1,x,self.D,self.config.mm1_overlap_ratio, self.config.mm1_prefetch_ratio)
        return x

if __name__ == "__main__":

    config = FeedForwardPDLConfig()
    ffn = FeedForward(1024,2048,config)
    x = torch.randn((128,1024)).to(device="cuda").to(dtype=torch.float8_e4m3fn)
    ffn(x)