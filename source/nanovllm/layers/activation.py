import torch
from torch import nn
import torch.nn.functional as F

#这是一层 SwiGLU（Swish-Gated Linear Unit）激活层，是 Transformer 中 FFN（前馈神经网络）常用的门控激活结构。

#在 Nano-vLLM 的 Qwen3 模型中，你会经常看到它。它的作用是：通过一个分支控制另一个分支的信息流，而不是简单地对输入做一次激活。
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #把张量按最后一维切开
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
