import torch
from torch import nn, Tensor
from torch.nn import Linear, Dropout
from typing import Tuple

class CausalAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias=False) -> None:
        super().__init__()
        self.d_out: int = d_out
        self.W_query: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout: Dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.size()
        keys: Tensor = self.W_key(x)
        queries: Tensor = self.W_query(x)
        values: Tensor = self.W_value(x)
        attn_scores: Tensor = queries @ keys.transpose(1, 2)
        print(self.mask)
        print(self.__dir__())
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # 这句没明白self.mask.bool()
        attn_weights: Tensor = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights: Tensor = self.dropout(attn_weights)
        context_vec: Tensor = attn_weights @ values
        return context_vec


if __name__ == '__main__':
    inputs: Tensor = torch.tensor([[0.43, 0.15, 0.89],  # Your (x^1)
                                   [0.55, 0.87, 0.66],  # journey (x^2)
                                   [0.57, 0.85, 0.64],  # starts (x^3)
                                   [0.22, 0.58, 0.33],  # with (x^4)
                                   [0.77, 0.25, 0.10],  # one (x^5)
                                   [0.05, 0.80, 0.55]]  # step (x^6)
                                  )
    d_in = inputs.shape[1]
    d_out = 2
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca=CausalAttention(d_in,d_out,context_length,0.0)
    context_vecs=ca(batch)
    print(context_vecs)
