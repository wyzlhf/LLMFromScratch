import torch
from torch import nn, Tensor
from torch.nn import Parameter, Linear


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out) -> None:
        super().__init__()
        self.W_query: Parameter = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key: Parameter = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value: Parameter = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x) -> Tensor:
        queries: Parameter = x @ self.W_query
        keys: Parameter = x @ self.W_key
        values: Parameter = x @ self.W_value
        attn_scores: Tensor = queries @ keys.T
        attn_weights: Tensor = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec: Tensor = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        self.W_query: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x) -> Tensor:
        queries: Tensor = self.W_query(x)
        keys: Tensor = self.W_key(x)
        values: Tensor = self.W_value(x)
        attn_scores: Tensor = queries @ keys.T
        attn_weights: Tensor = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
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
    x_2: Tensor = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
