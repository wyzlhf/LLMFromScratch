import torch
from torch import nn, Tensor
from torch.nn import Linear


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

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)

    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)

    masked_simple = attn_weights * mask_simple
    print(masked_simple)

    row_sums=masked_simple.sum(dim=-1,keepdim=True)
    masked_simple_norm=masked_simple/row_sums
    print(masked_simple_norm)

    mask=torch.triu(torch.ones(context_length,context_length),diagonal=1)
    masked=attn_scores.masked_fill(mask.bool(),-torch.inf)
    print(masked)

    attn_weights=torch.softmax(masked/keys.shape[-1]**0.5,dim=-1)
    print(attn_weights)

    torch.manual_seed(123)
    dropout=torch.nn.Dropout(0.5)
    example=torch.ones(6,6)
    print(dropout(example))
    print(dropout(attn_weights))
