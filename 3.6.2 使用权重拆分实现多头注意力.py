import torch
from torch import nn, Tensor
from torch.nn import Linear, Dropout


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int,
                 qkv_bias: bool = False) -> None:
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out: int = d_out
        self.num_heads: int = num_heads
        self.head_dim: int = d_out // num_heads
        self.W_query: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value: Linear = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj: Linear = nn.Linear(d_out, d_out)
        self.dropout: Dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys: Tensor = self.W_key(x)
        queries: Tensor = self.W_query(x)
        values: Tensor = self.W_value(x)

        keys: Tensor = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values: Tensor = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries: Tensor = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys: Tensor = keys.transpose(1, 2)
        queries: Tensor = queries.transpose(1, 2)
        values: Tensor = values.transpose(1, 2)

        attn_scores: Tensor = queries @ keys.transpose(2, 3)
        mask_bool: Tensor = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights: Tensor = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights: Tensor = self.dropout(attn_weights)

        context_vec: Tensor = (attn_weights @ values).transpose(1, 2)
        context_vec: Tensor = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec: Tensor = self.out_proj(context_vec)

        return context_vec


if __name__ == '__main__':
    inputs: Tensor = torch.tensor([[0.43, 0.15, 0.89],  # Your (x^1)
                                   [0.55, 0.87, 0.66],  # journey (x^2)
                                   [0.57, 0.85, 0.64],  # starts (x^3)
                                   [0.22, 0.58, 0.33],  # with (x^4)
                                   [0.77, 0.25, 0.10],  # one (x^5)
                                   [0.05, 0.80, 0.55]]  # step (x^6)
                                  )
    # d_in = inputs.shape[1]
    # d_out = 2
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print(f'context_vecs.shape: {context_vecs.shape}')
