import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

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
W_query: Parameter = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key: Parameter = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value: Parameter = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# print(query_2)
# print(np.asarray([0.4,1.4]).dot(np.asarray([0.4,1.1])))
keys = inputs @ W_key
values = inputs @ W_value
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)
attn_scores_2=query_2@keys.T
# print(attn_scores_2)
d_k=keys.shape[-1]
attn_weights_2=torch.softmax(attn_scores_2/d_k**0.5, dim=-1)
# print(attn_weights_2)
context_vec_2=attn_weights_2@values
print(context_vec_2)