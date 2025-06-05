import torch
from torch import Tensor

inputs: Tensor = torch.tensor([[0.43, 0.15, 0.89],  # Your (x^1)
                               [0.55, 0.87, 0.66],  # journey (x^2)
                               [0.57, 0.85, 0.64],  # starts (x^3)
                               [0.22, 0.58, 0.33],  # with (x^4)
                               [0.77, 0.25, 0.10],  # one (x^5)
                               [0.05, 0.80, 0.55]]  # step (x^6)
                              )
query = inputs[1]  # tensor([0.5500, 0.8700, 0.6600])就是journey (x^2)
# print(query)
attn_scores_2 = torch.empty(inputs.shape[0])  # 1×6的一个tensor
# print(attn_scores_2)
for i, x_i in enumerate(inputs):
    # assert type(x_i)==torch.Tensor
    attn_scores_2[i] = torch.dot(x_i, query)  # attn_scores_2就是inputs中的每个元素和第二个元素的dot生成的一个6元素向量
# print(attn_scores_2)
attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum()


# print(attn_scores_2_tmp)
# print(attn_scores_2_tmp.sum())
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_scores_2_naive=softmax_naive(attn_scores_2)
print(attn_scores_2_naive)
attn_weights_2=torch.softmax(attn_scores_2,dim=0)
print(attn_weights_2)