import torch
from torch import Tensor, nn

input_ids: Tensor=torch.tensor([2,3,4,1])
vocab_size:int=6
output_dim:int=3
torch.manual_seed(123)
embedding_layer: nn.Embedding=nn.Embedding(vocab_size,output_dim)
print(embedding_layer)
print(embedding_layer.weight)
# print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))