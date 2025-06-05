import tiktoken
import torch
from tiktoken import Encoding
from typing import List
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

# tokenizer: Encoding = tiktoken.get_encoding("gpt2")
#
# with open('the-verdict.txt', 'r', encoding='utf-8') as f:
#     raw_text = f.read()
# enc_text = tokenizer.encode(raw_text)
# print(len(raw_text.split(' ')))
# print(len(enc_text))
# print(enc_text)
# context_size=len(enc_text)
# enc_sample = enc_text[50:]
# context_size = 4
# x=enc_sample[:context_size]
# y=enc_sample[1:context_size+1]
# print(f'x:{x}\ny:{y}')
# for i in range(1, context_size + 1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(f'{context}---->{desired}')
#     print(f'{tokenizer.decode(context)}---->{tokenizer.decode([desired])}')


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int):
        self.input_ids: List[Tensor] = []
        self.target_ids: List[Tensor] = []
        token_ids: List[int] = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt: str, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True,
                         drop_last: bool = True, num_workers: int = 0)->DataLoader:
    tokenizer: Encoding = tiktoken.get_encoding("gpt2")
    dataset: GPTDatasetV1 = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return dataloader
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
# dataloader=create_dataloader_v1(
#     raw_text,
#     batch_size=1,
#     max_length=4,
#     stride=1,
#     shuffle=False
# )
# data_iter=iter(dataloader)
# first_batch=next(data_iter)
# second_batch=next(data_iter)
# print(first_batch)
# print(second_batch)
vocab_size=50257
output_dim=256
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
# print(token_embedding_layer.weight)
max_length=4
dataloader=create_dataloader_v1(raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False)
data_iter=iter(dataloader)
inputs,targets=next(data_iter)
# print(f'Token IDs:\n{inputs},{inputs.shape}')
# print(f'Target IDs:\n{targets}')
token_embeddings=token_embedding_layer(inputs)
# print(token_embeddings.shape)
context_length=max_length
pos_embedding_layer=nn.Embedding(context_length, output_dim)
pos_embeddings=pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings.shape)
input_embeddings=token_embeddings+pos_embeddings