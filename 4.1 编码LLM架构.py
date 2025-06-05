import torch
import tiktoken
import torch.nn as nn
from torch import Tensor
from config import GPT_CONFIG_124M, MultiHeadAttention
import matplotlib

matplotlib.use('TkAgg')  # 或者其他可用的后端，如 'Qt5Agg'
import matplotlib.pyplot as plt


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])],
        )
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias'],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])],
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        logits = self.out_head(x)
        return logits
def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        logits=logits[:,-1,:]
        probs=torch.softmax(logits,dim=-1)
        idx_next=torch.argmax(probs,dim=-1,keepdim=True)
        idx=torch.cat((idx,idx_next),dim=1)
    return idx

if __name__ == '__main__':
    # 4.1节
    # tokenizer=tiktoken.get_encoding('gpt2')
    # batch=[]
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch=torch.stack(batch, dim=0)
    # # print(batch)
    # torch.manual_seed(123)
    # model = DummyGPTModel(GPT_CONFIG_124M)
    # logits=model(batch)
    # # print(f'输出形状：{logits.shape}')
    # # print(logits)

    # 4.2节
    # torch.manual_seed(123)
    # batch_example = torch.randn(2, 5)
    # # layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    # # out = layer(batch_example)
    # # # print(out)
    # # mean = out.mean(dim=-1, keepdim=True)
    # # var = out.var(dim=-1, keepdim=True)
    # # # print(mean)
    # # # print(var)
    # # out_norm = (out - mean) / torch.sqrt(var)
    # # norm_mean = out_norm.mean(dim=-1, keepdim=True)
    # # norm_var = out_norm.var(dim=-1, keepdim=True)
    # torch.set_printoptions(sci_mode=False)
    # # # print(out_norm)
    # # # print(norm_mean)
    # # # print(norm_var)
    # ln=LayerNorm(emb_dim=5)
    # out_ln=ln(batch_example)
    # mean=out_ln.mean(dim=-1, keepdim=True)
    # var=out_ln.var(dim=-1, keepdim=True,unbiased=False)
    # print(mean)
    # print(var)

    # 4.3节
    # gelu, relu = GELU(), nn.ReLU()
    # x = torch.linspace(-3, 3, 100)
    # y_gelu, y_relu = gelu(x), relu(x)
    # plt.figure(figsize=(8, 3))
    # for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['gelu', 'relu']), 1):
    #     plt.subplot(1, 2, i)
    #     plt.plot(x, y)
    #     plt.title(f'{label}激活函数')
    #     plt.xlabel('x')
    #     plt.ylabel(f'{label}(x)')
    #     plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # ffn = FeedForward(GPT_CONFIG_124M)
    # x = torch.randn(2, 3, 768)
    # out = ffn(x)
    # print(out.shape)
    # print(out)

    # 4.4节
    # layer_sizes = [3, 3, 3, 3, 3, 1]
    # sample_input = torch.tensor([[1., 0., -1.]])
    # torch.manual_seed(123)
    # # model_without_shorcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    # # print_gradients(model_without_shorcut,sample_input)
    # # print()
    # model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, True)
    # print_gradients(model_with_shortcut, sample_input)

    # 4.5节
    # torch.manual_seed(123)
    # x = torch.rand(2, 4, 768)
    # block = TransformerBlock(GPT_CONFIG_124M)
    # output = block(x)
    # print(f'Input shape: {x.shape}')
    # print(f'Output shape: {output.shape}')

    #4.6节
    # tokenizer = tiktoken.get_encoding('gpt2')
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)
    # torch.manual_seed(123)
    # model=GPTModel(GPT_CONFIG_124M)
    # out=model(batch)
    # print(f'Input batch:\n{batch}')
    # print(f'Output batch:\n{out.shape}')
    # print(out)
    #
    # total_params=sum(p.numel() for p in model.parameters())
    # print(f'Total number of parameters:{format(total_params,",d")}')
    #
    # print(f'Token embedding layer shape:{model.tok_emb.weight.shape}')
    # print(f'Output layer shape:{model.out_head.weight.shape}')
    #
    # total_params_gpt2=(
    #     total_params-sum(p.numel() for p in model.out_head.parameters())
    # )
    # print(f'Number of trainable parameters considering weight tying:{format(total_params_gpt2,",d")}')
    #
    # total_size_bytes=total_params*4
    # total_size_mb=total_size_bytes/(1024*1024)
    # print(f'Total size of the model:{total_size_mb:.2f} MB')

    #4.7节
    tokenizer = tiktoken.get_encoding('gpt2')
    # start_context="Hello, I am"
    start_context = "你好啊！"
    encoded=tokenizer.encode(start_context)
    # print(f'encoded:{encoded}')
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)
    # print(f'encoded_tensor.shape:{encoded_tensor.shape}')

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    out=generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M['context_length']
    )
    print(f'Output:{out}')
    print(f'Output length:{len(out[0])}')

    decoded_text=tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


