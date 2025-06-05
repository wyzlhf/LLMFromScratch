import matplotlib
import urllib.request
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import tiktoken
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from torch import device
from tiktoken import Encoding
from torch import Tensor
from torch.utils.data import DataLoader
from config import GPTModel, generate_text_simple, create_dataloader_v1, model_configs
from config import GPT_CONFIG_124M_ch05 as GPT_CONFIG_124M
from gpt_download import download_and_load_gpt2

torch.set_printoptions(precision=4, sci_mode=False)


def text_to_token_ids(text: str, tokenizer: Encoding) -> Tensor:
    encoded: list[int] = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor: Tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: Tensor, tokenizer: Encoding) -> str:
    flat: Tensor = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch: Tensor, target_batch: Tensor, model: GPTModel, device: device):
    input_batch: Tensor = input_batch.to(device)
    target_batch: Tensor = target_batch.to(device)
    logits: Tensor = model(input_batch)
    loss: Tensor = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model: GPTModel, device: device, num_batches: int = None):
    total_loss: float = 0.
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches: int | None = len(data_loader)
    else:
        num_batches: int | None = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss: Tensor = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        model.train()
        return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace('\n', " "))
        model.train()


def train_model_simple(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader,
                       optimizer: torch.optim.Optimizer, device: device, num_epochs: int, eval_freq: int,
                       eval_iter: int, start_context: str, tokenizer: Encoding):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f'Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}')
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label='Training loss')
    ax1.plot(epochs_seen, val_losses, linestyle='-.', label='Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel('Epochs')
    fig.tight_layout()
    plt.show()


# def print_sample_tokens(probas):
#     torch.manual_seed(123)
#     sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1000)]
#     sample_ids = torch.bincount(torch.tensor(sample))
#     for i, freq in enumerate(sample_ids):
#         print(f'{freq}×{inverse_vocab[i]}')


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f'shape mismatch. Left:{left.shape}, Right:{right.shape}')
    return torch.nn.Parameter(torch.tensor(right))

#这个函数是copy过来的，要弄明白
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

if __name__ == '__main__':
    # 5.1节
    # torch.manual_seed(123)
    # model = GPTModel(GPT_CONFIG_124M)
    # model.eval()
    #
    # # start_context: str = 'Every effort moves you'
    # start_context: str = "Every effort moves you"
    # tokenizer = tiktoken.get_encoding('gpt2')
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids(start_context, tokenizer),
    #     max_new_tokens=10,
    #     context_size=GPT_CONFIG_124M['context_length']
    # )
    # # print(f'Output text:\n{token_ids_to_text(token_ids,tokenizer)}')
    #
    # inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
    #                        [40, 1107, 588]])  # "I really like"]
    # targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
    #                         [1107, 588, 11311]])  # " really like chocolate"]
    # with torch.no_grad():
    #     logits = model(inputs)
    # probas = torch.softmax(logits, dim=-1)
    # # print(probas)
    # token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # # print(f'Token ids: \n{token_ids}')
    # # print(f'Targets batch 1:{token_ids_to_text(targets[0],tokenizer)}')
    # # print(f'Outputs batch 1:{token_ids_to_text(token_ids[0].flatten(),tokenizer)}')
    #
    # text_idx = 0
    # target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # # print(f'Text 1:{target_probas_1}')
    #
    # text_idx = 1
    # target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # # print(f'Text 2:{target_probas_2}')
    # #此处输出和教程中不一样————————————————————需要注意！！！！！！！！！！！！！！
    # # text_idx = 0
    # # target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # # print("Text 1:", target_probas_1)
    # # text_idx = 1
    # # target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # # print("Text 2:", target_probas_2)
    #
    # log_probas=torch.log(torch.cat((target_probas_1,target_probas_2)))
    # # print(log_probas)
    # avg_log_probas = torch.mean(log_probas)
    # # print(avg_log_probas)
    # neg_avg_log_probas=avg_log_probas*-1
    # # print(neg_avg_log_probas)
    #
    # # print(f'Logits shape:{logits.shape}')
    # # print(f'Targets shape:{targets.shape}')
    #
    # logits_flat=logits.flatten(0,1)
    # targets_flat=targets.flatten()
    # # print(f'Flattened logits: {logits_flat.shape}')
    # # print(f'Flattened targets: {targets_flat.shape}')
    # loss=torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    # # print(loss)

    # 开始文本处理
    # file_path = 'the-verdict.txt'
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     text_data = f.read()
    # total_characters = len(text_data)
    # tokenizer = tiktoken.get_encoding('gpt2')
    # total_tokens = len(tokenizer.encode(text_data))
    # # print(f'Characters: {total_characters}')
    # # print(f'Tokens: {total_tokens}')
    #
    # train_ratio = 0.90
    # split_idx = int(train_ratio * len(text_data))
    # train_data = text_data[:split_idx]
    # val_data = text_data[split_idx:]
    #
    # torch.manual_seed(123)
    # train_loader = create_dataloader_v1(
    #     train_data,
    #     batch_size=2,
    #     max_length=GPT_CONFIG_124M['context_length'],
    #     stride=GPT_CONFIG_124M['context_length'],
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=0,
    # )
    # val_loader = create_dataloader_v1(
    #     val_data,
    #     batch_size=2,
    #     max_length=GPT_CONFIG_124M['context_length'],
    #     stride=GPT_CONFIG_124M['context_length'],
    #     drop_last=False,
    #     shuffle=False,
    #     num_workers=0,
    # )
    # print('Train loader:')
    # for x,y in train_loader:
    #     print(x.shape,y.shape)
    # print('Validation loader:')
    # for x,y in val_loader:
    #     print(x,y)

    # model = GPTModel(GPT_CONFIG_124M)
    # device: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # with torch.no_grad():
    #     train_loss = calc_loss_loader(train_loader, model, device)
    #     val_loss = calc_loss_loader(val_loader, model, device)
    # print(f'Training loss:{train_loss}')
    # print(f'Validation loss:{val_loss}')

    # 5.2节
    # file_path = 'the-verdict.txt'
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     text_data = f.read()
    # total_characters = len(text_data)
    # tokenizer = tiktoken.get_encoding('gpt2')
    # total_tokens = len(tokenizer.encode(text_data))
    # train_ratio = 0.90
    # split_idx = int(train_ratio * len(text_data))
    # train_data = text_data[:split_idx]
    # val_data = text_data[split_idx:]
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_loader = create_dataloader_v1(
    #     train_data,
    #     batch_size=2,
    #     max_length=GPT_CONFIG_124M['context_length'],
    #     stride=GPT_CONFIG_124M['context_length'],
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=0,
    # )
    # val_loader = create_dataloader_v1(
    #     val_data,
    #     batch_size=2,
    #     max_length=GPT_CONFIG_124M['context_length'],
    #     stride=GPT_CONFIG_124M['context_length'],
    #     drop_last=False,
    #     shuffle=False,
    #     num_workers=0,
    # )
    # torch.manual_seed(123)
    # model = GPTModel(GPT_CONFIG_124M)
    # model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004,weight_decay=0.1)
    # num_epochs = 10
    # train_losses,val_losses,tokens_seen=train_model_simple(
    #     model,train_loader,val_loader,optimizer,device,
    #     num_epochs=num_epochs,eval_freq=5,eval_iter=5,
    #     start_context="Every effort moves you", tokenizer=tokenizer
    # )
    # epochs_tensor=torch.linspace(0,num_epochs,len(train_losses))
    # plot_losses(epochs_tensor,tokens_seen,train_losses,val_losses)
    # tokenizer=tiktoken.get_encoding('gpt2')
    # token_ids=generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids("Every effort moves you",tokenizer),
    #     max_new_tokens=25,
    #     context_size=GPT_CONFIG_124M['context_length'],
    # )
    # print(f'Output text:\n{token_ids_to_text(token_ids,tokenizer)}')

    # 5.3.1小节

    # vocab = {
    #     "closer": 0,
    #     "every": 1,
    #     "effort": 2,
    #     "forward": 3,
    #     "inches": 4,
    #     "moves": 5,
    #     "pizza": 6,
    #     "toward": 7,
    #     "you": 8
    # }
    # inverse_vocab = {v: k for k, v in vocab.items()}
    # # print(inverse_vocab)
    # next_token_logits = torch.tensor([
    #     4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79
    # ])
    # probas = torch.softmax(next_token_logits, dim=0)
    # next_token_id=torch.argmax(probas).item()
    # print(inverse_vocab[next_token_id])
    # torch.manual_seed(123)
    # next_token_id=torch.multinomial(probas,num_samples=1).item()
    # print(inverse_vocab[next_token_id])
    # print_sample_tokens(probas)
    # temperatures = [1, 0.1, 5]
    # scaled_probas=[softmax_with_temperature(next_token_logits,T) for T in temperatures]
    # x=torch.arange(len(vocab))
    # bar_width=0.15
    # fig,ax=plt.subplots(figsize=(5,3))
    # for i,T in enumerate(temperatures):
    #     rects=ax.bar(x+i*bar_width,scaled_probas[i],bar_width,label=f'Temperature={T}')
    # ax.set_ylabel('Probability')
    # ax.set_xticks(x)
    # ax.set_xticklabels(vocab.keys(),rotation=90)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    # top_k = 3
    # top_logits, top_pos = torch.topk(next_token_logits, top_k)
    # # print(f'Top logits:{top_logits}')
    # # print(f'Top positions:{top_pos}')
    # new_logits = torch.where(
    #     condition=next_token_logits < top_logits[-1],
    #     input=torch.tensor(float('-inf')),
    #     other=next_token_logits
    # )
    # # print(new_logits)
    # topk_probas = torch.softmax(next_token_logits, dim=0)
    # # print(topk_probas)

    # model = GPTModel(GPT_CONFIG_124M)
    # tokenizer = tiktoken.get_encoding('gpt2')
    # torch.manual_seed(123)
    # token_ids = generate(
    #     model=model,
    #     idx=text_to_token_ids('Every effort moves you', tokenizer),
    #     max_new_tokens=15,
    #     context_size=GPT_CONFIG_124M['context_length'],
    #     top_k=25,
    #     temperature=1.4
    # )
    # print(f'Output text:\n{token_ids_to_text(token_ids,tokenizer)}')

    # 5.5节

    # url=(
    #     "https://raw.githubusercontent.com/rasbt/"
    #     "LLMs-from-scratch/main/ch05/"
    #     "01_main-chapter-code/gpt_download.py"
    # )
    # filename=url.split('/')[-1]
    # urllib.request.urlretrieve(url,filename)

    settings,params=download_and_load_gpt2(
        model_size='124M',models_dir='gpt2'
    )
    # print(f'Settings: {settings}')
    # print(f'Parameter dictionary keys:{params.keys()}')
    # print(params['wte'])
    # print(f'Token embedding weight tensor dimensions:{params['wte'].shape}')

    model_name = 'gpt2-small (124M)'
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({'context_length': 1024})
    NEW_CONFIG.update({'qkv_bias': True})
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    tokenizer = tiktoken.get_encoding('gpt2')
    device: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_weights_into_gpt(gpt,params)
    gpt.to(device)

    torch.manual_seed(123)
    token_ids=generate(
        model=gpt,
        idx=text_to_token_ids('Every effort moves you', tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG['context_length'],
        top_k=50,
        temperature=1.5
    )
    print(f'Output text\n{token_ids_to_text(token_ids,tokenizer)}')
