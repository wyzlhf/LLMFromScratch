import json
import os
import re
import time
import torch
import urllib
import urllib.request
import tiktoken
from typing import Dict, List, Tuple

from tqdm import tqdm
from pydantic import Json
from functools import partial
from pydantic.types import AnyType
from tiktoken import Encoding
from torch import Tensor, device
from torch.utils.data import Dataset, DataLoader
from gpt_download import download_and_load_gpt2
from config import (
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    generate,
    calc_loss_loader,
    train_model_simple,
    plot_losses
)


def download_and_load_file(file_path: str, url: str) -> Json:
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    with open(file_path, 'r') as file:
        data: Json = json.load(file)
    return data


def format_input(entry: Dict) -> str:
    instruction_text: str = (
        f'Below is an instruction that describes a task.'
        f'Below is an instruction that describes a task.'
        f'\n\n### Instrucion:\n{entry['instruction']}'
    )
    input_text: str = (
        f'\n\n### Input:\n{entry["input"]}' if entry["input"] else ''
    )
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: Encoding) -> None:
        self.data: List[Dict[str, str]] = data
        self.encoded_texts: List[List[int]] = []
        for entry in data:
            instruction_plus_input: str = format_input(entry)
            response_text: str = f'\n\n### Response:\n{entry['output']}'
            full_text: str = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index: int) -> list[int]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)


def custom_collate_draft_1(batch: List[list[int]], pad_token_id: int = 50256,
                           device: device = 'cpu') -> Tensor:
    batch_max_length: int = max(len(item) + 1 for item in batch)
    inputs_lst: List[Tensor] = []
    for item in batch:
        new_item: List = item.copy()
        new_item += [pad_token_id]
        padded: List = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs: Tensor = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    inputs_tensor: Tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


def custom_collate_draft_2(
        batch: List[List[int]],
        pad_token_id: int = 50256,
        device: device = 'cpu') -> Tuple[Tensor, Tensor]:
    batch_max_length: int = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item: List = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs: Tensor = torch.tensor(padded[:-1])
        targets: Tensor = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor: Tensor = torch.stack(inputs_lst).to(device)
    targets_tensor: Tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def custom_collate_fn(batch: List[List[int]], pad_token_id: int = 50256, ignore_index: int = -100,
                      allowed_max_length: int | bool = None, device: device = 'cpu') -> Tuple[Tensor, Tensor]:
    batch_max_length: int = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item: List = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs: Tensor = torch.tensor(padded[:-1])
        targets: Tensor = torch.tensor(padded[1:])
        mask: Tensor = torch.tensor(targets == pad_token_id)
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor: Tensor = torch.stack(inputs_lst).to(device)
    targets_tensor: Tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024,
)

if __name__ == '__main__':
    # 7.2 准备用于监督指令微调的数据集
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data: Json | AnyType = download_and_load_file(file_path, url)
    # # print("Number of entries:", len(data))
    # print("Example entry:\n", data[0])

    model_input: str = format_input(data[999])
    desired_response: str = f'\n\n### Response:\n{data[50]['output']}'
    # print(model_input+desired_response)

    train_portion: int = int(len(data) * 0.85)
    test_portion: int = int(len(data) * 0.1)
    val_portion: int = len(data) - train_portion - test_portion

    train_data: Json | AnyType = data[:train_portion]
    test_data: Json | AnyType = data[train_portion:train_portion + test_portion]
    val_data: Json | AnyType = data[train_portion + test_portion:]

    # print(f'Training set length: {len(train_data)}')
    # print(f'Validation set length: {len(val_data)}')
    # print(f'Test set length: {len(test_data)}')
    tokenizer = tiktoken.get_encoding('gpt2')
    # print(tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'}))
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = [
        inputs_1,
        inputs_2,
        inputs_3
    ]
    # print(custom_collate_draft_1(batch))
    # inputs, targets = custom_collate_draft_2(batch)
    # print(inputs)
    # print(targets)
    inputs, targets = custom_collate_fn(batch)
    # print(inputs)
    # print(targets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    # for inputs, targets in train_loader:
    #     print(inputs.shape, targets.shape)

    # 7.5 Loading a pretrained LLM
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    # print(input_text)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].strip()
    # print(response_text)

    model.to(device)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    torch.manual_seed(123)
    for entry in test_data[:3]:  # 1
        input_text = format_input(entry)
    token_ids = generate(  # 2
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\\n>> {entry['output']}")
    print(f"\nModel response:\\n>> {response_text.strip()}")
    print("-------------------------------------")

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"  # 1
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    print()
