import os
import time

import matplotlib.pyplot as plt
import torch
import zipfile
import tiktoken
import pandas as pd
import urllib.request
from pathlib import Path
from torch import device
from torch import Tensor
from pydantic import HttpUrl
from pandas import DataFrame
from tiktoken import Encoding
from typing import Tuple, List
from gpt_download import download_and_load_gpt2
from torch.utils.data import Dataset, DataLoader
from config import GPTModel, load_weights_into_gpt, generate_text_simple, text_to_token_ids, token_ids_to_text

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 300)


def download_and_unzip_spam_data(url: HttpUrl | str, zip_path: str, extracted_path: str, data_file_path) -> None:
    if data_file_path.exists():
        print(f'{data_file_path} already exists. Skipping download and extraction.')
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / 'SMSSpamCollection'
    os.rename(original_file_path, data_file_path)
    print(f'File downloaded and saved as {data_file_path}')


def creat_balanced_dataset(df: DataFrame) -> DataFrame:
    num_spam: int = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df: DataFrame = pd.concat([ham_subset, df[df['Label'] == 'spam']])
    return balanced_df


def random_split(df: DataFrame, train_frac: float, validation_frac: float) -> Tuple[DataFrame, DataFrame, DataFrame]:
    df: DataFrame = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end: int = int(len(df) * train_frac)
    validation_end: int = train_end + int(len(df) * validation_frac)

    train_df: DataFrame = df[:train_end]
    validation_df: DataFrame = df[train_end:validation_end]
    test_df: DataFrame = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: Encoding, max_length: int = None, pad_token_id: int = 50256) -> None:
        self.data: DataFrame = pd.read_csv(csv_file)
        self.encoded_texts: List[List[int]] = [
            tokenizer.encode(text) for text in self.data['Text']
        ]
        if max_length is None:
            self.max_length = self._longest_encode_length()
        else:
            self.max_length: int = max_length
            self.encoded_texts: List[List[int]] = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        self.encoded_texts: List[List[int]] = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        encoded: list[int] | list[list[int]] = self.encoded_texts[index]
        label: int = self.data.iloc[index]['Label']
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.data)

    def _longest_encode_length(self) -> int:
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_accuracy_loader(data_loader: DataLoader, model: GPTModel,
                         device: device, num_batches: int = None) -> float:
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_examples = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = logits.argmax(dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 1
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader: DataLoader, model: GPTModel, device: device, num_batches: int = None) -> float:
    total_loss = 0.
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        eval_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, eval_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f'Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}')
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f'Train accuracy:{train_accuracy * 100:.2f}% |', end='')
        print(f'Validation accuracy:{val_accuracy * 100:.2f}% |')
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label='loss'):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f'Training {label}')
    ax1.plot(epochs_seen, val_values, linestyle='-.', label=f'Validation {label}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel('Examples seen')

    fig.tight_layout()
    plt.savefig(f'{label}-plot.pdf')
    plt.show()


def classify_review(text: str, model: GPTModel, tokenizer: Encoding, device: device, max_length: int = None,
                    pad_token_id: int = 50256):
    model.eval()
    input_ids: list[int] = tokenizer.encode(text)
    supported_context_length: int = model.pos_emb.weight.shape[1]
    input_ids: list[int] = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor: Tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids)[:, -1, :]
    predicted_label: int | float | bool = torch.argmax(logits, dim=-1).item()
    return 'spam' if predicted_label == 1 else 'not spam'


if __name__ == '__main__':
    # 6.2节
    # url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    # zip_path = "sms_spam_collection.zip"
    # extracted_path = "sms_spam_collection"
    # data_file_path = Path(extracted_path) / 'SMSSpamCollection.tsv'
    # download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    # df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])
    # print(df.head())
    # print(df['Label'].value_counts())
    # balanced_df = creat_balanced_dataset(df)
    # print(balanced_df['Label'].value_counts())
    # balanced_df['Label'] = balanced_df['Label'].map({'ham': 0, 'spam': 1})
    # print(balanced_df.tail())
    # print(df.head())
    # train_df, validation_df, test_df=random_split(balanced_df,0.7,0.1)
    # train_df.to_csv('train.csv', index=False)
    # validation_df.to_csv('validation.csv', index=False)
    # test_df.to_csv('test.csv', index=False)

    # 6.3 创建数据加载器
    tokenizer = tiktoken.get_encoding('gpt2')
    # # print(tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'}))
    train_dataset = SpamDataset(csv_file='train.csv', max_length=None, tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file='test.csv', max_length=None, tokenizer=tokenizer)
    validation_dataset = SpamDataset(csv_file='validation.csv', max_length=None, tokenizer=tokenizer)
    # # print(train_data.max_length)
    num_workers = 0
    batch_size = 8
    # torch.manual_seed(123)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             drop_last=False)
    # for input_batch,target_batch in train_loader:
    # print(input_batch.shape)
    # print(target_batch.shape)
    # pass
    # print(f'Input batch dimensions: {input_batch.shape}')
    # print(f'Target batch dimensions: {target_batch.shape}')
    # print(f"{len(train_loader)} training batches")
    # print(f"{len(val_loader)} validation batches")
    # print(f"{len(test_loader)} test batches")

    # 6.4 Initializing a model with pretrained weights
    CHOOSE_MODEL = 'gpt2-small (124M)'
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,  # 1
        "context_length": 1024,  # 2
        "drop_rate": 0.0,  # 3
        "qkv_bias": True  # 4
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    # print(BASE_CONFIG)
    model_size = CHOOSE_MODEL.split(' ')[-1].lstrip('(').rstrip(')')
    # print(model_size)
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir='../gpt2'
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    # text_1 = 'Every effort moves you'
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids(text_1, tokenizer=tokenizer),
    #     max_new_tokens=15,
    #     context_size=BASE_CONFIG['context_length'],
    # )
    # print(token_ids_to_text(token_ids, tokenizer=tokenizer))
    #
    # text_2 = (
    #     "Is the following text 'spam'? Answer with 'yes' or 'no':"
    #     " 'You are a winner you have been specially"
    #     " selected to receive $1000 cash or a $2000 award.'"
    # )
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids(text_2, tokenizer),
    #     max_new_tokens=23,
    #     context_size=BASE_CONFIG["context_length"]
    # )
    # print(token_ids_to_text(token_ids, tokenizer))
    # for param in model.parameters():
    #     param.requires_grad = False
    # torch.manual_seed(123)
    # num_classes = 2
    # model.out_head = torch.nn.Linear(
    #     in_features=BASE_CONFIG['emb_dim'],
    #     out_features=num_classes
    # )
    # for param in model.trf_blocks[-1].parameters():
    #     param.requires_grad = True
    # for param in model.final_norm.parameters():
    #     param.requires_grad = True
    # inputs = tokenizer.encode("Do you have time")
    # inputs = torch.tensor(inputs).unsqueeze(0)
    # print("Inputs:", inputs)
    # print("Inputs dimensions:", inputs.shape)
    # with torch.no_grad():
    #     outputs = model(inputs)
    # print("Outputs:\n", outputs)
    # print("Outputs dimensions:", outputs.shape)
    # print("Last output token:", outputs[:, -1, :])
    # probas = torch.softmax(outputs[:, -1, :], dim=-1)
    # label = torch.argmax(probas)
    # print("Class label:", label.item())
    # logits = outputs[:, -1, :]
    # label = torch.argmax(logits)
    # print("Class label:", label.item())

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # torch.manual_seed(123)
    # train_accuracy = calc_accuracy_loader(
    #     train_loader, model, device, num_batches=10
    # )
    # val_accuracy = calc_accuracy_loader(
    #     val_loader, model, device, num_batches=10
    # )
    # test_accuracy = calc_accuracy_loader(
    #     test_loader, model, device, num_batches=10
    # )
    # print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    # print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    # print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # with torch.no_grad():  # 1
    #     train_loss = calc_loss_loader(
    #         train_loader, model, device, num_batches=5
    #     )
    #     val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    #     test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
    # print(f"Training loss: {train_loss:.3f}")
    # print(f"Validation loss: {val_loss:.3f}")
    # print(f"Test loss: {test_loss:.3f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f'Training completed in {execution_time_minutes:.2f} minutes.')

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(
        epochs_tensor, examples_seen_tensor, train_accs, val_accs,
        label="accuracy"
    )

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    #6.8节
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    torch.save(model.state_dict(), "review_classifier.pth")

    model_state_dict = torch.load("review_classifier.pth, map_location=device")
    model.load_state_dict(model_state_dict)
