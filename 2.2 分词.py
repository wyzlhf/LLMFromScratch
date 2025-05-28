import re
from typing import Dict, List

with open('the-verdict.txt', 'r', encoding='utf8') as f:
    raw_text = f.read()
# preprocessed=re.split(r'([,.:l_!\']|——|\s)',raw_text)
preprocessed: list = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed: list = [item.strip() for item in preprocessed if item.strip()]
all_words: list = sorted(set(preprocessed))
vocab_size: int = len(all_words)
vocab: Dict[str, int] = {token: integer for integer, token in enumerate(all_words)}


# print(vocab.items())
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break
# for item in vocab.items():
#     print(item)

class SimpleTokenizerV1(object):
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int: Dict[str, int] = vocab
        self.int_to_str: Dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        preprocessed: list = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed: list = [item.strip() for item in preprocessed if item.strip()]
        ids: list = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        text: str = ' '.join([self.int_to_str[i] for i in ids])
        text: str = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
text= "Hello, do you like tea?"
ids = tokenizer.encode(text)
print(ids)
text=tokenizer.decode(ids)
print(text)
