import re
from typing import Dict, List

with open('the-verdict.txt', 'r', encoding='utf8') as f:
    raw_text = f.read()
# preprocessed=re.split(r'([,.:l_!\']|——|\s)',raw_text)
preprocessed: list = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed: list = [item.strip() for item in preprocessed if item.strip()]
all_tokens: list = sorted(set(preprocessed))
all_tokens.extend(['<|unk|>', ''])
# vocab_size: int = len(all_words)
vocab: Dict[str, int] = {token: integer for integer, token in enumerate(all_tokens)}


# for i,item in enumerate(list(vocab.items())[-5:]):
#     print(item)
class SimpleTokenizerV2(object):
    def __init__(self, vocab):
        self.str_to_int: Dict[str, int] = vocab
        self.int_to_str: Dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        preprocessed: List[str] = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed: List[str] = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids: List[int] = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        text: str = ' '.join([self.int_to_str[i] for i in ids])
        text: str = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " ".join((text1, text2))
tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
# print(tokenizer.decode(ids))