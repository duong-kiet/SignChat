from collections import Counter
from typing import List
from torch.utils.data import Dataset

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN


class Vocabulary:
    """Đại diện cho ánh xạ giữa tokens và chỉ số."""
    def __init__(self, tokens: List[str] = None, file: str = None) -> None:
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.stoi = {token: idx for idx, token in enumerate(self.specials)}
        self.itos = self.specials.copy()

        if tokens is not None:
            self.add_tokens(tokens)
        elif file is not None:
            self._from_file(file)

    def _from_file(self, file: str) -> None:
        with open(file, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f]
        self.add_tokens(tokens)

    def add_tokens(self, tokens: List[str]) -> None:
        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
    def tokens_to_indices(self, sentence: List[str]) -> List[int]:
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in sentence]

    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.itos[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.itos)

def build_vocab(field: str, max_size: int, min_freq: int, dataset: Dataset, vocab_file: str = None) -> Vocabulary:
    """
    Xây dựng từ vựng từ dataset hoặc file từ vựng.

    :param field: trường dữ liệu ("src" hoặc "trg")
    :param max_size: kích thước tối đa của từ vựng
    :param min_freq: tần suất tối thiểu để đưa vào từ vựng
    :param dataset: dataset để trích xuất từ vựng
    :param vocab_file: file chứa từ vựng nếu có
    :return: đối tượng Vocabulary
    """

    if vocab_file is not None:
        return Vocabulary(file=vocab_file)
    else:
        counter = Counter()
        for example in dataset:
            if field == "src":
                counter.update(example['src'])
            elif field == "trg":
                counter.update(example['trg'])
        
        tokens = [token for token, freq in counter.items() if freq >= min_freq]
        tokens = sorted(tokens, key=lambda x: (-counter[x], x))[:max_size]

        return Vocabulary(tokens=tokens)


