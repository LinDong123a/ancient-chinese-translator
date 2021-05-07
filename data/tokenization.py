from collections import Counter
from typing import List

from .vocab import Vocab


class BaseTokenizer(object):
    def __init__(self) -> None:
        self.vocab = Vocab()

    def __len__(self):
        return len(self.vocab)

    def tokenize(self, sent: str) -> List[str]:
        raise NotImplementedError

    def load_vocab(self, fpath: str):
        self.vocab.load(fpath)

    def save_vocab(self, fpath: str):
        self.vocab.save(fpath)

    def build_vocab(self, sent_list: List[List[str]], min_freq: int = 1):
        token_list = []
        for sent in sent_list:
            token_list.extend(self.tokenize(sent))

        counter = Counter(token_list)
        for token, num in counter.items():
            if num >= min_freq:
                self.vocab.add_token(token)


class CharTokenizer(BaseTokenizer):
    def tokenize(self, sent: str, map_to_id: bool = False) -> List[str]:
        token_list = list(sent.replace(" ", ""))

        if map_to_id:
            return [self.vocab.stoi(s) for s in token_list]
        else:
            return token_list
