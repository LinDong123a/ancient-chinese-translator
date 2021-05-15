import logging
from collections import Counter
from typing import List

import jiayan
import jieba

from .vocab import Vocab

logger = logging.getLogger(__name__)


class BaseTokenizer(object):
    def __init__(self) -> None:
        self.vocab = Vocab()

    def __len__(self):
        return len(self.vocab)

    def load_vocab(self, fpath: str):
        self.vocab.load(fpath)

    def save_vocab(self, fpath: str):
        self.vocab.save(fpath)

    def build_vocab(self, sent_list: List[List[str]], min_freq: int = 1):
        token_list = []
        for sent in sent_list:
            token_list.extend(self.tokenize(sent))

        counter = Counter(token_list)
        logger.info(f"origin token number: {len(counter)}")

        for token, num in counter.items():
            if num >= min_freq:
                self.vocab.add_token(token)

    def _tokenize(self, sent: str) -> List[str]:
        raise NotImplementedError

    def tokenize(self, sent: str, map_to_id: bool = False) -> List[str]:
        token_list = self._tokenize(sent)

        if map_to_id:
            return [self.vocab.stoi(s) for s in token_list]
        else:
            return token_list


class CharTokenizer(BaseTokenizer):
    def _tokenize(self, sent: str) -> List[str]:
        return list(sent.replace(" ", ""))


class AncientTokenTokenizer(BaseTokenizer):
    tokenizer = None

    def load_tokenizer(self):
        if self.tokenizer is None:
            lm = jiayan.load_lm("source/jiayan.klm")
            self.tokenizer = jiayan.CharHMMTokenizer(lm)

    def _tokenize(self, sent: str) -> List[str]:
        self.load_tokenizer()
        return self.tokenizer.tokenize(sent)


class VernacularTokenTokenizer(BaseTokenizer):
    def _tokenize(self, sent: str) -> List[str]:
        return jieba.cut(sent)
