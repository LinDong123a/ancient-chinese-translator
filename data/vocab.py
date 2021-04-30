import json


class Vocab(object):
    def __init__(self) -> None:
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"

        self.unk_idx = 0
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        self.token_to_idx = {
            self.unk_token: self.unk_idx,
            self.pad_token: self.pad_idx,
            self.sos_token: self.sos_idx,
            self.eos_token: self.eos_idx,
        }
        self.count_idx = len(self.token_to_idx)

        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

    def __len__(self):
        return len(self.token_to_idx)

    def add_token(self, token: str):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.count_idx
            self.idx_to_token[self.count_idx] = token
            self.count_idx += 1

    def stoi(self, token: str):
        """将token映射为idx"""
        return self.token_to_idx.get(token, self.unk_idx)

    def itos(self, idx: int):
        """将idx映射为token"""
        return self.idx_to_token[idx]

    def save(self, fpath: str):
        with open(fpath, "w", encoding="utf-8") as wfile:
            json.dump(self.token_to_idx, wfile, ensure_ascii=False, indent=4)

    def load(self, fpath: str):
        with open(fpath, "r", encoding="utf-8") as rfile:
            self.token_to_idx = json.load(rfile)

        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
