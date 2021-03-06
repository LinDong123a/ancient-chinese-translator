import argparse

import torch

from data.tokenization import CharTokenizer, VernacularTokenTokenizer
from data.vocab import Vocab
from model import ModelInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--token_type", type=str, default="char", choices=["char", "token"])
    parser.add_argument("--src_vocab_path", type=str, required=True, help="白话文词表路径")
    parser.add_argument("--trg_vocab_path", type=str, required=True, help="文言文词表路径")

    parser = ModelInterface.add_trainer_args(parser)

    args = parser.parse_args()

    if args.token_type == "char":
        src_tokenizer = CharTokenizer()
    elif args.token_type == "token":
        src_tokenizer = VernacularTokenTokenizer()

    src_tokenizer.load_vocab(args.src_vocab_path)

    trg_vocab = Vocab()
    trg_vocab.load(args.trg_vocab_path)

    model = ModelInterface.load_from_checkpoint(
        args.checkpoint_path,
        src_vocab=src_tokenizer.vocab,
        trg_vocab=trg_vocab,
    )

    model = model.eval()
    while True:
        sent = input("原始白话文:")

        input_token_list = src_tokenizer.tokenize(sent, map_to_id=True)
        res_sent = model.inference(
            torch.LongTensor([input_token_list]),
            torch.LongTensor([len(input_token_list)]),
        )[0]

        print(f"古文:{res_sent}")
