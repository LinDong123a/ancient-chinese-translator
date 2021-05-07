import argparse

import torch

from data.tokenization import CharTokenizer
from data.vocab import Vocab
from model import ModelInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--src_vocab_path", type=str, required=True, help="白话文词表路径")
    parser.add_argument("--trg_vocab_path", type=str, required=True, help="文言文词表路径")

    parser = ModelInterface.add_trainer_args(parser)

    args = parser.parse_args()

    src_tokenizer = CharTokenizer()
    src_tokenizer.load_vocab(args.src_vocab_path)

    trg_vocab = Vocab()
    trg_vocab.load(args.trg_vocab_path)

    model = ModelInterface.load_from_checkpoint(
        args.checkpoint_path,
        src_vocab=src_tokenizer.vocab,
        trg_vocab=trg_vocab,
        lr=0.01, num_epoch=100, steps_per_epoch=20,
        model_config={
            "embedding_dim": args.embedding_dim,
            "enc_hidden_dim": args.enc_hidden_dim,
            "dec_hidden_dim": args.dec_hidden_dim,
            "dropout": 0,
        },
    )

    model = model.eval()
    while True:
        sent = input("原始白话文:")

        input_token_list = src_tokenizer.tokenize(sent, map_to_id=True)
        res_sent = model.inference(
            torch.LongTensor([input_token_list]),
            torch.LongTensor([len(input_token_list)]),
        )[0]

        print(res_sent)
