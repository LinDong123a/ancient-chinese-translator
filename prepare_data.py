import argparse
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Union

from data.tokenization import (
    AncientTokenTokenizer, CharTokenizer, VernacularTokenTokenizer,
)

logging.basicConfig(level=logging.INFO)


TRAIN_FILE_NAME = "train.tsv"
TEST_FILE_NAME = "test.tsv"
VALID_FILE_NAME = "valid.tsv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="原始数据文件夹保存路径")
    parser.add_argument("--num_worker", type=int, default=None, help="处理数据的线程数")

    parser.add_argument("--dataset_size", type=int, default=-1, help="数据集的大小")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集占比")
    parser.add_argument("--valid_size", type=float, default=0.1, help="验证集占比")

    parser.add_argument("--save_dir", type=str, default="data", help="要保存处理后的文件路径")
    parser.add_argument(
        "--train_file_name", type=str, default=TRAIN_FILE_NAME, help="保存的训练集文件名",
    )
    parser.add_argument(
        "--test_file_name", type=str, default=TEST_FILE_NAME, help="保存的测试集文件名",
    )
    parser.add_argument(
        "--valid_file_name", type=str, default=VALID_FILE_NAME, help="保存的验证集文件名",
    )

    parser.add_argument("--seed", type=int, default=100, help="划分数据集时的随机种子")
    parser.add_argument("--min_freq", type=int, default=100, help="词表的最小值")
    parser.add_argument(
        "--token_type", type=str, choices=["char", "token"], default="char",
        help="划分的词表类型",
    )

    return parser.parse_args()


def get_ancient_chinese_and_vernacular_file_mapper(data_dir_path: str) -> dict:
    """获取古文原文和现代文的文件对应表

    Args:
        data_dir_path (str): 数据文件夹路径

    Returns:
        dict: 文件映射表
    """
    data_dir_path: Path = Path(data_dir_path)

    fpath_set = set(data_dir_path.iterdir())

    mapper = {}

    for fpath in fpath_set:
        vernacular_fpath = fpath.parent / f"{fpath.name}翻译"

        if vernacular_fpath in fpath_set:
            mapper[str(vernacular_fpath)] = str(fpath)

    return mapper


def build_vernacular_file_to_ancient_chinese_mapper(
    a2v_mapper: dict, num_worker: int = None,
) -> dict:
    """根据文言文和现代文的文件映射关系，生成句对

    Args:
        a2v_mapper (dict): 输入的文件映射文件，格式为:

            .. code-block:: json
                {
                    "元史翻译": "元史"
                }
        num_worker (int): 多线程的并发数

    Returns:
        dict: 生成的句对，格式为:

            .. code-block:: json

                {
                    "你好骚啊": "汝甚骚",
                }
    """
    def _build_mapper(v_fpath: str, a_fpath: str):
        with open(v_fpath, "r", encoding="utf-8") as v_rfile, \
                open(a_fpath, "r", encoding="utf-8") as a_rfile:
            for vernacular_text, ancient_text in zip(v_rfile, a_rfile):
                text_pair[vernacular_text.strip()] = ancient_text.strip()

    text_pair = {}
    if num_worker is None:
        for v_fpath, a_fpath in a2v_mapper.items():
            _build_mapper(a_fpath=a_fpath, v_fpath=v_fpath)
    else:
        map_item_list = list(a2v_mapper.items())
        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            res = executor.map(
                _build_mapper,
                [i[1] for i in map_item_list],
                [i[0] for i in map_item_list],
            )

            for idx, _ in enumerate(res):
                print(f"{idx}/{len(map_item_list)}")

    return text_pair


def save_pair_to_tsv_file(pairs: List[Tuple[str, str]], fpath: Union[str, Path]):
    """将句对保存为tsv文件用于训练

    Args:
        pairs (Dict[str, str]): key为现代汉语，value为古代汉语
        fpath (Union[str, Path]): 要保存的文件路径
    """
    if isinstance(fpath, str):
        fpath: Path = Path(fpath)

    if not fpath.parent.exists():
        fpath.parent.mkdir()

    with fpath.open("w", encoding="utf-8") as wfile:
        for pair0, pair1 in pairs:
            wfile.write(f"{pair0}\t{pair1}\n")


if __name__ == "__main__":
    args = parse_args()

    a2v_mapper = get_ancient_chinese_and_vernacular_file_mapper(args.data_path)
    all_text_pair = build_vernacular_file_to_ancient_chinese_mapper(
        a2v_mapper, args.num_worker,
    )
    all_text_pair = list(all_text_pair.items())

    print("number of pairs", len(all_text_pair))

    if args.token_type == "char":
        src_tokenizer, trg_tokenizer = CharTokenizer(), CharTokenizer()
    elif args.token_type == "token":
        src_tokenizer, trg_tokenizer = (
            VernacularTokenTokenizer(), AncientTokenTokenizer(),
        )

    random.seed(args.seed)
    random.shuffle(all_text_pair)

    if args.dataset_size > 0:
        all_text_pair = all_text_pair[:args.dataset_size]

    test_size, valid_size = args.test_size, args.valid_size
    if test_size + valid_size >= 1:
        raise ValueError("Sum of test size and valid size must be less than 1")

    # tokenize text
    for idx, (src_text, trg_text) in enumerate(all_text_pair):
        all_text_pair[idx] = (
            " ".join(src_tokenizer.tokenize(src_text)),
            " ".join(trg_tokenizer.tokenize(trg_text)),
        )

    num_test_dataset = round(len(all_text_pair) * test_size)
    num_valid_dataset = round(len(all_text_pair) * valid_size)
    num_train_dataset = len(all_text_pair) - num_test_dataset - num_valid_dataset

    training_text_pairs = all_text_pair[:num_train_dataset]
    valid_text_pairs = all_text_pair[
        num_train_dataset: num_train_dataset + num_valid_dataset
    ]
    test_text_pairs = all_text_pair[num_train_dataset + num_valid_dataset:]

    # Build vocab
    src_tokenizer.build_vocab(
        [src for src, _ in training_text_pairs], min_freq=args.min_freq,
    )
    trg_tokenizer.build_vocab(
        [trg for _, trg in training_text_pairs], min_freq=args.min_freq,
    )

    save_dir = Path(args.save_dir)
    save_pair_to_tsv_file(training_text_pairs, save_dir / args.train_file_name)
    save_pair_to_tsv_file(test_text_pairs, save_dir / args.test_file_name)
    save_pair_to_tsv_file(valid_text_pairs, save_dir / args.valid_file_name)

    src_tokenizer.save_vocab(str(save_dir / "src_vocab.json"))
    trg_tokenizer.save_vocab(str(save_dir / "trg_vocab.json"))

    print(f"num src vocab size: {len(src_tokenizer)}")
    print(f"num trg vocab size: {len(trg_tokenizer)}")

    print(
        f"num training: {num_train_dataset}, num test: {num_test_dataset}, "
        f"num valid: {num_valid_dataset}",
    )
