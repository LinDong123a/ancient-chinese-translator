import os

import flask
import torch
from flask import abort, jsonify, render_template, request

from data.tokenization import CharTokenizer
from data.vocab import Vocab
from model import ModelInterface

app = flask.Flask(
    __name__, template_folder="templates", static_folder="./", static_url_path="",
)

if "MODEL_DIR" not in os.environ:
    print("MODEL_DIR must be speicified before launching server")
    exit(1)

model_dir = os.environ["MODEL_DIR"]

src_tokenizer = CharTokenizer()
src_tokenizer.load_vocab(os.path.join(model_dir, "src_vocab.json"))

trg_vocab = Vocab()
trg_vocab.load(os.path.join(model_dir, "trg_vocab.json"))

model = ModelInterface.load_from_checkpoint(
    os.path.join(model_dir, "checkpoint.pt"),
    src_vocab=src_tokenizer.vocab,
    trg_vocab=trg_vocab,
    model_name="transformer",
).to("cuda" if torch.cuda.is_available() else "cpu")

model = model.eval()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/query/", methods=["POST"])
def query():
    if not request.json:
        abort(400)

    sent = request.json["text"]
    if sent == "":
        return jsonify({"text": ""})

    # 按照标点符号进行切分
    valid_punc = {".", "。", "?", "？", "!", "！"}
    prev_start_idx, sent_list = 0, []
    for idx, char in enumerate(sent):
        if char in valid_punc:
            sent_list.append(sent[prev_start_idx: idx + 1])
            prev_start_idx = idx + 1

    if prev_start_idx < len(sent):
        part = sent[prev_start_idx:]
        add_punc = True
        for punc in valid_punc:
            if part.endswith(punc):
                add_punc = False
                break

        if add_punc:
            part += "。"

        sent_list.append(part)

    output_list = []
    for sent in sent_list:
        input_token_list = src_tokenizer.tokenize(sent, map_to_id=True)
        res_sent = model.inference(
            torch.LongTensor([input_token_list]),
            torch.LongTensor([len(input_token_list)]),
        )[0]
        output_list.append(res_sent)

    output = "".join(output_list)
    print(sent, "=====", output)

    return jsonify({"text": "".join(output_list)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
