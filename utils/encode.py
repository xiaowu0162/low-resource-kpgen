#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys
import argparse
import contextlib

from collections import Counter
from multiprocessing import Pool
from transformers import BertTokenizer
from fairseq.data.encoders.gpt2_bpe import get_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help='model name',
    )
    parser.add_argument(
        "--encoder-json",
        type=str,
        default=None,
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        default=None,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        if self.args.model == 'bart':
            bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)
        elif self.args.model in ['mass', 'prophetnet']:
            bpe = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode(self, line):
        global bpe
        if self.args.model == 'bart':
            ids = bpe.encode(line)
            return list(map(str, ids))
        elif self.args.model in ['mass', 'prophetnet']:
            return bpe._tokenize(line)
        else:
            return line.lower().split()

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            tokens = self.encode(line.strip())
            tokens = tokens[:self.args.max_len]
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]


if __name__ == "__main__":
    main()
