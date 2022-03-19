import sys

sys.path.insert(0, '../..')

import os
import json
from tqdm import tqdm
from pathlib import Path
from utils.constants import *

DATA_DIR = '../../data/'


def process_titlegen(infile, outdir, split):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(infile, "r", encoding='utf-8') as fin, \
            open(os.path.join(outdir, '{}.source'.format(split)), 'w', encoding='utf-8') as fsrc, \
            open(os.path.join(outdir, '{}.target'.format(split)), 'w', encoding='utf-8') as ftgt:
        for line in tqdm(fin):
            ex = json.loads(line)
            source = ex['abstract']['text']
            target = ex['title']['text']
            if len(source) > 0 and len(target) > 0:
                fsrc.write(source + '\n')
                ftgt.write(target + '\n')


if __name__ == '__main__':
    process_titlegen(DATA_DIR + '/scikp/kp20k/processed/train.json', DATA_DIR + '/scikp/kp20k-titlegen/fairseq', 'train')
    process_titlegen(DATA_DIR + '/scikp/kp20k/processed/valid.json', DATA_DIR + '/scikp/kp20k-titlegen/fairseq', 'valid')
    process_titlegen(DATA_DIR + '/scikp/kp20k/processed/test.json', DATA_DIR + '/scikp/kp20k-titlegen/fairseq', 'test')
