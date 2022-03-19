import sys

sys.path.insert(0, '../..')

import os
import json
from tqdm import tqdm
from pathlib import Path
from utils.constants import *

DATA_DIR = '../../data/'


def process_denoising(infile, outdir, split):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(infile, "r", encoding='utf-8') as fin, open(os.path.join(outdir, f'{split}'), 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            ex = json.loads(line)
            source = ex['title']['text'] + ' {} '.format(TITLE_SEP) + ex['abstract']['text']
            if len(source) > 0:
                fout.write(source + '\n')


if __name__ == '__main__':
    process_denoising(DATA_DIR + '/scikp/kp20k/processed/train.json', DATA_DIR + '/scikp/kp20k-lm/fairseq', 'train')
    process_denoising(DATA_DIR + '/scikp/kp20k/processed/valid.json', DATA_DIR + '/scikp/kp20k-lm/fairseq', 'valid')
    process_denoising(DATA_DIR + '/scikp/kp20k/processed/test.json', DATA_DIR + '/scikp/kp20k-lm/fairseq', 'test')
