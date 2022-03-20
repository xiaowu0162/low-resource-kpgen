import sys

sys.path.insert(0, '../..')

import os
import json
from tqdm import tqdm
from pathlib import Path
from utils.constants import *

DATA_DIR = '../../data/'


def process_salient_span(infile, outdir, split):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(infile, "r", encoding='utf-8') as fin, \
            open(os.path.join(outdir, '{}.source'.format(split)), 'w', encoding='utf-8') as fsrc, \
            open(os.path.join(outdir, '{}.target'.format(split)), 'w', encoding='utf-8') as ftgt:
        for line in tqdm(fin):
            ex = json.loads(line)
            source = ex['title']['text'] + ' {} '.format(TITLE_SEP) + ex['abstract']['text']
            kps = ex['present_kps']['text'] + ex['absent_kps']['text']
            # prepend a KP_SEP to sequence to make encoding of the first kp consistent
            target = KP_SEP + ' ' + ' {} '.format(KP_SEP).join([t for t in kps if t])
            if len(source) > 0 and len(target) > 0:
                fsrc.write(source + '\n')
                ftgt.write(target + '\n')


if __name__ == '__main__':
    process_salient_span('tfidf/merged/kp20k.train.tfidfpred.jsonl', DATA_DIR + '/scikp/kp20k-salient-span/fairseq', 'train')
    process_salient_span('tfidf/merged/kp20k.valid.tfidfpred.jsonl', DATA_DIR + '/scikp/kp20k-salient-span/fairseq', 'valid')
    process_salient_span('tfidf/merged/kp20k.test.tfidfpred.jsonl', DATA_DIR + '/scikp/kp20k-salient-span/fairseq', 'test')
