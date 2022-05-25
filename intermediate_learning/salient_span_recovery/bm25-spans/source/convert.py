import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import argparse
import subprocess
from tqdm import tqdm
from nltk.stem.porter import *
from nltk.tokenize import wordpunct_tokenize

stemmer = PorterStemmer()


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').strip().split(' ')
    return int(num[0])


def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


def stem_text(text):
    return ' '.join(stem_word_list(text.split()))


def separate_present_absent(source_text, keyphrases):
    present_kps, absent_kps = [], []
    stemmed_source = stem_text(source_text)
    for kp in keyphrases:
        stemmed_kp = stem_text(kp)
        if stemmed_kp in stemmed_source:
            present_kps.append(kp)
        else:
            absent_kps.append(kp)

    return present_kps, absent_kps


def scikp(args):
    idx = 0
    with open(args.out_file, 'w', encoding='utf8') as fw:
        with open(args.src_file) as f1, open(args.tgt_file) as f2:
            for source, target in zip(f1, f2):
                source = source.strip()
                target = target.strip()
                if len(source) == 0 or len(target) == 0:
                    continue
                title, abstract = [s.strip() for s in source.split('<eos>')]
                if len(title) == 0 or len(abstract) == 0:
                    continue
                pkps, akps = [kp.strip().split(';') for kp in target.split('<peos>')]
                pkps = [kp for kp in pkps if kp]
                akps = [kp for kp in akps if kp]
                if len(pkps) == 0 and len(akps) == 0:
                    continue
                ex_idx = ''
                if args.dataset:
                    ex_idx += args.dataset + '.'
                if args.split:
                    ex_idx += args.split + '.'
                ex_idx += str(idx)
                obj = {
                    'id': ex_idx,
                    'title': title,
                    'abstract': abstract,
                    'present': pkps,
                    'absent': akps
                }
                idx += 1
                fw.write(json.dumps(obj) + '\n')


def kptimes(args):
    idx = 0
    with open(args.out_file, 'w', encoding='utf8') as fw:
        with open(args.input_file) as f:
            for line in tqdm(f, total=count_file_lines(args.input_file)):
                ex = json.loads(line.strip())
                if len(ex['title']) == 0 or len(ex['abstract']) == 0:
                    continue
                keyphrases = ex['keyword'].split(';')
                text = (ex['title'] + ' ' + ex['abstract']).strip().lower()
                text = ' '.join(wordpunct_tokenize(text))
                pkps, akps = separate_present_absent(text, keyphrases)
                obj = {
                    'id': ex['id'],
                    'title': ex['title'],
                    'abstract': ex['abstract'],
                    'present': pkps,
                    'absent': akps
                }
                idx += 1
                fw.write(json.dumps(obj) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-input_file', default=None, help='Input json file')
    parser.add_argument('-src_file', default=None, help='Source *.txt file')
    parser.add_argument('-tgt_file', default=None, help='Target *.txt file')
    parser.add_argument('-out_file', required=True, help='Output file path')
    parser.add_argument('-dataset', default=None, help='Dataset name',
                        choices=['kp20k', 'inspec', 'nus', 'krapivin', 'semeval', 'kptimes'])
    parser.add_argument('-split', default=None, help='Dataset name')
    args = parser.parse_args()
    if args.dataset == 'kptimes':
        assert args.input_file is not None
        kptimes(args)
    else:
        assert args.src_file is not None
        assert args.tgt_file is not None
        scikp(args)
