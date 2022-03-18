from collections import Counter
import argparse
from tqdm import tqdm


special_chars = set(["<unk>", "[digit]", "[sep]", '.', ',', "'", '%', '(', ')', "''"])
def process(infile, outfile):
    with open(infile) as in_f, open(outfile, 'w') as out_f:
        words = [x for line in in_f.readlines() for x in line.strip().split() if (x not in special_chars and len(x)>1)]
        words_freq = Counter(words).most_common()
        for w, c in words_freq:
            out_f.write("{} {}\n".format(w, c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate_wordcount_vocab.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-input_source', default='', help='input source file')
    parser.add_argument('-output_vocab', default='', help='output vocab file')
    args = parser.parse_args()
    process(args.input_source, args.output_vocab)
