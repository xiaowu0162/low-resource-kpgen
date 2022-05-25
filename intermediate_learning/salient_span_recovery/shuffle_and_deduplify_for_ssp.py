import sys
import random
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from tqdm import tqdm


global bpe
bpe = GPT2BPE(None)


def dedupe_bpe_line(bpe_line):
    global bpe
    decoded_string = bpe.decode(bpe_line.strip())
    phrases = [x.strip() for x in decoded_string.split(';') if len(x.strip()) > 0]
    phrases_new = []
    for x in phrases:
        dup = False
        for y in phrases:
            if x != y and x in y:
                dup = True
                break
        if not dup:
            phrases_new.append(x)

    return bpe.encode('; ' + ' ; '.join(phrases_new))


def main():
    data = []
    with open(sys.argv[1]) as src_f, open(sys.argv[2]) as tgt_f:
        for src_line in src_f.readlines():
            tgt_line = tgt_f.readline()
            tgt_line = dedupe_bpe_line(tgt_line) + '\n'
            data.append([src_line, tgt_line])

    random.shuffle(data)
        
    with open(sys.argv[3] + '/train.bpe.source', 'w') as src_f, open(sys.argv[3] + '/train.bpe.target', 'w') as tgt_f:
        for src_line, tgt_line in tqdm(data, desc=sys.argv[3].split('/')[-1]):
            src_f.write(src_line)
            tgt_f.write(tgt_line)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python shuffle_for_ssp.py train.bpe.source train.bpe.target out_dir")
        exit()
    main()

