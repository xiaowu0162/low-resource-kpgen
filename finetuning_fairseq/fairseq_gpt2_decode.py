import sys
from tqdm import tqdm
from fairseq.data.encoders.gpt2_bpe import GPT2BPE


def decode(in_file, out_file):
    bpe = GPT2BPE(None)
    with open(in_file) as in_f, open(out_file, 'w') as out_f:
        for line in tqdm(in_f.readlines()):
            decoded_string = bpe.decode(line.strip())
            out_f.write(decoded_string)
            out_f.write('\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python fairseq_gpt2_decode.py in_file_name out_file_name")
        exit()

    decode(sys.argv[1], sys.argv[2])
