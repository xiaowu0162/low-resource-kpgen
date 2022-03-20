# prepare data for salient span masking
import sys
import numpy as np
import random
from tqdm import tqdm
from fairseq.data.encoders.gpt2_bpe import GPT2BPE


def prepare_ssr_masking_sharding(source_file_bpe, target_file_bpe, input_dir, output_dir, n_shards):
    for i in range(n_shards):
        print('Generating shard{}...'.format(i+1))
        prepare_ssr_masking_oneshot(source_file_bpe, target_file_bpe, input_dir, output_dir, shard_idx=i+1)


def prepare_ssr_masking_oneshot(source_file_bpe, target_file_bpe, input_dir, output_dir, shard_idx=None):
    bpe = GPT2BPE(None)
    REPEAT_TIMES = 1    # repeat each document for how many times
    MASK_PROB_KP = 0.8        # prob of masking each keyphrase
    MASK_PROB_OTHER = 0.2     # prob of masking each word that is not a keyphrase

    # Step 1: build, repeat, and shuffle data
    print("building data...")
    data = []
    with open(input_dir + source_file_bpe) as src_f, open(input_dir + target_file_bpe) as tgt_f:
        for src_line in tqdm(src_f.readlines()):
            src_line = src_line.strip()
            tgt_line = tgt_f.readline().strip()
            for _ in range(REPEAT_TIMES):
                data.append([src_line, tgt_line])
    random.shuffle(data)
    
    # Step 2: mask and output
    print("generating mask...")
    if shard_idx is None:
        out_src_file = output_dir + source_file_bpe #+ ".masked.source"
        out_tgt_file = output_dir + target_file_bpe #+ ".masked.target"
    else:
        out_src_file = output_dir + source_file_bpe + ".shard{}".format(shard_idx)#+ ".masked.source.shard{}".format(shard_idx)
        out_tgt_file = output_dir + target_file_bpe + ".shard{}".format(shard_idx) #+ ".masked.target.shard{}".format(shard_idx)
    mask_ratios = []
    with open(out_src_file, 'w') as out_src_f, open(out_tgt_file, 'w') as out_tgt_f:
        for original_src, original_tgt in tqdm(data):
            original_ntokens = len(original_src.split())
            out_tgt_f.write(original_src + '\n')

            # first mask out keyphrases
            original_tgt = original_tgt[3:].split('2162')
            original_tgt = [x.strip() for x in original_tgt]
            original_tgt.sort(key=(lambda x: len(x.split())), reverse=True)
            masked_src = original_src
            for kp in original_tgt:
                if kp == "":
                    continue
                src_line_temp = masked_src.strip().split(kp)
                masked_src = []
                for cur_i in range(len(src_line_temp)):
                    masked_src.append(src_line_temp[cur_i])
                    if cur_i != len(src_line_temp)-1:
                        r = random.random()
                        if r <= MASK_PROB_KP:
                            masked_src.append('51200')
                        else:
                            masked_src.append(kp)
                masked_src = " ".join(masked_src).replace('  ', ' ')

            # then mask out random words, replacing each span with a single mask
            masked_src_final = []
            append_flag_nonstarting = True
            for cur_token in masked_src.split():
                if cur_token == "51200":
                    masked_src_final.append(cur_token)
                elif not bpe.is_beginning_of_word(cur_token):
                    if append_flag_nonstarting:
                        masked_src_final.append(cur_token) 
                    else:
                        continue
                else:
                    r = random.random()
                    if r <= MASK_PROB_OTHER:
                        masked_src_final.append('51200')
                        append_flag_nonstarting = False
                    else:
                        masked_src_final.append(cur_token)
                        append_flag_nonstarting = True

            masked_src_final = ' '.join(masked_src_final)
            nonmasked_ntokens = len([x for x in masked_src_final.split() if x != "51200"])
            mask_ratios.append(1-nonmasked_ntokens/original_ntokens)
            
            out_src_f.write(masked_src + '\n')
    
    # print("Average mask ratio:", np.mean(np.array(mask_ratios)))



if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python prepare_ssr.py dataset_name n_shards input_dir output_dir")
        exit()
    
    dataset_name = sys.argv[1]
    n_shards = int(sys.argv[2])
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]

    for split in ['train', 'valid', 'test']:
        source_file_bpe = '{}.bpe.source'.format(split)
        target_file_bpe = '{}.bpe.target'.format(split)
        print(source_file_bpe)
        if split == 'train':
            prepare_ssr_masking_sharding(source_file_bpe, target_file_bpe, input_dir, output_dir, n_shards) 
        else:
            prepare_ssr_masking_oneshot(source_file_bpe, target_file_bpe, input_dir, output_dir)

    # for ds in ['kp20k']:
    #     for split in ['valid', 'test']:
    #         source_file_bpe = '/home/diwu/kpgen/NeuralKpGen/bart/data/{}-ssr-title-v2/{}.bpe.source'.format(ds, split)
    #         target_file_bpe = '/home/diwu/kpgen/NeuralKpGen/bart/data/{}-ssr-title-v2/{}.bpe.target'.format(ds, split)
    #         print(source_file_bpe)   
    #         prepare_ssp_masking_v3_oneshot(source_file_bpe, target_file_bpe)

    # for ds in ['kp20k']: 
    #     for split in ['train']:
    #         source_file_bpe = '/home/diwu/kpgen/NeuralKpGen/bart/data/{}-ssr-title-v2/{}.bpe.source'.format(ds, split)
    #         target_file_bpe = '/home/diwu/kpgen/NeuralKpGen/bart/data/{}-ssr-title-v2/{}.bpe.target'.format(ds, split)
    #         print(source_file_bpe)
    #         prepare_ssp_masking_v3_sharding(source_file_bpe, target_file_bpe, 35)
