import json
from tqdm import tqdm


BM25_PHRASELEN_TO_THRESHOLD = {1: 500, 2: 430, 3: 360}
REMOVE_DUPE = False


for subset in ['train', 'valid', 'test']:
    src_file = "../data/kp20k.{}.jsonl".format(subset)
    in_pred_file = "raw_preds/kp20k-{}-bm25-salient1000.json".format(subset)
    out_file = "merged/kp20k.{}.salient1000.merged.json".format(subset)

    with open(src_file) as in_src_f, open(in_pred_file) as in_pred_f, open(out_file, 'w') as out_f:
        for src_line in tqdm(in_src_f.readlines(), desc=subset):
            src_entry = json.loads(src_line)
            bm25_pred_entry = json.loads(in_pred_f.readline())
            bm25_spans = []
            for x in bm25_pred_entry['candidates']:
                if int(x[1]) <= BM25_PHRASELEN_TO_THRESHOLD[len(x[0].split())]:
                    bm25_spans.append(x[0].strip())
    
            out_entry = {}
            out_entry['id'] = int(bm25_pred_entry['id'][-1])
            out_entry['title'] = {'text': src_entry["title"], 'tokenized': src_entry["title"]}
            out_entry['abstract'] = {'text': src_entry["abstract"], 'tokenized': src_entry["abstract"]}
            out_entry['present_kps_manual'] = {'text': src_entry['present'], 'tokenized': src_entry['present']}
            out_entry['absent_kps_manual'] = {'text': src_entry['present'], 'tokenized': src_entry['present']}
            out_entry['present_kps'] = {'text': bm25_spans, 'tokenized': bm25_spans}
            out_entry['absent_kps'] = {'text': [], 'tokenized': []}

            out_f.write(json.dumps(out_entry).replace('<digit>', '[digit]'))
            out_f.write("\n")
