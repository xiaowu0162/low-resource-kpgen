import sys
import csv
import json 
from tqdm import tqdm


if len(sys.argv) != 4:
    print("Usage: python collect_tfidf_predictions.py tfidf_all_predictions input_json_data output_transformed_data")
    exit()


tfidf_all_predictions = sys.argv[1]
input_json_data = sys.argv[2]
output_transformed_data = sys.argv[3]


# build up index
print('building index')
docid2pred = {}
with open(tfidf_all_predictions, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    c = 0
    for row in tqdm(csvreader):
        c += 1
        if c == 1:
            continue
        docid = row[0].replace(".xml", "") .strip()
        kps = [x.replace("'", "").strip() for x in row[1][1:-1].split(',')]
        docid2pred[docid] = kps

print(list(docid2pred.keys())[:50])


# transform data
print('transforming data')
with open(input_json_data) as in_f, open(output_transformed_data, 'w') as out_f:
    for line in tqdm(in_f.readlines()):
        entry = json.loads(line)
        entry['present_kps_manual'] = entry['present_kps']
        entry['absent_kps_manual'] = entry['absent_kps']
        try:
            entry['present_kps'] = {'text':[], 'tokenized':[]}
            entry['absent_kps'] = {'text':[], 'tokenized':[]}
            entry['present_kps']['text'] = docid2pred[str(entry['id'])]
            out_f.write(json.dumps(entry))
            out_f.write('\n')
        except:
            print("Exception thrown when processing", entry['id'])
            continue
