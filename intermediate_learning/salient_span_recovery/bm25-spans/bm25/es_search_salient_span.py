import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import argparse
import jsonlines

from tqdm import tqdm
from elasticsearch import Elasticsearch
from utils import filter_ngram


def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def search_es(es_obj, index_name, keywords, match_type='match', n_results=5):
    # keywords are a list of keyphrases
    # so, we use Boolean query to search for all keyphrases

    # match_type == "match" -> fuzzy match (keyphrases do not need to be exactly present in the document)
    # match_type == "match_phrase" -> exact match (keyphrases must be exactly present in the document)

    match = [{match_type: {"document_text": kw}} for kw in keywords]
    match_all_keywords = True
    if match_all_keywords:
        # if we want all keyword matches, then use "must" (equivalent to AND)
        # we form query as follows: keyphrase1 AND keyphrase2 AND ...
        query_dsl = {
            "query": {
                "bool": {
                    "must": match
                }
            }
        }
    else:
        # if we want 1+ keyword matches, then use "should" (equivalent to OR)
        # we form query as follows: keyphrase1 OR keyphrase2 OR ...
        query_dsl = {
            "query": {
                "bool": {
                    "should": match
                }
            }
        }

    res = es_obj.search(index=index_name, body=query_dsl, size=n_results)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', type=str, required=True, help='Path to index')
    parser.add_argument('--input_data_file', type=str, required=True, help='Path to index')
    parser.add_argument('--port', type=int, required=True, help='Port number')
    parser.add_argument('--output_fp', type=str, required=True)
    parser.add_argument('--keyword', type=str, required=True)
    parser.add_argument('--n_docs', type=int, default=100)

    args = parser.parse_args()
    input_data = read_jsonlines(args.input_data_file)
    config = {'host': 'localhost', 'port': args.port}
    es = Elasticsearch([config])
    result = {}

    with open(args.output_fp, 'w') as outfile:
        for i_item, item in tqdm(enumerate(input_data)):
            # get candidates upto MAX_N-gram
            MAX_N = 3
            tokens = [x.strip() for x in (item['title'] + ' ' + item['abstract']).strip().split()]
            candidates = set()
            for n in range(1, MAX_N + 1):
                for start in range(len(tokens) - n + 1):
                    end = start + n
                    cur_candidate = tokens[start:end]
                    if filter_ngram(cur_candidate, mode='any'):
                        continue
                    else:
                        candidates.add(' '.join(cur_candidate))

            candidate2pos = []
            for candidate_ngram in candidates:
                res = search_es(
                    es_obj=es,
                    index_name=args.index_name,
                    keywords=[candidate_ngram],
                    match_type='match',
                    n_results=args.n_docs
                )
                for i_doc, retrieved_doc in enumerate(res["hits"]["hits"]):
                    if retrieved_doc['_source']['document_title'] == item['id']:
                        candidate2pos.append([candidate_ngram, i_doc, len(res["hits"]["hits"])])
                        break

            candidate2pos.sort(key=(lambda x: x[1]))
            out_result = {"id": item['id'], "candidates": candidate2pos}

            outfile.write(json.dumps(out_result))
            outfile.write('\n')
            if i_item % 100 == 0:
                outfile.flush()


if __name__ == '__main__':
    main()
