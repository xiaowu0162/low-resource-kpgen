import json
import argparse


def MAP(inputfile):
    with open(inputfile) as f:
        examples = json.load(f)
    map = 0
    for id, ex in examples.items():
        average_precision = 0
        if ex["found"]:
            for j, hit in enumerate(ex["hits"]):
                if hit["doc_id"] == id:
                    average_precision = 1 / (j + 1)
                    break
        map += average_precision

    return map / len(examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Log JSON file')
    args = parser.parse_args()

    map = MAP(args.input_file)
    print("MAP - ", map)
