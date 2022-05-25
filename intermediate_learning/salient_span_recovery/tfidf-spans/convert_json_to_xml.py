from dicttoxml import dicttoxml
import json
from tqdm import tqdm
import sys


if len(sys.argv) != 3:
    print("Usage: python convert_json_to_xml.py in_json_file out_xml_dir")
    exit()

in_file = sys.argv[1]
out_dir = sys.argv[2]

with open(in_file) as in_f:
    print("Converting to xml:", in_file)
    for line in tqdm(in_f.readlines()):
        entry = json.loads(line)
        docid = str(entry['id'])
        out_name = out_dir + '/' + docid + '.xml'
        out_entry = {'description': entry['title']['text'] + ' ' + entry['abstract']['text']}
        with open(out_name, 'wb') as out_f:
            out_f.write(dicttoxml(out_entry, attr_type=False))

