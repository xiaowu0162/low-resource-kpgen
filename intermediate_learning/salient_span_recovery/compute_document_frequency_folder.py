import argparse
import os
import sys
from collections import defaultdict
from typing import List

from tqdm import tqdm

from kpe.data import parse_xml_file, process_folder, process_folder_directly
from kpe.document_frequency import DocumentFrequency


def compute_df_xml_folder(
        input_dir: str,
        output_dir: str,
        n: int = 5,
        stopwords: bool = False,
        allowed_languages: List[str] = None,
        tags: List[str] = None,
):
    df_models = {}

    def compute_df_xml_file(f, name: str = None):
        content = parse_xml_file(f)
        content = defaultdict(lambda: None, content)
        contents = []
        for tag in tags:
            contents.append((content['lang_{}'.format(tag)], content[tag]))
        for lang, text in contents:
            if lang is not None and text is not None:
                if allowed_languages and lang not in allowed_languages:
                    continue

                if lang not in df_models:
                    try:
                        df_models[lang] = DocumentFrequency(language=lang, n=n,
                                                            spacy_model=None, stemmer=None,
                                                            stopwords=stopwords, normalization='stemming')
                    except FileNotFoundError:
                        pass
                model = df_models.get(lang, None)
                if model is not None:
                    model.process(text)
                else:
                    tqdm.write('Warning: {}: no spaCy model for language: {}'.format(name, lang), file=sys.stdout)

    process_folder_directly(input_dir, func=compute_df_xml_file)

    # Write document frequency to tsv files
    for lang, model in df_models.items():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'docfreq_{}.tsv'.format(lang))
        model.to_tsv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compute_document_frequency.py',
        description='Compute document frequency from patient documents for TfIdf models'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='input data, a folder containing patient documents (.xml)')
    parser.add_argument('--output', type=str, required=True,
                        help='output folder for document frequency files (.tsv)')
    parser.add_argument('-n', '--size', type=int, default=5,
                        help='maximum size (n-gram) of extracted keyphrases')
    parser.add_argument('--stopwords', action='store_true',
                        help='use stopwords to filter candidates')
    parser.add_argument('--languages', type=str, required=False, nargs='+',
                        help='only compute document frequency on documents in these languages')
    parser.add_argument('--tags', type=str, required=False, default=['abstract', 'description', 'claims'], nargs='+',
                        help='only compute document frequency from these fields')

    args = parser.parse_args()
    print(args)

    compute_df_xml_folder(
        input_dir=args.input,
        output_dir=args.output,
        n=args.size,
        stopwords=args.stopwords,
        allowed_languages=args.languages,
        tags=args.tags,
    )
