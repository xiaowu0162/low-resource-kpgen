import argparse
import logging
import os
from collections import defaultdict
from typing import List

import pandas as pd
from tqdm import tqdm

from kpe.base_structures import Document
from kpe.data import parse_xml_file, process_folder, process_folder_directly
from kpe.document_frequency import DocumentFrequency
from kpe.tfidf import TfIdf

logger = logging.getLogger(__name__)


def extract_kp_xml_folder(
        input_dir: str,
        model_dir: str,
        output_file: str,
        n: int = 4,
        k: int = 30,
        stopwords: bool = False,
        allowed_languages: List[str] = None,
        tags: List[str] = None,
        redundancy_removal: bool = False,
):
    tfidf_models = {}

    def extract_kp_xml_file(f, name: str = None):
        content = parse_xml_file(f)
        content = defaultdict(lambda: None, content)

        lang = content['lang_abstract']
        # fall back: using the description for KPE if there is no abstract
        if lang is None:
            lang = content['lang_description']

        text = content['abstract']
        # fall back: using the description for KPE if there is no abstract
        if text is None and content['description'] is not None:
            text = []

        if lang is not None and text is not None:
            if allowed_languages and lang not in allowed_languages:
                return None

            if lang not in tfidf_models:
                df_file = os.path.join(model_dir, 'docfreq_{}.tsv'.format(lang))
                if not os.path.exists(df_file):
                    tqdm.write('Warning: {}: not support language: {}'.format(name, lang))
                    return None
                num_docs, doc_freq = DocumentFrequency.read_tsv(df_file)
                tfidf_models[lang] = TfIdf(
                    document_frequency=doc_freq, num_documents=num_docs,
                    language=lang, n=n,
                    spacy_model=None, stemmer=None,
                    stopwords=stopwords, normalization='stemming'
                )
                tqdm.write('Load KPE model for language: {}'.format(lang))

            model = tfidf_models[lang]

            all_text = []
            if 'abstract' not in tags:
                all_text.extend(text)
            for tag in tags:
                if content['lang_{}'.format(tag)] == lang and content[tag] is not None:
                    all_text.extend(content[tag])
            document = model.read_text(all_text)

            if text == []:  # fall back: using first 100 tokens of the description as abstract
                description = model.read_text(content['description'])
                limit = 100
                sentences = []
                for sentence in description.sentences:
                    sentences.append(sentence[:limit])
                    limit -= len(sentence)
                    if limit <= 0:
                        break
                abstract = Document()
                abstract.sentences = sentences
                abstract.language = description.language

                if 'description' not in tags:  # description is originally not in the list of considered tags
                    document.sentences.extend(abstract.sentences)
            else:
                abstract = model.read_text(text)

            # Use all fields to compute term frequency
            all_candidates = model.extract_candidates(document)
            scores = model.score_candidates(all_candidates)

            # Extract candidates from abstract
            candidates = model.extract_candidates(abstract)
            keyphrases = model.extract_keyphrases(candidates, scores, k=k, redundancy_removal=redundancy_removal)

            return {
                'name': name,
                'keyphrases': [kp for kp, _ in keyphrases],
                'lang': lang,
            }

        return None

    results = process_folder_directly(input_dir, func=extract_kp_xml_file)

    # Write results to csv file
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_tfidf.py',
        description='Extract keyphrases from patient documents'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='input data, a folder contained compressed files (.tgz) of patient documents (.xml)')
    parser.add_argument('--model', type=str, required=True,
                        help='folder of document frequency files (.tsv) for TfIdf KPE models')
    parser.add_argument('--output', type=str, required=True,
                        help='output file in .csv format')
    parser.add_argument('-n', '--size', type=int, default=3,
                        help='maximum size (n-gram) of extracted keyphrases')
    parser.add_argument('-k', '--top', type=int, default=30,
                        help='maximum number of keyphrases to extract')
    parser.add_argument('--stopwords', action='store_true',
                        help='use stopwords to filter candidates')
    parser.add_argument('--languages', type=str, required=False, nargs='+',
                        help='only extract keyphrases on documents in these languages')
    parser.add_argument('--tags', type=str, required=False, default=['abstract', 'description', 'claims'], nargs='+',
                        help='only compute document frequency from these fields')
    parser.add_argument('--redundancy_removal', type=bool, default=False,
                        help='remove redundant keyphrases (slow)')

    args = parser.parse_args()
    print(args)

    extract_kp_xml_folder(
        input_dir=args.input,
        model_dir=args.model,
        output_file=args.output,
        n=args.size,
        k=args.top,
        stopwords=args.stopwords,
        allowed_languages=args.languages,
        tags=args.tags,
        redundancy_removal=args.redundancy_removal,
    )
