import logging
from typing import Union, List, Tuple

import spacy
from nltk import SnowballStemmer

from .base_structures import Document, Sentence, Candidate
from .filters import AbstractCandidateFilter

__all__ = [
    'default_stemmer', 'default_spacy_model',
    'process_text', 'generate_candidates', 'filter_candidates',
    'remove_redundant_keyphrases',
]

logger = logging.getLogger(__name__)

_spacy_models = {}  # storing default models to avoid re-initialization

LANGUAGE_ISO6391 = {
    'en': 'english',
    'de': 'german',
}


def default_stemmer(language: str = 'en'):
    code = language
    language = LANGUAGE_ISO6391.get(code, None)
    if language is not None:
        try:
            return SnowballStemmer(language)
        except ValueError as e:
            logger.warning(e)
            return None
    logger.warning('Unknown language code: {}'.format(code))
    return None


def default_spacy_model(language: str = 'en'):
    if language in _spacy_models is not None:
        spacy_model = _spacy_models[language]
    else:

        spacy_model = spacy.load(language, disable=['parser', 'ner'])
        spacy_model.add_pipe(spacy_model.create_pipe('sentencizer'))
        _spacy_models[language] = spacy_model

    return spacy_model


def process_text(
        text: Union[str, List[str]],
        language: str = 'en',
        spacy_model=None,
        stemmer=None,
) -> Document:
    """
    Process (sentence segmentation, tokenization, lemmatization, stemming, POS tagging) text
    and save results into a :class:`Document` object.

    Stemming can be done by a Snowball stemmer (of the corresponding language) in package :mod:`nltk.stem.snowball`
    (using :func:`default_stemmer`).
    The rest of processing is done by a spaCy model (of the corresponding language)
    (using :func:`default_spacy_model`).
    If providing no stemmer, stems fall back to lemmas.

    Parameters
         text:
            input text
        language: str, default to *en*
            the language code (ISO 639-1) of the text
        spacy_model:
            spaCy model
        stemmer: default to the :class:`nltk.SnowballStemmer` stemmer of the corresponding language
            stemmer

    Return
        document: :class:`Document`
            processed text
    """
    document = Document()
    document.language = language
    assert spacy_model is not None

    if isinstance(text, str):
        text = [text]

    for line in text:
        doc = spacy_model(line)
        for sent in doc.sents:
            sentence = Sentence()
            sentence.words = [token.text for token in sent]
            sentence.lemmas = [token.lemma_ for token in sent]
            sentence.utags = [token.pos_ for token in sent]
            sentence.xtags = [token.tag_ for token in sent]
            sentence.stems = [stemmer.stem(word) for word in sentence.words] \
                if stemmer is not None \
                else [token.lemma_ for token in sent]  # fall back if no available stemmer
            sentence.stopwords = [token.is_stop for token in sent]
            document.sentences.append(sentence)

    return document


def generate_candidates(
        document: Document,
        n: int = 3,
) -> List[Candidate]:
    """
    Generate *n*-gram candidates up to a certain length
    """
    candidates = []

    for sentence in document.sentences:
        for i in range(len(sentence)):
            for k in range(1, min(n + 1, len(sentence) - i + 1)):
                j = i + k
                candidates.append(Candidate(sentence=sentence, i=i, j=j))

    return candidates


def filter_candidates(
        candidates: List[Candidate],
        filters: List[AbstractCandidateFilter]
) -> List[Candidate]:
    """
    Filter candidates using a list of filters
    """
    filtered_candidates = []
    for candidate in candidates:
        if all([not filter.filter(candidate) for filter in filters]):
            filtered_candidates.append(candidate)
    return filtered_candidates


def remove_redundant_keyphrases(ranked_keyphrases: List[Tuple[str, float]]):
    """
    Remove redundant keyphrases (in normalized/lexical forms) from a ranked list of keyphrases

    A keyphrase is redundant if it is a substring of higher ranked keyphrases.
    """
    non_redundant = []
    for keyphrase, score in ranked_keyphrases:
        if any([keyphrase in other for other, _ in non_redundant]):
            continue
        non_redundant.append((keyphrase, score))
    return non_redundant
