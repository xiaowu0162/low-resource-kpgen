from typing import Union, List

from .base_methods import default_spacy_model, default_stemmer, process_text
from .base_structures import Document


class NLPPipeline(object):
    """
    Base class for NLP pipelines

    Parameters
        ``language``: str, default to *en*
            language to process documents (ISO 639-1)
        ``spacy_model``: default to ``None``
            spaCy model to process documents,
            if ``None`` use the default spaCy model of the language return by :attr:`default_spacy_model`
        ``stemmer``: default to ``None``
            stemmer to process documents,
            if ``None`` use the default NLTK stemmer of the language return by :attr:`default_stemmer`
    """

    def __init__(
            self,
            language: str = 'en',
            spacy_model=None,
            stemmer=None,
    ):
        self.language = language
        if spacy_model is None:
            spacy_model = default_spacy_model(language)
        self.spacy_model = spacy_model
        if stemmer is None:
            stemmer = default_stemmer(language)
        self.stemmer = stemmer

    def read_text(self, text: Union[str, List[str]]) -> Document:
        """
        Read a string or a list of strings as a document.

        Parameter
            ``text``: str or a list of str
                input text

        Return
            ``document``: :class:`Document``
        """
        document = process_text(
            text,
            language=self.language,
            spacy_model=self.spacy_model,
            stemmer=self.stemmer
        )
        return document
