import math
from collections import Counter
from typing import Union, List, Set, Dict

from kpe.base_kpe import KeyphraseExtractor
from kpe.base_structures import Document, Candidate


class TfIdf(KeyphraseExtractor):
    """
    Keyphrase extractor based on Tf-Idf

    Parameters (additional, for the rest see :class:`KeyphraseExtractor`)
        ``document_frequency``: Dict[str, int]
            document frequency of terms
        ``num_documents``: int
            total number of documents
        ``n``: int, default to 3
            maximum length of terms (in number of tokens)
    """

    def __init__(
            self,
            document_frequency: Dict[str, int],
            num_documents: int,
            n: int = 3,
            language: str = 'en',
            spacy_model=None,
            stemmer=None,
            stopwords: Union[bool, Set[str]] = True,
            normalization: str = 'stemming',
    ):
        super().__init__(
            language=language,
            spacy_model=spacy_model,
            stemmer=stemmer,
            stopwords=stopwords,
            normalization=normalization,
        )
        self.doc_freq = document_frequency
        self.num_docs = num_documents
        self.n = n

    def extract_candidates(
            self,
            document: Document,
            n: int = None,
    ) -> List[Candidate]:
        return super().extract_candidates(document, n=self.n)

    def score_candidates(self, candidates: List[Candidate]) -> Dict[str, float]:
        term_freq = Counter()
        tfidf = {}
        num_docs = self.num_docs + 1
        for candidate in candidates:
            term = getattr(candidate, self.form)()
            term_freq[term] += 1
        for candidate in candidates:
            term = getattr(candidate, self.form)()
            tf = term_freq[term]
            idf = math.log2(num_docs / (self.doc_freq[term] + 1))
            tfidf[term] = tf * idf
        return tfidf
