import abc
from collections import defaultdict
from typing import Union, List, Tuple, Set, Dict

from . import filters
from .base_methods import generate_candidates, filter_candidates, remove_redundant_keyphrases
from .base_model import NLPPipeline
from .base_structures import Document, Candidate

_forms_map = {
    'stemming': 'lexical_repr',
    'lowercase': 'surface_repr',
}


class KeyphraseExtractor(NLPPipeline):
    """
    Base class for keyphrase extractors

    Parameters
        ``language``: str, default to *en*
            language to process documents (ISO 639-1)
        ``spacy_model``: default to ``None``
            spaCy model to process documents,
            if ``None`` use the default spaCy model of the language return by :attr:`default_spacy_model`
        ``stemmer``: default to ``None``
            stemmer to process documents,
            if ``None`` use the default NLTK stemmer of the language return by :func:`default_stemmer`
        ``stopwords``: bool or Set[str], default to ``True``
            list of stopwords used to filter candidates.
            If ``False``, not using stopwords to filter candidates.
            If ``True``, use the stopword indicator provided by the spaCy model.
            If it is set to a set of strings, use the set as the list of stopwords.
        ``normalization``: str, *stemming* or *lowercase*, default to  *stemming*
            types of normalization applied to candidates
    """

    def __init__(
            self,
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
        )

        if isinstance(stopwords, bool):
            if stopwords:
                stopwords = None
            else:
                stopwords = set()
        self.stopwords = stopwords

        self.candidate_filters = [
            filters.EmptyTokenFilter(),
            filters.PunctuationFilter(),
            filters.TokenLengthFilter(min_length=1),
            filters.StopwordFilter(stopwords),
        ]

        assert normalization in ['stemming', 'lowercase']
        self.normalization = normalization
        self.form = _forms_map[self.normalization]

    def extract_candidates(
            self,
            document: Document,
            n: int,
    ) -> List[Candidate]:
        """
        Extract (generate and filter) candidates from a document with the maximum length of ``n``.
        """
        candidates = generate_candidates(document, n=n)
        candidates = filter_candidates(candidates, filters=self.candidate_filters)
        return candidates

    @abc.abstractmethod
    def score_candidates(self, candidates: List[Candidate]) -> Dict[str, float]:
        pass

    def extract_keyphrases(
            self,
            candidates: List[Candidate],
            scores: Dict[str, float],
            k: int = None,
            redundancy_removal: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Extract keyphrases from a list of candidates based on a scoring

        Parameters
            candidates:
                list of candidates
            scores:
                a scoring mapping based on the normalized form of candidates
            k: int, default to ``None``
                the number of keyphrases to extract,
                if ``None`` extract all keyphrases
            redundancy_removal: bool, default to ``False``
                remove *redundant* keyphrases (slow).
                A keyphrase is redundant if it is a substring of higher ranked keyphrases.
        """
        if k is None:
            k = len(candidates)

        term_scores = {}
        term2keyphrases = defaultdict(list)

        for candidate in candidates:
            term = getattr(candidate, self.form)()
            term_scores[term] = scores[term]
            term2keyphrases[term].append(candidate.surface_repr())

        ranked_terms = sorted(term_scores.items(),
                              key=lambda kv: (kv[1], len(kv[0])),  # rank longer term higher
                              reverse=True)
        if redundancy_removal:
            ranked_terms = remove_redundant_keyphrases(ranked_terms)

        results = []
        for term, score in ranked_terms:
            results.append((term2keyphrases[term][0], score))
            if len(results) == k:
                break
        return results
