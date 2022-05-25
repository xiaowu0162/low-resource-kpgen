import math
from collections import defaultdict
from typing import Union, List, Tuple, Set, Dict

import networkx as nx

from kpe.base_kpe import KeyphraseExtractor
from kpe.base_methods import remove_redundant_keyphrases
from kpe.base_structures import Document, Candidate


class TextRank(KeyphraseExtractor):
    """
    Keyphrase extractor based on TextRank

    Parameters (additional, for the rest see :class:`KeyphraseExtractor`)
        ``pos``: Set[str], default to ``{'NOUN', 'PROPN', 'ADJ'}``
            POS tags for candidates
        ``window``: int, default to 2
            window size for words co-occurrence
        ``top``: int or float, default to 0.33
            number or ratio of top vertices to keep
    """

    def __init__(
            self,
            pos: Set[str] = None,
            window: int = 2,
            top: Union[int, float, None] = 0.33,
            language: str = 'en',
            spacy_model=None,
            stemmer=None,
            normalization: str = 'lowercase',
    ):
        super().__init__(
            language=language,
            spacy_model=spacy_model,
            stemmer=stemmer,
            stopwords=False,
            normalization=normalization,
        )

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}
        self.pos = pos
        self.window = window
        self.top = top

    def extract_candidates(
            self,
            document: Document,
            whitelist: Set[str] = None,
    ) -> List[Candidate]:
        """
        Extract candidates as the longest sequences of tokens with POS tags in the allowed list
        """
        candidates = []
        for sentence in document.sentences:
            words = [word.lower() for word in sentence.words] \
                if self.normalization == 'lowercase' \
                else sentence.stems
            tags = sentence.utags

            i = 0
            while i < len(sentence):
                if (whitelist and words[i] not in whitelist) or tags[i] not in self.pos:
                    i += 1
                    continue

                j = i + 1
                while j < len(sentence) and (not whitelist or words[j] in whitelist) and tags[j] in self.pos:
                    j += 1

                candidate = Candidate(sentence=sentence, i=i, j=j)
                candidates.append(candidate)

                i = j
        return candidates

    def build_graph(self, document: Document) -> nx.classes.Graph:
        """
        Build the word graph of the document
        """
        graph = nx.Graph()
        for sentence in document.sentences:
            words = [word.lower() for word in sentence.words] \
                if self.normalization == 'lowercase' \
                else sentence.stems
            tags = sentence.utags
            for i in range(0, len(sentence)):
                for j in range(i + 1, min(i + self.window, len(sentence))):
                    if tags[i] in self.pos and tags[j] in self.pos:
                        graph.add_edge(words[i], words[j])
        return graph

    def score_candidates(self, graph: nx.classes.Graph) -> Dict[str, float]:
        weights = nx.pagerank_scipy(graph, alpha=0.85, tol=0.0001, weight=None)
        if self.top is not None:
            top_weights = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
            limit = self.top if isinstance(self.top, int) else math.ceil(self.top * len(top_weights))
            weights = {k: v for k, v in top_weights[:limit]}
        return weights

    def extract_keyphrases(
            self,
            candidates: List[Candidate],
            scores: Dict[str, float],
            k: int = None,
            redundancy_removal: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Extract keyphrases from a list of candidates based on a scoring

        See :meth:`KeyphraseExtractor.extract_keyphrases`

        Parameters
            scores:
                a scoring mapping of *word vertices*
        """
        if k is None:
            k = len(candidates)

        term_scores = {}
        term2keyphrases = defaultdict(list)

        for candidate in candidates:
            forms = candidate.surface_forms \
                if self.normalization == 'lowercase' \
                else candidate.lexical_forms
            term = ' '.join(forms)
            term_scores[term] = sum([scores.get(form, float('-inf')) for form in forms])
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
