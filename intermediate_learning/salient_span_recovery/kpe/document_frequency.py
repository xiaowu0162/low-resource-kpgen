import csv
from collections import Counter
from typing import Union, List, Set, Dict

from . import filters
from .base_kpe import KeyphraseExtractor
from .base_structures import Candidate


class DocumentFrequency(KeyphraseExtractor):
    """
    :class:`DocumentFrequency` stores and updates document frequencies.

    Parameters (additional, for the rest see :class:`KeyphraseExtractor`)
        ``n``: int, default to 3
            maximum length of terms (in number of tokens)

        ``stopwords``: bool or Set[str], default to ``False`` (not used)
    """

    def __init__(
            self,
            language: str = 'en',
            n: int = 3,
            spacy_model=None,
            stemmer=None,
            stopwords: Union[bool, Set[str]] = False,
            normalization: str = 'stemming',
    ):
        super().__init__(
            language=language,
            spacy_model=spacy_model,
            stemmer=stemmer,
            stopwords=stopwords,
            normalization=normalization,
        )
        self.n = n

        self.candidate_filters = [
            filters.EmptyTokenFilter(),
            filters.PunctuationFilter(),
            filters.StopwordFilter(self.stopwords),
        ]

        self.counter = Counter()
        self.num_docs = 0

    def process(self, text: Union[str, List[str]]):
        document = self.read_text(text)
        candidates = self.extract_candidates(document, n=self.n)
        for candidate in candidates:
            form = getattr(candidate, self.form)()
            self.counter[form] += 1

        self.num_docs += 1

    def __getitem__(self, term):
        return self.counter[term]

    def __len__(self):
        return len(self.counter)

    def __repr__(self):
        return 'DocumentFrequency(num_docs={}, counter={})'.format(self.num_docs, self.counter)

    def to_tsv(self, path):
        with open(path, 'w') as f:
            f.write('{}\n'.format(self.num_docs))
            for k, v in self.counter.items():
                f.write('{}\t{}\n'.format(k, v))

    def from_tsv(self, path):
        num_docs, counter = DocumentFrequency.read_tsv(path)

        self.counter = counter
        self.num_docs = num_docs

    @classmethod
    def read_tsv(cls, path):
        counter = Counter()
        with open(path, 'r') as f:
            csv_file = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            num_docs = int(next(csv_file)[0])
            for row in csv_file:
                counter[row[0]] = int(row[1])
        return num_docs, counter

    def score_candidates(self, candidates: List[Candidate]) -> Dict[str, float]:
        raise NotImplementedError
