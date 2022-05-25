import abc
from string import punctuation
from typing import Set

from kpe.base_structures import Candidate


class AbstractCandidateFilter(abc.ABC):
    """
    :class:`Filter` is the abstract class for candidate filters.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def filter(self, candidate: Candidate) -> bool:
        """
        Decide if the candidate should be filtered.
        If return ``True``, it means that the candidate should be removed.
        """
        pass


class PunctuationFilter(AbstractCandidateFilter):
    """
    :class:`PunctuationFilter` filters candidates that have *any* token made of only punctuations.
    """

    def __init__(self, punctuations: Set[str] = None):
        super().__init__()
        if punctuations is None:
            punctuations = (set(punctuation))
            punctuations.remove('-')  # allow hyphenation
        self.punctuations = punctuations

    def filter(self, candidate: Candidate) -> bool:
        for token in candidate.surface_forms:
            if all([ch in self.punctuations for ch in token]):
                return True
        return False


class EmptyTokenFilter(AbstractCandidateFilter):
    """
    :class:`EmptyTokenFilter` filters candidates that have *at least* one empty string token.
    """

    def __init__(self):
        super().__init__()

    def filter(self, candidate: Candidate) -> bool:
        for token in candidate.surface_forms:
            if token.strip() == '':
                return True
        return False


class TokenLengthFilter(AbstractCandidateFilter):
    """
    :class:`TokenLengthFilter` filters candidates that have *any* token with its length not in a predefined range.
    """

    def __init__(self, min_length: int = 0, max_length: int = float('inf')):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length

    def filter(self, candidate: Candidate) -> bool:
        if any([not (word == '-' or self.min_length < len(word) < self.max_length)  # allow hyphenation
                for word in candidate.surface_forms]):
            return True
        return False


class MinimumLengthFilter(AbstractCandidateFilter):
    """
    :class:`MinimumLengthFilter` filters candidates with their lengths below a threshold.
    """

    def __init__(self, length: int = 2):
        super().__init__()
        self.length = length

    def filter(self, candidate: Candidate) -> bool:
        if len(''.join(candidate.surface_forms)) < self.length:
            return True
        return False


class StopwordFilter(AbstractCandidateFilter):
    """
    :class:`StopwordFilter` filters candidates that contains a stopword.
    If the stopword list is ``None``, filter by the attribute :attr:`Candidate.stopwords` of the candidate.
    """

    def __init__(self, stopwords: Set[str] = None):
        super().__init__()
        self.stopwords = stopwords

    def filter(self, candidate: Candidate) -> bool:
        if self.stopwords is not None:
            if any([word in self.stopwords for word in candidate.surface_forms]):
                return True
        else:
            if any([is_stopword for is_stopword in candidate.stopwords()]):
                return True
        return False


class UTagBlacklistFilter(AbstractCandidateFilter):
    """
    :class:`UTagBlacklistFilter` filters candidates that have *any* of their token utags in a blacklist.
    """

    def __init__(self, blacklist: Set[str]):
        super().__init__()
        self.blacklist = blacklist

    def filter(self, candidate: Candidate) -> bool:
        if any([tag in self.blacklist for tag in candidate.utags()]):
            return True
        return False


class XTagBlacklistFilter(AbstractCandidateFilter):
    """
    :class:`XTagBlacklistFilter` filters candidates that have *any* of their token xtags in a blacklist.
    """

    def __init__(self, blacklist: Set[str]):
        super().__init__()
        self.blacklist = blacklist

    def filter(self, candidate: Candidate) -> bool:
        if any([tag in self.blacklist for tag in candidate.xtags()]):
            return True
        return False


class UTagWhitelistFilter(AbstractCandidateFilter):
    """
    :class:`UTagWhitelistFilter` keeps *only* candidates that have *all* of their token utags in a whitelist.
    """

    def __init__(self, whitelist: Set[str]):
        super().__init__()
        self.whitelist = whitelist

    def filter(self, candidate: Candidate) -> bool:
        if all([tag in self.whitelist for tag in candidate.utags()]):
            return False
        return True


class XTagWhitelistFilter(AbstractCandidateFilter):
    """
    :class:`XTagWhitelistFilter` keeps *only* candidates that have *all* of their token xtags in a whitelist.
    """

    def __init__(self, whitelist: Set[str]):
        super().__init__()
        self.whitelist = whitelist

    def filter(self, candidate: Candidate) -> bool:
        if all([tag in self.whitelist for tag in candidate.xtags()]):
            return False
        return True
