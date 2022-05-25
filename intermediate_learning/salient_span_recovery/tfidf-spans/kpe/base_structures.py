__all__ = [
    'Sentence',
    'Document',
    'Candidate',
]


class Sentence(object):
    """
    Base class for a processed sentence
    """

    def __init__(self):
        self.words = None
        self.lemmas = None
        self.utags = None
        self.xtags = None
        self.stems = None
        self.stopwords = None

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        sentence = Sentence()
        sentence.words = self.words[item]
        sentence.lemmas = self.lemmas[item]
        sentence.utags = self.utags[item]
        sentence.xtags = self.xtags[item]
        sentence.stems = self.stems[item]
        sentence.stopwords = self.stopwords[item]
        return sentence

    def __repr__(self):
        return ' '.join(self.words)


class Document(object):
    """
    Base class for a document, which is a collection of sentences
    """

    def __init__(self):
        self.sentences = []
        self.language = None

    def __repr__(self):
        return '\n'.join([' '.join(sentence.words) for sentence in self.sentences])


class Candidate(object):
    """
    Base class for a keyphrase candidate, which is a *n*-gram of a :class:`Sentence`
    """

    def __init__(
            self,
            sentence: Sentence,
            i: int,
            j: int
    ):
        self.sentence = sentence
        self.i = i
        self.j = j

        self.surface_forms = [token.lower() for token in self.words()]
        self.lexical_forms = [token.lower() for token in self.stems()]

    def __len__(self):
        return self.j - self.i

    def __repr__(self):
        return self.surface_repr()

    def surface_repr(self):
        return ' '.join(self.surface_forms)

    def lexical_repr(self):
        return ' '.join(self.lexical_forms)

    def words(self):
        return self.sentence.words[self.i:self.j]

    def lemmas(self):
        return self.sentence.lemmas[self.i:self.j]

    def utags(self):
        return self.sentence.utags[self.i:self.j]

    def xtags(self):
        return self.sentence.xtags[self.i:self.j]

    def stems(self):
        return self.sentence.stems[self.i:self.j]

    def stopwords(self):
        return self.sentence.stopwords[self.i:self.j]
