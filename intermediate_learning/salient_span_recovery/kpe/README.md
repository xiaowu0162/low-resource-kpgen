# Package Information

Package ``kpe`` provides a basic solution for developing keyphrase extraction systems.

## Structure

### Base modules

- [base_structures.py](base_structures.py): base data structures, i.e. ``Sentence``, ``Document`` and ``Candidate``
- [base_methods.py](base_methods.py): basic functions to process text data
- [base_model.py](base_model.py): base class for NLP models ``NLPPipeline``
- [base_kpe.py](base_kpe.py): base class for KPE models ``KeyphraseExtractor``
- [filters.py](filters.py): different methods to filter (to remove unnecessary) candidates

### Models

- [document_frequency.py](document_frequency.py): model to compute document frequency
- [tfidf.py](tfidf.py): KPE model based on TF-IDF
- [textrank.py](textrank.py): KPE model based on TextRank

### Challenge-specific modules

- [data.py](data.py): helper methods to process patent data