# BM25 based Retriever

This tool is borrowed from https://github.com/AkariAsai/XORQA/tree/main/baselines/bm25.

## Elastic Search Insolation

To run ElasticSearch in your local environment, you need to install ElasticSearch first. We install ES by running 
the scripts provided by [CLIReval](https://github.com/ssun32/CLIReval) library (Sun et al., ACL demo 2020).

```
git clone https://github.com/ssun32/CLIReval.git
cd CLIReval
pip install -r requirements.txt
bash scripts/install_external_tools.sh
```

Whenever you run ES in your local environment, you need to start an ES instance.

```
bash scripts/server.sh [start|stop]
```

## Index documents and search

Steps involved are:

- Create db from preprocessed data.
- Index the preprocessed documents.
- Search documents based on BM25 scores.
