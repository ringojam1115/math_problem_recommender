from __future__ import annotations

from typing import Any
import re
import numpy as np
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> list[str]:
    """
    A simple tokenizer that lowercases the text and splits on non-alphanumeric characters.

    Parameters:
    str: text
            The input text to tokenize.
    Returns:
        list[str]
            A list of tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


class BM25Searcher:
    def __init__(self, docs: list[str]) -> None:
        """
        Initialize the BM25 searcher with a list of documents.
        
        Parameters:
        list[str]: docs
            A list of documents (problems) to index for BM25 search.
        """
        tokenized_docs = [simple_tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search_top_k_text(self, query_text: str, metadata: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search for the top-k most relevant documents based on the input query text.
        Parameters:
            str: query_text
                The user's query as a string.
            list[dict[str, Any]]: metadata
                A list of metadata dictionaries corresponding to the indexed documents. 
                The length of this list should match the number of documents used to initialize the BM25 searcher.
            int: top_k
                The number of top results to return. Default is 5.
        Returns:
            list[dict[str, Any]]
                A list of metadata dictionaries for the top-k most relevant documents, each with an additional "score" key representing the BM25 relevance score.
        """
        q_tokens = simple_tokenize(query_text)
        scores = np.array(self.bm25.get_scores(q_tokens))  # (N,)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            m = metadata[idx].copy()
            m["score"] = float(scores[idx])
            results.append(m)
        return results