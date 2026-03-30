from __future__ import annotations

from typing import Any
import numpy as np


def _cosine_similarities(query_vec: np.ndarray, dataset_vecs: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarities between a single query vector and a set of dataset vectors.

    Parameters:
        np.ndarray: query_vec
            A single vector representing the user's query. Shape should be (d,).
        np.ndarray: dataset_vecs
            A 2D array where each raw is a vector representing a problem in the dataset. Shape should be (n, d).

    Returns:
        np.ndarray
            An array of cosine similarity scores between the query vector and each dataset vector. Shape is (n,)
    """

    # Check input shapes and dimensions
    if query_vec.ndim != 1:
        raise ValueError(f"query_vec must be 1D, got shape={query_vec.shape}")
    if dataset_vecs.ndim != 2:
        raise ValueError(f"dataset_vecs must be 2D, got shape={dataset_vecs.shape}")
    if query_vec.shape[0] != dataset_vecs.shape[1]:
        raise ValueError(
            f"Dim mismatch: query_vec has d={query_vec.shape[0]}, "
            f"but dataset_vecs has d={dataset_vecs.shape[1]}"
        )

    # normalize the query vector to unit length
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        raise ValueError("query_vec has zero norm.")
    q_unit = query_vec / q_norm # shape = (d,)

    # normalize the dataset vectors to unit length
    d_norms = np.linalg.norm(dataset_vecs, axis=1)
    d_norms_safe = np.where(d_norms == 0, 1.0, d_norms)
    d_unit = dataset_vecs / d_norms_safe[:, None]  # shape = (n, d)

    # cosine similarity is the dot product of unit vectors
    # (n, d) @ (d,) → (n,)
    sims = d_unit @ q_unit
    return sims


def search_top_k(query_vec: np.ndarray, dataset_vecs: np.ndarray, metadata: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """
    english

    Return the top-k most similar items from the dataset based on cosine similarity between the query vector and dataset vectors.

    Parameters:
        np.ndarray: query_vec
            A single vector representing the user's query. Shape should be (d,).
        np.ndarray: dataset_vecs
            A 2D array where each raw is a vector representing a problem in the dataset. Shape should be (n, d).
        list[dict[str, Any]]: metadata
            A list of metadata dictionaries corresponding to each problem in the dataset. Ex: path, level, type etc.
        int: top_k
            The number of top similar items to return. Default is 5.

    Returns:
        list[dict[str, Any]]
            A list of metadata dictionaries for the top-k most similar problems, each with an additional " score" key representing the cosine similarity score.
    """
    if dataset_vecs.shape[0] != len(metadata):
        raise ValueError(
            f"dataset_vecs has N={dataset_vecs.shape[0]} but "
            f"metadata has len={len(metadata)}."
        )

    # Compute cosine similarities between the query vector and all dataset vectors
    sims = _cosine_similarities(query_vec, dataset_vecs)

    # Get the indices of the top-k most similar items
    top_indices = np.argsort(sims)[::-1][:top_k]

    # Prepare the results with metadata and similarity scores
    results: list[dict[str, Any]] = []
    for idx in top_indices:
        raw = metadata[idx]
        if isinstance(raw, dict):
            m = raw.copy()
        else:
            m = {"text": str(raw)}
        m["score"] = float(sims[idx])
        results.append(m)

    return results

