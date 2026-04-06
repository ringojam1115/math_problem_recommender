"""
This module contains functions for embedding datasets.

- Load problem JSON files from a directory (one problem per file)
- Embed all problem texts
- Save the embedding results as JSON
- If existing data is found, skip re-computation
- If force_recompute_dataset=True, delete existing data and re-compute

You can choose between different embedding models when vectorizing: default (vanilla BERT), MathBERT+SBERT, or SBERT.
The model choice and the force recompute flag are set from main.py (args.math_bert, args.force_recompute_dataset).  
"""
from __future__ import annotations

import os
import json
from typing import Literal

import numpy as np

from ..models.bert_vanilla import vanilla_bert_embed_texts
from ..models.bert_math_sbert import mathbert_sbert_embed_texts
from ..models.bert_sbert import sbert_embed_texts

def load_problems_texts_from_dir(dataset_dir: str) -> tuple[list[str], list[dict]]:
    """
    Return a list of problem texts by loading all problem JSON files in the specified directory.

    Parameters
    str: dataset_dir
        Path to the directory containing problem JSON files. Each file should contain a single problem with its metadata.
        Example JSON structure for each file:
        {
            "problem": "text of the math problem",
            "level": "easy",  # optional
            "type": "algebra"  # optional
        }

    Returns:
    tuple: (texts, metadata)
        texts: list[str]
            List of problem texts. Same order as metadata, used as the target for embedding.
        metadata: list[dict]
            List of metadata for each problem. Same order as texts, used for display and search results.
             - filename: JSON file name (e.g., "problem_001.json")
             - level: Problem difficulty (e.g., "easy", "medium", "hard")
             - type: Problem type (e.g., "algebra", "geometry")
             - Other fields can be added as needed
    """
    texts: list[str] = []
    metadata: list[dict] = []

    json_files = sorted(
        f for f in os.listdir(dataset_dir) if f.endswith(".json")
    )

    for filename in json_files:
        path = os.path.join(dataset_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)

        texts.append(item["problem"])
        metadata.append({
            "filename": filename,
            "level": item.get("level"),
            "type": item.get("type"),
        })

    return texts, metadata

def embed_dataset(dataset_dir: str, output_path: str, model_type: Literal["vanilla", "sbert", "mathbert_sbert"], pooling: str = "cls", force_recompute_dataset: bool = False) -> None:
    """
    Embed the dataset of problems from the specified directory and save the embeddings to a JSON file.

    Parameters:
        dataset_dir (str): The directory containing problem JSON files to embed. Each file should contain a single problem with its metadata.
        output_path (str): The file path where the resulting embeddings and metadata will be saved as JSON.
        model_type (str): The type of embedding model to use. Options are "vanilla" (default BERT), "mathbert_sbert" (MathBERT + SBERT), or "sbert" (SBERT).
        pooling (str): The pooling strategy to use for vanilla BERT and MathBERT+SBERT. Options are "cls" (use [CLS] token) or "mean" (average of token embeddings).
        force_recompute_dataset (bool): If True, existing embedding file at output_path will be deleted and embeddings will be re-computed. If False, existing file will be used if found, and embedding will be skipped.

    Returns:
        None: The function saves the embeddings and metadata to the specified output_path as JSON.
    """
    # Delete existing embedding file if force_recompute_dataset is True
    if force_recompute_dataset and os.path.exists(output_path):
        print(f"[INFO] Removing existing embeddings: {output_path}")
        os.remove(output_path)

    # Skip embedding if output file already exists
    if os.path.exists(output_path):
        print(f"[INFO] Found existing embeddings. Skipping: {output_path}")
        return

    print("[INFO] Loading dataset...")
    texts, metadata = load_problems_texts_from_dir(dataset_dir)

    print(f"[INFO] Embedding {len(texts)} problems using model={model_type}")

    # Embed the texts using the specified model and pooling strategy
    if model_type == "vanilla":
        embeddings = embed_in_batches(
            texts,
            batch_size=16,
            model="vanilla",
            pooling=pooling,
        )
    elif model_type == "mathbert_sbert":
        embeddings = embed_in_batches(
            texts,
            batch_size=16,
            model="mathbert_sbert",
            pooling=pooling,
        )
    elif model_type == "sbert":
        embeddings = embed_in_batches(
            texts,
            batch_size=16,
            model="sbert",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"[INFO] Embedding shape: {embeddings.shape}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embeddings": embeddings.tolist(),
                "metadata": metadata,
                "model_type": model_type,
                "pooling": pooling,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Saved embeddings to {output_path}")

def embed_in_batches(texts: list[str], batch_size: int, model: Literal["vanilla", "mathbert_sbert", "sbert"],pooling: str = "cls") -> np.ndarray:
    """
    Embed the list of texts in batches using the specified model and pooling strategy.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        if model == "mathbert_sbert":
            emb = mathbert_sbert_embed_texts(batch, pooling=pooling)
        elif model == "sbert":
            emb = sbert_embed_texts(batch)
        elif model == "vanilla":
            emb = vanilla_bert_embed_texts(batch, pooling=pooling)
        else:
            raise ValueError(f"Unknown model: {model}")

        all_embeddings.append(emb)

    return np.vstack(all_embeddings)
