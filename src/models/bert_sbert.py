from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME_SBERT

_sbert_model: SentenceTransformer | None = None


def get_sbert_model() -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(MODEL_NAME_SBERT)
    return _sbert_model


def sbert_embed_texts(
    texts: list[str],
    normalize: bool = True,
) -> np.ndarray:
    """
    SBERT (off-the-shelf dense baseline) で texts を埋め込み。
    normalize_embeddings=True にすると cosine=dot が成立して安定。
    """
    model = get_sbert_model()
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return embs