import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME_SBERT, MODEL_NAME_MATHBERT

_mathbert_model = None
_mathbert_tokenizer = None
_sbert_model = None


def get_models():
    global _mathbert_model, _mathbert_tokenizer, _sbert_model

    if _mathbert_model is None or _mathbert_tokenizer is None:
        _mathbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MATHBERT)
        _mathbert_model = AutoModel.from_pretrained(MODEL_NAME_MATHBERT)
        _mathbert_model.eval()

    if _sbert_model is None:
        _sbert_model = SentenceTransformer(MODEL_NAME_SBERT)

    return _mathbert_model, _mathbert_tokenizer, _sbert_model


def mathbert_sbert_embed_texts(
    texts: list[str],
    pooling: str = "cls",  # "cls", "mean", or "max"
) -> np.ndarray:
    """
    MathBERT + Sentence-BERT を併用してテキスト群をベクトル化する。

    - MathBERT: 数式・数学表現の局所的意味
    - SBERT   : 文全体の意味

    Returns
    -------
    np.ndarray
        shape = (N, D_mathbert + D_sbert)
    """
    mathbert_model, mathbert_tokenizer, sbert_model = get_models()

    # --- MathBERT embedding ---
    encodings = mathbert_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )

    with torch.no_grad():
        outputs = mathbert_model(**encodings)

    last_hidden_states = outputs.last_hidden_state  # (N, L, H)

    print(f"Pooling method: {pooling}")

    if pooling == "cls":
        mathbert_emb = last_hidden_states[:, 0, :]
    elif pooling == "mean":
        attention_mask = encodings["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mathbert_emb = sum_embeddings / sum_mask
    elif pooling == "max":
        attention_mask = encodings["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        masked_hidden_states = last_hidden_states.masked_fill(input_mask_expanded == 0, torch.finfo(last_hidden_states.dtype).min)
        mathbert_emb, _ = torch.max(masked_hidden_states, dim=1)
    else:
        raise ValueError(f"Invalid pooling method: {pooling}")

    # ✅ normalize for ALL pooling types
    mathbert_emb = torch.nn.functional.normalize(mathbert_emb, p=2, dim=1)

    mathbert_emb = mathbert_emb.cpu().numpy()

    # --- Sentence-BERT embedding ---
    sbert_emb = sbert_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --- Concatenate ---
    embeddings = np.concatenate([mathbert_emb, sbert_emb], axis=1)

    return embeddings