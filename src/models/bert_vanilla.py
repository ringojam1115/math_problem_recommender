import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from config import MODEL_NAME_VANILLA

_model = None
_tokenizer = None

def get_model_and_tokenizer():
    """
    Load and return the BERT model and tokenizer. Uses global variables to cache them after the first load.
    Returns:
        model: The loaded BERT model.
        tokenizer: The loaded BERT tokenizer.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_VANILLA)
        _model = AutoModel.from_pretrained(MODEL_NAME_VANILLA)
        _model.eval()
    return _model, _tokenizer

def vanilla_bert_embed_texts(texts: list[str], pooling: str = "cls",) -> np.ndarray:
    """
    Vanilla BERT を使ってテキスト群をベクトル化する。

    Parameters
    ----------
    texts : list[str]
        ベクトル化したいテキストのリスト。
    pooling : str, optional
        プーリング方法。"cls", "mean", または "max"。デフォルトは "cls"。

    Returns
    -------
    np.ndarray
        ベクトル化結果。shape = (N, D)
    """
    model, tokenizer = get_model_and_tokenizer()

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**encodings)

    last_hidden_states = outputs.last_hidden_state  # shape = (N, L, H)

    print(f"Pooling method: {pooling}")

    if pooling == "cls":
        embeddings = last_hidden_states[:, 0, :]  # shape = (N, H)
    elif pooling == "mean":
        attention_mask = encodings['attention_mask']  # shape = (N, L)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask  # shape = (N, H)
    elif pooling == "max":
        attention_mask = encodings["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        masked_hidden_states = last_hidden_states.masked_fill(input_mask_expanded == 0, torch.finfo(last_hidden_states.dtype).min)
        embeddings, _ = torch.max(masked_hidden_states, dim=1)
    else:
        raise ValueError(f"Invalid pooling method: {pooling}")
    
    # ✅ normalize for ALL pooling types
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()