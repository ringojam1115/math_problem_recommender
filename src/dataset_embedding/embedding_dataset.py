"""
This module contains functions for embedding datasets.
問題データセットをベクトル化する関数群を含みます。

問題データセットを読み込む
↓
全ての問題文をベクトル化する
↓
ベクトル化されたデータセットを保存する(JSON形式)

- ディレクトリ内の問題 JSON(1問1ファイル)を読み込む
- 全問題文をベクトル化する
- ベクトル化結果を JSON として保存する
- 既存データがあれば再計算をスキップ
- force_recompute_dataset=True の場合は既存データを削除して再計算

ベクトル化する時のモデルは、defaultからMathBERT+SBERTかを選べます。
モデルと再計算フラグの選択は main.py から行います。(args.math_bert, args.force_recompute_dataset)
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
    指定ディレクトリ内の全問題 JSON を読み込み、問題文リストを返す。

    Parameters
    str: dataset_dir
         問題 JSON ファイルが格納されているディレクトリのパ

    Returns:
    tuple: (texts, metadata)
        texts: list[str]
            問題文のリスト。metadata と同順で、embedding の対象となる。
        metadata: list[dict]
            各問題のメタデータのリスト。texts と同順で、embedding の対象ではないが検索結果表示などで使う。
             - filename: JSONファイル名(例: "problem_001.json")
             - level: 問題の難易度（例: "easy", "medium", "hard")
             - type: 問題の種類（例: "algebra", "geometry")
             - その他、必要に応じてフィールドを追加可能
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
    問題データセットをベクトル化して JSON として保存する。
    """

    # 強制再計算なら既存ファイルを削除
    if force_recompute_dataset and os.path.exists(output_path):
        print(f"[INFO] Removing existing embeddings: {output_path}")
        os.remove(output_path)

    # 既に存在していればスキップ
    if os.path.exists(output_path):
        print(f"[INFO] Found existing embeddings. Skipping: {output_path}")
        return

    print("[INFO] Loading dataset...")
    texts, metadata = load_problems_texts_from_dir(dataset_dir)

    print(f"[INFO] Embedding {len(texts)} problems using model={model_type}")

    # モデル選択
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

    # 保存
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
