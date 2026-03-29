from __future__ import annotations
from typing import Callable, Any
import numpy as np
from similarity_search import search_top_k


def compute_dcg(relevance_list):
    """
    Calculate DCG (Discounted Cumulative Gain) for a list of relevance scores.
    Return:
        float: The DCG score.
    """
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_list))


def compute_ndcg(relevance_list, num_relevant):
    """
    Calculate nDCG (Normalized Discounted Cumulative Gain) for a list of relevance scores.
    Return:
        float: The nDCG score.
    """
    dcg = compute_dcg(relevance_list)

    # 理想的な順位での relevance_list を作成
    ideal_rels = [1] * min(num_relevant, len(relevance_list))
    ideal_rels += [0] * (len(relevance_list) - len(ideal_rels))
    idcg = compute_dcg(ideal_rels)

    return dcg / idcg if idcg > 0 else 0


def evaluate_all_queries(eval_queries, dataset_embs, metadata, embed_query_fn, top_k=5, use_chatgpt=False, bm25_search_fn: Callable[[str, int], list[dict[str, Any]]] | None = None) -> dict[str, float]:
    total_precision = 0
    total_recall = 0
    total_ndcg = 0
    num_queries = len(eval_queries)

    if bm25_search_fn is None:
        if embed_query_fn is None:
            raise ValueError("embed_query_fn is required for dense evaluation.")
        if dataset_embs is None:
            raise ValueError("dataset_embs is required for dense evaluation.")

    for q in eval_queries:
        query_text = q["hypo_query"] if use_chatgpt else q["query"]

        # BM25 を使う場合はそちらで検索し、そうでない場合はベクトル検索を行う --- IGNORE ---
        if bm25_search_fn is not None:
            results = bm25_search_fn(query_text, top_k)
        else:
            query_vec = embed_query_fn([query_text])[0]
            results = search_top_k(
                query_vec=query_vec,
                dataset_embs=dataset_embs,
                metadata=metadata,
                top_k=top_k,
            )

        retrieved_ids = [r["filename"] for r in results]
        relevant_ids = set(q["relevant_problem_ids"])

        # relevance_list は、上位 top_k 件のうち何件が relevant_ids に含まれるかを示す 0/1 のリスト --- IGNORE ---
        relevance_list = [1 if rid in relevant_ids else 0 for rid in retrieved_ids]
        overlap_count = sum(relevance_list)
        num_relevant = len(relevant_ids)

        # Precision と Recall を計算 --- IGNORE ---
        precision = overlap_count / top_k if top_k > 0 else 0
        recall = overlap_count / num_relevant if num_relevant > 0 else 0

        # nDCG を計算 --- IGNORE ---
        ndcg = compute_ndcg(relevance_list, num_relevant)

        total_precision += precision
        total_recall += recall
        total_ndcg += ndcg

    return {
        "Precision@K": total_precision / num_queries if num_queries > 0 else 0,
        "Recall@K": total_recall / num_queries if num_queries > 0 else 0,
        "nDCG@K": total_ndcg / num_queries if num_queries > 0 else 0,
        "num_queries": num_queries,
    }
