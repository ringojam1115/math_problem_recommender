import json

from .recommender.query_cli import get_query
from .recommender.generate_hypo_problem import generate_hypothetical_problem
from .recommender.similarity_search import search_top_k
from .recommender.evaluate_queries import evaluate_all_queries

from .models.bert_math_sbert import mathbert_sbert_embed_texts
from .models.bert_vanilla import vanilla_bert_embed_texts
from .models.bert_sbert import sbert_embed_texts

from config import (
    TOP_K, EVAL_QUERIES_PATH_HYPO,
)

# === 単問検索モードの実行 ===
def run_single_query_mode(args, dataset_embs, metadata, pooling: str = "cls", bm25_searcher=None):
    # === 1. ユーザーからの検索文取得 ===
    user_text = get_query()

    # === 2. ChatGPTで仮問題を作るかどうか ===
    if args.use_chatgpt:
        print("\n[ChatGPT] 仮問題を生成中...\n")
        hypo = generate_hypothetical_problem(user_text)
        print("=== 仮問題(ChatGPT生成) ===")
        print(hypo)
        print("================================\n")
        query_text = hypo
    else:
        query_text = user_text

    # ===== BM25 =====
    if args.retriever == "bm25":
        if bm25_searcher is None:
            raise ValueError("bm25_searcher is required when retriever=bm25")

        results = bm25_searcher.search_top_k_text(
            query_text=query_text,
            metadata=metadata,
            top_k=TOP_K,
        )

    # ===== Dense =====
    else:
        if args.retriever == "mathbert_sbert":
            q_vec = mathbert_sbert_embed_texts([query_text], pooling=pooling)[0]
        elif args.retriever == "sbert":
            q_vec = sbert_embed_texts([query_text])[0]
        elif args.retriever == "vanilla":
            q_vec = vanilla_bert_embed_texts([query_text], pooling=pooling)[0]
        else:
            raise ValueError(f"Unknown retriever: {args.retriever}")

        results = search_top_k(
            query_vec=q_vec,
            dataset_embs=dataset_embs,
            metadata=metadata,
            top_k=TOP_K
        )

    # === 5. 結果表示 ===
    print("\n=== 類似問題候補 ===")
    for i, r in enumerate(results, start=1):
        score = r["score"]
        print(f"[{i}] score={score:.4f}  file={r['filename']}")


# === 複数クエリ一括評価モードの実行 ===
def run_batch_evaluation_mode(args, dataset_embs, metadata, pooling: str = "cls", bm25_searcher=None):
    # 1. 評価用クエリ JSON を読み込む
    with open(EVAL_QUERIES_PATH_HYPO, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    # ===== BM25 =====
    if args.retriever == "bm25":
        if bm25_searcher is None:
            raise ValueError("bm25_searcher is required when retriever=bm25")

        bm25_search_fn = lambda query_text, k: bm25_searcher.search_top_k_text(
            query_text=query_text,
            metadata=metadata,
            top_k=k,
        )

        results = evaluate_all_queries(
            eval_queries=eval_queries,
            dataset_embs=None,
            metadata=metadata,
            embed_query_fn=None,
            top_k=TOP_K,
            use_chatgpt=args.use_chatgpt,
            bm25_search_fn=bm25_search_fn,
        )

    # ===== Dense =====
    else:
        if args.retriever == "mathbert_sbert":
            embed_query_fn = lambda qs: mathbert_sbert_embed_texts(qs, pooling=pooling)
        elif args.retriever == "sbert":
            embed_query_fn = lambda qs: sbert_embed_texts(qs)
        elif args.retriever == "vanilla":
            embed_query_fn = lambda qs: vanilla_bert_embed_texts(qs, pooling=pooling)
        else:
            raise ValueError(f"Unknown retriever: {args.retriever}")

        results = evaluate_all_queries(
            eval_queries=eval_queries,
            dataset_embs=dataset_embs,
            metadata=metadata,
            embed_query_fn=embed_query_fn,
            top_k=TOP_K,
            use_chatgpt=args.use_chatgpt,
            bm25_search_fn=None,
        )

    print("================================")
    print("=== Evaluation Results ===")
    print("================================")
    print(f"Number of queries: {results['num_queries']}")

    print(f"Precision@K : {results['Precision@K']:.4f}")
    print(f"Recall@K    : {results['Recall@K']:.4f}")
    print(f"nDCG@K      : {results['nDCG@K']:.4f}")

    print("================================")