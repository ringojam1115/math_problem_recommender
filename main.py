import argparse
import json
import numpy as np

from src.models.bm25_seacher import BM25Searcher
from src.dataset_embedding.embedding_dataset import embed_dataset, load_problems_texts_from_dir
from src.run import run_batch_evaluation_mode, run_single_query_mode
from config import (
    EMB_PATH_SBERT,
    EMB_PATH_VANILLA,
    EMB_PATH_MATHBERT_SBERT,
    DATASET_DIR,
)

# Parse command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Math problem semantic search with BERT embeddings."
    )
    parser.add_argument("--use_chatgpt", action="store_true",
        help="検索文から ChatGPT で仮問題を生成する。")

    parser.add_argument("--force_recompute_dataset", action="store_true",
        help="既存の埋め込みがあっても再度ベクトル化を行う。")


    parser.add_argument(
        "--retriever",
        choices=["vanilla", "mathbert_sbert", "sbert", "bm25"],
        default="vanilla",
        help="retriever method"
    )

    parser.add_argument(
    "--mode",
    choices=["single", "batch"],
    default="single",
    help="single: 単問検索 / batch: 評価用複数クエリ"
    )
    
    parser.add_argument(
    "--pooling",
    choices=["cls", "mean", "max"],
    default="cls",
    help="pooling method: cls (CLS token), mean (mean pooling), or max (max pooling)"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = DATASET_DIR

    # ✅ まずテキストとmetadataを読む（BM25にも必要）
    texts, metadata = load_problems_texts_from_dir(dataset_dir)

    bm25_searcher = None
    dataset_embs = None

    # ===== BM25の場合 =====
    if args.retriever == "bm25":
        bm25_searcher = BM25Searcher(texts)

    # ===== Denseの場合 =====
    else:
        if args.retriever == "mathbert_sbert":
            emb_path = EMB_PATH_MATHBERT_SBERT
            model_type = "mathbert_sbert"
        elif args.retriever == "sbert":
            emb_path = EMB_PATH_SBERT
            model_type = "sbert"
        elif args.retriever == "vanilla":
            emb_path = EMB_PATH_VANILLA
            model_type = "vanilla"
        else:
            raise ValueError(f"Unknown retriever: {args.retriever}")

        embed_dataset(
            dataset_dir=dataset_dir,
            output_path=emb_path,
            model_type=model_type,
            pooling=args.pooling,
            force_recompute_dataset=args.force_recompute_dataset,
        )

        with open(emb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_embs = np.array(data["embeddings"])
        # metadataは load_problems_texts_from_dir のものを使う
        # （embeddingファイル内のmetadataと一致する想定）
        # metadata = data["metadata"]  ← どちらでもOK

    # ===== モード別 =====
    if args.mode == "single":
        run_single_query_mode(
            args=args,
            dataset_embs=dataset_embs,
            metadata=metadata,
            pooling=args.pooling,
            bm25_searcher=bm25_searcher,
        )
    elif args.mode == "batch":
        run_batch_evaluation_mode(
            args=args,
            dataset_embs=dataset_embs,
            metadata=metadata,
            pooling=args.pooling,
            bm25_searcher=bm25_searcher,
        )

if __name__ == "__main__":
    main()
