import argparse
import json
import numpy as np

from src.models.bm25_searcher import BM25Searcher
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
        help="Get user query and generate a hypothetical problem with ChatGPT to use as the search query. If not set, use the original user query for search.")

    parser.add_argument("--force_recompute_dataset", action="store_true",
        help="Force re-computation of dataset embeddings even if existing ones are available.")


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
    help="single: Single query search / batch: Multiple queries for evaluation"
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

    # Load problem texts and metadata from dataset directory
    texts, metadata = load_problems_texts_from_dir(dataset_dir)

    bm25_searcher = None
    dataset_vecs = None

    # ===== BM25 =====
    if args.retriever == "bm25":
        bm25_searcher = BM25Searcher(texts)

    # ===== Dense =====
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

        dataset_vecs = np.array(data["embeddings"])

    # ===== Mode-specific =====
    if args.mode == "single":
        run_single_query_mode(
            args=args,
            dataset_vecs=dataset_vecs,
            metadata=metadata,
            pooling=args.pooling,
            bm25_searcher=bm25_searcher,
        )
    elif args.mode == "batch":
        run_batch_evaluation_mode(
            args=args,
            dataset_vecs=dataset_vecs,
            metadata=metadata,
            pooling=args.pooling,
            bm25_searcher=bm25_searcher,
        )

if __name__ == "__main__":
    main()
