import json
from ..src.recommender.generate_hypo_problem import generate_hypothetical_problem
from config import EVAL_QUERIES_PATH

# 出力先（上書きしたくない場合は別ファイルにする）
OUTPUT_PATH = "data/data_evaluation/data_evaluation_with_hypo"


def main():
    # 1. 評価用クエリ JSON を読み込む
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    updated = 0
    skipped = 0

    # 2. 各クエリを順番に処理
    for q in eval_queries:
        # すでに hypo_query がある場合はスキップ
        if "hypo_query" in q and q["hypo_query"].strip():
            skipped += 1
            continue

        query_id = q.get("query_id", "UNKNOWN")
        print(f"[INFO] Generating hypo_query for query_id={query_id}")

        # 3. ChatGPT で仮問題を生成
        hypo = generate_hypothetical_problem(q["query"])

        # 4. JSON に追記
        q["hypo_query"] = hypo
        updated += 1

    # 5. 結果を書き出す
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, indent=2, ensure_ascii=False)

    # 6. 実行結果サマリ
    print("================================")
    print(f"Updated : {updated}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {OUTPUT_PATH}")
    print("================================")


if __name__ == "__main__":
    main()
