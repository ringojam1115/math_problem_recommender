import json
from ..src.recommender.generate_hypo_problem import generate_hypothetical_problem
from config import EVAL_QUERIES_PATH

# output path for the updated JSON with hypo_query
OUTPUT_PATH = "data/data_evaluation/data_evaluation_with_hypo/eval_queries_with_hypo.json"


def main():
    # 1. Load the evaluation queries from the JSON file
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    updated = 0
    skipped = 0

    # 2. Load each query and process it
    for q in eval_queries:
        # Skip if "hypo_query" already exists and is not empty`
        if "hypo_query" in q and q["hypo_query"].strip():
            skipped += 1
            continue

        query_id = q.get("query_id", "UNKNOWN")
        print(f"[INFO] Generating hypo_query for query_id={query_id}")

        # 3. Generate the hypothetical problem using the existing query
        hypo = generate_hypothetical_problem(q["query"])

        # 4. Add the generated hypo_query to the query dictionary
        q["hypo_query"] = hypo
        updated += 1

    # 5. Save the updated evaluation queries back to a new JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, indent=2, ensure_ascii=False)

    # 6. Print execution summary
    print("================================")
    print(f"Updated : {updated}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {OUTPUT_PATH}")
    print("================================")


if __name__ == "__main__":
    main()
