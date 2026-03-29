MODEL_NAME_VANILLA = "bert-base-uncased"
MODEL_NAME_MATHBERT = "tbs17/MathBERT"
MODEL_NAME_SBERT = "sentence-transformers/all-MiniLM-L6-v2"

# 精度を評価するためのクエリと正解問題のセットパス
EVAL_QUERIES_PATH_HYPO = "data/data_evaluation/data_evaluation_with_hypo/sample_queries_with_hypo.json"
EVAL_QUERIES_PATH = "data/data_evaluation/data_evaluation/sample_queries.json"

# 問題セット json の場所（必要に応じて変えてください）
DATASET_GLOB = "dataset_maker/outlet_problems_100/run_20251211-2328/**/*.json"

DATASET_DIR = "dataset_maker/outlet_problems_100/run_20251211-2328"

# 類似問題を何問返すか
TOP_K = 5

# 埋め込み保存先
EMB_PATH_VANILLA = "artifacts/dataset_vanilla_embeddings.json"
EMB_PATH_MATHBERT_SBERT = "artifacts/dataset_mathbert_sbert_embeddings.json"
EMB_PATH_SBERT = "artifacts/dataset_sbert_embeddings.json"

# Name of the model to use for ChatGPT API.
CHATGPT_MODEL_NAME = "gpt-5-mini-2025-08-07" 

# Name of the environment variable that holds the ChatGPT API key.
CHATGPT_API_ENV_NAME = "OPENAI_API_KEY"

# The prompt template for generating hypothetical problems using ChatGPT.
# MATH_DATASETのProblemを参考にして作成
CHATGPT_HYPO_PROMPT = (
    "You are a helpful math teacher.\n"
    "Based on the following student's description, write ONE math problem "
    "that matches the student's intent.\n"
    "Use LaTeX for formulas if appropriate.\n\n"
    "Write the problem as a self-contained, formal mathematics question in natural English. "
    "Start directly with the problem statement (e.g., \"Find\", \"Determine\", \"What is\"). "
    "Do not include any introduction, title, numbering, or commentary. "
    "State all necessary conditions within the sentence(s) of the problem itself.\n\n"
    "Student's description:\n{user_query}\n\n"
    "Output only the problem statement. Do not include solution or explanation."
)
