# A Mathematics Problem Recommendation System Using Proxy Problem Generation with Large Language Models

A semantic search system that retrieves similar math problems 
from natural language queries, built as a graduation research project.

## Overview

When a user inputs a natural language query like  
"problems about finding the minimum value of a quadratic function", "problems where the graph goes down to its lowest point",
the system returns Top-K similar math problems from the MATH dataset

## Tech Stack

- Python / PyTorch 
- HuggingFace Transformers 
- MathBERT / Sentence-BERT 
- BM25 (baseline) 
- OpenAI API (HyDE) 
- NumPy

## Key Features

- **Multi-model comparison**: BM25 / Vanilla BERT / SBERT / MathBERT+SBERT
- **HyDE implementation**: Converts natural language queries into formal math problems via ChatGPT before embedding
- **Evaluation metrics**: Precision@K, Recall@K, nDCG@K

## Results

## Results

### Experiment 1: Embedding Method Comparison (without HyDE)

| Model | Precision@5 (normal) | nDCG@5 (normal) | Precision@5 (gap) | nDCG@5 (gap) |
|-------|---------------------|-----------------|-------------------|--------------|
| Vanilla BERT + CLS | 0.0020 | 0.0013 | 0.0060 | 0.0089 |
| Vanilla BERT + Mean | 0.0040 | 0.0047 | 0.0040 | 0.0029 |
| Vanilla BERT + Max | 0.0180 | 0.0230 | 0.0020 | 0.0034 |
| SBERT | 0.2720 | 0.2938 | 0.0240 | 0.0272 |
| SBERT + MathBERT + CLS | 0.2260 | 0.2404 | 0.0180 | 0.0187 |
| **SBERT + MathBERT + Mean** | **0.2760** | **0.3039** | **0.0280** | **0.0278** |
| SBERT + MathBERT + Max | 0.2620 | 0.2880 | 0.0260 | 0.0290 |

### Experiment 2: Proposed Method vs Baselines (with HyDE)

| Model | Precision@5 (normal) | nDCG@5 (normal) | Precision@5 (gap) | nDCG@5 (gap) |
|-------|---------------------|-----------------|-------------------|--------------|
| BM25 (baseline) | 0.0920 | 0.1023 | 0.0140 | 0.0125 |
| SBERT | 0.2720 | 0.2938 | 0.0240 | 0.0272 |
| **Proposed (SBERT + MathBERT + Mean + HyDE)** | **0.2460** | **0.2781** | **0.1180** | **0.1294** |

### Key Finding

On the Type-gap condition (queries with large semantic gap from problem text),  
the proposed method achieved **4.9x improvement in Precision** and  
**4.8x improvement in nDCG** compared to SBERT alone.

## Future

## Setup

> **Note:** This setup guide is intended for **single query mode** only.
> Batch evaluation mode (for measuring retrieval performance) requires
> additional configuration and is not covered here.

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API key
```bash
export OPENAI_API_KEY="your-api-key"
```

### 4. Download the MATH dataset

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/awsaf49/math-dataset

Place the raw data in the following directory:
```
data/raw/math_original/
```

### 5. Sample problems from the dataset

Run the following script to randomly sample problems from the raw data:
```bash
python scripts/problem_sampler.py data/raw/math_original data/outlet_problem_sampler
```

Available options:

| Option | Default | Description |
|--------|---------|-------------|
| `-n`, `--num_problems` | 100 | Number of problems to sample |
| `--seed` | 42 | Random seed for reproducibility |
| `--dry-run` | - | Preview without copying files |

The sampled data will be saved in:
```
data/outlet_problem_sampler/<run_timestamp>/
```

### 6. Set the dataset path in config

Open `config.py` and update `DATASET_DIR` to point to the sampled data:
```python
DATASET_DIR = "data/outlet_problem_sampler/<run_timestamp>"
```

Example:
```python
DATASET_DIR = "data/outlet_problem_sampler/run_20251211-2328"
```

## Usage

## Usage

Run `main.py` from the project root:
```bash
python main.py [options]
```

### Options

| Option | Choices | Default | Description |
|--------|---------|---------|-------------|
| `--retriever` | `vanilla` `mathbert_sbert` `sbert` `bm25` | `vanilla` | Retrieval method |
| `--pooling` | `cls` `mean` `max` | `cls` | Pooling method for BERT embeddings |
| `--use_chatgpt` | - | `False` | Enable HyDE (generate proxy problem via ChatGPT before search) |
| `--force_recompute_dataset` | - | `False` | Force recompute dataset embeddings even if cache exists |
| `--mode` | `single` `batch` | `single` | `single`: interactive query / `batch`: evaluation mode |

### Examples

**Basic search (Vanilla BERT):**
```bash
python main.py --retriever vanilla --pooling mean
```

**Proposed method (SBERT + MathBERT + HyDE):**
```bash
python main.py --retriever mathbert_sbert --pooling mean --use_chatgpt
```

**After running, enter your query in the prompt:**
```
Enter your query: find a problem about probability distributions
```

**Results are displayed in the terminal:**
```
=== 類似問題候補 ===
[1] score=0.9485  file=555_precalculus.json
[2] score=0.9483  file=870_intermediate_algebra.json
[3] score=0.9470  file=1048_intermediate_algebra.json
[4] score=0.9465  file=2083_algebra.json
[5] score=0.9442  file=1499_algebra.json
```
