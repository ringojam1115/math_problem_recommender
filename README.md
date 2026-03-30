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

## Future

## Setup

## Usage

