# System_failed1

Advanced RAG Pipeline
🚀 Overview

This project implements an Advanced Hybrid Retrieval-Augmented Generation (RAG) pipeline designed for high-accuracy, low-latency question answering from documents.
It combines dense vector search (FAISS), sparse retrieval (BM25), cross-encoder reranking, semantic caching, and Google Gemini LLM to achieve 90%+ accuracy with <2s response time — optimized for HackRX competition standards.

🔑 Features

Hybrid Retrieval: Combines dense embeddings (OpenAI via GitHub Models) with sparse BM25 search.

FAISS HNSW Indexing: High-performance similarity search with sub-linear complexity.

Cross-encoder Reranking: Improves answer precision.

Google Gemini Integration: Fast and high-quality text generation.

Smart Caching: Semantic similarity-based cache (Redis or in-memory).

Context Compression: Optimized prompts to reduce token usage.

Concurrent Processing: Async pipeline supports multiple queries in parallel.

Explainability: Detailed retrieval traces with confidence scores.

📂 Project Structure
├── hybrid_rag_pipeline.py   # Core RAG pipeline (retrieval, generation, caching, optimizations)
├── main.py                  # FastAPI app exposing REST API endpoints
├── run_pipeline.py           # Benchmark & evaluation runner for performance testing
├── data/                    # Document storage
├── faiss_index/             # Persisted FAISS index
└── .env                     # API keys & config (not included in repo)

⚡ API Endpoints

The FastAPI service (main.py) provides:

POST /hackrx/run → Process PDF and answer multiple questions

GET /stats → Retrieve pipeline statistics

GET /health → Health check

Auto-generated docs at /docs

▶️ Quick Start

Clone repository and install dependencies:

pip install -r requirements.txt


Add a .env file:

GITHUB_TOKEN=your_github_models_token
GEMINI_API_KEY=your_google_gemini_api_key
BEARER_TOKEN=your_auth_token


Run API server:

uvicorn main:app --reload


Benchmark pipeline:

python run_pipeline.py

📊 Performance (HackRX Benchmark)

Target: 90%+ accuracy, <2s latency

Achieved: Significant improvements with hybrid retrieval, caching, and reranking

Cold Start vs Warm Start shows 2–3x speedup with caching.
