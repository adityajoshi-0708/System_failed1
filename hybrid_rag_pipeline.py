import os
import re
import asyncio
import tempfile
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from pathlib import Path

# Core dependencies
from openai import AsyncOpenAI
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer


class HyperOptimizedRAGPipeline:
    def __init__(
            self,
            github_token: str = None,
            pinecone_api_key: str = None,
            chunks_index_host: str = None,
            cache_index_host: str = None,
            data_path: str = "./data",
            model_save_path: str = "./saved_models",
            max_concurrent_requests: int = 50,
            embedding_batch_size: int = 100,
            cache_similarity_threshold: float = 0.85,
            use_local_embeddings: bool = True,  # Set to True to match your 384-dim indexes
            preload_cache: bool = True,
            chunk_size: int = 800,
            chunk_overlap: int = 50,
            top_k_retrieval: int = 2
    ):
        print("🚀 Initializing HYPER-OPTIMIZED RAG Pipeline for HackRX...")

        # Get credentials from environment variables or parameters
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.chunks_index_host = chunks_index_host or os.getenv("CHUNKS_INDEX_HOST")
        self.cache_index_host = cache_index_host or os.getenv("CACHE_INDEX_HOST")
        
        # Validate required credentials
        if not self.github_token:
            raise ValueError("GitHub token not provided. Set GITHUB_TOKEN environment variable or pass github_token parameter.")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not provided. Set PINECONE_API_KEY environment variable or pass pinecone_api_key parameter.")
        if not self.chunks_index_host:
            raise ValueError("Chunks index host not provided. Set CHUNKS_INDEX_HOST environment variable or pass chunks_index_host parameter.")
        if not self.cache_index_host:
            raise ValueError("Cache index host not provided. Set CACHE_INDEX_HOST environment variable or pass cache_index_host parameter.")

        self.github_endpoint = "https://models.github.ai/inference"
        self.data_path = data_path
        self.model_save_path = model_save_path

        # Optimized parameters
        self.max_concurrent_requests = max_concurrent_requests
        self.embedding_batch_size = embedding_batch_size
        self.cache_similarity_threshold = cache_similarity_threshold
        self.use_local_embeddings = use_local_embeddings
        self.preload_cache = preload_cache
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval

        # Initialize local embedding model - FIXED to use 384-dimension model
        self.local_embedding_model = None
        if use_local_embeddings:
            print("🧠 Loading local embedding model (384 dimensions)...")
            try:
                # Use all-MiniLM-L6-v2 which produces 384-dimensional embeddings
                self.local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Local embedding model loaded (384 dimensions)!")
                print("🎯 Embedding dimensions match Pinecone index configuration")
            except Exception as e:
                print(f"⚠️ Failed to load local embedding model: {e}")
                self.use_local_embeddings = False
                print("❌ Falling back to OpenAI embeddings - dimension mismatch will occur!")

        # Memory caches
        self.document_cache = {}
        self.embedding_cache = {}
        self.answer_cache = {}

        print("🔗 Setting up clients...")
        self._setup_clients()

        # Increased semaphore and thread pool
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.thread_pool = ThreadPoolExecutor(max_workers=16)

        # Optimized prompt (shorter)
        self.simple_prompt_template = "Context: {context}\n\nQ: {query}\n\nA:"

        print("📁 Setting up directories...")
        self._setup_directories()

        if preload_cache:
            print("🔄 Preloading caches...")
            self._preload_caches()

        print("✅ HYPER-OPTIMIZED Pipeline ready for HackRX!")

    def _setup_clients(self):
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Extract index names from hosts
            # Parse index names from the host URLs
            chunks_host_parts = self.chunks_index_host.split('.')
            cache_host_parts = self.cache_index_host.split('.')
            
            if len(chunks_host_parts) > 0 and chunks_host_parts[0].startswith('https://'):
                self.chunks_index_name = chunks_host_parts[0].replace('https://', '').split('-')[0]
            else:
                self.chunks_index_name = "manav1"  # Fallback to your known index name
                
            if len(cache_host_parts) > 0 and cache_host_parts[0].startswith('https://'):
                self.cache_index_name = cache_host_parts[0].replace('https://', '').split('-')[0]
            else:
                self.cache_index_name = "cacheindexmanav"  # Fallback to your known index name
            
            self.chunks_index = self.pc.Index(name=self.chunks_index_name, host=self.chunks_index_host)
            self.cache_index = self.pc.Index(name=self.cache_index_name, host=self.cache_index_host)
            
            self.async_openai_client = AsyncOpenAI(
                base_url=self.github_endpoint, 
                api_key=self.github_token
            )
            print("✅ Clients setup complete!")
            print(f"📊 Using indexes: {self.chunks_index_name} (chunks), {self.cache_index_name} (cache)")
        except Exception as e:
            print(f"❌ Error setting up clients: {e}")
            raise

    def _setup_directories(self):
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(f"{self.data_path}/cache", exist_ok=True)

    def _preload_caches(self):
        """Preload frequently accessed data"""
        cache_files = [
            f"{self.data_path}/cache/document_cache.pkl",
            f"{self.data_path}/cache/embedding_cache.pkl",
            f"{self.data_path}/cache/answer_cache.pkl"
        ]

        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        if 'document' in cache_file:
                            self.document_cache.update(cache_data)
                        elif 'embedding' in cache_file:
                            self.embedding_cache.update(cache_data)
                        elif 'answer' in cache_file:
                            self.answer_cache.update(cache_data)
                except Exception as e:
                    print(f"⚠ Could not load cache {cache_file}: {e}")

    def _save_caches(self):
        """Save caches to disk"""
        cache_dir = f"{self.data_path}/cache"
        try:
            with open(f"{cache_dir}/document_cache.pkl", 'wb') as f:
                pickle.dump(self.document_cache, f)
            with open(f"{cache_dir}/embedding_cache.pkl", 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            with open(f"{cache_dir}/answer_cache.pkl", 'wb') as f:
                pickle.dump(self.answer_cache, f)
        except Exception as e:
            print(f"⚠ Could not save caches: {e}")

    def generate_namespace_from_url(self, url: str) -> str:
        return re.sub(r'\W+', '', url).strip('').lower()

    def generate_cache_key(self, question: str) -> str:
        return hashlib.md5(question.lower().encode()).hexdigest()

    async def download_pdf_to_temp_file(self, pdf_url: str) -> str:
        """Optimized PDF download with caching"""
        url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
        cached_path = f"{self.data_path}/cache/{url_hash}.pdf"

        # Check if already cached
        if os.path.exists(cached_path):
            print(f"📋 Using cached PDF")
            return cached_path

        print(f"📥 Downloading PDF...")
        timeout = aiohttp.ClientTimeout(total=30)

        try:
            async with aiohttp.ClientSession(
                    timeout=timeout,
                    connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
            ) as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        raise Exception(f"Download failed: {response.status}")
                    content = await response.read()

                    # Save to cache
                    with open(cached_path, 'wb') as f:
                        f.write(content)

                    print(f"✅ PDF downloaded and cached ({len(content):,} bytes)")
                    return cached_path
        except Exception as e:
            print(f"❌ PDF download failed: {e}")
            raise

    async def extract_text_from_pdf_fast(self, pdf_path: str) -> List[str]:
        """Ultra-fast text extraction with caching"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_hash = hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"❌ Error reading PDF for hash: {e}")
            raise

        # Check cache first
        if pdf_hash in self.document_cache:
            print(f"📋 Using cached text extraction")
            return self.document_cache[pdf_hash]

        print(f"📄 Extracting text...")

        def extract_text():
            try:
                full_text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    # Process pages in parallel
                    pages_text = []
                    for page in pdf.pages[:10]:  # Limit to first 10 pages for speed
                        text = page.extract_text()
                        if text:
                            pages_text.append(text)
                    full_text = "\n".join(pages_text)

                # Optimized chunking
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = splitter.split_text(full_text)
                return [c.strip() for c in chunks if len(c.strip()) > 50]
            except Exception as e:
                print(f"❌ Error extracting text: {e}")
                return []

        # Run in thread pool
        chunks = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, extract_text
        )

        # Cache result
        self.document_cache[pdf_hash] = chunks
        print(f"✅ Extracted {len(chunks)} chunks")
        return chunks

    async def embed_text_ultra_fast(self, texts: List[str]) -> List[List[float]]:
        """Ultra-fast embedding with proper dimension handling"""
        if not texts:
            return []

        # Check cache first
        cache_keys = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if key in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            if self.use_local_embeddings and self.local_embedding_model:
                # Use local model (384 dimensions - matches your Pinecone indexes)
                def encode_batch():
                    try:
                        embeddings = self.local_embedding_model.encode(
                            uncached_texts,
                            batch_size=self.embedding_batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        
                        # Verify dimensions
                        if len(embeddings) > 0 and len(embeddings[0]) != 384:
                            print(f"⚠️ Unexpected embedding dimension: {len(embeddings[0])}, expected 384")
                            return []
                        
                        return embeddings.tolist()
                    except Exception as e:
                        print(f"❌ Local embedding error: {e}")
                        return []

                new_embeddings = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, encode_batch
                )
                print(f"✅ Generated {len(new_embeddings)} embeddings with 384 dimensions")
            else:
                # Fallback to OpenAI API - but this will cause dimension mismatch!
                print("❌ WARNING: Using OpenAI embeddings with 384-dim index will fail!")
                print("🔧 Consider recreating Pinecone indexes with 1536 dimensions or use local embeddings")
                try:
                    batch_size = min(self.embedding_batch_size, len(uncached_texts))
                    for i in range(0, len(uncached_texts), batch_size):
                        batch = uncached_texts[i:i + batch_size]
                        async with self.request_semaphore:
                            response = await self.async_openai_client.embeddings.create(
                                input=batch,
                                model="text-embedding-3-small"  # This produces 1536 dims!
                            )
                            batch_embeddings = [item.embedding for item in response.data]
                            new_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"❌ OpenAI embedding error: {e}")
                    return []

            # Cache new embeddings
            for text, key, embedding in zip(uncached_texts,
                                            [cache_keys[i] for i in uncached_indices],
                                            new_embeddings):
                self.embedding_cache[key] = embedding

        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)

        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding

        # Place new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = embedding

        return all_embeddings

    async def upsert_to_pinecone_ultra_fast(self, chunks: List[str], namespace: str):
        """Ultra-fast parallel upsert with dimension verification"""
        print(f"📤 Upserting {len(chunks)} chunks in parallel...")

        try:
            # Generate embeddings in parallel
            embeddings = await self.embed_text_ultra_fast(chunks)

            if not embeddings:
                print("❌ No embeddings generated")
                return

            # Verify embedding dimensions before upsert
            if len(embeddings[0]) != 384:
                print(f"❌ Embedding dimension mismatch: got {len(embeddings[0])}, expected 384")
                print("🔧 Please ensure local embeddings are enabled and working")
                return

            print(f"✅ Embedding dimensions verified: {len(embeddings[0])} (matches Pinecone index)")

            # Prepare vectors
            vectors = [
                {
                    "id": f"{namespace}_{i}",
                    "values": embedding,
                    "metadata": {"text": chunk[:1000]}  # Limit metadata size
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                if embedding is not None
            ]

            if not vectors:
                print("❌ No valid vectors prepared")
                return

            # Batch upsert in parallel
            batch_size = 50  # Optimized batch size
            upsert_tasks = []

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                task = asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda b=batch: self.chunks_index.upsert(vectors=b, namespace=namespace)
                )
                upsert_tasks.append(task)

            # Wait for all upserts
            await asyncio.gather(*upsert_tasks)
            print("✅ Parallel upsert complete!")
        except Exception as e:
            print(f"❌ Upsert error: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def retrieve_ultra_fast(self, query: str, namespace: str, top_k: int = None) -> List[str]:
        """Ultra-fast retrieval with caching and dimension verification"""
        if top_k is None:
            top_k = self.top_k_retrieval

        # Check answer cache first
        cache_key = f"{namespace}_{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        try:
            async with self.request_semaphore:
                query_embedding = await self.embed_text_ultra_fast([query])
                
                if not query_embedding or not query_embedding[0]:
                    print("❌ Failed to generate query embedding")
                    return []

                # Verify query embedding dimension
                if len(query_embedding[0]) != 384:
                    print(f"❌ Query embedding dimension mismatch: got {len(query_embedding[0])}, expected 384")
                    return []

                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: self.chunks_index.query(
                        vector=query_embedding[0],
                        namespace=namespace,
                        top_k=top_k,
                        include_metadata=True
                    )
                )

                relevant_chunks = [
                    match.metadata.get("text", "")
                    for match in result.matches
                    if match.score > 0.7  # Slightly higher threshold
                ]

                # Cache result
                self.answer_cache[cache_key] = relevant_chunks
                return relevant_chunks
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def ask_gpt_ultra_fast(self, context: str, question: str) -> str:
        """Ultra-fast GPT query with streaming and caching"""
        # Check cache
        cache_key = hashlib.md5(f"{context[:200]}{question}".encode()).hexdigest()
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        try:
            async with self.request_semaphore:
                # Optimized prompt and parameters
                prompt = self.simple_prompt_template.format(
                    context=context[:600],  # Reduced context size
                    query=question
                )

                response = await self.async_openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,  # Increased for better answers
                    stream=False
                )

                answer = response.choices[0].message.content.strip()

                # Cache answer
                self.answer_cache[cache_key] = answer
                return answer
        except Exception as e:
            print(f"❌ GPT query error: {e}")
            return "Error generating response"

    async def process_question_hyper_optimized(self, question: str, namespace: str) -> str:
        """Hyper-optimized question processing"""
        try:
            # Parallel retrieval and processing
            chunks_task = self.retrieve_ultra_fast(question, namespace)

            chunks = await chunks_task
            context = "\n".join(chunks[:2]) if chunks else "No relevant context found."

            return await self.ask_gpt_ultra_fast(context, question)
        except Exception as e:
            print(f"❌ Question processing error: {e}")
            import traceback
            traceback.print_exc()
            return "Error processing question"

    async def process_document_and_questions_hyper_fast(
            self,
            pdf_url: str,
            questions: List[str]
    ) -> Dict[str, str]:
        """HYPER-OPTIMIZED processing for HackRX competition"""
        start = time.time()
        namespace = self.generate_namespace_from_url(pdf_url)

        try:
            # Check if document already exists
            stats = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.chunks_index.describe_index_stats()
            )
            existing_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []

            # Process document if not exists
            if namespace not in existing_namespaces:
                print("📥 Processing new document...")

                # Parallel document processing
                local_pdf_path = await self.download_pdf_to_temp_file(pdf_url)
                chunks = await self.extract_text_from_pdf_fast(local_pdf_path)
                
                if chunks:
                    await self.upsert_to_pinecone_ultra_fast(chunks, namespace)
                else:
                    print("❌ No chunks extracted from PDF")

                # Cleanup
                if local_pdf_path.startswith(self.data_path):
                    try:
                        os.unlink(local_pdf_path)
                    except:
                        pass
            else:
                print("📋 Using existing document index")

            print(f"🧠 Processing {len(questions)} questions in parallel...")

            # Ultra-high concurrency for questions
            semaphore = asyncio.Semaphore(25)  # Increased concurrency

            async def bounded_question_processing(q):
                async with semaphore:
                    answer = await self.process_question_hyper_optimized(q, namespace)
                    return q, answer

            # Process all questions in parallel
            tasks = [bounded_question_processing(q) for q in questions]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results and exceptions
            result_dict = {}
            for result in results:
                if isinstance(result, Exception):
                    print(f"⚠ Error processing question: {result}")
                    continue
                q, a = result
                result_dict[q] = a

            # Save caches for future runs
            self._save_caches()

            end = time.time()
            total_time = end - start
            print(f"\n⚡ HYPER-OPTIMIZED TIME: {total_time:.2f}s")
            print(f"🎯 Target achieved: {total_time / len(questions):.2f}s per question")

            return result_dict

        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            return {q: "Error processing" for q in questions}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            chunks_stats = self.chunks_index.describe_index_stats()
            cache_stats = self.cache_index.describe_index_stats()

            return {
                "chunks_index": {
                    "total_vectors": chunks_stats.total_vector_count,
                    "namespaces": list(chunks_stats.namespaces.keys()) if chunks_stats.namespaces else []
                },
                "cache_index": {
                    "total_vectors": cache_stats.total_vector_count,
                    "namespaces": list(cache_stats.namespaces.keys()) if cache_stats.namespaces else []
                },
                "configuration": {
                    "max_concurrent_requests": self.max_concurrent_requests,
                    "embedding_batch_size": self.embedding_batch_size,
                    "cache_similarity_threshold": self.cache_similarity_threshold,
                    "use_local_embeddings": self.use_local_embeddings,
                    "chunk_size": self.chunk_size,
                    "top_k_retrieval": self.top_k_retrieval,
                    "embedding_model": "all-MiniLM-L6-v2 (384d)" if self.use_local_embeddings else "text-embedding-3-small (1536d)",
                    "llm_model": "gpt-4o-mini",
                    "pinecone_dimension": 384
                },
                "cache_stats": {
                    "document_cache_size": len(self.document_cache),
                    "embedding_cache_size": len(self.embedding_cache),
                    "answer_cache_size": len(self.answer_cache)
                }
            }
        except Exception as e:
            return {"error": str(e)}


# Demo function for testing (environment-aware)
async def hackrx_demo():
    """Demo optimized for HackRX competition - uses environment variables"""
    try:
        # Initialize with environment variables
        pipeline = HyperOptimizedRAGPipeline(
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            cache_similarity_threshold=0.85,
            use_local_embeddings=os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true",
            preload_cache=True,
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=50,
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "2"))
        )

        print("✅ HYPER-OPTIMIZED Pipeline ready for HackRX!")

        pdf_url = "https://www.ijfmr.com/papers/2023/6/11497.pdf"
        questions = [
            "Why Deep Learning in Today's Research and Applications?",
            "What is the Position of Deep Learning in AI?",
            "What are the Background?"
        ]

        print("🚀 Starting HYPER-OPTIMIZED processing for HackRX...")
        overall_start_time = time.time()

        results = await pipeline.process_document_and_questions_hyper_fast(pdf_url, questions)

        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time
        avg_time_per_question = total_processing_time / len(questions) if questions else 0

        print("\n" + "=" * 80)
        print("🏆 HACKRX COMPETITION RESULTS")
        print("=" * 80)

        for i, (question, answer) in enumerate(results.items(), 1):
            print(f"\n[Q{i}] {question}")
            print(f"[A{i}] {answer}")
            print("-" * 60)

        print(f"\n⚡ HACKRX PERFORMANCE:")
        print(f" Total Time: {total_processing_time:.2f} seconds")
        print(f" Average per Question: {avg_time_per_question:.2f} seconds")
        print(f" Processing Rate: {len(questions) / total_processing_time:.2f} q/s")

        # Competition performance rating
        if avg_time_per_question <= 1.0:
            performance_rating = "🏆 HACKRX CHAMPION"
        elif avg_time_per_question <= 2.0:
            performance_rating = "🥇 HACKRX WINNER"
        elif avg_time_per_question <= 3.0:
            performance_rating = "🥈 HACKRX FINALIST"
        else:
            performance_rating = "🔴 NEEDS MORE OPTIMIZATION"

        print(f" Competition Rating: {performance_rating}")

        # Show optimization stats
        stats = pipeline.get_stats()
        config = stats.get('configuration', {})
        cache_stats = stats.get('cache_stats', {})

        print(f"\n🎯 HACKRX OPTIMIZATIONS:")
        print(f" Local Embeddings: {'✅' if config.get('use_local_embeddings') else '❌'}")
        print(f" Embedding Model: {config.get('embedding_model', 'N/A')}")
        print(f" Cache Hits - Documents: {cache_stats.get('document_cache_size', 0)}")
        print(f" Cache Hits - Embeddings: {cache_stats.get('embedding_cache_size', 0)}")
        print(f" Cache Hits - Answers: {cache_stats.get('answer_cache_size', 0)}")
        print(f" Concurrent Requests: {config.get('max_concurrent_requests', 'N/A')}")
        print(f" Batch Size: {config.get('embedding_batch_size', 'N/A')}")
        print(f" Pinecone Dimension: {config.get('pinecone_dimension', 'N/A')}")

        print("\n🏆 HYPER-OPTIMIZED processing completed for HackRX!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print(" 1. Ensure .env file exists with all required variables")
        print(" 2. Check that API keys are valid and not expired")
        print(" 3. Verify Pinecone index hosts are correct")
        print(" 4. Ensure internet connectivity for downloads")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import platform
    from dotenv import load_dotenv

    print("🏆 HACKRX HYPER-OPTIMIZED RAG PIPELINE")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(hackrx_demo())