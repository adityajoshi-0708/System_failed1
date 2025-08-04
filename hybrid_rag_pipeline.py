import os
import re
import asyncio
import hashlib
import time
from typing import Dict, Any, List, Optional
import json
import pickle
from pathlib import Path
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Core dependencies
from openai import AsyncOpenAI
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
from sentence_transformers import SentenceTransformer


class UltraFastRAGPipeline:
    def __init__(
            self,
            github_token: str = None,
            pinecone_api_key: str = None,
            chunks_index_host: str = None,
            cache_index_host: str = None,
            data_path: str = "./data",
            max_concurrent_requests: int = 100,  # Will be adjusted for LLM calls
            embedding_batch_size: int = 200,     
            use_local_embeddings: bool = True,
            chunk_size: int = 600,               
            chunk_overlap: int = 30,             
            top_k_retrieval: int = 1,            
            max_tokens_response: int = 80,        
            embedding_cache_size : int = 10000,   
            answer_cache_size: int = 5000        
    ):
        print("🚀 Initializing ULTRA-FAST RAG Pipeline (Target: 1.5s per question)")

        # Environment variables
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.chunks_index_host = chunks_index_host or os.getenv("CHUNKS_INDEX_HOST")
        self.cache_index_host = cache_index_host or os.getenv("CACHE_INDEX_HOST")
        
        # Validate credentials
        required_vars = [self.github_token, self.pinecone_api_key, self.chunks_index_host]
        if not all(required_vars):
            raise ValueError("Missing required environment variables")

        self.github_endpoint = "https://models.github.ai/inference"
        self.data_path = data_path

        # OPTIMIZATION 1: Aggressive parameters for speed
        self.max_concurrent_requests = max_concurrent_requests
        self.embedding_batch_size = embedding_batch_size
        self.use_local_embeddings = use_local_embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval  
        self.max_tokens_response = max_tokens_response

        # OPTIMIZATION 2: Enhanced caching system
        from functools import lru_cache
        self.embedding_cache = {}
        self.answer_cache = {}
        self.embedding_cache_size = embedding_cache_size
        self.answer_cache_size = answer_cache_size

        # OPTIMIZATION 3: Pre-load and optimize embedding model
        self.local_embedding_model = None
        if use_local_embeddings:
            print("🧠 Loading optimized embedding model...")
            try:
                # Use faster, smaller model for speed
                self.local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Pre-compile for faster inference
                _ = self.local_embedding_model.encode(["warmup text"], show_progress_bar=False)
                print("✅ Embedding model optimized and warmed up!")
            except Exception as e:
                print(f"⚠️ Local embedding failed: {e}")
                self.use_local_embeddings = False

        print("🔗 Setting up optimized clients...")
        self._setup_clients()

        # RATE LIMIT FIX: Different semaphores for different operations
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.llm_semaphore = asyncio.Semaphore(3)  # CRITICAL: Only 3 concurrent LLM calls
        self.thread_pool = ThreadPoolExecutor(max_workers=32)  

        # OPTIMIZATION 5: Ultra-minimal prompt for speed
        self.ultra_minimal_prompt = "Context: {context}\nQ: {query}\nA (2 lines max):"

        # RATE LIMIT FIX: Add retry configuration
        self.max_retries = 3
        self.base_delay = 0.5
        self.backoff_factor = 2

        self._setup_directories()
        self._preload_critical_caches()

        print("✅ ULTRA-FAST Pipeline ready! Target: 1.5s per question")
        print(f"🔒 Rate limit protection: {3} concurrent LLM calls max")

    def _setup_clients(self):
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Extract index names from hosts
            chunks_host_parts = self.chunks_index_host.split('.')
            if len(chunks_host_parts) > 0 and chunks_host_parts[0].startswith('https://'):
                self.chunks_index_name = chunks_host_parts[0].replace('https://', '').split('-')[0]
            else:
                self.chunks_index_name = "manav1"
                
            self.chunks_index = self.pc.Index(name=self.chunks_index_name, host=self.chunks_index_host)
            
            # OPTIMIZATION 6: Faster OpenAI client with connection pooling
            self.async_openai_client = AsyncOpenAI(
                base_url=self.github_endpoint, 
                api_key=self.github_token,
                timeout=15.0,  # Increased timeout for stability
                max_retries=1   
            )
            print("✅ Optimized clients ready!")
        except Exception as e:
            print(f"❌ Client setup error: {e}")
            raise

    def _setup_directories(self):
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(f"{self.data_path}/cache", exist_ok=True)

    def _preload_critical_caches(self):
        """Preload only critical caches for speed"""
        cache_files = [
            f"{self.data_path}/cache/embedding_cache.pkl",
            f"{self.data_path}/cache/answer_cache.pkl"
        ]

        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        if 'embedding' in cache_file:
                            self.embedding_cache.update(cache_data)
                        elif 'answer' in cache_file:
                            self.answer_cache.update(cache_data)
                except Exception:
                    pass  # Ignore cache errors for speed

    def _manage_cache_size(self, cache_dict: dict, max_size: int):
        """Simple LRU cache management"""
        if len(cache_dict) > max_size:
            # Remove oldest 20% of entries
            items_to_remove = len(cache_dict) - int(max_size * 0.8)
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache_dict[key]

    async def ultra_fast_embedding(self, texts: List[str]) -> List[List[float]]:
        """OPTIMIZATION 7: Ultra-fast embedding with aggressive caching"""
        if not texts:
            return []

        # Check cache first
        cache_keys = [hashlib.md5(text.encode()).hexdigest()[:16] for text in texts]  # Shorter keys
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if key in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts only
        new_embeddings = []
        if uncached_texts and self.use_local_embeddings and self.local_embedding_model:
            def encode_batch():
                try:
                    # OPTIMIZATION 8: Batch processing with no progress bar
                    embeddings = self.local_embedding_model.encode(
                        uncached_texts,
                        batch_size=self.embedding_batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Pre-normalize for faster similarity
                    )
                    return embeddings.tolist()
                except Exception as e:
                    print(f"❌ Embedding error: {e}")
                    return []

            new_embeddings = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, encode_batch
            )

            # Cache new embeddings with size management
            for text, key, embedding in zip(uncached_texts,
                                          [cache_keys[i] for i in uncached_indices],
                                          new_embeddings):
                self.embedding_cache[key] = embedding
            
            # Manage cache size
            self._manage_cache_size(self.embedding_cache, self.embedding_cache_size)

        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        for i, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = embedding

        return all_embeddings

    async def lightning_fast_retrieval(self, query: str, namespace: str) -> List[str]:
        """OPTIMIZATION 9: Lightning-fast retrieval with minimal top_k"""
        # Ultra-aggressive caching
        cache_key = f"{namespace}_{hashlib.md5(query.encode()).hexdigest()[:12]}"
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        try:
            async with self.request_semaphore:
                # Single query embedding
                query_embedding = await self.ultra_fast_embedding([query])
                
                if not query_embedding or not query_embedding[0]:
                    return []

                # OPTIMIZATION 10: Minimal retrieval with higher score threshold
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: self.chunks_index.query(
                        vector=query_embedding[0],
                        namespace=namespace,
                        top_k=self.top_k_retrieval,  # Only 1 chunk!
                        include_metadata=True
                    )
                )

                # Only get the best match
                relevant_chunks = []
                if result.matches and result.matches[0].score > 0.75:  # Higher threshold
                    relevant_chunks = [result.matches[0].metadata.get("text", "")]

                # Cache with size management
                self.answer_cache[cache_key] = relevant_chunks
                self._manage_cache_size(self.answer_cache, self.answer_cache_size)
                
                return relevant_chunks
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

    # RATE LIMIT FIX: Add exponential backoff retry mechanism
    async def _llm_call_with_retry(self, prompt: str, retry_count: int = 0) -> str:
        """LLM call with exponential backoff retry for rate limits"""
        try:
            response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=self.max_tokens_response,
                stream=False,
                timeout=15.0
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle rate limit errors with exponential backoff
            if any(keyword in error_str for keyword in ['rate limit', '429', 'too many requests']):
                if retry_count < self.max_retries:
                    delay = self.base_delay * (self.backoff_factor ** retry_count)
                    print(f"⏳ Rate limit hit, retrying in {delay:.1f}s (attempt {retry_count + 1})")
                    await asyncio.sleep(delay)
                    return await self._llm_call_with_retry(prompt, retry_count + 1)
                else:
                    print(f"❌ Max retries reached for rate limit")
                    return "Rate limit exceeded - please try again later"
            else:
                print(f"❌ LLM error: {e}")
                return "Error generating response"

    async def ultra_fast_llm_call(self, context: str, question: str) -> str:
        """OPTIMIZATION 11: Ultra-fast LLM call with rate limit protection"""
        # Aggressive answer caching
        cache_key = hashlib.md5(f"{context[:100]}{question}".encode()).hexdigest()[:16]
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        # RATE LIMIT FIX: Use LLM-specific semaphore
        async with self.llm_semaphore:
            # OPTIMIZATION 12: Minimal context and ultra-short prompt
            prompt = self.ultra_minimal_prompt.format(
                context=context[:400],  # Very limited context
                query=question
            )

            answer = await self._llm_call_with_retry(prompt)

            # Cache with size management
            self.answer_cache[cache_key] = answer
            self._manage_cache_size(self.answer_cache, self.answer_cache_size)
            
            return answer

    async def process_question_ultra_fast(self, question: str, namespace: str) -> str:
        """OPTIMIZATION 13: Ultra-fast question processing pipeline"""
        try:
            # Single retrieval call with minimal chunks
            chunks = await self.lightning_fast_retrieval(question, namespace)
            context = chunks[0] if chunks else "No context available."

            # Single LLM call with minimal tokens and rate limit protection
            return await self.ultra_fast_llm_call(context, question)
        except Exception as e:
            print(f"❌ Question processing error: {e}")
            return "Error processing question"

    async def download_pdf_cached(self, pdf_url: str) -> str:
        """Fast PDF download with aggressive caching"""
        url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
        cached_path = f"{self.data_path}/cache/{url_hash}.pdf"

        if os.path.exists(cached_path):
            return cached_path

        timeout = aiohttp.ClientTimeout(total=20)  # Reduced timeout
        try:
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=100, limit_per_host=50)
            ) as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        raise Exception(f"Download failed: {response.status}")
                    content = await response.read()

                    with open(cached_path, 'wb') as f:
                        f.write(content)
                    return cached_path
        except Exception as e:
            print(f"❌ PDF download failed: {e}")
            raise

    async def extract_text_minimal(self, pdf_path: str) -> List[str]:
        """Minimal text extraction for speed"""
        def extract_text():
            try:
                full_text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    # Only first 5 pages for speed
                    for page in pdf.pages[:5]:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"

                # Smaller chunks for faster processing
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". "]
                )
                chunks = splitter.split_text(full_text)
                return [c.strip() for c in chunks if len(c.strip()) > 30]  # Lower threshold
            except Exception as e:
                print(f"❌ Text extraction error: {e}")
                return []

        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, extract_text
        )

    async def upsert_ultra_fast(self, chunks: List[str], namespace: str):
        """Ultra-fast upsert with maximum parallelism"""
        print(f"📤 Ultra-fast upsert: {len(chunks)} chunks...")

        try:
            embeddings = await self.ultra_fast_embedding(chunks)
            if not embeddings:
                return

            vectors = [
                {
                    "id": f"{namespace}_{i}",
                    "values": embedding,
                    "metadata": {"text": chunk[:800]}  # Limited metadata
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                if embedding is not None
            ]

            if not vectors:
                return

            # Maximum parallelism for upsert
            batch_size = 100  # Larger batches
            tasks = []

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                task = asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda b=batch: self.chunks_index.upsert(vectors=b, namespace=namespace)
                )
                tasks.append(task)

            await asyncio.gather(*tasks)
            print("✅ Ultra-fast upsert complete!")
        except Exception as e:
            print(f"❌ Upsert error: {e}")
            raise

    async def process_document_and_questions_ultra_fast(
            self,
            pdf_url: str,
            questions: List[str]
    ) -> Dict[str, str]:
        """OPTIMIZATION 14: Complete ultra-fast processing pipeline with rate limit protection"""
        start = time.time()
        namespace = re.sub(r'\W+', '', pdf_url).strip('').lower()

        try:
            # Check if document exists
            stats = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.chunks_index.describe_index_stats()
            )
            existing_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []

            # Process document if needed
            if namespace not in existing_namespaces:
                print("📥 Processing document ultra-fast...")
                local_pdf_path = await self.download_pdf_cached(pdf_url)
                chunks = await self.extract_text_minimal(local_pdf_path)
                
                if chunks:
                    await self.upsert_ultra_fast(chunks, namespace)

                # Cleanup
                try:
                    if local_pdf_path.startswith(self.data_path):
                        os.unlink(local_pdf_path)
                except:
                    pass
            else:
                print("📋 Using existing document")

            print(f"🧠 Processing {len(questions)} questions ultra-fast...")

            # RATE LIMIT FIX: Process questions in batches to respect rate limits
            batch_size = 3  # Process 3 questions at a time to stay within rate limits
            result_dict = {}

            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                print(f"📝 Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
                
                # Process batch with limited concurrency
                async def process_single_question(q):
                    return q, await self.process_question_ultra_fast(q, namespace)

                # Process batch in parallel but limited by semaphore
                tasks = [process_single_question(q) for q in batch_questions]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle batch results
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"❌ Question processing failed: {result}")
                        continue
                    q, a = result
                    result_dict[q] = a

                # Small delay between batches to be safe
                if i + batch_size < len(questions):
                    await asyncio.sleep(0.2)

            # Save critical caches only
            self._save_critical_caches()

            end = time.time()
            total_time = end - start
            avg_time = total_time / len(questions) if questions else 0

            print(f"\n⚡ ULTRA-FAST TIME: {total_time:.2f}s")
            print(f"🎯 Average per question: {avg_time:.2f}s")
            
            if avg_time <= 1.5:
                print("🏆 TARGET ACHIEVED: ≤ 1.5s per question!")
            else:
                print(f"🔴 Target missed by {avg_time - 1.5:.2f}s (due to rate limiting)")

            return result_dict

        except Exception as e:
            print(f"❌ Processing error: {e}")
            return {q: "Error processing" for q in questions}

    def _save_critical_caches(self):
        """Save only critical caches for speed"""
        try:
            cache_dir = f"{self.data_path}/cache"
            # Only save if cache has meaningful content
            if len(self.embedding_cache) > 10:
                with open(f"{cache_dir}/embedding_cache.pkl", 'wb') as f:
                    pickle.dump(dict(list(self.embedding_cache.items())[-1000:]), f)  # Only recent items
            if len(self.answer_cache) > 10:
                with open(f"{cache_dir}/answer_cache.pkl", 'wb') as f:
                    pickle.dump(dict(list(self.answer_cache.items())[-500:]), f)  # Only recent items
        except Exception:
            pass  # Ignore save errors for speed

    def get_stats(self):
        """Get pipeline statistics"""
        return {
            "embedding_cache_size": len(self.embedding_cache),
            "answer_cache_size": len(self.answer_cache),
            "max_concurrent_llm_calls": 3,
            "rate_limit_protection": True
        }


# Ultra-fast demo function
async def ultra_fast_demo():
    """Ultra-fast demo targeting 1.5s per question with rate limit protection"""
    try:
        # OPTIMIZATION 16: Aggressive initialization parameters with rate limit protection
        pipeline = UltraFastRAGPipeline(
            max_concurrent_requests=100,
            embedding_batch_size=200,
            use_local_embeddings=True,
            chunk_size=600,
            chunk_overlap=30,
            top_k_retrieval=1,           # Only 1 chunk!
            max_tokens_response=80,      # Very limited response
            embedding_cache_size=10000,
            answer_cache_size=5000
        )

        print("✅ ULTRA-FAST Pipeline ready with rate limit protection!")

        pdf_url = "https://www.ijfmr.com/papers/2023/6/11497.pdf"
        questions = [
            "Why Deep Learning in Today's Research?",
            "What is Deep Learning's position in AI?",
            "What are the key backgrounds?"
        ]

        print("🚀 Starting ULTRA-FAST processing...")
        start_time = time.time()

        results = await pipeline.process_document_and_questions_ultra_fast(pdf_url, questions)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(questions)

        print("\n" + "=" * 60)
        print("🏆 ULTRA-FAST RESULTS (Target: 1.5s per question)")
        print("=" * 60)

        for i, (question, answer) in enumerate(results.items(), 1):
            print(f"\n[Q{i}] {question}")
            print(f"[A{i}] {answer}")
            print("-" * 40)

        print(f"\n⚡ PERFORMANCE:")
        print(f" Total Time: {total_time:.2f}s")
        print(f" Average per Question: {avg_time:.2f}s")
        print(f" Target Achievement: {'✅' if avg_time <= 1.5 else '❌'}")
        
        if avg_time <= 1.5:
            print(" 🏆 SUCCESS: Target achieved!")
        else:
            print(f" 🔴 Target missed by: {avg_time - 1.5:.2f}s")

        print("\n🎯 OPTIMIZATIONS APPLIED:")
        print(" ✅ Only 1 chunk retrieval (top_k=1)")
        print(" ✅ Minimal response tokens (80 max)")
        print(" ✅ Aggressive caching with LRU")
        print(" ✅ Rate limit protection (3 concurrent LLM calls)")
        print(" ✅ Exponential backoff retry")
        print(" ✅ Batch processing for questions")
        print(" ✅ Minimal context (400 chars)")
        print(" ✅ Pre-normalized embeddings")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import platform
    from dotenv import load_dotenv

    print("🏆 ULTRA-FAST RAG PIPELINE (Target: 1.5s per question)")
    print("🔒 WITH RATE LIMIT PROTECTION")
    print("=" * 60)
    
    load_dotenv()

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(ultra_fast_demo())