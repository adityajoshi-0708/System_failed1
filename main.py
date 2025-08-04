import os
import asyncio
import platform
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import time
import hashlib

# Load environment variables
load_dotenv()

# Import the new advanced pipeline
try:
    from hybrid_rag_pipeline import HybridAdvancedRAGPipeline
except ImportError:
    raise ImportError("hybrid_rag_pipeline.py not found. Please ensure it's in the same directory.")

# Pydantic models for request/response
class HackRXRequest(BaseModel):
    documents: str = Field(..., description="PDF URL to process")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackRXResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

# Global pipeline instance
pipeline = None

# FastAPI app initialization
app = FastAPI(
    title="HackRX Advanced Hybrid RAG API",
    description="Advanced Hybrid RAG Pipeline for HackRX Competition (Target: 90%+ accuracy, <2s latency)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

def verify_bearer_token(authorization: str = Header(None)):
    """Verify Bearer token from environment"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    
    token = authorization.split(" ")[1]
    expected_token = os.getenv("Bearer_Token")
    
    if not expected_token:
        raise HTTPException(status_code=500, detail="Server authentication not configured")
    
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid bearer token")
    
    return token

async def get_pipeline():
    """Get or initialize the advanced hybrid pipeline"""
    global pipeline
    if pipeline is None:
        try:
            # Initialize pipeline with advanced features and environment variables
            pipeline = HybridAdvancedRAGPipeline(
                github_token=os.getenv("GITHUB_TOKEN"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                chunks_index_host=os.getenv("CHUNKS_INDEX_HOST"),
                cache_index_host=os.getenv("CACHE_INDEX_HOST"),
                max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
                embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "200")),
                use_local_embeddings=os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true",
                chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
                top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "10")),
                max_tokens_response=int(os.getenv("MAX_TOKEN_RESPONSE", "150")),
                embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "15000")),
                answer_cache_size=int(os.getenv("ANSWER_CACHE_SIZE", "8000")),
                reranker_top_k=int(os.getenv("TOP_N_RERANK", "3")),
                use_hybrid_search=True,  # Enable hybrid search
                use_reranking=True,      # Enable reranking
                use_context_compression=True,  # Enable context compression
                compression_ratio=float(os.getenv("HYBRID_ALPHA", "0.7"))  # Context compression ratio
            )
            print("✅ Advanced Hybrid RAG Pipeline initialized with all features!")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")
    
    return pipeline

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup with advanced optimizations"""
    print("🚀 Starting HackRX Advanced Hybrid RAG API...")
    
    # Check required environment variables
    required_vars = [
        "GITHUB_TOKEN",
        "PINECONE_API_KEY",
        "CHUNKS_INDEX_HOST",
        "CACHE_INDEX_HOST",
        "Bearer_Token"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")
    
    # Set Windows event loop policy
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Initialize pipeline
    await get_pipeline()
    print("✅ HackRX Advanced Hybrid RAG API ready!")
    print("🎯 Features: Hybrid Search + Reranking + Context Compression")
    print("📊 Target: 90%+ accuracy, <2s latency")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown with cache saving"""
    print("🛑 Shutting down HackRX Advanced Hybrid RAG API...")
    global pipeline
    if pipeline:
        try:
            pipeline.save_caches()
            print("💾 Advanced caches saved successfully")
        except Exception as e:
            print(f"⚠ Cache save warning: {e}")
    
    print("✅ HackRX Advanced Hybrid RAG API shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HackRX Advanced Hybrid RAG API</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 900px; margin: 0 auto; background: rgba(255,255,255,0.95); padding: 40px; border-radius: 15px; color: #333; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
            .header { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; text-align: center; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
            .feature-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
            .endpoint { background: #ecf0f1; padding: 20px; margin: 15px 0; border-radius: 10px; border-left: 5px solid #3498db; }
            .method { background: #e74c3c; color: white; padding: 5px 12px; border-radius: 5px; font-size: 12px; font-weight: bold; }
            .url { font-family: 'Consolas', monospace; background: #34495e; color: #ecf0f1; padding: 8px 12px; border-radius: 5px; margin: 10px 0; }
            .example { background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 8px; font-family: 'Consolas', monospace; overflow-x: auto; }
            .status { padding: 15px; margin: 15px 0; border-radius: 8px; text-align: center; font-weight: bold; }
            .success { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); color: #155724; }
            .highlight { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🏆 HackRX Advanced Hybrid RAG API v2.0</h1>
            <div class="status success">
                ✅ Advanced Hybrid RAG Pipeline is operational! 🚀
            </div>
            
            <div class="highlight">
                <h2>🎯 Competition-Ready Features</h2>
                <p><strong>90%+ Accuracy Target</strong> • <strong>&lt;2s Latency</strong> • <strong>Token Optimized</strong></p>
            </div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>🔍 Hybrid Search</h3>
                    <p>Semantic + Keyword search for maximum accuracy</p>
                </div>
                <div class="feature-card">
                    <h3>🎯 Cross-Encoder Reranking</h3>
                    <p>AI-powered result reranking for precision</p>
                </div>
                <div class="feature-card">
                    <h3>🗜 Context Compression</h3>
                    <p>Intelligent filtering for token efficiency</p>
                </div>
                <div class="feature-card">
                    <h3>🧠 BAAI Embeddings</h3>
                    <p>1024-dim high-quality embeddings</p>
                </div>
            </div>
            
            <h2>📋 API Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> Main Processing Endpoint</h3>
                <div class="url">/hackrx/run</div>
                <p><strong>Description:</strong> Process PDF document and answer questions with advanced hybrid RAG</p>
                <p><strong>Authentication:</strong> Bearer token required</p>
                <p><strong>Features:</strong> Hybrid search, reranking, context compression, explainable results</p>
            </div>
            
            <h3>📝 Request Format:</h3>
            <div class="example">
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic?",
        "What are the key findings?",
        "What are the conclusions?"
    ]
}
            </div>
            
            <h3>📋 Response Format:</h3>
            <div class="example">
{
    "answers": [
        "The main topic is advanced machine learning techniques...",
        "The key findings include improved accuracy of 95%...",
        "The conclusions demonstrate significant improvements..."
    ]
}
            </div>
            
            <h3>🔧 Authentication:</h3>
            <div class="example">
Authorization: Bearer &lt;your_token&gt;
Content-Type: application/json
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> API Documentation</h3>
                <div class="url">/docs</div>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> Health Check</h3>
                <div class="url">/health</div>
                <p>Check API health and advanced pipeline status</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> Advanced Statistics</h3>
                <div class="url">/stats</div>
                <p>Detailed pipeline performance metrics and configuration</p>
            </div>
            
            <h2>⚡ Advanced Optimization Features</h2>
            <ul style="font-size: 16px; line-height: 1.8;">
                <li>🧠 <strong>BAAI/bge-large-en-v1.5</strong> embeddings (1024-dimension)</li>
                <li>🎯 <strong>Cross-encoder reranking</strong> for maximum accuracy</li>
                <li>🔍 <strong>Hybrid search</strong> (semantic + keyword TF-IDF)</li>
                <li>🗜 <strong>LLM-based context compression</strong> for token efficiency</li>
                <li>💾 <strong>Multi-level caching</strong> (embeddings, answers, keywords)</li>
                <li>🚀 <strong>High concurrency</strong> (100+ concurrent requests)</li>
                <li>📊 <strong>Explainable results</strong> with detailed metrics</li>
                <li>🔒 <strong>Secure authentication</strong> with Bearer tokens</li>
                <li>⚡ <strong>Performance optimized</strong> for competition standards</li>
            </ul>
            
            <div class="highlight">
                <h2>🏆 HackRX Competition Ready</h2>
                <p><strong>Advanced Hybrid RAG Pipeline v2.0</strong><br>
                Engineered for maximum accuracy and optimal performance!</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint with advanced pipeline stats"""
    try:
        pipeline_instance = await get_pipeline()
        stats = pipeline_instance.get_advanced_stats()
        
        return JSONResponse({
            "status": "healthy",
            "message": "HackRX Advanced Hybrid RAG API is operational",
            "pipeline_ready": True,
            "version": "2.0.0",
            "features": {
                "hybrid_search": "✅ Active",
                "reranking": "✅ Active", 
                "context_compression": "✅ Active",
                "local_embeddings": "✅ Active"
            },
            "cache_stats": stats.get("cache_stats", {}),
            "performance_stats": stats.get("performance_stats", {}),
            "configuration": stats.get("configuration", {})
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"Pipeline error: {str(e)}",
                "pipeline_ready": False
            }
        )

@app.get("/hackrx/run")
async def hackrx_run_info():
    """HackRX endpoint - GET request shows API status and usage info"""
    return JSONResponse({
        "status": "✅ HackRX Advanced Hybrid RAG API is WORKING!",
        "message": "Advanced pipeline with hybrid search, reranking, and context compression",
        "version": "2.0.0",
        "endpoint": "/hackrx/run",
        "method": "POST (for processing)",
        "current_method": "GET (for status check)",
        "features": {
            "hybrid_search": "Semantic + Keyword search",
            "reranking": "Cross-encoder optimization",
            "context_compression": "LLM-based intelligent filtering",
            "embeddings": "BAAI/bge-large-en-v1.5 (1024-dim)",
            "performance_target": "90%+ accuracy, <2s latency"
        },
        "team_info": {
            "success": "🎉 Your Advanced Hybrid RAG API is ready for competition!",
            "next_step": "Use POST method with JSON payload to process documents",
            "performance_target": "90%+ accuracy with <2 second latency",
            "authentication": "Bearer token required"
        },
        "usage": {
            "url": "/hackrx/run",
            "method": "POST",
            "content_type": "application/json",
            "headers": {
                "Authorization": "Bearer <your_token>",
                "Content-Type": "application/json"
            },
            "body_example": {
                "documents": "https://example.com/document.pdf",
                "questions": ["Question 1?", "Question 2?", "Question 3?"]
            }
        },
        "advanced_features": {
            "explainable_results": "Detailed reasoning for each answer",
            "metrics_tracking": "Performance metrics for each query",
            "token_optimization": "Context compression for efficiency",
            "accuracy_optimization": "Hybrid search + reranking pipeline"
        },
        "documentation": "Visit /docs for full API documentation"
    })

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run_api(
    request: HackRXRequest,
    token: str = Depends(verify_bearer_token)
):
    """Main HackRX endpoint - POST request processes PDF with advanced hybrid RAG"""
    try:
        pipeline_instance = await get_pipeline()
        
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        print(f"🚀 Processing {len(request.questions)} questions with Advanced Hybrid RAG...")
        print("🔍 Features: Hybrid Search + Reranking + Context Compression")
        start_time = time.time()
        
        # Use the advanced processing method
        results = await pipeline_instance.process_document_and_questions_advanced(
            pdf_url=request.documents,
            questions=request.questions
        )
        
        # Extract answers from results (excluding _summary)
        answers = []
        for question in request.questions:
            if question in results:
                answer_data = results[question]
                if isinstance(answer_data, dict):
                    answers.append(answer_data.get("answer", "Unable to process question"))
                else:
                    answers.append(str(answer_data))
            else:
                answers.append("Unable to process question")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(request.questions) if request.questions else 0
        
        # Get summary metrics if available
        summary = results.get("_summary", {})
        avg_accuracy = summary.get("average_accuracy", 0)
        total_tokens = summary.get("total_tokens_used", 0)
        
        print(f"✅ Successfully processed {len(answers)} answers")
        print(f"⚡ Total Time: {total_time:.2f}s, Avg: {avg_time:.2f}s per question")
        print(f"🎯 Average Accuracy: {avg_accuracy:.1f}%")
        print(f"💰 Total Tokens: {total_tokens}")
        print(f"🏆 Performance Target: {'✅ ACHIEVED' if avg_time <= 2.0 and avg_accuracy >= 90 else '🔧 NEEDS OPTIMIZATION'}")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in hackrx_run: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/stats")
async def get_pipeline_stats(token: str = Depends(verify_bearer_token)):
    """Get detailed advanced pipeline statistics"""
    try:
        pipeline_instance = await get_pipeline()
        stats = pipeline_instance.get_advanced_stats()
        
        return JSONResponse({
            "pipeline_version": "Advanced Hybrid RAG v2.0",
            "cache_stats": stats.get("cache_stats", {}),
            "performance_stats": stats.get("performance_stats", {}),
            "configuration": stats.get("configuration", {}),
            "features": stats.get("features", {}),
            "system_info": {
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "embedding_dimensions": 1024,
                "max_concurrent_requests": pipeline_instance.max_concurrent_requests,
                "chunk_size": pipeline_instance.chunk_size,
                "top_k_retrieval": pipeline_instance.top_k_retrieval,
                "reranker_top_k": pipeline_instance.reranker_top_k,
                "use_hybrid_search": pipeline_instance.use_hybrid_search,
                "use_reranking": pipeline_instance.use_reranking,
                "use_context_compression": pipeline_instance.use_context_compression
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Available endpoints: / (GET), /hackrx/run (POST), /health (GET), /docs (GET), /stats (GET)",
            "api_version": "Advanced Hybrid RAG v2.0"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please check server logs for details",
            "api_version": "Advanced Hybrid RAG v2.0"
        }
    )

if __name__ == "__main__":
    print("🚀 Starting HackRX Advanced Hybrid RAG API Server...")
    print("🔒 Environment variables loaded from .env file")
    print("🎯 Features: Hybrid Search + Reranking + Context Compression")
    print("📊 Target: 90%+ accuracy, <2s latency")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏆 HackRX endpoint: http://localhost:8000/hackrx/run")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,
        log_level="info",
        access_log=True,
        timeout_keep_alive=10
    )