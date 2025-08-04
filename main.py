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

# Import the new pipeline
try:
    from hybrid_rag_pipeline import UltraFastRAGPipeline
except ImportError:
    raise ImportError("ultra_fast_rag_pipeline.py not found. Please ensure it's in the same directory.")

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
    title="HackRX Ultra-Fast RAG API",
    description="Ultra-Fast RAG Pipeline for HackRX Competition (Target: 1.5s per question)",
    version="1.0.0",
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
    """Get or initialize the ultra-fast pipeline"""
    global pipeline
    if pipeline is None:
        try:
            # Initialize pipeline with optimized parameters
            pipeline = UltraFastRAGPipeline(
                github_token=os.getenv("GITHUB_TOKEN"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                chunks_index_host=os.getenv("CHUNKS_INDEX_HOST"),
                cache_index_host=os.getenv("CACHE_INDEX_HOST"),
                max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
                embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "200")),
                use_local_embeddings=os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true",
                chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
                top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "1")),  # Only 1 chunk for speed
                max_tokens_response=int(os.getenv("MAX_TOKENS_RESPONSE", "80")),  # Minimal response
                embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "10000")),
                answer_cache_size=int(os.getenv("ANSWER_CACHE_SIZE", "5000"))
            )
            print("✅ Ultra-Fast Pipeline initialized with optimized parameters!")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")
    
    return pipeline

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup with optimizations"""
    print("🚀 Starting HackRX Ultra-Fast RAG API...")
    
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
    print("✅ HackRX Ultra-Fast RAG API ready! Target: 1.5s per question")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown with cache saving"""
    print("🛑 Shutting down HackRX Ultra-Fast RAG API...")
    global pipeline
    if pipeline:
        try:
            pipeline._save_critical_caches()
            print("💾 Critical caches saved successfully")
        except Exception as e:
            print(f"⚠️ Cache save warning: {e}")
    
    print("✅ HackRX Ultra-Fast RAG API shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HackRX Ultra-Fast RAG API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .url { font-family: monospace; background: #34495e; color: white; padding: 5px; border-radius: 3px; }
            .example { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d5edda; color: #155724; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🏆 HackRX Ultra-Fast RAG API</h1>
            <div class="status success">
                ✅ API is running and optimized for HackRX competition! (Target: 1.5s per question)
            </div>
            
            <h2>📋 API Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> Main Processing Endpoint</h3>
                <div class="url">/hackrx/run</div>
                <p><strong>Description:</strong> Process PDF document and answer questions ultra-fast</p>
                <p><strong>Authentication:</strong> Bearer token required</p>
                <p><strong>Content-Type:</strong> application/json</p>
            </div>
            
            <h3>📝 Request Format:</h3>
            <div class="example">
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic?",
        "What are the key findings?"
    ]
}
            </div>
            
            <h3>📋 Response Format:</h3>
            <div class="example">
{
    "answers": [
        "The main topic is...",
        "The key findings are..."
    ]
}
            </div>
            
            <h3>🔧 Authentication:</h3>
            <div class="example">
Authorization: Bearer &lt;your_token&gt;
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> API Documentation</h3>
                <div class="url">/docs</div>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> Health Check</h3>
                <div class="url">/health</div>
                <p>Check API health and system status</p>
            </div>
            
            <h2>⚡ Optimization Features</h2>
            <ul>
                <li>🧠 Local embeddings (all-MiniLM-L6-v2, 384-dimension)</li>
                <li>💾 Aggressive LRU caching system</li>
                <li>🚀 Maximum concurrency (100 requests)</li>
                <li>⚡ Single chunk retrieval (top_k=1)</li>
                <li>📝 Minimal response tokens (80 max)</li>
                <li>🔒 Secure authentication</li>
                <li>🎯 1.5s per question target</li>
            </ul>
            
            <p><strong>🎯 HackRX Ready:</strong> Ultra-optimized for competition performance!</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint with optimized pipeline stats"""
    try:
        pipeline_instance = await get_pipeline()
        stats = pipeline_instance.get_stats()
        
        return JSONResponse({
            "status": "healthy",
            "message": "HackRX Ultra-Fast RAG API is operational",
            "pipeline_ready": True,
            "cache_stats": {
                "embedding_cache_size": len(pipeline_instance.embedding_cache),
                "answer_cache_size": len(pipeline_instance.answer_cache)
            },
            "configuration": {
                "use_local_embeddings": pipeline_instance.use_local_embeddings,
                "embedding_model": "all-MiniLM-L6-v2" if pipeline_instance.use_local_embeddings else "remote",
                "max_concurrent_requests": pipeline_instance.max_concurrent_requests,
                "top_k_retrieval": pipeline_instance.top_k_retrieval,
                "max_tokens_response": pipeline_instance.max_tokens_response
            }
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
        "status": "✅ HackRX API is WORKING!",
        "message": "API endpoint is operational and ready for hackathon submissions",
        "endpoint": "/hackrx/run",
        "method": "POST (for processing)",
        "current_method": "GET (for status check)",
        "team_info": {
            "success": "🎉 Your API endpoint is correctly configured!",
            "next_step": "Use POST method with JSON payload to process documents",
            "performance_target": "1.5 seconds per question",
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
                "questions": ["Question 1?", "Question 2?"]
            }
        },
        "test_interface": "Visit / (root) for interactive test form",
        "documentation": "Visit /docs for full API documentation"
    })

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run_api(
    request: HackRXRequest,
    token: str = Depends(verify_bearer_token)
):
    """Main HackRX endpoint - POST request processes PDF and returns answers"""
    try:
        pipeline_instance = await get_pipeline()
        
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        print(f"🚀 Processing {len(request.questions)} questions ultra-fast...")
        start_time = time.time()
        
        results = await pipeline_instance.process_document_and_questions_ultra_fast(
            pdf_url=request.documents,
            questions=request.questions
        )
        
        answers = [results.get(question, "Unable to process question") for question in request.questions]
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(request.questions) if request.questions else 0
        
        print(f"✅ Successfully processed {len(answers)} answers")
        print(f"⚡ Total Time: {total_time:.2f}s, Avg: {avg_time:.2f}s")
        print(f"🎯 Target Achievement: {'✅' if avg_time <= 1.5 else '❌'}")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in hackrx_run: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/stats")
async def get_pipeline_stats(token: str = Depends(verify_bearer_token)):
    """Get detailed pipeline statistics"""
    try:
        pipeline_instance = await get_pipeline()
        stats = pipeline_instance.get_stats()
        return JSONResponse({
            "cache_stats": {
                "embedding_cache_size": len(pipeline_instance.embedding_cache),
                "answer_cache_size": len(pipeline_instance.answer_cache)
            },
            "configuration": {
                "use_local_embeddings": pipeline_instance.use_local_embeddings,
                "max_concurrent_requests": pipeline_instance.max_concurrent_requests,
                "top_k_retrieval": pipeline_instance.top_k_retrieval,
                "max_tokens_response": pipeline_instance.max_tokens_response,
                "chunk_size": pipeline_instance.chunk_size,
                "chunk_overlap": pipeline_instance.chunk_overlap
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
            "message": "Available endpoints: / (GET), /hackrx/run (POST), /health (GET), /docs (GET), /stats (GET)"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please check server logs for details"
        }
    )

if __name__ == "__main__":
    print("🚀 Starting HackRX Ultra-Fast RAG API Server...")
    print("🔒 Environment variables loaded from .env file")
    print("📊 API Documentation available at: http://localhost:8000/docs")
    print("🏆 HackRX endpoint: http://localhost:8000/hackrx/run")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,  # Increased for production
        log_level="info",
        access_log=True,
        timeout_keep_alive=5  # Reduced for faster connections
    )