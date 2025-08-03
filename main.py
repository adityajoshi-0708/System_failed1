"""
🚀 HackRX RAG Pipeline - Production FastAPI Server
Deployment-ready for Railway with robust error handling
"""

import os
import asyncio
import platform
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import uvicorn

# Import your existing pipeline (UNCHANGED)
from hybrid_rag_pipeline import HyperOptimizedRAGPipeline

# Load environment variables
load_dotenv()

class ProcessRequest(BaseModel):
    """Enhanced request model with validation"""
    pdf_url: str = Field(..., description="URL of the PDF to process")
    questions: list[str] = Field(..., min_items=1, max_items=20, description="List of questions (max 20)")
    
    @validator('pdf_url')
    def validate_pdf_url(cls, v):
        if not v or not v.startswith(('http://', 'https://')):
            raise ValueError('PDF URL must be a valid HTTP/HTTPS URL')
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one question is required')
        # Filter out empty questions
        filtered = [q.strip() for q in v if q.strip()]
        if not filtered:
            raise ValueError('All questions cannot be empty')
        return filtered

class ProcessResponse(BaseModel):
    """Response model for process endpoint"""
    status: str
    results: Dict[str, str]
    processing_time: Optional[float] = None
    questions_count: int

class StatsResponse(BaseModel):
    """Response model for stats endpoint"""
    status: str
    stats: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    pipeline_ready: bool

# Global pipeline instance
pipeline_instance: Optional[HyperOptimizedRAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    global pipeline_instance
    
    print("🚀 Starting HackRX RAG Pipeline...")
    
    # Startup
    try:
        # Get environment variables with defaults
        github_token = os.getenv("GITHUB_TOKEN")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        chunks_index_host = os.getenv("CHUNKS_INDEX_HOST")
        cache_index_host = os.getenv("CACHE_INDEX_HOST")
        
        # Validate required environment variables
        required_vars = {
            "GITHUB_TOKEN": github_token,
            "PINECONE_API_KEY": pinecone_api_key,
            "CHUNKS_INDEX_HOST": chunks_index_host,
            "CACHE_INDEX_HOST": cache_index_host,
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize pipeline with your exact configuration (UNCHANGED LOGIC)
        pipeline_instance = HyperOptimizedRAGPipeline(
            github_token=github_token,
            pinecone_api_key=pinecone_api_key,
            chunks_index_host=chunks_index_host,
            cache_index_host=cache_index_host,
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            cache_similarity_threshold=0.85,
            use_local_embeddings=os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true",
            preload_cache=True,
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=50,
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "2"))
        )
        
        print("✅ Pipeline initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        pipeline_instance = None
        # Don't raise here - let the app start but mark pipeline as unavailable
    
    yield
    
    # Shutdown
    print("🛑 Shutting down HackRX RAG Pipeline...")
    pipeline_instance = None
    print("✅ Shutdown complete!")

# Create FastAPI app with lifespan
app = FastAPI(
    title="HackRX RAG Pipeline API",
    description="High-performance RAG pipeline optimized for HackRX competition",
    version="1.0.0",
    lifespan=lifespan
)

# Custom OpenAPI schema to create proper URL structure
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="HackRX RAG Pipeline API",
        version="1.0.0",
        description="High-performance RAG pipeline optimized for HackRX competition",
        routes=app.routes,
    )
    
    # Define custom tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "hackrx",
            "description": "HackRX pipeline operations"
        },
        {
            "name": "system",
            "description": "System health and monitoring"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Apply custom OpenAPI
app.openapi = custom_openapi

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully"""
    print(f"❌ Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "Contact support"
        }
    )

@app.get("/", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="HackRX RAG Pipeline is running!",
        pipeline_ready=pipeline_instance is not None
    )

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def detailed_health_check():
    """Detailed health check with pipeline status"""
    pipeline_ready = pipeline_instance is not None
    
    return HealthResponse(
        status="healthy" if pipeline_ready else "degraded",
        message=f"Pipeline Status: {'Ready' if pipeline_ready else 'Not Available'}",
        pipeline_ready=pipeline_ready
    )

@app.get("/stats", response_model=StatsResponse, tags=["system"])
async def get_system_stats():
    """Get system statistics"""
    if not pipeline_instance:
        raise HTTPException(
            status_code=503, 
            detail="Pipeline not available. Check environment variables and restart."
        )
    
    try:
        stats = pipeline_instance.get_stats()
        return StatsResponse(status="success", stats=stats)
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/hackrx/run", response_model=ProcessResponse, tags=["hackrx"], operation_id="run")
async def process_pdf_and_questions(request: ProcessRequest):
    """Process PDF and answer questions - YOUR CORE LOGIC (UNCHANGED)"""
    if not pipeline_instance:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available. Check environment variables and try again."
        )
    
    try:
        print(f"📝 Processing request with {len(request.questions)} questions")
        print(f"📄 PDF URL: {request.pdf_url}")
        
        import time
        start_time = time.time()
        
        # Use your existing pipeline method (UNCHANGED)
        results = await pipeline_instance.process_document_and_questions_hyper_fast(
            request.pdf_url, 
            request.questions
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Successfully processed {len(results)} questions in {processing_time:.2f}s")
        
        return ProcessResponse(
            status="success",
            results=results,
            processing_time=processing_time,
            questions_count=len(results)
        )
        
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return detailed error in development, generic in production
        error_detail = str(e) if os.getenv("DEBUG", "false").lower() == "true" else "Processing failed"
        raise HTTPException(status_code=500, detail=f"Processing failed: {error_detail}")

# Optional: Add a simple test endpoint
@app.get("/test", tags=["system"])
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "status": "success",
        "message": "API is working!",
        "pipeline_ready": pipeline_instance is not None,
        "environment": "production" if os.getenv("RAILWAY_ENVIRONMENT") else "development"
    }

if __name__ == "__main__":
    # Development server configuration
    print("🏆 Starting HackRX RAG API Server...")
    print("🌐 Development server - use Railway for production")
    
    # Windows optimization
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,  # Auto-reload in development
        log_level="info"
    )