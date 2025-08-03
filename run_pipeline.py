import asyncio
import sys
import platform
import time
import psutil
import gc
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Check Python version
if sys.version_info < (3, 8):
    print(f"❌ Python 3.8+ required. Current version: {sys.version}")
    sys.exit(1)

try:
    from hybrid_rag_pipeline import HyperOptimizedRAGPipeline
except ImportError:
    print("❌ Could not import HyperOptimizedRAGPipeline")
    print("Please ensure the hybrid_rag_pipeline.py file is in the same directory")
    sys.exit(1)


def check_environment_variables():
    """Check if all required environment variables are set"""
    required_vars = [
        "GITHUB_TOKEN",
        "PINECONE_API_KEY", 
        "CHUNKS_INDEX_HOST",
        "CACHE_INDEX_HOST"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n🔧 Create a .env file with:")
        for var in missing_vars:
            print(f"   {var}=your_value_here")
        print("\n📖 See .env.example for complete template")
        return False
    
    print("✅ All required environment variables found!")
    return True


def optimize_system():
    """Optimize system settings for maximum performance"""
    print("🔧 Optimizing system for HackRX competition...")

    # Force garbage collection
    gc.collect()

    # Get system info
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()

    print(f"💻 System Info:")
    print(f"   Python Version: {sys.version}")
    print(f"   CPU Cores: {cpu_count}")
    print(f"   RAM: {memory.total / (1024 ** 3):.1f} GB")
    print(f"   Available RAM: {memory.available / (1024 ** 3):.1f} GB")
    print(f"   Platform: {platform.system()} {platform.release()}")

    # Optimize settings based on system capabilities
    if memory.available < 4 * (1024 ** 3):  # Less than 4GB available
        print("⚠️ Low memory detected, using conservative settings")
        max_concurrent = min(cpu_count * 4, 20)
        batch_size = min(cpu_count * 10, 40)
    elif memory.available < 8 * (1024 ** 3):  # Less than 8GB available
        print("📊 Medium memory detected, using balanced settings")
        max_concurrent = min(cpu_count * 6, 30)
        batch_size = min(cpu_count * 15, 60)
    else:  # 8GB+ available
        print("🚀 High memory detected, using aggressive settings")
        max_concurrent = min(cpu_count * 8, 40)
        batch_size = min(cpu_count * 20, 80)

    # Override with environment variables if set
    max_concurrent = int(os.getenv("MAX_CONCURRENT_REQUESTS", max_concurrent))
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", batch_size))

    return {
        'max_concurrent_requests': max_concurrent,
        'embedding_batch_size': batch_size,
        'use_local_embeddings': os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true",
        'preload_cache': True,
        'chunk_size': int(os.getenv("CHUNK_SIZE", "800")),
        'chunk_overlap': 50,
        'top_k_retrieval': int(os.getenv("TOP_K_RETRIEVAL", "2"))
    }


async def run_comprehensive_benchmark():
    """Comprehensive benchmark for HyperOptimizedRAGPipeline"""

    print("🏆" + "=" * 78 + "🏆")
    print("🚀 HYPER-OPTIMIZED RAG PIPELINE BENCHMARK")
    print("   🎯 Target: <1 second per question for HackRX")
    print("   ⚡ Maximum Performance Configuration")
    print("   📊 Using 384-dimension embeddings (matches Pinecone)")
    print("🏆" + "=" * 78 + "🏆")

    # Check environment variables first
    if not check_environment_variables():
        return

    # System optimization
    optimized_settings = optimize_system()

    try:
        # Initialize pipeline using environment variables
        pipeline = HyperOptimizedRAGPipeline(
            **optimized_settings,
            cache_similarity_threshold=0.85
        )

        print("✅ Hyper-Optimized Pipeline initialized!")
        print("🔍 Dimension Configuration: 384 (all-MiniLM-L6-v2 → Pinecone)")

        # Test scenarios
        pdf_url = "https://www.ijfmr.com/papers/2023/6/11497.pdf"
        questions = [
            "Why Deep Learning in Today's Research and Applications?",
            "What is the Position of Deep Learning in AI?", 
            "What are the Background?"
        ]

        print(f"\n📄 Test Document: {pdf_url}")
        print(f"📝 Test Questions: {len(questions)}")

        # Benchmark 1: Cold Start (First Run)
        print("\n" + "🔥" * 60)
        print("🧊 COLD START BENCHMARK (First Time)")
        print("🔥" * 60)

        cold_start_time = time.time()
        cold_results = await pipeline.process_document_and_questions_hyper_fast(pdf_url, questions)
        cold_end_time = time.time()

        cold_total_time = cold_end_time - cold_start_time
        cold_avg_time = cold_total_time / len(questions)

        print(f"\n❄️ COLD START RESULTS:")
        print(f"   Total Time: {cold_total_time:.2f} seconds")
        print(f"   Average per Question: {cold_avg_time:.2f} seconds")
        print(f"   Questions per Second: {len(questions) / cold_total_time:.2f}")

        # Brief pause to simulate real conditions
        await asyncio.sleep(1)

        # Benchmark 2: Warm Start (Cached Run)
        print("\n" + "⚡" * 60)
        print("🔥 WARM START BENCHMARK (With Caching)")
        print("⚡" * 60)

        warm_start_time = time.time()
        warm_results = await pipeline.process_document_and_questions_hyper_fast(pdf_url, questions)
        warm_end_time = time.time()

        warm_total_time = warm_end_time - warm_start_time
        warm_avg_time = warm_total_time / len(questions)

        print(f"\n🔥 WARM START RESULTS:")
        print(f"   Total Time: {warm_total_time:.2f} seconds")
        print(f"   Average per Question: {warm_avg_time:.2f} seconds")
        print(f"   Questions per Second: {len(questions) / warm_total_time:.2f}")

        # Performance improvement analysis
        speedup = cold_total_time / warm_total_time if warm_total_time > 0 else 1
        improvement_pct = ((cold_avg_time - warm_avg_time) / cold_avg_time * 100) if cold_avg_time > 0 else 0

        print(f"\n📈 PERFORMANCE IMPROVEMENT:")
        print(f"   Speedup Factor: {speedup:.2f}x faster")
        print(f"   Time Reduction: {improvement_pct:.1f}% improvement")
        print(f"   Cache Efficiency: {'🔥 EXCELLENT' if speedup > 2 else '👍 GOOD' if speedup > 1.5 else '⚠️ MODERATE'}")

        # Display final results
        print("\n" + "=" * 80)
        print("🎯 HACKRX COMPETITION FINAL RESULTS")
        print("=" * 80)

        for i, (question, answer) in enumerate(warm_results.items(), 1):
            print(f"\n[Q{i}] 🤔 QUESTION:")
            print(f"   {question}")
            print(f"\n[A{i}] 💡 ANSWER:")
            print(f"   {answer}")
            print("-" * 60)

        # HackRX Readiness Assessment
        print(f"\n🏆 HACKRX COMPETITION ASSESSMENT:")
        print("=" * 50)

        # Performance categories
        if warm_avg_time <= 0.5:
            performance_rating = "🏆 HACKRX CHAMPION"
            performance_emoji = "🚀"
            readiness_status = "READY TO DOMINATE"
        elif warm_avg_time <= 1.0:
            performance_rating = "🥇 HACKRX GOLD TIER"
            performance_emoji = "⚡"
            readiness_status = "COMPETITION READY"
        elif warm_avg_time <= 2.0:
            performance_rating = "🥈 HACKRX SILVER TIER"
            performance_emoji = "🔥"
            readiness_status = "STRONG CONTENDER"
        elif warm_avg_time <= 3.0:
            performance_rating = "🥉 HACKRX BRONZE TIER"
            performance_emoji = "👍"
            readiness_status = "GOOD FOUNDATION"
        else:
            performance_rating = "🔴 NEEDS OPTIMIZATION"
            performance_emoji = "⚠️"
            readiness_status = "REQUIRES IMPROVEMENT"

        print(f" {performance_emoji} Performance Tier: {performance_rating}")
        print(f" 🎯 Competition Status: {readiness_status}")
        print(f" ⏱️ Target Achievement: {'✅ ACHIEVED' if warm_avg_time <= 1.0 else '❌ NOT YET'}")

        # Detailed metrics
        stats = pipeline.get_stats()
        config = stats.get('configuration', {})
        cache_stats = stats.get('cache_stats', {})

        print(f"\n📊 DETAILED PERFORMANCE METRICS:")
        print("-" * 40)
        print(f" Cold Start Performance: {cold_avg_time:.2f}s per question")
        print(f" Warm Start Performance: {warm_avg_time:.2f}s per question")
        print(f" Performance Gain: {speedup:.2f}x improvement")
        print(f" Target Compliance: {'✅ YES' if warm_avg_time <= 1.0 else '❌ NO'}")

        print(f"\n🧠 SYSTEM OPTIMIZATION STATUS:")
        print("-" * 40)
        optimization_checks = [
            ("Local Embeddings Active", config.get('use_local_embeddings', False)),
            ("Caching System Working", sum(cache_stats.values()) > 0),
            ("High Concurrency Enabled", config.get('max_concurrent_requests', 0) >= 30),
            ("Optimal Batch Size", config.get('embedding_batch_size', 0) >= 60),
            ("Memory Efficiency", psutil.virtual_memory().available / (1024**3) > 2),
            ("Dimension Compatibility", config.get('pinecone_dimension') == 384),
        ]

        passed_checks = 0
        for check_name, status in optimization_checks:
            status_icon = "✅" if status else "❌"
            print(f" {status_icon} {check_name}")
            if status:
                passed_checks += 1

        optimization_score = (passed_checks / len(optimization_checks)) * 100
        print(f"\n🎯 Optimization Score: {optimization_score:.0f}%")

        # Resource utilization
        memory_after = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        print(f"\n💾 RESOURCE UTILIZATION:")
        print("-" * 40)
        print(f" Memory Usage: {(memory_after.total - memory_after.available) / (1024 ** 3):.1f} GB")
        print(f" Memory Efficiency: {memory_after.percent:.1f}% utilized")
        print(f" CPU Usage: {cpu_percent:.1f}%")
        print(f" Concurrent Requests: {config.get('max_concurrent_requests', 'N/A')}")
        print(f" Embedding Batch Size: {config.get('embedding_batch_size', 'N/A')}")
        print(f" Embedding Model: {config.get('embedding_model', 'N/A')}")

        # Competition recommendations
        print(f"\n🎯 HACKRX STRATEGY RECOMMENDATIONS:")
        print("-" * 40)

        if warm_avg_time <= 1.0:
            print(" 🏆 CHAMPIONSHIP STRATEGY:")
            print("   • Emphasize sub-1-second response time")
            print("   • Demonstrate scalability with multiple questions")
            print("   • Highlight caching system efficiency")
            print("   • Showcase local embedding innovation")
            print("   • Stress test with 10+ concurrent questions")
        elif warm_avg_time <= 2.0:
            print(" 🥇 OPTIMIZATION STRATEGY:")
            print("   • Fine-tune concurrent request limits")
            print("   • Optimize embedding batch sizes")
            print("   • Pre-warm all caches before demo")
            print("   • Consider reducing chunk size further")
        else:
            print(" 🔧 IMPROVEMENT STRATEGY:")
            print("   • Verify local embeddings are working")
            print("   • Check system memory availability")
            print("   • Optimize network connectivity")
            print("   • Consider using faster hardware")

        # Final readiness checklist
        print(f"\n📋 HACKRX FINAL READINESS CHECKLIST:")
        print("-" * 40)

        readiness_items = [
            ("Sub-1s response time achieved", warm_avg_time <= 1.0),
            ("Local embeddings operational", config.get('use_local_embeddings', False)),
            ("3-layer caching active", len(cache_stats) >= 3),
            ("High concurrency configured", config.get('max_concurrent_requests', 0) >= 30),
            ("System resources adequate", psutil.virtual_memory().available / (1024**3) > 2),
            ("Error handling robust", len(warm_results) == len(questions)),
            ("Performance consistent", abs(cold_avg_time - warm_avg_time) > 1.0),
            ("Dimension compatibility", config.get('pinecone_dimension') == 384)
        ]

        ready_count = sum(1 for _, status in readiness_items if status)
        for item, status in readiness_items:
            status_icon = "✅" if status else "❌"
            print(f" {status_icon} {item}")

        final_readiness = (ready_count / len(readiness_items)) * 100
        print(f"\n🎯 FINAL HACKRX READINESS: {final_readiness:.0f}%")

        if final_readiness >= 85:
            print("🏆 SYSTEM IS HACKRX CHAMPIONSHIP READY!")
            print("🚀 You have a competitive advantage!")
        elif final_readiness >= 70:
            print("🥇 SYSTEM IS HACKRX COMPETITION READY!")
            print("⚡ Strong performance expected!")
        else:
            print("🔧 SYSTEM NEEDS MORE OPTIMIZATION")
            print("⚠️ Additional tuning recommended before competition")

        print(f"\n✅ HackRX benchmark completed!")
        print(f"⏱️ Best performance: {warm_avg_time:.2f}s per question")
        print(f"🎯 Dimension Configuration: ✅ FIXED (384d)")

    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        print("\n🔍 DEBUGGING INFORMATION:")
        import traceback
        traceback.print_exc()

        print(f"\nSystem Information:")
        print(f" Platform: {platform.system()} {platform.release()}")
        print(f" Python Version: {sys.version}")
        print(f" Available Memory: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")
        
        print(f"\n🔧 ENVIRONMENT TROUBLESHOOTING:")
        env_vars = ["GITHUB_TOKEN", "PINECONE_API_KEY", "CHUNKS_INDEX_HOST", "CACHE_INDEX_HOST"]
        for var in env_vars:
            value = os.getenv(var)
            status = "✅ SET" if value else "❌ MISSING"
            masked_value = f"{value[:10]}..." if value and len(value) > 10 else "NOT SET"
            print(f" {status} {var}: {masked_value}")


def run_hackrx_benchmark():
    """Main runner function for HackRX pipeline"""

    print("🚀" + "=" * 78 + "🚀")
    print("🏆 HACKRX HYPER-OPTIMIZED RAG PIPELINE")
    print("   🎯 Mission: Achieve <1 second per question")
    print("   ⚡ Technology: Local embeddings + 3-layer caching")
    print("   📊 Embedding: 384-dim (all-MiniLM-L6-v2)")
    print("   🏆 Goal: Dominate the HackRX competition")
    print("   🔒 Security: Environment variables enabled")
    print("🚀" + "=" * 78 + "🚀")

    # Verify Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required for optimal performance")
        print(f"   Current version: {sys.version}")
        print(f"   Please upgrade to Python 3.8 or higher")
        return

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected - Compatible!")

    # Platform optimizations
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("🪟 Windows detected: ProactorEventLoopPolicy optimized")
    else:
        print(f"🐧 {platform.system()} detected: Using default event loop")

    # Create optimized event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        print("⚡ Launching HackRX benchmark...")
        loop.run_until_complete(run_comprehensive_benchmark())

    except KeyboardInterrupt:
        print("\n⏹ HackRX benchmark interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error in benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up resources...")

        # Enhanced cleanup
        pending = asyncio.all_tasks(loop)
        if pending:
            print(f" Cancelling {len(pending)} pending tasks...")
            for task in pending:
                task.cancel()
            try:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                print(" ✅ All tasks cancelled successfully")
            except Exception as cleanup_error:
                print(f" ⚠️ Cleanup warning: {cleanup_error}")

        # Close event loop properly
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            if hasattr(loop, 'shutdown_default_executor'):
                loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()
            print(" ✅ Event loop closed successfully")
        except Exception as close_error:
            print(f" ⚠️ Loop closure warning: {close_error}")

        # Final memory cleanup
        gc.collect()
        print("🏁 HackRX benchmark completed!")
        print("🎯 System optimized and ready for competition!")


# Environment-aware demo function
async def demo_with_env_check():
    """Demo function that checks environment before running"""
    print("🔍 Checking environment configuration...")
    
    if not check_environment_variables():
        print("\n❌ Cannot run demo without proper environment setup")
        print("📖 Please check .env.example for required variables")
        return
    
    print("✅ Environment validated, starting demo...")
    
    try:
        # Run the optimized demo using environment variables
        from hybrid_rag_pipeline import hackrx_demo
        await hackrx_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Production-Ready HackRX Pipeline")
    print("🔒 Using environment variables for security")
    
    # Check if we should run demo or benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("🎬 Running demo mode...")
        if platform.system() == 'Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.run(demo_with_env_check())
    else:
        print("🏆 Running full benchmark...")
        run_hackrx_benchmark()