"""
FastAPI application entry point.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

Or for development:
    python -m api.main
"""

import os
import sys
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import router, set_job_manager
from api.tasks import JobManager
from api.storage import StorageManager, InMemoryStorage
from utils.logging import setup_logging, get_logger
from utils.config import load_inference_config

logger = get_logger(__name__)

# Global state
_job_manager: Optional[JobManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _job_manager

    logger.info("Starting mesh-segmentor API...")

    # Load configuration
    try:
        config = load_inference_config()
    except FileNotFoundError:
        logger.warning("Config not found, using defaults")
        config = None

    # Initialize storage
    s3_bucket = os.environ.get("S3_BUCKET", "mesh-segmentor")
    use_s3 = os.environ.get("USE_S3", "true").lower() == "true"

    if use_s3:
        storage = StorageManager(
            s3_bucket=s3_bucket,
            s3_prefix="outputs/",
            presigned_expiry=3600,
            temp_dir="/tmp/mesh-segmentor",
        )
    else:
        logger.warning("Using in-memory storage (development mode)")
        storage = InMemoryStorage()

    # Initialize job manager
    max_concurrent = int(os.environ.get("MAX_CONCURRENT_JOBS", "4"))
    job_timeout = int(os.environ.get("JOB_TIMEOUT", "300"))

    _job_manager = JobManager(
        storage=storage,
        max_concurrent_jobs=max_concurrent,
        job_timeout=job_timeout,
        job_retention_hours=24,
    )

    # Set job manager for routes
    set_job_manager(_job_manager)

    # Start job workers
    await _job_manager.start()

    # Preload model if configured
    preload_model = os.environ.get("PRELOAD_MODEL", "true").lower() == "true"
    if preload_model:
        logger.info("Preloading model...")
        try:
            _job_manager._get_pipeline()
            logger.info("Model preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")

    logger.info("API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down API...")
    await _job_manager.stop()
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Setup logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_format = os.environ.get("LOG_FORMAT", "json")
    setup_logging(level=log_level, format_type=log_format)

    # Create app
    app = FastAPI(
        title="Mesh Segmentor API",
        description="""
        Automatic 3D jewelry segmentation service.

        Upload a Rhino (.3dm) file to segment it into Metal and Gem components.
        The output is a GLB file with named mesh nodes.

        ## Workflow

        1. **POST /segment** - Upload .3dm file, receive job_id
        2. **GET /segment/{job_id}** - Poll for job status
        3. When complete, download GLB from the presigned URL

        ## Output Format

        The GLB file contains:
        - **Metal_01, Metal_02, ...** - Metal components (sorted by volume)
        - **Gem_01, Gem_02, ...** - Gem components (sorted by volume)
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time
        import uuid

        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Add request ID to state
        request.state.request_id = request_id

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"time={process_time:.3f}s "
            f"request_id={request_id}"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Invalid request",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An internal error occurred",
            },
        )

    # Include routes
    app.include_router(router, prefix="/api/v1")

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Mesh Segmentor API", "docs": "/docs"}

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=1,  # Single worker for GPU
        reload=False,
        log_level="info",
    )
