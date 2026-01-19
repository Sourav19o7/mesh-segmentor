"""
FastAPI route definitions.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, Response

from api.schemas import (
    JobResponse,
    CreateJobResponse,
    HealthResponse,
    ErrorResponse,
    JobStatus,
)
from api.tasks import JobManager
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global job manager (set during app startup)
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Dependency to get job manager."""
    if _job_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized",
        )
    return _job_manager


def set_job_manager(manager: JobManager):
    """Set the global job manager."""
    global _job_manager
    _job_manager = manager


@router.post(
    "/segment",
    response_model=CreateJobResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["segmentation"],
    summary="Upload a .3dm file for segmentation",
    description="""
    Upload a Rhino (.3dm) file to segment into Metal and Gem components.

    The file is processed asynchronously. Use the returned job_id to
    check status via GET /segment/{job_id}.

    **Supported formats:** .3dm (Rhino)
    **Maximum file size:** 100 MB
    """,
)
async def create_segmentation_job(
    file: UploadFile = File(..., description="Rhino .3dm file to segment"),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Create a new segmentation job."""
    # Validate file extension
    if not file.filename.lower().endswith(".3dm"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file_type",
                "message": "Only .3dm files are supported",
            },
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Check file size (100 MB limit)
    max_size = 100 * 1024 * 1024  # 100 MB
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds maximum size of {max_size // (1024*1024)} MB",
            },
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = await job_manager.storage.save_upload(
        job_id, file.filename, content
    )

    # Create job
    job = job_manager.create_job(
        job_id=job_id,
        filename=file.filename,
        file_size=file_size,
        file_path=file_path,
    )

    # Submit for processing
    await job_manager.submit_job(job_id)

    logger.info(
        f"Created segmentation job {job_id} for {file.filename}",
        extra={"job_id": job_id},
    )

    return CreateJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job submitted for processing",
        estimated_time_seconds=60,  # Rough estimate
    )


@router.get(
    "/segment/{job_id}",
    response_model=JobResponse,
    responses={
        404: {"model": ErrorResponse},
    },
    tags=["segmentation"],
    summary="Get segmentation job status",
    description="""
    Check the status of a segmentation job.

    When status is "completed", the result will include:
    - output_url: Presigned S3 URL to download the GLB file
    - components: List of segmented components with names and volumes

    The output_url is valid for 1 hour.
    """,
)
async def get_segmentation_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    """Get job status and result."""
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"Job {job_id} not found",
            },
        )

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        error_message=job.error_message,
        result=job.result,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
    description="Check if the service is healthy and ready to accept requests.",
)
async def health_check(
    job_manager: JobManager = Depends(get_job_manager),
):
    """Health check endpoint."""
    import torch

    # Check if model is loaded
    pipeline = job_manager._pipeline
    model_loaded = pipeline is not None

    # Check GPU
    gpu_available = torch.cuda.is_available()

    # Count active jobs
    active_jobs = job_manager.get_active_job_count()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_loaded,
        gpu_available=gpu_available,
        active_jobs=active_jobs,
    )


@router.get(
    "/ready",
    tags=["system"],
    summary="Readiness check",
    description="Check if the service is ready to process requests.",
)
async def readiness_check(
    job_manager: JobManager = Depends(get_job_manager),
):
    """Readiness check for Kubernetes."""
    # Service is ready if job manager exists
    return {"status": "ready"}


@router.get(
    "/live",
    tags=["system"],
    summary="Liveness check",
    description="Check if the service is alive.",
)
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}
