"""
Pydantic models for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ComponentInfo(BaseModel):
    """Information about a segmented component."""

    name: str = Field(..., description="Component name (e.g., Metal_01, Gem_02)")
    class_type: str = Field(
        ..., alias="class", description="Class type (metal or gem)"
    )
    volume: float = Field(..., description="Component volume")
    face_count: int = Field(..., description="Number of faces in component")

    class Config:
        populate_by_name = True


class SegmentationResult(BaseModel):
    """Result of a segmentation job."""

    output_url: str = Field(..., description="Presigned S3 URL for output GLB file")
    output_key: str = Field(..., description="S3 key for output file")
    components: List[ComponentInfo] = Field(
        default_factory=list, description="List of segmented components"
    )
    processing_time_seconds: float = Field(
        ..., description="Total processing time"
    )
    input_vertices: int = Field(..., description="Number of vertices in input mesh")
    input_faces: int = Field(..., description="Number of faces in input mesh")


class JobResponse(BaseModel):
    """Response for job status endpoint."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    progress: Optional[float] = Field(
        None, ge=0, le=100, description="Progress percentage"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[SegmentationResult] = Field(
        None, description="Result when completed"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SegmentationRequest(BaseModel):
    """Request body for segmentation endpoint (used with form data)."""

    # These would be used if we had JSON body, but we use multipart
    pass


class SegmentJobCreate(BaseModel):
    """Internal model for creating a segmentation job."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    filename: str
    file_size: int
    progress: float = 0.0
    error_message: Optional[str] = None
    result: Optional[SegmentationResult] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    active_jobs: int = Field(..., description="Number of active jobs")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


# Job creation response
class CreateJobResponse(BaseModel):
    """Response when creating a new segmentation job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")
    estimated_time_seconds: Optional[int] = Field(
        None, description="Estimated processing time"
    )
