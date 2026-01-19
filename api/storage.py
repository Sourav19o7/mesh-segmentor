"""
S3 storage integration for API.

Handles:
- Uploading output GLB files
- Generating presigned URLs
- Temporary file management
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import aiofiles
from utils.s3 import S3Client
from utils.logging import get_logger

logger = get_logger(__name__)


class StorageManager:
    """
    Manage file storage for the API.

    Handles:
    - Temporary file storage for uploads
    - S3 upload of results
    - Presigned URL generation
    """

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str = "outputs/",
        presigned_expiry: int = 3600,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize storage manager.

        Args:
            s3_bucket: S3 bucket name
            s3_prefix: Prefix for output files
            presigned_expiry: Presigned URL expiry in seconds
            temp_dir: Temporary directory for files
        """
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.presigned_expiry = presigned_expiry
        self.temp_dir = temp_dir or tempfile.gettempdir()

        self.s3_client = S3Client(bucket=s3_bucket)

        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"StorageManager initialized: bucket={s3_bucket}, "
            f"prefix={s3_prefix}, temp_dir={self.temp_dir}"
        )

    def get_temp_path(self, job_id: str, filename: str) -> Path:
        """Get temporary file path for a job."""
        job_dir = Path(self.temp_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir / filename

    async def save_upload(
        self,
        job_id: str,
        filename: str,
        content: bytes,
    ) -> Path:
        """
        Save uploaded file to temporary storage.

        Args:
            job_id: Job identifier
            filename: Original filename
            content: File content

        Returns:
            Path to saved file
        """
        file_path = self.get_temp_path(job_id, filename)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        logger.debug(f"Saved upload: {file_path} ({len(content)} bytes)")
        return file_path

    def upload_result(
        self,
        job_id: str,
        glb_bytes: bytes,
    ) -> Tuple[str, str]:
        """
        Upload GLB result to S3.

        Args:
            job_id: Job identifier
            glb_bytes: GLB file content

        Returns:
            Tuple of (s3_key, presigned_url)
        """
        s3_key = f"{self.s3_prefix}{job_id}/segmented.glb"

        # Upload to S3
        self.s3_client.upload_bytes(
            glb_bytes, s3_key, content_type="model/gltf-binary"
        )

        # Generate presigned URL
        presigned_url = self.s3_client.generate_presigned_url(
            s3_key, expiration=self.presigned_expiry
        )

        logger.info(f"Uploaded result: s3://{self.s3_bucket}/{s3_key}")
        return s3_key, presigned_url

    def cleanup_job(self, job_id: str):
        """Clean up temporary files for a job."""
        job_dir = Path(self.temp_dir) / job_id

        if job_dir.exists():
            import shutil

            shutil.rmtree(job_dir)
            logger.debug(f"Cleaned up job directory: {job_dir}")

    def get_model_path(self) -> Path:
        """Get local path for model file, downloading if needed."""
        model_path = Path(self.temp_dir) / "models" / "best_model.pt"

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Download from S3
            s3_key = "models/best_model.pt"
            logger.info(f"Downloading model from S3: {s3_key}")
            self.s3_client.download_file(s3_key, model_path)

        return model_path


class InMemoryStorage:
    """
    In-memory storage for development/testing.

    Stores job data in memory instead of S3.
    """

    def __init__(self):
        self.files: dict = {}
        self.temp_dir = tempfile.mkdtemp()

    def get_temp_path(self, job_id: str, filename: str) -> Path:
        job_dir = Path(self.temp_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir / filename

    async def save_upload(
        self, job_id: str, filename: str, content: bytes
    ) -> Path:
        file_path = self.get_temp_path(job_id, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        return file_path

    def upload_result(
        self, job_id: str, glb_bytes: bytes
    ) -> Tuple[str, str]:
        key = f"outputs/{job_id}/segmented.glb"
        self.files[key] = glb_bytes

        # For in-memory storage, return a placeholder URL
        url = f"/download/{job_id}/segmented.glb"
        return key, url

    def get_result(self, job_id: str) -> Optional[bytes]:
        key = f"outputs/{job_id}/segmented.glb"
        return self.files.get(key)

    def cleanup_job(self, job_id: str):
        job_dir = Path(self.temp_dir) / job_id
        if job_dir.exists():
            import shutil
            shutil.rmtree(job_dir)

        key = f"outputs/{job_id}/segmented.glb"
        self.files.pop(key, None)
