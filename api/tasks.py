"""
Background task processing for segmentation jobs.

Handles:
- Async job queue management
- Segmentation pipeline execution
- Job status tracking
"""

import asyncio
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

from api.schemas import JobStatus, SegmentationResult, ComponentInfo, SegmentJobCreate
from api.storage import StorageManager, InMemoryStorage
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Job:
    """Internal job representation."""

    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    filename: str
    file_size: int
    file_path: Optional[Path] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    result: Optional[SegmentationResult] = None


class JobManager:
    """
    Manage segmentation jobs with async processing.

    Features:
    - In-memory job queue
    - Concurrent job processing
    - Job status tracking
    - Automatic cleanup
    """

    def __init__(
        self,
        storage: StorageManager,
        max_concurrent_jobs: int = 4,
        job_timeout: int = 300,
        job_retention_hours: int = 24,
    ):
        """
        Initialize job manager.

        Args:
            storage: Storage manager instance
            max_concurrent_jobs: Maximum concurrent jobs
            job_timeout: Job timeout in seconds
            job_retention_hours: How long to keep job data
        """
        self.storage = storage
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_timeout = job_timeout
        self.job_retention_hours = job_retention_hours

        # Job storage
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

        # Thread pool for CPU-bound inference
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)

        # Processing queue
        self._queue: asyncio.Queue = None
        self._workers: list = []
        self._running = False

        # Segmentation pipeline (lazy loaded)
        self._pipeline = None
        self._pipeline_lock = threading.Lock()

    def _get_pipeline(self):
        """Lazy load segmentation pipeline."""
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    from inference.predictor import Predictor
                    from inference.mesh_segmenter import MeshSegmenter
                    from inference.component_splitter import ComponentSplitter
                    from inference.glb_exporter import GLBExporter

                    # Load model
                    model_path = self.storage.get_model_path()
                    predictor = Predictor(
                        model_path=str(model_path),
                        device="cuda",
                        use_amp=True,
                    )

                    self._pipeline = {
                        "predictor": predictor,
                        "segmenter": MeshSegmenter(predictor, num_points=20000),
                        "splitter": ComponentSplitter(min_volume_ratio=0.001),
                        "exporter": GLBExporter(include_materials=True),
                    }
                    logger.info("Segmentation pipeline loaded")

        return self._pipeline

    async def start(self):
        """Start the job processing workers."""
        if self._running:
            return

        self._running = True
        self._queue = asyncio.Queue()

        # Start worker tasks
        for i in range(self.max_concurrent_jobs):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(f"Started {self.max_concurrent_jobs} job workers")

    async def stop(self):
        """Stop job processing workers."""
        self._running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        self._workers = []
        self._executor.shutdown(wait=True)

        logger.info("Job workers stopped")

    def create_job(
        self,
        job_id: str,
        filename: str,
        file_size: int,
        file_path: Path,
    ) -> Job:
        """Create a new job."""
        now = datetime.utcnow()
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            filename=filename,
            file_size=file_size,
            file_path=file_path,
        )

        with self._lock:
            self.jobs[job_id] = job

        logger.info(f"Created job {job_id}: {filename} ({file_size} bytes)")
        return job

    async def submit_job(self, job_id: str):
        """Submit a job for processing."""
        await self._queue.put(job_id)
        logger.info(f"Submitted job {job_id} to queue")

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def get_active_job_count(self) -> int:
        """Get number of active jobs."""
        with self._lock:
            return sum(
                1
                for job in self.jobs.values()
                if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
            )

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes jobs."""
        logger.info(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get job from queue with timeout
                try:
                    job_id = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process job
                await self._process_job(job_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"Worker {worker_id} stopped")

    async def _process_job(self, job_id: str):
        """Process a single job."""
        job = self.get_job(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found")
            return

        logger.info(f"Processing job {job_id}")
        start_time = time.time()

        try:
            # Update status
            self._update_job(job_id, status=JobStatus.PROCESSING, progress=0.0)

            # Run segmentation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._run_segmentation,
                job_id,
                job.file_path,
            )

            # Update with result
            processing_time = time.time() - start_time
            self._update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100.0,
                result=result,
            )

            logger.info(
                f"Job {job_id} completed in {processing_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            logger.error(traceback.format_exc())
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
            )

        finally:
            # Cleanup temp files
            self.storage.cleanup_job(job_id)

    def _run_segmentation(
        self, job_id: str, file_path: Path
    ) -> SegmentationResult:
        """
        Run the segmentation pipeline (synchronous, runs in thread pool).

        Steps:
        1. Load .3dm file
        2. Convert to trimesh
        3. Sample points
        4. Run model inference
        5. Map labels to faces
        6. Split into components
        7. Export GLB
        8. Upload to S3
        """
        import time
        from preprocessing.rhino_loader import RhinoLoader, merge_meshes
        from preprocessing.mesh_converter import MeshConverter

        start_time = time.time()

        # Step 1: Load .3dm file
        self._update_job(job_id, progress=10.0)
        loader = RhinoLoader(triangulate=True)
        extracted_meshes = loader.load(file_path)

        if len(extracted_meshes) == 0:
            raise ValueError("No valid meshes found in file")

        # Step 2: Convert to trimesh and merge
        self._update_job(job_id, progress=20.0)
        converter = MeshConverter()
        trimeshes = converter.convert_all(extracted_meshes)

        if len(trimeshes) == 0:
            raise ValueError("Failed to convert meshes")

        mesh = converter.merge(trimeshes)
        logger.info(
            f"Job {job_id}: Loaded mesh with "
            f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        # Step 3-4: Segment mesh (includes point sampling and inference)
        self._update_job(job_id, progress=40.0)
        pipeline = self._get_pipeline()
        segmenter = pipeline["segmenter"]
        face_labels = segmenter.segment(mesh)

        # Step 5: Split into components
        self._update_job(job_id, progress=70.0)
        splitter = pipeline["splitter"]
        components = splitter.split(mesh, face_labels)

        if len(components) == 0:
            raise ValueError("No components found after segmentation")

        # Step 6: Export GLB
        self._update_job(job_id, progress=85.0)
        exporter = pipeline["exporter"]
        glb_bytes = exporter.export(components)

        # Step 7: Upload to S3
        self._update_job(job_id, progress=95.0)
        s3_key, presigned_url = self.storage.upload_result(job_id, glb_bytes)

        processing_time = time.time() - start_time

        # Build result
        component_info = [
            ComponentInfo(
                name=c.name,
                class_type=c.class_name,
                volume=c.volume,
                face_count=c.face_count,
            )
            for c in components
        ]

        return SegmentationResult(
            output_url=presigned_url,
            output_key=s3_key,
            components=component_info,
            processing_time_seconds=processing_time,
            input_vertices=len(mesh.vertices),
            input_faces=len(mesh.faces),
        )

    def _update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        result: Optional[SegmentationResult] = None,
    ):
        """Update job state."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return

            job.updated_at = datetime.utcnow()

            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if error_message is not None:
                job.error_message = error_message
            if result is not None:
                job.result = result

    async def cleanup_old_jobs(self):
        """Remove old completed/failed jobs."""
        now = datetime.utcnow()
        cutoff = now.timestamp() - (self.job_retention_hours * 3600)

        with self._lock:
            to_remove = [
                job_id
                for job_id, job in self.jobs.items()
                if (
                    job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
                    and job.updated_at.timestamp() < cutoff
                )
            ]

            for job_id in to_remove:
                del self.jobs[job_id]
                self.storage.cleanup_job(job_id)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")
