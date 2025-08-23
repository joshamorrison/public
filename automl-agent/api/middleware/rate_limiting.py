import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from ..models.response_models import JobStatus, JobStatusEnum, AgentResult, JobResult

logger = logging.getLogger(__name__)

@dataclass
class Job:
    job_id: str
    task_function: Callable
    args: tuple
    kwargs: dict
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: JobStatusEnum = JobStatusEnum.QUEUED
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    priority: int = 1
    max_retries: int = 3
    retry_count: int = 0
    estimated_duration: Optional[float] = None
    
    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.completed_at or self.started_at or self.created_at,
            estimated_completion=self._calculate_estimated_completion()
        )
    
    def _calculate_estimated_completion(self) -> Optional[datetime]:
        if self.status == JobStatusEnum.COMPLETED or self.status == JobStatusEnum.FAILED:
            return self.completed_at
        
        if self.status == JobStatusEnum.RUNNING and self.started_at and self.estimated_duration:
            if self.progress > 0:
                estimated_total = (time.time() - self.started_at.timestamp()) / self.progress
                return datetime.fromtimestamp(self.started_at.timestamp() + estimated_total)
            elif self.estimated_duration:
                return self.started_at + timedelta(seconds=self.estimated_duration)
        
        return None

class JobManager:
    def __init__(self, max_concurrent_jobs: int = 5, cleanup_interval: int = 3600):
        self.jobs: Dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.cleanup_interval = cleanup_interval
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        # Progress update callbacks
        self.progress_callbacks: Dict[str, List[Callable[[str, float], None]]] = {}
    
    def _start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_completed_jobs())
    
    async def submit_job(
        self, 
        task_function: Callable, 
        *args, 
        priority: int = 1,
        estimated_duration: Optional[float] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Submit a new job to the queue"""
        job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            task_function=task_function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            estimated_duration=estimated_duration,
            max_retries=max_retries
        )
        
        self.jobs[job_id] = job
        logger.info(f"Job {job_id} submitted to queue")
        
        # Start job execution if we have capacity
        await self._try_start_job(job_id)
        
        return job_id
    
    async def _try_start_job(self, job_id: str):
        """Try to start a job if we have capacity"""
        if len(self.running_tasks) >= self.max_concurrent_jobs:
            return
        
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatusEnum.QUEUED:
            return
        
        # Start the job
        task = asyncio.create_task(self._execute_job(job_id))
        self.running_tasks[job_id] = task
        
        job.status = JobStatusEnum.RUNNING
        job.started_at = datetime.now()
        logger.info(f"Job {job_id} started execution")
    
    async def _execute_job(self, job_id: str):
        """Execute a single job"""
        job = self.jobs[job_id]
        
        try:
            # Create progress callback for this job
            def update_progress(progress: float):
                job.progress = max(0.0, min(1.0, progress))
                # Trigger progress callbacks
                for callback in self.progress_callbacks.get(job_id, []):
                    try:
                        callback(job_id, progress)
                    except Exception as e:
                        logger.warning(f"Progress callback error for job {job_id}: {e}")
            
            # Add progress callback to kwargs
            job.kwargs['progress_callback'] = update_progress
            
            # Execute the task function
            if asyncio.iscoroutinefunction(job.task_function):
                result = await job.task_function(*job.args, **job.kwargs)
            else:
                # Run in thread pool for CPU-bound tasks
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, 
                    lambda: job.task_function(*job.args, **job.kwargs)
                )
            
            job.result = result
            job.status = JobStatusEnum.COMPLETED
            job.progress = 1.0
            job.completed_at = datetime.now()
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job.error = str(e)
            job.status = JobStatusEnum.FAILED
            job.completed_at = datetime.now()
            
            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatusEnum.QUEUED
                job.started_at = None
                job.completed_at = None
                job.error = None
                logger.info(f"Job {job_id} queued for retry ({job.retry_count}/{job.max_retries})")
                
                # Retry after a delay
                await asyncio.sleep(2 ** job.retry_count)  # Exponential backoff
                await self._try_start_job(job_id)
                return
        
        finally:
            # Remove from running tasks
            self.running_tasks.pop(job_id, None)
            
            # Try to start next queued job
            await self._start_next_queued_job()
    
    async def _start_next_queued_job(self):
        """Start the next highest priority queued job"""
        if len(self.running_tasks) >= self.max_concurrent_jobs:
            return
        
        # Find highest priority queued job
        queued_jobs = [
            (job_id, job) for job_id, job in self.jobs.items() 
            if job.status == JobStatusEnum.QUEUED
        ]
        
        if not queued_jobs:
            return
        
        # Sort by priority (higher number = higher priority)
        queued_jobs.sort(key=lambda x: x[1].priority, reverse=True)
        next_job_id = queued_jobs[0][0]
        
        await self._try_start_job(next_job_id)
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the current status of a job"""
        job = self.jobs.get(job_id)
        return job.to_status() if job else None
    
    def list_jobs(
        self, 
        status: Optional[JobStatusEnum] = None, 
        limit: int = 50
    ) -> List[JobStatus]:
        """List jobs with optional filtering"""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return [job.to_status() for job in jobs[:limit]]
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's not completed"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED]:
            return False
        
        # Cancel running task
        task = self.running_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        job.status = JobStatusEnum.CANCELLED
        job.completed_at = datetime.now()
        
        # Remove from running tasks
        self.running_tasks.pop(job_id, None)
        
        # Try to start next job
        await self._start_next_queued_job()
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    def add_progress_callback(self, job_id: str, callback: Callable[[str, float], None]):
        """Add a progress callback for a specific job"""
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)
    
    def remove_progress_callback(self, job_id: str, callback: Callable[[str, float], None]):
        """Remove a progress callback for a specific job"""
        if job_id in self.progress_callbacks:
            try:
                self.progress_callbacks[job_id].remove(callback)
            except ValueError:
                pass
    
    async def _cleanup_completed_jobs(self):
        """Background task to clean up old completed jobs"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                cutoff_time = datetime.now() - timedelta(hours=24)  # Keep jobs for 24 hours
                jobs_to_remove = []
                
                for job_id, job in self.jobs.items():
                    if (job.status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED, JobStatusEnum.CANCELLED] 
                        and job.completed_at and job.completed_at < cutoff_time):
                        jobs_to_remove.append(job_id)
                
                for job_id in jobs_to_remove:
                    del self.jobs[job_id]
                    self.progress_callbacks.pop(job_id, None)
                    logger.info(f"Cleaned up old job {job_id}")
                
                if jobs_to_remove:
                    logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for monitoring"""
        total_jobs = len(self.jobs)
        running_jobs = len(self.running_tasks)
        queued_jobs = len([j for j in self.jobs.values() if j.status == JobStatusEnum.QUEUED])
        completed_jobs = len([j for j in self.jobs.values() if j.status == JobStatusEnum.COMPLETED])
        failed_jobs = len([j for j in self.jobs.values() if j.status == JobStatusEnum.FAILED])
        
        return {
            "total_jobs": total_jobs,
            "running_jobs": running_jobs,
            "queued_jobs": queued_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "executor_threads": self.executor._max_workers
        }
    
    async def shutdown(self):
        """Gracefully shutdown the job manager"""
        logger.info("Shutting down job manager...")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Job manager shutdown complete")

# Global job manager instance
job_manager = JobManager()