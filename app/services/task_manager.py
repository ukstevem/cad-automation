"""
Background task manager.

Supports two task modes:
  - submit()       : sync callable run in a single-threaded ThreadPoolExecutor
                     (for OCP-based STL generation which holds the GIL)
  - submit_async() : async coroutine run as an asyncio.Task on the event loop
                     (for subprocess-based analysis which is truly non-blocking)

Tasks are tracked in-memory with status, progress, and results.
"""
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    filename: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    total_items: int = 0
    completed_items: int = 0
    current_item: str = ""
    results: Optional[Any] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "filename": self.filename,
            "status": self.status.value,
            "progress": self.progress,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "current_item": self.current_item,
            "results": self.results,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class TaskManager:
    """In-memory task manager backed by a single-threaded executor."""

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, TaskInfo] = {}

    def submit(self, task_type: str, filename: str, func: Callable) -> str:
        """Submit a sync callable to run in the ThreadPoolExecutor. Returns task_id."""
        task_id = uuid.uuid4().hex[:12]
        info = TaskInfo(task_id=task_id, task_type=task_type, filename=filename)
        self._tasks[task_id] = info

        loop = asyncio.get_event_loop()
        loop.run_in_executor(self._executor, self._run_task, task_id, func)

        logger.info("task_submitted", task_id=task_id, task_type=task_type, filename=filename)
        return task_id

    def submit_async(self, task_type: str, filename: str, async_func: Callable) -> str:
        """
        Submit an async callable to run as an asyncio.Task on the event loop.

        The callable must have the signature: async def f(progress_callback) -> Any
        It runs on the event loop itself, so it must be truly non-blocking
        (e.g. using asyncio.create_subprocess_exec, not subprocess.run).
        Returns task_id immediately.
        """
        task_id = uuid.uuid4().hex[:12]
        info = TaskInfo(task_id=task_id, task_type=task_type, filename=filename)
        self._tasks[task_id] = info

        asyncio.create_task(self._run_async_task(task_id, async_func))

        logger.info("task_submitted", task_id=task_id, task_type=task_type, filename=filename)
        return task_id

    async def _run_async_task(self, task_id: str, async_func: Callable):
        """Coroutine wrapper that tracks task lifecycle for async submissions."""
        info = self._tasks[task_id]
        info.status = TaskStatus.RUNNING
        info.started_at = datetime.now(timezone.utc).isoformat()

        def progress_callback(completed: int, total: int, current_name: str):
            info.completed_items = completed
            info.total_items = total
            info.current_item = current_name
            info.progress = int((completed / total) * 100) if total > 0 else 0

        try:
            results = await async_func(progress_callback)
            info.status = TaskStatus.COMPLETED
            info.progress = 100
            info.results = results
            logger.info("task_completed", task_id=task_id)
        except Exception as e:
            info.status = TaskStatus.FAILED
            info.error = str(e)
            logger.error("task_failed", task_id=task_id, error=str(e))
        finally:
            info.completed_at = datetime.now(timezone.utc).isoformat()

    def _run_task(self, task_id: str, func: Callable):
        info = self._tasks[task_id]
        info.status = TaskStatus.RUNNING
        info.started_at = datetime.now(timezone.utc).isoformat()

        def progress_callback(completed: int, total: int, current_name: str):
            info.completed_items = completed
            info.total_items = total
            info.current_item = current_name
            info.progress = int((completed / total) * 100) if total > 0 else 0

        try:
            results = func(progress_callback)
            info.status = TaskStatus.COMPLETED
            info.progress = 100
            info.results = results
            logger.info("task_completed", task_id=task_id)
        except Exception as e:
            info.status = TaskStatus.FAILED
            info.error = str(e)
            logger.error("task_failed", task_id=task_id, error=str(e))
        finally:
            info.completed_at = datetime.now(timezone.utc).isoformat()

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def get_tasks_for_file(self, filename: str, task_type: str = None) -> List[TaskInfo]:
        tasks = [t for t in self._tasks.values() if t.filename == filename]
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        return tasks

    def shutdown(self):
        self._executor.shutdown(wait=False)
        logger.info("task_manager_shutdown")


# Module-level singleton
task_manager = TaskManager(max_workers=1)
