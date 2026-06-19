from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol

from apscheduler.job import Job
from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


class SchedulerPort(Protocol):
    def add_job(self, agent_id: str, job_id: str, func: Any, trigger: Any, **kwargs: Any) -> Any: ...

    def remove_jobs_by_prefix(self, agent_id: str, prefix: str) -> int: ...

    def list_jobs(self, agent_id: str) -> list[dict[str, Any]]: ...


class AgentScheduler:
    """
    Thin wrapper around APScheduler's AsyncIOScheduler.

    Job IDs are namespaced as ``{agent_id}:{job_id}`` so each specialist
    can manage its own jobs without collisions.

    The optional ``task_runner`` callback lets the scheduler execute an
    arbitrary specialist task when a scheduled job fires:
        await task_runner(agent_id, task_description, context)
    """

    def __init__(
        self,
        scheduler: AsyncIOScheduler | None = None,
        *,
        task_runner: Callable[[str, str, dict[str, Any]], Awaitable[Any]] | None = None,
    ) -> None:
        self._scheduler = scheduler or AsyncIOScheduler(timezone=MOSCOW_TZ)
        self._task_runner = task_runner

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("AgentScheduler started")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("AgentScheduler stopped")

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_job(
        self,
        agent_id: str,
        job_id: str,
        func: Callable,
        trigger: Any,
        *,
        kwargs: dict[str, Any] | None = None,
        misfire_grace_time: int = 300,
        replace_existing: bool = True,
    ) -> Job:
        full_id = _full_id(agent_id, job_id)
        job = self._scheduler.add_job(
            func,
            trigger,
            id=full_id,
            kwargs=kwargs or {},
            misfire_grace_time=misfire_grace_time,
            replace_existing=replace_existing,
        )
        logger.debug("Scheduled job %s next_run=%s", full_id, job.next_run_time)
        return job

    def remove_job(self, agent_id: str, job_id: str) -> bool:
        full_id = _full_id(agent_id, job_id)
        try:
            self._scheduler.remove_job(full_id)
            logger.debug("Removed job %s", full_id)
            return True
        except JobLookupError:
            return False

    def remove_jobs_by_prefix(self, agent_id: str, prefix: str) -> int:
        """Remove all jobs whose id starts with ``{agent_id}:{prefix}``."""
        namespace = _full_id(agent_id, prefix)
        removed = 0
        for job in self._scheduler.get_jobs():
            if job.id.startswith(namespace):
                try:
                    job.remove()
                    removed += 1
                except JobLookupError:
                    pass
        return removed

    def list_jobs(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        jobs = self._scheduler.get_jobs()
        if agent_id is not None:
            prefix = f"{agent_id}:"
            jobs = [j for j in jobs if j.id.startswith(prefix)]
        return [
            {
                "id": j.id,
                "next_run_time": j.next_run_time.isoformat() if j.next_run_time else None,
                "trigger": str(j.trigger),
            }
            for j in jobs
        ]

    # ------------------------------------------------------------------
    # Scheduled task runner (used by SchedulerSpecialist / p.3)
    # ------------------------------------------------------------------

    async def run_task(self, agent_id: str, task_description: str, context: dict[str, Any] | None = None) -> Any:
        if self._task_runner is None:
            logger.warning("AgentScheduler.run_task called but no task_runner configured")
            return None
        return await self._task_runner(agent_id, task_description, context or {})

    def schedule_task(
        self,
        agent_id: str,
        job_id: str,
        trigger: Any,
        task_description: str,
        context: dict[str, Any] | None = None,
        *,
        replace_existing: bool = True,
    ) -> Job:
        """Schedule a task_runner call for the given agent at trigger time."""
        ctx = context or {}

        async def _run() -> None:
            await self.run_task(agent_id, task_description, ctx)

        return self.add_job(agent_id, job_id, _run, trigger, replace_existing=replace_existing)


def _full_id(agent_id: str, job_id: str) -> str:
    return f"{agent_id}:{job_id}"


def next_run_times(scheduler: AgentScheduler, agent_id: str) -> list[datetime]:
    return [job["next_run_time"] for job in scheduler.list_jobs(agent_id) if job["next_run_time"] is not None]
