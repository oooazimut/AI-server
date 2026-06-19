from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class WebhookEventStatus(TypedDict):
    enabled: bool
    mode: str
    webhook_url_configured: bool
    secret_required: bool
    last_received_at: str | None
    last_event: str | None
    events_seen: int
    duplicates_seen: int


class WebhookEventQueueStatus(TypedDict):
    enabled: bool
    running: bool
    path: str
    worker_enabled: bool
    worker_count: int
    last_enqueued_at: str | None
    last_enqueued_event_id: str | None
    last_enqueued_event: str | None
    enqueued: int
    duplicates_seen: int
    processed: int
    errors: int
    last_error: str | None
    # set by worker loop
    last_check_at: str | None
    last_event_id: int | None
    last_event: str | None
    last_worker_id: int | None
    active_workers: int
    active_partition_keys: list[str]


class SearchWebhookIndexerStatus(TypedDict):
    enabled: bool
    events_seen: int
    processed: int
    errors: int
    last_received_at: str | None
    last_event: str | None
    last_file_id: int | None
    last_action: str | None
    last_reason: str | None
    last_error: str | None
    last_result: dict[str, Any] | None


class QualityControlWebhookStatus(TypedDict):
    enabled: bool
    auto_managed_only: bool
    auto_manage_project_id: int | None
    dry_run: bool
    actor_user_id: int | None
    last_received_at: str | None
    last_event: str | None
    last_task_id: str | None
    last_reason: str | None
    last_error: str | None
    last_actions: list[Any]
    events_seen: int
    tasks_processed: int
    duplicates_seen: int
    ignored: int
    errors: int


class TaskSupervisorStatus(TypedDict):
    enabled: bool
    running: bool
    dry_run: bool
    interval_seconds: int
    last_check_at: str | None
    last_success_at: str | None
    last_error: str | None
    last_result: dict[str, Any] | None
    next_check_at: str | None
    runs: int
    errors: int


class ReconcilerStatus(TypedDict):
    enabled: bool
    running: bool
    interval_seconds: int
    task_lookback_hours: int
    last_check_at: str | None
    last_success_at: str | None
    last_error: str | None
    last_result: dict[str, Any] | None
    next_check_at: str | None
    runs: int
    errors: int


class VehicleUsageStatus(TypedDict):
    enabled: bool
    running: bool
    dry_run: bool
    interval_seconds: int
    dialog_id: str
    manager_user_id: int | None
    admin_notify_user_ids: list[int]
    request_time: str
    db_path: str
    last_check_at: str | None
    last_sent_at: str | None
    last_escalated_at: str | None
    last_error: str | None
    runs: int
    errors: int
    last_skipped_date: str | None
    last_skip_reason: str | None
    last_result: dict[str, Any] | None
    last_delivery: str | None
    last_request_date: str | None
    last_reminder_number: int | None
