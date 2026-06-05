# Runtime var

`var/` is the local runtime state directory for AI Server.

Only this README and empty directory markers are committed. Runtime data is not
stored in Git because it can contain company data, OAuth tokens, dialog history,
search indexes, learning events, attachments and generated documents.

Expected Bitrix runtime paths:

- `search_index.sqlite`
- `search_content/`
- `search_indexer_state.json`
- `webhook_event_queue.sqlite`
- `dialog_state.sqlite`
- `bitrix_oauth.sqlite`
- `bitrix_write_audit.jsonl`
- `quality_control_state.json`
- `supervisor_state.json`
- `vehicle_usage.sqlite`
- `learning_events.jsonl`
- `attachments/`
- `document_drafts/`
- `embedding_models/`
- `tmp/`
- `legacy/`

For cutover from the old `BitrixAIAgent` project, use:

```powershell
uv run python scripts/import_bitrix_var.py --profile cutover
uv run python scripts/import_bitrix_var.py --profile cutover --execute
```

Run the execute step only after the old service is stopped and the target runtime
directory has been backed up or is safe to replace.
