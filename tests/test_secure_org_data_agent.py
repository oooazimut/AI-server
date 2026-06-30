import asyncio
import json

from ai_server.agents.secure_org_data import SecureOrgDataLLMService, SecureOrgDataStore
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest, load_agent_manifests
from ai_server.specialists import build_specialist_registry
from tests.fakes import RecordingLLMClient


def test_secure_org_data_manifest_exists_and_is_employee_audience():
    manifest = get_agent_manifest("secure_org_data")

    assert manifest is not None
    assert manifest.kind == "specialist"
    assert manifest.audience == "employee"
    assert manifest.instructions_file == "agents/secure_org_data/instructions.md"
    assert manifest.knowledge_path == "agents/secure_org_data/knowledge"
    assert "search_org_data" in manifest.tools


def test_employee_specialist_registry_includes_secure_org_data():
    manifests = load_agent_manifests()

    specialists = build_specialist_registry(manifests, audience="employee")

    assert "secure_org_data" in specialists


def test_secure_org_data_llm_payload_includes_loaded_access_rules():
    client = RecordingLLMClient(
        '{"status":"completed","answer":"Поищу в базе.","confidence":0.7,'
        '"tool_calls":[{"name":"search_org_data","args":{"query":"пароль АЯКС","limit":5},"summary":"поиск"}]}'
    )
    manifest = get_agent_manifest("secure_org_data")

    result = asyncio.run(
        SecureOrgDataLLMService(client).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="secure-1",
                request="Найди пароль от АЯКС",
            ),
            retrieval_hits=[],
            tool_definitions=[],
        )
    )

    system_prompt = client.calls[0]["messages"][0]["content"]
    payload = json.loads(client.calls[0]["messages"][1]["content"])

    assert "Secure Org Data Agent" in system_prompt
    assert any(rule["id"] == "access_model" for rule in payload["loaded_rules"])
    assert any(rule["id"] == "access_model" for rule in result.raw["loaded_rules"])
    assert result.decision.tool_calls[0].name == "search_org_data"


def test_secure_org_data_store_uses_explicit_access_markers(tmp_path):
    metadata_dir = tmp_path / "kb_data"
    index_dir = metadata_dir / "content_index"
    index_dir.mkdir(parents=True)
    _write_jsonl(
        index_dir / "stage1_open_chunks.jsonl",
        [
            {
                "path": "objects/ajax/open.txt",
                "title": "Открытая инструкция",
                "text": "Открытый пароль alpha123 для тестового устройства.",
            },
            {
                "path": "objects/ajax/secret.txt",
                "title": "Явно секретный файл",
                "text": "secret-token-123",
            },
        ],
    )
    _write_jsonl(
        index_dir / "stage1_protected_chunks.jsonl",
        [
            {
                "path": "objects/ajax/protected.txt",
                "title": "Закрытая инструкция",
                "access": "restricted_review",
                "text": "protected server settings",
            },
            {
                "relativePath": "objects/ajax/secret_from_index.txt",
                "name": "Секрет из индекса",
                "access": "secret",
                "text": "index-secret-456",
            }
        ],
    )
    (metadata_dir / "file_access_overrides.json").write_text(
        json.dumps({"objects/ajax/secret.txt": "secret"}, ensure_ascii=False),
        encoding="utf-8",
    )
    store = SecureOrgDataStore(
        metadata_dir=metadata_dir,
        protected_user_ids="7",
        secret_user_ids="",
    )

    open_result = store.search("пароль", user_id=None)
    protected_denied = store.search("server", user_id=None)
    protected_allowed = store.search("server", user_id=7)
    secret_denied = store.search("secret-token", user_id=7)
    secret_from_index_denied = store.search("index-secret", user_id=7)

    assert open_result["results"][0]["title"] == "Открытая инструкция"
    assert "alpha123" in open_result["results"][0]["snippet"]
    assert protected_denied["results"] == []
    assert protected_denied["access"]["denied_counts"]["protected"] == 1
    assert protected_allowed["results"][0]["title"] == "Закрытая инструкция"
    assert secret_denied["results"] == []
    assert secret_denied["access"]["denied_counts"]["secret"] == 1
    assert secret_from_index_denied["results"] == []
    assert secret_from_index_denied["access"]["denied_counts"]["secret"] == 1


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
