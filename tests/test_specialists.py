import ai_server.specialists as _specialists_module
from ai_server.models import AgentManifest
from ai_server.registry import load_agent_manifests
from ai_server.specialists import build_specialist_registry, manifest_by_id


class _FakeSpecialist:
    @classmethod
    def build(cls, manifest, **deps):
        inst = cls()
        inst.manifest = manifest
        return inst

    async def handle(self, task):
        pass


# manifest_by_id
def test_manifest_by_id_found():
    manifests = load_agent_manifests()
    result = manifest_by_id(manifests, "bitrix24")
    assert result is not None
    assert result.id == "bitrix24"


def test_manifest_by_id_not_found():
    manifests = load_agent_manifests()
    assert manifest_by_id(manifests, "nonexistent_agent") is None


def test_manifest_by_id_empty_list():
    assert manifest_by_id([], "bitrix24") is None


def test_manifest_by_id_with_fake_manifests():
    manifests = [
        AgentManifest(id="alpha", name="Alpha", kind="specialist", description="A"),
        AgentManifest(id="beta", name="Beta", kind="orchestrator", description="B"),
    ]
    assert manifest_by_id(manifests, "alpha").id == "alpha"
    assert manifest_by_id(manifests, "beta").id == "beta"
    assert manifest_by_id(manifests, "gamma") is None


# build_specialist_registry — audience filter
# Uses monkeypatch to bypass entrypoint loading (some manifests are placeholders with no real module).
def test_build_specialist_registry_employee_audience_contains_known_specialists(monkeypatch):
    monkeypatch.setattr(_specialists_module, "_load_entrypoint", lambda ep: _FakeSpecialist)
    manifests = load_agent_manifests()
    registry = build_specialist_registry(manifests, audience="employee")
    assert "bitrix24" in registry
    assert "pto" in registry
    assert "logistics" in registry


def test_build_specialist_registry_customer_audience_returns_empty():
    # No manifests have audience="customer" — no entrypoints loaded, safe without patch
    manifests = load_agent_manifests()
    registry = build_specialist_registry(manifests, audience="customer")
    assert registry == {}


def test_build_specialist_registry_no_audience_filter_includes_internal_specialists(monkeypatch):
    monkeypatch.setattr(_specialists_module, "_load_entrypoint", lambda ep: _FakeSpecialist)
    manifests = load_agent_manifests()
    without_filter = build_specialist_registry(manifests)
    with_employee = build_specialist_registry(manifests, audience="employee")
    assert set(with_employee.keys()).issubset(set(without_filter.keys()))
    assert "diagnostic_agent" in without_filter
    assert "diagnostic_agent" not in with_employee


def test_build_specialist_registry_skips_non_specialists(monkeypatch):
    monkeypatch.setattr(_specialists_module, "_load_entrypoint", lambda ep: _FakeSpecialist)
    manifests = load_agent_manifests()
    registry = build_specialist_registry(manifests)
    for manifest in manifests:
        if manifest.kind != "specialist":
            assert manifest.id not in registry


def test_build_specialist_registry_skips_manifests_without_entrypoint():
    manifests = [
        AgentManifest(
            id="no_entry",
            name="No Entry",
            kind="specialist",
            description="Has no entrypoint",
            entrypoint=None,
        ),
    ]
    registry = build_specialist_registry(manifests)
    assert "no_entry" not in registry
