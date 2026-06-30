from ai_server.models import AgentManifest
from ai_server.rule_loader import load_rules_for_task
from ai_server.skill_loader import load_skills_for_task


def test_common_index_loader_supports_always_load_and_statuses_for_rules(tmp_path):
    agent_dir = tmp_path / "agent"
    knowledge_dir = agent_dir / "knowledge"
    knowledge_dir.mkdir(parents=True)
    instructions = agent_dir / "instructions.md"
    instructions.write_text("core", encoding="utf-8")
    (knowledge_dir / "always.md").write_text("always chapter", encoding="utf-8")
    (knowledge_dir / "status.md").write_text("status chapter", encoding="utf-8")
    (agent_dir / "rule_index.yaml").write_text(
        """
rules:
  - id: always
    title: Always
    file: knowledge/always.md
    priority: 100
    use_when:
      always_load: true
    load_reason: always
  - id: status
    title: Status
    file: knowledge/status.md
    priority: 90
    use_when:
      statuses:
        - needs_human
    load_reason: status
""",
        encoding="utf-8",
    )
    manifest = AgentManifest(
        id="agent",
        name="Agent",
        kind="specialist",
        description="Test agent",
        instructions_file=str(instructions),
    )

    rules = load_rules_for_task(
        manifest,
        request="plain request",
        statuses=["needs_human"],
    )

    assert [rule.id for rule in rules] == ["always", "status"]
    assert rules[0].match_reasons == ["always_load"]
    assert "needs_human" in rules[1].matched_statuses
    assert "statuses" in rules[1].match_reasons


def test_common_index_loader_supports_always_load_statuses_and_default_file_for_skills(tmp_path):
    agent_dir = tmp_path / "agent"
    skills_dir = agent_dir / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "always.md").write_text("# Always skill\n\nAlways content.", encoding="utf-8")
    (skills_dir / "status.md").write_text("# Status skill\n\nStatus content.", encoding="utf-8")
    (agent_dir / "skill_index.yaml").write_text(
        """
skills:
  - id: always
    priority: 100
    use_when:
      always_load: true
    load_reason: always
  - id: status
    priority: 90
    use_when:
      statuses:
        - needs_human
    load_reason: status
""",
        encoding="utf-8",
    )
    manifest = AgentManifest(
        id="agent",
        name="Agent",
        kind="specialist",
        description="Test agent",
        skills_path=str(skills_dir),
    )

    skills = load_skills_for_task(
        manifest,
        request="plain request",
        statuses=["needs_human"],
    )

    assert [skill.id for skill in skills] == ["always", "status"]
    assert skills[0].file == "skills/always.md"
    assert skills[0].title == "Always skill"
    assert skills[0].match_reasons == ["always_load"]
    assert "needs_human" in skills[1].matched_statuses
    assert "statuses" in skills[1].match_reasons


def test_common_index_loader_uses_fallback_rule_only_when_nothing_matches(tmp_path):
    agent_dir = tmp_path / "agent"
    knowledge_dir = agent_dir / "knowledge"
    knowledge_dir.mkdir(parents=True)
    instructions = agent_dir / "instructions.md"
    instructions.write_text("core", encoding="utf-8")
    (knowledge_dir / "matched.md").write_text("matched chapter", encoding="utf-8")
    (knowledge_dir / "fallback.md").write_text("fallback chapter", encoding="utf-8")
    (agent_dir / "rule_index.yaml").write_text(
        """
rules:
  - id: matched
    title: Matched
    file: knowledge/matched.md
    priority: 100
    use_when:
      request_topics:
        - exact-topic
  - id: fallback
    title: Fallback
    file: knowledge/fallback.md
    priority: 1
    use_when:
      fallback: true
""",
        encoding="utf-8",
    )
    manifest = AgentManifest(
        id="agent",
        name="Agent",
        kind="specialist",
        description="Test agent",
        instructions_file=str(instructions),
    )

    assert [rule.id for rule in load_rules_for_task(manifest, request="exact-topic request")] == ["matched"]

    fallback_rules = load_rules_for_task(manifest, request="unrelated request")

    assert [rule.id for rule in fallback_rules] == ["fallback"]
    assert fallback_rules[0].matched_keywords == []
    assert fallback_rules[0].matched_statuses == []
    assert fallback_rules[0].match_reasons == ["fallback"]


def test_common_index_loader_uses_fallback_skill_only_when_nothing_matches(tmp_path):
    agent_dir = tmp_path / "agent"
    skills_dir = agent_dir / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "matched.md").write_text("# Matched skill\n\nMatched content.", encoding="utf-8")
    (skills_dir / "fallback.md").write_text("# Fallback skill\n\nFallback content.", encoding="utf-8")
    (agent_dir / "skill_index.yaml").write_text(
        """
skills:
  - id: matched
    priority: 100
    use_when:
      request_topics:
        - exact-topic
  - id: fallback
    priority: 1
    use_when:
      fallback: true
""",
        encoding="utf-8",
    )
    manifest = AgentManifest(
        id="agent",
        name="Agent",
        kind="specialist",
        description="Test agent",
        skills_path=str(skills_dir),
    )

    assert [skill.id for skill in load_skills_for_task(manifest, request="exact-topic request")] == ["matched"]

    fallback_skills = load_skills_for_task(manifest, request="unrelated request")

    assert [skill.id for skill in fallback_skills] == ["fallback"]
    assert fallback_skills[0].file == "skills/fallback.md"
    assert fallback_skills[0].matched_keywords == []
    assert fallback_skills[0].matched_statuses == []
    assert fallback_skills[0].match_reasons == ["fallback"]
