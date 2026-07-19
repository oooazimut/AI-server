import asyncio

from ai_server.orchestrators.local_harness import LocalOrchestratorHarness


def run(case_id, request):
    return asyncio.run(LocalOrchestratorHarness().run_case(case_id, request))


def test_only_bitrix_is_a_hard_boundary():
    result = run("Q05", "Только Bitrix: найди склад Борисова")
    assert result.route == ["bitrix"]
    assert result.executor_calls == 1


def test_not_supported_has_no_executor_calls():
    result = run("Q10", "Какая сейчас погода")
    assert result.verdict == "NOT_SUPPORTED"
    assert result.executor_calls == 0


def test_parallel_fixture_gap_is_partial_and_does_not_abort():
    result = run("Q03", "Найди три склада и остатки")
    assert result.parallel is True
    assert result.verdict == "PARTIAL"
    assert result.executor_calls == 3
    assert [branch.status for branch in result.branches].count("not_found") == 2


def test_loop_guard_caps_round_trips():
    result = run("Q13", "Найди склад и проверь доставку")
    assert result.round_trips == 3
    assert result.verdict == "PARTIAL"
    assert result.executor_calls == 6
    assert all(branch.status == "not_mine" for branch in result.branches)


def test_contexts_are_isolated_and_duplicate_is_suppressed():
    harness = LocalOrchestratorHarness()
    q15 = asyncio.run(harness.run_case("Q15", "Two simultaneous users"))
    q16 = asyncio.run(harness.run_case("Q16", "Late duplicate executor result"))
    assert q15.verdict == "PASS"
    assert harness.contexts["chat:a:user:1"] != harness.contexts["chat:b:user:2"]
    assert "duplicate_accepted=False" in q16.notes[1]
    assert q16.correlation_ids["suppressed_attempt_id"] == q16.branches[0].attempt_id
