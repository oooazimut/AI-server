from __future__ import annotations

import anyio

from ai_server.agents.bitrix24.tools.tasks import BitrixProjectSearchTool, BitrixTaskSearchTool


class _FakeBitrixSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.comments: dict[str, list[dict]] = {}
        self.tasks = [
            {
                "id": "101",
                "title": "Ответственная задача",
                "status": "2",
                "responsibleId": "13",
                "createdBy": "9",
                "createdDate": "2026-06-01T09:00:00+03:00",
                "deadline": "2026-07-10T19:00:00+03:00",
                "groupId": "45",
            },
            {
                "id": "102",
                "title": "Поставленная мной",
                "status": "3",
                "responsibleId": "9",
                "createdBy": "13",
                "createdDate": "2026-06-15T09:00:00+03:00",
                "deadline": "2026-07-09T19:00:00+03:00",
                "groupId": "45",
            },
            {
                "id": "103",
                "title": "Наблюдение",
                "status": "2",
                "responsibleId": "9",
                "createdBy": "9",
                "auditors": ["13"],
                "createdDate": "2026-06-20T09:00:00+03:00",
                "deadline": "2026-07-08T19:00:00+03:00",
                "groupId": "39",
            },
            {
                "id": "104",
                "title": "Закрытая задача",
                "status": "5",
                "responsibleId": "13",
                "createdBy": "13",
                "createdDate": "2026-05-20T09:00:00+03:00",
                "closedDate": "2026-07-08T10:00:00+03:00",
                "deadline": "2026-07-07T19:00:00+03:00",
                "groupId": "45",
            },
            {
                "id": "139",
                "title": "Обучение сотрудников",
                "status": "2",
                "responsibleId": "35",
                "createdBy": "35",
                "createdDate": "2026-06-25T09:00:00+03:00",
                "deadline": None,
                "groupId": "45",
            },
        ]

    async def result(self, method: str, params: dict):
        self.calls.append((method, params))
        if method == "tasks.task.get":
            task_id = str(params.get("taskId"))
            return {"task": next((task for task in self.tasks if str(task["id"]) == task_id), None)}
        if method == "tasks.task.list":
            task_filter = params.get("filter") or {}
            return {"tasks": [task for task in self.tasks if _matches_task_filter(task, task_filter)]}
        if method == "task.commentitem.getlist":
            return {"comments": self.comments.get(str(params.get("TASKID")), [])}
        if method == "sonet_group.get" and (params.get("FILTER") or {}).get("%NAME"):
            return []
        if method == "sonet_group.get":
            return [
                {"ID": "39", "NAME": "Логан"},
                {"ID": "45", "NAME": "Ларгус 2"},
                {"ID": "53", "NAME": "Ларгус 3"},
            ]
        return []


def test_task_search_responsible_scope_uses_current_user_and_active_statuses():
    client = _FakeBitrixSearchClient()
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"scope": "responsible"}, user_id=13))

    assert result.status == "ok"
    assert [item["title"] for item in result.data["items"]] == ["Ответственная задача"]
    assert result.data["total"] == 1
    assert client.calls[0][1]["filter"] == {"STATUS": [1, 2, 3, 4], "RESPONSIBLE_ID": 13}


def test_task_search_status_all_requires_explicit_include_closed_flag():
    client = _FakeBitrixSearchClient()
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"scope": "created_by", "status": "all"}, user_id=13))

    assert result.status == "ok"
    assert [item["title"] for item in result.data["items"]] == ["Поставленная мной"]
    assert result.data["status"] == "active"
    assert client.calls[0][1]["filter"] == {"STATUS": [1, 2, 3, 4], "CREATED_BY": 13}


def test_task_search_text_query_uses_title_filter_for_all_scope():
    client = _FakeBitrixSearchClient()
    client.tasks.append(
        {
            "id": "140",
            "title": "Обучение сотрудников",
            "status": "5",
            "responsibleId": "35",
            "createdBy": "35",
            "deadline": None,
            "groupId": "45",
        }
    )
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"scope": "all", "query": "Обучение сотрудников"}, user_id=13))

    assert result.status == "ok"
    assert [item["title"] for item in result.data["items"]] == ["Обучение сотрудников"]
    assert result.data["total"] == 1
    assert client.calls[0][1]["filter"] == {"%TITLE": "Обучение сотрудников"}


def test_task_search_defaults_to_ten_and_reports_more_after_sorting():
    client = _FakeBitrixSearchClient()
    client.tasks = [
        {
            "id": str(200 + index),
            "title": f"Задача {index}",
            "status": "2",
            "responsibleId": "13",
            "createdBy": "9",
            "createdDate": "2026-06-01T09:00:00+03:00",
            "deadline": f"2026-07-{index + 1:02d}T19:00:00+03:00",
        }
        for index in range(12)
    ]
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"scope": "responsible"}, user_id=13))

    assert len(result.data["items"]) == 10
    assert result.data["total"] == 12
    assert result.data["has_more"] is True


def test_task_search_can_match_query_in_comments_without_title_filter():
    client = _FakeBitrixSearchClient()
    client.comments["101"] = [{"POST_MESSAGE": "В комментарии есть фраза шкаф автоматики."}]
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(
        lambda: tool.execute({"scope": "responsible", "query": "шкаф автоматики", "include_comments": True}, user_id=13)
    )

    assert result.status == "ok"
    assert [item["title"] for item in result.data["items"]] == ["Ответственная задача"]
    assert result.data["items"][0]["matched_comment_count"] == 1
    assert "шкаф автоматики" in result.data["items"][0]["comment_snippets"][0]
    task_call = next(payload for method, payload in client.calls if method == "tasks.task.list")
    assert "%TITLE" not in task_call["filter"]
    assert ("task.commentitem.getlist", {"TASKID": "101"}) in client.calls


def test_task_search_comment_query_requires_comment_match():
    client = _FakeBitrixSearchClient()
    client.comments["101"] = [{"POST_MESSAGE": "Неподходящий комментарий"}]
    client.comments["102"] = [{"POST_MESSAGE": "Нужен акт сверки по задаче"}]
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"scope": "my", "comment_query": "акт сверки"}, user_id=13))

    assert result.status == "ok"
    assert [item["title"] for item in result.data["items"]] == ["Поставленная мной"]
    assert result.data["comment_query"] == "акт сверки"


def test_task_search_comment_snippets_hide_bitrix_user_markup():
    client = _FakeBitrixSearchClient()
    client.comments["101"] = [{"POST_MESSAGE": "Before [USER=13]Dmitry[/USER], added observer."}]
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"scope": "responsible", "comment_query": "observer"}, user_id=13))

    assert result.status == "ok"
    snippet = result.data["items"][0]["comment_snippets"][0]
    assert snippet == "Before Dmitry, added observer."
    assert "[USER=13]" not in snippet
    assert "[/USER]" not in snippet


def test_task_search_filters_dates_before_loading_comments():
    client = _FakeBitrixSearchClient()
    client.comments["104"] = [{"POST_MESSAGE": "closed observer comment"}]
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(
        lambda: tool.execute(
            {
                "scope": "all",
                "status": "closed",
                "include_closed": True,
                "closed_from": "2026-07-08",
                "closed_to": "2026-07-08",
                "comment_query": "observer",
                "comment_lookup_task_limit": 50,
            },
            user_id=13,
        )
    )

    assert result.status == "ok"
    assert [item["id"] for item in result.data["items"]] == ["104"]
    comment_calls = [payload for method, payload in client.calls if method == "task.commentitem.getlist"]
    assert comment_calls == [{"TASKID": "104"}]


def test_task_search_closed_date_range_includes_date_only_upper_bound():
    client = _FakeBitrixSearchClient()
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(
        lambda: tool.execute(
            {
                "scope": "all",
                "status": "closed",
                "include_closed": True,
                "closed_from": "2026-07-08",
                "closed_to": "2026-07-08",
            },
            user_id=13,
        )
    )

    assert result.status == "ok"
    assert [item["title"] for item in result.data["items"]] == ["Закрытая задача"]
    assert result.data["items"][0]["closed_date"] == "2026-07-08T10:00:00+03:00"


def test_task_search_resolves_hyphenated_project_name_before_task_lookup():
    client = _FakeBitrixSearchClient()
    tool = BitrixTaskSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"scope": "my", "project_name": "Ларгус-2"}, user_id=13))

    assert result.status == "ok"
    assert result.data["project"] == {"id": "45", "name": "Ларгус 2", "description": ""}
    assert [item["title"] for item in result.data["items"]] == ["Поставленная мной", "Ответственная задача"]
    task_calls = [payload for method, payload in client.calls if method == "tasks.task.list"]
    assert task_calls[0]["filter"] == {"STATUS": [1, 2, 3, 4], "GROUP_ID": 45, "MEMBER": 13}
    assert task_calls[1]["filter"] == {"STATUS": [1, 2, 3, 4], "GROUP_ID": 45, "CREATED_BY": 13}


def test_project_search_resolves_hyphenated_project_name():
    client = _FakeBitrixSearchClient()
    tool = BitrixProjectSearchTool(client=client)

    result = anyio.run(lambda: tool.execute({"query": "Ларгус-2"}, user_id=13))

    assert result.status == "ok"
    assert result.data["items"] == [{"id": "45", "name": "Ларгус 2", "description": ""}]
    assert client.calls == [
        ("sonet_group.get", {"FILTER": {"%NAME": "Ларгус-2"}, "ORDER": {"NAME": "ASC"}}),
        ("sonet_group.get", {"FILTER": {}, "ORDER": {"NAME": "ASC"}}),
    ]


def _matches_task_filter(task: dict, task_filter: dict) -> bool:
    statuses = task_filter.get("STATUS")
    if isinstance(statuses, list) and int(task["status"]) not in statuses:
        return False
    if isinstance(statuses, int) and int(task["status"]) != statuses:
        return False
    if task_filter.get("RESPONSIBLE_ID") and str(task["responsibleId"]) != str(task_filter["RESPONSIBLE_ID"]):
        return False
    if task_filter.get("CREATED_BY") and str(task["createdBy"]) != str(task_filter["CREATED_BY"]):
        return False
    if task_filter.get("GROUP_ID") and str(task.get("groupId")) != str(task_filter["GROUP_ID"]):
        return False
    if task_filter.get("%TITLE") and str(task_filter["%TITLE"]).casefold() not in str(task.get("title")).casefold():
        return False
    if task_filter.get("MEMBER"):
        user_id = str(task_filter["MEMBER"])
        if user_id not in {
            str(task.get("responsibleId")),
            str(task.get("createdBy")),
            *(str(item) for item in task.get("accomplices", [])),
            *(str(item) for item in task.get("auditors", [])),
        }:
            return False
    return True
