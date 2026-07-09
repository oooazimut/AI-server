from __future__ import annotations

import anyio

from ai_server.agents.bitrix24.tools.tasks import BitrixMyTasksTool


class _FakeTaskClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def result(self, method: str, params: dict):
        self.calls.append((method, params))
        task_filter = params["filter"]
        if "MEMBER" in task_filter:
            return {
                "tasks": [
                    {
                        "id": "3",
                        "title": "Без срока",
                        "status": "2",
                        "responsibleId": "9",
                        "createdBy": "1",
                        "deadline": None,
                    },
                    {
                        "id": "1",
                        "title": "Срочная",
                        "status": "3",
                        "responsibleId": "9",
                        "createdBy": "9",
                        "deadline": "2026-07-09T19:00:00+03:00",
                    },
                    {
                        "id": "2",
                        "title": "Соисполнитель",
                        "status": "2",
                        "responsibleId": "15",
                        "createdBy": "1",
                        "accomplices": ["9"],
                        "deadline": "2026-07-10T19:00:00+03:00",
                    },
                ]
            }
        return {
            "tasks": [
                {
                    "id": "1",
                    "title": "Срочная",
                    "status": "3",
                    "responsibleId": "9",
                    "createdBy": "9",
                    "deadline": "2026-07-09T19:00:00+03:00",
                },
                {
                    "id": "4",
                    "title": "Поставленная мной",
                    "status": "2",
                    "responsibleId": "15",
                    "createdBy": "9",
                    "deadline": "2026-07-08T19:00:00+03:00",
                },
            ]
        }


class _FakeBitrixOAuth:
    def __init__(self, client: _FakeTaskClient) -> None:
        self.client = client
        self.user_ids: list[int] = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client


def test_my_tasks_merges_created_and_member_tasks_sorted_by_deadline():
    client = _FakeTaskClient()
    tool = BitrixMyTasksTool(client=client)

    result = anyio.run(lambda: tool.execute({"status": "open", "limit": 10}, user_id=9))

    assert result.status == "ok"
    assert [item["title"] for item in result.data["items"]] == [
        "Поставленная мной",
        "Срочная",
        "Соисполнитель",
        "Без срока",
    ]
    assert result.data["total"] == 4
    assert result.data["items"][0]["roles"] == ["постановщик"]
    assert result.data["items"][1]["roles"] == ["исполнитель", "постановщик"]
    assert result.data["items"][2]["roles"] == ["соисполнитель"]
    assert client.calls[0][1]["filter"] == {"STATUS": [1, 2, 3, 4], "MEMBER": 9}
    assert client.calls[1][1]["filter"] == {"STATUS": [1, 2, 3, 4], "CREATED_BY": 9}


def test_my_tasks_uses_oauth_client_for_current_user_reads():
    fallback_client = _FakeTaskClient()
    oauth_client = _FakeTaskClient()
    oauth = _FakeBitrixOAuth(oauth_client)
    tool = BitrixMyTasksTool(client=fallback_client, bitrix_oauth=oauth)

    result = anyio.run(lambda: tool.execute({"status": "open", "limit": 10}, user_id=9))

    assert result.status == "ok"
    assert result.data["access_actor"] == "oauth_current_user"
    assert oauth.user_ids == [9]
    assert fallback_client.calls == []
    assert oauth_client.calls[0][1]["filter"] == {"STATUS": [1, 2, 3, 4], "MEMBER": 9}
