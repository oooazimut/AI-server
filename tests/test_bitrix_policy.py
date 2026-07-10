import pytest

from ai_server.tools.bitrix_policy import decide_bitrix_method_policy


def test_empty_method_denied():
    result = decide_bitrix_method_policy("")
    assert result.decision == "deny"


def test_whitespace_only_denied():
    result = decide_bitrix_method_policy("   ")
    assert result.decision == "deny"


# Explicitly denied methods
@pytest.mark.parametrize(
    "method",
    [
        "user.delete",
        "user.update",
        "user.add",
        "user.dismiss",
        "imbot.bot.unregister",
    ],
)
def test_explicitly_denied_methods(method):
    assert decide_bitrix_method_policy(method).decision == "deny"


# Calendar — read vs write
def test_calendar_read_allowed():
    assert decide_bitrix_method_policy("calendar.event.get").decision == "allow"


def test_calendar_list_allowed():
    assert decide_bitrix_method_policy("calendar.event.list").decision == "allow"


def test_calendar_write_requires_confirmation():
    assert decide_bitrix_method_policy("calendar.event.add").decision == "confirm"


def test_calendar_update_requires_confirmation():
    assert decide_bitrix_method_policy("calendar.event.update").decision == "confirm"


# Known read methods
@pytest.mark.parametrize(
    "method",
    [
        "profile",
        "user.current",
        "user.get",
        "user.search",
        "tasks.task.get",
        "tasks.task.list",
        "disk.file.get",
        "crm.lead.get",
        "crm.deal.list",
    ],
)
def test_known_read_methods_allowed(method):
    assert decide_bitrix_method_policy(method).decision == "allow"


# Known write methods requiring confirmation
@pytest.mark.parametrize(
    "method",
    [
        "disk.storage.addfolder",
        "disk.storage.uploadfile",
        "disk.folder.addsubfolder",
        "disk.folder.uploadfile",
    ],
)
def test_known_write_methods_require_confirmation(method):
    assert decide_bitrix_method_policy(method).decision == "confirm"


# Denied prefixes
@pytest.mark.parametrize(
    "method",
    [
        "user.makeAdmin",
        "department.get",
        "humanresources.employee.list",
        "imbot.message.send",
        "im.chat.get",
        "rest.app.get",
    ],
)
def test_denied_prefix_methods_denied(method):
    assert decide_bitrix_method_policy(method).decision == "deny"


# Read-like suffix
@pytest.mark.parametrize(
    "method",
    [
        "crm.contact.list",
        "crm.activity.search",
        "sonet_group.get",
    ],
)
def test_read_like_suffix_allowed(method):
    assert decide_bitrix_method_policy(method).decision == "allow"


# Write-like suffix with confirmation-required prefix
@pytest.mark.parametrize(
    "method",
    [
        "tasks.task.add",
        "tasks.task.update",
        "tasks.task.delete",
        "crm.deal.update",
        "crm.lead.create",
        "sonet_group.user.add",
        "disk.folder.delete",
    ],
)
def test_write_like_suffix_requires_confirmation(method):
    assert decide_bitrix_method_policy(method).decision == "confirm"


# Unknown / not allowlisted
def test_unknown_method_denied():
    assert decide_bitrix_method_policy("someapi.customthing").decision == "deny"


def test_unsupported_bitrix_search_method_denied():
    assert decide_bitrix_method_policy("search.search").decision == "deny"


def test_method_with_no_dot_denied():
    assert decide_bitrix_method_policy("getcustomdata").decision == "deny"


# Case insensitive
def test_case_insensitive_read():
    assert decide_bitrix_method_policy("Tasks.Task.Get").decision == "allow"


def test_case_insensitive_write():
    assert decide_bitrix_method_policy("TASKS.TASK.ADD").decision == "confirm"


# Whitespace stripped
def test_whitespace_stripped():
    assert decide_bitrix_method_policy("  profile  ").decision == "allow"
