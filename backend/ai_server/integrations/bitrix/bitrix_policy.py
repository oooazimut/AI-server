from ai_server.models import PolicyDecision

READ_METHODS = {
    "batch",
    "app.info",
    "profile",
    "user.current",
    "user.get",
    "user.search",
    "tasks.task.get",
    "tasks.task.list",
    "tasks.task.result.list",
    "task.commentitem.get",
    "task.commentitem.getlist",
    "disk.file.get",
    "disk.folder.get",
    "disk.folder.getchildren",
    "disk.attachedobject.get",
    "disk.storage.getlist",
    "disk.storage.getchildren",
    "sonet_group.get",
    "sonet_group.user.get",
    "crm.lead.get",
    "crm.lead.list",
    "crm.deal.get",
    "crm.deal.list",
    "crm.status.list",
    "catalog.catalog.list",
    "catalog.catalog.get",
    "catalog.product.list",
    "catalog.product.get",
    "catalog.section.list",
    "catalog.section.get",
    "catalog.store.list",
    "catalog.store.get",
    "catalog.storeproduct.list",
    "catalog.measure.list",
    "catalog.price.list",
    "catalog.pricetype.list",
}
READ_SUFFIXES = (".get", ".list", ".search")
WRITE_SUFFIXES = (
    ".add",
    ".create",
    ".update",
    ".delete",
    ".remove",
    ".complete",
    ".approve",
    ".disapprove",
    ".renew",
    ".start",
    ".pause",
    ".delegate",
)
WRITE_PREFIXES_WITH_CONFIRMATION = ("tasks.", "task.", "sonet_group.", "crm.", "disk.", "calendar.", "catalog.")
WRITE_METHODS_WITH_CONFIRMATION = {
    "disk.storage.addfolder",
    "disk.storage.uploadfile",
    "disk.folder.addsubfolder",
    "disk.folder.uploadfile",
}
DENIED_PREFIXES = ("user.", "department.", "humanresources.", "imbot.", "im.", "rest.")
DENIED_METHODS = {"user.add", "user.update", "user.delete", "user.dismiss", "imbot.bot.unregister"}


def decide_bitrix_method_policy(method: str) -> PolicyDecision:
    normalized = method.strip().lower()
    if not normalized:
        return PolicyDecision(decision="deny", reason="empty Bitrix REST method")
    if normalized in DENIED_METHODS:
        return PolicyDecision(decision="deny", reason="method is explicitly denied")
    if normalized.startswith("calendar."):
        if normalized.endswith(WRITE_SUFFIXES):
            return PolicyDecision(decision="confirm", reason="calendar write requires confirmation")
        return PolicyDecision(decision="allow", reason="calendar read")
    if normalized in READ_METHODS:
        return PolicyDecision(decision="allow", reason="known read method")
    if normalized in WRITE_METHODS_WITH_CONFIRMATION:
        return PolicyDecision(decision="confirm", reason="known write method requires confirmation")
    if normalized.startswith(DENIED_PREFIXES):
        return PolicyDecision(decision="deny", reason="method prefix is denied")
    if normalized.endswith(READ_SUFFIXES):
        return PolicyDecision(decision="allow", reason="read-like suffix")
    if normalized.endswith(WRITE_SUFFIXES) and normalized.startswith(WRITE_PREFIXES_WITH_CONFIRMATION):
        return PolicyDecision(decision="confirm", reason="write-like Bitrix method requires confirmation")
    return PolicyDecision(decision="deny", reason="method is not allowlisted")
