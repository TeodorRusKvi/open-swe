"""Main entry point and CLI loop for Open SWE agent."""
# ruff: noqa: E402

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
# ruff: noqa: E402
import logging
import os
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from langgraph.graph.state import RunnableConfig
from langgraph.pregel import Pregel

warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")

import asyncio

# Suppress Pydantic v1 compatibility warnings from langchain on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

# Now safe to import agent (which imports LangChain modules)
from deepagents import create_deep_agent
from deepagents.backends.protocol import SandboxBackendProtocol
from langsmith.sandbox import SandboxClientError

from agent.integrations.langsmith import _configure_github_proxy
from agent.middleware import (
    ToolErrorMiddleware,
    check_message_queue_before_model,
    open_pr_if_needed,
)
from agent.prompt import construct_system_prompt
from agent.tools import (
    commit_and_open_pr,
    create_pr_review,
    dismiss_pr_review,
    fetch_url,
    get_branch_name,
    get_pr_review,
    github_comment,
    http_request,
    linear_comment,
    linear_create_issue,
    linear_delete_issue,
    linear_get_issue,
    linear_get_issue_comments,
    linear_list_teams,
    linear_update_issue,
    list_pr_review_comments,
    list_pr_reviews,
    list_repos,
    slack_thread_reply,
    submit_pr_review,
    update_pr_review,
    web_search,
)
from agent.utils.auth import resolve_github_token
from agent.utils.github_app import get_github_app_installation_token
from agent.utils.model import make_model
from agent.utils.sandbox import create_sandbox
from agent.utils.sandbox_paths import aresolve_sandbox_work_dir

SANDBOX_CREATING = "__creating__"
SANDBOX_CREATION_TIMEOUT = 180
SANDBOX_POLL_INTERVAL = 1.0
CHECKPOINT_DB_PATH = Path(
    os.environ.get(
        "OPEN_SWE_CHECKPOINT_DB",
        Path(__file__).resolve().parents[4] / ".open_swe_langgraph_checkpoints.sqlite3",
    )
)
CHECKPOINTER_BACKEND = os.environ.get("OPEN_SWE_CHECKPOINTER", "sqlite").strip().lower()
CHECKPOINTER_ASYNC = os.environ.get("OPEN_SWE_CHECKPOINTER_ASYNC", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
from acp_agent_server.checkpointer import get_checkpointer as get_base_checkpointer


async def get_checkpointer():
    """Return a shared LangGraph checkpointer for the Open SWE agent."""
    dsn = os.environ.get("OPEN_SWE_CHECKPOINTER_DSN") or os.environ.get("DATABASE_URL")
    return await get_base_checkpointer(
        db_path=CHECKPOINT_DB_PATH,
        backend=CHECKPOINTER_BACKEND,
        is_async=CHECKPOINTER_ASYNC,
        dsn=dsn,
    )


from agent.utils.sandbox_state import (
    SANDBOX_BACKENDS,
    get_sandbox_id_from_metadata,
    update_thread_metadata,
)


async def _create_sandbox_with_proxy() -> SandboxBackendProtocol:
    """Create a new sandbox with GitHub proxy auth configured.

    Uses create_sandbox (generic factory) so non-langsmith providers still work.
    For langsmith sandboxes, configures the proxy with the installation token.
    """
    sandbox_backend = await asyncio.to_thread(create_sandbox)

    sandbox_type = os.getenv("SANDBOX_TYPE", "langsmith")
    if sandbox_type == "langsmith":
        installation_token = await get_github_app_installation_token()
        if not installation_token:
            msg = "Cannot configure proxy: GitHub App installation token is unavailable"
            logger.error(msg)
            raise ValueError(msg)
        await asyncio.to_thread(_configure_github_proxy, sandbox_backend.id, installation_token)

    return sandbox_backend


async def _refresh_github_proxy(
    sandbox_backend: SandboxBackendProtocol,
) -> None:
    """Refresh GitHub proxy credentials for reused LangSmith sandboxes."""
    if os.getenv("SANDBOX_TYPE", "langsmith") != "langsmith":
        return

    installation_token = await get_github_app_installation_token()
    if not installation_token:
        logger.warning(
            "Skipping GitHub proxy refresh for sandbox %s: installation token unavailable",
            sandbox_backend.id,
        )
        return

    await asyncio.to_thread(_configure_github_proxy, sandbox_backend.id, installation_token)


async def _recreate_sandbox(thread_id: str) -> SandboxBackendProtocol:
    """Recreate a sandbox after a connection failure.

    Clears the stale cache entry, sets the SANDBOX_CREATING sentinel,
    and creates a fresh sandbox (with proxy auth configured).
    The agent is responsible for cloning repos via tools.
    """
    SANDBOX_BACKENDS.pop(thread_id, None)
    update_thread_metadata(thread_id, {"sandbox_id": SANDBOX_CREATING})
    try:
        sandbox_backend = await _create_sandbox_with_proxy()
    except Exception:
        logger.exception("Failed to recreate sandbox after connection failure")
        update_thread_metadata(thread_id, {"sandbox_id": None})
        raise
    return sandbox_backend


async def check_or_recreate_sandbox(
    sandbox_backend: SandboxBackendProtocol, thread_id: str
) -> SandboxBackendProtocol:
    """Check if a cached sandbox is reachable; recreate it if not.

    Pings the sandbox with a lightweight command. If the sandbox is
    unreachable (SandboxClientError), it is torn down and a fresh one
    is created via _recreate_sandbox.

    Returns the original backend if healthy, or a new one if recreated.
    """
    try:
        await asyncio.to_thread(sandbox_backend.execute, "echo ok")
    except SandboxClientError:
        logger.warning(
            "Cached sandbox is no longer reachable for thread %s, recreating",
            thread_id,
        )
        sandbox_backend = await _recreate_sandbox(thread_id)
    return sandbox_backend


async def _wait_for_sandbox_id(thread_id: str) -> str:
    """Wait for sandbox_id to be set in thread metadata.

    Polls thread metadata until sandbox_id is set to a real value
    (not the creating sentinel).

    Raises:
        TimeoutError: If sandbox creation takes too long
    """
    elapsed = 0.0
    while elapsed < SANDBOX_CREATION_TIMEOUT:
        sandbox_id = await get_sandbox_id_from_metadata(thread_id)
        if sandbox_id is not None and sandbox_id != SANDBOX_CREATING:
            return sandbox_id
        await asyncio.sleep(SANDBOX_POLL_INTERVAL)
        elapsed += SANDBOX_POLL_INTERVAL

    msg = f"Timeout waiting for sandbox creation for thread {thread_id}"
    raise TimeoutError(msg)


def graph_loaded_for_execution(config: RunnableConfig) -> bool:
    """Check if the graph is loaded for actual execution vs introspection."""
    return (
        config["configurable"].get("__is_for_execution__", False)
        if "configurable" in config
        else False
    )


DEFAULT_LLM_MODEL_ID = "openai:gpt-4o"
DEFAULT_RECURSION_LIMIT = 1_000
ASK_BEFORE_EDITS_MODE = "ask_before_edits"


async def get_thread_messages(agent: Pregel, thread_id: str) -> list[Any]:
    """Read all messages for a thread by traversing the full history.
    """
    from acp_agent_server import BaseAgentServer
    from acp_agent_server.context import ACPSessionContext

    class DummyServer(BaseAgentServer):
        async def build_agent(self, context: ACPSessionContext, for_execution: bool = True) -> Pregel:
            return agent

    server = DummyServer()
    server._agents[thread_id] = agent
    return await server.get_all_messages(thread_id)


def _build_interrupt_policy(mode: str) -> dict[str, Any] | None:
    if mode != ASK_BEFORE_EDITS_MODE:
        return None
    return {
        "write_file": True,
        "edit_file": True,
        "delete_file": True,
    }




async def rewind_thread_to_message_id(agent: Pregel, thread_id: str, message_id: str) -> str:
    """Destructive rewind in the current thread."""
    checkpoint_id = await resolve_checkpoint_id(agent, thread_id, message_id)
    update_thread_metadata(thread_id, {"checkpoint_id": checkpoint_id})
    return message_id


async def resolve_checkpoint_id(agent: Pregel, thread_id: str, target: str) -> str:
    """Resolve a message ID or ordinal to the checkpoint AFTER that message."""
    res = await _resolve_checkpoint_info(agent, thread_id, target)
    return res["checkpoint_id"]


async def resolve_parent_checkpoint_id(agent: Pregel, thread_id: str, target: str) -> str:
    """Resolve a message ID or ordinal to the checkpoint BEFORE that message."""
    res = await _resolve_checkpoint_info(agent, thread_id, target)
    parent = res.get("parent_checkpoint_id")
    if parent is None:
        # If no parent, it might be the first message, so we return the empty START state
        return "" 
    return parent


async def _resolve_checkpoint_info(agent: Pregel, thread_id: str, target: str) -> dict[str, Any]:
    """Internal helper to find checkpoint and its parent."""
    if not target.strip():
        raise ValueError("Target ID must not be empty")

    all_snapshots = []
    async for snapshot in agent.aget_state_history({"configurable": {"thread_id": thread_id}}):
        all_snapshots.append(snapshot)

    # Ordinal lookup (1-based)
    if target.isdigit():
        idx = int(target)
        user_msg_count = 0
        for snapshot in reversed(all_snapshots):
            messages = (getattr(snapshot, "values", None) or {}).get("messages", [])
            if messages and getattr(messages[-1], "type", None) == "human":
                user_msg_count += 1
                if user_msg_count == idx:
                    return {
                        "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
                        "parent_checkpoint_id": snapshot.parent_config["configurable"].get("checkpoint_id") if snapshot.parent_config else None
                    }

    # ID lookup
    # We want the EARLIEST snapshot that contains this message ID
    for snapshot in reversed(all_snapshots): # Search from oldest to newest
        messages = (getattr(snapshot, "values", None) or {}).get("messages", [])
        for msg in messages:
            if getattr(msg, "id", None) == target:
                return {
                    "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
                    "parent_checkpoint_id": snapshot.parent_config["configurable"].get("checkpoint_id") if snapshot.parent_config else None
                }

    raise ValueError(f"Could not resolve '{target}' in session {thread_id}")


async def get_agent(config: RunnableConfig) -> Pregel:
    """Get or create an agent with a sandbox for the given thread."""
    thread_id = config["configurable"].get("thread_id", None)

    config["recursion_limit"] = DEFAULT_RECURSION_LIMIT

    if thread_id is None or not graph_loaded_for_execution(config):
        logger.info("No thread_id or not for execution, returning agent without sandbox")
        return create_deep_agent(
            system_prompt="",
            tools=[],
            checkpointer=await get_checkpointer(),
        ).with_config(config)

    github_token, new_encrypted = await resolve_github_token(config, thread_id)
    config["metadata"]["github_token_encrypted"] = new_encrypted

    sandbox_backend = SANDBOX_BACKENDS.get(thread_id)
    sandbox_id = await get_sandbox_id_from_metadata(thread_id)

    if sandbox_id == SANDBOX_CREATING and not sandbox_backend:
        logger.info("Sandbox creation in progress, waiting...")
        sandbox_id = await _wait_for_sandbox_id(thread_id)

    if sandbox_backend:
        logger.info("Using cached sandbox backend for thread %s", thread_id)
        await _refresh_github_proxy(sandbox_backend)
        sandbox_backend = await check_or_recreate_sandbox(sandbox_backend, thread_id)

    elif sandbox_id is None:
        logger.info("Creating new sandbox for thread %s", thread_id)
        update_thread_metadata(thread_id, {"sandbox_id": SANDBOX_CREATING})

        try:
            sandbox_backend = await _create_sandbox_with_proxy()
            logger.info("Sandbox created: %s", sandbox_backend.id)
        except Exception:
            logger.exception("Failed to create sandbox")
            try:
                update_thread_metadata(thread_id, {"sandbox_id": None})
                logger.info("Reset sandbox_id to None for thread %s", thread_id)
            except Exception:
                logger.exception("Failed to reset sandbox_id metadata")
            raise
    else:
        logger.info("Connecting to existing sandbox %s", sandbox_id)
        try:
            sandbox_backend = await asyncio.to_thread(create_sandbox, sandbox_id)
            logger.info("Connected to existing sandbox %s", sandbox_id)
        except Exception:
            logger.warning("Failed to connect to existing sandbox %s, creating new one", sandbox_id)
            # Reset sandbox_id and create a new sandbox with proxy auth configured
            update_thread_metadata(thread_id, {"sandbox_id": SANDBOX_CREATING})

            try:
                sandbox_backend = await _create_sandbox_with_proxy()
                logger.info("New sandbox created: %s", sandbox_backend.id)
            except Exception:
                logger.exception("Failed to create replacement sandbox")
                update_thread_metadata(thread_id, {"sandbox_id": None})
                raise

        await _refresh_github_proxy(sandbox_backend)
        sandbox_backend = await check_or_recreate_sandbox(sandbox_backend, thread_id)

    SANDBOX_BACKENDS[thread_id] = sandbox_backend

    if sandbox_id != sandbox_backend.id:
        update_thread_metadata(thread_id, {"sandbox_id": sandbox_backend.id})

        await asyncio.to_thread(
            sandbox_backend.execute,
            "git config --global user.name 'open-swe[bot]' && git config --global user.email 'open-swe@users.noreply.github.com'",
        )

    linear_issue = config["configurable"].get("linear_issue", {})
    linear_project_id = linear_issue.get("linear_project_id", "")
    linear_issue_number = linear_issue.get("linear_issue_number", "")

    work_dir = await aresolve_sandbox_work_dir(sandbox_backend)

    mode = config["configurable"].get("mode", ASK_BEFORE_EDITS_MODE)

    logger.info("Returning agent with sandbox for thread %s", thread_id)
    return create_deep_agent(
        model=make_model(
            os.environ.get("LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID),
            max_tokens=20_000,
        ),
        system_prompt=construct_system_prompt(
            working_dir=work_dir,
            linear_project_id=linear_project_id,
            linear_issue_number=linear_issue_number,
        ),
        tools=[
            http_request,
            fetch_url,
            web_search,
            list_repos,
            get_branch_name,
            commit_and_open_pr,
            linear_comment,
            linear_create_issue,
            linear_delete_issue,
            linear_get_issue,
            linear_get_issue_comments,
            linear_list_teams,
            linear_update_issue,
            slack_thread_reply,
            github_comment,
            list_pr_reviews,
            get_pr_review,
            create_pr_review,
            update_pr_review,
            dismiss_pr_review,
            submit_pr_review,
            list_pr_review_comments,
        ]
        + config["configurable"].get("extra_tools", []),
        backend=sandbox_backend,
        checkpointer=await get_checkpointer(),
        interrupt_on=_build_interrupt_policy(mode),
        middleware=[
            ToolErrorMiddleware(),
            check_message_queue_before_model,
            open_pr_if_needed,
        ],
    ).with_config(config)
