import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

# Import from the absolute path within the workspace
import agent.agent_acp
from agent.agent_acp import SertiAgentServerACP
print(f"DEBUG: SertiAgentServerACP file: {agent.agent_acp.__file__}")
from acp.schema import ForkSessionResponse


@pytest.mark.asyncio
async def test_fork_session_lifecycle():
    """Tests the full lifecycle of forking a session including checkpoint resolution."""

    # 1. Setup Mocks
    mock_conn = AsyncMock()
    agent_server = SertiAgentServerACP(mock_conn)

    original_session_id = "orig-session-123"
    checkpoint_id = "check-456"
    new_thread_id = "thread-789"

    original_session_data = {"id": original_session_id, "cwd": "/test/path", "mcp_servers": []}

    # Mock the internal compiled agent getter
    mock_agent = MagicMock()
    agent_server._get_compiled_agent = AsyncMock(return_value=mock_agent)

    # Correct Patching of source modules (since imports are local in agent_acp.py)
    with (
        patch("agent.utils.sandbox_state.get_acp_session", return_value=original_session_data),
        patch("agent.utils.sandbox_state.persist_acp_session") as mock_persist,
        patch(
            "agent.utils.sandbox_state.get_thread_metadata",
            return_value={"checkpoint_id": "latest-check"},
        ),
        patch("agent.utils.sandbox_state.update_thread_metadata") as mock_update_meta,
        patch("agent.server.resolve_checkpoint_id", new_callable=AsyncMock) as mock_resolve,
        patch("agent.server.fork_thread", new_callable=AsyncMock) as mock_fork,
    ):
        mock_resolve.return_value = checkpoint_id
        mock_fork.return_value = new_thread_id

        # 2. Execute Fork
        response = await agent_server.fork_session(
            cwd="/test/path", session_id=original_session_id, checkpoint_id=checkpoint_id
        )

        # 3. Verify Response
        assert isinstance(response, ForkSessionResponse)
        assert response.session_id is not None

        # 4. Verify Logic Steps
        mock_resolve.assert_called_once()
        mock_fork.assert_called_once_with(mock_agent, original_session_id, checkpoint_id)
        mock_persist.assert_called_once()
        mock_update_meta.assert_called_once()


@pytest.mark.asyncio
async def test_fork_session_fallback_logic():
    """Tests that forking falls back to the latest checkpoint if no ID is provided."""

    mock_conn = AsyncMock()
    agent_server = SertiAgentServerACP(mock_conn)

    original_session_id = "orig-session-fallback"
    latest_check = "latest-check-999"

    original_session_data = {"id": original_session_id}

    with (
        patch("agent.utils.sandbox_state.get_acp_session", return_value=original_session_data),
        patch("agent.utils.sandbox_state.persist_acp_session"),
        patch(
            "agent.utils.sandbox_state.get_thread_metadata",
            return_value={"checkpoint_id": latest_check},
        ),
        patch("agent.utils.sandbox_state.update_thread_metadata"),
        patch("agent.server.fork_thread", new_callable=AsyncMock) as mock_fork,
    ):
        mock_fork.return_value = "new-thread"

        # Fork WITHOUT checkpoint_id
        await agent_server.fork_session(cwd="/test", session_id=original_session_id)
        # Verify it used the latest checkpoint from metadata
        mock_fork.assert_called_once()
        args, _ = mock_fork.call_args
        assert args[2] == latest_check


if __name__ == "__main__":
    asyncio.run(test_fork_session_lifecycle())
    asyncio.run(test_fork_session_fallback_logic())
    print("All ACP rewind/fork tests passed!")
