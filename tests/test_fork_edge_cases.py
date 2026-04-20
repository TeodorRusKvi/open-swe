import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4
from dataclasses import dataclass
from typing import Any, List

# Generic Universal ACP Agent Server framework
from acp.schema import ForkSessionResponse
try:
    from packages.open_swe.agent.agent_acp import OpenSWEAgentServer
except ImportError:
    from agent.agent_acp import OpenSWEAgentServer

@dataclass
class MockSnapshot:
    config: dict
    parent_config: dict
    values: dict
    next: List[str] = None

class MockAgent:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.aupdate_state = AsyncMock()

    async def aget_state_history(self, config):
        for s in self.snapshots:
            yield s

    async def aget_state(self, config):
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not checkpoint_id:
            return self.snapshots[0] # Return newest if not specified
        for s in self.snapshots:
            if s.config["configurable"]["checkpoint_id"] == checkpoint_id:
                return s
        return None

@pytest.mark.asyncio
async def test_fork_during_tool_execution():
    """
    Tests forking a session where the latest state has a tool call.
    Verifies that the fork-at-parent logic correctly resolves to the state BEFORE the tool call.
    """
    
    # 1. Setup History
    # Snapshot 1: User message
    # Snapshot 2: Assistant message with tool call (Current Tip)
    
    msg_user_id = "msg-user-1"
    msg_assistant_id = "msg-assistant-1"
    
    snapshot_1 = MockSnapshot(
        config={"configurable": {"thread_id": "orig", "checkpoint_id": "check-1"}},
        parent_config=None,
        values={"messages": [{"id": msg_user_id, "type": "human", "content": "Hello"}]},
        next=["agent"]
    )
    
    snapshot_2 = MockSnapshot(
        config={"configurable": {"thread_id": "orig", "checkpoint_id": "check-2"}},
        parent_config={"configurable": {"thread_id": "orig", "checkpoint_id": "check-1"}},
        values={
            "messages": [
                {"id": msg_user_id, "type": "human", "content": "Hello"},
                {"id": msg_assistant_id, "type": "ai", "content": "", "tool_calls": [{"id": "call-1", "name": "test_tool"}]}
            ]
        },
        next=["tools"] # Indicates it's about to run tools
    )
    
    # Snapshots are returned newest first by LangGraph
    mock_agent = MockAgent([snapshot_2, snapshot_1])
    
    # 2. Setup Server
    server = OpenSWEAgentServer(db_path=":memory:")
    server._get_compiled_agent = AsyncMock(return_value=mock_agent)
    
    # Mock metadata and session persistence
    with (
        patch("agent.agent_acp.get_acp_session", return_value={"cwd": "/tmp"}),
        patch("agent.agent_acp.persist_acp_session"),
        patch("agent.agent_acp.get_thread_metadata", return_value={}),
        patch("agent.agent_acp.update_thread_metadata"),
        # Also need to mock 'agent.server' which is imported inside fork_session
        patch("agent.server.resolve_parent_checkpoint_id", new_callable=AsyncMock) as mock_resolve_parent,
        patch("agent.server.resolve_checkpoint_id", new_callable=AsyncMock) as mock_resolve_check,
    ):
        # Setup resolution mocks to match our snapshot history
        mock_resolve_parent.side_effect = lambda agent, tid, target: "check-1" if target == msg_assistant_id else None
        
        # 3. Execute Fork at the assistant message (e.g. to "edit" it or fork before it)
        # In our "Edit" workflow, we fork at the message we want to REPLACE.
        response = await server.fork_session(
            session_id="orig",
            messageId=msg_assistant_id
        )
        
        assert response.session_id != "orig"
        
        # Verify aupdate_state was called with Snapshot 1's values (the parent)
        # Because fork_at_parent defaults to True for message_id forks
        mock_agent.aupdate_state.assert_called()
        args, kwargs = mock_agent.aupdate_state.call_args
        
        # args[0] is target_config, args[1] is values
        assert args[1]["messages"][-1]["id"] == msg_user_id
        assert len(args[1]["messages"]) == 1
        # It should NOT contain the assistant message or the tool call
        assert msg_assistant_id not in [m["id"] for m in args[1]["messages"]]

@pytest.mark.asyncio
async def test_fork_with_divergence_detection():
    """
    Tests the "Smart Divergence Detection" logic where we provide new values.
    """
    msg_user_id = "msg-user-1"
    
    snapshot_1 = MockSnapshot(
        config={"configurable": {"thread_id": "orig", "checkpoint_id": "check-1"}},
        parent_config=None,
        values={"messages": [{"id": msg_user_id, "type": "human", "content": "Original content"}]},
        next=["agent"]
    )
    
    mock_agent = MockAgent([snapshot_1])
    server = OpenSWEAgentServer(db_path=":memory:")
    server._get_compiled_agent = AsyncMock(return_value=mock_agent)
    
    with (
        patch("agent.agent_acp.get_acp_session", return_value={"cwd": "/tmp"}),
        patch("agent.agent_acp.persist_acp_session"),
        patch("agent.agent_acp.get_thread_metadata", return_value={}),
        patch("agent.agent_acp.update_thread_metadata"),
        patch("agent.server.resolve_parent_checkpoint_id", new_callable=AsyncMock) as mock_resolve_parent,
    ):
        # We provide a new version of the message in 'values'
        new_values = {
            "messages": [
                {"id": msg_user_id, "type": "human", "content": "Edited content"}
            ]
        }
        
        # Mock resolution for divergence
        mock_resolve_parent.return_value = "__START__" # Divergence at first message
        
        response = await server.fork_session(
            session_id="orig",
            values=new_values
        )
        
        # Verify it detected divergence and applied new values
        # Since it's a divergence at msg 0, it should have used aupdate_state with new_values
        # and cloned from START.
        
        # The code does two aupdate_state calls: 
        # 1. Materialize source (empty if START)
        # 2. Apply initial values
        
        calls = mock_agent.aupdate_state.call_args_list
        # Second call should be our new values
        assert calls[-1].args[1]["messages"][0]["content"] == "Edited content"

if __name__ == "__main__":
    import sys
    pytest.main([__file__])
