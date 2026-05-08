import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from acp_agent_server import (
    BaseAgentServer,
    ACPSessionContext,
    load_env_configs,
)

# Standardized model list for this agent
AVAILABLE_MODELS = [
    {
        "value": "openai:gpt-4o-mini",
        "name": "GPT-4o Mini",
        "description": "Fast and efficient for daily tasks",
    },
    {
        "value": "openai:gpt-4o",
        "name": "GPT-4o",
        "description": "Powerful reasoning and creativity",
    },
    {
        "value": "anthropic:claude-3-5-sonnet-latest",
        "name": "Claude 3.5 Sonnet",
        "description": "Exceptional coding and nuance",
    },
]
from acp_agent_server.launcher import serve_acp_stdio
from langgraph.graph.state import RunnableConfig


# Load environment variables using universal utility
load_env_configs()

# Default to local sandbox for ACP execution
if "SANDBOX_TYPE" not in os.environ:
    os.environ["SANDBOX_TYPE"] = "local"

# Configure line buffering for logs/stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logger = logging.getLogger(__name__)

class OpenSWEAgentServer(BaseAgentServer):
    """Simplified ACP server for Open SWE agent focusing on graph construction."""

    async def build_agent(
        self, context: ACPSessionContext, for_execution: bool = True
    ) -> Any:
        """Build the Open SWE agent graph for the current ACP session."""
        from agent.server import get_agent

        # Extract context fields safely
        session_id = context.session_id
        mode = context.mode
        user_id = context.user_id

        logger.info(f"Building Open SWE agent for session {session_id} (mode={mode}, execution={for_execution})")

        try:
            # Note: Persistence is handled internally by agent.server.get_agent 
            # using the OPEN_SWE_CHECKPOINT_DB environment variable.
            
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": session_id,
                    "mode": mode,
                    "model_id": getattr(context, "model", None),
                    "__is_for_execution__": for_execution,
                },
                "metadata": {
                    "user_id": user_id,
                },
            }

            # Optional: Pass checkpoint_id if present in context (for future fork stability)
            if context.checkpoint_id:
                config["configurable"]["checkpoint_id"] = context.checkpoint_id

            # Pass the config to the factory. The factory handles checkpointer initialization.
            agent = await get_agent(config)
            return agent

        except Exception:
            logger.error(f"Failed to build agent: {traceback.format_exc()}")
            raise

if __name__ == "__main__":
    # Default persistence path
    default_db = Path.home() / ".deepagents" / "cache" / "swe_checkpoints.db"
    
    # Ensure our environment variable for the checkpointer is set before the loop starts
    if "OPEN_SWE_CHECKPOINT_DB" not in os.environ:
        os.environ["OPEN_SWE_CHECKPOINT_DB"] = str(default_db)
        
    # Launch the server using the standard launcher
    serve_acp_stdio(
        server_class=OpenSWEAgentServer,
        db_path=os.environ["OPEN_SWE_CHECKPOINT_DB"],
        name="Open SWE Agent",
        version="1.1.0",
        models=AVAILABLE_MODELS
    )
