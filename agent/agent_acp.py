import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

# Generic Universal ACP Agent Server framework
from acp_agent_server import (
    BaseAgentServer,
    ACPSessionContext,
)
from acp_agent_server.checkpointer import get_checkpointer

from acp import run_agent
from acp.schema import (
    HttpMcpServer,
    McpServerStdio,
    SseMcpServer,
)
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from dotenv import load_dotenv

# Setup project paths
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent.parent

# Load environment variables from identified locations
load_dotenv(project_root / ".env")
load_dotenv(project_root / "apps" / "api" / ".env.development")

# Default to local sandbox for ACP execution
if "SANDBOX_TYPE" not in os.environ:
    os.environ["SANDBOX_TYPE"] = "local"

# Configure logging to flush immediately
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
    force=True
)
# Ensure buffers are flushed even if the process is piped
sys.stderr.reconfigure(line_buffering=True)
sys.stdout.reconfigure(line_buffering=True)

logger = logging.getLogger(__name__)
logger.info("SWE AGENT STARTING (Simplified Mode)...")

DEFAULT_MODEL = os.environ.get("DEEPAGENTS_MODEL", "openai:gpt-4o")
DEFAULT_REPO_OWNER = os.environ.get("DEFAULT_REPO_OWNER", "TeodorRusKvi")

from agent.utils.sandbox_state import (
    get_thread_metadata,
)

def infer_repo_from_cwd(cwd: str) -> dict[str, str]:
    """Try to infer GitHub repo owner/name from git remote."""
    try:
        import subprocess
        import re

        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            match = re.search(r"github\.com[:/](.+?)/(.+?)(\.git)?$", url)
            if match:
                return {"owner": match.group(1), "name": match.group(2)}
    except Exception:
        pass

    path = Path(cwd)
    return {"owner": DEFAULT_REPO_OWNER, "name": path.name}


class OpenSWEAgentServer(BaseAgentServer):
    """Simplified ACP server for Open SWE agent with local SQLite persistence."""

    async def build_agent(
        self, context: ACPSessionContext, for_execution: bool = True
    ) -> CompiledStateGraph:
        """Build the Open SWE agent graph for the current ACP session."""
        logger.info(f"DEBUG: build_agent called for session {context.session_id}")
        try:
            from agent import server

            get_agent = server.get_agent
            workspace_root = Path(context.cwd).expanduser()
            os.environ["LOCAL_SANDBOX_ROOT_DIR"] = str(workspace_root)
            repo_info = infer_repo_from_cwd(str(workspace_root))

            mcp_tools = []
            mcp_servers = getattr(context, "mcp_servers", [])
            if mcp_servers:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                server_configs = {}
                for i, server_cfg in enumerate(mcp_servers):
                    server_id = getattr(server_cfg, "name", f"server_{i}")
                    if isinstance(server_cfg, McpServerStdio):
                        server_configs[server_id] = {
                            "transport": "stdio",
                            "command": server_cfg.command,
                            "args": server_cfg.args,
                            "env": {e.name: e.value for e in server_cfg.env}
                            if server_cfg.env
                            else None,
                        }
                    elif isinstance(server_cfg, HttpMcpServer):
                        server_configs[server_id] = {
                            "transport": "streamable_http",
                            "url": server_cfg.url,
                            "headers": server_cfg.headers,
                        }
                    elif isinstance(server_cfg, SseMcpServer):
                        server_configs[server_id] = {
                            "transport": "sse",
                            "url": server_cfg.url,
                            "headers": server_cfg.headers,
                        }

                if server_configs:
                    try:
                        mcp_client = MultiServerMCPClient(server_configs)
                        mcp_tools = await mcp_client.get_tools()
                        logger.info(
                            f"Loaded {len(mcp_tools)} tools from {len(server_configs)} MCP servers"
                        )
                    except Exception as mcp_err:
                        logger.error(f"Failed to load MCP tools: {mcp_err}")

            # Use local SQLite checkpointer
            db_path = self.db_path or (Path.home() / ".deepagents" / "cache" / "swe_checkpoints.db")
            checkpointer = await get_checkpointer(db_path=db_path, backend="sqlite", is_async=True)

            config: RunnableConfig = {
                "configurable": {
                    "thread_id": context.session_id or str(uuid4()),
                    "mode": context.mode,
                    "github_token": os.environ.get("GITHUB_TOKEN"),
                    "__is_for_execution__": for_execution,
                    "extra_tools": mcp_tools,
                },
                "metadata": {
                    "user_id": context.user_id,
                    "repo_owner": repo_info["owner"],
                    "repo_name": repo_info["name"],
                },
            }

            # Optional: Resume from a specific checkpoint if stored in metadata
            thread_id = config["configurable"]["thread_id"]
            meta = get_thread_metadata(str(thread_id))
            checkpoint_id = meta.get("checkpoint_id")
            if checkpoint_id:
                config["configurable"]["checkpoint_id"] = checkpoint_id

            model = context.model or DEFAULT_MODEL
            os.environ["LLM_MODEL_ID"] = model
            os.environ["OPEN_SWE_CHECKPOINT_DB"] = str(db_path)

            logger.info(f"Building Open SWE agent for workspace: {workspace_root}, model: {model}, thread: {thread_id}")
            
            # Pass our checkpointer to the factory
            agent = await get_agent(config)
            return agent

        except Exception as e:
            logger.error(f"FAILED TO BUILD OPEN SWE AGENT: {e}")
            traceback.print_exc(file=sys.stderr)
            raise

    # Note: We NO LONGER override fork_session. 
    # The BaseAgentServer will use the default protocol (or we can implement standard 
    # branching if needed, but for now we remove the materialized cloning complexity).

if __name__ == "__main__":
    db_path = Path.home() / ".deepagents" / "cache" / "swe_checkpoints.db"
    server = OpenSWEAgentServer(db_path=db_path)
    asyncio.run(run_agent(server, use_unstable_protocol=True))
