import os

def create_sandbox(sandbox_id: str | None = None):
    """Create or reconnect to a sandbox using the configured provider.

    The provider is selected via the SANDBOX_TYPE environment variable.
    Supported values: langsmith (default), daytona, modal, runloop, local.

    Args:
        sandbox_id: Optional existing sandbox ID to reconnect to.

    Returns:
        A sandbox backend implementing SandboxBackendProtocol.
    """
    sandbox_type = os.getenv("SANDBOX_TYPE", "langsmith")
    
    if sandbox_type == "langsmith":
        from agent.integrations.langsmith import create_langsmith_sandbox
        return create_langsmith_sandbox(sandbox_id)
    elif sandbox_type == "daytona":
        from agent.integrations.daytona import create_daytona_sandbox
        return create_daytona_sandbox(sandbox_id)
    elif sandbox_type == "modal":
        from agent.integrations.modal import create_modal_sandbox
        return create_modal_sandbox(sandbox_id)
    elif sandbox_type == "runloop":
        from agent.integrations.runloop import create_runloop_sandbox
        return create_runloop_sandbox(sandbox_id)
    elif sandbox_type == "local":
        from agent.integrations.local import create_local_sandbox
        return create_local_sandbox(sandbox_id)
    else:
        supported = "langsmith, daytona, modal, runloop, local"
        raise ValueError(f"Invalid sandbox type: {sandbox_type}. Supported types: {supported}")
