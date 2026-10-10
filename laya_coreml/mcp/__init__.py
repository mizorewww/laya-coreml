"""MCP (Model Context Protocol) stdio server for Laya (optional extra).

Install the extra to use it:

    pip install "laya[mcp]"

Then run the server:

    laya-mcp-server            # console script
    python -m laya.mcp.server  # equivalent module form

The submodules ``device`` and ``tools`` are importable without the ``mcp``
dependency; ``server`` requires the extra.
"""

__all__ = ["device", "server", "tools"]
