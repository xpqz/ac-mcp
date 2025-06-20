#!/bin/bash
# Wrapper script for MCP server
cd "$(dirname "$0")"
export APLCART_USE_DB=1
exec uv run python aplcart_mcp_semantic.py