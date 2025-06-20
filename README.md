# APLCart MCP Server

A Model Context Protocol (MCP) server that exposes the APLCart idiom collection with semantic search capabilities. APLCart is a searchable collection of APL expressions and their descriptions.

## Features

- Find APL expressions by exact syntax match
- Search across syntax, descriptions, and keywords
- Get keywords for specific APL expressions
- Natural language queries using OpenAI embeddings

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (for semantic search functionality)

### Using uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd ac2jsonl

# Install dependencies
uv sync
```

## Setup

### Convert APLCart Data

First, fetch and convert the APLCart TSV data to JSONL format:

```bash
# Using uv
uv run python aplcart2json.py

# Or with activated venv
python aplcart2json.py

# Optional: Generate SQLite database for faster searches
python aplcart2json.py --db
```

### Generate Embeddings (Optional; for Semantic Search)

To enable semantic search functionality:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Generate embeddings
uv run python generate_embeddings.py

# Or with activated venv
python generate_embeddings.py
```

This creates:
- `aplcart.index` - FAISS index file containing embeddings
- `aplcart_metadata.pkl` - Metadata for semantic search results

## Usage

### Running the MCP Server

```bash
# Basic usage
uv run python aplcart_mcp_semantic.py

# With SQLite database backend
APLCART_USE_DB=1 uv run python aplcart_mcp_semantic.py
```

### Using with Claude Code

The project includes a `.mcp.json.template` file that automatically configures the MCP server. Save that as `.mcp.json`, update it with your details, and run `/mcp` in Claude Code to see available servers.

You can also manually add the server:
```bash
claude mcp add aplcart "uv run python aplcart_mcp_semantic.py"
```

### Using with Claude Desktop

Add this to your Claude Desktop configuration file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`  
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`  
- Linux: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "aplcart": {
      "command": "uv",
      "args": ["run", "python", "YOUR/PATH/HERE/ac-mcp/aplcart_mcp_semantic.py"],
      "cwd": "YOUR/PATH/HERE/ac-mcp",
      "env": {
        "OPENAI_API_KEY": "your-api-key-here",
        "APLCART_USE_DB": "1"
      }
    }
  }
}
```

Then restart Claude Desktop to load the MCP server.

### Available MCP Tools

- `lookup-syntax` - Exact match on APL syntax
  ```
  Example: lookup-syntax "⍳10"
  ```

- `search` - Substring search across syntax, description, and keywords
  ```
  Example: search "matrix" limit=10
  ```

- `keywords-for` - Get keywords for a specific syntax
  ```
  Example: keywords-for "∘.≤⍨∘⍳"
  ```

- `semantic-search` - Natural language search using embeddings
  ```
  Example: semantic-search "how to split a string on a separator"
  ```

### Standalone Search Tool

You can also use the semantic search functionality directly:

```bash
# Interactive mode
uv run python search_embeddings.py

# Single query
uv run python search_embeddings.py "find the largest number"

# JSON output
uv run python search_embeddings.py "reverse an array" --json

# More results
uv run python search_embeddings.py "matrix operations" -k 10
```

Interactive mode commands:
- Type your query and press Enter to search
- Type `quit`, `exit`, or `q` to exit (or Ctrl+D or Ctrl+C)

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for semantic search functionality
- `APLCART_USE_DB` - Set to `1`, `true`, or `yes` to use SQLite database backend

### File Structure

```
ac-mcp/
├── aplcart.jsonl           # Converted APLCart data (run aplcart2json.py to generate)
├── aplcart.db              # SQLite database (optional)
├── aplcart.index           # FAISS embeddings index (run generate_embeddings.py to generate)
├── aplcart_metadata.pkl    # Metadata for semantic search (run generate_embeddings.py to generate)
├── aplcart2json.py         # Converter script
├── generate_embeddings.py  # Embedding generator
├── aplcart_mcp_semantic.py # MCP server with semantic search
├── search_embeddings.py    # Standalone search tool
└── pyproject.toml          # Project dependencies
```

## About APLCart

APLCart is a searchable collection of APL idioms and expressions maintained at https://aplcart.info/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note: The APLCart data itself is subject to its own [licensing terms](https://github.com/abrudz/aplcart/blob/master/LICENSE).
