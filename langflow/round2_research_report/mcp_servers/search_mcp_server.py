"""
Web Search MCP Server (DuckDuckGo — no API key needed)
=======================================================
Exposes a `web_search` tool over the Model Context Protocol (MCP).
The Research Agent in Langflow connects to this via the MCP Tools component.

RUN IT (easiest, with uv — auto-installs deps):
    uv run --with mcp --with ddgs python search_mcp_server.py

In Langflow's MCP Tools component (STDIO mode), set the command to exactly that.
"""

from mcp.server.fastmcp import FastMCP

# Create the MCP server. The name shows when the agent lists available tools.
mcp = FastMCP("web-search")


def _search(query: str, max_results: int):
    """Try the modern `ddgs` package first, fall back to `duckduckgo_search`."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # older package name
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current, up-to-date information about a query.
    Returns a list of results, each with a title, a snippet, and a source URL.
    Use this to research a topic before writing a report.
    """
    try:
        results = _search(query, max_results)
    except Exception as e:
        return f"SEARCH_ERROR: could not complete the search ({e}). Try again or rephrase."

    if not results:
        return "NO_RESULTS: the search returned nothing. Try a broader or different query."

    blocks = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "Untitled"
        snippet = r.get("body") or r.get("snippet") or ""
        url = r.get("href") or r.get("url") or ""
        blocks.append(f"[{i}] {title}\n{snippet}\nSource: {url}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    # Runs over STDIO by default — exactly what Langflow's MCP Tools (STDIO) expects.
    mcp.run()
