import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Ensure API key is set (or load from .env)
# os.environ["TAVILY_API_KEY"] = "your-key-here" 

@tool("market_news_search")
def market_news_search(query: str) -> str:
    """
    Search for real-time financial news, earnings dates, or macro events.
    Useful for finding 'why' a stock is moving.
    """
    search = TavilySearchResults(max_results=3)
    try:
        results = search.invoke({"query": query})
        output = []
        for res in results:
            output.append(f"- {res['content']} (Source: {res['url']})")
        return "\n".join(output)
    except Exception as e:
        return f"Search failed: {str(e)}"