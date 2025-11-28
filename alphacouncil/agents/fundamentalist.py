from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from alphacouncil.schema import SectorIntel
from alphacouncil.utils.langchain_stub import tool

# 1. Define Batched Tool
@tool
def search_sector_news(sector: str) -> str:
    """
    Batched Search: Finds major regulatory/earnings news for an ENTIRE sector.
    Efficiently covers multiple tickers in one query.
    """
    # Max results = 5 gives good breadth without burning tokens
    tavily = TavilySearchResults(max_results=5) 
    query = f"Major market-moving news, earnings, or regulation for {sector} sector today"
    return str(tavily.invoke({"query": query}))

# 2. Initialize Gemini (Flash is fine here)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0
)

# 3. Create Structured Agent
fundamentalist_runnable = llm.bind_tools([search_sector_news]).with_structured_output(SectorIntel)

# 4. System Prompt
SYSTEM_PROMPT = """You are 'The Fundamentalist'.
1. Identify the SECTOR of the input ticker (e.g. NVDA -> Technology).
2. Call `search_sector_news` for that SECTOR.
3. Extract risks relevant to the specific ticker from the sector-wide news.
4. Output strictly valid JSON matching the SectorIntel schema."""

# 5. Node Function
def fundamentalist_agent(state):
    # Same fix: Ensure there is a trigger message
    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=f"Analyze the sector risks for {state['ticker']}")]

    response = fundamentalist_runnable.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    return {"fundamental_signal": response}