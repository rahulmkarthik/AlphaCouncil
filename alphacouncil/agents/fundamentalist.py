from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from urllib.parse import urlparse
import json
import re

from alphacouncil.schema import SectorIntel, NewsStory
from alphacouncil.utils.langchain_stub import tool

# Import VolSense sector mapping
try:
    from volsense_inference.sector_mapping import get_sector_map
    SECTOR_MAP = get_sector_map("v507")
    print(f"[FUNDAMENTALIST v6] Loaded VolSense sector map with {len(SECTOR_MAP)} tickers")
except ImportError as e:
    print(f"[FUNDAMENTALIST v6] Could not import VolSense sector_mapping: {e}")
    SECTOR_MAP = {}

print("[FUNDAMENTALIST v6] Module loaded successfully")

# Helper function to extract source from URL
def extract_source(url: str) -> str:
    """Extract a readable source name from a URL."""
    try:
        domain = urlparse(url).netloc
        domain = domain.replace('www.', '')
        source_map = {
            'bloomberg.com': 'Bloomberg',
            'reuters.com': 'Reuters',
            'wsj.com': 'Wall Street Journal',
            'cnbc.com': 'CNBC',
            'ft.com': 'Financial Times',
            'marketwatch.com': 'MarketWatch',
            'forbes.com': 'Forbes',
            'seekingalpha.com': 'Seeking Alpha',
            'yahoo.com': 'Yahoo Finance',
            'benzinga.com': 'Benzinga',
            'investopedia.com': 'Investopedia',
            'thestreet.com': 'TheStreet',
            'barrons.com': 'Barron\'s',
            'investors.com': 'Investor\'s Business Daily',
            'fool.com': 'Motley Fool',
        }
        for key, value in source_map.items():
            if key in domain:
                return value
        return domain.split('.')[0].capitalize()
    except:
        return "Market News"

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    
    # Try to find raw JSON object
    json_match = re.search(r'\{[^{}]*"headline"[^{}]*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    
    # Last resort: try parsing the whole thing
    return json.loads(text)

def search_news(query: str, limit: int = 10):
    """Generic news search."""
    tavily = TavilySearchResults(max_results=limit)
    print(f"[FUNDAMENTALIST v6] Search query: {query}")
    results = tavily.invoke({"query": query})
    return results

def search_sector_news(sector: str, limit: int = 10):
    """Search for broad sector-wide news (for Sector Scan mode)."""
    query = f"{sector} sector news latest market trends earnings regulations announcements"
    return search_news(query, limit)

def search_ticker_news(ticker: str, sector: str, limit: int = 10):
    """Search for ticker-specific news (for Ticker Deep Dive mode)."""
    query = f"{ticker} stock news latest announcements earnings acquisitions mergers {sector}"
    return search_news(query, limit)

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0.2
)

fundamentalist_runnable = llm.with_structured_output(SectorIntel)

RESTRICTED_PROMPT = """You are 'The Fundamentalist'.
You will receive sector news search results.

Analyze them and return:
- sector: The sector name
- risk_level: HIGH, MEDIUM, or LOW
- major_events: List of 3-5 brief one-sentence summaries of key events
- sentiment_score: Overall sentiment from -1.0 (bearish) to 1.0 (bullish)
- relevance_to_ticker: How these sector events affect the specific ticker
- expanded_news: Leave as null

Keep it concise."""

def get_sector_for_ticker(ticker: str) -> str:
    """Get sector for a ticker using VolSense mapping, with fallback."""
    return SECTOR_MAP.get(ticker.upper(), 'Technology')

def create_expanded_news_stories(search_results, sector: str, ticker: str, mode: str):
    """Process search results into NewsStory objects with LLM enrichment."""
    stories = []
    
    context = f"the {sector} sector" if mode == "Sector Scan" else f"{ticker}"
    print(f"[FUNDAMENTALIST v6] Processing {len(search_results)} search results for {context}")
    
    for i, result in enumerate(search_results[:10]):
        content = result.get('content', result.get('snippet', ''))
        url = result.get('url', 'https://markets.example.com')
        source = extract_source(url)
        
        print(f"[FUNDAMENTALIST v6] Processing story {i+1}: {content[:60]}...")
        
        # Context-aware enrichment prompt
        if mode == "Sector Scan":
            enrichment_prompt = f"""Analyze this {sector} sector news:

"{content}"

IMPORTANT SENTIMENT RULES:
- If the news mentions growth, expansion, positive earnings, gains, or bullish outlook → "Positive"
- If the news mentions layoffs, losses, regulatory issues, declines, or bearish outlook → "Negative"  
- ONLY use "Neutral" if purely factual with no clear positive or negative implications

Respond with ONLY valid JSON (no markdown):
{{"headline": "Clear headline about what happened in {sector} sector", "summary": "3-5 sentence summary explaining the news and its significance for {sector} sector", "sentiment": "Positive or Negative or Neutral"}}"""
        else:
            enrichment_prompt = f"""Analyze this financial news for {ticker}:

"{content}"

IMPORTANT SENTIMENT RULES:
- If the news mentions acquisitions, growth, expansion, positive earnings, stock gains → "Positive"
- If the news mentions layoffs, losses, regulatory issues, stock declines, lawsuits → "Negative"  
- ONLY use "Neutral" if purely factual with no clear positive or negative implications

Respond with ONLY valid JSON (no markdown):
{{"headline": "Clear headline about what happened with {ticker}", "summary": "3-5 sentence summary explaining the news and its likely impact on {ticker} stock", "sentiment": "Positive or Negative or Neutral"}}"""
        
        try:
            enriched = llm.invoke(enrichment_prompt)
            response_text = enriched.content if hasattr(enriched, 'content') else str(enriched)
            
            enriched_data = extract_json_from_response(response_text)
            
            # Validate sentiment value
            sentiment = enriched_data.get('sentiment', 'Neutral')
            if sentiment not in ['Positive', 'Negative', 'Neutral']:
                # Try to infer from content
                content_lower = content.lower()
                if any(word in content_lower for word in ['acquisition', 'growth', 'beats', 'surge', 'rally', 'gains']):
                    sentiment = 'Positive'
                elif any(word in content_lower for word in ['layoffs', 'decline', 'loss', 'lawsuit', 'fall', 'drops']):
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
            
            story = NewsStory(
                headline=enriched_data.get('headline', content[:100]),
                summary=enriched_data.get('summary', content),
                source=source,
                url=url,
                sentiment=sentiment
            )
            stories.append(story)
            print(f"[FUNDAMENTALIST v6] Story {i+1} → {sentiment}")
            
        except Exception as e:
            print(f"[FUNDAMENTALIST v6] Failed to enrich story {i+1}: {type(e).__name__}: {e}")
            # Fallback with keyword-based sentiment
            content_lower = content.lower()
            if any(word in content_lower for word in ['acquisition', 'growth', 'beats', 'surge', 'rally', 'gains', 'expands']):
                sentiment = 'Positive'
            elif any(word in content_lower for word in ['layoffs', 'decline', 'loss', 'lawsuit', 'fall', 'drops', 'cuts']):
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            story = NewsStory(
                headline=content[:100] if len(content) > 100 else content,
                summary=content if len(content) > 50 else f"News update regarding {context}.",
                source=source,
                url=url,
                sentiment=sentiment
            )
            stories.append(story)
    
    print(f"[FUNDAMENTALIST v6] Created {len(stories)} total stories")
    return stories

def fundamentalist_agent(state):
    """
    Main agent function supporting both expanded and restricted modes.
    
    State should contain:
    - ticker: The ticker symbol
    - expanded: True for expanded mode (with news stories), False for restricted
    - mode: "Sector Scan" for sector-wide news, "Ticker Deep Dive" for ticker-specific
    - sector: (optional) The sector name if already known (for Sector Scan mode)
    """
    ticker = state['ticker']
    expanded = state.get("expanded", False)
    mode = state.get("mode", "Ticker Deep Dive")  # Default to ticker-specific
    
    print(f"\n[FUNDAMENTALIST v6] ========== NEW REQUEST ==========")
    print(f"[FUNDAMENTALIST v6] Ticker: {ticker}, Mode: {mode}, Expanded: {expanded}")
    
    # Get sector - either from state (for Sector Scan) or from VolSense mapping
    if mode == "Sector Scan" and "sector" in state:
        sector = state["sector"]
    else:
        sector = get_sector_for_ticker(ticker)
    print(f"[FUNDAMENTALIST v6] Sector: {sector}")
    
    # Search based on mode
    limit = 10 if expanded else 5
    if mode == "Sector Scan":
        # Sector-wide news for Sector Scan
        search_results = search_sector_news(sector, limit)
    else:
        # Ticker-specific news for Ticker Deep Dive
        search_results = search_ticker_news(ticker, sector, limit)
    
    print(f"[FUNDAMENTALIST v6] Got {len(search_results)} search results")
    
    if expanded:
        # Create expanded news stories
        expanded_news = create_expanded_news_stories(search_results, sector, ticker, mode)
        print(f"[FUNDAMENTALIST v6] Created {len(expanded_news)} expanded news stories")
        
        # Calculate sentiment from stories
        sentiment_values = {'Positive': 0.5, 'Negative': -0.5, 'Neutral': 0.0}
        if expanded_news:
            sentiments = [sentiment_values.get(s.sentiment, 0.0) for s in expanded_news]
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Log sentiment distribution
            pos_count = sum(1 for s in expanded_news if s.sentiment == 'Positive')
            neg_count = sum(1 for s in expanded_news if s.sentiment == 'Negative')
            neu_count = sum(1 for s in expanded_news if s.sentiment == 'Neutral')
            print(f"[FUNDAMENTALIST v6] Sentiment distribution: +{pos_count} / -{neg_count} / ~{neu_count}")
        else:
            avg_sentiment = 0.0
        
        # Determine risk level based on sentiment
        if avg_sentiment < -0.15:
            risk_level = "HIGH"
        elif avg_sentiment > 0.15:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        # Build relevance message based on mode
        if mode == "Sector Scan":
            relevance_msg = f"The {sector} sector trends will impact all related stocks including {ticker}. Monitor sector-wide developments for portfolio positioning."
        else:
            relevance_msg = f"{ticker}'s performance is directly affected by these developments. Recent news will likely impact {ticker}'s stock price and investor sentiment."
        
        # Manually construct SectorIntel
        intel = SectorIntel(
            sector=sector,
            risk_level=risk_level,
            major_events=[story.headline for story in expanded_news[:5]],
            sentiment_score=round(avg_sentiment, 2),
            relevance_to_ticker=relevance_msg,
            expanded_news=expanded_news
        )
        
        print(f"[FUNDAMENTALIST v6] Final: risk={risk_level}, sentiment={avg_sentiment:.2f}")
        return {"fundamental_signal": intel}
    
    else:
        # Restricted mode: use structured output
        messages = [HumanMessage(content=f"Analyze sector risks for {ticker}. Search results: {json.dumps(search_results, indent=2)}")]
        response = fundamentalist_runnable.invoke([SystemMessage(content=RESTRICTED_PROMPT)] + messages)
        return {"fundamental_signal": response}