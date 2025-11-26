from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# --- Technical Agent Output ---
class TechnicalSignal(BaseModel):
    ticker: str
    signal: Literal["BUY", "SELL", "WAIT", "HEDGE"]
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 based on signal strength")
    regime: Literal["Low Vol", "Normal", "High Vol Spike", "Extreme Stress"]
    key_drivers: List[str] = Field(description="Top 3 reasons (e.g., 'Sector Rank > 0.9', 'Negative Vol Spread', 'Z-Score Breakout')")
    reasoning: str = Field(description="A concise, single-sentence summary of the technical view.")

# --- Fundamental Agent Output ---
class SectorIntel(BaseModel):
    sector: str
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    major_events: List[str] = Field(description="List of specific earnings, regulatory, or macro events found")
    sentiment_score: float = Field(description="-1.0 (Bearish) to 1.0 (Bullish)")
    relevance_to_ticker: str = Field(description="How these sector-wide events specifically affect the requested ticker.")