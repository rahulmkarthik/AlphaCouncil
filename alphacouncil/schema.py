from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# --- Technical Agent Output ---
class TechnicalSignal(BaseModel):
    ticker: str
    
    # UPDATED: Broader vocabulary matching SignalEngine 2.0
    signal: Literal["BUY", "SELL", "WAIT", "HEDGE", "STRONG_BUY", "STRONG_SELL"] 
    
    confidence: float = Field(description="Confidence score (0.0-1.0) derived from signal strength")
    
    # UPDATED: Allow new regimes like 'spike', 'calm', etc.
    regime: str = Field(description="Volatility Regime (e.g. 'Spike', 'Normal', 'Calm')")
    
    # UPDATED: Context for the new momentum factors
    key_drivers: List[str] = Field(description="Top reasons (e.g. 'Positive 20d Trend', 'Term Structure Inversion')")
    
    reasoning: str = Field(description="Concise summary of the trade rationale")

# --- Fundamental Agent Output ---
class SectorIntel(BaseModel):
    sector: str
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    major_events: List[str] = Field(description="List of specific earnings, regulatory, or macro events found")
    sentiment_score: float = Field(description="-1.0 (Bearish) to 1.0 (Bullish)")
    relevance_to_ticker: str = Field(description="How these sector-wide events specifically affect the requested ticker.")

# --- Risk Agent Output ---
class RiskAssessment(BaseModel):
    verdict: Literal["APPROVED", "REJECTED", "MODIFIED"]
    reason: str = Field(description="Explanation for the verdict")
    approved_quantity: int = Field(description="Number of shares approved (0 if rejected)")
    max_exposure_allowed: float = Field(description="The dollar limit calculated for this trade")
    risk_score: int = Field(description="1-10 scale of trade danger")