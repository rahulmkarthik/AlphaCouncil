import os
import builtins
from typing import Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# The SignalEngine expects to be in a Jupyter notebook. We mock 'display' to silence it.
try:
    display
except NameError:
    def display(obj):
        pass
    builtins.display = display

# Import from your vendored library
from volsense_inference.forecast_engine import Forecast
from volsense_inference.signal_engine import SignalEngine

# ------------------------------------------------------------------
# 1. The Service (Singleton to manage heavy model loading)
# ------------------------------------------------------------------
class VolSenseService:
    _instance = None
    _forecast_engine: Optional[Forecast] = None

    @classmethod
    def get_instance(cls):
        """Returns the singleton instance of the service."""
        if cls._instance is None:
            cls._instance = VolSenseService()
        return cls._instance

    def __init__(self):
        # Configuration - adjust these paths if your setup differs
        self.model_version = "v507" # Using the larger model by default
        # Point to the models folder inside the vendored module
        self.checkpoints_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "modules", "VolSense", "models"
        )
        
    def _ensure_loaded(self):
        """Lazy loader: Only loads PyTorch model when actually requested."""
        if self._forecast_engine is None:
            print(f"🔄 VolSenseService: Loading model {self.model_version} from {self.checkpoints_dir}...")
            try:
                self._forecast_engine = Forecast(
                    model_version=self.model_version,
                    checkpoints_dir=self.checkpoints_dir,
                    start="2010-01-01" # Start date for feature fetch
                )
                print("✅ VolSenseService: Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load VolSense model: {e}")

    def analyze_ticker(self, ticker: str) -> str:
        self._ensure_loaded()
        
        # FIX: Single-ticker Z-scores return NaN because std() of 1 item is undefined.
        # We must fetch a "Reference Universe" to calculate relative stats.
        reference_tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "VXX"]
        
        # Ensure target is in the list, but don't duplicate
        universe = list(set([ticker] + reference_tickers))
        
        # 1. Run Forecast on the whole group
        preds = self._forecast_engine.run(tickers=universe)
        
        # 2. Pass to Signal Engine (Now valid because N > 1)
        sig_engine = SignalEngine(model_version=self.model_version)
        sig_engine.set_data(preds)
        sig_engine.compute_signals(enrich_with_sectors=True)
        
        # 3. Get Text Summary ONLY for the requested ticker
        return sig_engine.ticker_summary(ticker)

    def analyze_market_sectors(self) -> str:
        """Runs a broader market scan to get sector trends."""
        self._ensure_loaded()
        
        # For a "Market Scan", we usually need a representative list.
        # We'll use a small subset of major tickers to proxy the sectors for speed.
        # (Scanning 500 tickers might be too slow for an interactive agent right now)
        proxy_universe = [
            "SPY", "QQQ", "XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "GLD", "TLT"
        ]
        
        preds = self._forecast_engine.run(tickers=proxy_universe)
        
        sig_engine = SignalEngine(model_version=self.model_version)
        sig_engine.set_data(preds)
        sig_engine.compute_signals(enrich_with_sectors=True)
        
        # Get the summary dataframe and convert to string
        summary_df = sig_engine.sector_summary()
        return summary_df.to_markdown(index=False)

# ------------------------------------------------------------------
# 2. The LangChain Tool Definitions
# ------------------------------------------------------------------

class TickerInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol to analyze (e.g. 'AAPL', 'NVDA')")

@tool("get_volatility_forecast", args_schema=TickerInput)
def get_volatility_forecast(ticker: str) -> str:
    """
    Retrieves the quantitative volatility forecast, regime, and signal for a given ticker.
    Use this tool to get the technical view on a stock.
    Returns a text summary including Z-scores, forecasts, and regime flags.
    """
    service = VolSenseService.get_instance()
    try:
        return service.analyze_ticker(ticker.upper())
    except Exception as e:
        return f"Error analyzing {ticker}: {str(e)}"

@tool("get_sector_trends")
def get_sector_trends() -> str:
    """
    Retrieves the current volatility heatmap of market sectors.
    Use this to understand if specific sectors (Tech, Energy, etc.) are under stress.
    """
    service = VolSenseService.get_instance()
    try:
        return service.analyze_market_sectors()
    except Exception as e:
        return f"Error analyzing sectors: {str(e)}"