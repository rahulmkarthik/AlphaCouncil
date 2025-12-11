"""
Test script for strict risk controls in risk_manager.py
Tests: GOOG, NVDA, KOLD, BOIL

Run from project root:
    python tests/test_risk_controls.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables FIRST (needed for GOOGLE_API_KEY)
from dotenv import load_dotenv
load_dotenv()

from alphacouncil.agents.risk_manager import risk_manager_agent, _calculate_daily_pnl, _resolve_live_price
from alphacouncil.schema import TechnicalSignal, SectorIntel
from alphacouncil.execution.portfolio import PortfolioService
from langchain_core.messages import HumanMessage


def create_tech_signal(ticker: str, confidence: float = 0.75, signal: str = "BUY", regime: str = "NORMAL"):
    """Create a mock TechnicalSignal."""
    return TechnicalSignal(
        ticker=ticker,
        signal=signal,
        confidence=confidence,
        regime=regime,
        reasoning="Test signal",
        key_drivers=["Test driver"]
    )


def create_fund_signal(risk_level: str = "LOW", sentiment: float = 0.5):
    """Create a mock SectorIntel."""
    return SectorIntel(
        sector="Technology",
        risk_level=risk_level,
        major_events=["Test event"],
        sentiment_score=sentiment,
        relevance_to_ticker="Test relevance"
    )


def test_case(name: str, state: dict, expected_verdict: str = None, expected_contains: str = None):
    """Run a test case and print results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    try:
        result = risk_manager_agent(state)
        assessment = result.get("risk_assessment")
        
        if assessment:
            print(f"  Verdict: {assessment.verdict}")
            print(f"  Reason: {assessment.reason[:100]}..." if len(assessment.reason) > 100 else f"  Reason: {assessment.reason}")
            print(f"  Approved Qty: {assessment.approved_quantity}")
            print(f"  Max Exposure: ${assessment.max_exposure_allowed:,.2f}")
            print(f"  Risk Score: {assessment.risk_score}")
            
            # Check expectations
            if expected_verdict and assessment.verdict != expected_verdict:
                print(f"  ❌ FAIL: Expected verdict '{expected_verdict}', got '{assessment.verdict}'")
            elif expected_verdict:
                print(f"  ✅ PASS: Verdict matches expected '{expected_verdict}'")
                
            if expected_contains and expected_contains not in assessment.reason:
                print(f"  ❌ FAIL: Expected reason to contain '{expected_contains}'")
            elif expected_contains:
                print(f"  ✅ PASS: Reason contains expected text")
        else:
            print("  ❌ ERROR: No risk_assessment in result")
            
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*60)
    print("RISK CONTROLS TEST SUITE")
    print("="*60)
    
    # Get current portfolio state
    portfolio = PortfolioService()
    state = portfolio.get_state()
    print(f"\nPortfolio Cash: ${state.cash_balance:,.2f}")
    print(f"Holdings: {list(state.holdings.keys())}")
    
    tickers = ["GOOG", "NVDA", "KOLD", "BOIL"]
    
    # Print current prices
    print("\nCurrent Prices:")
    for ticker in tickers:
        price = _resolve_live_price(ticker)
        print(f"  {ticker}: ${price:.2f}")
    
    # =========================================================
    # TEST 1: SOFT STOP - Low Confidence
    # =========================================================
    for ticker in tickers:
        test_case(
            name=f"{ticker} - SOFT STOP: Low Confidence (50%)",
            state={
                "ticker": ticker,
                "technical_signal": create_tech_signal(ticker=ticker, confidence=0.50),
                "fundamental_signal": create_fund_signal(),
                "messages": [HumanMessage(content=f"User requests to BUY 10 shares of {ticker}")]
            },
            expected_verdict="REJECTED",
            expected_contains="SOFT STOP"
        )
    
    # =========================================================
    # TEST 2: SOFT STOP - High Risk + Negative Sentiment
    # =========================================================
    test_case(
        name="NVDA - SOFT STOP: HIGH Risk + Negative Sentiment",
        state={
            "ticker": "NVDA",
            "technical_signal": create_tech_signal(ticker="NVDA", confidence=0.70),
            "fundamental_signal": create_fund_signal(risk_level="HIGH", sentiment=-0.5),
            "messages": [HumanMessage(content="User requests to BUY 10 shares of NVDA")]
        },
        expected_verdict="REJECTED",
        expected_contains="SOFT STOP"
    )
    
    # =========================================================
    # TEST 3: LIMIT EXCEEDED - Request too many shares
    # =========================================================
    test_case(
        name="GOOG - LIMIT EXCEEDED: Request 1000 shares",
        state={
            "ticker": "GOOG",
            "technical_signal": create_tech_signal(ticker="GOOG", confidence=0.80, regime="MANUAL_OVERRIDE"),
            "fundamental_signal": create_fund_signal(),
            "messages": [HumanMessage(content="User requests to BUY 1000 shares of GOOG")]
        },
        expected_verdict="REJECTED",
        expected_contains="LIMIT EXCEEDED"
    )
    
    test_case(
        name="NVDA - LIMIT EXCEEDED: Request 500 shares",
        state={
            "ticker": "NVDA",
            "technical_signal": create_tech_signal(ticker="NVDA", confidence=0.80, regime="MANUAL_OVERRIDE"),
            "fundamental_signal": create_fund_signal(),
            "messages": [HumanMessage(content="User requests to BUY 500 shares of NVDA")]
        },
        expected_verdict="REJECTED",
        expected_contains="LIMIT EXCEEDED"
    )
    
    # =========================================================
    # TEST 4: VALID TRADE - Should be approved
    # =========================================================
    for ticker in ["KOLD", "BOIL"]:  # Cheaper ETFs for easier testing
        test_case(
            name=f"{ticker} - VALID: Small trade with good signals",
            state={
                "ticker": ticker,
                "technical_signal": create_tech_signal(ticker=ticker, confidence=0.75, regime="MANUAL_OVERRIDE"),
                "fundamental_signal": create_fund_signal(risk_level="LOW", sentiment=0.3),
                "messages": [HumanMessage(content=f"User requests to BUY 5 shares of {ticker}")]
            },
            expected_verdict="APPROVED",
            expected_contains=None
        )
    
    # =========================================================
    # TEST 5: SELL ORDER - Should bypass soft stops
    # =========================================================
    # Only test if we have a position
    if state.holdings:
        ticker_to_sell = list(state.holdings.keys())[0]
        test_case(
            name=f"{ticker_to_sell} - SELL: Should bypass soft stops",
            state={
                "ticker": ticker_to_sell,
                "technical_signal": create_tech_signal(ticker=ticker_to_sell, confidence=0.30, signal="SELL"),  # Low confidence
                "fundamental_signal": create_fund_signal(risk_level="HIGH", sentiment=-0.8),  # Negative
                "messages": [HumanMessage(content=f"User requests to SELL 1 shares of {ticker_to_sell}")]
            },
            expected_verdict="APPROVED",
            expected_contains=None
        )
    
    # =========================================================
    # TEST 6: Daily P&L Calculation
    # =========================================================
    print(f"\n{'='*60}")
    print("TEST: Daily P&L Calculation")
    print(f"{'='*60}")
    daily_pnl = _calculate_daily_pnl(state)
    print(f"  Today's Realized P&L: ${daily_pnl:,.2f}")
    print(f"  Trade History Count: {len(state.trade_history)}")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
