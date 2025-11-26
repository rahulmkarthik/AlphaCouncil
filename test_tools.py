# test_tools.py
from alphacouncil.tools.vol_tools import get_volatility_forecast, get_sector_trends

print("--- Testing Ticker Forecast ---")
# This should trigger the model load (might take 10-20s)
result = get_volatility_forecast.invoke({"ticker": "SPY"})
print(result)

print("\n--- Testing Sector Trends ---")
# This should use the already loaded model
result_sector = get_sector_trends.invoke({})
print(result_sector)