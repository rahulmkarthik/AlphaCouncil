from alphacouncil.graph import app

result = app.invoke({"ticker": "NVDA"})
print("--- TECH SIGNAL ---\n", result['technical_signal'])
print("\n--- FUND SIGNAL ---\n", result['fundamental_signal'])
print("\n--- RISK ASSESSMENT ---\n", result['risk_assessment'])