# AlphaCouncil: Autonomous Investment Committee 🏛️

**AlphaCouncil** is an agentic risk-overlay system that orchestrates a "Man-vs-Machine" debate to validate algorithmic trading signals. 

Unlike traditional black-box quant models, AlphaCouncil uses a **Multi-Agent Architecture** (powered by LangGraph) to simulate a hedge fund Investment Committee. It combines deep-learning volatility forecasts with semantic reasoning to filter out false positives caused by event risk (earnings, macro news) or sector concentration.

## 🏗 Architecture

AlphaCouncil operates as a Directed Acyclic Graph (DAG) with three specialized agents:

1.  **The Technician (Quant Agent):**
    * *Role:* Signal detection & Regime classification.
    * *Core Engine:* **VolSense** (Custom PyTorch Volatility Forecaster).
    * *Logic:* Analyzes Term Structure, Z-Scores, and Volatility Cones.
    
2.  **The Fundamentalist (Research Agent):**
    * *Role:* Event Risk & Sentiment analysis.
    * *Tools:* Tavily Search API / RAG.
    * *Logic:* Scans for earnings calls, lawsuits, and macro headwinds to reject "gambling" setups.

3.  **The Risk Manager (CQF Agent):**
    * *Role:* Portfolio construction & Limits.
    * *Logic:* Enforces Sector Limits, Correlation checks, and CVaR constraints.

## 🚀 Quick Start

### Prerequisites
* Python 3.10+
* OpenAI API Key (or Anthropic/Gemini)
* Tavily API Key (for web search)

### Installation
```bash
git clone [https://github.com/rahulmkarthik/AlphaCouncil.git](https://github.com/rahulmkarthik/AlphaCouncil.git)
cd AlphaCouncil
pip install -r requirements.txt
```

### Usage
Run the dashboard to see the agents in action:
```bash
streamlit run app/Home.py
```

## 🧠 Core Technologies

- VolSense: Custom Deep Learning Library for Volatility Forecasting (LSTM/Transformer).

- LangGraph: Stateful multi-agent orchestration.

- Streamlit: Interactive frontend for signal visualization.