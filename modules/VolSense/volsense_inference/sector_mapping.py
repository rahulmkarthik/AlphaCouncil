# ============================================================
# 🧭 VolSense Sector Mappings + Color Maps
# ============================================================
"""
VolSense Inference — Sector mappings and color palettes.

This module centralizes:
  1) Ticker-to-sector dictionaries for different model universes:
     - SECTOR_MAP_109: compact universe used by v109 models
     - SECTOR_MAP_507: extended universe used by v507 models
  2) SECTOR_COLORS: consistent hex colors per sector for plots/dashboards

Helpers:
  - get_sector_map(version): obtain the appropriate ticker→sector map
  - export_to_json(path): write both maps to a JSON file for external use
  - get_color(sector): fetch a stable color for a sector label

Intended for analytics, dashboards, and reporting layers in volsense_inference.
"""
import json
from pathlib import Path

# ------------------------------------------------------------
# 1️⃣  SECTOR MAPS
# ------------------------------------------------------------

SECTOR_MAP_109 = {
    # Index / ETFs
    "SPY": "Index",
    "QQQ": "Index",
    "DIA": "Index",
    "IWM": "Index",
    "VXX": "Volatility",
    "GLD": "Commodities",
    "SLV": "Commodities",
    "TLT": "Fixed Income",
    "HYG": "Fixed Income",
    "EEM": "Index",
    "VIX": "Volatility",
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOG": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "TXN": "Technology",
    "QCOM": "Technology",
    "ADI": "Technology",
    "MU": "Technology",
    "CSCO": "Technology",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "C": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "AXP": "Financials",
    "SCHW": "Financials",
    "CBOE": "Financials",
    # Healthcare
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "MRK": "Healthcare",
    "UNH": "Healthcare",
    "ABBV": "Healthcare",
    "ABT": "Healthcare",
    "LLY": "Healthcare",
    "BMY": "Healthcare",
    "TMO": "Healthcare",
    "CVS": "Healthcare",
    # Energy / Materials
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "HAL": "Energy",
    "BP": "Energy",
    "SHEL": "Energy",
    "BHP": "Materials",
    "RIO": "Materials",
    "FCX": "Materials",
    # Consumer Discretionary
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    "ROST": "Consumer Discretionary",
    # Industrials
    "CAT": "Industrials",
    "BA": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "FDX": "Industrials",
    "UNP": "Industrials",
    "DE": "Industrials",
    "LMT": "Industrials",
    "GE": "Industrials",
    "RTX": "Industrials",
    # Consumer Staples / Retail
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "KMB": "Consumer Staples",
    "GIS": "Consumer Staples",
    "KR": "Consumer Staples",
    "WBA": "Consumer Staples",
    # Utilities / Real Estate
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "XEL": "Utilities",
    "AEP": "Utilities",
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    # Communications / Media
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "NFLX": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "TWX": "Communication Services",
    "PARA": "Communication Services",
    "WBD": "Communication Services",
    # Bonus / Volatile
    "GME": "Consumer Discretionary",
    "AMC": "Consumer Discretionary",
    "PYPL": "Financials",
    "IBM": "Technology",
    "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary",
}


# --- 507-ticker model (v507) --------------------------------
SECTOR_MAP_507 = {
    # Technology
    **{
        t: "Technology"
        for t in [
            "AAPL",
            "MSFT",
            "GOOG",
            "NVDA",
            "META",
            "AVGO",
            "ADBE",
            "CSCO",
            "CRM",
            "AMD",
            "INTC",
            "QCOM",
            "TXN",
            "ORCL",
            "IBM",
            "AMAT",
            "NOW",
            "SNPS",
            "PANW",
            "MU",
            "ADI",
            "LRCX",
            "NXPI",
            "KLAC",
            "WDAY",
            "MCHP",
            "CRWD",
            "CDNS",
            "APH",
            "MSI",
            "FTNT",
            "INTU",
            "ADSK",
            "TEAM",
            "ANET",
            "ZS",
            "DDOG",
            "ENPH",
            "ACN",
            "SHOP",
            "PYPL",
            "SQ",
            "UBER",
            "ABNB",
            "ZM",
            "DOCU",
            "NET",
            "OKTA",
            "HUBS",
            "RBLX",
            "TTD",
            "MDB",
            "SPLK",
            "SNOW",
            "FSLR",
            "TSLA",
            "ON",
            "MRVL",
            "ALTR",
        ]
    },
    # Financials
    **{
        t: "Financials"
        for t in [
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            "MS",
            "BLK",
            "SPGI",
            "AXP",
            "SCHW",
            "BK",
            "TROW",
            "PNC",
            "USB",
            "CB",
            "AIG",
            "CME",
            "ICE",
            "COF",
            "AFL",
            "PAYX",
            "TRV",
            "MET",
            "PRU",
            "ALL",
            "MTB",
            "FITB",
            "CFG",
            "KEY",
            "RF",
            "HBAN",
            "NTRS",
            "STT",
            "ZION",
            "DFS",
            "ALLY",
            "BEN",
            "VIRT",
            "CINF",
            "FNF",
            "RJF",
            "PGR",
            "MMC",
            "AON",
            "LNC",
            "AJG",
            "WRB",
            "MKTX",
            "AMP",
            "FDS",
            "CBOE",
            "MSCI",
            "CASH",
            "EWBC",
            "PNFP",
        ]
    },
    # Healthcare
    **{
        t: "Healthcare"
        for t in [
            "LLY",
            "JNJ",
            "PFE",
            "ABT",
            "TMO",
            "MRK",
            "AMGN",
            "UNH",
            "DHR",
            "BMY",
            "CVS",
            "CI",
            "ISRG",
            "REGN",
            "GILD",
            "VRTX",
            "ZBH",
            "BSX",
            "MDT",
            "SYK",
            "HCA",
            "EW",
            "IDXX",
            "IQV",
            "BDX",
            "BIO",
            "DGX",
            "HOLX",
            "LH",
            "MRNA",
            "ALGN",
            "PODD",
            "GEHC",
            "MTD",
            "RMD",
            "CRL",
            "WAT",
            "TFX",
            "INCY",
            "BAX",
            "ABC",
            "CAH",
            "MCK",
            "ELV",
            "CNC",
            "HUM",
            "EXAS",
            "TECH",
            "SGEN",
            "ILMN",
            "BIOC",
            "STE",
            "HIMS",
            "DOCS",
            "PSTG",
        ]
    },
    # Energy
    **{
        t: "Energy"
        for t in [
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "EOG",
            "PSX",
            "MPC",
            "OXY",
            "HAL",
            "KMI",
            "WMB",
            "BKR",
            "DVN",
            "FANG",
            "VLO",
            "PXD",
            "HES",
            "APA",
            "OVV",
            "CTRA",
            "MUR",
            "CHK",
            "AR",
            "PR",
            "SM",
            "MTDR",
            "RRC",
            "HP",
            "NBR",
            "PTEN",
            "OII",
            "NOV",
            "SLCA",
            "OIS",
            "RES",
            "TRGP",
            "LNG",
            "ET",
            "OKE",
            "MRO",
        ]
    },
    # Industrials
    **{
        t: "Industrials"
        for t in [
            "HON",
            "GE",
            "CAT",
            "BA",
            "UPS",
            "UNP",
            "MMM",
            "ETN",
            "GD",
            "DE",
            "NSC",
            "EMR",
            "PCAR",
            "ITW",
            "LMT",
            "DAL",
            "CSX",
            "WM",
            "ROK",
            "AME",
            "FAST",
            "PH",
            "IR",
            "FDX",
            "TXT",
            "UAL",
            "CMI",
            "CARR",
            "OTIS",
            "BALL",
            "WAB",
            "PWR",
            "MAS",
            "LEN",
            "NUE",
            "DHI",
            "FLS",
            "JCI",
            "VMC",
            "MLM",
            "URI",
            "HWM",
            "CTAS",
            "TT",
            "RHI",
            "ROL",
            "GWW",
            "ALK",
            "HUBB",
            "XYL",
        ]
    },
    # Consumer Discretionary
    **{
        t: "Consumer Discretionary"
        for t in [
            "AMZN",
            "HD",
            "MCD",
            "NKE",
            "LOW",
            "SBUX",
            "TJX",
            "TGT",
            "MAR",
            "BKNG",
            "GM",
            "F",
            "YUM",
            "ROST",
            "EBAY",
            "ORLY",
            "AZO",
            "DG",
            "DLTR",
            "ULTA",
            "CMG",
            "DRI",
            "BBY",
            "RL",
            "TPR",
            "DKS",
            "EXPE",
            "CCL",
            "RCL",
            "NCLH",
            "MGM",
            "WYNN",
            "LVS",
            "HLT",
            "H",
            "CZR",
            "ADNT",
            "HAS",
            "MAT",
            "ETSY",
            "COST",
            "POOL",
            "NVR",
            "WHR",
            "CHDN",
            "PVH",
            "SIG",
            "WSM",
            "KMX",
            "CVNA",
        ]
    },
    # Consumer Staples
    **{
        t: "Consumer Staples"
        for t in [
            "PG",
            "KO",
            "PEP",
            "WMT",
            "COST",
            "MDLZ",
            "PM",
            "MO",
            "KMB",
            "CL",
            "HSY",
            "K",
            "GIS",
            "SYY",
            "TSN",
            "CAG",
            "CPB",
            "CHD",
            "TAP",
            "KDP",
            "EL",
            "CLX",
            "HRL",
            "KR",
            "WBA",
            "ACI",
            "BJ",
            "STZ",
            "BF.B",
            "ADM",
            "BG",
            "LW",
            "MKC",
            "TGT",
            "DG",
            "KHC",
            "SAM",
            "MNST",
            "CHWY",
            "UL",
        ]
    },
    # Utilities
    **{
        t: "Utilities"
        for t in [
            "NEE",
            "DUK",
            "SO",
            "D",
            "AEP",
            "EXC",
            "SRE",
            "XEL",
            "PEG",
            "ED",
            "WEC",
            "PCG",
            "EIX",
            "PPL",
            "DTE",
            "ES",
            "CMS",
            "AEE",
            "LNT",
            "ATO",
            "NI",
            "NRG",
            "AWK",
            "EVRG",
            "AES",
        ]
    },
    # Materials
    **{
        t: "Materials"
        for t in [
            "LIN",
            "APD",
            "SHW",
            "ECL",
            "NEM",
            "DD",
            "MLM",
            "VMC",
            "BALL",
            "CE",
            "ALB",
            "FCX",
            "NUE",
            "STLD",
            "CF",
            "MOS",
            "LYB",
            "EMN",
            "AVY",
            "PKG",
            "IP",
            "WRK",
            "IFF",
            "RPM",
            "FMC",
        ]
    },
    # Real Estate
    **{
        t: "Real Estate"
        for t in [
            "PLD",
            "AMT",
            "CCI",
            "EQIX",
            "PSA",
            "SPG",
            "O",
            "DLR",
            "VTR",
            "WELL",
            "AVB",
            "EQR",
            "UDR",
            "MAA",
            "EXR",
            "ARE",
            "SBAC",
            "IRM",
            "CBRE",
            "WY",
            "INVH",
            "BXP",
            "REG",
            "KIM",
            "VNO",
        ]
    },
    # Index / ETF
    **{
        t: "Index/ETF"
        for t in [
            "SPY",
            "QQQ",
            "DIA",
            "IWM",
            "VOO",
            "VTI",
            "VTV",
            "VUG",
            "IWF",
            "IWD",
            "EEM",
            "EWJ",
            "EWU",
            "EWG",
            "EWZ",
            "FXI",
            "VGK",
            "EWW",
            "EWT",
            "EWS",
            "XLE",
            "XLF",
            "XLK",
            "XLV",
            "XLY",
            "XLP",
            "XLU",
            "XLI",
            "XLB",
            "XOP",
        ]
    },
    # Commodities
    **{
        t: "Commodities"
        for t in [
            "GLD",
            "SLV",
            "USO",
            "UNG",
            "DBC",
            "GDX",
            "GDXJ",
            "DBA",
            "PPLT",
            "PALL",
            "URA",
            "CORN",
            "JO",
            "CPER",
            "SIL",
            "NIB",
            "KOLD",
            "BOIL",
            "UCO",
            "BNO",
            "WEAT",
            "COW",
            "SOYB",
            "SPYV",
            "SPYG",
        ]
    },
    # Fixed Income
    **{
        t: "Fixed Income"
        for t in [
            "TLT",
            "IEF",
            "SHY",
            "LQD",
            "HYG",
            "BND",
            "AGG",
            "TIP",
            "JNK",
            "EMB",
            "BIL",
            "VGSH",
            "GOVT",
            "MUB",
            "BSV",
            "ZROZ",
            "SPTL",
            "BIV",
            "SCHZ",
            "VCIT",
        ]
    },
    # FX
    **{
        t: "FX / Currency"
        for t in ["UUP", "FXE", "FXY", "FXB", "FXA", "FXC", "CEW", "CYB", "EUO", "YCS"]
    },
    # Crypto
    **{
        t: "Crypto / Blockchain"
        for t in [
            "BTC-USD",
            "ETH-USD",
            "GBTC",
            "BITO",
            "COIN",
            "MARA",
            "RIOT",
            "HUT",
            "WGMI",
            "BTDR",
        ]
    },
    # Volatility / Hedge
    **{
        t: "Volatility / Hedge"
        for t in [
            "VXX",
            "UVXY",
            "SVXY",
            "VIXY",
            "VIXM",
            "SPXU",
            "SDS",
            "SH",
            "DOG",
            "SQQQ",
        ]
    },
}


# ------------------------------------------------------------
# 2️⃣  UTILITY HELPERS
# ------------------------------------------------------------


def get_sector_map(version: str = "v109") -> dict[str, str]:
    """
    Return the ticker-to-sector mapping for a given model version.

    :param version: Model version key ('v109' for small map, 'v507' for large map).
    :type version: str
    :raises ValueError: If the version key is not recognized.
    :return: Dictionary mapping ticker symbols to sector names.
    :rtype: dict[str, str]
    """
    version = version.lower()
    if version in ("v109", "109", "small"):
        return SECTOR_MAP_109
    elif version in ("v507", "507", "large", "volnetx"):
        return SECTOR_MAP_507
    else:
        raise ValueError(f"Unknown model version: {version}")


def get_ticker_type_map(version: str = "v507") -> dict[str, str]:
    """
    Categorize each ticker into a high-level type (e.g. 'Equity', 'ETF', 'Crypto').

    :param version: Sector map version to use ('v109' or 'v507').
    :return: Dictionary mapping ticker → ticker_type
    """
    sector_map = get_sector_map(version)
    ticker_type = {}

    for t, sector in sector_map.items():
        if "Crypto" in sector:
            ticker_type[t] = "Crypto"
        elif "Volatility" in sector:
            ticker_type[t] = "Hedge"
        elif "Index" in sector or "ETF" in sector:
            ticker_type[t] = "ETF"
        elif "FX" in sector:
            ticker_type[t] = "FX"
        elif "Fixed Income" in sector:
            ticker_type[t] = "Bond"
        else:
            ticker_type[t] = "Equity"

    return ticker_type


def export_to_json(path="sector_map.json"):
    """
    Export available sector maps to a single JSON file.

    The JSON contains two top-level keys: "v109" and "v507".

    :param path: Filesystem path to write the JSON file.
    :type path: str
    :return: None
    :rtype: None
    """
    combined = {"v109": SECTOR_MAP_109, "v507": SECTOR_MAP_507}
    Path(path).write_text(json.dumps(combined, indent=2))
    print(f"💾 Sector mappings exported to {path}")


# ------------------------------------------------------------
# 3️⃣  COLOR MAPS (for plots / dashboards)
# ------------------------------------------------------------
SECTOR_COLORS = {
    "Technology": "#1f77b4",
    "Financials": "#ff7f0e",
    "Healthcare": "#2ca02c",
    "Energy": "#d62728",
    "Industrials": "#9467bd",
    "Consumer Discretionary": "#8c564b",
    "Consumer Staples": "#e377c2",
    "Utilities": "#7f7f7f",
    "Materials": "#bcbd22",
    "Real Estate": "#17becf",
    "Communication Services": "#9edae5",
    "Commodities": "#d9b38c",
    "Index/ETF": "#aec7e8",
    "Volatility": "#c49c94",
    "Fixed Income": "#98df8a",
    "Crypto / Blockchain": "#ff9896",
    "FX / Currency": "#c7c7c7",
    "Volatility / Hedge": "#dbdb8d",
    "Unknown": "#dddddd",
}


def get_color(sector: str) -> str:
    """
    Get a consistent hex color for a given sector label.

    :param sector: Sector name (e.g., 'Technology', 'Financials').
    :type sector: str
    :return: Hex color string (e.g., '#1f77b4'); defaults to light gray if unknown.
    :rtype: str
    """
    """Return a consistent color hex code for the given sector."""
    return SECTOR_COLORS.get(sector, "#cccccc")
