"""The investable universe: ~130 diverse ETFs (distinct exposures) + the S&P 500.

Curated for *distinct* exposures, not raw count: near-duplicate ETFs (SPY/VOO/IVV
are all the S&P 500) make the covariance matrix singular and the optimizer pile into
one of them arbitrarily. One representative per exposure; breadth comes from spanning
sectors, regions, factors, the whole bond curve, commodities, and real assets, then
adding the S&P 500 single names for single-stock richness (~630 tickers total).

download/estimation lives in data.py; this module just defines *what* to pull.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path(__file__).parent / "data_cache"
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# category -> {ticker: short name}.  One clean representative per exposure.
ETFS: dict[str, dict[str, str]] = {
    "US Sectors": {
        "XLB": "Materials", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials",
        "XLK": "Technology", "XLP": "Cons Staples", "XLU": "Utilities", "XLV": "Health Care",
        "XLY": "Cons Discretionary", "XLRE": "Real Estate (sector)", "XLC": "Comm Services",
    },
    "US Broad / Style": {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000",
        "VTI": "US Total Market", "MDY": "S&P MidCap 400", "IJR": "S&P SmallCap 600",
        "IWF": "R1000 Growth", "IWD": "R1000 Value",
    },
    "Factors": {
        "MTUM": "Momentum", "QUAL": "Quality", "USMV": "Min Volatility", "VLUE": "Value factor",
        "SIZE": "Size factor", "SPLV": "Low Volatility", "VIG": "Dividend Growth",
        "VYM": "High Dividend", "NOBL": "Dividend Aristocrats", "COWZ": "Cash Cows",
    },
    "Intl Developed": {
        "EFA": "EAFE", "VEA": "Developed ex-US", "VGK": "Europe", "EZU": "Eurozone",
        "EWU": "UK", "EWG": "Germany", "EWQ": "France", "EWL": "Switzerland",
        "EWJ": "Japan", "DXJ": "Japan (hedged)", "EWA": "Australia", "EWC": "Canada",
        "EWY": "South Korea", "EWT": "Taiwan", "EWH": "Hong Kong", "EWS": "Singapore",
    },
    "Emerging Markets": {
        "EEM": "EM broad", "VWO": "EM (Vanguard)", "FXI": "China large-cap", "MCHI": "China",
        "KWEB": "China internet", "INDA": "India", "EWZ": "Brazil", "EWW": "Mexico",
        "EZA": "South Africa", "EIDO": "Indonesia", "THD": "Thailand",
    },
    "Treasuries": {
        "SHV": "T-bills", "SHY": "1-3y UST", "IEI": "3-7y UST", "IEF": "7-10y UST",
        "TLH": "10-20y UST", "TLT": "20y+ UST", "GOVT": "US Treasury broad",
    },
    "Credit": {
        "LQD": "IG corporate", "VCSH": "Short corporate", "HYG": "High yield",
        "JNK": "High yield (SPDR)", "ANGL": "Fallen angels", "BKLN": "Bank loans",
        "EMB": "EM USD bonds", "EMLC": "EM local bonds",
    },
    "Bonds Other": {
        "AGG": "US Aggregate", "BNDX": "Intl bonds", "TIP": "TIPS", "VTIP": "Short TIPS",
        "MUB": "Municipals", "MBB": "MBS", "FLOT": "Floating rate", "CWB": "Convertibles",
    },
    "Commodities": {
        "DBC": "Broad commodity", "PDBC": "Broad commodity (no K-1)", "GLD": "Gold",
        "SLV": "Silver", "PPLT": "Platinum", "USO": "Oil (WTI)", "BNO": "Brent",
        "UNG": "Natural gas", "DBA": "Agriculture", "CPER": "Copper",
    },
    "Real Estate": {
        "VNQ": "US REITs", "IYR": "US real estate", "VNQI": "Intl REITs", "REM": "Mortgage REITs",
    },
    "Thematic / Industry": {
        "SOXX": "Semiconductors", "IBB": "Biotech", "XBI": "Biotech (equal-wt)", "SKYY": "Cloud",
        "BOTZ": "Robotics/AI", "ICLN": "Clean energy", "TAN": "Solar", "LIT": "Lithium",
        "JETS": "Airlines", "GDX": "Gold miners", "XOP": "Oil & gas E&P", "KRE": "Regional banks",
        "ITB": "Homebuilders", "ARKK": "Disruptive innovation", "HACK": "Cybersecurity",
    },
}


def etf_rows() -> list[dict]:
    rows = []
    for cat, d in ETFS.items():
        for tkr, name in d.items():
            rows.append({"ticker": tkr, "name": name, "group": "ETF", "category": cat})
    return rows


def sp500_rows(refresh: bool = False) -> list[dict]:
    """S&P 500 constituents from the datahub CSV, cached locally."""
    CACHE.mkdir(exist_ok=True)
    local = CACHE / "sp500.csv"
    if local.exists() and not refresh:
        df = pd.read_csv(local)
    else:
        df = pd.read_csv(SP500_URL)
        df.to_csv(local, index=False)
    return [
        {"ticker": r["Symbol"].replace(".", "-"),   # BRK.B -> BRK-B for yfinance
         "name": r["Security"], "group": "Stock",
         "category": "S&P 500 — " + str(r.get("GICS Sector", "n/a"))}
        for _, r in df.iterrows()
    ]


def build_universe(include_stocks: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Full universe as a DataFrame[ticker, name, group, category], deduped on ticker."""
    rows = etf_rows() + (sp500_rows(refresh) if include_stocks else [])
    return pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)


if __name__ == "__main__":
    u = build_universe()
    print(f"universe: {len(u)} tickers "
          f"({(u.group == 'ETF').sum()} ETFs + {(u.group == 'Stock').sum()} stocks)")
    print(u.groupby("group").size().to_string())
    print("\nETF categories:")
    print(u[u.group == "ETF"].groupby("category").size().to_string())
