# fx_sr/config.py
class FXConfig:
    """
    Defines the fixed universe and time horizons.
    """

    """
    Defines the fixed universe, time horizons, and data sources.
    """

    # Nodes: graph states (currencies).
    UNIVERSE = {
        "EUR": {"ticker": "EURUSD=X", "inverted": False},
        "GBP": {"ticker": "GBPUSD=X", "inverted": False},
        "AUD": {"ticker": "AUDUSD=X", "inverted": False},
        "NZD": {"ticker": "NZDUSD=X", "inverted": False},
        "JPY": {"ticker": "JPY=X", "inverted": True},
        "CHF": {"ticker": "CHF=X", "inverted": True},
        "CAD": {"ticker": "CAD=X", "inverted": True},
        # "XAU": {"ticker": "GC=F", "inverted": False},  # Gold Futures
        # "SPY": {"ticker": "SPY", "inverted": False},  # S&P 500 ETF
    }

    # Regime variables (context, not nodes)
    REGIME_TICKERS = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "XAU": "GC=F",
        "SPY": "SPY",
    }

    # NEW: Yield Proxies (2Y or similar short-end rates)
    # Note: Free data for INT 2Y is hard to find. We use these or fallback.
    YIELD_TICKERS = {
        "USD": "^TNX",  # US 10Y
        "EUR": "0P0000JW1Q.F",  # German Bund 10Y (Frankfurt)
        "GBP": "0P00007P6E.L",  # UK Gilt 10Y (London)
        "JPY": "0P0000XW95.T",  # Japan 10Y (Tokyo)
        "AUD": "0P0000XW6S.S",  # Australia 10Y (Sydney)
        "CAD": "0P0000XW77.TO",  # Canada 10Y (Toronto)
        "CHF": "0P0000XW9B.SW",  # Swiss 10Y (Zurich)
        "NZD": "0P0000XW84.NZ",  # New Zealand 10Y (Auckland)
    }

    # Horizons in trading days
    HORIZONS = {
        "short": 10,
        "medium": 63,
        "long": 252,
    }

    # Safe havens for regime logic
    SAFE_HAVENS = {"JPY", "CHF", "USD"}
