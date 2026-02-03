import numpy as np
import pandas as pd
import yfinance as yf

from .config import FXConfig


class MarketDataLoader:
    def __init__(self, config: FXConfig):
        self.config = config

    def fetch_history(self, lookback_days: int = 500) -> pd.DataFrame:
        all_tickers = [meta["ticker"] for meta in self.config.UNIVERSE.values()]
        all_tickers += list(self.config.REGIME_TICKERS.values())

        print(f"Fetching {len(all_tickers)} tickers from Yahoo Finance...")
        # auto_adjust=True avoids most 'NoneType' price issues
        raw = yf.download(
            all_tickers,
            period=f"{lookback_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )["Close"]

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = pd.DataFrame(index=raw.index)

        # 1. Process Currencies
        for iso, meta in self.config.UNIVERSE.items():
            ticker = meta["ticker"]
            if ticker in raw.columns:
                series = raw[ticker]
                # INSTEAD OF DROPNA: Forward fill missing data points (like the CHF timeout)
                # then backfill any early-history gaps
                series = series.ffill().bfill()
                df[iso] = (1.0 / series) if meta["inverted"] else series
            else:
                print(f"Warning: {iso} ({ticker}) missing from download.")

        # 2. Process Regime Vars
        for key, ticker in self.config.REGIME_TICKERS.items():
            if ticker in raw.columns:
                df[key] = raw[ticker].ffill().bfill()

        # 3. Handle Yields (RiskFree)
        usd_yield_ticker = self.config.YIELD_TICKERS["USD"]
        if usd_yield_ticker in raw.columns:
            df["RiskFree"] = raw[usd_yield_ticker].ffill().bfill()
        else:
            # Synthetic fallback if yield download fails
            df["RiskFree"] = 4.5

        # Drop only if the entire row is empty
        df = df.dropna(how="all").sort_index()

        if df.empty:
            raise ValueError(
                "Data download resulted in an empty DataFrame. Check internet connection."
            )

        return df

    def fetch_yields(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        # Simple synthetic map to ensure logic works even if bond data times out
        yields = pd.DataFrame(index=dates)
        base = 4.5  # Fallback
        if "RiskFree" in self.config.YIELD_TICKERS:
            # Logic to fetch or use pre-downloaded RiskFree
            pass

        yields["USD"] = 4.5
        yields["JPY"] = 0.1
        yields["CHF"] = 1.5
        yields["EUR"] = 3.0
        yields["GBP"] = 5.0
        yields["AUD"] = 4.3
        yields["NZD"] = 5.5
        yields["CAD"] = 4.5
        return yields
