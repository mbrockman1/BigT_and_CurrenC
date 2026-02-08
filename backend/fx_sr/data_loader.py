import numpy as np
import pandas as pd
import yfinance as yf

from .config import FXConfig


class MarketDataLoader:
    def __init__(self, config: FXConfig):
        self.config = config

    def fetch_history(self, lookback_days: int = 750) -> pd.DataFrame:
        """
        Fetches EOD FX rates, Regime variables, and the Global Yield Anchor (^TNX).
        """
        # 1. Collect all required tickers
        all_tickers = [meta["ticker"] for meta in self.config.UNIVERSE.values()]
        all_tickers += list(self.config.REGIME_TICKERS.values())

        # Add the Global Yield Anchor (US 10-Year Treasury)
        yield_anchor = "^TNX"
        if yield_anchor not in all_tickers:
            all_tickers.append(yield_anchor)

        print(
            f"Fetching {len(all_tickers)} tickers {[_ for _ in all_tickers]} from Yahoo Finance..."
        )

        # Download data
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

        # 2. Process Currencies (Normalize to 'Value in USD')
        for iso, meta in self.config.UNIVERSE.items():
            ticker = meta["ticker"]
            if ticker in raw.columns:
                # Forward fill gaps (like timeouts or holidays) to prevent NaN crashes
                series = raw[ticker].ffill().bfill()
                df[iso] = (1.0 / series) if meta["inverted"] else series
            else:
                print(f"Warning: {iso} missing from download.")

        # 3. Process Regime Variables (VIX, DXY)
        for key, ticker in self.config.REGIME_TICKERS.items():
            if ticker in raw.columns:
                df[key] = raw[ticker].ffill().bfill()

        # 4. Handle Global Yield Anchor (RiskFree / Gravity Base)
        if yield_anchor in raw.columns:
            df["RiskFree"] = raw[yield_anchor].ffill().bfill()
        else:
            # Hard fallback if download fails (approximate 10Y yield)
            df["RiskFree"] = 4.2

        # Final cleanup
        df = df.dropna(how="all").sort_index()

        if df.empty:
            raise ValueError("Download resulted in empty DataFrame. Check connection.")

        return df

    def fetch_yields(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Constructs a dynamic Incentive Map.
        """
        try:
            usd_yield = yf.download(
                "^TNX",
                start=dates[0],
                end=dates[-1],
                progress=False,
                auto_adjust=False,
            )["Close"]
            # FIX: Ensure it is a percentage (e.g. 4.2 not 42.0 or 0.042)
            if usd_yield.mean() > 20:
                usd_yield /= 10.0
            usd_yield = usd_yield.reindex(dates).ffill().bfill()
        except:
            usd_yield = pd.Series(4.2, index=dates)

        yields = pd.DataFrame(index=dates)
        yields["USD"] = usd_yield

        # Structural Spreads (The "Gravity")
        yields["JPY"] = 0.9
        yields["CHF"] = 0.7
        yields["EUR"] = usd_yield - 1.5
        yields["GBP"] = usd_yield + 0.2
        yields["AUD"] = usd_yield + 0.3
        yields["NZD"] = usd_yield + 0.6
        yields["CAD"] = usd_yield - 0.3

        # yields["XAU"] = 0.0  # Gold pays no rent
        # yields["SPY"] = 1.5  # Proxy for S&P 500 Dividend Yield

        return yields
