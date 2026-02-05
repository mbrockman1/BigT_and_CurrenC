from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class FeatureEngine:
    """
    Computes macro features: volatility + momentum + USD Pressure.
    """

    @staticmethod
    def compute_features(
        df: pd.DataFrame, universe_cols: List[str]
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        features = {}
        log_rets = np.log(df[universe_cols] / df[universe_cols].shift(1))

        # 1. Volatility (Risk)
        features["volatility"] = log_rets.rolling(21).std() * np.sqrt(252)

        # 2. Momentum (Secondary Driver)
        features["mom_21d"] = df[universe_cols].pct_change(21)

        return features, log_rets

    @staticmethod
    def compute_usd_pressure(
        df: pd.DataFrame, universe_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calculates 'USD Pressure Index' (0-100).
        High = USD Wrecking Ball.
        """
        # 1. Calculate Daily Log Returns (Value in USD)
        # If EUR goes down, USD is getting stronger.
        # We assume universe_cols are already "Value of 1 unit in USD"
        rets = np.log(df[universe_cols] / df[universe_cols].shift(1))

        # 2. Breadth: % of currencies down vs USD
        breadth = (rets < 0).mean(axis=1)

        # 3. Magnitude: Average return of basket (inverted)
        magnitude = -rets.mean(axis=1)

        # 4. Volatility (Dispersion)
        dispersion = rets.std(axis=1)

        # 5. Composite Score (Heuristic weighting)
        # Normalize roughly to 0-1 range before smoothing
        raw_score = (
            (breadth * 0.4) + (magnitude * 50.0 * 0.4) + (dispersion * 50.0 * 0.2)
        )

        # Smooth and Scale to 0-100
        pressure = raw_score.rolling(5).mean().clip(0, 1) * 100

        return pd.DataFrame(
            {
                "level": pressure,
                "breadth": breadth.rolling(5).mean(),
                "magnitude": magnitude.rolling(5).mean(),
            },
            index=df.index,
        )

    @staticmethod
    def compute_carry_scores(
        yields: pd.DataFrame, universe_cols: List[str]
    ) -> pd.DataFrame:
        diffs = yields[universe_cols].sub(yields["USD"], axis=0)
        z_scores = diffs.sub(diffs.mean(axis=1), axis=0).div(
            diffs.std(axis=1) + 1e-6, axis=0
        )
        return z_scores.clip(-3, 3)

    @staticmethod
    def compute_macro_regime_indices(
        df: pd.DataFrame, universe_cols: List[str]
    ) -> pd.DataFrame:
        # ... (Keep existing implementation for Breadth/Direction) ...
        # Simplified for brevity here, assume same logic as before
        rets = np.log(df[universe_cols] / df[universe_cols].shift(1))
        breadth = (rets < 0).mean(axis=1)
        magnitude = rets.abs().mean(axis=1)
        dispersion = rets.std(axis=1)

        raw_stress = (breadth * 0.3) + (magnitude * 35.0) + (dispersion * 35.0)
        stress_idx = raw_stress.rolling(5).mean().clip(0, 1) * 100

        dxy_ret = (
            df["DXY"].pct_change(5).fillna(0)
            if "DXY" in df.columns
            else pd.Series(0, index=df.index)
        )
        yield_chg = (
            df["RiskFree"].diff(5).fillna(0)
            if "RiskFree" in df.columns
            else pd.Series(0, index=df.index)
        )

        direction_idx = ((dxy_ret * 50.0) + (yield_chg * 100.0)).clip(-1, 1) * 100

        return pd.DataFrame(
            {
                "stress_score": stress_idx,
                "direction_score": direction_idx,
                "stress_breadth": breadth.rolling(5).mean(),
                "stress_vol": dispersion.rolling(5).mean(),
                "usd_mom": dxy_ret,
                "yield_delta": yield_chg,
            },
            index=df.index,
        )

    @staticmethod
    def compute_yield_differentials(
        yields: pd.DataFrame, universe_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calculates Real Yield Spread vs USD.
        Positive = Higher Yield than USD (Attractor).
        Negative = Lower Yield than USD (Repulsor).
        """
        if "USD" not in yields.columns:
            return pd.DataFrame(0.0, index=yields.index, columns=universe_cols)

        # Spread = Local - USD
        diffs = yields[universe_cols].sub(yields["USD"], axis=0)
        return diffs

    @staticmethod
    def compute_adaptive_tuning(series: pd.Series, window: int = 126) -> pd.Series:
        """
        Calculates Rolling Z-Score for adaptive constant tuning.
        (Value - Mean) / Std
        """
        roll_mean = series.rolling(window).mean()
        roll_std = series.rolling(window).std()
        z_score = (series - roll_mean) / (roll_std + 1e-6)
        return z_score.fillna(0)
