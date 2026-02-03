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
        features["volatility"] = log_rets.rolling(21).std() * np.sqrt(252)
        features["mom_5d"] = df[universe_cols].pct_change(5)
        features["mom_21d"] = df[universe_cols].pct_change(21)
        features["mom_63d"] = df[universe_cols].pct_change(63)
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
