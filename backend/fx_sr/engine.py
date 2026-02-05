from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import FXConfig
from .data_loader import MarketDataLoader
from .features import FeatureEngine
from .schemas import (
    BeliefParams,
    HorizonResult,
    MacroIndices,
    ModelPosture,
    RegimeConfidence,
    RegimeData,
    RollingDataPoint,
    RollingOutput,
    SimulationOutput,
    WeeklyDelta,
)
from .sr import SREngine
from .transitions import TransitionModel


class FXMacroEngine:
    def __init__(self):
        self.config = FXConfig()
        self.loader = MarketDataLoader(self.config)
        self.features = FeatureEngine()
        self.physics = TransitionModel(self.config)
        self.math = SREngine(self.config)
        self.currencies = list(self.config.UNIVERSE.keys())
        self.refresh_data()

    def refresh_data(self):
        """Pre-computes all inputs for the physics engine."""
        self._global_df = self.loader.fetch_history(lookback_days=750)
        self._feat_dict, _ = self.features.compute_features(
            self._global_df, self.currencies
        )
        self._macro_indices = self.features.compute_macro_regime_indices(
            self._global_df, self.currencies
        )

        # New Physics Inputs
        self._yields = self.loader.fetch_yields(self._global_df.index)
        self._yield_diffs = self.features.compute_yield_differentials(
            self._yields, self.currencies
        )
        self._vix_z = self.features.compute_adaptive_tuning(self._global_df["VIX"])

        print(f"Engine Ready. Records: {len(self._global_df)}")

    def _compute_horizon_results(
        self, T_adj: np.ndarray
    ) -> Dict[str, List[HorizonResult]]:
        results = {}
        for h_name, days in self.config.HORIZONS.items():
            gamma = self.math.get_gamma(days)
            M = self.math.compute_sr_matrix(T_adj, gamma)
            scores = self.math.compute_strength_scores(M)
            ranks = np.argsort(scores)[::-1]
            results[h_name] = [
                HorizonResult(
                    iso=self.currencies[i],
                    score=round(float(scores[i]), 4),
                    rank=int(np.where(ranks == i)[0][0] + 1),
                    delta=0,
                    trend="Gaining",
                )
                for i in range(len(self.currencies))
            ]
        return results

    def run_eod_cycle(self) -> Dict[str, Any]:
        """Calculates /latest dashboard."""
        idx_rec = self._macro_indices.iloc[-1]
        prev_idx = (
            self._macro_indices.iloc[-2] if len(self._macro_indices) > 1 else idx_rec
        )

        # Physics Inputs
        mom = self._feat_dict["mom_21d"].iloc[-1]
        vol = self._feat_dict["volatility"].iloc[-1]
        y_diff = self._yield_diffs.iloc[-1]
        vix_z = self._vix_z.iloc[-1]

        T_base = self.physics.construct_physics_matrix(mom, vol, y_diff, BeliefParams())
        T_adj, leakage, net_flow, regime = self.physics.apply_adaptive_leakage(
            T_base, self.currencies, vix_z, idx_rec
        )

        horizons = self._compute_horizon_results(T_adj)
        posture = self._compute_posture(regime)

        return {
            "date": self._global_df.index[-1].strftime("%Y-%m-%d"),
            "regime": regime.model_dump(),
            "posture": posture.model_dump(),
            "confidence": {
                "score": float(max(0, 100 - (abs(vix_z) * 20))),
                "persistence": 1,
                "is_stable": True,
            },
            "delta": {
                "stress_chg": float(idx_rec.stress_score - prev_idx.stress_score),
                "direction_chg": float(
                    idx_rec.direction_score - prev_idx.direction_score
                ),
                "regime_shift": (regime.label != "Unknown"),
                "prev_label": None,
            },
            "horizons": {
                k: [item.model_dump() for item in v] for k, v in horizons.items()
            },
        }

    def run_simulation(self, beliefs: BeliefParams) -> SimulationOutput:
        """Calculates POST /simulate."""
        mom = self._feat_dict["mom_21d"].iloc[-1]
        vol = self._feat_dict["volatility"].iloc[-1]
        y_diff = self._yield_diffs.iloc[-1]

        # Recalculate VIX Z for simulation override
        vix_actual = float(self._global_df["VIX"].iloc[-1])
        vix_to_use = (
            beliefs.vix_override if beliefs.vix_override is not None else vix_actual
        )

        # Approx Z-score update
        mean_vix = self._global_df["VIX"].rolling(126).mean().iloc[-1]
        std_vix = self._global_df["VIX"].rolling(126).std().iloc[-1]
        vix_z = (vix_to_use - mean_vix) / (std_vix + 1e-6)

        T_base = self.physics.construct_physics_matrix(mom, vol, y_diff, beliefs)

        # Adjust stress score for regime label
        idx_rec = self._macro_indices.iloc[-1].copy()
        if beliefs.vix_override:
            idx_rec["stress_score"] = min(100.0, beliefs.vix_override * 2.5)

        T_adj, _, _, _ = self.physics.apply_adaptive_leakage(
            T_base, self.currencies, vix_z, idx_rec
        )

        return SimulationOutput(
            mode="counterfactual",
            params_used=beliefs,
            horizons=self._compute_horizon_results(T_adj),
            posture=None,
        )

    def run_rolling_analysis(self, lookback_window: int = 90) -> RollingOutput:
        """Calculates GET /history."""
        history = []
        dates = self._global_df.index[-lookback_window:]

        for date in dates:
            if date not in self._macro_indices.index:
                continue

            # --- FIX: Retrieve Historical Physics Inputs ---
            mom = self._feat_dict["mom_21d"].loc[date]
            vol = self._feat_dict["volatility"].loc[date]
            y_diff = self._yield_diffs.loc[date]
            vix_z = self._vix_z.loc[date]
            idx_rec = self._macro_indices.loc[date]

            # Run Physics
            T_base = self.physics.construct_physics_matrix(
                mom, vol, y_diff, BeliefParams()
            )
            T_adj, _, _, _ = self.physics.apply_adaptive_leakage(
                T_base, self.currencies, vix_z, idx_rec
            )

            # SR (Medium Term)
            M = self.math.compute_sr_matrix(T_adj, self.math.get_gamma(63))
            scores = self.math.compute_strength_scores(M)
            ranks = np.argsort(scores)[::-1]

            history.append(
                RollingDataPoint(
                    date=date.strftime("%Y-%m-%d"),
                    rankings={
                        self.currencies[i]: int(np.where(ranks == i)[0][0] + 1)
                        for i in range(len(self.currencies))
                    },
                    top_iso=self.currencies[ranks[0]],
                    regime_vix=float(self._global_df.loc[date, "VIX"]),
                )
            )

        return RollingOutput(history=history)

    def _compute_confidence(
        self, regime: RegimeData, prev_regime_label: str, persistence: int
    ) -> RegimeConfidence:
        """
        Confidence = (100 - Stress/2) penalized by regime flipping
        """
        stress = regime.indices.stress_score
        base_score = max(0, 100 - (stress * 0.4))

        # Penalize if regime just changed
        if regime.label != prev_regime_label and prev_regime_label != "":
            base_score *= 0.8

        # Reward persistence
        base_score = min(100, base_score + (persistence * 2))

        return RegimeConfidence(
            score=round(base_score, 1),
            persistence=persistence,
            is_stable=base_score > 60,
        )

    def _compute_posture(self, regime: RegimeData) -> ModelPosture:
        label, indices = regime.label, regime.indices
        stress, direction = indices.stress_score, indices.direction_score

        p = {
            "usd_view": "Neutral",
            "fx_risk": "Selective",
            "carry_view": "Neutral",
            "hedging": "Light",
            "trust_ranking": True,
        }

        if label == "USD Wrecking Ball":
            p.update(
                {
                    "usd_view": "Overweight",
                    "fx_risk": "Defensive",
                    "carry_view": "Avoid",
                    "hedging": "Heavy",
                }
            )
        elif label == "US-Centric Stress":
            p.update(
                {
                    "usd_view": "Underweight",
                    "fx_risk": "Defensive",
                    "carry_view": "Avoid",
                    "hedging": "Moderate",
                }
            )
        elif label == "Reflation / Risk-On":
            p.update(
                {
                    "usd_view": "Underweight",
                    "fx_risk": "Aggressive",
                    "carry_view": "Favor",
                    "hedging": "None",
                }
            )
        elif label == "Tightening / Carry":
            p.update(
                {
                    "usd_view": "Overweight",
                    "fx_risk": "Selective",
                    "carry_view": "Maximum Favor",
                    "hedging": "Light",
                    "trust_ranking": False,
                }
            )

        if stress > 75:
            p.update({"fx_risk": "Prohibited", "hedging": "Maximum"})
        if direction > 60:
            p["usd_view"] = "Maximum Overweight"
        elif direction < -60:
            p["usd_view"] = "Maximum Underweight"

        return ModelPosture(**p)

    def run_rolling_analysis(self, lookback_window: int = 90) -> RollingOutput:
        """Calculates GET /history with robust error handling."""
        history = []
        dates = self._global_df.index[-lookback_window:]

        for date in dates:
            if date not in self._macro_indices.index:
                continue

            try:
                # Retrieve inputs with 0.0 fill to prevent NaN crashes
                mom = self._feat_dict["mom_21d"].loc[date].fillna(0)
                vol = self._feat_dict["volatility"].loc[date].fillna(0)
                y_diff = self._yield_diffs.loc[date].fillna(0)
                vix_z = self._vix_z.loc[date]
                # Handle scalar NaN
                if pd.isna(vix_z):
                    vix_z = 0.0

                idx_rec = self._macro_indices.loc[date]

                # Run Physics
                T_base = self.physics.construct_physics_matrix(
                    mom, vol, y_diff, BeliefParams()
                )
                T_adj, _, _, _ = self.physics.apply_adaptive_leakage(
                    T_base, self.currencies, vix_z, idx_rec
                )

                # SR (Medium Term)
                M = self.math.compute_sr_matrix(T_adj, self.math.get_gamma(63))
                scores = self.math.compute_strength_scores(M)
                ranks = np.argsort(scores)[::-1]

                history.append(
                    RollingDataPoint(
                        date=date.strftime("%Y-%m-%d"),
                        rankings={
                            self.currencies[i]: int(np.where(ranks == i)[0][0] + 1)
                            for i in range(len(self.currencies))
                        },
                        top_iso=self.currencies[ranks[0]],
                        regime_vix=float(self._global_df.loc[date, "VIX"]),
                    )
                )
            except Exception as e:
                # Skip bad days silently to keep the timeline intact
                continue

        return RollingOutput(history=history)
