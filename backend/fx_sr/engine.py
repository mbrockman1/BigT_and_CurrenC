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

        # Internal state to hold pre-processed data
        self._global_df = None
        self._feat_dict = None
        self._macro_indices = None

        # Initial data load
        self.refresh_data()

    def refresh_data(self):
        self._global_df = self.loader.fetch_history()
        self._feat_dict, _ = self.features.compute_features(
            self._global_df, self.currencies
        )
        self._macro_indices = self.features.compute_macro_regime_indices(
            self._global_df, self.currencies
        )

    def _compute_horizon_results(
        self, T_adj: np.ndarray
    ) -> Dict[str, List[HorizonResult]]:
        results = {}
        for h_name, days in self.config.HORIZONS.items():
            gamma = self.math.get_gamma(days)
            M = self.math.compute_sr_matrix(T_adj, gamma)
            scores = self.math.compute_strength_scores(M)
            ranks = np.argsort(scores)[::-1]
            h_list = []
            for i, iso in enumerate(self.currencies):
                h_list.append(
                    HorizonResult(
                        iso=iso,
                        score=round(float(scores[i]), 4),
                        rank=int(np.where(ranks == i)[0][0] + 1),
                        delta=0.0,
                        trend="Gaining",
                    )
                )
            h_list.sort(key=lambda x: x.rank)
            results[h_name] = h_list
        return results

    def run_eod_cycle(self) -> Dict[str, Any]:
        if self._macro_indices is None:
            self.refresh_data()

        # 1. Latest Data
        idx_rec = self._macro_indices.iloc[-1]
        prev_idx = (
            self._macro_indices.iloc[-2] if len(self._macro_indices) > 1 else idx_rec
        )

        mom = self._feat_dict["mom_21d"].iloc[-1]
        vol = self._feat_dict["volatility"].iloc[-1]
        vix = float(self._global_df["VIX"].iloc[-1])

        # 2. Physics
        T_base = self.physics.construct_scenario_matrix(mom, vol, vix, BeliefParams())
        T_adj, _, _, regime = self.physics.apply_regime_physics(
            T_base, self.currencies, idx_rec
        )

        # 3. History check for persistence (Simplified for EOD: look back 3 weeks)
        # In a real DB, you'd query the persistent state. Here we approximate.
        # We classify the previous week to see if it changed.
        _, _, _, prev_regime = self.physics.apply_regime_physics(
            T_base, self.currencies, prev_idx
        )

        persistence = 1  # Default
        if prev_regime.label == regime.label:
            persistence = 2  # Placeholder logic

        # 4. Interpretability Layer
        posture = self._compute_posture(regime)
        conf = self._compute_confidence(regime, prev_regime.label, persistence)
        delta = WeeklyDelta(
            stress_chg=float(idx_rec.stress_score - prev_idx.stress_score),
            direction_chg=float(idx_rec.direction_score - prev_idx.direction_score),
            regime_shift=(regime.label != prev_regime.label),
            prev_label=prev_regime.label,
        )

        horizons = self._compute_horizon_results(T_adj)

        return {
            "date": self._global_df.index[-1].strftime("%Y-%m-%d"),
            "regime": regime.model_dump(),
            "posture": posture.model_dump(),
            "confidence": conf.model_dump(),
            "delta": delta.model_dump(),
            "horizons": {
                k: [item.model_dump() for item in v] for k, v in horizons.items()
            },
        }

        def run_eod_cycle(self) -> Dict[str, Any]:
            # 1. Fetch latest
            idx_rec = self._macro_indices.iloc[-1]
            prev_idx = (
                self._macro_indices.iloc[-2]
                if len(self._macro_indices) > 1
                else idx_rec
            )

            mom = self._feat_dict["mom_21d"].iloc[-1]
            vol = self._feat_dict["volatility"].iloc[-1]
            vix = float(self._global_df["VIX"].iloc[-1])

            # 2. Physics
            T_base = self.physics.construct_scenario_matrix(
                mom, vol, vix, BeliefParams()
            )
            T_adj, _, _, regime = self.physics.apply_regime_physics(
                T_base, self.currencies, idx_rec
            )

            # 3. Interpretations
            posture = self._compute_posture(regime)
            conf = RegimeConfidence(
                score=max(0, 100 - (regime.indices.stress_score * 0.5)),
                persistence=1,
                is_stable=True,
            )
            delta = WeeklyDelta(
                stress_chg=float(idx_rec.stress_score - prev_idx.stress_score),
                direction_chg=float(idx_rec.direction_score - prev_idx.direction_score),
                regime_shift=(regime.label != "Unknown"),  # Simplified
                prev_label=None,
            )

            # 4. Horizons
            horizons = {}
            for h_name, days in self.config.HORIZONS.items():
                M = self.math.compute_sr_matrix(T_adj, self.math.get_gamma(days))
                scores = self.math.compute_strength_scores(M)
                ranks = np.argsort(scores)[::-1]
                horizons[h_name] = [
                    HorizonResult(
                        iso=self.currencies[i],
                        score=round(float(scores[i]), 4),
                        rank=int(np.where(ranks == i)[0][0] + 1),
                        delta=0,
                        trend="Gaining",
                    )
                    for i in range(len(self.currencies))
                ]

            # 5. Final Package (Matches EngineOutput schema)
            return {
                "date": self._global_df.index[-1].strftime("%Y-%m-%d"),
                "regime": regime.model_dump(),
                "posture": posture.model_dump(),
                "confidence": conf.model_dump(),
                "delta": delta.model_dump(),
                "horizons": {
                    k: [item.model_dump() for item in v] for k, v in horizons.items()
                },
            }

    def run_simulation(self, beliefs: BeliefParams) -> SimulationOutput:
        # Simplified for validation stability
        current_mom = self._feat_dict["mom_21d"].iloc[-1]
        current_vol = self._feat_dict["volatility"].iloc[-1]
        vix = (
            beliefs.vix_override
            if beliefs.vix_override
            else float(self._global_df["VIX"].iloc[-1])
        )
        T = self.physics.construct_scenario_matrix(
            current_mom, current_vol, vix, beliefs
        )
        T_adj, _, _, _ = self.physics.apply_regime_physics(
            T, self.currencies, self._macro_indices.iloc[-1]
        )

        results = {}
        for h_name, days in self.config.HORIZONS.items():
            M = self.math.compute_sr_matrix(T_adj, self.math.get_gamma(days))
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
        return SimulationOutput(
            mode="counterfactual", params_used=beliefs, horizons=results, posture=None
        )

    def run_rolling_analysis(self, lookback_window: int = 90) -> RollingOutput:
        history = []
        dates = self._global_df.index[-lookback_window:]
        for date in dates:
            history.append(
                RollingDataPoint(
                    date=date.strftime("%Y-%m-%d"),
                    rankings={c: 1 for c in self.currencies},
                    top_iso=self.currencies[0],
                    regime_vix=20.0,
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
        label = regime.label
        stress = regime.indices.stress_score

        p = {
            "usd_view": "Neutral",
            "fx_risk": "Selective",
            "carry_view": "Neutral",
            "hedging": "Light",
            "trust_ranking": True,
        }

        if label == "USD Wrecking Ball":
            p = {
                "usd_view": "Overweight",
                "fx_risk": "Defensive",
                "carry_view": "Avoid",
                "hedging": "Heavy",
                "trust_ranking": True,
            }
        elif label == "Reflation / Risk-On":
            p = {
                "usd_view": "Underweight",
                "fx_risk": "Aggressive",
                "carry_view": "Favor",
                "hedging": "None",
                "trust_ranking": True,
            }
        elif label == "Tightening / Carry":
            p["carry_view"] = "Favor"
            p["trust_ranking"] = False

        if stress > 70:
            p["fx_risk"] = "Prohibited"
            p["hedging"] = "Maximum"
        return ModelPosture(**p)

    def run_simulation(self, beliefs: BeliefParams) -> SimulationOutput:
        # (Standard simulation logic from previous steps)
        # Just ensure the return matches the new schema (posture can be None)
        # ...
        current_mom = self._feat_dict["mom_21d"].iloc[-1]
        current_vol = self._feat_dict["volatility"].iloc[-1]
        vix = (
            beliefs.vix_override
            if beliefs.vix_override
            else float(self._global_df["VIX"].iloc[-1])
        )
        T = self.physics.construct_scenario_matrix(
            current_mom, current_vol, vix, beliefs
        )
        # Dummy indices
        idx_rec = self._macro_indices.iloc[-1].copy()
        if beliefs.vix_override:
            idx_rec["stress_score"] = min(100.0, beliefs.vix_override * 2.5)

        T_adj, _, _, _ = self.physics.apply_regime_physics(T, self.currencies, idx_rec)
        horizons = self._compute_horizon_results(T_adj)

        # Create a dummy posture based on the simulated regime logic
        # (Simplified: we won't fully recalculate regime object here for brevity)
        return SimulationOutput(
            mode="counterfactual", params_used=beliefs, horizons=horizons, posture=None
        )

    def run_rolling_analysis(self, lookback_window: int = 90) -> RollingOutput:
        """
        Generates historical ranking time-series for the evolution chart (/history).
        """
        if self._global_df is None:
            self.refresh_data()

        history = []
        dates = self._global_df.index[-lookback_window:]

        for date in dates:
            if date not in self._macro_indices.index:
                continue

            # For history, we compute a quick 'Medium' horizon rank
            mom = self._feat_dict["mom_21d"].loc[date]
            vol = self._feat_dict["volatility"].loc[date]
            vix = float(self._global_df.loc[date, "VIX"])

            T_base = self.physics.construct_scenario_matrix(
                mom, vol, vix, BeliefParams()
            )
            T_adj, _, _, _ = self.physics.apply_regime_physics(
                T_base, self.currencies, self._macro_indices.loc[date]
            )

            # SR for 63 days
            gamma = self.math.get_gamma(63)
            M = self.math.compute_sr_matrix(T_adj, gamma)
            scores = self.math.compute_strength_scores(M)

            ranks = np.argsort(scores)[::-1]
            rank_map = {
                self.currencies[i]: int(np.where(ranks == i)[0][0] + 1)
                for i in range(len(self.currencies))
            }

            history.append(
                RollingDataPoint(
                    date=date.strftime("%Y-%m-%d"),
                    rankings=rank_map,
                    top_iso=self.currencies[ranks[0]],
                    regime_vix=vix,
                )
            )

        return RollingOutput(history=history)
