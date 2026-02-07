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
    RegimeTransition,
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

    def _compute_transition_risk(
        self, indices: MacroIndices, delta: WeeklyDelta
    ) -> RegimeTransition:
        """
        Calculates the probability of a regime shift based on proximity to
        boundaries (Stress=40, Dir=0) and velocity (Delta).
        """
        s = indices.stress_score
        d = indices.direction_score

        # 1. Distances to Boundaries
        # Stress Boundary is 40. Direction Boundary is 0.
        dist_s = abs(s - 40)
        dist_d = abs(d - 0)

        # Normalize distance to a Risk Score (Closer = Higher Risk)
        # We assume a distance of 20 units is "Safe". Distance 0 is "Breaking".
        risk_s = max(
            0, 100 - (dist_s * 5)
        )  # 5x multiplier implies 20 units away = 0 risk
        risk_d = max(
            0, 100 - (dist_d * 2)
        )  # Direction is wider (-100 to 100), so 50 units = 0 risk

        # 2. Velocity Adjustment
        # If we are moving TOWARD the boundary, risk increases.
        # Check Stress Direction
        moving_to_stress_boundary = (s < 40 and delta.stress_chg > 0) or (
            s > 40 and delta.stress_chg < 0
        )
        if moving_to_stress_boundary:
            risk_s *= 1.2

        # Check Direction Boundary
        moving_to_dir_boundary = (d < 0 and delta.direction_chg > 0) or (
            d > 0 and delta.direction_chg < 0
        )
        if moving_to_dir_boundary:
            risk_d *= 1.2

        # 3. Identify Primary Risk Vector
        total_risk = max(risk_s, risk_d)
        total_risk = min(99.0, total_risk)  # Cap at 99

        # 4. Identify Next Regime
        # Hypothetical: If we cross the nearest boundary, where are we?
        next_s = s + (10 if s < 40 else -10)  # Cross S boundary
        next_d = d + (10 if d < 0 else -10)  # Cross D boundary

        # Which boundary is closer?
        next_regime = "Uncertain"
        if risk_s > risk_d:
            # Crossing Stress Boundary
            is_high_now = s > 40
            # If high now, next is low. If low now, next is high.
            target_high = not is_high_now
            target_up = d > 0
            if target_high and target_up:
                next_regime = "USD Wrecking Ball"
            elif target_high and not target_up:
                next_regime = "US-Centric Stress"
            elif not target_high and target_up:
                next_regime = "Tightening / Carry"
            else:
                next_regime = "Reflation / Risk-On"
        else:
            # Crossing Direction Boundary
            target_high = s > 40
            target_up = not (d > 0)
            if target_high and target_up:
                next_regime = "USD Wrecking Ball"
            elif target_high and not target_up:
                next_regime = "US-Centric Stress"
            elif not target_high and target_up:
                next_regime = "Tightening / Carry"
            else:
                next_regime = "Reflation / Risk-On"

        return RegimeTransition(
            risk_score=float(total_risk),
            next_likely_regime=next_regime,
            vector_desc=f"Drifting toward {next_regime}",
            is_breaking=total_risk > 75,
        )

    def _compute_horizon_results(
        self, T_adj: np.ndarray
    ) -> Dict[str, List[HorizonResult]]:
        """Computes SR scores and STRICTLY ranks them 1-8."""
        results = {}
        for h_name, days in self.config.HORIZONS.items():
            gamma = self.math.get_gamma(days)
            M = self.math.compute_sr_matrix(T_adj, gamma)
            scores = self.math.compute_strength_scores(M)

            # Create pairs of (iso, score)
            scored_nodes = []
            for i, iso in enumerate(self.currencies):
                scored_nodes.append({"iso": iso, "score": float(scores[i])})

            # STRICT SORT: Highest score first
            scored_nodes.sort(key=lambda x: x["score"], reverse=True)

            # Assign ranks based on sorted position
            h_list = []
            for rank_idx, node in enumerate(scored_nodes, 1):
                h_list.append(
                    HorizonResult(
                        iso=node["iso"],
                        score=round(node["score"], 4),
                        rank=rank_idx,
                        delta=0.0,
                        trend="Gaining",
                    )
                )
            results[h_name] = h_list
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

        # 1. Instantiate Delta Object (Needed for Transition Risk calculation)
        delta = WeeklyDelta(
            stress_chg=float(idx_rec.stress_score - prev_idx.stress_score),
            direction_chg=float(idx_rec.direction_score - prev_idx.direction_score),
            regime_shift=(regime.label != "Unknown"),
            prev_label=None,
        )

        # 2. Calculate Transition Risk
        transition = self._compute_transition_risk(idx_rec, delta)

        return {
            "date": self._global_df.index[-1].strftime("%Y-%m-%d"),
            "regime": regime.model_dump(),
            "transition": transition.model_dump(),  # <--- Added Field
            "posture": posture.model_dump(),
            "confidence": {
                "score": float(max(0, 100 - (abs(vix_z) * 20))),
                "persistence": 1,
                "is_stable": True,
            },
            "delta": delta.model_dump(),
            "horizons": {
                k: [item.model_dump() for item in v] for k, v in horizons.items()
            },
            "transition": transition.model_dump(),
        }

    def run_simulation(self, beliefs: BeliefParams) -> SimulationOutput:
        """Handles POST /simulate - Now with simulated Posture and Stability."""
        mom = self._feat_dict["mom_21d"].iloc[-1]
        vol = self._feat_dict["volatility"].iloc[-1]
        y_diff = self._yield_diffs.iloc[-1]

        # 1. Logic Overrides
        vix_actual = float(self._global_df["VIX"].iloc[-1])
        vix_to_use = (
            beliefs.vix_override if beliefs.vix_override is not None else vix_actual
        )

        # Calculate a simulated VIX Z-Score for the physics engine
        mean_vix = self._global_df["VIX"].rolling(126).mean().iloc[-1]
        std_vix = self._global_df["VIX"].rolling(126).std().iloc[-1]
        vix_z = (vix_to_use - mean_vix) / (std_vix + 1e-6)

        T_base = self.physics.construct_physics_matrix(mom, vol, y_diff, beliefs)

        # 2. Adjust Simulated Indices
        # If user slides to 'Wrecking Ball', we simulate high stress indices
        idx_rec = self._macro_indices.iloc[-1].copy()
        idx_rec["stress_score"] = beliefs.risk_mix * 100.0
        idx_rec["direction_score"] = (
            beliefs.risk_mix * 50.0
        )  # Assume USD strength in risk-off

        T_adj, _, _, regime = self.physics.apply_adaptive_leakage(
            T_base, self.currencies, vix_z, idx_rec
        )

        # 3. Re-calculate Posture and Horizons for the 'What-If' scenario
        horizons = self._compute_horizon_results(T_adj)
        posture = self._compute_posture(regime)  # <--- NOW SIMULATED

        return SimulationOutput(
            mode="counterfactual",
            params_used=beliefs,
            horizons=horizons,
            posture=posture,  # <--- NOW RETURNED TO FRONTEND
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
