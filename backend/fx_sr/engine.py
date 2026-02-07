import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FXConfig
from .data_loader import MarketDataLoader
from .features import FeatureEngine
from .schemas import (
    BeliefParams,
    DailyDiff,
    HorizonResult,
    InstitutionalPosture,
    MacroIndices,
    MarketForces,
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

STATE_FILE = "state_store.json"


class FXMacroEngine:
    def __init__(self):
        self.config = FXConfig()
        self.loader = MarketDataLoader(self.config)
        self.features = FeatureEngine()
        self.physics = TransitionModel(self.config)
        self.math = SREngine(self.config)
        self.currencies = list(self.config.UNIVERSE.keys())
        self.refresh_data()

    def _load_prev_state(self) -> Dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_state(
        self, date: str, regime_label: str, stress: float, direction: float, conf: float
    ):
        state = {
            "date": date,
            "regime": regime_label,
            "stress": stress,
            "direction": direction,
            "confidence": conf,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)

    def refresh_data(self):
        """Pre-computes all inputs for the physics engine."""
        self._global_df = self.loader.fetch_history(lookback_days=750)
        self._feat_dict, _ = self.features.compute_features(
            self._global_df, self.currencies
        )
        self._macro_indices = self.features.compute_macro_regime_indices(
            self._global_df, self.currencies
        )
        self._yields = self.loader.fetch_yields(self._global_df.index)
        self._yield_diffs = self.features.compute_yield_differentials(
            self._yields, self.currencies
        )
        self._vix_z = self.features.compute_adaptive_tuning(self._global_df["VIX"])
        print(f"Engine Ready. Records: {len(self._global_df)}")

    def _compute_institutional_posture(
        self, regime: RegimeData, beliefs: Optional[BeliefParams] = None
    ) -> InstitutionalPosture:
        """
        Translates Regime + Beliefs into Strategy Posture.
        Now sensitive to Vol Penalty and Trend Persistence sliders.
        """
        label = regime.label
        stress = regime.indices.stress_score

        # 1. Base Posture from Regime
        p = {
            "headline": "Neutral Rotation",
            "usd_gauge": 50,
            "risk_gauge": 50,
            "carry_gauge": 50,
            "hedging_gauge": 20,
            "rewards": ["Selectivity"],
            "penalties": ["Indecision"],
            "interp": "Markets are transitioning. Maintain balanced exposure.",
            "trust_ranking": True,
        }

        if label == "USD Wrecking Ball":
            p.update(
                {
                    "headline": "Defensive Lockdown",
                    "usd_gauge": 90,
                    "risk_gauge": 15,
                    "carry_gauge": 5,
                    "hedging_gauge": 95,
                    "rewards": ["USD Liquidity", "Volatility Longs"],
                    "penalties": ["High Beta", "Carry"],
                    "interp": "USD acts as the sole safe haven. Reduce all non-USD structural exposure.",
                }
            )
        elif label == "US-Centric Stress":
            p.update(
                {
                    "headline": "Safety Rotation",
                    "usd_gauge": 25,
                    "risk_gauge": 30,
                    "carry_gauge": 10,
                    "hedging_gauge": 65,
                    "rewards": ["Gold", "Swiss Franc", "JPY"],
                    "penalties": ["USD Longs", "Banking Sector"],
                    "interp": "Capital flees USD due to domestic stress. Rotate into traditional havens.",
                }
            )
        elif label == "Reflation / Risk-On":
            p.update(
                {
                    "headline": "Broad Expansion",
                    "usd_gauge": 20,
                    "risk_gauge": 85,
                    "carry_gauge": 80,
                    "hedging_gauge": 10,
                    "rewards": ["Growth FX (AUD/NZD)", "Risk"],
                    "penalties": ["Cash", "USD Longs"],
                    "interp": "USD weakens structurally. Maximize exposure to growth-linked currencies.",
                }
            )
        elif label == "Tightening / Carry":
            p.update(
                {
                    "headline": "Selective Yield Seeking",
                    "usd_gauge": 75,
                    "risk_gauge": 55,
                    "carry_gauge": 95,
                    "hedging_gauge": 30,
                    "rewards": ["Carry Spreads", "USD Yields"],
                    "penalties": ["Funding Currencies"],
                    "interp": "Yield differentials dominate. Long USD vs JPY/CHF.",
                    "trust_ranking": False,
                }
            )

        if stress > 75:
            p.update(
                {"headline": "LIQUIDITY CRISIS", "risk_gauge": 0, "hedging_gauge": 100}
            )

        # 2. Apply Belief Injection Overrides (The fix for static gauges)
        if beliefs:
            # Volatility Penalty: Higher = More Hedging, Less Risk
            if beliefs.vol_penalty > 1.0:
                p["hedging_gauge"] = min(
                    100, p["hedging_gauge"] + int((beliefs.vol_penalty - 1.0) * 30)
                )
                p["risk_gauge"] = max(
                    0, p["risk_gauge"] - int((beliefs.vol_penalty - 1.0) * 30)
                )
                if beliefs.vol_penalty > 2.0:
                    p["headline"] += " (Vol Filtered)"

            # Trend Persistence: Higher = More Risk (Trusting the trend), Less Hedging
            if beliefs.trend_sensitivity > 1.0:
                p["risk_gauge"] = min(
                    100, p["risk_gauge"] + int((beliefs.trend_sensitivity - 1.0) * 20)
                )
                p["hedging_gauge"] = max(
                    0, p["hedging_gauge"] - int((beliefs.trend_sensitivity - 1.0) * 10)
                )

            # Trend Persistence: Lower = Mean Reversion (Less Risk, More Hedging)
            if beliefs.trend_sensitivity < 1.0:
                p["risk_gauge"] = max(
                    0, p["risk_gauge"] - int((1.0 - beliefs.trend_sensitivity) * 30)
                )
                p["headline"] += " (Mean Reversion)"

        return InstitutionalPosture(
            headline=p["headline"],
            usd_gauge=p["usd_gauge"],
            risk_gauge=p["risk_gauge"],
            carry_gauge=p["carry_gauge"],
            hedging_gauge=p["hedging_gauge"],
            forces=MarketForces(rewarding=p["rewards"], penalizing=p["penalties"]),
            interpretation=p["interp"],
            trust_ranking=p["trust_ranking"],
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

            scored_nodes = [
                {"iso": iso, "score": float(scores[i])}
                for i, iso in enumerate(self.currencies)
            ]
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
        if self._global_df is None:
            self.refresh_data()
        idx_rec = self._macro_indices.iloc[-1]
        date_str = self._global_df.index[-1].strftime("%Y-%m-%d")
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
        posture = self._compute_institutional_posture(regime)

        conf_score = float(max(0, 100 - (abs(vix_z) * 20)))
        prev_state = self._load_prev_state()
        diff = DailyDiff(
            stress_delta=float(
                idx_rec.stress_score - prev_state.get("stress", idx_rec.stress_score)
            ),
            direction_delta=float(
                idx_rec.direction_score
                - prev_state.get("direction", idx_rec.direction_score)
            ),
            regime_changed=(regime.label != prev_state.get("regime", regime.label)),
            prev_regime=prev_state.get("regime", None),
            confidence_delta=float(
                conf_score - prev_state.get("confidence", conf_score)
            ),
        )

        self._save_state(
            date_str,
            regime.label,
            float(idx_rec.stress_score),
            float(idx_rec.direction_score),
            conf_score,
        )

        horizons = self._compute_horizon_results(T_adj)

        # 1. Instantiate Delta Object (Needed for Transition Risk calculation)
        delta = WeeklyDelta(
            stress_delta=float(idx_rec.stress_score - prev_idx.stress_score),
            direction_delta=float(idx_rec.direction_score - prev_idx.direction_score),
            regime_shift=(regime.label != "Unknown"),
            prev_label=None,
        )

        # 2. Calculate Transition Risk
        transition = self._compute_transition_risk(
            idx_rec, diff
        )  # Assuming helper exists or inline it

        return {
            "date": date_str,
            "regime": regime.model_dump(),
            "transition": transition.model_dump(),  # <--- Added Field
            "posture": posture.model_dump(),
            "confidence": {
                "score": float(conf_score),
                "persistence": 1,
                "is_stable": True,
            },
            "delta": {
                "stress_delta": diff.stress_delta,
                "direction_delta": diff.direction_delta,
                "regime_shift": diff.regime_changed,
                "prev_label": diff.prev_regime,
            },
            "horizons": {
                k: [item.model_dump() for item in v] for k, v in horizons.items()
            },
            "diff": diff.model_dump(),  # New Field
        }

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
        moving_to_stress_boundary = (s < 40 and delta.stress_delta > 0) or (
            s > 40 and delta.stress_delta < 0
        )
        if moving_to_stress_boundary:
            risk_s *= 1.2

        # Check Direction Boundary
        moving_to_dir_boundary = (d < 0 and delta.direction_delta > 0) or (
            d > 0 and delta.direction_delta < 0
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

    def run_simulation(self, beliefs: BeliefParams) -> SimulationOutput:
        """Updated to ensure sliders actually change the posture."""
        mom = self._feat_dict["mom_21d"].iloc[-1]
        vol = self._feat_dict["volatility"].iloc[-1]
        y_diff = self._yield_diffs.iloc[-1]

        # 1. Force VIX Z-Score based on Risk Slider
        # 0.5 = Neutral (0.0). 0.0 = Calm (-2.0). 1.0 = Panic (+3.0)
        # This ensures we actually CROSS the threshold for Reflation/Wrecking Ball
        vix_z_forced = (beliefs.risk_mix - 0.5) * 6.0

        T_base = self.physics.construct_physics_matrix(mom, vol, y_diff, beliefs)

        # 2. Force Indices based on Risk Slider
        idx_rec = self._macro_indices.iloc[-1].copy()

        # If slider < 0.4, Force Reflation (Low Stress, Negative Direction)
        if beliefs.risk_mix < 0.4:
            idx_rec["stress_score"] = 20.0
            idx_rec["direction_score"] = -50.0
        # If slider > 0.6, Force Wrecking Ball (High Stress, Positive Direction)
        elif beliefs.risk_mix > 0.6:
            idx_rec["stress_score"] = 80.0
            idx_rec["direction_score"] = 50.0
        else:
            # Neutral zone, stick closer to actuals but bias slightly
            pass

        T_adj, _, _, regime = self.physics.apply_adaptive_leakage(
            T_base, self.currencies, vix_z_forced, idx_rec
        )

        horizons = self._compute_horizon_results(T_adj)

        # 3. Pass beliefs to posture computation so sliders affect gauges directly
        posture = self._compute_institutional_posture(regime, beliefs)

        return SimulationOutput(
            mode="counterfactual",
            params_used=beliefs,
            horizons=horizons,
            posture=posture,
        )

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
