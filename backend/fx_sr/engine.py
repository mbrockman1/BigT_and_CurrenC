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
        Translates Regime + Beliefs into Strategy Posture using dynamic scaling.
        Gauges now move based on raw scores, not just discrete labels.
        """
        indices = regime.indices
        stress = indices.stress_score
        direction = indices.direction_score

        # 1. DYNAMIC GAUGE CALCULATION (The "Live" Logic)
        # USD Gauge: Center at 50, moves to 100 with DXY strength, 0 with weakness
        usd_val = int(np.clip(50 + (direction * 0.5), 0, 100))

        # Risk Budget: Shrinks as Stress rises
        risk_val = int(np.clip(100 - stress, 0, 100))

        # Hedging Pressure: Increases with Stress
        hedge_val = int(np.clip(stress, 0, 100))

        # Carry Viability: High when Stress is low AND Direction is positive (USD Carry)
        carry_val = int(
            np.clip((100 - stress) * (1.0 if direction > 0 else 0.5), 0, 100)
        )

        # 2. Base Posture Structure
        p = {
            "headline": regime.label,
            "usd_gauge": usd_val,
            "risk_gauge": risk_val,
            "carry_gauge": carry_val,
            "hedging_gauge": hedge_val,
            "rewards": ["Yield" if direction > 0 else "Momentum"],
            "penalties": ["Volatility"],
            "interp": regime.desc,
            "trust_ranking": True,
        }

        # 3. Fine-tune Headlines and Forces based on Quadrants
        if regime.label == "USD Wrecking Ball":
            p.update(
                {
                    "headline": "Defensive Lockdown",
                    "rewards": ["USD Liquidity", "Vol"],
                    "penalties": ["High Beta", "Carry"],
                }
            )
        elif regime.label == "US-Centric Stress":
            p.update(
                {
                    "headline": "Safety Rotation",
                    "rewards": ["JPY/CHF", "Gold"],
                    "penalties": ["USD Assets"],
                }
            )
        elif regime.label == "Reflation / Risk-On":
            p.update(
                {
                    "headline": "Broad Expansion",
                    "rewards": ["Commodity FX", "Growth"],
                    "penalties": ["Cash", "USD"],
                }
            )
        elif regime.label == "Tightening / Carry":
            p.update(
                {
                    "headline": "Yield Seeking",
                    "rewards": ["Carry Spreads", "USD Yields"],
                    "penalties": ["Funding FX"],
                    "trust_ranking": False,
                }
            )

        # 4. Apply Slider Overrides (Belief Injection)
        if beliefs:
            # Vol Penalty moves Hedging up further
            vol_mod = (beliefs.vol_penalty - 1.0) * 15
            p["hedging_gauge"] = int(np.clip(p["hedging_gauge"] + vol_mod, 0, 100))
            p["risk_gauge"] = int(np.clip(p["risk_gauge"] - vol_mod, 0, 100))

            # Trend Slider moves Risk budget
            trend_mod = (beliefs.trend_sensitivity - 1.0) * 15
            p["risk_gauge"] = int(np.clip(p["risk_gauge"] + trend_mod, 0, 100))

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

    def _get_signal_direction(self, regime_label: str) -> float:
        """
        Determines if we should invert the SR Score.
        -1.0 = Invert (Fade Crowding/Traps)
        +1.0 = Trust (Buy Safety/Sinks)

        Logic:
        In 'US-Centric Stress', the model correctly identifies Safety (CHF/JPY). Trust it.
        In all other regimes (especially Wrecking Ball), high scores indicate Crowding. Fade it.
        """
        if regime_label == "US-Centric Stress":
            return 1.0  # Buy Safety
        return -1.0  # Fade Crowding

    def _compute_horizon_results(
        self, T_adj: np.ndarray, regime_label: str
    ) -> Dict[str, List[HorizonResult]]:
        results = {}

        # Get the Smart Switch Factor
        signal_dir = self._get_signal_direction(regime_label)

        for h_name, days in self.config.HORIZONS.items():
            gamma = self.math.get_gamma(days)
            M = self.math.compute_sr_matrix(T_adj, gamma)
            raw_scores = self.math.compute_strength_scores(M)

            # APPLY SMART SWITCH
            final_scores = raw_scores * signal_dir

            scored_nodes = [
                {"iso": iso, "score": float(final_scores[i])}
                for i, iso in enumerate(self.currencies)
            ]
            scored_nodes.sort(key=lambda x: x["score"], reverse=True)

            results[h_name] = [
                HorizonResult(
                    iso=n["iso"],
                    score=round(n["score"], 4),
                    rank=i,
                    delta=0.0,
                    trend="Gaining",
                )
                for i, n in enumerate(scored_nodes, 1)
            ]

        return results

    def run_eod_cycle(self) -> Dict[str, Any]:
        if self._global_df is None:
            self.refresh_data()

        idx_rec = self._macro_indices.iloc[-1]
        prev_idx = (
            self._macro_indices.iloc[-2] if len(self._macro_indices) > 1 else idx_rec
        )
        date_str = self._global_df.index[-1].strftime("%Y-%m-%d")

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

        # Pass regime label to handle Smart Switch
        horizons = self._compute_horizon_results(T_adj, regime.label)
        transition = self._compute_transition_risk(idx_rec, diff)

        return {
            "date": date_str,
            "regime": regime.model_dump(),
            "transition": transition.model_dump(),
            "posture": posture.model_dump(),
            "confidence": {"score": conf_score, "persistence": 1, "is_stable": True},
            "diff": diff.model_dump(),
            "horizons": {
                k: [item.model_dump() for item in v] for k, v in horizons.items()
            },
        }

    def _compute_transition_risk(
        self, indices: Any, delta: DailyDiff
    ) -> RegimeTransition:
        s = float(indices["stress_score"])
        d = float(indices["direction_score"])
        risk_s = max(0, 100 - (abs(s - 40) * 5))
        risk_d = max(0, 100 - (abs(d - 0) * 2))

        if (s < 40 and delta.stress_delta > 0) or (s > 40 and delta.stress_delta < 0):
            risk_s *= 1.2
        if (d < 0 and delta.direction_delta > 0) or (
            d > 0 and delta.direction_delta < 0
        ):
            risk_d *= 1.2

        total_risk = float(min(99.0, max(risk_s, risk_d)))

        next_regime = "Uncertain"
        if risk_s > risk_d:
            target_high = not (s > 40)
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
            risk_score=total_risk,
            next_likely_regime=next_regime,
            vector_desc=f"Drifting toward {next_regime}",
            is_breaking=total_risk > 75,
        )

    def run_simulation(self, beliefs: BeliefParams) -> SimulationOutput:
        """Handles POST /simulate - Ensures all sliders trigger updates."""
        # Standard Data Fetch
        mom, vol, y_diff = (
            self._feat_dict["mom_21d"].iloc[-1],
            self._feat_dict["volatility"].iloc[-1],
            self._yield_diffs.iloc[-1],
        )

        # 1. Dynamic VIX Z-Score (driven by Risk Mix)
        # We simulate VIX Z from -1.5 (Risk-On) to +3.5 (Crisis)
        vix_z_sim = (beliefs.risk_mix - 0.5) * 6.0

        # 2. Physics Matrix (Uses Trend/Vol sliders)
        T_base = self.physics.construct_physics_matrix(
            mom, vol, y_diff, beliefs, vix_z_sim
        )

        # 3. Simulated Indices (Quadrant positions)
        idx_rec = self._macro_indices.iloc[-1].copy()
        idx_rec["stress_score"] = float(np.clip(beliefs.risk_mix * 100, 0, 100))
        idx_rec["direction_score"] = float((beliefs.risk_mix - 0.5) * 100)

        T_adj, _, _, regime = self.physics.apply_adaptive_leakage(
            T_base, self.currencies, vix_z_sim, idx_rec
        )

        # 4. Generate Output (Passing beliefs to the posture logic)
        posture = self._compute_institutional_posture(regime, beliefs)
        horizons = self._compute_horizon_results(T_adj, regime.label)

        # Compute Transition Risk for the radar
        real_indices = self._macro_indices.iloc[-1]
        sim_delta = DailyDiff(
            stress_delta=float(idx_rec["stress_score"] - real_indices["stress_score"]),
            direction_delta=float(
                idx_rec["direction_score"] - real_indices["direction_score"]
            ),
            regime_changed=(regime.label != "Neutral"),
            prev_regime=None,
            confidence_delta=0.0,
        )
        transition = self._compute_transition_risk(idx_rec, sim_delta)

        return SimulationOutput(
            mode="counterfactual",
            params_used=beliefs,
            horizons=horizons,
            posture=posture,
            transition=transition,
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
        """Calculates GET /history with robust error handling and Smart Switch logic."""
        history = []
        dates = self._global_df.index[-lookback_window:]

        for date in dates:
            if date not in self._macro_indices.index:
                continue

            try:
                # 1. Retrieve inputs
                mom = self._feat_dict["mom_21d"].loc[date].fillna(0)
                vol = self._feat_dict["volatility"].loc[date].fillna(0)
                y_diff = self._yield_diffs.loc[date].fillna(0)
                vix_z = self._vix_z.loc[date]
                if pd.isna(vix_z):
                    vix_z = 0.0

                idx_rec = self._macro_indices.loc[date]

                # 2. Run Physics
                T_base = self.physics.construct_physics_matrix(
                    mom, vol, y_diff, BeliefParams(), vix_z
                )

                # Capture 'regime' here to feed the Smart Switch
                T_adj, _, _, regime = self.physics.apply_adaptive_leakage(
                    T_base, self.currencies, vix_z, idx_rec
                )

                # 3. Compute SR (Medium Term)
                M = self.math.compute_sr_matrix(T_adj, self.math.get_gamma(63))
                raw_scores = self.math.compute_strength_scores(M)

                # --- APPLY THE SMART SWITCH ---
                # This ensures the history matches the 'Tradeable' logic established in the audit
                signal_dir = self._get_signal_direction(regime.label)
                final_scores = raw_scores * signal_dir
                # ------------------------------

                # 4. Rank based on the Strategy Scores (Final Scores)
                ranks = np.argsort(final_scores)[::-1]

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
                # Log errors for visibility during development
                print(f"Rolling analysis error on {date}: {e}")
                continue

        return RollingOutput(history=history)
