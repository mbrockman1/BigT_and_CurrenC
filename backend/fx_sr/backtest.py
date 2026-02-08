import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import FXConfig
from .data_loader import MarketDataLoader
from .features import FeatureEngine
from .schemas import (
    BeliefParams,
    CarryData,
    RegimeTransition,
    ValidationMetrics,
    WalkForwardOutput,
    WalkForwardSnapshot,
    WeeklyDelta,
)
from .sr import SREngine
from .transitions import TransitionModel


class WalkForwardEngine:
    def __init__(self):
        self.config = FXConfig()
        self.loader = MarketDataLoader(self.config)
        self.features = FeatureEngine()
        self.physics = TransitionModel(self.config)
        self.math = SREngine(self.config)
        self.currencies = list(self.config.UNIVERSE.keys())

        print("Backtester: Loading History...")
        self._df = self.loader.fetch_history(lookback_days=750)
        self._feats, _ = self.features.compute_features(self._df, self.currencies)

        # Pre-compute Physics Inputs
        self._yields = self.loader.fetch_yields(self._df.index)
        self._yield_diffs = self.features.compute_yield_differentials(
            self._yields, self.currencies
        )
        self._vix_z = self.features.compute_adaptive_tuning(self._df["VIX"])
        self._carry_z = self.features.compute_carry_scores(
            self._yields, self.currencies
        )

    def _get_top_edges(
        self, T: np.ndarray, leakage_nodes: List[Any], k: int = 2
    ) -> List[dict]:
        edges = []
        n = len(self.currencies)
        T_viz = T.copy()
        np.fill_diagonal(T_viz, 0)

        for i in range(n):
            top_indices = np.argsort(T_viz[i])[-k:]
            for j in top_indices:
                weight = float(T_viz[i, j])
                if weight > 0.005:
                    edges.append(
                        {
                            "source": self.currencies[i],
                            "target": self.currencies[j],
                            "weight": weight,
                        }
                    )

        for node in leakage_nodes:
            # Handle both object and dict access for robustness
            prob = (
                node.leakage_prob
                if hasattr(node, "leakage_prob")
                else node.get("leakage_prob", 0)
            )
            iso = node.iso if hasattr(node, "iso") else node.get("iso", "")
            if prob > 0.02:
                edges.append({"source": iso, "target": "USD", "weight": float(prob)})
        return edges

    def _compute_transition_risk(
        self, indices: Any, delta: WeeklyDelta
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

    def print_performance_report(
        self, history: List[WalkForwardSnapshot], horizon: str = "medium"
    ):
        """Reporter with dict-safe access."""
        print(
            f"\n{'=' * 60}\nFX SR MODEL PERFORMANCE REPORT ({horizon.upper()})\n{'=' * 60}"
        )

        valid_ics = []
        valid_blended = []
        tight_ics = []
        tight_blended = []

        for s in history:
            # Pydantic models in history might be dicts or objects depending on version/init
            m = s.metrics.get(horizon)
            if m is None:
                continue

            # Extract values safely
            if isinstance(m, dict):
                ic = m.get("rank_ic")
                sr_ic = m.get("sr_only_ic")
                bl_ic = m.get("blended_ic")
            else:
                ic = m.rank_ic
                sr_ic = m.sr_only_ic
                bl_ic = m.blended_ic

            if ic is not None:
                valid_ics.append(sr_ic)
                valid_blended.append(bl_ic)

                # Check Carry status
                is_active = (
                    s.carry_data.is_active
                    if not isinstance(s.carry_data, dict)
                    else s.carry_data.get("is_active")
                )
                if is_active:
                    tight_ics.append(sr_ic)
                    tight_blended.append(bl_ic)

        if not valid_ics:
            print("No completed horizons to report.")
            return

        print(f"Completed Weeks:  {len(valid_ics)}")
        print(f"Avg SR-Only IC:   {np.mean(valid_ics):.4f}")
        print(f"Avg Blended IC:   {np.mean(valid_blended):.4f}")

        if tight_ics:
            print(f"\n--- TIGHTENING REGIME IMPACT ({len(tight_ics)} weeks) ---")
            print(f"SR Only IC:   {np.mean(tight_ics):.4f}")
            print(f"SR+Carry IC:  {np.mean(tight_blended):.4f}")
            print(f"Improvement:  {np.mean(tight_blended) - np.mean(tight_ics):.4f}")
        print(f"{'=' * 60}\n")

    def run_walk_forward(self, weeks=52) -> WalkForwardOutput:
        print(f"Running WalkForward for {weeks} weeks...")
        try:
            macro_df = self.features.compute_macro_regime_indices(
                self._df, self.currencies
            )
            available_dates = self._df.index
            step = 5
            start_idx = len(available_dates) - (weeks * step) - 1
            if start_idx < 63:
                start_idx = 63
            anchor_dates = available_dates[start_idx::step]

            history_snapshots = []
            all_preds, all_actuals = (
                {h: [] for h in self.config.HORIZONS},
                {h: [] for h in self.config.HORIZONS},
            )
            prev_label = ""
            persistence = 0

            for i, date in enumerate(anchor_dates):
                try:
                    if date not in self._feats["mom_21d"].index:
                        continue
                    mom, vol = (
                        self._feats["mom_21d"].loc[date],
                        self._feats["volatility"].loc[date],
                    )
                    y_diff, vix_z, indices_rec = (
                        self._yield_diffs.loc[date],
                        self._vix_z.loc[date],
                        macro_df.loc[date],
                    )

                    T_base = self.physics.construct_physics_matrix(
                        mom, vol, y_diff, BeliefParams(), vix_z
                    )
                    T_adj, leakage, net_flow, regime = (
                        self.physics.apply_adaptive_leakage(
                            T_base, self.currencies, vix_z, indices_rec
                        )
                    )

                    if regime.label == prev_label:
                        persistence += 1
                    else:
                        persistence = 1

                    p_dict = {
                        "usd_view": "Neutral",
                        "fx_risk": "Selective",
                        "carry_view": "Neutral",
                        "hedging": "Light",
                        "trust_ranking": True,
                    }
                    if regime.label == "USD Wrecking Ball":
                        p_dict.update(
                            {
                                "usd_view": "Overweight",
                                "fx_risk": "Defensive",
                                "carry_view": "Avoid",
                                "hedging": "Heavy",
                            }
                        )
                    elif regime.label == "US-Centric Stress":
                        p_dict.update(
                            {"usd_view": "Underweight", "fx_risk": "Defensive"}
                        )
                    elif regime.label == "Reflation / Risk-On":
                        p_dict.update(
                            {
                                "usd_view": "Underweight",
                                "fx_risk": "Aggressive",
                                "carry_view": "Favor",
                                "hedging": "None",
                            }
                        )
                    elif regime.label == "Tightening / Carry":
                        p_dict.update(
                            {
                                "usd_view": "Overweight",
                                "fx_risk": "Selective",
                                "carry_view": "Maximum Favor",
                                "trust_ranking": False,
                            }
                        )

                    prev_stress = (
                        float(macro_df.loc[anchor_dates[i - 1]]["stress_score"])
                        if i > 0
                        else 50.0
                    )
                    prev_dir = (
                        float(macro_df.loc[anchor_dates[i - 1]]["direction_score"])
                        if i > 0
                        else 0.0
                    )
                    delta_obj = WeeklyDelta(
                        stress_delta=float(indices_rec["stress_score"] - prev_stress),
                        direction_delta=float(
                            indices_rec["direction_score"] - prev_dir
                        ),
                        regime_shift=(regime.label != prev_label),
                        prev_label=prev_label,
                    )
                    transition_obj = self._compute_transition_risk(
                        indices_rec, delta_obj
                    )

                    is_carry_active = regime.label == "Tightening / Carry"
                    curr_carry = self._carry_z.loc[date].to_dict()
                    carry_obj = CarryData(
                        is_active=is_carry_active,
                        lambda_param=1.5 if is_carry_active else 0.0,
                        raw_yields=self._yields.loc[date].to_dict(),
                        yield_diffs={},
                        carry_scores=curr_carry,
                    )
                    prev_label = regime.label

                    # Build Snapshot using objects, let WalkForwardSnapshot handle serialization
                    snapshot_obj = {
                        "date": date.strftime("%Y-%m-%d"),
                        "horizon_results": {},
                        "edges": {},
                        "realized_returns": {},
                        "metrics": {},
                        "regime": regime,
                        "posture": p_dict,
                        "confidence": {
                            "score": float(max(0, 100 - abs(vix_z) * 20)),
                            "persistence": persistence,
                            "is_stable": True,
                        },
                        "delta": delta_obj,
                        "transition": transition_obj,
                        "usd_leakage": leakage,
                        "net_usd_flow": float(net_flow),
                        "carry_data": carry_obj,
                    }

                    for h_name, days in self.config.HORIZONS.items():
                        gamma = self.math.get_gamma(days)
                        M = self.math.compute_sr_matrix(T_adj, gamma)
                        sr_scores = self.math.compute_strength_scores(M)
                        sr_z = (sr_scores - np.mean(sr_scores)) / (
                            np.std(sr_scores) + 1e-6
                        )
                        final_scores = (
                            sr_z
                            + (1.5 * np.array([curr_carry[c] for c in self.currencies]))
                            if is_carry_active
                            else sr_scores
                        )
                        ranks = np.argsort(final_scores)[::-1]
                        snapshot_obj["horizon_results"][h_name] = [
                            {
                                "iso": self.currencies[i],
                                "score": float(final_scores[i]),
                                "rank": int(np.where(ranks == i)[0][0] + 1),
                                "delta": 0,
                                "trend": "Gaining",
                            }
                            for i in range(len(self.currencies))
                        ]
                        if h_name == "medium":
                            snapshot_obj["edges"][h_name] = self._get_top_edges(
                                T_adj, leakage
                            )

                        try:
                            fut_idx = self._df.index.get_loc(date) + days
                            if fut_idx < len(self._df):
                                rets = (
                                    self._df.iloc[fut_idx][self.currencies]
                                    - self._df.loc[date, self.currencies]
                                ) / self._df.loc[date, self.currencies]
                                snapshot_obj["realized_returns"][h_name] = (
                                    rets.to_dict()
                                )
                                corr_sr, _ = spearmanr(sr_scores, rets.values)
                                corr_blend, _ = spearmanr(final_scores, rets.values)
                                snapshot_obj["metrics"][h_name] = ValidationMetrics(
                                    rank_ic=float(corr_blend),
                                    top_quartile_ret=float(rets.iloc[ranks[:2]].mean()),
                                    btm_quartile_ret=float(
                                        rets.iloc[ranks[-2:]].mean()
                                    ),
                                    resilience_gap=float(
                                        rets.iloc[ranks[:2]].mean()
                                        - rets.iloc[ranks[-2:]].mean()
                                    ),
                                    sr_only_ic=float(corr_sr),
                                    blended_ic=float(corr_blend),
                                )
                                all_preds[h_name].extend(final_scores)
                                all_actuals[h_name].extend(rets.values)
                            else:
                                snapshot_obj["metrics"][h_name] = ValidationMetrics(
                                    rank_ic=None,
                                    top_quartile_ret=None,
                                    btm_quartile_ret=None,
                                    resilience_gap=None,
                                )
                                snapshot_obj["realized_returns"][h_name] = {}
                        except:
                            pass

                    # Instantiate snapshot properly
                    history_snapshots.append(WalkForwardSnapshot(**snapshot_obj))
                except Exception as loop_e:
                    print(f"Error in Backtest Loop for {date}: {loop_e}")
                    continue

            agg_corrs = {
                h: float(spearmanr(all_preds[h], all_actuals[h])[0])
                if len(all_actuals[h]) > 10
                else 0.0
                for h in self.config.HORIZONS
            }
            self.print_performance_report(history_snapshots)
            return WalkForwardOutput(history=history_snapshots, correlations=agg_corrs)
        except Exception as e:
            traceback.print_exc()
            raise e
