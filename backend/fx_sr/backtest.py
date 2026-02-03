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
    RegimeData,
    ValidationMetrics,
    WalkForwardOutput,
    WalkForwardSnapshot,
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
        self._df = self.loader.fetch_history(lookback_days=750)
        self._feats, _ = self.features.compute_features(self._df, self.currencies)
        self._yields = self.loader.fetch_yields(self._df.index)
        self._carry_z = self.features.compute_carry_scores(
            self._yields, self.currencies
        )

    def _get_top_edges(self, T, k=3):
        edges = []
        T_viz = T.copy()
        np.fill_diagonal(T_viz, 0)
        for i in range(len(self.currencies)):
            top_indices = np.argsort(T_viz[i])[-k:]
            for j in top_indices:
                if T_viz[i, j] > 0.05:
                    edges.append(
                        {
                            "source": self.currencies[i],
                            "target": self.currencies[j],
                            "weight": float(T_viz[i, j]),
                        }
                    )
        return edges

    def print_performance_report(
        self, history: List[WalkForwardSnapshot], horizon: str = "medium"
    ):
        """Hardened report that filters out None values from pending future weeks."""
        print(
            f"\n{'=' * 60}\nFX SR MODEL PERFORMANCE REPORT ({horizon.upper()})\n{'=' * 60}"
        )

        # Filter snapshots that have realized metrics
        valid_snaps = [s for s in history if s.metrics[horizon].rank_ic is not None]

        if not valid_snaps:
            print("No completed horizons found to report on.")
            return

        base_ics = [s.metrics[horizon].sr_only_ic for s in valid_snaps]
        blend_ics = [s.metrics[horizon].blended_ic for s in valid_snaps]

        print(f"Completed Weeks:  {len(valid_snaps)}")
        print(f"Avg SR-Only IC:   {np.mean(base_ics):.4f}")
        print(f"Avg Blended IC:   {np.mean(blend_ics):.4f}")

        # Tightening Regime Filtering
        tight_snaps = [s for s in valid_snaps if s.carry_data.is_active]
        if tight_snaps:
            t_sr = [s.metrics[horizon].sr_only_ic for s in tight_snaps]
            t_bl = [s.metrics[horizon].blended_ic for s in tight_snaps]
            print(f"\n--- TIGHTENING REGIME IMPACT ({len(tight_snaps)} weeks) ---")
            print(f"SR Only IC:   {np.mean(t_sr):.4f}")
            print(f"SR+Carry IC:  {np.mean(t_bl):.4f}")
            print(f"Improvement:  {np.mean(t_bl) - np.mean(t_sr):.4f}")

        print(f"{'=' * 60}\n")

    def run_walk_forward(self, weeks=52):
        macro_df = self.features.compute_macro_regime_indices(self._df, self.currencies)

        # Calculate sample step
        available_dates = self._df.index
        step = 5
        start_idx = len(available_dates) - (weeks * step) - 1
        if start_idx < 63:
            start_idx = 63
        anchor_dates = available_dates[start_idx::step]

        history_snapshots = []
        prev_label = ""
        persistence = 0

        all_preds, all_actuals = (
            {h: [] for h in self.config.HORIZONS},
            {h: [] for h in self.config.HORIZONS},
        )

        for date in anchor_dates:
            if date not in self._feats["mom_21d"].index:
                continue

            # Data Fetch
            mom, vol, vix = (
                self._feats["mom_21d"].loc[date],
                self._feats["volatility"].loc[date],
                float(self._df.loc[date, "VIX"]),
            )
            indices_rec = macro_df.loc[date]

            # Physics
            T_base = self.physics.construct_scenario_matrix(
                mom, vol, vix, BeliefParams()
            )
            T_adj, leakage, net_flow, regime = self.physics.apply_regime_physics(
                T_base, self.currencies, indices_rec
            )

            # --- Persistence Logic ---
            if regime.label == prev_label:
                persistence += 1
            else:
                persistence = 1

            # --- New Interpretation ---
            # Using the previous loop's data for delta
            prev_stress = 50.0  # Default for first week
            prev_dir = 0.0
            if len(history_snapshots) > 0:
                last_regime = history_snapshots[-1].regime
                prev_stress = last_regime.indices.stress_score
                prev_dir = last_regime.indices.direction_score

            posture = {
                # Logic copied from engine._compute_posture for consistency
                "usd_view": "Neutral",
                "fx_risk": "Selective",
                "carry_view": "Neutral",
                "hedging": "Light",
                "trust_ranking": True,
            }
            if regime.label == "USD Wrecking Ball":
                posture = {
                    "usd_view": "Overweight",
                    "fx_risk": "Defensive",
                    "carry_view": "Avoid",
                    "hedging": "Heavy",
                    "trust_ranking": True,
                }
            elif regime.label == "Reflation / Risk-On":
                posture = {
                    "usd_view": "Underweight",
                    "fx_risk": "Aggressive",
                    "carry_view": "Favor",
                    "hedging": "None",
                    "trust_ranking": True,
                }
            elif regime.label == "Tightening / Carry":
                posture["trust_ranking"] = False
                posture["carry_view"] = "Favor"

            # Confidence
            base_conf = max(0, 100 - (regime.indices.stress_score * 0.4)) + (
                persistence * 2
            )

            # Delta
            delta_obj = {
                "stress_chg": float(regime.indices.stress_score - prev_stress),
                "direction_chg": float(regime.indices.direction_score - prev_dir),
                "regime_shift": (regime.label != prev_label),
                "prev_label": prev_label,
            }

            prev_label = regime.label
            # Carry
            is_carry_active = regime.label == "Tightening / Carry"
            carry_z = self._carry_z.loc[date].to_dict()
            carry_obj = CarryData(
                is_active=is_carry_active,
                lambda_param=1.5 if is_carry_active else 0,
                raw_yields=self._yields.loc[date].to_dict(),
                yield_diffs={},
                carry_scores=carry_z,
            )

            snapshot = {
                "date": date.strftime("%Y-%m-%d"),
                "horizon_results": {},
                "edges": {},
                "realized_returns": {},
                "metrics": {},
                "regime": regime,
                "usd_leakage": leakage,
                "net_usd_flow": net_flow,
                "carry_data": carry_obj,
                "posture": posture,
                "confidence": {
                    "score": base_conf,
                    "persistence": persistence,
                    "is_stable": base_conf > 60,
                },
                "delta": delta_obj,
            }

            for h_name, days in self.config.HORIZONS.items():
                gamma = self.math.get_gamma(days)
                M = self.math.compute_sr_matrix(T_adj, gamma)
                sr_scores = self.math.compute_strength_scores(M)

                # V1 Blending
                sr_mean, sr_std = np.mean(sr_scores), np.std(sr_scores) + 1e-6
                sr_z = (sr_scores - sr_mean) / sr_std

                if is_carry_active:
                    carry_vec = np.array([carry_z[c] for c in self.currencies])
                    final_scores = sr_z + (1.5 * carry_vec)
                else:
                    final_scores = sr_scores

                ranks = np.argsort(final_scores)[::-1]
                snapshot["horizon_results"][h_name] = [
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
                    snapshot["edges"][h_name] = self._get_top_edges(T_adj)

                # Validation (Causal check)
                try:
                    fut_idx = self._df.index.get_loc(date) + days
                    if fut_idx < len(self._df):
                        rets = (
                            self._df.iloc[fut_idx][self.currencies]
                            - self._df.loc[date, self.currencies]
                        ) / self._df.loc[date, self.currencies]
                        snapshot["realized_returns"][h_name] = rets.to_dict()

                        corr_sr, _ = spearmanr(sr_scores, rets.values)
                        corr_blend, _ = spearmanr(final_scores, rets.values)

                        snapshot["metrics"][h_name] = ValidationMetrics(
                            rank_ic=float(corr_blend),
                            top_quartile_ret=float(rets.iloc[ranks[:2]].mean()),
                            btm_quartile_ret=float(rets.iloc[ranks[-2:]].mean()),
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
                        # Future has not happened yet
                        snapshot["metrics"][h_name] = ValidationMetrics(
                            rank_ic=None,
                            top_quartile_ret=None,
                            btm_quartile_ret=None,
                            resilience_gap=None,
                            sr_only_ic=None,
                            blended_ic=None,
                        )
                        snapshot["realized_returns"][h_name] = {}
                except:
                    pass

            history_snapshots.append(WalkForwardSnapshot(**snapshot))

        # Final Aggregation
        agg_corrs = {}
        for h in self.config.HORIZONS:
            if len(all_actuals[h]) > 10:
                c, _ = spearmanr(all_preds[h], all_actuals[h])
                agg_corrs[h] = float(c)
            else:
                agg_corrs[h] = 0.0

        self.print_performance_report(history_snapshots)
        return WalkForwardOutput(history=history_snapshots, correlations=agg_corrs)
