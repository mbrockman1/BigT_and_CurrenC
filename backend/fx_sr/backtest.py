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
    NetworkEdge,
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
        """
        Extracts the most significant capital flows.
        Excludes self-loops to force 'rewiring' visibility.
        Includes USD leakage as explicit edges.
        """
        edges = []
        n = len(self.currencies)

        # 1. PEER-TO-PEER FLOWS (Blue Lines)
        for i in range(n):
            row = T[i].copy()
            # FORCE REWIRING: Set self-probability to zero so we see where capital EXITS to
            row[i] = 0

            # Get Top K destinations for this currency
            top_indices = np.argsort(row)[-k:]
            for j in top_indices:
                weight = float(row[j])
                if weight > 0.001:  # High sensitivity threshold
                    edges.append(
                        {
                            "source": self.currencies[i],
                            "target": self.currencies[j],
                            "weight": weight,
                        }
                    )

        # 2. FLIGHT TO USD (Red Lines)
        # We turn the leakage stats into explicit graph edges pointing to 'USD'
        for node in leakage_nodes:
            if node.leakage_prob > 0.01:
                edges.append(
                    {
                        "source": node.iso,
                        "target": "USD",  # The central reservoir
                        "weight": float(node.leakage_prob),
                    }
                )

        return edges

    def run_walk_forward(self, weeks=52):
        macro_df = self.features.compute_macro_regime_indices(self._df, self.currencies)
        available_dates = self._df.index
        step = 5
        start_idx = len(available_dates) - (weeks * step) - 1
        if start_idx < 63:
            start_idx = 63
        anchor_dates = available_dates[start_idx::step]

        history_snapshots = []
        # Correlation accumulators
        all_preds = {h: [] for h in self.config.HORIZONS}
        all_actuals = {h: [] for h in self.config.HORIZONS}

        prev_label = ""
        persistence = 0

        for i, date in enumerate(anchor_dates):
            if date not in self._feats["mom_21d"].index:
                continue

            # --- FIX: Fetch Historical Physics Inputs ---
            mom = self._feats["mom_21d"].loc[date]
            vol = self._feats["volatility"].loc[date]
            y_diff = self._yield_diffs.loc[date]
            vix_z = self._vix_z.loc[date]
            indices_rec = macro_df.loc[date]

            # --- Physics ---
            T_base = self.physics.construct_physics_matrix(
                mom, vol, y_diff, BeliefParams()
            )
            T_adj, leakage, net_flow, regime = self.physics.apply_adaptive_leakage(
                T_base, self.currencies, vix_z, indices_rec
            )

            # Persistence
            if regime.label == prev_label:
                persistence += 1
            else:
                persistence = 1

            # Posture (Replicated logic)
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
                p_dict.update({"usd_view": "Underweight", "fx_risk": "Defensive"})
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
            if indices_rec.stress_score > 75:
                p_dict.update({"fx_risk": "Prohibited", "hedging": "Maximum"})

            # Carry Logic
            is_carry_active = regime.label == "Tightening / Carry"
            curr_carry = (
                self._carry_z.loc[date].to_dict()
                if date in self._carry_z.index
                else {c: 0.0 for c in self.currencies}
            )
            carry_obj = CarryData(
                is_active=is_carry_active,
                lambda_param=1.5 if is_carry_active else 0,
                raw_yields=self._yields.loc[date].to_dict(),
                yield_diffs={},
                carry_scores=curr_carry,
            )

            # Delta Logic
            prev_stress = (
                macro_df.loc[anchor_dates[i - 1]].stress_score if i > 0 else 50
            )
            prev_dir = macro_df.loc[anchor_dates[i - 1]].direction_score if i > 0 else 0

            snapshot = {
                "date": date.strftime("%Y-%m-%d"),
                "horizon_results": {},
                "edges": {},
                "realized_returns": {},
                "metrics": {},
                "regime": regime.model_dump(),
                "posture": p_dict,
                "confidence": {
                    "score": float(max(0, 100 - abs(vix_z) * 20)),
                    "persistence": persistence,
                    "is_stable": True,
                },
                "delta": {
                    "stress_chg": float(indices_rec.stress_score - prev_stress),
                    "direction_chg": float(indices_rec.direction_score - prev_dir),
                    "regime_shift": (regime.label != prev_label),
                    "prev_label": prev_label,
                },
                "usd_leakage": [l.model_dump() for l in leakage],
                "net_usd_flow": float(net_flow),
                "carry_data": carry_obj.model_dump(),
            }

            prev_label = regime.label

            # SR & Validation
            for h_name, days in self.config.HORIZONS.items():
                gamma = self.math.get_gamma(days)
                M = self.math.compute_sr_matrix(T_adj, gamma)
                sr_scores = self.math.compute_strength_scores(M)

                # Blend V1
                sr_z = (sr_scores - np.mean(sr_scores)) / (np.std(sr_scores) + 1e-6)
                final_scores = sr_z
                if is_carry_active:
                    c_vec = np.array([curr_carry.get(c, 0) for c in self.currencies])
                    final_scores = sr_z + (1.5 * c_vec)

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
                    snapshot["edges"][h_name] = self._get_top_edges(T_adj, leakage)

                # Validation
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

                        all_preds[h_name].extend(final_scores)
                        all_actuals[h_name].extend(rets.values)

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
                        ).model_dump()
                    else:
                        snapshot["metrics"][h_name] = ValidationMetrics(
                            rank_ic=None,
                            top_quartile_ret=None,
                            btm_quartile_ret=None,
                            resilience_gap=None,
                        ).model_dump()
                        snapshot["realized_returns"][h_name] = {}
                except:
                    pass

            history_snapshots.append(WalkForwardSnapshot(**snapshot))

        # Aggregate Stats
        agg_corrs = {}
        for h in self.config.HORIZONS:
            if len(all_actuals[h]) > 10:
                c, _ = spearmanr(all_preds[h], all_actuals[h])
                agg_corrs[h] = float(c)
            else:
                agg_corrs[h] = 0.0

        return WalkForwardOutput(history=history_snapshots, correlations=agg_corrs)
