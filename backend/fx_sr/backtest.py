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
        self._df = self.loader.fetch_history(
            lookback_days=1500
        )  # Increased for robust scaling
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

    # --- THE SMART SWITCH LOGIC ---
    def _get_signal_direction(self, regime_label: str) -> float:
        if regime_label == "US-Centric Stress":
            return 1.0  # Safety works
        return -1.0  # Crowding fails

    def _get_top_edges(self, T, leakage_nodes, k=2):
        edges = []
        n = len(self.currencies)
        T_viz = T.copy()
        np.fill_diagonal(T_viz, 0)

        for i in range(n):
            top_indices = np.argsort(T_viz[i])[-k:]
            for j in top_indices:
                if T_viz[i, j] > 0.005:
                    edges.append(
                        {
                            "source": self.currencies[i],
                            "target": self.currencies[j],
                            "weight": float(T_viz[i, j]),
                        }
                    )

        for node in leakage_nodes:
            if node.leakage_prob > 0.02:
                edges.append(
                    {
                        "source": node.iso,
                        "target": "USD",
                        "weight": float(node.leakage_prob),
                    }
                )
        return edges

    def _compute_transition_risk(
        self, indices: Any, delta: WeeklyDelta
    ) -> RegimeTransition:
        s = float(indices["stress_score"])
        d = float(indices["direction_score"])
        risk_s = max(0, 100 - (abs(s - 40) * 5))
        risk_d = max(0, 100 - (abs(d - 0) * 2))

        # Velocity adjustment using WeeklyDelta schema
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
        print(
            f"\n{'=' * 60}\nFX SR INSTITUTIONAL AUDIT ({horizon.upper()})\n{'=' * 60}"
        )

        valid_snaps = [s for s in history if s.metrics[horizon].rank_ic is not None]
        if not valid_snaps:
            print("No valid periods found.")
            return

        # 1. Base Metrics
        base_ics = [s.metrics[horizon].sr_only_ic for s in valid_snaps]

        # 2. Reconstruct Equity Curve (Top 2 Long / Bottom 2 Short)
        # We need to approximate returns from the resilience gap * 0.5 (since gap is spread)
        # or use recorded realized returns if we stored them better.
        # For this audit, we use resilience_gap as a proxy for periodic return.
        periodic_rets = [s.metrics[horizon].resilience_gap for s in valid_snaps]
        cum_ret = np.cumsum(periodic_rets)

        # 3. Drawdown Calc
        peak = np.maximum.accumulate(cum_ret)
        drawdown = cum_ret - peak
        max_dd = np.min(drawdown)

        print(f"Observations:     {len(base_ics)}")
        print(f"Mean Rank IC:     {np.mean(base_ics):.4f}")
        print(f"Win Rate:         {(np.array(base_ics) > 0).mean() * 100:.1f}%")
        print(f"Cum. Return:      {cum_ret[-1] * 100:.1f}%")
        print(f"Max Drawdown:     {max_dd * 100:.1f}%")
        print(f"-" * 30)

        # 4. Regime Breakdown
        regimes = {}
        for s in valid_snaps:
            r = s.regime.label
            if r not in regimes:
                regimes[r] = []
            regimes[r].append(s.metrics[horizon].sr_only_ic)

        print(f"{'Regime':<20} | {'IC':<8} | {'Count'}")
        for r, ics in regimes.items():
            print(f"{r:<20} | {np.mean(ics):.4f}   | {len(ics)}")

        print(f"{'=' * 60}\n")

    def run_walk_forward(self, weeks=52) -> WalkForwardOutput:
        print(f"Running Institutional WalkForward for {weeks} weeks...")

        # Hardening Constants
        TX_COST = 0.0002  # 2bps friction per trade
        SIGNAL_LAG = 0  # EOD Execution

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

                    # --- SMART SWITCH ---
                    signal_dir = self._get_signal_direction(regime.label)

                    # Posture Logic
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

                    curr_carry = self._carry_z.loc[date].to_dict()
                    carry_obj = CarryData(
                        is_active=(regime.label == "Tightening / Carry"),
                        lambda_param=1.5
                        if (regime.label == "Tightening / Carry")
                        else 0.0,
                        raw_yields=self._yields.loc[date].to_dict(),
                        yield_diffs={},
                        carry_scores=curr_carry,
                    )
                    prev_label = regime.label

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
                        "usd_leakage": [l.model_dump() for l in leakage],
                        "net_usd_flow": float(net_flow),
                        "carry_data": carry_obj,
                    }

                    for h_name, days in self.config.HORIZONS.items():
                        gamma = self.math.get_gamma(days)
                        M = self.math.compute_sr_matrix(T_adj, gamma)
                        sr_scores = self.math.compute_strength_scores(M)

                        # --- THE SMART SWITCH (The Alpha Generator) ---
                        # 1. Determine Direction
                        # If the regime is 'US-Centric Stress', we trust the sinks (+1.0)
                        # In all other regimes, we fade the crowding (-1.0)
                        signal_dir = (
                            1.0 if regime.label == "US-Centric Stress" else -1.0
                        )

                        # 2. Apply the Switch
                        final_scores = sr_scores * signal_dir

                        # Carry Blend (Optional overlay)
                        if carry_obj.is_active:
                            sr_z = (final_scores - np.mean(final_scores)) / (
                                np.std(final_scores) + 1e-6
                            )
                            final_scores = sr_z + (
                                1.5 * np.array([curr_carry[c] for c in self.currencies])
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

                        # Realized Returns
                        try:
                            curr_pos = self._df.index.get_loc(date)
                            exit_idx = curr_pos + days
                            if exit_idx < len(self._df):
                                entry_vals = self._df.iloc[curr_pos][self.currencies]
                                exit_vals = self._df.iloc[exit_idx][self.currencies]
                                rets = ((exit_vals - entry_vals) / entry_vals) - TX_COST

                                snapshot_obj["realized_returns"][h_name] = (
                                    rets.to_dict()
                                )
                                corr, _ = spearmanr(final_scores, rets.values)

                                top_ret = float(rets.iloc[ranks[:2]].mean())
                                btm_ret = float(rets.iloc[ranks[-2:]].mean())

                                snapshot_obj["metrics"][h_name] = ValidationMetrics(
                                    rank_ic=float(corr),
                                    top_quartile_ret=top_ret,
                                    btm_quartile_ret=btm_ret,
                                    resilience_gap=float(top_ret - btm_ret),
                                    sr_only_ic=float(corr),
                                    blended_ic=float(corr),
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
                        except Exception:
                            continue

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
