from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import FXConfig
from .schemas import BeliefParams, MacroIndices, RegimeData, USDLeakageNode


class TransitionModel:
    """
    Constructs the Transition Matrix (T).
    """

    def __init__(self, config: FXConfig):
        self.config = config

    def construct_scenario_matrix(
        self,
        snapshot_mom: pd.Series,
        snapshot_vol: pd.Series,
        base_vix: float,
        beliefs: BeliefParams,
    ) -> np.ndarray:
        """
        Builds T based on user beliefs blended with market data.
        """
        currencies = list(self.config.UNIVERSE.keys())
        n = len(currencies)
        weights = np.zeros((n, n), dtype=float)

        effective_vix = (
            beliefs.vix_override if beliefs.vix_override is not None else base_vix
        )

        mix = beliefs.risk_mix
        base_trend_weight = (1 - mix) * 5.0 + (mix) * 1.0
        base_vol_weight = (1 - mix) * 1.0 + (mix) * 4.0
        base_haven_bonus = (mix) * 1.5

        w_mom = base_trend_weight * beliefs.trend_sensitivity
        w_vol = base_vol_weight * beliefs.vol_penalty

        mom_vec = snapshot_mom.reindex(currencies).to_numpy(dtype=float)
        vol_vec = snapshot_vol.reindex(currencies).to_numpy(dtype=float)

        if beliefs.shocks:
            for shock in beliefs.shocks:
                if shock.iso in currencies:
                    idx = currencies.index(shock.iso)
                    vol_vec[idx] *= shock.vol_shock
                    mom_vec[idx] *= shock.mom_shock

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                curr_target = currencies[j]

                mom_diff = mom_vec[j] - mom_vec[i]
                vol_diff = vol_vec[i] - vol_vec[j]

                haven_score = 0.0
                if curr_target in self.config.SAFE_HAVENS:
                    haven_score = base_haven_bonus

                score = (mom_diff * w_mom) + (vol_diff * w_vol) + haven_score
                score = float(np.clip(score, -15.0, 15.0))
                weights[i, j] = np.exp(score)

        np.fill_diagonal(weights, 0.1)
        row_sums = weights.sum(axis=1, keepdims=True)
        T = weights / (row_sums + 1e-12)

        return T

    def apply_regime_physics(
        self,
        T: np.ndarray,
        currencies: List[str],
        indices: pd.Series,  # The row from compute_macro_regime_indices
    ) -> Tuple[np.ndarray, List[USDLeakageNode], float, RegimeData]:
        """
        Applies USD Leakage conditionally based on Stress vs Direction.
        Returns modified T, leakage nodes, net flow, and the classification.
        """
        stress = float(indices["stress_score"])  # 0 to 100
        direction = float(indices["direction_score"])  # -100 to 100

        # 1. Classify Regime
        # Thresholds: Stress > 50 is High. Direction > 0 is Up.
        is_high_stress = stress > 40.0
        is_usd_up = direction > 0.0

        label = ""
        desc = ""
        leakage_intensity = 0.0
        is_source_mode = False

        if is_high_stress and is_usd_up:
            label = "USD Wrecking Ball"
            desc = "High Stress + USD Strength. Capital flees peers to USD safety."
            leakage_intensity = (stress / 100.0) * 0.30  # Up to 30% leakage

        elif is_high_stress and not is_usd_up:
            label = "US-Centric Stress"
            desc = "High Stress + USD Weakness. Disorder originates in US (e.g. Banking/Debt). No USD Sink."
            leakage_intensity = 0.0  # Do NOT sink to USD
            # Optional: Could sink to Gold/CHF here if nodes existed

        elif not is_high_stress and not is_usd_up:
            label = "Reflation / Risk-On"
            desc = "Low Stress + USD Weakness. Capital flows from USD to peers."
            leakage_intensity = 0.0
            is_source_mode = True  # Inject flow

        else:  # Low Stress + USD Up
            label = "Tightening / Carry"
            desc = "Orderly USD Strength. Moderate flow to USD driven by yields."
            leakage_intensity = (direction / 100.0) * 0.10  # Gentle leakage

        # 2. Apply Physics
        n = len(currencies)
        leakage_nodes = []
        net_flow = 0.0
        T_adj = T.copy()

        # Leakage Logic (Sink)
        if leakage_intensity > 0:
            for i in range(n):
                # Idiosyncratic nuance: weaker momentum currencies leak faster?
                # For MVP, uniform leakage based on intensity
                p_leak = leakage_intensity

                # Update Matrix (Row sum reduction)
                T_adj[i, :] = T_adj[i, :] * (1.0 - p_leak)

                leakage_nodes.append(
                    USDLeakageNode(
                        iso=currencies[i], leakage_prob=p_leak, is_source=False
                    )
                )
                net_flow += p_leak

        # Injection Logic (Source)
        if is_source_mode:
            # We simulate source by 'negative net flow' reporting
            # T matrix assumes peer circulation, so we can't easily add external probability
            # without expanding N. We represent this as Net Flow < 0.
            magnitude = abs(direction / 100.0) * 0.2
            net_flow -= magnitude * n

        # 3. Package Result
        regime_data = RegimeData(
            label=label,
            desc=desc,
            indices=MacroIndices(
                stress_score=stress,
                direction_score=direction,
                stress_breadth=float(indices["stress_breadth"]),
                stress_vol=float(indices["stress_vol"]),
                usd_momentum=float(indices["usd_mom"]),
                yield_delta=float(indices["yield_delta"]),
            ),
        )

        return T_adj, leakage_nodes, net_flow, regime_data

    def apply_usd_leakage(
        self,
        T: np.ndarray,
        currencies: List[str],
        pressure: float,
        vix: float,
        yield_chg: float,
    ) -> Tuple[np.ndarray, List[USDLeakageNode], float]:
        """
        Modifies T so it is no longer strictly row-stochastic within the peer group.
        Mass 'leaks' to the USD node based on pressure.
        """
        n = len(currencies)
        leakage_nodes = []
        net_flow = 0.0

        # 1. Base Leakage Probability
        # If Pressure is 80/100, high leakage
        base_prob = (pressure / 100.0) * 0.25

        if vix > 25:
            base_prob *= 1.5
        if yield_chg > 0.05:
            base_prob += 0.05

        base_prob = min(base_prob, 0.8)

        # 2. Apply to Matrix
        T_adj = T.copy()

        for i in range(n):
            p_leak = float(base_prob)

            leakage_nodes.append(
                USDLeakageNode(iso=currencies[i], leakage_prob=p_leak, is_source=False)
            )

            # Reduce row weights
            # New Sum = 1.0 - p_leak
            T_adj[i, :] = T_adj[i, :] * (1.0 - p_leak)

            net_flow += p_leak

        # 3. USD Release (Source Mode)
        # If pressure is low (<20), USD acts as a source
        if pressure < 20:
            release = (20 - pressure) / 100.0 * 0.5
            net_flow -= release * n

        return T_adj, leakage_nodes, net_flow
