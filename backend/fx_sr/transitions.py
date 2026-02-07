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
                vix=float(indices["vix"]),
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

    def construct_physics_matrix(
        self,
        mom: pd.Series,
        vol: pd.Series,
        yield_diffs: pd.Series,
        beliefs: BeliefParams,
    ) -> np.ndarray:
        currencies = list(self.config.UNIVERSE.keys())
        n = len(currencies)
        weights = np.zeros((n, n), dtype=float)

        # --- SENSITIVITY TUNING ---
        # Temperature controls how "concentrated" the flows are.
        # Higher (1.0+) = Spiderweb (lots of rewiring). Lower (0.2) = Winner-Take-All.
        temperature = 0.8

        # We Z-Score the inputs so a 1% yield move has the same "Gravity" as a 1% price move.
        def z_score(s):
            return (s - s.mean()) / (s.std() + 1e-6)

        y_z = z_score(yield_diffs.reindex(currencies).fillna(0)).to_numpy()
        m_z = (
            z_score(mom.reindex(currencies).fillna(0)).to_numpy()
            * beliefs.trend_sensitivity
        )
        v_z = (
            z_score(vol.reindex(currencies).fillna(0)).to_numpy() * beliefs.vol_penalty
        )

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # DIFFERENTIAL PHYSICS: Attraction is based on the RELATIVE advantage of j over i
                # Capital flows from Low Yield -> High Yield
                # Capital flows from Low Momentum -> High Momentum
                # Capital flows from High Vol -> Low Vol

                score = (y_z[j] - y_z[i]) + (m_z[j] - m_z[i]) - (v_z[j] - v_z[i])

                weights[i, j] = np.exp(score / temperature)

        # Ensure small baseline connectivity
        weights = np.clip(weights, 1e-6, None)

        # Self-retention (diagonal)
        np.fill_diagonal(weights, 1.0)

        # Normalize rows to 1.0
        row_sums = weights.sum(axis=1, keepdims=True)
        T = weights / (row_sums + 1e-12)

        return T

    def apply_adaptive_leakage(
        self,
        T: np.ndarray,
        currencies: List[str],
        vix_z_score: float,  # <--- NEW: Dynamic Input
        indices: pd.Series,
    ) -> Tuple[np.ndarray, List[USDLeakageNode], float, RegimeData]:
        """
        Applies USD Leakage based on Adaptive Z-Scores, not magic numbers.
        """
        # 1. Calculate Dynamic Leakage Intensity using Sigmoid
        # Z-Score of 0 (Mean VIX) -> 10% Leakage
        # Z-Score of +2 (Crisis) -> 40% Leakage
        # Z-Score of -1 (Calm) -> 0% Leakage
        sigmoid = 1 / (1 + np.exp(-(vix_z_score - 0.5)))
        leakage_intensity = float(sigmoid * 0.40)  # Max 40%

        # Direction filter: Only leak if USD is structurally strong
        if indices["direction_score"] < 0:
            leakage_intensity = 0.0  # No leakage if USD is crashing

        # 2. Apply Physics
        n = len(currencies)
        leakage_nodes = []
        net_flow = 0.0
        T_adj = T.copy()

        if leakage_intensity > 0.01:
            for i in range(n):
                p_leak = leakage_intensity
                T_adj[i, :] = T_adj[i, :] * (1.0 - p_leak)
                leakage_nodes.append(
                    USDLeakageNode(
                        iso=currencies[i], leakage_prob=p_leak, is_source=False
                    )
                )
                net_flow += p_leak

        # 3. Construct Regime Label (Data-Driven)
        label = "Neutral"
        desc = "Markets are near historical averages."

        if vix_z_score > 1.0:
            label = "High Stress"
            desc = "VIX is > 1 std dev above mean. Capital constraints active."
            if indices["direction_score"] > 0:
                label = "USD Wrecking Ball"
                desc = (
                    f"Crisis Logic: High Stress (Z={vix_z_score:.1f}) + USD Strength."
                )
        elif vix_z_score < -0.5:
            label = "Reflation / Carry"
            desc = "Low Volatility Regime. Yield seeking behavior dominant."

        regime_data = RegimeData(
            label=label,
            desc=desc,
            indices=MacroIndices(
                stress_score=float((vix_z_score + 2) * 20),  # Proxy for UI 0-100
                direction_score=float(indices["direction_score"]),
                stress_breadth=0.5,
                stress_vol=0.5,
                usd_momentum=0.0,
                yield_delta=0.0,
                vix=float(indices["vix"]),
            ),
        )

        return T_adj, leakage_nodes, net_flow, regime_data
