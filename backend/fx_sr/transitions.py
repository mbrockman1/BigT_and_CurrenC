from typing import Any, List, Tuple

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
        vix_z: float = 0.0,
    ) -> np.ndarray:
        currencies = list(self.config.UNIVERSE.keys())
        n = len(currencies)
        weights = np.zeros((n, n), dtype=float)

        # --- THE 'CENTER OF GRAVITY' FIX ---
        # We must acknowledge that in the current era, High Yield = High Risk.

        # 1. Define Regime Intensity
        # If VIX Z is high, we enter 'Preservation Mode'
        is_stressed = vix_z > 0.5

        if is_stressed:
            # CRISIS PHYSICS: Capital seeks the 'Deepest Sinks'
            w_yield = -1.5  # FLIP YIELD: High yield is now a 'Leaky' signal (Risk)
            w_mom = -1.0  # ANTI-MOMENTUM: Chasing trends in a crisis is a trap
            w_vol = 5.0  # TOTAL SAFETY: Gravity pulls only to low-volatility
            temperature = 0.3
        else:
            # CALM PHYSICS: Relative Value
            w_yield = 0.5  # DAMPEN YIELD: Don't let carry bully the model
            w_mom = 0.5  # MILD TREND: Respect the move, but don't marry it
            w_vol = 2.0  # RISK AVERSION: Even in calm, favor quality
            temperature = 0.9

        # Normalize via Z-Score
        def z_score(s):
            return (s - s.mean()) / (s.std() + 1e-6)

        y_z = z_score(yield_diffs.reindex(currencies).fillna(0)).to_numpy()
        m_z = z_score(mom.reindex(currencies).fillna(0)).to_numpy()
        v_z = z_score(vol.reindex(currencies).fillna(0)).to_numpy()

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # The Equation: High Score = Lower Volatility & Lower (Riskier) Yield
                # This forces capital to flow toward JPY, CHF, and USD during stress.
                score = (
                    (y_z[j] - y_z[i]) * w_yield
                    + (m_z[j] - m_z[i]) * w_mom
                    - (v_z[j] - v_z[i]) * w_vol
                )

                weights[i, j] = np.exp(np.clip(score / temperature, -12, 12))

        np.fill_diagonal(weights, 1.0)
        T = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
        return T

    def apply_adaptive_leakage(
        self, T: np.ndarray, currencies: List[str], vix_z: float, indices: Any
    ) -> Tuple[np.ndarray, List[USDLeakageNode], float, RegimeData]:
        # 1. Leakage Intensity (Floating)
        # Higher VIX = Higher Leakage to USD
        # sigmoid = 1 / (1 + np.exp(-(vix_z - 0.5)))
        leakage_intensity = float(np.clip(0.10 + (vix_z * 0.10), 0.0, 0.50))
        # Direction Check
        d_score = (
            float(indices.direction_score)
            if hasattr(indices, "direction_score")
            else float(indices["direction_score"])
        )
        if d_score < 0:
            leakage_intensity = 0.0

        # 2. Apply
        n = len(currencies)
        leakage_nodes = []
        net_flow = 0.0
        T_adj = T.copy()

        for i in range(len(currencies)):
            T_adj[i, :] = T_adj[i, :] * (1.0 - leakage_intensity)
            leakage_nodes.append(
                USDLeakageNode(
                    iso=currencies[i], leakage_prob=leakage_intensity, is_source=False
                )
            )
            net_flow += leakage_intensity

        # 3. Labeling
        s_score = (
            float(indices.stress_score)
            if hasattr(indices, "stress_score")
            else float(indices["stress_score"])
        )

        label = "Neutral"
        desc = "Markets are near historical averages."

        if s_score > 60:
            label = "USD Wrecking Ball" if d_score > 0 else "US-Centric Stress"
            desc = "High Stress Regime. Yields deprioritized in favor of Safety."
        elif s_score < 40:
            label = "Tightening / Carry" if d_score > 0 else "Reflation / Risk-On"
            desc = "Low Stress Regime. Yields and Growth driving capital flows."

        # Package Result (Safe access)
        # ... (Same safe packaging logic as before) ...
        idx_data = MacroIndices(
            stress_score=s_score,
            direction_score=d_score,
            stress_breadth=float(indices.stress_breadth)
            if hasattr(indices, "stress_breadth")
            else float(indices.get("stress_breadth", 0)),
            stress_vol=float(indices.stress_vol)
            if hasattr(indices, "stress_vol")
            else float(indices.get("stress_vol", 0)),
            usd_momentum=float(indices.usd_momentum)
            if hasattr(indices, "usd_momentum")
            else float(indices.get("usd_momentum", 0)),
            yield_delta=float(indices.yield_delta)
            if hasattr(indices, "yield_delta")
            else float(indices.get("yield_delta", 0)),
            vix=float(indices.vix)
            if hasattr(indices, "vix")
            else float(indices.get("vix", 0)),
        )

        return (
            T_adj,
            leakage_nodes,
            net_flow,
            RegimeData(label=label, desc=desc, indices=idx_data),
        )
