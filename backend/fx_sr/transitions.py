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

    def _robust_scale(self, series: pd.Series) -> np.ndarray:
        """
        Robust Normalization using Median and MAD.
        Prevents outliers (e.g., flash crashes) from distorting the Physics.
        """
        # arr = series.fillna(0).to_numpy(dtype=float)
        # median = np.median(arr)
        # # MAD = Median(|x - median|)
        # mad = np.median(np.abs(arr - median))

        # # Avoid division by zero. If MAD is 0, fallback to std.
        # scale = mad * 1.4826 if mad > 1e-6 else (np.std(arr) + 1e-6)

        # # Calculate Robust Z
        # z = (arr - median) / scale

        # # Clip extreme outliers (-3 to +3) to stabilize Softmax
        # return np.clip(z, -3.0, 3.0)
        arr = series.fillna(0).to_numpy(dtype=float)
        return (arr - arr.mean()) / (arr.std() + 1e-6)

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

        # --- DATA-DERIVED PHYSICS (OPTIMIZER OUTPUT) ---
        # The optimizer found that the regime break is at Mean VIX (0.0).
        is_stress = vix_z > 0.00

        if is_stress:
            # STRESS PHYSICS (IC: 0.1176)
            # The market trends hard in stress. Yield provides a floor.
            w_yield = 1.0
            w_mom = 2.0  # Aggressive Trend Following
            w_vol = 1.0  # Moderate Risk Aversion
            temperature = 0.4  # High Conviction
        else:
            # CALM PHYSICS (IC: 0.2071)
            # The market chases Beta (Volatility) in calm.
            w_yield = 0.0  # Yield doesn't drive alpha here
            w_mom = 0.0  # Trend doesn't drive alpha here
            w_vol = -0.5  # REWARD Volatility (High Beta)
            temperature = 0.4

        # Standard Z-Score Normalization (Optimizer used this)
        def z_score(s):
            return (s - s.mean()) / (s.std() + 1e-6)

        y_z = z_score(yield_diffs.reindex(currencies).fillna(0)).to_numpy()
        m_z = z_score(mom.reindex(currencies).fillna(0)).to_numpy()
        v_z = z_score(vol.reindex(currencies).fillna(0)).to_numpy()

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                score = (
                    (y_z[j] - y_z[i]) * w_yield
                    + (m_z[j] - m_z[i]) * w_mom
                    - (v_z[j] - v_z[i]) * w_vol
                )

                weights[i, j] = np.exp(np.clip(score / temperature, -10, 10))

        np.fill_diagonal(weights, 1.0)
        T = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
        return T

    def apply_adaptive_leakage(
        self, T: np.ndarray, currencies: List[str], vix_z: float, indices: Any
    ) -> Tuple[np.ndarray, List[USDLeakageNode], float, RegimeData]:
        # 1. Leakage Intensity
        # Scales with VIX. If VIX Z > 1.5, Leakage -> 60%
        sigmoid = 1 / (1 + np.exp(-(vix_z - 1.0)))
        leakage_intensity = float(sigmoid * 0.60)

        # Direction Check (USD Trend)
        d_score = (
            float(indices.direction_score)
            if hasattr(indices, "direction_score")
            else float(indices["direction_score"])
        )
        if d_score < -20:
            leakage_intensity *= 0.2  # Reduce leakage if USD is weak

        n = len(currencies)
        leakage_nodes = []
        net_flow = 0.0
        T_adj = T.copy()

        # Apply Leakage
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

        # 2. Labeling
        s_score = (
            float(indices.stress_score)
            if hasattr(indices, "stress_score")
            else float(indices["stress_score"])
        )
        label, desc = "Neutral", "Markets are near historical averages."

        if s_score > 60:
            if d_score > 0:
                label = "USD Wrecking Ball"
                desc = "Systemic Stress + USD Strength. Defensive positioning required."
            else:
                label = "US-Centric Stress"
                desc = "Systemic Stress + USD Weakness. Rotate to CHF/JPY/Gold."
        elif s_score < 40:
            if d_score < 0:
                label = "Reflation / Risk-On"
                desc = "Low Stress + Weak USD. Favor High-Beta and Commodity FX."
            else:
                label = "Tightening / Carry"
                desc = "Low Stress + Strong USD. Favor Yield."

        # Package Result
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
            else float(indices.get("vix", 20.0)),
        )

        return (
            T_adj,
            leakage_nodes,
            net_flow,
            RegimeData(label=label, desc=desc, indices=idx_data),
        )
