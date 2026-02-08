# fx_sr/sr.py
import numpy as np
from scipy.linalg import inv

from .config import FXConfig


class SREngine:
    """
    Computes multi-scale successor representations:
      M = (I - gamma * T)^-1
    """

    def __init__(self, config: FXConfig):
        self.config = config

    @staticmethod
    def get_gamma(days: float) -> float:
        """
        Continuous-time formulation: gamma = exp(-1/tau).
        This is more stable than (1 - 1/days) for small horizons.
        """
        # Avoid division by zero
        if days <= 0:
            return 0.0
        return np.exp(-1.0 / days)

    @staticmethod
    def compute_sr_matrix(T: np.ndarray, gamma: float) -> np.ndarray:
        n = T.shape[0]
        I = np.eye(n)
        g = min(float(gamma), 0.9995)
        try:
            return inv(I - g * T)
        except Exception:
            return np.linalg.pinv(I - g * T)

    @staticmethod
    def compute_strength_scores(M: np.ndarray) -> np.ndarray:
        """
        Standard Score: Column Sums (Eigenvector Centrality Approximation).
        PURE MATH: No sign flipping here. Strategy logic belongs in the Engine.
        """
        return M.sum(axis=0)

    @staticmethod
    def compute_multiscale_features(T: np.ndarray, currencies: list) -> dict:
        scales = [10, 21, 63, 126, 252]
        scores_by_scale = []

        for days in scales:
            g = np.exp(-1.0 / days)
            M = SREngine.compute_sr_matrix(T, g)
            raw = M.sum(axis=0)
            norm = (raw - raw.mean()) / (raw.std() + 1e-6)
            scores_by_scale.append(norm)

        scores_by_scale = np.array(scores_by_scale)

        features = {}
        for i, iso in enumerate(currencies):
            short_term = scores_by_scale[0, i]
            long_term = scores_by_scale[-1, i]
            slope = long_term - short_term
            persistence = np.mean(scores_by_scale[:, i])

            features[iso] = {"slope": float(slope), "persistence": float(persistence)}

        return features
