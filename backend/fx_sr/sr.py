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
    def get_gamma(days: int) -> float:
        days = max(int(days), 2)
        gamma = 1.0 - (1.0 / float(days))
        return min(gamma, 0.999)

    @staticmethod
    def compute_sr_matrix(T: np.ndarray, gamma: float) -> np.ndarray:
        n = T.shape[0]
        I = np.eye(n)
        gamma = min(float(gamma), 0.999)
        try:
            return inv(I - gamma * T)
        except Exception:
            return np.linalg.pinv(I - gamma * T)

    @staticmethod
    def compute_strength_scores(M: np.ndarray) -> np.ndarray:
        """
        Strength = column sums (expected future occupancy across all starts).
        """
        return M.sum(axis=0)
