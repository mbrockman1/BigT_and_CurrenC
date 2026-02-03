from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# ==========================================
# 1. Basic Components
# ==========================================


class HorizonResult(BaseModel):
    iso: str
    score: float
    rank: int
    delta: float
    trend: str


class MacroIndices(BaseModel):
    stress_score: float
    direction_score: float
    stress_breadth: float
    stress_vol: float
    usd_momentum: float
    yield_delta: float


class RegimeData(BaseModel):
    label: str
    desc: str
    indices: MacroIndices


# ==========================================
# 2. Input Parameters (Beliefs & Shocks)
# ==========================================


class ShockParams(BaseModel):
    iso: Optional[str] = None
    vol_shock: float = 1.0
    mom_shock: float = 1.0


class BeliefParams(BaseModel):
    risk_mix: float = 0.5
    trend_sensitivity: float = 1.0
    vol_penalty: float = 1.0
    vix_override: Optional[float] = None
    shocks: Optional[List[ShockParams]] = []


# ==========================================
# 3. Carry & USD Layer
# ==========================================


class CarryData(BaseModel):
    is_active: bool
    lambda_param: float
    raw_yields: Dict[str, float]
    yield_diffs: Dict[str, float]
    carry_scores: Dict[str, float]


class USDLeakageNode(BaseModel):
    iso: str
    leakage_prob: float
    is_source: bool


class NetworkEdge(BaseModel):  # <--- WAS MISSING
    source: str
    target: str
    weight: float


# ==========================================
# 4. API Output Models
# ==========================================


class ModelPosture(BaseModel):
    usd_view: str  # "Overweight", "Neutral", "Underweight"
    fx_risk: str  # "Aggressive", "Selective", "Defensive"
    carry_view: str  # "Favor", "Neutral", "Avoid"
    hedging: str  # "None", "Light", "Heavy"
    trust_ranking: bool  # False if historical IC for this regime is poor


class RegimeConfidence(BaseModel):
    score: float  # 0-100
    persistence: int  # Weeks in current regime
    is_stable: bool  # True if score > 50


class WeeklyDelta(BaseModel):
    stress_chg: float
    direction_chg: float
    regime_shift: bool
    prev_label: Optional[str]


class EngineOutput(BaseModel):
    date: str
    regime: Any  # RegimeData
    posture: ModelPosture  # <--- NEW
    confidence: RegimeConfidence  # <--- NEW
    delta: WeeklyDelta  # <--- NEW
    horizons: Dict[str, List[Any]]  # HorizonResult


class SimulationOutput(BaseModel):
    mode: str
    params_used: Any
    horizons: Dict[str, List[Any]]
    # Add dummy posture for simulation
    posture: Optional[ModelPosture] = None


class RollingDataPoint(BaseModel):
    date: str
    rankings: Dict[str, int]
    top_iso: str
    regime_vix: float


class RollingOutput(BaseModel):
    history: List[RollingDataPoint]


# ==========================================
# 5. Walk-Forward Validation
# ==========================================


class ValidationMetrics(BaseModel):
    rank_ic: Optional[float]
    top_quartile_ret: Optional[float]
    btm_quartile_ret: Optional[float]
    resilience_gap: Optional[float]
    sr_only_ic: Optional[float] = None
    blended_ic: Optional[float] = None


class WalkForwardSnapshot(BaseModel):
    date: str
    horizon_results: Dict[str, List[Dict[str, Any]]]
    edges: Dict[str, List[Dict[str, Any]]]
    realized_returns: Dict[str, Dict[str, float]]
    metrics: Dict[str, Any]
    regime: Any
    usd_leakage: List[Any]
    net_usd_flow: float
    carry_data: Any

    # NEW fields for history viz
    posture: ModelPosture
    confidence: RegimeConfidence
    delta: WeeklyDelta


class WalkForwardOutput(BaseModel):
    history: List[WalkForwardSnapshot]
    correlations: Dict[str, float]


class RegimeConfidence(BaseModel):
    score: float  # 0-100
    persistence: int  # Weeks in current regime
    is_stable: bool  # True if score > 50
