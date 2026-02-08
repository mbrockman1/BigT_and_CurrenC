from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# --- 1. Basic Components ---
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
    vix: Optional[float]  # <--- Add this field


class RegimeData(BaseModel):
    label: str
    desc: str
    indices: MacroIndices


# --- 2. Posture Definitions ---


# OLD Simple Posture (Keep for Backtest History)
class ModelPosture(BaseModel):
    usd_view: str
    fx_risk: str
    carry_view: str
    hedging: str
    trust_ranking: bool


# NEW Institutional Posture (For Dashboard & Simulation)
class MarketForces(BaseModel):
    rewarding: List[str]
    penalizing: List[str]


class InstitutionalPosture(BaseModel):
    headline: str
    usd_gauge: int
    risk_gauge: int
    carry_gauge: int
    hedging_gauge: int
    forces: MarketForces
    interpretation: str
    trust_ranking: bool


# --- 3. Inputs & History ---
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


class DailyDiff(BaseModel):
    stress_delta: float
    direction_delta: float
    regime_changed: bool
    prev_regime: Optional[str]
    confidence_delta: float


class RegimeConfidence(BaseModel):
    score: float
    persistence: int
    is_stable: bool


class WeeklyDelta(BaseModel):
    stress_delta: float
    direction_delta: float
    regime_shift: bool
    prev_label: Optional[str]


class RegimeTransition(BaseModel):
    risk_score: float
    next_likely_regime: str
    vector_desc: str
    is_breaking: bool


class CarryData(BaseModel):
    is_active: bool
    lambda_param: float
    raw_yields: Dict[str, float]
    yield_diffs: Dict[str, float]
    carry_scores: Dict[str, float]


class NetworkEdge(BaseModel):
    source: str
    target: str
    weight: float


class USDLeakageNode(BaseModel):
    iso: str
    leakage_prob: float
    is_source: bool


# --- 4. API Output Models ---


# Dashboard (/latest)
class EngineOutput(BaseModel):
    date: str
    regime: RegimeData
    posture: InstitutionalPosture
    confidence: RegimeConfidence
    diff: DailyDiff
    transition: RegimeTransition
    horizons: Dict[str, List[HorizonResult]]


# Simulation (/simulate)
# !!! THIS MUST BE InstitutionalPosture !!!
class SimulationOutput(BaseModel):
    mode: str
    params_used: BeliefParams
    horizons: Dict[str, List[HorizonResult]]
    posture: InstitutionalPosture
    transition: RegimeTransition


# Rolling History (/history)
class RollingDataPoint(BaseModel):
    date: str
    rankings: Dict[str, int]
    top_iso: str
    regime_vix: float


class RollingOutput(BaseModel):
    history: List[RollingDataPoint]


# Backtest (/walkforward)
class WalkForwardSnapshot(BaseModel):
    date: str
    horizon_results: Dict[str, List[Dict[str, Any]]]
    edges: Dict[str, List[NetworkEdge]]
    realized_returns: Dict[str, Dict[str, float]]
    metrics: Dict[str, Any]
    regime: RegimeData
    posture: ModelPosture  # Backtest uses the Simple version
    confidence: RegimeConfidence
    delta: WeeklyDelta
    transition: RegimeTransition
    usd_leakage: List[USDLeakageNode]
    net_usd_flow: float
    carry_data: CarryData


class WalkForwardOutput(BaseModel):
    history: List[WalkForwardSnapshot]
    correlations: Dict[str, float]


class ValidationMetrics(BaseModel):
    rank_ic: Optional[float]
    top_quartile_ret: Optional[float]
    btm_quartile_ret: Optional[float]
    resilience_gap: Optional[float]
    sr_only_ic: Optional[float] = None
    blended_ic: Optional[float] = None
