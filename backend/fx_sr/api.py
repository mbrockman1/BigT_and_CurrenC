from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .backtest import WalkForwardEngine
from .engine import FXMacroEngine
from .schemas import (
    BeliefParams,
    EngineOutput,
    RollingOutput,
    SimulationOutput,
    WalkForwardOutput,
)

app = FastAPI(title="SR-FX Macro Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
engine = FXMacroEngine()
backtester = WalkForwardEngine()


@app.get("/latest", response_model=EngineOutput)
async def get_latest():
    """Returns latest EOD dashboard with Posture and Confidence."""
    try:
        # engine.run_eod_cycle() MUST return date, regime, posture, confidence, delta, horizons
        return engine.run_eod_cycle()
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate", response_model=SimulationOutput)
async def run_simulation(beliefs: BeliefParams):
    try:
        return engine.run_simulation(beliefs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=RollingOutput)
async def get_history(days: int = 90):
    try:
        return engine.run_rolling_analysis(lookback_window=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/walkforward", response_model=WalkForwardOutput)
async def get_walkforward(weeks: int = 52):
    try:
        return backtester.run_walk_forward(weeks=weeks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
