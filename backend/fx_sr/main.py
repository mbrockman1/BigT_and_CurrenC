import traceback

import uvicorn

from .api import app, engine
from .schemas import BeliefParams


def main():
    print("--- Starting FX Macro Engine ---")

    try:
        print("Running initial warm-up analysis...")
        # Warm up the EOD cycle which is what /latest uses
        data = engine.run_eod_cycle()
        print(f"Warm-up complete. Analysis Date: {data['date']}")
    except Exception as e:
        print("!!! WARM-UP CRASHED !!!")
        traceback.print_exc()  # This shows you the EXACT line of failure

    print("Starting API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
