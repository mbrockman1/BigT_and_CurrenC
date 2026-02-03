# SR-FX Macro Intelligence Engine

An interpretable macro-financial FX engine based on **Multi-scale Successor Representations (SR)**.

This application translates computational models into capital-flow physics engine for G10 currencies.

---

## 🧠 Methodology

Unlike black-box ML models, this system uses a structural graph-based approach:

1. **Transition Matrix (`T`)**  
   Models one-step "attraction" between currencies based on relative momentum and volatility.

2. **Successor Representation (`M`)**  
   Computes expected future capital occupancy across multiple time horizons using:

   \[
   M = (I - \gamma T)^{-1}
   \]

3. **Regime Conditioning**  
   Decouples systemic **Stress** (disorder) from **USD Direction** (trend) to dynamically rewire the network.

4. **USD Reservoir**  
   Models USD not as a node, but as an external liquidity sink/source that absorbs probability mass during "Wrecking Ball" regimes.

5. **Carry Blending**  
   Dynamically integrates yield differentials into structural rankings during low-volatility tightening regimes.

---

## 🚀 Features

- **Macro Simulator**  
  Inject counterfactual beliefs (risk mix, trend persistence, volatility penalties) to observe how structural flows rewire.

- **Regime Evolution**  
  Track historical paths of relative currency ranks to identify regime-shift crossovers.

- **Walk-Forward Validation**  
  Causal backtesting framework that samples weekly, runs SR using only trailing data, and validates against realized forward returns.

- **Model Posture**  
  Automated strategy constraints (USD bias, hedging intensity, risk appetite) derived from the active regime.

---

## 🛠️ Tech Stack

**Backend**
- Python 3.10+
- FastAPI
- Pandas, NumPy
- SciPy (Linear Algebra)
- yfinance

**Frontend**
- React
- TypeScript
- Vite
- Tailwind CSS
- Recharts
- Lucide-React

---

## 💻 Installation & Setup

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

---

## 🏃 Running the Application

### Start the Backend API

```bash
# From the project root
cd backend
python -m fx_sr.main
```

The API will be available at:  
`http://localhost:8000`

A performance report will print to the terminal once the backtest warms up.

### Start the Frontend Dashboard

```bash
# From the project root
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 📊 Evaluation Metrics

- **Rank IC (Spearman Correlation)**  
  Measures alignment between predicted structural preference and realized forward returns.

- **Resilience Gap**  
  Measures outperformance of the model’s "Safe" picks versus "Risky" picks during high-stress regimes.

---

## 📌 Notes

This project is designed to be **interpretable-first**.  
Every ranking, transition, and regime shift is inspectable and grounded in explicit macro assumptions rather than opaque optimization.
