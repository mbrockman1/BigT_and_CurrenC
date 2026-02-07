import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  ArrowUp,
  ArrowDown,
  Minus,
  Shield,
  TrendingUp,
  Anchor,
  Activity,
  Calendar,
  ShieldAlert,
  Zap,
  Sliders,
  RotateCcw,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";
// Import the new chart component from WalkForward (or define locally if preferred, but reusing logic is better)
// To keep this file drop-in simple, I will redefine ModernOccupancyMap here locally so you don't need to mess with exports.
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";
import WalkForward from "./WalkForward";
import { HelpCircle } from "lucide-react"; // Add to your imports

const InfoTooltip = ({
  title,
  content,
}: {
  title: string;
  content: string;
}) => (
  <div className="group relative inline-block ml-1.5 align-middle">
    <HelpCircle
      size={14}
      className="text-slate-500 hover:text-blue-400 cursor-help transition-colors"
    />
    <div className="absolute z-[100] hidden group-hover:block w-72 p-3 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl text-[11px] -left-36 bottom-6 pointer-events-none animate-in fade-in zoom-in duration-200">
      <div className="font-black text-blue-400 mb-1.5 uppercase tracking-widest border-b border-slate-800 pb-1">
        {title}
      </div>
      <div className="text-slate-300 leading-relaxed font-medium">
        {content}
      </div>
      {/* Small Arrow */}
      <div className="absolute h-2 w-2 bg-slate-900 border-r border-b border-slate-700 transform rotate-45 left-1/2 -translate-x-1/2 -bottom-1"></div>
    </div>
  </div>
);

// --- REUSED COMPONENT: ModernOccupancyMap ---
const ModernOccupancyMap = ({ data }: { data: any[] }) => {
  const sortedData = [...data].sort((a, b) => b.score - a.score);
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={sortedData}
        layout="vertical"
        margin={{ left: 0, right: 15, top: 10, bottom: 5 }}
      >
        <defs>
          <linearGradient id="barBlueApp" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.9} />
          </linearGradient>
          <linearGradient id="barDarkApp" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#1e293b" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#334155" stopOpacity={0.9} />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="2 4"
          stroke="#1e293b"
          horizontal={false}
          vertical={true}
        />
        <XAxis type="number" hide />
        <YAxis
          dataKey="iso"
          type="category"
          width={40}
          tick={{ fill: "#94a3b8", fontSize: 11, fontWeight: 700 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "#ffffff", opacity: 0.05 }}
          content={({ active, payload }) => {
            if (active && payload && payload.length) {
              const d = payload[0].payload;
              return (
                <div className="bg-slate-950 border border-slate-700 p-3 rounded-lg shadow-xl text-xs z-50">
                  <div className="font-black text-white text-sm mb-2 border-b border-slate-800 pb-1">
                    {d.iso}
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-400">Score:</span>
                    <span className="text-blue-400 font-mono font-bold">
                      {d.score.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4 mt-1">
                    <span className="text-slate-400">Rank:</span>
                    <span className="text-white font-mono font-bold">
                      #{d.rank}
                    </span>
                  </div>
                </div>
              );
            }
            return null;
          }}
        />
        <Bar
          dataKey="score"
          radius={[0, 4, 4, 0]}
          barSize={24}
          animationDuration={500}
        >
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={index < 2 ? "url(#barBlueApp)" : "url(#barDarkApp)"}
              stroke={index < 2 ? "#60a5fa" : "none"}
              strokeWidth={index < 2 ? 1 : 0}
            />
          ))}
          <LabelList
            dataKey="score"
            position="right"
            fill="#64748b"
            fontSize={9}
            formatter={(v: number) => v.toFixed(2)}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

// --- Type Definitions ---
interface HorizonResult {
  iso: string;
  score: number;
  rank: number;
  delta: number;
  trend: string;
}
interface ModelPosture {
  usd_view: string;
  fx_risk: string;
  carry_view: string;
  hedging: string;
  trust_ranking: boolean;
}
interface MacroIndices {
  stress_score: number;
  direction_score: number;
  stress_breadth: number;
  stress_vol: number;
  usd_momentum: number;
  yield_delta: number;
  vix: number; // <--- Add this
}
interface RegimeData {
  label: string;
  desc: string;
  indices: MacroIndices;
}
interface RegimeConfidence {
  score: number;
  persistence: number;
  is_stable: boolean;
}
interface EngineOutput {
  date: string;
  regime: RegimeData;
  posture: InstitutionalPosture; // Dashboard uses this
  confidence: RegimeConfidence;
  delta: WeeklyDelta;
  transition: RegimeTransition;
  horizons: Record<string, HorizonResult[]>;
  diff: DailyDiff;
}
interface SimulationOutput {
  mode: string;
  horizons: Record<string, HorizonResult[]>;
  posture: InstitutionalPosture; // Simulator uses this now
}
interface BeliefParams {
  risk_mix: number;
  trend_sensitivity: number;
  vol_penalty: number;
  vix_override?: number | null;
}

interface RegimeTransition {
  risk_score: number;
  next_likely_regime: string;
  vector_desc: string;
  is_breaking: boolean;
}

interface MarketForces {
  rewarding: string[];
  penalizing: string[];
}

interface InstitutionalPosture {
  headline: string;
  usd_gauge: number;
  risk_gauge: number;
  carry_gauge: number;
  hedging_gauge: number;
  forces: MarketForces;
  interpretation: string;
  trust_ranking: boolean;
}

interface DailyDiff {
  stress_delta: number;
  direction_delta: number;
  regime_changed: boolean;
  prev_regime: string | null;
  confidence_delta: number;
}

// --- Internal Components ---
const MarketPosturePanel = ({
  posture,
  diff,
}: {
  posture: InstitutionalPosture;
  diff: DailyDiff;
}) => {
  // Helper for the gauge bars
  const Gauge = ({
    label,
    value,
    color,
  }: {
    label: string;
    value: number;
    color: string;
  }) => (
    <div className="mb-3">
      <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest mb-1 text-slate-400">
        <span>{label}</span>
        <span className="text-white">{value}%</span>
      </div>
      <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-1000 ${color}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );

  // Helper for Diffs
  const DeltaIndicator = ({ val, label }: { val: number; label: string }) => {
    if (Math.abs(val) < 1) return null; // Hide noise
    const isPos = val > 0;
    return (
      <div
        className={`flex items-center gap-1 text-[9px] font-bold uppercase px-2 py-1 rounded border ${isPos ? "bg-slate-800 border-slate-700 text-slate-300" : "bg-slate-800 border-slate-700 text-slate-300"}`}
      >
        {isPos ? (
          <ArrowUp size={10} className="text-emerald-400" />
        ) : (
          <ArrowDown size={10} className="text-rose-400" />
        )}
        <span>
          {label} {Math.abs(val).toFixed(0)}
        </span>
      </div>
    );
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl mb-6">
      {/* 1. Header: Headline & Change */}
      <div className="p-5 border-b border-slate-800 bg-gradient-to-r from-slate-900 to-slate-800/50">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-2xl font-black text-white uppercase tracking-tighter leading-none">
              {posture.headline}
            </h2>
            <div className="flex gap-2 mt-3">
              {diff.regime_changed && (
                <span className="text-[9px] font-bold bg-blue-600 text-white px-2 py-1 rounded animate-pulse">
                  Regime Shift: Was {diff.prev_regime}
                </span>
              )}
              <DeltaIndicator val={diff.stress_delta} label="Stress" />
              <DeltaIndicator val={diff.direction_delta} label="USD Trend" />
              <DeltaIndicator val={diff.confidence_delta} label="Confidence" />
            </div>
          </div>
          <Shield className="text-slate-700 opacity-20" size={48} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2">
        {/* 2. Gauges (Constraints) */}
        <div className="p-5 border-r border-slate-800">
          <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">
            Operational Constraints
          </h4>
          <Gauge
            label="USD Bias (Long)"
            value={posture.usd_gauge}
            color="bg-blue-500"
          />
          <Gauge
            label="Risk Budget"
            value={posture.risk_gauge}
            color="bg-emerald-500"
          />
          <Gauge
            label="Carry Viability"
            value={posture.carry_gauge}
            color="bg-yellow-500"
          />
          <Gauge
            label="Hedging Pressure"
            value={posture.hedging_gauge}
            color="bg-rose-500"
          />
        </div>

        {/* 3. Forces & Interpretation */}
        <div className="p-5 flex flex-col justify-between">
          <div>
            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3">
              Market Forces
            </h4>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <span className="text-[9px] text-emerald-400 font-bold block mb-1">
                  REWARDING
                </span>
                <ul className="text-[10px] text-slate-300 space-y-1">
                  {posture.forces.rewarding.map((f) => (
                    <li key={f} className="flex items-center gap-1">
                      <Zap size={8} className="text-emerald-500" /> {f}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <span className="text-[9px] text-rose-400 font-bold block mb-1">
                  PENALIZING
                </span>
                <ul className="text-[10px] text-slate-300 space-y-1">
                  {posture.forces.penalizing.map((f) => (
                    <li key={f} className="flex items-center gap-1">
                      <Minus size={8} className="text-rose-500" /> {f}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <h5 className="text-[9px] font-bold text-blue-400 uppercase mb-1">
              Interpretation
            </h5>
            <p className="text-xs text-slate-300 italic leading-snug">
              "{posture.interpretation}"
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, subtext, icon: Icon, colorClass }: any) => (
  <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-start space-x-4 shadow-sm hover:border-slate-700 transition-colors">
    <div
      className={`p-3 rounded-lg bg-opacity-10 ${colorClass.replace("text-", "bg-")}`}
    >
      <Icon className={`w-6 h-6 ${colorClass}`} />
    </div>
    <div>
      <p className="text-slate-400 text-[10px] font-bold uppercase tracking-widest">
        {label}
      </p>
      <h3 className="text-lg font-bold text-white mt-0.5 leading-tight">
        {value}
      </h3>
      <p className="text-[10px] text-slate-500 mt-0.5">{subtext}</p>
    </div>
  </div>
);

const PostureCard = ({
  posture,
  confidence,
}: {
  posture: ModelPosture;
  confidence: RegimeConfidence;
}) => (
  <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6 shadow-lg">
    <div className="flex justify-between items-start mb-4">
      <div>
        <h3 className="text-sm font-black text-white uppercase tracking-widest flex items-center gap-2">
          <ShieldAlert size={16} className="text-blue-500" /> Model Stance
          <InfoTooltip
            title="Strategy Posture"
            content="Deterministic rules for a portfolio. USD View: Hold more/less dollars. FX Risk: Aggressive vs Defensive positioning. Hedging: Intensity of crash insurance required."
          />
        </h3>
      </div>
      <div className="text-right">
        <div className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">
          Confidence
        </div>
        <div
          className={`text-xl font-mono font-bold ${confidence?.score > 60 ? "text-emerald-400" : "text-orange-400"}`}
        >
          {confidence?.score?.toFixed(0)}%
        </div>
      </div>
    </div>
    <div className="grid grid-cols-2 gap-3 text-center">
      <div className="bg-slate-950 p-2 rounded border border-slate-800">
        <div className="text-[9px] text-slate-500 uppercase font-bold">USD</div>
        <div className="text-xs font-black uppercase mt-1 text-white">
          {posture?.usd_view}
        </div>
      </div>
      <div className="bg-slate-950 p-2 rounded border border-slate-800">
        <div className="text-[9px] text-slate-500 uppercase font-bold">
          Risk
        </div>
        <div className="text-xs font-black uppercase mt-1 text-white">
          {posture?.fx_risk}
        </div>
      </div>
      <div className="bg-slate-950 p-2 rounded border border-slate-800">
        <div className="text-[9px] text-slate-500 uppercase font-bold">
          Hedging
        </div>
        <div className="text-xs font-black uppercase mt-1 text-white">
          {posture?.hedging}
        </div>
      </div>
      <div className="bg-slate-950 p-2 rounded border border-slate-800">
        <div className="text-[9px] text-slate-500 uppercase font-bold">
          Carry
        </div>
        <div className="text-xs font-black uppercase mt-1 text-white">
          {posture?.carry_view}
        </div>
      </div>
    </div>
    {!posture?.trust_ranking && (
      <div className="mt-4 bg-rose-900/20 border border-rose-500/30 p-2 rounded-lg flex items-center justify-center gap-2">
        <AlertTriangle size={14} className="text-rose-500" />
        <span className="text-[10px] font-black text-rose-400 uppercase tracking-tighter">
          Rankings Unreliable
        </span>
      </div>
    )}
  </div>
);

const SliderControl = ({
  label,
  value,
  onChange,
  min,
  max,
  step,
  resetVal,
}: any) => (
  <div className="mb-5">
    <div className="flex justify-between mb-1.5">
      <label className="text-xs font-bold text-slate-400 uppercase tracking-tighter">
        {label}
      </label>
      <div className="flex items-center gap-2">
        <span className="text-blue-400 font-mono text-xs font-bold">
          {value.toFixed(1)}
        </span>
        <button
          onClick={() => onChange(resetVal)}
          className="text-slate-600 hover:text-slate-300"
        >
          <RotateCcw size={10} />
        </button>
      </div>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
    />
  </div>
);

const TransitionRadar = ({ transition }: { transition: RegimeTransition }) => {
  const isSafe = transition.risk_score < 40;
  const isCritical = transition.risk_score > 75;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 shadow-sm">
      {" "}
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
          <Activity
            size={14}
            className={
              isCritical ? "text-rose-500 animate-pulse" : "text-blue-500"
            }
          />
          Regime Stability
          <InfoTooltip
            title="Transition Radar"
            content="Calculates the probability of a regime shift by measuring the 'Euclidean distance' of current macro data to the quadrant boundaries. High risk scores indicate the model is drifting toward a tipping point (e.g., Stress crossing 40 or DXY crossing 0), suggesting the current strategy posture may soon break."
          />
        </h3>
        <span
          className={`text-xs font-mono font-bold ${isCritical ? "text-rose-400" : isSafe ? "text-emerald-400" : "text-yellow-400"}`}
        >
          {transition.risk_score.toFixed(0)}% Risk
        </span>
      </div>
      {/* Progress Bar */}
      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mb-3">
        <div
          className={`h-full transition-all duration-1000 ease-out ${isCritical ? "bg-rose-500" : isSafe ? "bg-emerald-500" : "bg-yellow-500"}`}
          style={{ width: `${transition.risk_score}%` }}
        />
      </div>
      <div className="text-[10px] text-slate-300 font-medium flex justify-between items-center">
        <span>
          Status:{" "}
          <span className="text-white">
            {isSafe ? "Locked In" : "Transitioning"}
          </span>
        </span>
        {transition.risk_score > 20 && (
          <span className="flex items-center gap-1 text-slate-400">
            → {transition.next_likely_regime}
          </span>
        )}
      </div>
    </div>
  );
};

// --- MAIN APPLICATION ---
export default function App() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "walkforward">(
    "dashboard",
  );
  const [baseData, setBaseData] = useState<EngineOutput | null>(null);
  const [simData, setSimData] = useState<SimulationOutput | null>(null);
  const [loading, setLoading] = useState(true);
  const [horizon, setHorizon] = useState<"short" | "medium" | "long">("medium");
  const [isSimulating, setIsSimulating] = useState(false);
  const [riskMix, setRiskMix] = useState(0.5);
  const [trendSens, setTrendSens] = useState(1.0);
  const [volPen, setVolPen] = useState(1.0);

  useEffect(() => {
    fetchLatest();
  }, []);

  const fetchLatest = async () => {
    try {
      const res = await axios.get("http://localhost:8000/latest");
      setBaseData(res.data);
    } catch (e) {
      console.error("API Error:", e);
    } finally {
      setLoading(false);
    }
  };

  const runSimulation = useCallback(async () => {
    setIsSimulating(true);
    try {
      const payload: BeliefParams = {
        risk_mix: riskMix,
        trend_sensitivity: trendSens,
        vol_penalty: volPen,
      };
      const res = await axios.post("http://localhost:8000/simulate", payload);
      setSimData(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSimulating(false);
    }
  }, [riskMix, trendSens, volPen]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (baseData) runSimulation();
    }, 400);
    return () => clearTimeout(timer);
  }, [riskMix, trendSens, volPen, runSimulation, baseData]);

  if (loading)
    return (
      <div className="h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-500 gap-4">
        <RefreshCw className="animate-spin text-blue-500" size={32} />
        <p className="font-bold tracking-widest uppercase text-xs">
          Initializing Engine...
        </p>
      </div>
    );
  if (!baseData)
    return (
      <div className="h-screen bg-slate-950 flex flex-col items-center justify-center text-rose-500 p-10 text-center font-sans">
        <ShieldAlert size={48} className="mb-4" />
        <h1 className="text-xl font-bold">API Connection Lost</h1>
        <button
          onClick={fetchLatest}
          className="mt-4 px-4 py-2 bg-slate-800 text-white rounded text-xs uppercase font-bold"
        >
          Retry Connection
        </button>
      </div>
    );

  // Inside the App() component function:
  const activeHorizons = simData ? simData.horizons : baseData.horizons;
  const isCounterfactual =
    Math.abs(riskMix - 0.5) > 0.01 ||
    Math.abs(trendSens - 1.0) > 0.01 ||
    Math.abs(volPen - 1.0) > 0.01;

  const useSim = isCounterfactual && simData;
  const currentList = useSim
    ? simData.horizons[horizon]
    : baseData.horizons[horizon];
  // NEW: Logic to switch between Real Advice and Simulated Advice
  const currentPosture = useSim ? simData.posture : baseData.posture;

  const currentConfidence = baseData.confidence;
  const currentRegime = useSim
    ? { ...baseData.regime, label: `Simulated: ${simData.posture.headline}` }
    : baseData.regime;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex flex-col md:flex-row justify-between items-center border-b border-slate-800 pb-6">
          <div className="flex items-center gap-4">
            <div className="p-2 bg-blue-600 rounded-lg shadow-lg shadow-blue-900/20">
              <Activity className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-2xl font-black text-white tracking-tight uppercase">
                SR-FX Macro
              </h1>
              <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest">
                Structural Intelligence • {baseData.date}
              </p>
            </div>
          </div>
          <div className="flex bg-slate-900 p-1 rounded-xl mt-6 md:mt-0 border border-slate-800">
            {(["dashboard", "walkforward"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-5 py-2 rounded-lg text-xs font-bold uppercase tracking-tighter transition-all ${activeTab === tab ? "bg-slate-800 text-blue-400 shadow-inner" : "text-slate-500 hover:text-slate-300"}`}
              >
                {tab === "dashboard" ? "Simulator" : "Backtest"}
              </button>
            ))}
          </div>
        </div>

        {activeTab === "dashboard" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-3 bg-slate-900 border border-slate-800 rounded-2xl p-6 h-fit shadow-xl">
              <div className="flex items-center gap-2 mb-8 text-white font-black text-sm uppercase tracking-widest">
                <Sliders size={16} className="text-blue-500" /> Belief Injection
                <InfoTooltip
                  title="Belief Injection"
                  content="Manually shift the model’s 'Brain'. Mix Slider: Wrecking Ball (Crisis) vs Reflation (Growth). Persistence: Chase trends. Penalty: Filter out noisy/volatile currencies."
                />
              </div>
              <SliderControl
                label="Regime Mix (Risk)"
                value={riskMix}
                min={0}
                max={1}
                step={0.1}
                resetVal={0.5}
                onChange={setRiskMix}
              />
              <div className="flex justify-between text-[9px] font-black text-slate-600 uppercase mb-8 -mt-4">
                <span>Reflation</span>
                <span>Wrecking Ball</span>
              </div>
              <SliderControl
                label="Trend Persistence"
                value={trendSens}
                min={0}
                max={3}
                step={0.1}
                resetVal={1.0}
                onChange={setTrendSens}
              />
              <SliderControl
                label="Volatility Penalty"
                value={volPen}
                min={0}
                max={5}
                step={0.5}
                resetVal={1.0}
                onChange={setVolPen}
              />
              <div
                className={`mt-10 p-4 rounded-xl border text-[10px] font-bold uppercase tracking-widest text-center transition-all duration-500 ${isCounterfactual ? "bg-blue-600/10 border-blue-500/50 text-blue-400 animate-pulse" : "bg-slate-800/50 border-slate-700 text-slate-500"}`}
              >
                {isCounterfactual
                  ? "Counterfactual Active"
                  : "Live Market Mode"}
              </div>
            </div>

            <div className="lg:col-span-9 space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <StatCard
                  label={
                    <>
                      Regime{" "}
                      <InfoTooltip
                        title="Market Regime"
                        content="The overall macro 'season'. Wrecking Ball (Panic/USD Strength), Reflation (Growth/USD Weakness), Tightening (Orderly USD Strength/Yield driven), or US-Stress (US-specific crisis)."
                      />
                    </>
                  }
                  value={useSim ? "Simulation" : currentRegime.label}
                  subtext={
                    useSim
                      ? "Viewing counterfactual scenario"
                      : currentRegime.desc
                  }
                  icon={ShieldAlert}
                  colorClass={
                    baseData.regime.indices.stress_score > 40
                      ? "text-orange-500"
                      : "text-emerald-500"
                  }
                />
                <StatCard
                  label="DXY Direction"
                  value={
                    baseData.regime.indices.direction_score > 0
                      ? "BULLISH"
                      : "BEARISH"
                  }
                  subtext={`Score: ${baseData.regime.indices.direction_score.toFixed(0)}`}
                  icon={Zap}
                  colorClass={
                    baseData.regime.indices.direction_score > 0
                      ? "text-yellow-500"
                      : "text-blue-500"
                  }
                />
                <StatCard
                  label={
                    <>
                      FX Stress{" "}
                      <InfoTooltip
                        title="FX Stress Level"
                        content="Market 'Fever' index. Below 40 is healthy. Above 60 suggests the global liquidity plumbing is starting to clog with fear and volatility."
                      />
                    </>
                  }
                  value={`${baseData.regime.indices.stress_score.toFixed(0)}/100`}
                  subtext="Vol + Breadth"
                  icon={Activity}
                  colorClass="text-purple-500"
                />
                <StatCard
                  label={
                    <>
                      VIX Index{" "}
                      <InfoTooltip
                        title="Market Fear"
                        content="The CBOE Volatility Index. Measures S&P 500 implied volatility; used here as a global risk-off baseline."
                      />
                    </>
                  }
                  value={baseData.regime.indices.vix.toFixed(2)}
                  subtext="Equity Vol Proxy"
                  icon={Activity}
                  colorClass="text-rose-400"
                />
                <StatCard
                  label="Top Pick"
                  value={currentList[0]?.iso || "---"}
                  subtext={`Horizon: ${horizon}`}
                  icon={Calendar}
                  colorClass="text-blue-500"
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-5 space-y-4">
                  <MarketPosturePanel
                    posture={currentPosture}
                    // If simulating, pass a 'empty' diff so we don't show real-world deltas
                    // on a hypothetical scenario
                    diff={
                      useSim
                        ? {
                            stress_delta: 0,
                            direction_delta: 0,
                            regime_changed: false,
                            prev_regime: null,
                            confidence_delta: 0,
                          }
                        : baseData.diff
                    }
                  />
                  <TransitionRadar transition={baseData.transition} />
                </div>
                <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-[10px] font-black text-white uppercase tracking-widest">
                      Future Occupancy Map
                    </h3>
                    <div className="flex bg-slate-800 rounded-xl p-1 border border-slate-700">
                      {(["short", "medium", "long"] as const).map((h) => (
                        <button
                          key={h}
                          onClick={() => setHorizon(h)}
                          className={`px-4 py-1 text-[9px] font-black uppercase rounded-lg transition-all ${horizon === h ? "bg-slate-700 text-white shadow-lg" : "text-slate-400 hover:text-white"}`}
                        >
                          {h}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="h-[300px] w-full">
                    <ModernOccupancyMap data={currentList} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "walkforward" && <WalkForward />}
      </div>
    </div>
  );
}
