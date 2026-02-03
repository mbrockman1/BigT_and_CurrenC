import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  Legend,
  Label,
} from "recharts";
import {
  Activity,
  Calendar,
  ShieldAlert,
  Zap,
  Sliders,
  RotateCcw,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";
import WalkForward from "./WalkForward";

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
  posture: ModelPosture;
  confidence: RegimeConfidence;
  horizons: Record<string, HorizonResult[]>;
}

interface SimulationOutput {
  mode: string;
  horizons: Record<string, HorizonResult[]>;
}

interface HistoryPoint {
  date: string;
  rankings: Record<string, number>;
  top_iso: string;
  regime_vix: number;
}

interface BeliefParams {
  risk_mix: number;
  trend_sensitivity: number;
  vol_penalty: number;
  vix_override?: number | null;
}

// --- Internal Components ---

const StatCard = ({ label, value, subtext, icon: Icon, colorClass }: any) => (
  <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-start space-x-4 shadow-sm">
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

// --- MAIN APPLICATION ---

export default function App() {
  const [activeTab, setActiveTab] = useState<
    "dashboard" | "history" | "walkforward"
  >("dashboard");
  const [baseData, setBaseData] = useState<EngineOutput | null>(null);
  const [simData, setSimData] = useState<SimulationOutput | null>(null);
  const [loading, setLoading] = useState(true);
  const [horizon, setHorizon] = useState<"short" | "medium" | "long">("medium");

  // Simulation Sliders
  const [isSimulating, setIsSimulating] = useState(false);
  const [riskMix, setRiskMix] = useState(0.5);
  const [trendSens, setTrendSens] = useState(1.0);
  const [volPen, setVolPen] = useState(1.0);
  const [historyData, setHistoryData] = useState<HistoryPoint[]>([]);

  useEffect(() => {
    fetchLatest();
    fetchHistory();
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

  const fetchHistory = async () => {
    try {
      const res = await axios.get("http://localhost:8000/history?days=60");
      setHistoryData(res.data.history);
    } catch (e) {
      console.error(e);
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

  // Trigger simulation on slider change
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

  const activeHorizons = simData ? simData.horizons : baseData.horizons;
  const currentList = activeHorizons[horizon] || [];
  const isCounterfactual =
    riskMix !== 0.5 || trendSens !== 1.0 || volPen !== 1.0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
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
            {(["dashboard", "history", "walkforward"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-5 py-2 rounded-lg text-xs font-bold uppercase tracking-tighter transition-all ${activeTab === tab ? "bg-slate-800 text-blue-400 shadow-inner" : "text-slate-500 hover:text-slate-300"}`}
              >
                {tab
                  .replace("dashboard", "Simulator")
                  .replace("history", "Evolution")
                  .replace("walkforward", "Backtest")}
              </button>
            ))}
          </div>
        </div>

        {activeTab === "dashboard" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* --- SIDEBAR: SLIDERS --- */}
            <div className="lg:col-span-3 bg-slate-900 border border-slate-800 rounded-2xl p-6 h-fit shadow-xl">
              <div className="flex items-center gap-2 mb-8 text-white font-black text-sm uppercase tracking-widest">
                <Sliders size={16} className="text-blue-500" /> Belief Injection
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
                className={`mt-10 p-4 rounded-xl border text-[10px] font-bold uppercase tracking-widest text-center transition-all duration-500
                ${isCounterfactual ? "bg-blue-600/10 border-blue-500/50 text-blue-400 animate-pulse" : "bg-slate-800/50 border-slate-700 text-slate-500"}`}
              >
                {isCounterfactual
                  ? "Counterfactual Active"
                  : "Live Market Mode"}
              </div>
            </div>

            {/* --- MAIN CONTENT --- */}
            <div className="lg:col-span-9 space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                  label="Regime"
                  value={baseData.regime.label}
                  subtext={baseData.regime.desc}
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
                  label="FX Stress"
                  value={`${baseData.regime.indices.stress_score.toFixed(0)}/100`}
                  subtext="Vol + Breadth"
                  icon={Activity}
                  colorClass="text-purple-500"
                />
                <StatCard
                  label="Top Predicted"
                  value={currentList[0]?.iso || "---"}
                  subtext={`Horizon: ${horizon}`}
                  icon={Calendar}
                  colorClass="text-blue-500"
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-5">
                  <PostureCard
                    posture={baseData.posture}
                    confidence={baseData.confidence}
                  />
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

                  <div className="h-[350px] w-full mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={currentList}
                        layout="vertical"
                        margin={{ left: 10, right: 30, bottom: 20 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#1e293b"
                          horizontal={true}
                          vertical={false}
                        />

                        {/* Correct X-Axis with Label */}
                        <XAxis
                          type="number"
                          stroke="#475569"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                        >
                          <Label
                            value="Structural Flow Accumulation (SR Score)"
                            offset={-10}
                            position="insideBottom"
                            fill="#475569"
                            fontSize={10}
                            fontWeight="bold"
                          />
                        </XAxis>

                        <YAxis
                          dataKey="iso"
                          type="category"
                          stroke="#94a3b8"
                          width={40}
                          fontSize={12}
                          tickLine={false}
                          axisLine={false}
                          tick={{ fontWeight: "bold" }}
                        />

                        <RechartsTooltip
                          cursor={{ fill: "#1e293b", opacity: 0.4 }}
                          content={({ payload }) => {
                            if (payload && payload.length) {
                              const d = payload[0].payload;
                              return (
                                <div className="bg-slate-950 border border-slate-700 p-3 rounded-xl shadow-2xl text-[11px] font-sans">
                                  <div className="font-black text-white text-sm mb-2 border-b border-slate-800 pb-1">
                                    {d.iso} Detail
                                  </div>
                                  <div className="flex justify-between gap-6 mb-1">
                                    <span className="text-slate-500 uppercase font-bold tracking-tighter">
                                      SR Score
                                    </span>
                                    <span className="text-blue-400 font-mono font-bold">
                                      {d.score.toFixed(4)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between gap-6">
                                    <span className="text-slate-500 uppercase font-bold tracking-tighter">
                                      Current Rank
                                    </span>
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

                        <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={22}>
                          {currentList.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              /* Coloring: Top 2 blue, others professional slate-blue */
                              fill={index < 2 ? "#3b82f6" : "#2d3748"}
                              stroke={index < 2 ? "#60a5fa" : "none"}
                              strokeWidth={1}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Evolution View */}
        {activeTab === "history" && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl h-[500px]">
            <h3 className="text-xl font-black text-white mb-6 uppercase tracking-tight">
              Rank History
            </h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="date"
                  stroke="#475569"
                  fontSize={10}
                  tickFormatter={(s) => s.slice(5)}
                />
                <YAxis
                  stroke="#475569"
                  reversed
                  domain={[1, 8]}
                  fontSize={10}
                />
                <RechartsTooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #334155",
                  }}
                />
                <Legend iconType="circle" />
                {["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"].map(
                  (iso, i) => (
                    <Line
                      key={iso}
                      type="stepAfter"
                      dataKey={`rankings.${iso}`}
                      name={iso}
                      stroke={
                        [
                          "#22c55e",
                          "#3b82f6",
                          "#ef4444",
                          "#eab308",
                          "#a855f7",
                          "#ec4899",
                          "#6366f1",
                          "#14b8a6",
                        ][i]
                      }
                      strokeWidth={3}
                      dot={false}
                    />
                  ),
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {activeTab === "walkforward" && <WalkForward />}
      </div>
    </div>
  );
}
