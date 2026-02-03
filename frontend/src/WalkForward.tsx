import { useState, useEffect } from "react";
import axios from "axios";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  Label,
} from "recharts";
import {
  Play,
  Pause,
  GitCommit,
  TrendingUp,
  Activity,
  Zap,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";

// --- Types (Matched to Backend) ---

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

interface ModelPosture {
  usd_view: string;
  fx_risk: string;
  carry_view: string;
  hedging: string;
  trust_ranking: boolean;
}

interface RegimeConfidence {
  score: number;
  persistence: number;
  is_stable: boolean;
}

interface WeeklyDelta {
  stress_chg: number;
  direction_chg: number;
  regime_shift: boolean;
  prev_label: string | null;
}

interface ValidationMetrics {
  rank_ic: number | null;
  top_quartile_ret: number | null;
  btm_quartile_ret: number | null;
  resilience_gap: number | null;
  sr_only_ic?: number | null;
  blended_ic?: number | null;
}

interface CarryData {
  is_active: boolean;
  lambda_param: number;
  raw_yields: Record<string, number>;
  yield_diffs: Record<string, number>;
  carry_scores: Record<string, number>;
}

interface Snapshot {
  date: string;
  horizon_results: Record<
    string,
    { iso: string; score: number; rank: number }[]
  >;
  edges: Record<string, { source: string; target: string; weight: number }[]>;
  realized_returns: Record<string, Record<string, number>>;
  metrics: Record<string, ValidationMetrics>;
  regime: RegimeData;
  posture: ModelPosture;
  confidence: RegimeConfidence;
  delta: WeeklyDelta;
  usd_leakage: { iso: string; leakage_prob: number }[];
  net_usd_flow: number;
  carry_data: CarryData;
}

interface WalkForwardData {
  history: Snapshot[];
  correlations: Record<string, number>;
}

// --- SUB-COMPONENTS ---

const WeeklyDeltaBadge = ({ delta }: { delta: WeeklyDelta }) => (
  <div className="flex flex-wrap gap-3 text-[9px] text-slate-500 uppercase font-black border-t border-slate-800 pt-2 mt-2">
    <span className="text-slate-600">Delta:</span>
    <span
      className={delta.stress_chg > 0 ? "text-rose-400" : "text-emerald-400"}
    >
      Stress {delta.stress_chg > 0 ? "↑" : "↓"}{" "}
      {Math.abs(delta.stress_chg).toFixed(0)}
    </span>
    <span
      className={delta.direction_chg > 0 ? "text-yellow-400" : "text-blue-400"}
    >
      USD {delta.direction_chg > 0 ? "↑" : "↓"}
    </span>
    {delta.regime_shift && (
      <span className="text-white bg-blue-600 px-1.5 rounded-sm">
        Regime Shift
      </span>
    )}
  </div>
);

const MarketNarrative = ({ snapshot }: { snapshot: Snapshot }) => {
  const { regime, posture, delta } = snapshot;
  return (
    <div className="bg-slate-900 border-l-2 border-blue-500 p-4 rounded-r-lg shadow-sm">
      <div className="flex justify-between items-center mb-1">
        <div className="flex items-center gap-2">
          <Zap size={14} className="text-yellow-500" />
          <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
            Macro Context
          </h4>
        </div>
        {!posture.trust_ranking && (
          <div className="bg-rose-500/10 text-rose-400 text-[9px] font-black px-2 rounded border border-rose-500/20 flex items-center gap-1">
            <AlertTriangle size={10} /> LOW ACCURACY REGIME
          </div>
        )}
      </div>
      <p className="text-xs text-slate-300 font-medium leading-relaxed">
        "{regime.desc}{" "}
        {posture.usd_view === "Overweight"
          ? "USD acts as a sink."
          : "Flows favor peer rotation."}
        "
      </p>
      <WeeklyDeltaBadge delta={delta} />
    </div>
  );
};

const NetworkGraph = ({
  edges,
  leakage,
  scores,
}: {
  edges: any[];
  leakage: any[];
  scores: any[];
}) => {
  const currencies = ["EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"];
  const radius = 85; // Tighter radius
  const center = 150;
  const [hoverNode, setHoverNode] = useState<string | null>(null);

  const nodes = currencies.map((iso, i) => {
    const angle = (i / currencies.length) * 2 * Math.PI - Math.PI / 2;
    const rawScore = scores?.find((s) => s.iso === iso)?.score || 0;
    // Compact bubble sizes
    const size = Math.max(8, Math.min(22, 6 + Math.sqrt(rawScore) * 4));
    return {
      iso,
      x: center + radius * Math.cos(angle),
      y: center + radius * Math.sin(angle),
      size,
    };
  });

  return (
    <div className="relative w-full h-full">
      <svg viewBox="0 0 300 300" className="w-full h-full">
        {/* USD Center */}
        <circle
          cx={center}
          cy={center}
          r={20}
          fill="#0f172a"
          stroke="#eab308"
          strokeWidth={2}
        />
        <text
          x={center}
          y={center}
          dy={4}
          textAnchor="middle"
          className="text-[10px] font-black fill-yellow-500 select-none"
        >
          USD
        </text>

        {/* Leakage (Red Dashed) */}
        {leakage?.map((l: any) => {
          const node = nodes.find((n) => n.iso === l.iso);
          if (!node || l.leakage_prob < 0.02) return null;
          return (
            <line
              key={`l-${l.iso}`}
              x1={node.x}
              y1={node.y}
              x2={center}
              y2={center}
              stroke="#ef4444"
              strokeWidth={l.leakage_prob * 10}
              strokeDasharray="3 2"
              opacity={0.4}
            />
          );
        })}

        {/* Peer Flows (Blue) */}
        {edges?.map((e: any, i: number) => {
          const start = nodes.find((n) => n.iso === e.source);
          const end = nodes.find((n) => n.iso === e.target);
          if (!start || !end) return null;
          const isDimmed =
            hoverNode && e.source !== hoverNode && e.target !== hoverNode;
          return (
            <line
              key={i}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              stroke="#3b82f6"
              strokeWidth={Math.max(1, e.weight * 4)}
              opacity={isDimmed ? 0.1 : 0.5}
            />
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => (
          <g
            key={node.iso}
            onMouseEnter={() => setHoverNode(node.iso)}
            onMouseLeave={() => setHoverNode(null)}
            className="cursor-pointer"
          >
            <circle
              cx={node.x}
              cy={node.y}
              r={node.size}
              fill={hoverNode === node.iso ? "#3b82f6" : "#1e293b"}
              stroke={hoverNode === node.iso ? "#fff" : "#475569"}
              strokeWidth={1.5}
            />
            <text
              x={node.x}
              y={node.y}
              dy={3}
              textAnchor="middle"
              className="text-[8px] font-bold fill-white select-none pointer-events-none"
            >
              {node.iso}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

// --- MAIN COMPONENT ---

export default function WalkForward() {
  const [data, setData] = useState<WalkForwardData | null>(null);
  const [idx, setIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [horizon, setHorizon] = useState<"short" | "medium" | "long">("medium");

  useEffect(() => {
    axios.get("http://localhost:8000/walkforward?weeks=52").then((res) => {
      setData(res.data);
      setIdx(res.data.history.length - 1);
    });
  }, []);

  useEffect(() => {
    let interval: any;
    if (isPlaying && data) {
      interval = setInterval(() => {
        setIdx((prev) => (prev + 1) % data.history.length);
      }, 600);
    }
    return () => clearInterval(interval);
  }, [isPlaying, data]);

  if (!data)
    return (
      <div className="flex flex-col items-center justify-center h-[50vh] text-slate-500 animate-pulse">
        <RefreshCw className="mb-4 animate-spin" />
        <span className="text-xs font-black uppercase tracking-widest">
          Running Historical Simulation...
        </span>
      </div>
    );

  const snap = data.history[idx];
  const scores = snap?.horizon_results[horizon] || [];
  const metrics = snap?.metrics[horizon];

  // Scatter Data
  const scatterData = scores
    .map((s: any) => ({
      iso: s.iso,
      score: s.score,
      ret: (snap.realized_returns[horizon]?.[s.iso] || 0) * 100,
    }))
    .filter((d: any) => d.ret !== 0);

  return (
    <div className="space-y-4">
      {/* 1. Timeline Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex gap-4 items-center">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="p-3 bg-blue-600 rounded-full text-white hover:bg-blue-500 transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <div className="flex-1">
          <div className="flex justify-between text-[10px] font-black uppercase text-slate-500 mb-1">
            <span>{data.history[0].date}</span>
            <span className="text-blue-400 font-mono">{snap.date}</span>
            <span>{data.history[data.history.length - 1].date}</span>
          </div>
          <input
            type="range"
            min={0}
            max={data.history.length - 1}
            value={idx}
            onChange={(e) => setIdx(parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>
        <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
          {["short", "medium", "long"].map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h as any)}
              className={`px-3 py-1 text-[9px] font-black uppercase rounded transition-all ${horizon === h ? "bg-slate-800 text-white" : "text-slate-500 hover:text-slate-300"}`}
            >
              {h}
            </button>
          ))}
        </div>
      </div>

      {/* 2. Narrative & Regime */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MarketNarrative snapshot={snap} />

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col justify-center">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-[9px] font-black text-slate-500 uppercase flex items-center gap-1">
                <Activity size={12} /> Current Regime
              </div>
              <div
                className={`text-xl font-black uppercase mt-1 ${snap.regime.indices.stress_score > 40 ? "text-rose-500" : "text-emerald-500"}`}
              >
                {snap.regime.label}
              </div>
            </div>
            <div className="text-right">
              <div className="text-[9px] font-black text-slate-500 uppercase">
                Rank IC
              </div>
              <div
                className={`text-xl font-mono font-bold ${metrics?.rank_ic && metrics.rank_ic > 0 ? "text-emerald-400" : "text-slate-500"}`}
              >
                {metrics?.rank_ic?.toFixed(3) || "--"}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Visualizations (Compact) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Network */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[300px] relative">
          <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest absolute top-4 left-4 flex items-center gap-2">
            <GitCommit size={14} /> Capital Topology
          </h3>
          <NetworkGraph
            edges={snap.edges["medium"]}
            leakage={snap.usd_leakage}
            scores={scores}
          />
        </div>

        {/* Scatter */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[300px] flex flex-col">
          <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
            <TrendingUp size={14} /> Prediction vs Outcome
          </h3>
          <div className="flex-grow">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 10, right: 10, bottom: 20, left: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  type="number"
                  dataKey="score"
                  stroke="#64748b"
                  fontSize={9}
                  tickLine={false}
                  axisLine={false}
                >
                  <Label
                    value="SR Score"
                    offset={-10}
                    position="insideBottom"
                    fill="#475569"
                    fontSize={9}
                    fontWeight="bold"
                  />
                </XAxis>
                <YAxis
                  type="number"
                  dataKey="ret"
                  unit="%"
                  stroke="#64748b"
                  fontSize={9}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #334155",
                    borderRadius: "6px",
                    fontSize: "10px",
                  }}
                />
                <ReferenceLine y={0} stroke="#334155" />
                <Scatter name="FX" data={scatterData}>
                  {scatterData.map((entry: any, index: number) => (
                    <Cell
                      key={index}
                      fill={entry.ret > 0 ? "#10b981" : "#ef4444"}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
