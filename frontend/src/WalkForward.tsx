import { useState, useEffect } from "react";
import axios from "axios";
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
  ScatterChart,
  Scatter,
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
  HelpCircle,
} from "lucide-react";

// --- Types ---
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
    {/* Use fixed z-index and ensure it's not clipped by overflow */}
    <div className="absolute z-[9999] hidden group-hover:block w-64 p-3 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl text-[10px] -left-32 bottom-6 pointer-events-none animate-in fade-in zoom-in duration-200">
      <div className="font-black text-blue-400 mb-1 uppercase tracking-widest border-b border-slate-800 pb-1">
        {title}
      </div>
      <div className="text-slate-300 leading-relaxed font-medium">
        {content}
      </div>
      <div className="absolute h-2 w-2 bg-slate-900 border-r border-b border-slate-700 transform rotate-45 left-1/2 -translate-x-1/2 -bottom-1"></div>
    </div>
  </div>
);

// --- 1. MODERN TOPOLOGY GRAPH (Custom SVG) ---
const ModernCapitalTopology = ({
  edges,
  leakage,
  scores,
}: {
  edges: any[];
  leakage: any[];
  scores: any[];
}) => {
  const currencies = [
    "EUR",
    "JPY",
    "GBP",
    "AUD",
    "CAD",
    "CHF",
    "NZD",
    // "XAU",
    // "SPY",
  ];
  const radius = 85;
  const center = 150;
  const [hoverNode, setHoverNode] = useState<string | null>(null);

  // Calculate Node Positions & Sizes
  const nodes = currencies.map((iso, i) => {
    const angle = (i / currencies.length) * 2 * Math.PI - Math.PI / 2;
    const rawScore = scores?.find((s) => s.iso === iso)?.score || 0;
    // Logarithmic scaling to prevent giant bubbles
    const size = Math.max(10, Math.min(26, 8 + Math.log(rawScore + 1) * 8));
    return {
      iso,
      x: center + radius * Math.cos(angle),
      y: center + radius * Math.sin(angle),
      size,
      rawScore,
    };
  });

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <svg viewBox="0 0 300 300" className="w-full h-full">
        <defs>
          {/* Neon Glow Filters */}
          <filter id="glow-blue" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="1.5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Gradients */}
          <radialGradient id="grad-usd" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0%" stopColor="#eab308" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#854d0e" stopOpacity="0.2" />
          </radialGradient>
          <linearGradient id="grad-flow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.1" />
            <stop offset="50%" stopColor="#60a5fa" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.1" />
          </linearGradient>
        </defs>

        <style>
          {`
            @keyframes flow { to { stroke-dashoffset: -20; } }
            @keyframes pulse { 0% { r: 20px; opacity: 0.8; } 50% { r: 22px; opacity: 1; } 100% { r: 20px; opacity: 0.8; } }
            .flow-line { animation: flow 1s linear infinite; }
            .usd-core { animation: pulse 3s ease-in-out infinite; }
            .node-group { transition: opacity 0.3s ease; }
          `}
        </style>

        {/* Links: Leakage (Red) */}
        {leakage?.map((l: any) => {
          const node = nodes.find((n) => n.iso === l.iso);
          if (!node || l.leakage_prob < 0.02) return null;
          return (
            <g key={`leak-${l.iso}`} className="node-group">
              <line
                x1={node.x}
                y1={node.y}
                x2={center}
                y2={center}
                stroke="#ef4444"
                strokeWidth={1}
                strokeDasharray="2 4"
                opacity={0.3}
              />
              {/* Animated Packet */}
              <line
                x1={node.x}
                y1={node.y}
                x2={center}
                y2={center}
                stroke="#f87171"
                strokeWidth={l.leakage_prob * 10}
                strokeDasharray="4 6"
                className="flow-line"
                opacity={0.6}
                filter="url(#glow-red)"
              />
            </g>
          );
        })}

        {/* Links: Peer Flows (Blue) */}
        {edges?.map((e: any, i: number) => {
          const start = nodes.find((n) => n.iso === e.source);
          // FIX: If target is USD, use the 'center' coordinates, otherwise find the node
          const end =
            e.target === "USD"
              ? { x: center, y: center }
              : nodes.find((n) => n.iso === e.target);

          if (!start || !end) return null;

          const isUSD = e.target === "USD";
          const isFocused =
            hoverNode && (e.source === hoverNode || e.target === hoverNode);
          const isDimmed = hoverNode && !isFocused;

          return (
            <g
              key={`edge-${i}`}
              style={{
                opacity: isDimmed ? 0.1 : 1,
                transition: "opacity 0.3s",
              }}
            >
              <line
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke={isUSD ? "#ef4444" : "#3b82f6"}
                strokeWidth={Math.max(1, e.weight * 5)}
                strokeDasharray={isUSD ? "4 2" : "0"}
                className={isUSD ? "leak-line" : "flow-line"}
                opacity={isUSD ? 0.4 : 0.6}
              />
            </g>
          );
        })}

        {/* USD Core */}
        <circle
          cx={center}
          cy={center}
          r={28}
          fill="url(#grad-usd)"
          className="usd-core"
        />
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
          className="text-[10px] font-black fill-white select-none"
        >
          USD
        </text>

        {/* Currency Nodes */}
        {nodes.map((node) => {
          const isHovered = hoverNode === node.iso;
          return (
            <g
              key={node.iso}
              onMouseEnter={() => setHoverNode(node.iso)}
              onMouseLeave={() => setHoverNode(null)}
              style={{
                cursor: "pointer",
                opacity: hoverNode && !isHovered ? 0.3 : 1,
              }}
              className="node-group"
            >
              {/* Outer Score Ring */}
              <circle
                cx={node.x}
                cy={node.y}
                r={node.size + 3}
                fill="none"
                stroke={isHovered ? "#60a5fa" : "#1e293b"}
                strokeWidth={isHovered ? 2 : 1}
                strokeDasharray={isHovered ? "0" : "2 2"}
                className="transition-all duration-300"
              />

              {/* Inner Node */}
              <circle
                cx={node.x}
                cy={node.y}
                r={node.size}
                fill="#1e293b"
                stroke={isHovered ? "#fff" : "#3b82f6"}
                strokeWidth={2}
                filter={isHovered ? "url(#glow-blue)" : ""}
              />

              <text
                x={node.x}
                y={node.y}
                dy={3}
                textAnchor="middle"
                className="text-[9px] font-bold fill-white select-none pointer-events-none"
                style={{ textShadow: "0px 2px 4px rgba(0,0,0,0.8)" }}
              >
                {node.iso}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

// --- 2. MODERN OCCUPANCY MAP (Bar Chart) ---
export const ModernOccupancyMap = ({ data }: { data: any[] }) => {
  const sortedData = [...data].sort((a, b) => b.score - a.score);
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={sortedData}
        layout="vertical"
        margin={{ left: 10, right: 15, top: 10, bottom: 5 }}
      >
        <defs>
          <linearGradient id="barBlue" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.9} />
          </linearGradient>
          <linearGradient id="barDark" x1="0" y1="0" x2="1" y2="0">
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
          width={55} // <- more space
          tick={{ fill: "#94a3b8", fontSize: 11, fontWeight: 700 }}
          axisLine={false}
          tickLine={false}
        />

        {/* READABLE TOOLTIP FIX */}
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
          barSize={20}
          animationDuration={500}
        >
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={index < 2 ? "url(#barBlue)" : "url(#barDark)"}
              stroke={index < 2 ? "#60a5fa" : "none"}
              strokeWidth={1}
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

// --- SUB-COMPONENTS (Narrative & Delta) ---
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
      <span className="text-white bg-blue-600 px-1.5 rounded-sm animate-pulse">
        Regime Shift
      </span>
    )}
  </div>
);

const MarketNarrative = ({ snapshot }: { snapshot: Snapshot }) => (
  <div className="bg-slate-900 border-l-2 border-blue-500 p-4 rounded-r-lg shadow-sm">
    <div className="flex justify-between items-center mb-1">
      <div className="flex items-center gap-2">
        <Zap size={14} className="text-yellow-500" />
        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
          Macro Context
          <InfoTooltip
            title="Market Intelligence"
            content="A high-level synthesis of the active regime. This section translates mathematical network flows, liquidity leakage, and risk indices into a qualitative summary of current capital behavior and USD dominance."
          />
        </h4>
      </div>
      {!snapshot.posture.trust_ranking && (
        <div className="bg-rose-500/10 text-rose-400 text-[9px] font-black px-2 rounded border border-rose-500/30 flex items-center gap-1">
          <AlertTriangle size={10} /> LOW ACCURACY REGIME
        </div>
      )}
    </div>
    <p className="text-xs text-slate-300 font-medium leading-relaxed">
      "{snapshot.regime.desc}{" "}
      {snapshot.posture.usd_view === "Overweight"
        ? "USD acts as a sink."
        : "Flows favor peer rotation."}
      "
    </p>
    <WeeklyDeltaBadge delta={snapshot.delta} />
  </div>
);

// --- MAIN PAGE ---
export default function WalkForward() {
  const [data, setData] = useState<WalkForwardData | null>(null);
  const [idx, setIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [horizon, setHorizon] = useState<"short" | "medium" | "long">("medium");

  useEffect(() => {
    axios
      .get("http://localhost:8000/walkforward?weeks=52")
      .then((res) => {
        setData(res.data);
        setIdx(res.data.history.length - 1);
      })
      .catch((e) => console.error("Walkforward Error:", e));
  }, []);

  useEffect(() => {
    let interval: any;
    if (isPlaying && data) {
      interval = setInterval(() => {
        setIdx((prev) => (prev + 1) % data.history.length);
      }, 700);
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
      {/* 1. Timeline */}
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

      {/* 2. Narrative */}
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

      {/* 3. Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* NETWORK (Topology) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[280px] relative shadow-xl">
          <h3 className="text-[9px] font-black text-slate-500 uppercase tracking-widest absolute top-3 left-4 flex items-center gap-2 z-10">
            <GitCommit size={14} className="text-blue-500" /> Capital Flow
            Topology
            <InfoTooltip
              title="Plumbing Map"
              content="Blue: Peer rotation. Red: Safety leakage to USD. Blips: Strength of gravity."
            />
          </h3>
          <ModernCapitalTopology
            edges={snap.edges["medium"]}
            leakage={snap.usd_leakage}
            scores={scores}
          />
        </div>

        {/* Prediction Bar Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[350px] flex flex-col shadow-2xl">
          <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
            <TrendingUp size={14} className="text-emerald-500" /> Future
            Occupancy Map
            <InfoTooltip
              title="Structural Sinks"
              content="Where will capital 'settle' after 3 months? High bars are 'Sinks'—stable places where capital stays. Blue bars highlight the top structural winners."
            />
          </h3>
          <div className="flex-grow">
            <ModernOccupancyMap data={scores} />
          </div>
        </div>

        {/* SCATTER (Validation) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[280px] flex flex-col shadow-xl">
          <h3 className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
            <Activity size={14} className="text-purple-500" /> Prediction vs
            Outcome
            <InfoTooltip
              title="Report Card"
              content="A diagnostic plot correlating model conviction against market reality. The X-axis represents predicted structural strength, while the Y-axis shows the actual realized return. A positive diagonal distribution proves the model’s 'Capital Sinks' are successfully capturing market alpha."
            />
          </h3>
          <div className="flex-grow">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 10, right: 30, bottom: 20, left: -20 }}
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
                    value="Predicted SR Strength"
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

                {/* HUD Tooltip */}
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ payload }) => {
                    if (payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-slate-950 border border-slate-700 p-2 rounded-lg shadow-2xl text-[10px] font-sans">
                          <div className="font-black text-blue-400 border-b border-slate-800 pb-1 mb-1 uppercase tracking-tighter">
                            {d.iso} Snapshot
                          </div>
                          <div className="flex justify-between gap-4">
                            <span className="text-slate-500">
                              Predicted Score:
                            </span>
                            <span className="text-white font-mono">
                              {d.score.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span className="text-slate-500">
                              Realized Return:
                            </span>
                            <span
                              className={
                                d.ret >= 0
                                  ? "text-emerald-400"
                                  : "text-rose-400"
                              }
                            >
                              {d.ret >= 0 ? "+" : ""}
                              {d.ret.toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      );
                    }
                    return null;
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
                  {/* ADDED LABELS TO THE DOTS */}
                  <LabelList
                    dataKey="iso"
                    position="top"
                    fill="#94a3b8"
                    fontSize={10}
                    fontWeight="bold"
                    offset={8} // Pushes text slightly above the dot
                  />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
