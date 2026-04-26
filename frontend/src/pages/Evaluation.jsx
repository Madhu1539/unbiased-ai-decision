import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  TrendingUp, RefreshCw, AlertCircle, Target, Play,
  ArrowRight, Clock, Database, CheckCircle, Sliders,
  Zap, Info, ChevronDown, ChevronUp, BarChart2,
  ShieldCheck, AlertTriangle, Crosshair, Bug, Eye,
  Activity, Hash, Calendar, Cpu,
  XCircle, Search, Filter, ChevronLeft, ChevronRight,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Legend, ScatterChart, Scatter,
  ReferenceLine, ReferenceDot,
} from 'recharts'
import StatCard from '../components/StatCard'
import ConfusionMatrix from '../components/ConfusionMatrix'
import {
  getEvaluation, getLiveMetrics,
  getOptimalThreshold, applyThreshold, getErrorAnalysis,
} from '../services/api'

// ══════════════════════════════════════════════════════════════════════
//  Pure helpers (no metric computation — backend is sole source of truth)
// ══════════════════════════════════════════════════════════════════════
function isTimeoutError(err) {
  return err.code === 'ECONNABORTED' || err.message?.toLowerCase().includes('timeout')
}
function formatElapsed(ms) { return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s` }
function pct(v)  { return v != null ? `${(v * 100).toFixed(2)}%` : '—' }
function num(v)  { return v != null ? Number(v).toLocaleString() : '—' }

// ── Extreme-threshold UX messages ─────────────────────────────────────
function ExtremeThresholdBanner({ threshold }) {
  if (threshold <= 0) {
    return (
      <div id="extreme-threshold-all-positive"
        className="flex items-center gap-3 p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/30 animate-fade-in">
        <AlertTriangle size={15} className="text-indigo-400 shrink-0" />
        <div>
          <span className="text-indigo-300 font-semibold text-sm">All predictions are positive at this threshold. </span>
          <span className="text-slate-400 text-xs">Recall = 100%, Precision may be very low.</span>
        </div>
      </div>
    )
  }
  if (threshold >= 1) {
    return (
      <div id="extreme-threshold-no-positive"
        className="flex items-center gap-3 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 animate-fade-in">
        <AlertTriangle size={15} className="text-rose-400 shrink-0" />
        <div>
          <span className="text-rose-300 font-semibold text-sm">No positive predictions at this threshold. </span>
          <span className="text-slate-400 text-xs">Precision = undefined, Recall = 0%.</span>
        </div>
      </div>
    )
  }
  return null
}

// ── Skeleton metric card ──────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div className="card animate-pulse border-surface-700">
      <div className="h-3 bg-slate-700 rounded w-1/3 mb-3" />
      <div className="h-8 bg-slate-700/60 rounded w-1/2" />
    </div>
  )
}

// ── Metric badge ──────────────────────────────────────────────────────
function MetricBadge({ label, value, color = 'text-brand-400', highlight, loading }) {
  return (
    <div className={`flex flex-col items-center gap-1 px-4 py-3 rounded-xl bg-surface-800 border transition-all ${
      highlight ? 'border-brand-500/50 bg-brand-500/10' : 'border-surface-700'
    }`}>
      <span className="text-xs text-slate-500 font-medium">{label}</span>
      {loading
        ? <div className="h-6 w-16 bg-slate-700 rounded animate-pulse mt-1" />
        : <span className={`text-xl font-bold tabular-nums ${color}`}>{value}</span>
      }
    </div>
  )
}

// ── Loading panel ─────────────────────────────────────────────────────
function LoadingPanel({ label = 'Computing evaluation metrics…', elapsed }) {
  return (
    <div className="card border-brand-500/20 bg-brand-500/5 flex flex-col items-center gap-4 py-10">
      <div className="spinner w-10 h-10" />
      <p className="text-brand-300 font-medium text-lg">{label}</p>
      {elapsed > 5000 && (
        <div className="flex items-center gap-2 text-slate-500 text-xs">
          <Clock size={12} /><span>Elapsed: {formatElapsed(elapsed)}</span>
        </div>
      )}
    </div>
  )
}

// ── Reproducibility badge strip ───────────────────────────────────────
function ReproducibilityStrip({ modelId, datasetHash, timestamp, calibrated }) {
  if (!modelId || modelId === 'unknown') return null
  return (
    <div className="card border-surface-700 bg-surface-800/50">
      <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-1">
        <ShieldCheck size={11} /> Reproducibility Metadata
      </p>
      <div className="flex flex-wrap gap-x-6 gap-y-1">
        <div className="flex items-center gap-1.5 text-xs">
          <Cpu size={11} className="text-brand-400" />
          <span className="text-slate-500">Model ID:</span>
          <code className="text-slate-300 font-mono text-[11px]">{modelId}</code>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <Hash size={11} className="text-accent-400" />
          <span className="text-slate-500">Dataset Hash:</span>
          <code className="text-slate-300 font-mono text-[11px]">{datasetHash}</code>
        </div>
        {timestamp && (
          <div className="flex items-center gap-1.5 text-xs">
            <Calendar size={11} className="text-green-400" />
            <span className="text-slate-500">Trained:</span>
            <span className="text-slate-300 text-[11px]">{new Date(timestamp).toLocaleString()}</span>
          </div>
        )}
        {calibrated && (
          <div className="flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/25">
            <CheckCircle size={10} className="text-emerald-400" />
            <span className="text-emerald-300 font-medium">Calibrated Probabilities</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Strategy selector (for optimizer) ────────────────────────────────
function StrategyOption({ id, label, desc, icon: Icon, active, onClick }) {
  return (
    <button id={id} onClick={onClick}
      className={`flex-1 flex flex-col items-start gap-1 px-4 py-3 rounded-xl border text-left transition-all ${
        active
          ? 'border-brand-500 bg-brand-500/10 text-brand-300'
          : 'border-surface-600 bg-surface-800 text-slate-400 hover:border-slate-500'
      }`}>
      <div className="flex items-center gap-2 font-semibold text-sm"><Icon size={14} /> {label}</div>
      <p className="text-xs text-slate-500 leading-snug">{desc}</p>
    </button>
  )
}

// ── Explanation renderer ──────────────────────────────────────────────
function Explanation({ text }) {
  if (!text) return null
  const parts = text.split(/\*\*(.*?)\*\*/g)
  return (
    <p className="text-sm text-slate-300 leading-relaxed">
      {parts.map((seg, i) =>
        i % 2 === 1 ? <strong key={i} className="text-white font-semibold">{seg}</strong> : seg
      )}
    </p>
  )
}

// ── Threshold chart tooltip ───────────────────────────────────────────
function ThresholdTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface-800 border border-surface-600 rounded-xl p-3 shadow-xl text-xs">
      <p className="text-slate-400 mb-2 font-medium">Threshold = {Number(label).toFixed(3)}</p>
      {payload.map(p => (
        <div key={p.dataKey} className="flex items-center justify-between gap-6 mb-1">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="text-white font-semibold tabular-nums">{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

// ── Regression: Predicted vs Actual + Residual plots ─────────────────
function RegressionPlots({ avpData }) {
  // Validate, align lengths, strip NaN
  const clean = (avpData || [])
    .filter(d =>
      d.actual   != null && isFinite(d.actual) &&
      d.predicted != null && isFinite(d.predicted)
    )

  if (clean.length === 0) {
    return (
      <div className="card border-surface-700 text-center py-10 text-slate-500 text-sm">
        No prediction data available for visualisation.
      </div>
    )
  }

  // Build residual data: residual = actual - predicted
  const residualData = clean.map(d => ({
    predicted : d.predicted,
    residual  : d.actual - d.predicted,
  }))

  // Perfect-line endpoints (min/max of actual range)
  const actuals   = clean.map(d => d.actual)
  const minVal    = Math.min(...actuals)
  const maxVal    = Math.max(...actuals)
  const perfLine  = [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]

  // Color palette: Actual = blue, Predicted = green, Reference = red
  const COLOR_ACTUAL    = '#60a5fa'  // blue-400   — actual  (y_true)
  const COLOR_PREDICTED = '#4ade80'  // green-400  — predicted (y_pred)
  const COLOR_PERFECT   = '#f87171'  // red-400    — reference / diagonal

  const chartStyle = {
    contentStyle: { background: '#1e293b', border: '1px solid #334155', borderRadius: 12, color: '#e2e8f0' },
    formatter: (v, name) => [typeof v === 'number' ? v.toFixed(4) : v, name],
  }

  // Legend pill helper
  const Pill = ({ color, label }) => (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium" style={{ color }}>
      <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )

  return (
    <div className="space-y-5">

      {/* ── 1. Predicted vs Actual ── */}
      <div className="card" id="regression-pred-vs-actual">
        <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <Crosshair size={16} className="text-brand-400" />
            Predicted vs Actual
          </h3>
          {/* Legend: dot = prediction point, dashed line = perfect fit */}
          <div className="flex items-center gap-4">
            <Pill color={COLOR_PREDICTED} label="Prediction point (y_pred)" />
            <span className="inline-flex items-center gap-1.5 text-[11px] font-medium" style={{ color: COLOR_PERFECT }}>
              <span className="inline-block w-5 border-t-2 border-dashed" style={{ borderColor: COLOR_PERFECT }} />
              Perfect fit (y = x)
            </span>
          </div>
        </div>
        <p className="text-slate-500 text-xs mb-4">
          Each <span style={{ color: COLOR_PREDICTED }} className="font-semibold">green dot</span> = one sample plotted at (actual value, predicted value).
          Dots on the <span style={{ color: COLOR_PERFECT }} className="font-semibold">red dashed diagonal</span> = perfect prediction. Vertical spread = prediction error.
        </p>

        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="actual"
              type="number"
              name="Actual (y_true)"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{ value: 'Actual (y_true)', position: 'insideBottom', offset: -14, fill: '#94a3b8', fontSize: 12 }}
            />
            <YAxis
              dataKey="predicted"
              type="number"
              name="Predicted (y_pred)"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{ value: 'Predicted (y_pred)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }}
            />
            <Tooltip
              contentStyle={chartStyle.contentStyle}
              formatter={chartStyle.formatter}
              cursor={{ strokeDasharray: '3 3', stroke: '#475569' }}
            />
            {/* Perfect-fit diagonal y = x — rendered as a line-only Scatter */}
            <Scatter
              data={perfLine}
              dataKey="y"
              line={{ stroke: COLOR_PERFECT, strokeWidth: 2, strokeDasharray: '7 4' }}
              shape={() => null}
              name="Perfect fit"
              legendType="none"
            />
            {/* ONE dot per sample at (actual, predicted) — this is the correct chart */}
            <Scatter
              data={clean}
              dataKey="predicted"
              name="Predicted (y_pred)"
              fill={COLOR_PREDICTED}
              fillOpacity={0.75}
              r={3.5}
            />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-[10px] text-slate-600 mt-1 text-right">
          Red dashed = perfect fit (y = x) &nbsp;|&nbsp; {clean.length} samples plotted
        </p>
      </div>


      {/* ── 2. Residual Plot ── */}
      <div className="card" id="regression-residual-plot">
        <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <Activity size={16} className="text-accent-400" />
            Residual Plot
          </h3>
          {/* Color legend */}
          <div className="flex items-center gap-4">
            <Pill color={COLOR_PREDICTED} label="Prediction error (residual)" />
            <Pill color={COLOR_PERFECT}   label="Zero-error line" />
          </div>
        </div>
        <p className="text-slate-500 text-xs mb-4">
          Residual = Actual − Predicted. <span style={{ color: COLOR_PREDICTED }} className="font-semibold">Green dots</span> above zero = under-prediction; below zero = over-prediction.
          Random scatter around the <span style={{ color: COLOR_PERFECT }} className="font-semibold">red line</span> = well-fitted model; patterns = systematic bias.
        </p>

        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="predicted"
              type="number"
              name="Predicted"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{ value: 'Predicted (y_pred)', position: 'insideBottom', offset: -14, fill: '#94a3b8', fontSize: 12 }}
            />
            <YAxis
              dataKey="residual"
              type="number"
              name="Residual"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{ value: 'Residual (y_true − y_pred)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }}
            />
            <Tooltip
              contentStyle={chartStyle.contentStyle}
              formatter={chartStyle.formatter}
              cursor={{ strokeDasharray: '3 3', stroke: '#475569' }}
            />
            {/* Zero reference line — red */}
            <ReferenceLine y={0} stroke={COLOR_PERFECT} strokeWidth={2} strokeDasharray="7 4"
              label={{ value: 'y = 0  (no error)', position: 'right', fill: COLOR_PERFECT, fontSize: 10 }}
            />
            {/* Residual dots — green */}
            <Scatter
              data={residualData}
              dataKey="residual"
              name="Prediction error"
              fill={COLOR_PREDICTED}
              fillOpacity={0.75}
              r={3.5}
            />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-[10px] text-slate-600 mt-1 text-right">
          Red dashed = zero residual &nbsp;|&nbsp; {clean.length} samples plotted
        </p>

      </div>

    </div>
  )
}

// ── ROC chart with current threshold marker ───────────────────────────
function RocChart({ rocData, currentFPR, currentTPR, threshold }) {
  if (!rocData) return null
  const chartData = (rocData.fpr || []).map((fpr, i) => ({ fpr, tpr: rocData.tpr[i] }))
  const hasDot = currentFPR != null && currentTPR != null &&
    isFinite(currentFPR) && isFinite(currentTPR)

  return (
    <div className="card">
      <h3 className="font-semibold text-white mb-1 flex items-center gap-2">
        <TrendingUp size={16} className="text-brand-400" /> ROC Curve
        {rocData.auc != null && (
          <span className="badge badge-success ml-auto">AUC = {rocData.auc.toFixed(3)}</span>
        )}
      </h3>
      <p className="text-slate-500 text-xs mb-3">
        AUC uses predicted probabilities — it does not change with threshold.
        {hasDot && (
          <span className="text-amber-400 ml-2">
            ● marks operating point at threshold {threshold?.toFixed(2)}.
          </span>
        )}
      </p>
      <ResponsiveContainer width="100%" height={290}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]}
            label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -4, fill: '#94a3b8', fontSize: 12 }}
            tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis domain={[0, 1]}
            label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }}
            tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 12, color: '#e2e8f0' }}
            formatter={v => v?.toFixed?.(3)} />
          <Line type="monotone" dataKey="tpr" stroke="#6366f1" strokeWidth={2.5} dot={false} name="ROC" />
          <Line type="monotone" data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
            dataKey="tpr" stroke="#334155" strokeDasharray="4 4" strokeWidth={1} dot={false} name="Random" />
          {hasDot && (
            <ReferenceDot x={currentFPR} y={currentTPR} r={7}
              fill="#f59e0b" stroke="#fff" strokeWidth={2}
              label={{ value: `t=${threshold?.toFixed(2)}`, position: 'top', fill: '#f59e0b', fontSize: 10, fontWeight: 700 }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}


// ══════════════════════════════════════════════════════════════════════
//  LiveThresholdPanel  — API-only (backend = single source of truth)
// ══════════════════════════════════════════════════════════════════════
function LiveThresholdPanel({ initialThreshold = 0.5, debugMode, onMetricsUpdate, staticData }) {
  const [threshold, setThreshold]   = useState(Number(initialThreshold.toFixed(2)))
  const [metrics,   setMetrics]     = useState(staticData?.metrics || null)
  const [loading,   setLoading]     = useState(false)
  const [error,     setError]       = useState(null)
  const [meta,      setMeta]        = useState(null)   // versioning from API
  const [elapsed,   setElapsed]     = useState(0)

  const debounceRef = useRef(null)
  const abortRef    = useRef(null)
  const timerRef    = useRef(null)

  // Prefill with static evaluation data on mount
  useEffect(() => {
    if (staticData?.metrics) {
      setMetrics(staticData.metrics)
      setMeta({
        model_id          : staticData.model_id,
        dataset_hash      : staticData.dataset_hash,
        training_timestamp: staticData.training_timestamp,
        calibrated        : staticData.calibrated,
      })
    }
    return () => {
      clearTimeout(debounceRef.current)
      clearInterval(timerRef.current)
      abortRef.current?.abort()
    }
  }, [staticData])

  const fetchMetrics = useCallback(async (t, controller) => {
    setLoading(true)
    setError(null)
    setElapsed(0)
    const t0 = Date.now()
    timerRef.current = setInterval(() => setElapsed(Date.now() - t0), 300)

    try {
      const res = await getLiveMetrics(t, debugMode, controller.signal)
      if (controller.signal.aborted) return
      const d = res.data
      setMetrics(d?.metrics || null)
      setMeta({
        model_id          : d?.model_id,
        dataset_hash      : d?.dataset_hash,
        training_timestamp: d?.training_timestamp,
        calibrated        : d?.calibrated,
      })
      onMetricsUpdate?.(d?.metrics)
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(e.response?.data?.detail || 'Failed to fetch live metrics.')
      }
    } finally {
      clearInterval(timerRef.current)
      setLoading(false)
    }
  }, [debugMode, onMetricsUpdate])

  const handleSlider = (e) => {
    const t = parseFloat(e.target.value)
    setThreshold(t)
    // Cancel in-flight request and debounce the next one
    clearTimeout(debounceRef.current)
    abortRef.current?.abort()
    const controller  = new AbortController()
    abortRef.current  = controller
    debounceRef.current = setTimeout(() => fetchMetrics(t, controller), 250)
  }

  const m = metrics || {}
  const rocData = m.roc_curve || staticData?.metrics?.roc_curve

  // Compute current ROC operating point from TP/FP/TN/FN (backend returns these)
  const currentFPR = (m.fp != null && m.tn != null && (m.fp + m.tn) > 0)
    ? m.fp / (m.fp + m.tn) : null
  const currentTPR = (m.tp != null && m.fn != null && (m.tp + m.fn) > 0)
    ? m.tp / (m.tp + m.fn) : null

  const classDist = m.class_distribution || {}
  const threshWarns = m.threshold_warnings || []

  // Task check (passed from parent via staticData)
  if (staticData?.task_type && staticData.task_type !== 'classification') {
    return (
      <div className="card border-slate-700/50">
        <div className="flex items-center gap-3 text-slate-500 text-sm">
          <Info size={16} />
          Live threshold control is only available for binary classification models.
        </div>
      </div>
    )
  }

  return (
    <div className="card space-y-5" id="live-threshold-panel">
      {/* ── Header ── */}
      <div className="flex items-center gap-3 justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-brand-500/15">
            <Activity size={18} className="text-brand-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white text-base">Live Threshold Control</h3>
            <p className="text-slate-500 text-xs mt-0.5">
              Backend recomputes all metrics — authoritative results
            </p>
          </div>
        </div>
        {loading && (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="spinner w-3.5 h-3.5" />
            {elapsed > 1000 && <span>{formatElapsed(elapsed)}</span>}
          </div>
        )}
      </div>

      {/* ── Slider ── */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-400 font-medium">Classification Threshold</span>
          <span className="text-2xl font-extrabold tabular-nums text-transparent bg-clip-text"
            style={{ backgroundImage: 'linear-gradient(135deg, #6366f1, #a78bfa)' }}>
            {threshold.toFixed(2)}
          </span>
        </div>
        <input id="threshold-slider"
          type="range" min="0" max="1" step="0.01" value={threshold}
          onChange={handleSlider}
          disabled={loading}
          className="w-full h-2 rounded-full cursor-pointer appearance-none disabled:cursor-wait"
          style={{
            background: `linear-gradient(to right, #6366f1 ${threshold * 100}%, #1e293b ${threshold * 100}%)`,
            opacity: loading ? 0.7 : 1,
          }}
        />
        <div className="flex justify-between text-[10px] text-slate-600">
          <span>0.00 — All Positive</span>
          <span className="text-slate-500">Default: 0.50</span>
          <span>1.00 — All Negative</span>
        </div>
        {/* Preset buttons */}
        <div className="flex gap-2 flex-wrap">
          {[0.3, 0.4, 0.5, 0.6, 0.7].map(t => (
            <button key={t} onClick={() => handleSlider({ target: { value: t } })}
              disabled={loading}
              className={`text-xs px-3 py-1 rounded-lg border transition-all disabled:opacity-50 ${
                Math.abs(threshold - t) < 0.005
                  ? 'border-brand-500 bg-brand-500/20 text-brand-300'
                  : 'border-surface-600 text-slate-500 hover:border-slate-500'
              }`}>
              {t.toFixed(2)}
            </button>
          ))}
        </div>
      </div>

      {/* ── Extreme threshold banner ── */}
      <ExtremeThresholdBanner threshold={threshold} />

      {/* ── API error ── */}
      {error && (
        <div className="flex items-start gap-3 p-3 rounded-xl bg-danger-500/10 border border-danger-500/30">
          <AlertCircle size={14} className="text-danger-400 mt-0.5 shrink-0" />
          <p className="text-danger-300 text-xs">{error}</p>
        </div>
      )}

      {/* ── Threshold-behaviour warnings (from backend) ── */}
      {threshWarns.length > 0 && (
        <div className="space-y-1">
          {threshWarns.map((w, i) => (
            <div key={i} className="flex items-start gap-2 p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <AlertTriangle size={12} className="text-amber-400 mt-0.5 shrink-0" />
              <p className="text-amber-300 text-xs">{w}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Live metric cards — skeleton while loading ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricBadge label="Accuracy"  value={pct(m.accuracy)}  color="text-green-400"   loading={loading && !metrics} />
        <MetricBadge label="Precision" value={pct(m.precision)} color="text-accent-400"  loading={loading && !metrics} />
        <MetricBadge label="Recall"    value={pct(m.recall)}    color="text-warn-400"    loading={loading && !metrics} />
        <MetricBadge label="F1 Score"  value={pct(m.f1_score)}  color="text-brand-400"  highlight loading={loading && !metrics} />
      </div>

      {/* ── Loading overlay on TP/FP/TN/FN tiles ── */}
      {metrics && (
        <div className="space-y-4">
          {/* TP / FP / TN / FN */}
          <div className="grid grid-cols-4 gap-2">
            {[
              { k: 'TP', v: m.tp, color: 'text-indigo-300',  bg: 'bg-indigo-500/10',  border: 'border-indigo-500/30' },
              { k: 'TN', v: m.tn, color: 'text-emerald-300', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
              { k: 'FP', v: m.fp, color: 'text-rose-300',    bg: 'bg-rose-500/10',    border: 'border-rose-500/30' },
              { k: 'FN', v: m.fn, color: 'text-amber-300',   bg: 'bg-amber-500/10',   border: 'border-amber-500/30' },
            ].map(({ k, v, color, bg, border }) => (
              <div key={k} className={`flex flex-col items-center gap-0.5 py-2 rounded-xl border ${bg} ${border} ${loading ? 'opacity-50' : ''} transition-opacity`}>
                <span className={`text-[10px] font-bold ${color}`}>{k}</span>
                <span className="text-white font-bold text-base tabular-nums">
                  {loading ? '…' : num(v)}
                </span>
              </div>
            ))}
          </div>

          {/* ROC-AUC badge */}
          {m.roc_auc != null && (
            <div className="flex items-center gap-2 p-3 rounded-xl bg-surface-800 border border-surface-600">
              <TrendingUp size={14} className="text-brand-400" />
              <span className="text-xs text-slate-400">ROC-AUC</span>
              <span className="text-white font-bold tabular-nums ml-auto">{m.roc_auc.toFixed(4)}</span>
              <span className="text-xs text-slate-500">(probability-based, threshold-independent)</span>
            </div>
          )}

          {/* Confusion matrix */}
          {m.confusion_matrix && (
            <div>
              <p className="text-xs font-medium text-slate-400 mb-3 flex items-center gap-1">
                <Target size={12} /> Confusion Matrix @ {threshold.toFixed(2)}
              </p>
              <ConfusionMatrix
                matrix={m.confusion_matrix}
                labels={m.class_labels}
                tp={m.tp} fp={m.fp} tn={m.tn} fn={m.fn}
              />
            </div>
          )}

          {/* ROC chart with current threshold marker */}
          {rocData && (
            <RocChart
              rocData={rocData}
              currentFPR={currentFPR}
              currentTPR={currentTPR}
              threshold={threshold}
            />
          )}

          {/* Class distribution */}
          {Object.keys(classDist).length > 0 && (
            <div className="p-3 rounded-xl bg-surface-800 border border-surface-600">
              <p className="text-xs font-medium text-slate-400 mb-2 flex items-center gap-1">
                <Database size={12} /> Class Distribution (test set)
              </p>
              <div className="flex flex-wrap gap-3">
                {Object.entries(classDist).map(([cls, info]) => (
                  <div key={cls} className="flex items-center gap-2">
                    <span className="text-white text-xs font-semibold">{cls}:</span>
                    <span className="text-slate-400 text-xs">
                      {info.count?.toLocaleString()} ({info.pct?.toFixed(1)}%)
                    </span>
                  </div>
                ))}
              </div>
              {(() => {
                const pcts = Object.values(classDist).map(d => d.pct || 0)
                const minP = Math.min(...pcts)
                return minP < 20 ? (
                  <div className="flex items-center gap-2 mt-2 text-warn-400 text-xs">
                    <AlertTriangle size={12} />
                    <span>Imbalanced dataset — minority class is {minP.toFixed(1)}%. Consider recall-priority threshold.</span>
                  </div>
                ) : null
              })()}
            </div>
          )}

          {/* sklearn cross-verification */}
          {m.sklearn_f1 != null && (
            <details className="group">
              <summary className="cursor-pointer text-xs text-slate-500 flex items-center gap-1 select-none hover:text-slate-400">
                <Eye size={11} /> sklearn cross-verification (backend)
              </summary>
              <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2">
                {[
                  ['Accuracy',  m.sklearn_accuracy],
                  ['Precision', m.sklearn_precision],
                  ['Recall',    m.sklearn_recall],
                  ['F1',        m.sklearn_f1],
                ].map(([label, val]) => (
                  <div key={label} className="flex flex-col gap-0.5 p-2 rounded-lg bg-surface-800 border border-surface-700">
                    <span className="text-[10px] text-slate-500">{label} (sklearn)</span>
                    <span className="text-xs font-semibold text-slate-300 tabular-nums">{pct(val)}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  )
}


// ══════════════════════════════════════════════════════════════════════
//  ThresholdOptimizerPanel
// ══════════════════════════════════════════════════════════════════════
function ThresholdOptimizerPanel({ taskType }) {
  const [strategy,  setStrategy]  = useState('auto')
  const [result,    setResult]    = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [applying,  setApplying]  = useState(false)
  const [error,     setError]     = useState(null)
  const [success,   setSuccess]   = useState(null)
  const [collapsed, setCollapsed] = useState(false)
  const [appliedAt, setAppliedAt] = useState(null)

  if (taskType && taskType !== 'classification') return null

  const run = async () => {
    setLoading(true); setError(null); setSuccess(null); setResult(null)
    try { setResult((await getOptimalThreshold(strategy)).data) }
    catch (e) { setError(e.response?.data?.detail || 'Threshold optimisation failed.') }
    finally { setLoading(false) }
  }

  const handleApply = async () => {
    if (!result) return
    setApplying(true); setSuccess(null); setError(null)
    try {
      await applyThreshold(result.best_threshold, strategy)
      setAppliedAt(result.best_threshold)
      setSuccess(`✅ Threshold ${result.best_threshold.toFixed(4)} applied! Reload the Evaluation tab to see updated metrics.`)
    } catch (e) { setError(e.response?.data?.detail || 'Failed to apply threshold.') }
    finally { setApplying(false) }
  }

  const curve    = result?.threshold_curve || []
  const bestThr  = result?.best_threshold

  return (
    <div className="card space-y-5" id="threshold-optimizer-panel">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-accent-500/15"><Sliders size={18} className="text-accent-400" /></div>
          <div>
            <h3 className="font-semibold text-white text-base">Optimal Probability Threshold</h3>
            <p className="text-slate-500 text-xs mt-0.5">Auto-select the best cutoff instead of default 0.5</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {appliedAt != null && (
            <span className="badge badge-success text-xs flex items-center gap-1">
              <ShieldCheck size={11} /> Applied: {appliedAt.toFixed(4)}
            </span>
          )}
          <button onClick={() => setCollapsed(v => !v)} className="btn-ghost text-slate-400 p-1.5 rounded-lg">
            {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          <div>
            <p className="text-xs font-medium text-slate-400 mb-2">Optimisation Strategy</p>
            <div className="flex gap-2 flex-wrap">
              <StrategyOption id="strategy-auto"   label="Auto"           icon={Zap}          active={strategy==='auto'}             onClick={() => setStrategy('auto')}             desc="F1 for balanced; Recall-priority when imbalanced" />
              <StrategyOption id="strategy-f1"     label="F1 Maximise"    icon={BarChart2}     active={strategy==='f1'}               onClick={() => setStrategy('f1')}               desc="Choose threshold with highest F1-score" />
              <StrategyOption id="strategy-recall" label="Recall Priority" icon={AlertTriangle} active={strategy==='recall_priority'}  onClick={() => setStrategy('recall_priority')}  desc="Maximise recall (minimise false negatives)" />
            </div>
          </div>

          <button id="run-threshold-btn" onClick={run} disabled={loading}
            className="btn-primary w-full justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
            {loading ? <><div className="spinner w-4 h-4" /> Analysing 200 thresholds…</> : <><Crosshair size={16} /> Find Optimal Threshold</>}
          </button>

          {error   && <div className="flex items-start gap-3 p-4 rounded-xl bg-danger-500/10 border border-danger-500/30"><AlertCircle size={16} className="text-danger-400 mt-0.5 shrink-0" /><p className="text-danger-300 text-sm">{error}</p></div>}
          {success && <div className="flex items-start gap-3 p-4 rounded-xl bg-green-500/10 border border-green-500/30"><CheckCircle size={16} className="text-green-400 mt-0.5 shrink-0" /><p className="text-green-300 text-sm">{success}</p></div>}

          {result && !loading && (
            <div className="space-y-5 animate-fade-in">
              <div className="flex flex-wrap gap-2">
                <span className="badge badge-info text-xs">Strategy: {result.strategy_used === 'recall_priority' ? 'Recall Priority' : 'F1 Maximisation'}</span>
                {result.is_imbalanced && <span className="badge badge-warn text-xs flex items-center gap-1"><AlertTriangle size={10} /> Minority {result.minority_pct?.toFixed(1)}%</span>}
                <span className="badge badge-secondary text-xs">{result.n_thresholds_swept} thresholds</span>
              </div>

              <div className="flex flex-col items-center gap-2 py-4 rounded-2xl bg-gradient-to-br from-brand-500/15 to-accent-500/10 border border-brand-500/30">
                <p className="text-slate-400 text-xs font-medium uppercase tracking-widest">Optimal Threshold</p>
                <p className="text-5xl font-extrabold text-transparent bg-clip-text"
                  style={{ backgroundImage: 'linear-gradient(135deg, #6366f1, #a78bfa)' }}>
                  {bestThr?.toFixed(4)}
                </p>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricBadge label="Precision" value={pct(result.precision)} color="text-accent-400" />
                <MetricBadge label="Recall"    value={pct(result.recall)}    color="text-warn-400" />
                <MetricBadge label="F1-Score"  value={pct(result.f1_score)}  color="text-brand-400" highlight />
                <MetricBadge label="Accuracy"  value={pct(result.accuracy)}  color="text-green-400" />
              </div>

              {result.confusion_matrix && (
                <div>
                  <p className="text-xs font-medium text-slate-400 mb-2 flex items-center gap-1"><Target size={12} /> Confusion Matrix @ {bestThr?.toFixed(4)}</p>
                  <ConfusionMatrix matrix={result.confusion_matrix} labels={result.class_labels} />
                </div>
              )}

              {curve.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-slate-400 mb-3 flex items-center gap-1"><TrendingUp size={12} /> Precision · Recall · F1 vs Threshold</p>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={curve} margin={{ top: 8, right: 16, bottom: 24, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="threshold" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={v => v.toFixed(2)}
                        label={{ value: 'Threshold', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={v => v.toFixed(1)} />
                      <Tooltip content={<ThresholdTooltip />} />
                      <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, color: '#94a3b8', paddingBottom: 8 }} />
                      {bestThr != null && (
                        <ReferenceLine x={bestThr} stroke="#6366f1" strokeDasharray="6 3" strokeWidth={2}
                          label={{ value: `Best: ${bestThr.toFixed(3)}`, position: 'top', fill: '#818cf8', fontSize: 10 }} />
                      )}
                      <Line type="monotone" dataKey="precision" name="Precision" stroke="#22d3ee"  strokeWidth={2}   dot={false} />
                      <Line type="monotone" dataKey="recall"    name="Recall"    stroke="#f59e0b"  strokeWidth={2}   dot={false} />
                      <Line type="monotone" dataKey="f1"        name="F1"        stroke="#a78bfa"  strokeWidth={2.5} dot={false} />
                      <Line type="monotone" dataKey="accuracy"  name="Accuracy"  stroke="#34d399"  strokeWidth={1.5} strokeDasharray="4 3" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {result.explanation && (
                <div className="p-4 rounded-xl bg-surface-800 border border-surface-600">
                  <p className="text-xs font-semibold text-slate-400 mb-2 flex items-center gap-1"><Info size={12} /> Why this threshold?</p>
                  <Explanation text={result.explanation} />
                </div>
              )}

              <button id="apply-threshold-btn" onClick={handleApply} disabled={applying}
                className="btn-primary w-full justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', boxShadow: '0 0 20px rgba(99,102,241,0.3)' }}>
                {applying ? <><div className="spinner w-4 h-4" /> Applying…</> : <><ShieldCheck size={16} /> Apply Threshold {bestThr?.toFixed(4)} to Session</>}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}


// ══════════════════════════════════════════════════════════════════════
//  ErrorAnalysisSection  — lazy-loaded on demand
// ══════════════════════════════════════════════════════════════════════
function ErrorAnalysisSection() {
  const [data,      setData]      = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)
  const [filter,    setFilter]    = useState('all')   // 'all' | 'FP' | 'FN'
  const [search,    setSearch]    = useState('')
  const [sortDir,   setSortDir]   = useState('desc')  // 'desc' | 'asc'
  const [page,      setPage]      = useState(1)
  const [collapsed, setCollapsed] = useState(false)
  const PAGE_SIZE = 15

  const load = async () => {
    setLoading(true); setError(null)
    try {
      const res = await getErrorAnalysis(100)
      setData(res.data)
      setPage(1)
    } catch (e) {
      setError(e.response?.data?.detail || 'Error analysis failed.')
    } finally { setLoading(false) }
  }

  const s = data?.summary || {}
  const rows = (data?.error_analysis || [])
    .filter(r => filter === 'all' || r.type === filter)
    .filter(r => !search || String(r.sample_id).includes(search) ||
                 r.actual.includes(search) || r.predicted.includes(search))
    .sort((a, b) => sortDir === 'desc' ? b.confidence - a.confidence : a.confidence - b.confidence)

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const pageRows   = rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  const hasFeatures = s.has_feature_info

  const TYPE_STYLE = {
    FP: { bg: 'bg-rose-500/15',   border: 'border-rose-500/35',   text: 'text-rose-300',   label: 'False Positive' },
    FN: { bg: 'bg-amber-500/15',  border: 'border-amber-500/35',  text: 'text-amber-300',  label: 'False Negative' },
  }

  const ConfBar = ({ value }) => (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-surface-600 overflow-hidden">
        <div className="h-full rounded-full bg-brand-500/70 transition-all"
             style={{ width: `${(value * 100).toFixed(1)}%` }} />
      </div>
      <span className="text-[11px] font-mono text-slate-300 w-10 text-right tabular-nums">
        {value.toFixed(3)}
      </span>
    </div>
  )

  const DistBar = ({ value, label }) => (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1 rounded-full bg-surface-600 overflow-hidden">
        <div className="h-full rounded-full bg-slate-400/50"
             style={{ width: `${Math.min(100, value * 100 * 5).toFixed(1)}%` }} />
      </div>
      <span className="text-[10px] text-slate-500 tabular-nums">{value.toFixed(3)}</span>
    </div>
  )

  return (
    <div className="card space-y-4" id="error-analysis-section">
      {/* ── Header ── */}
      <div className="flex items-center gap-3 justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-rose-500/15">
            <XCircle size={18} className="text-rose-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white text-base">Error Analysis</h3>
            <p className="text-slate-500 text-xs mt-0.5">
              Misclassified samples — ranked by confidence of wrong prediction
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!data && !loading && (
            <button onClick={load}
              className="btn-primary text-sm flex items-center gap-1.5">
              <Search size={14} /> Analyse Errors
            </button>
          )}
          {data && (
            <button onClick={load} disabled={loading}
              className="btn-secondary text-xs flex items-center gap-1 disabled:opacity-50">
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              {loading ? 'Loading…' : 'Refresh'}
            </button>
          )}
          <button onClick={() => setCollapsed(v => !v)} className="btn-ghost text-slate-400 p-1.5 rounded-lg">
            {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          {/* Loading state */}
          {loading && !data && (
            <div className="flex items-center gap-3 py-8 justify-center text-slate-500">
              <div className="spinner w-5 h-5" />
              <span className="text-sm">Scanning {100} test samples…</span>
            </div>
          )}

          {/* API error */}
          {error && (
            <div className="flex items-start gap-3 p-3 rounded-xl bg-danger-500/10 border border-danger-500/30">
              <AlertCircle size={14} className="text-danger-400 mt-0.5 shrink-0" />
              <p className="text-danger-300 text-xs">{error}</p>
            </div>
          )}

          {/* Prompt before first load */}
          {!data && !loading && !error && (
            <div className="flex flex-col items-center gap-3 py-10 text-slate-500">
              <XCircle size={32} className="text-rose-400/40" />
              <p className="text-sm">Click <strong className="text-slate-300">Analyse Errors</strong> to inspect misclassified test samples.</p>
            </div>
          )}

          {/* Results */}
          {data && !loading && (
            <>
              {/* ── Summary strip ── */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: 'Total Errors',    value: s.total_errors,    color: 'text-rose-400',   bg: 'bg-rose-500/10',   border: 'border-rose-500/25' },
                  { label: 'False Positives', value: s.fp_count,        color: 'text-rose-300',   bg: 'bg-rose-500/8',    border: 'border-rose-500/20' },
                  { label: 'False Negatives', value: s.fn_count,        color: 'text-amber-300',  bg: 'bg-amber-500/8',   border: 'border-amber-500/20' },
                  { label: 'Avg Confidence',  value: `${(s.avg_confidence * 100).toFixed(1)}%`, color: 'text-slate-200', bg: 'bg-surface-700', border: 'border-surface-600' },
                ].map(({ label, value, color, bg, border }) => (
                  <div key={label} className={`flex flex-col gap-1 px-4 py-3 rounded-xl border ${bg} ${border}`}>
                    <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">{label}</span>
                    <span className={`text-2xl font-bold tabular-nums ${color}`}>{value}</span>
                  </div>
                ))}
              </div>

              {/* ── Threshold info ── */}
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Target size={11} />
                <span>Threshold used: <span className="text-slate-300 font-mono font-semibold">{s.threshold_used?.toFixed(3)}</span></span>
                <span className="mx-2 text-slate-700">·</span>
                <span>Showing top <span className="text-slate-300 font-semibold">{s.returned}</span> of <span className="text-slate-300 font-semibold">{s.total_errors}</span> errors</span>
                {!hasFeatures && (
                  <span className="ml-auto text-slate-600 italic">Feature importance unavailable for this model type</span>
                )}
              </div>

              {/* ── Filters + search ── */}
              <div className="flex items-center gap-3 flex-wrap">
                {/* Type filter tabs */}
                <div className="flex rounded-lg border border-surface-600 overflow-hidden text-xs">
                  {['all', 'FP', 'FN'].map(f => (
                    <button key={f} onClick={() => { setFilter(f); setPage(1) }}
                      className={`px-3 py-1.5 font-semibold transition-all ${
                        filter === f
                          ? f === 'FP' ? 'bg-rose-500/20 text-rose-300'
                          : f === 'FN' ? 'bg-amber-500/20 text-amber-300'
                          : 'bg-brand-500/20 text-brand-300'
                          : 'text-slate-500 hover:text-slate-300'
                      }`}>
                      {f === 'all' ? `All (${(data?.error_analysis||[]).length})` : f}
                    </button>
                  ))}
                </div>
                {/* Sort toggle */}
                <button onClick={() => setSortDir(d => d === 'desc' ? 'asc' : 'desc')}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-surface-600 text-slate-400 hover:border-slate-500 transition-all">
                  <Filter size={11} />
                  Confidence {sortDir === 'desc' ? '↓ High→Low' : '↑ Low→High'}
                </button>
                {/* Search */}
                <div className="flex items-center gap-2 flex-1 min-w-[160px] px-3 py-1.5 rounded-lg border border-surface-600 bg-surface-800">
                  <Search size={12} className="text-slate-500 shrink-0" />
                  <input
                    type="text" placeholder="Search sample ID or label…"
                    value={search} onChange={e => { setSearch(e.target.value); setPage(1) }}
                    className="bg-transparent text-xs text-slate-300 placeholder-slate-600 outline-none w-full"
                  />
                </div>
              </div>

              {/* ── Table ── */}
              {rows.length === 0 ? (
                <div className="text-center py-8 text-slate-500 text-sm">
                  {filter !== 'all' || search ? 'No errors match the current filter.' : '🎉 No misclassifications found!'}
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-surface-700">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-surface-700 bg-surface-800/60">
                        <th className="px-3 py-2.5 text-left text-slate-500 font-medium">ID</th>
                        <th className="px-3 py-2.5 text-left text-slate-500 font-medium">Type</th>
                        <th className="px-3 py-2.5 text-left text-slate-500 font-medium">Actual</th>
                        <th className="px-3 py-2.5 text-left text-slate-500 font-medium">Predicted</th>
                        <th className="px-3 py-2.5 text-left text-slate-500 font-medium w-40">Confidence</th>
                        <th className="px-3 py-2.5 text-left text-slate-500 font-medium">Δ Threshold</th>
                        {hasFeatures && (
                          <th className="px-3 py-2.5 text-left text-slate-500 font-medium">Top Features</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {pageRows.map((row, idx) => {
                        const ts = TYPE_STYLE[row.type] || TYPE_STYLE.FP
                        return (
                          <tr key={row.sample_id}
                            className={`border-b border-surface-700/50 transition-colors hover:bg-surface-700/30 ${
                              idx % 2 === 0 ? '' : 'bg-surface-800/20'
                            }`}>
                            {/* ID */}
                            <td className="px-3 py-2.5">
                              <code className="text-slate-400 font-mono">#{row.sample_id}</code>
                            </td>
                            {/* Type badge */}
                            <td className="px-3 py-2.5">
                              <span className={`inline-block text-[10px] font-bold px-2 py-0.5 rounded-full border ${ts.bg} ${ts.border} ${ts.text}`}>
                                {row.type}
                              </span>
                            </td>
                            {/* Actual */}
                            <td className="px-3 py-2.5">
                              <span className="text-emerald-300 font-semibold">{row.actual}</span>
                            </td>
                            {/* Predicted */}
                            <td className="px-3 py-2.5">
                              <span className="text-rose-300 font-semibold">{row.predicted}</span>
                            </td>
                            {/* Confidence bar */}
                            <td className="px-3 py-2.5 min-w-[120px]">
                              <ConfBar value={row.confidence} />
                            </td>
                            {/* Distance from threshold */}
                            <td className="px-3 py-2.5">
                              <DistBar value={row.distance} />
                            </td>
                            {/* Top features */}
                            {hasFeatures && (
                              <td className="px-3 py-2.5 min-w-[180px]">
                                {row.top_features?.length > 0 ? (
                                  <div className="space-y-1">
                                    {row.top_features.map(f => (
                                      <div key={f.feature} className="flex items-center gap-2">
                                        <span className="text-slate-400 truncate max-w-[90px]" title={f.feature}>
                                          {f.feature}
                                        </span>
                                        <div className="flex-1 h-1 rounded-full bg-surface-600 overflow-hidden">
                                          <div className="h-full rounded-full bg-brand-400/60"
                                               style={{ width: `${(f.impact * 100).toFixed(1)}%` }} />
                                        </div>
                                        <span className="text-[10px] text-slate-500 w-8 text-right tabular-nums">
                                          {f.impact.toFixed(2)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <span className="text-slate-600 italic">—</span>
                                )}
                              </td>
                            )}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* ── Pagination ── */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span>{rows.length} error{rows.length !== 1 ? 's' : ''} · page {page} of {totalPages}</span>
                  <div className="flex gap-1">
                    <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                      className="p-1.5 rounded-lg border border-surface-600 hover:border-slate-500 disabled:opacity-30 transition-all">
                      <ChevronLeft size={13} />
                    </button>
                    <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                      className="p-1.5 rounded-lg border border-surface-600 hover:border-slate-500 disabled:opacity-30 transition-all">
                      <ChevronRight size={13} />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  )
}


// ══════════════════════════════════════════════════════════════════════
//  Main Evaluation page
// ══════════════════════════════════════════════════════════════════════
export default function Evaluation({ onNavigate }) {
  const [data,        setData]        = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState(null)
  const [elapsed,     setElapsed]     = useState(0)
  const [debugMode,   setDebugMode]   = useState(false)
  const [liveMetrics, setLiveMetrics] = useState(null)

  const load = useCallback(async () => {
    setLoading(true); setError(null); setElapsed(0)
    const t0    = Date.now()
    const timer = setInterval(() => setElapsed(Date.now() - t0), 500)
    try {
      const res = await getEvaluation()
      setData(res.data)
      if (res.data?.metrics) setLiveMetrics(res.data.metrics)
    } catch (e) {
      setError(
        isTimeoutError(e)
          ? 'Request timed out. Please try again.'
          : e.response?.data?.detail || 'Failed to load evaluation.'
      )
    } finally { clearInterval(timer); setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  const m    = liveMetrics || data?.metrics || {}
  const avp  = data?.actual_vs_predicted || {}
  const task = data?.task_type
  const initialThreshold = data?.applied_threshold ?? 0.5

  const metricCards = task === 'classification' ? [
    { label: 'Accuracy',  value: m.accuracy  != null ? pct(m.accuracy)  : '—', color: 'text-green-400' },
    { label: 'Precision', value: m.precision != null ? pct(m.precision) : '—', color: 'text-accent-400' },
    { label: 'Recall',    value: m.recall    != null ? pct(m.recall)    : '—', color: 'text-warn-400' },
    { label: 'F1 Score',  value: m.f1_score  != null ? pct(m.f1_score)  : '—', color: 'text-brand-300' },
  ] : [
    { label: 'RMSE',     value: m.rmse     != null ? m.rmse.toFixed(4)     : '—', color: 'text-danger-400' },
    { label: 'MAE',      value: m.mae      != null ? m.mae.toFixed(4)      : '—', color: 'text-warn-400' },
    { label: 'R² Score', value: m.r2_score != null ? m.r2_score.toFixed(4) : '—', color: 'text-accent-400' },
    { label: 'MSE',      value: m.mse      != null ? m.mse.toFixed(4)      : '—', color: 'text-brand-300' },
  ]

  const avpData = (avp.actual || [])
    .map((a, i) => ({ actual: Number(a), predicted: Number(avp.predicted?.[i]) }))
    .filter(d => isFinite(d.actual) && isFinite(d.predicted))

  return (
    <div className="space-y-6 animate-fade-in">
      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="section-title">Model Evaluation</h2>
          <p className="section-subtitle">
            {data ? (
              <span className="flex items-center gap-2 flex-wrap">
                <CheckCircle size={14} className="text-green-400" />
                {data.model} — {task}
                {data.test_samples != null && (
                  <span className="flex items-center gap-1 text-slate-500 ml-2">
                    <Database size={12} /> {data.test_samples.toLocaleString()} test samples
                  </span>
                )}
                {data.applied_threshold != null && (
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-brand-500/20 text-brand-300 border border-brand-500/30">
                    Session threshold: {Number(data.applied_threshold).toFixed(2)}
                  </span>
                )}
              </span>
            ) : 'Train a model first to see results'}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button id="debug-mode-toggle" onClick={() => setDebugMode(v => !v)}
            title="Debug Mode: backend logs TP/FP/TN/FN to server console"
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all ${
              debugMode ? 'border-amber-500/50 bg-amber-500/10 text-amber-300' : 'border-surface-600 text-slate-500 hover:border-slate-500'
            }`}>
            <Bug size={13} /> {debugMode ? 'Debug ON' : 'Debug'}
          </button>
          <button onClick={load} disabled={loading}
            className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Debug banner */}
      {debugMode && (
        <div className="flex items-center gap-3 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30">
          <Bug size={14} className="text-amber-400 shrink-0" />
          <p className="text-amber-300 text-xs">
            <strong>Debug Mode ON</strong> — backend logs TP/FP/TN/FN + sample predictions to server console on each threshold query.
          </p>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="card border-danger-500/30 bg-danger-500/5">
          <div className="flex items-center gap-3 text-danger-400 mb-3">
            <AlertCircle size={20} /><span className="font-medium">Evaluation Error</span>
          </div>
          <pre
            style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '18rem' }}
            className="text-danger-300 text-xs bg-slate-900/60 rounded-lg p-3 overflow-auto mb-4"
          >{error}</pre>
          <div className="flex gap-3">
            <button onClick={load} className="btn-secondary text-sm"><RefreshCw size={14} /> Retry</button>
            {onNavigate && (
              <button onClick={() => onNavigate('training')} className="btn-primary inline-flex items-center gap-2">
                <Play size={16} /> Go to Training <ArrowRight size={16} />
              </button>
            )}
          </div>
        </div>
      )}

      {loading && <LoadingPanel elapsed={elapsed} />}
      {loading && !data && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => <SkeletonCard key={i} />)}
        </div>
      )}

      {data && !loading && (
        <>
          {/* Reproducibility strip */}
          <ReproducibilityStrip
            modelId={data.model_id}
            datasetHash={data.dataset_hash}
            timestamp={data.training_timestamp}
            calibrated={data.calibrated}
          />

          {/* Summary metric cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {metricCards.map(({ label, value, color }) => (
              <StatCard key={label} label={label} value={value} color={color} icon={<TrendingUp size={18} />} />
            ))}
          </div>

          {/* ── REGRESSION PLOTS (Predicted vs Actual + Residual) ── */}
          {task !== 'classification' && (
            <RegressionPlots avpData={avpData} />
          )}

          {/* ── LIVE THRESHOLD PANEL (API-only, backend = source of truth) ── */}
          {task === 'classification' && (
            <LiveThresholdPanel
              initialThreshold={initialThreshold}
              debugMode={debugMode}
              onMetricsUpdate={setLiveMetrics}
              staticData={data}
            />
          )}

          {/* ── THRESHOLD OPTIMIZER ── */}
          {task === 'classification' && <ThresholdOptimizerPanel taskType={task} />}

          {/* ── ERROR ANALYSIS ── */}
          {task === 'classification' && <ErrorAnalysisSection />}

        </>
      )}
    </div>
  )
}
