import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  Grid, TrendingUp, RefreshCw, AlertCircle, UploadCloud,
  BarChart2, Info, ShieldAlert, ShieldCheck, ChevronRight,
  ChevronDown, Database, Zap, Target, Activity,
  AlertTriangle, CheckCircle, Layers, GitBranch,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell, PieChart, Pie, Legend,
} from 'recharts'
import {
  getSummary, getCorrelation, getClassDistribution,
} from '../services/api'
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 120_000 })

// ── Tab definitions ────────────────────────────────────────────────────────────
const TABS = [
  { id: 'overview',    label: 'Dataset Overview', icon: Database },
  { id: 'quality',     label: 'Data Quality',     icon: ShieldCheck },
  { id: 'target',      label: 'Target Analysis',  icon: Target },
  { id: 'correlation', label: 'Correlation',       icon: Grid },
  { id: 'actions',     label: 'Suggested Actions', icon: Zap },
]

const COLORS = ['#6366f1', '#34d399', '#f97316', '#f43f5e', '#06b6d4', '#8b5cf6', '#fb923c']

// ── Flag styles ────────────────────────────────────────────────────────────────
const FLAG_STYLE = {
  'Zero Variance':               'bg-danger-500/20 text-danger-300 border border-danger-500/30',
  'Near Constant':               'bg-warn-500/20 text-warn-300 border border-warn-500/30',
  'High Cardinality':            'bg-orange-500/20 text-orange-300 border border-orange-500/30',
  'Potential ID':                'bg-sky-500/20 text-sky-300 border border-sky-500/30',
  'High Risk Leakage':           'bg-danger-500/30 text-danger-200 border border-danger-400/50 font-semibold',
  'Very Strong Predictor (Check)': 'bg-amber-500/20 text-amber-300 border border-amber-400/40',
}

// ── Action level colours ───────────────────────────────────────────────────────
const LEVEL_STYLE = {
  critical: { card: 'border-danger-500/30 bg-danger-500/6',  badge: 'bg-danger-500/20 text-danger-300',  dot: 'bg-danger-500' },
  warn:     { card: 'border-warn-500/30 bg-warn-500/6',      badge: 'bg-warn-500/20 text-warn-300',      dot: 'bg-warn-500' },
  info:     { card: 'border-sky-500/20 bg-sky-500/6',        badge: 'bg-sky-500/20 text-sky-300',        dot: 'bg-sky-400' },
  good:     { card: 'border-accent-500/20 bg-accent-500/6',  badge: 'bg-accent-500/20 text-accent-300',  dot: 'bg-accent-500' },
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function Skeleton({ className = '' }) {
  return <div className={`animate-pulse rounded-lg bg-surface-700/50 ${className}`} />
}
function CardSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[0,1,2,3].map(i => <Skeleton key={i} className="h-24" />)}
      </div>
      <Skeleton className="h-52" />
    </div>
  )
}
function NoDataset() {
  return (
    <div className="card flex flex-col items-center justify-center py-16 gap-5 text-center">
      <div className="w-16 h-16 rounded-2xl bg-brand-600/20 flex items-center justify-center">
        <UploadCloud size={32} className="text-brand-400" />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-white mb-1">No Dataset Loaded</h3>
        <p className="text-slate-400 text-sm max-w-xs">Upload a CSV file and optionally run preprocessing before exploring your data here.</p>
      </div>
    </div>
  )
}
function ErrorBanner({ message, onRetry }) {
  return (
    <div className="card border border-danger-500/30 flex items-center justify-between gap-3 text-danger-400 py-3 px-4">
      <div className="flex items-center gap-2"><AlertCircle size={16} /><span className="text-sm">{message}</span></div>
      {onRetry && <button onClick={onRetry} className="text-xs text-brand-400 hover:text-brand-300 underline whitespace-nowrap">Retry</button>}
    </div>
  )
}

// ── Collapsible ───────────────────────────────────────────────────────────────
function Collapse({ title, icon: Icon, children, defaultOpen = true, badge }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card overflow-hidden">
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center justify-between text-left">
        <span className="flex items-center gap-2 font-semibold text-white text-sm">
          {Icon && <Icon size={15} className="text-brand-400" />}
          {title}
          {badge}
        </span>
        <ChevronDown size={16} className={`text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && <div className="mt-4">{children}</div>}
    </div>
  )
}

// ── ML Readiness Score Card ───────────────────────────────────────────────────
function ScoreGauge({ scoreObj }) {
  if (!scoreObj) return null
  const { score, breakdown = {} } = scoreObj
  const color  = score >= 80 ? '#34d399' : score >= 60 ? '#f97316' : '#f43f5e'
  const bg     = score >= 80 ? 'rgba(52,211,153,0.08)' : score >= 60 ? 'rgba(249,115,22,0.08)' : 'rgba(244,63,94,0.08)'
  const border = score >= 80 ? 'rgba(52,211,153,0.25)' : score >= 60 ? 'rgba(249,115,22,0.25)' : 'rgba(244,63,94,0.25)'
  const label  = score >= 80 ? 'ML-Ready' : score >= 60 ? 'Needs Work' : 'Issues Found'
  const pct    = Math.max(0, Math.min(100, score))

  const breakdownItems = [
    { key: 'leakage_penalty',     label: 'Leakage',      color: '#f43f5e' },
    { key: 'missing_penalty',     label: 'Missing',      color: '#f97316' },
    { key: 'duplicate_penalty',   label: 'Duplicates',   color: '#eab308' },
    { key: 'outlier_penalty',     label: 'Outliers',     color: '#8b5cf6' },
    { key: 'skewness_penalty',    label: 'Skewness',     color: '#06b6d4' },
    { key: 'cardinality_penalty', label: 'Cardinality',  color: '#fb923c' },
    { key: 'variance_penalty',    label: 'Zero Variance', color: '#f43f5e' },
  ]

  const maxPenalty = Math.max(...breakdownItems.map(b => breakdown[b.key] || 0), 1)
  const hasIssues  = breakdownItems.some(b => (breakdown[b.key] || 0) > 0)

  return (
    <div
      className="rounded-2xl px-5 py-4 flex items-center gap-5"
      style={{ background: bg, border: `1px solid ${border}`, minWidth: 320 }}
    >
      {/* Gauge */}
      <div className="relative shrink-0" style={{ width: 80, height: 80 }}>
        <svg viewBox="0 0 100 100" className="w-full h-full" style={{ transform: 'rotate(-90deg)' }}>
          {/* Track */}
          <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="10" />
          {/* Progress */}
          <circle cx="50" cy="50" r="42" fill="none"
            stroke={color} strokeWidth="10" strokeLinecap="round"
            strokeDasharray={`${(pct / 100) * 263.9} 263.9`}
            style={{ transition: 'stroke-dasharray 1.2s cubic-bezier(.4,0,.2,1)' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-black text-xl leading-none" style={{ color }}>{score}</span>
          <span className="text-[9px] text-slate-500 font-medium mt-0.5">/ 100</span>
        </div>
      </div>

      {/* Right side */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-bold text-white">Data Quality</span>
          <span className="text-[10px] px-2 py-0.5 rounded-full font-bold"
            style={{ background: `${color}22`, color, border: `1px solid ${color}44` }}>
            {label}
          </span>
        </div>

        {/* Penalty bars */}
        <div className="space-y-1.5 mt-2">
          {breakdownItems.map(({ key, label: lbl, color: c }) => {
            const val    = breakdown[key] || 0
            const barPct = val > 0 ? Math.max(8, Math.round((val / maxPenalty) * 100)) : 0
            return (
              <div key={key} className="flex items-center gap-2">
                <span className="text-[10px] text-slate-500 w-20 shrink-0 truncate">{lbl}</span>
                <div className="flex-1 h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
                  {val > 0 ? (
                    <div className="h-1.5 rounded-full transition-all duration-700"
                      style={{ width: `${barPct}%`, background: c, opacity: 0.85 }} />
                  ) : (
                    <div className="h-1.5 rounded-full" style={{ width: '100%', background: 'rgba(52,211,153,0.25)' }} />
                  )}
                </div>
                <span className="text-[10px] font-mono w-7 text-right shrink-0"
                  style={{ color: val > 0 ? c : '#34d399' }}>
                  {val > 0 ? `-${val}` : '+0'}
                </span>
              </div>
            )
          })}
        </div>

        {!hasIssues && (
          <p className="text-[10px] text-accent-400 mt-2">No deductions — clean dataset</p>
        )}
      </div>
    </div>
  )
}

// ── Quality tile ──────────────────────────────────────────────────────────────
function QTile({ label, value, sub, status }) {
  const s = { good: 'border-accent-500/30 bg-accent-500/8', warn: 'border-warn-500/30 bg-warn-500/8', danger: 'border-danger-500/30 bg-danger-500/8', neutral: 'border-surface-600 bg-surface-700/30' }
  const t = { good: 'text-accent-400', warn: 'text-warn-400', danger: 'text-danger-400', neutral: 'text-slate-300' }
  return (
    <div className={`rounded-xl border px-4 py-3 text-center ${s[status]}`}>
      <p className={`text-xl font-bold ${t[status]}`}>{value}</p>
      <p className="text-xs text-slate-400 mt-0.5">{label}</p>
      {sub && <p className="text-[10px] text-slate-500 mt-0.5">{sub}</p>}
    </div>
  )
}

// ── Tooltip wrapper ───────────────────────────────────────────────────────────
function Tip({ text, children }) {
  const [show, setShow] = useState(false)
  return (
    <span className="relative inline-block" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg bg-slate-800 border border-slate-700 text-xs text-slate-300 px-3 py-2 shadow-xl pointer-events-none">
          {text}
        </span>
      )}
    </span>
  )
}

// ── Correlation Heatmap (ORIGINAL — untouched) ────────────────────────────────
function CorrelationHeatmap({ data }) {
  if (!data) return null
  const { columns, matrix, min_val = 0, max_val = 0 } = data
  if (!columns.length || !matrix.length) {
    return <div className="flex flex-col items-center py-10 gap-3 text-slate-400"><Info size={24} /><p className="text-sm">No numeric columns found for correlation analysis.</p></div>
  }
  const absVals = matrix.flat().filter((v, idx) => { const i = Math.floor(idx / columns.length); const j = idx % columns.length; return i !== j && v !== null && v !== undefined })
  const maxAbs  = Math.max(...absVals.map(Math.abs), 0.001)
  const cellColor = (v, i, j) => {
    if (i === j) return 'rgba(99,102,241,0.9)'
    if (v === null || v === undefined) return 'rgba(30,41,59,0.6)'
    const intensity = 0.12 + 0.88 * (Math.abs(v) / maxAbs)
    return v > 0 ? `rgba(99,102,241,${intensity.toFixed(3)})` : `rgba(239,68,68,${intensity.toFixed(3)})`
  }
  const fmt = v => { if (v === null || v === undefined) return '–'; const r = parseFloat(v.toFixed(2)); return r === 0 ? '0.00' : r.toFixed(2) }
  const textColor = (v, i, j) => { if (i === j) return '#fff'; if (v === null) return '#64748b'; return Math.abs(v) / maxAbs > 0.5 ? '#fff' : '#94a3b8' }
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-xs text-slate-500 px-1">
        <span>Off-diagonal range: <span className="text-danger-400 font-medium">{fmt(min_val)}</span> to <span className="text-brand-400 font-medium">{fmt(max_val)}</span></span>
        <span className="italic">Colour intensity normalised to dataset max</span>
      </div>
      <div className="overflow-x-auto">
        <table className="mx-auto text-xs border-collapse">
          <thead><tr><th className="p-1.5 text-slate-500 min-w-[90px]" />{columns.map(c => <th key={c} className="p-1 text-slate-400 max-w-[52px]" title={c}><div style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', whiteSpace: 'nowrap', maxHeight: 80, overflow: 'hidden', fontSize: 10 }}>{c}</div></th>)}</tr></thead>
          <tbody>{matrix.map((row, i) => (<tr key={i}><td className="p-1.5 text-slate-300 text-right pr-3 font-medium truncate max-w-[90px] text-[11px]" title={columns[i]}>{columns[i]}</td>{row.map((val, j) => (<td key={j} title={`${columns[i]} ↔ ${columns[j]}: ${val !== null ? val.toFixed(4) : 'N/A'}`} className="p-0.5" style={{ width: 44, height: 44 }}><div className="w-9 h-9 rounded-md flex items-center justify-center font-bold transition-transform hover:scale-110 cursor-default" style={{ background: cellColor(val, i, j), color: textColor(val, i, j), fontSize: 9, letterSpacing: '-0.02em' }}>{fmt(val)}</div></td>))}</tr>))}</tbody>
        </table>
      </div>
      <div className="flex flex-col items-center gap-2 mt-2">
        <div className="w-64 h-4 rounded-full" style={{ background: 'linear-gradient(to right, rgb(239,68,68), rgba(30,41,59,0.3), rgb(99,102,241))' }} />
        <div className="flex justify-between w-64 text-[10px] text-slate-500"><span>Strong negative</span><span>No correlation</span><span>Strong positive</span></div>
      </div>
    </div>
  )
}

// ── Target Analysis (ORIGINAL ClassBalance — untouched logic) ─────────────────
function TargetAnalysisTab({ data, loading, error, onRetry }) {
  if (loading) return <CardSkeleton />
  if (error)   return <ErrorBanner message={error} onRetry={onRetry} />
  if (!data)   return null
  const { target_column, total_samples, num_classes, counts, percentages, majority_class, minority_class,
    majority_count, minority_count, minority_pct, imbalance_ratio, status, is_balanced, insight, recommendations } = data
  const labels  = Object.keys(counts)
  const barData = labels.map((lbl, i) => ({ label: lbl, count: counts[lbl], pct: percentages[lbl], fill: COLORS[i % COLORS.length] }))
  const pieData = labels.map((lbl, i) => ({ name: lbl, value: percentages[lbl], fill: COLORS[i % COLORS.length] }))
  const ratioBadge = imbalance_ratio >= 0.8 ? 'bg-accent-500/20 text-accent-300' : imbalance_ratio >= 0.4 ? 'bg-warn-500/20 text-warn-300' : 'bg-danger-500/20 text-danger-300'
  const statusStyle = is_balanced
    ? { bg: 'bg-accent-500/10', border: 'border-accent-500/30', text: 'text-accent-400', icon: <ShieldCheck size={20} /> }
    : { bg: 'bg-warn-500/10',   border: 'border-warn-500/30',   text: 'text-warn-400',   icon: <ShieldAlert size={20} /> }
  const severityMap = { well_balanced: { cls: 'bg-accent-500/20 text-accent-300', label: '✅ Well Balanced' }, slight: { cls: 'bg-sky-500/20 text-sky-300', label: '🔵 Slight Imbalance' }, moderate: { cls: 'bg-warn-500/20 text-warn-300', label: '⚠ Moderate Imbalance' }, severe: { cls: 'bg-danger-500/20 text-danger-300', label: '🔴 Severe Imbalance' } }
  return (
    <div className="space-y-5 animate-slide-up">
      <div className={`card border ${statusStyle.border} ${statusStyle.bg} flex items-start gap-3 py-4`}>
        <span className={`mt-0.5 ${statusStyle.text}`}>{statusStyle.icon}</span>
        <div className="flex-1">
          <div className="flex items-center gap-3 flex-wrap">
            <p className={`font-semibold text-base ${statusStyle.text}`}>{is_balanced ? '✅ Balanced Dataset' : '⚠️ Imbalanced Dataset'}</p>
            {data.severity && severityMap[data.severity] && <span className={`text-xs px-2 py-0.5 rounded-full ${severityMap[data.severity].cls}`}>{severityMap[data.severity].label}</span>}
          </div>
          <p className="text-slate-300 text-sm mt-1 leading-relaxed max-w-3xl">{insight}</p>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[{ label: 'Total Samples', value: total_samples.toLocaleString(), sub: `${num_classes} classes` }, { label: 'Minority Class', value: minority_class, sub: `${minority_count.toLocaleString()} samples` }, { label: 'Minority %', value: `${minority_pct.toFixed(1)}%`, sub: is_balanced ? 'Sufficient' : 'Too low' }, { label: 'Imbalance Ratio', value: imbalance_ratio.toFixed(3), sub: 'minority ÷ majority', badge: ratioBadge }].map(({ label, value, sub, badge }) => (
          <div key={label} className="card text-center py-5 flex flex-col items-center gap-1">
            <p className={`text-2xl font-bold gradient-text ${badge ? `px-2 py-0.5 rounded-full text-sm ${badge}` : ''}`}>{value}</p>
            <p className="text-xs text-slate-400 mt-0.5">{label}</p>
            <p className="text-[10px] text-slate-500">{sub}</p>
          </div>
        ))}
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <div className="card">
          <h3 className="font-semibold text-white mb-4 flex items-center gap-2 text-sm"><BarChart2 size={15} className="text-brand-400" /> Class Counts</h3>
          <ResponsiveContainer width="100%" height={220}><BarChart data={barData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}><CartesianGrid strokeDasharray="3 3" stroke="#334155" /><XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 12 }} /><YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} /><Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, color: '#e2e8f0' }} formatter={(v, n, p) => [`${v.toLocaleString()} (${p.payload.pct}%)`, 'Count']} /><Bar dataKey="count" radius={[6, 6, 0, 0]}>{barData.map((d, i) => <Cell key={i} fill={d.fill} />)}</Bar></BarChart></ResponsiveContainer>
        </div>
        <div className="card">
          <h3 className="font-semibold text-white mb-4 flex items-center gap-2 text-sm"><Grid size={15} className="text-brand-400" /> Class Distribution (%)</h3>
          <ResponsiveContainer width="100%" height={220}><PieChart><Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} innerRadius={40} paddingAngle={3} label={({ name, value }) => `${name}: ${value}%`} labelLine={{ stroke: '#64748b' }}>{pieData.map((d, i) => <Cell key={i} fill={d.fill} />)}</Pie><Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, color: '#e2e8f0' }} formatter={v => [`${v}%`, 'Share']} /><Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} iconType="circle" /></PieChart></ResponsiveContainer>
        </div>
      </div>
      <div className="card">
        <h3 className="font-semibold text-white mb-4 flex items-center gap-2 text-sm"><TrendingUp size={15} className="text-brand-400" /> Per-Class Breakdown</h3>
        <div className="table-wrapper">
          <table className="data-table">
            <thead><tr><th>Class</th><th>Count</th><th>Percentage</th><th>Role</th><th>Balance Bar</th></tr></thead>
            <tbody>{labels.map((lbl, i) => { const pct = percentages[lbl]; const role = lbl === majority_class ? 'Majority' : lbl === minority_class ? 'Minority' : 'Other'; const roleColor = role === 'Majority' ? 'badge-info' : role === 'Minority' ? 'bg-warn-500/20 text-warn-300' : 'badge'; return (<tr key={lbl}><td className="font-mono text-brand-300">{lbl}</td><td className="text-slate-200 font-medium">{counts[lbl].toLocaleString()}</td><td className="text-slate-200">{pct.toFixed(2)}%</td><td><span className={`badge ${roleColor}`}>{role}</span></td><td style={{ minWidth: 140 }}><div className="flex items-center gap-2"><div className="flex-1 bg-surface-700 rounded-full h-2"><div className="h-2 rounded-full transition-all" style={{ width: `${pct}%`, background: COLORS[i % COLORS.length] }} /></div><span className="text-xs text-slate-400 w-10 text-right">{pct.toFixed(1)}%</span></div></td></tr>) })}</tbody>
          </table>
        </div>
      </div>
      {!is_balanced && recommendations?.length > 0 && (
        <div className="card border border-warn-500/20">
          <h3 className="font-semibold text-warn-300 mb-4 flex items-center gap-2 text-sm"><ShieldAlert size={15} /> Recommended Actions <span className="text-[10px] font-normal text-slate-500 ml-1">(apply in the Class Imbalance step)</span></h3>
          <ul className="space-y-3">{recommendations.map((rec, i) => { const [title, ...rest] = rec.split(' – '); return (<li key={i} className="flex items-start gap-3 text-sm"><ChevronRight size={14} className="text-warn-400 mt-0.5 shrink-0" /><span><span className="font-semibold text-warn-300">{title}</span>{rest.length > 0 && <span className="text-slate-400"> – {rest.join(' – ')}</span>}</span></li>) })}</ul>
        </div>
      )}
      {is_balanced && <div className="card border border-accent-500/20 text-center py-5"><ShieldCheck size={28} className="text-accent-400 mx-auto mb-2" /><p className="text-accent-300 font-semibold">No resampling required</p><p className="text-slate-400 text-sm mt-1">Healthy class balance — proceed to preprocessing or feature engineering.</p></div>}
    </div>
  )
}

// ── Main EDA Page ─────────────────────────────────────────────────────────────
export default function EDA() {
  const [tab, setTab] = useState('overview')

  // Original state
  const [summary,     setSummary]     = useState(null)
  const [correlation, setCorrelation] = useState(null)
  const [classDist,   setClassDist]   = useState(null)
  const [summaryState,  setSummaryState]  = useState('idle')
  const [summaryError,  setSummaryError]  = useState('')
  const [corrLoading,   setCorrLoading]   = useState(false)
  const [corrError,     setCorrError]     = useState('')
  const [classLoading,  setClassLoading]  = useState(false)
  const [classError,    setClassError]    = useState('')

  // v2 state
  const [v2, setV2]           = useState(null)
  const [v2Loading, setV2L]   = useState(false)
  const [v2Error,   setV2Err] = useState('')

  const getMsg = e => e?.response?.data?.detail || e?.message || 'An unexpected error occurred.'

  const loadSummary = useCallback(async () => {
    setSummaryState('loading'); setSummaryError('')
    try {
      const res = await getSummary()
      setSummary(res.data)
      setSummaryState('ok')
    } catch (e) {
      e?.response?.status === 404 ? setSummaryState('no-data') : (setSummaryError(getMsg(e)), setSummaryState('error'))
    }
  }, [])

  const loadV2 = useCallback(async () => {
    setV2L(true); setV2Err('')
    try { const res = await api.get('/eda/v2/analysis'); setV2(res.data) }
    catch (e) { setV2Err(getMsg(e)) }
    finally { setV2L(false) }
  }, [])

  const loadCorrelation = useCallback(async () => {
    if (correlation) return
    setCorrLoading(true); setCorrError('')
    try { const res = await getCorrelation(); setCorrelation(res.data) }
    catch (e) { setCorrError(getMsg(e)) }
    finally { setCorrLoading(false) }
  }, [correlation])

  const loadClassDist = useCallback(async () => {
    if (classDist) return
    setClassLoading(true); setClassError('')
    try { const res = await getClassDistribution(); setClassDist(res.data) }
    catch (e) { setClassError(getMsg(e)) }
    finally { setClassLoading(false) }
  }, [classDist])

  useEffect(() => { loadSummary() }, [])
  useEffect(() => { if (summaryState === 'ok') loadV2() }, [summaryState])
  useEffect(() => { if (tab === 'correlation' && summaryState === 'ok') loadCorrelation() }, [tab, summaryState])
  useEffect(() => { if (tab === 'target'      && summaryState === 'ok') loadClassDist()  }, [tab, summaryState])

  const columns     = summary ? Object.keys(summary.dtypes || {}) : []
  const quality     = v2?.data_quality
  const diagnostics = v2?.feature_diagnostics || []
  const rels        = v2?.feature_relationships
  const highCorr    = v2?.correlation_analysis?.highly_correlated_pairs || []
  const actions     = v2?.suggested_actions || []
  const scoreObj    = v2?.data_quality_score
  const overview    = v2?.overview

  const flagCounts = useMemo(() => {
    const c = {}
    diagnostics.forEach(f => f.flags.forEach(flag => { c[flag] = (c[flag] || 0) + 1 }))
    return c
  }, [diagnostics])

  const criticalCount = (flagCounts['High Risk Leakage'] || 0) + (flagCounts['Zero Variance'] || 0)
  const actionCriticalCount = actions.filter(a => a.level === 'critical').length

  const refreshAll = () => {
    setCorrelation(null); setClassDist(null); setV2(null)
    setSummaryState('idle')
    loadSummary()
  }
  // load v2 when summary comes back
  useEffect(() => { if (summaryState === 'ok' && !v2 && !v2Loading) loadV2() }, [summaryState, v2, v2Loading])

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="section-title">Exploratory Data Analysis</h2>
          <p className="section-subtitle">ML readiness &amp; decision-support system</p>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={refreshAll} className="btn-secondary text-sm self-start" disabled={summaryState === 'loading' || v2Loading}>
            <RefreshCw size={14} className={(summaryState === 'loading' || v2Loading) ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </div>



      {summaryState === 'no-data' && <NoDataset />}

      {summaryState !== 'no-data' && (
        <>
          {/* Tab bar */}
          <div className="flex gap-1 bg-surface-800 border border-surface-700 p-1 rounded-xl flex-wrap">
            {TABS.map(({ id, label, icon: Icon }) => {
              const isActive = tab === id
              const badge = id === 'diagnostics' && criticalCount > 0
                ? <span className="ml-1 w-4 h-4 rounded-full bg-danger-500 text-white text-[8px] flex items-center justify-center">{criticalCount}</span>
                : id === 'actions' && actionCriticalCount > 0
                  ? <span className="ml-1 w-4 h-4 rounded-full bg-warn-500 text-white text-[8px] flex items-center justify-center">{actionCriticalCount}</span>
                  : null
              return (
                <button key={id} onClick={() => setTab(id)}
                  className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-all ${isActive ? 'bg-brand-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}>
                  <Icon size={13} />{label}{badge}
                </button>
              )
            })}
          </div>

          {summaryState === 'error' && <ErrorBanner message={summaryError} onRetry={loadSummary} />}

          {/* ══ DATASET OVERVIEW ══════════════════════════════════════════ */}
          {tab === 'overview' && (
            <div className="space-y-4 animate-slide-up">
              {summaryState === 'loading' && <CardSkeleton />}
              {summaryState === 'ok' && summary && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { label: 'Total Rows',  value: summary.shape.rows.toLocaleString(), icon: <BarChart2 size={18} className="text-brand-400" /> },
                      { label: 'Columns',     value: summary.shape.columns,               icon: <Grid size={18} className="text-brand-400" /> },
                      { label: 'Missing Cells', value: Object.values(summary.missing).reduce((a,b)=>a+b,0), icon: <AlertCircle size={18} className="text-warn-400" /> },
                      { label: 'Memory',      value: overview ? `${overview.memory_usage_mb} MB` : '—', icon: <Database size={18} className="text-purple-400" /> },
                    ].map(({ label, value, icon }) => (
                      <div key={label} className="card text-center py-5 flex flex-col items-center gap-2">
                        {icon}<p className="text-3xl font-bold gradient-text">{value}</p><p className="text-sm text-slate-400">{label}</p>
                      </div>
                    ))}
                  </div>
                  {overview && (
                    <div className="card">
                      <h3 className="font-semibold text-white mb-4 flex items-center gap-2 text-sm"><Layers size={15} className="text-brand-400" /> Feature Type Breakdown</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        {[{ label: 'Numerical', value: overview.feature_types.numerical, color: '#6366f1' }, { label: 'Categorical', value: overview.feature_types.categorical, color: '#34d399' }, { label: 'Boolean', value: overview.feature_types.boolean, color: '#f97316' }].map(({ label, value, color }) => (
                          <div key={label} className="flex flex-col items-center gap-1 py-3 rounded-xl" style={{ background: `${color}15`, border: `1px solid ${color}30` }}>
                            <span className="text-2xl font-bold" style={{ color }}>{value}</span>
                            <span className="text-xs text-slate-400">{label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="card">
                    <h3 className="font-semibold text-white mb-4 flex items-center gap-2 text-sm"><TrendingUp size={16} className="text-brand-400" /> Column Details</h3>
                    <div className="table-wrapper">
                      <table className="data-table">
                        <thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Missing</th></tr></thead>
                        <tbody>{columns.map(col => (<tr key={col}><td className="font-mono text-brand-300">{col}</td><td><span className="badge badge-info">{summary.dtypes[col]}</span></td><td className="text-slate-300">{summary.unique_counts[col]}</td><td className={summary.missing[col] > 0 ? 'text-warn-400 font-medium' : 'text-accent-400'}>{summary.missing[col]}</td></tr>))}</tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ══ DATA QUALITY ══════════════════════════════════════════════════ */}
          {tab === 'quality' && (
            <div className="space-y-4 animate-slide-up">
              {v2Loading && <CardSkeleton />}
              {v2Error && <ErrorBanner message={v2Error} onRetry={loadV2} />}
              {!v2Loading && quality && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <QTile label="Missing Values" value={`${quality.missing_values.missing_percentage}%`} sub={`${quality.missing_values.total_missing} cells`} status={quality.missing_values.missing_percentage > 10 ? 'danger' : quality.missing_values.missing_percentage > 0 ? 'warn' : 'good'} />
                    <QTile label="Duplicate Rows" value={`${quality.duplicates.percentage}%`} sub={`${quality.duplicates.count.toLocaleString()} rows`} status={quality.duplicates.percentage > 5 ? 'danger' : quality.duplicates.percentage > 0 ? 'warn' : 'good'} />
                    <QTile label="Outlier Features" value={quality.outliers.columns.length} sub="IQR method" status={quality.outliers.columns.length > 5 ? 'danger' : quality.outliers.columns.length > 0 ? 'warn' : 'good'} />
                    <QTile label="Skewed Features" value={quality.skewness.filter(s => s.severity === 'high').length} sub="|skew| > 2" status={quality.skewness.filter(s => s.severity === 'high').length > 3 ? 'warn' : 'good'} />
                  </div>

                  {quality.missing_values.columns.length > 0 ? (
                    <Collapse title={`Missing Values — ${quality.missing_values.columns.length} column(s)`} icon={AlertCircle} defaultOpen>
                      <div className="space-y-2">
                        {quality.missing_values.columns.map(col => (
                          <div key={col.name} className="flex items-center gap-3">
                            <span className="text-xs font-mono text-slate-300 w-36 truncate" title={col.name}>{col.name}</span>
                            <div className="flex-1 bg-surface-700 rounded-full h-2">
                              <div className="h-2 rounded-full transition-all" style={{ width: `${col.missing_percent}%`, background: col.missing_percent > 30 ? '#f43f5e' : col.missing_percent > 10 ? '#f97316' : '#eab308' }} />
                            </div>
                            <span className={`text-xs font-medium w-12 text-right ${col.missing_percent > 30 ? 'text-danger-400' : 'text-warn-400'}`}>{col.missing_percent.toFixed(1)}%</span>
                            {col.missing_percent > 30 && <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-danger-500/20 text-danger-300 border border-danger-500/30 whitespace-nowrap">Critical</span>}
                          </div>
                        ))}
                      </div>
                      <p className="text-[11px] text-slate-500 mt-3 border-t border-surface-700 pt-3">💡 Columns with &gt;30% missing: consider dropping. 5–30%: impute with median/mode. &lt;5%: simple imputation.</p>
                    </Collapse>
                  ) : (
                    <div className="card border border-accent-500/20 flex items-center gap-3 py-3 px-4"><CheckCircle size={16} className="text-accent-400" /><span className="text-sm text-accent-300 font-medium">No missing values — dataset is complete.</span></div>
                  )}

                  {quality.duplicates.count > 0 && (
                    <div className="card border border-warn-500/20 flex items-center gap-3 py-3 px-4">
                      <AlertTriangle size={16} className="text-warn-400 shrink-0" />
                      <div>
                        <p className="text-sm font-semibold text-warn-300">{quality.duplicates.count.toLocaleString()} duplicate rows ({quality.duplicates.percentage}%)</p>
                        <p className="text-xs text-slate-400 mt-0.5">Remove before training to prevent overfitting and inflated CV scores.</p>
                      </div>
                    </div>
                  )}

                  {quality.outliers.columns.length > 0 && (
                    <Collapse title={`Outlier Summary — ${quality.outliers.columns.length} feature(s)`} icon={Activity} defaultOpen={false}>
                      <div className="table-wrapper">
                        <table className="data-table">
                          <thead><tr><th>Feature</th><th>Outlier Count</th><th>Outlier %</th><th>Severity</th></tr></thead>
                          <tbody>{quality.outliers.columns.map(col => (<tr key={col.name}><td className="font-mono text-brand-300">{col.name}</td><td className="text-slate-200">{col.outlier_count.toLocaleString()}</td><td className={col.outlier_pct > 10 ? 'text-danger-400 font-medium' : 'text-warn-400'}>{col.outlier_pct.toFixed(1)}%</td><td><span className={`badge ${col.outlier_pct > 10 ? 'bg-danger-500/20 text-danger-300' : 'bg-warn-500/20 text-warn-300'}`}>{col.outlier_pct > 10 ? 'High' : 'Moderate'}</span></td></tr>))}</tbody>
                        </table>
                      </div>
                      <p className="text-[11px] text-slate-500 mt-3 border-t border-surface-700 pt-3">💡 Tree-based models are robust to outliers. For linear models, apply RobustScaler or IQR clipping.</p>
                    </Collapse>
                  )}

                  {quality.skewness.length > 0 ? (
                    <Collapse title={`Skewed Features — ${quality.skewness.length} feature(s) with |skew| > 1`} icon={TrendingUp} defaultOpen={false}>
                      <div className="space-y-2">
                        {quality.skewness.map(s => (
                          <div key={s.feature} className="flex items-center gap-3">
                            <Tip text={`${s.direction} — skewness measures asymmetry. Values >2 or <-2 significantly distort linear models.`}>
                              <span className="text-xs font-mono text-slate-300 w-36 truncate cursor-help underline decoration-dotted" title={s.feature}>{s.feature}</span>
                            </Tip>
                            <div className="flex-1 bg-surface-700 rounded-full h-2 overflow-hidden relative">
                              <div className="h-2 rounded-full absolute" style={{ width: `${Math.min(100, Math.abs(s.skew_value) * 20)}%`, background: s.severity === 'high' ? '#f43f5e' : '#f97316', left: s.skew_value > 0 ? '50%' : `${50 - Math.min(50, Math.abs(s.skew_value) * 10)}%` }} />
                              <div className="absolute left-1/2 top-0 w-px h-full bg-slate-600" />
                            </div>
                            <span className={`text-xs font-mono w-16 text-right ${s.severity === 'high' ? 'text-danger-400' : 'text-warn-400'}`}>{s.skew_value > 0 ? '+' : ''}{s.skew_value.toFixed(2)}</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded-full whitespace-nowrap ${s.severity === 'high' ? 'bg-danger-500/20 text-danger-300' : 'bg-warn-500/20 text-warn-300'}`}>{s.direction}</span>
                          </div>
                        ))}
                      </div>
                      <p className="text-[11px] text-slate-500 mt-4 border-t border-surface-700 pt-3">💡 Right-skewed → apply <code className="text-brand-300">log1p</code>. Left-skewed → try square transform. Skewness &lt;1 is acceptable.</p>
                    </Collapse>
                  ) : (
                    <div className="card border border-accent-500/20 flex items-center gap-3 py-3 px-4"><CheckCircle size={16} className="text-accent-400" /><span className="text-sm text-accent-300">No significantly skewed features detected (|skew| ≤ 1).</span></div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ══ TARGET ANALYSIS ══════════════════════════════════════════════ */}
          {tab === 'target' && <TargetAnalysisTab data={classDist} loading={classLoading} error={classError} onRetry={() => { setClassDist(null); loadClassDist() }} />}

          {/* ══ CORRELATION ════════════════════════════════════════════════ */}
          {tab === 'correlation' && (
            <div className="space-y-4 animate-slide-up">
              {highCorr.length > 0 && (
                <div className="flex items-center gap-3 rounded-xl border border-warn-500/30 bg-warn-500/8 px-4 py-3">
                  <AlertTriangle size={16} className="text-warn-400 shrink-0" />
                  <div className="flex-1"><p className="text-sm font-semibold text-warn-300">Multicollinearity Detected</p><p className="text-xs text-slate-400 mt-0.5">{highCorr.length} feature pair(s) with |r| &gt; 0.9. Consider removing one from each pair.</p></div>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-warn-500/20 text-warn-300 border border-warn-500/30 font-medium whitespace-nowrap">{highCorr.length} pair{highCorr.length > 1 ? 's' : ''}</span>
                </div>
              )}
              <div className="card">
                <h3 className="font-semibold text-white mb-5 flex items-center gap-2"><Grid size={16} className="text-brand-400" /> Pearson Correlation Matrix</h3>
                {corrLoading && <div className="flex items-center gap-3 text-brand-300 py-6"><div className="spinner" /><span>Computing correlation…</span></div>}
                {corrError   && <ErrorBanner message={corrError} onRetry={() => { setCorrelation(null); loadCorrelation() }} />}
                {!corrLoading && !corrError && <CorrelationHeatmap data={correlation} />}
              </div>
              {highCorr.length > 0 && (
                <Collapse title={`Highly Correlated Pairs (|r| > 0.9) — ${highCorr.length} pair(s)`} icon={GitBranch} defaultOpen>
                  <div className="table-wrapper">
                    <table className="data-table">
                      <thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th><th>Risk</th><th>Recommended Action</th></tr></thead>
                      <tbody>{highCorr.map((pair, i) => (<tr key={i}><td className="font-mono text-brand-300">{pair.feature_1}</td><td className="font-mono text-brand-300">{pair.feature_2}</td><td className={`font-mono font-bold ${Math.abs(pair.correlation) > 0.95 ? 'text-danger-400' : 'text-warn-400'}`}>{pair.correlation > 0 ? '+' : ''}{pair.correlation.toFixed(4)}</td><td><span className={`badge ${pair.risk === 'Critical' ? 'bg-danger-500/20 text-danger-300' : pair.risk === 'High' ? 'bg-warn-500/20 text-warn-300' : 'bg-orange-500/20 text-orange-300'}`}>{pair.risk}</span></td><td className="text-slate-400 text-xs">{pair.recommended_action}</td></tr>))}</tbody>
                    </table>
                  </div>
                </Collapse>
              )}
            </div>
          )}

          {/* ══ SUGGESTED ACTIONS ══════════════════════════════════════════ */}
          {tab === 'actions' && (
            <div className="space-y-3 animate-slide-up">
              {v2Loading && <CardSkeleton />}
              {v2Error && <ErrorBanner message={v2Error} onRetry={loadV2} />}
              {!v2Loading && actions.length > 0 && (
                <>
                  <div className="flex items-center gap-3 mb-1">
                    <Zap size={18} className="text-brand-400" />
                    <div>
                      <p className="font-semibold text-white text-sm">ML Readiness Recommendations</p>
                      <p className="text-xs text-slate-400">Sorted by priority · {actions.filter(a => a.level === 'critical').length} critical · {actions.filter(a => a.level === 'warn').length} warnings</p>
                    </div>
                  </div>
                  {actions.map((action, i) => {
                    const style = LEVEL_STYLE[action.level] || LEVEL_STYLE.info
                    return (
                      <div key={i} className={`rounded-xl border px-4 py-4 ${style.card}`}>
                        <div className="flex items-start gap-3">
                          <span className="text-lg shrink-0">{action.emoji}</span>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap mb-1">
                              <p className="font-semibold text-white text-sm">{action.title}</p>
                              <span className={`text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wide font-bold ${style.badge}`}>{action.level}</span>
                            </div>
                            <p className="text-xs text-slate-400 leading-relaxed mb-2">{action.detail}</p>
                            <div className="flex items-start gap-2 bg-surface-800/60 rounded-lg px-3 py-2">
                              <ChevronRight size={12} className="text-brand-400 mt-0.5 shrink-0" />
                              <p className="text-xs text-slate-300 leading-relaxed"><span className="text-brand-300 font-medium">Fix: </span>{action.fix}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                  <p className="text-[11px] text-slate-500 border-t border-surface-700 pt-3 mt-2">ℹ These recommendations are read-only. Apply fixes in the Preprocessing, Basic Cleaning, or Class Imbalance steps.</p>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
