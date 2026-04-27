import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  ShieldAlert, ShieldCheck, RefreshCw, CheckCircle, AlertCircle,
  AlertTriangle, Info, Zap, Scale, ChevronRight, BarChart2,
  Eye, EyeOff, XCircle, Sparkles, Activity,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell, Legend,
} from 'recharts'
import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : '/api'
const api = axios.create({ baseURL: BASE, timeout: 120_000 })

// ─── Palette ──────────────────────────────────────────────────────────────────
// Strategies that require imblearn (mirrors STRATEGY_META on backend)
const IMBLEARN_STRATEGIES = new Set([
  'smote', 'smotenc', 'adasyn', 'undersample', 'smoteenn', 'smotetomek',
])

const CLASS_COLORS = [
  '#6366f1', '#34d399', '#f97316', '#f43f5e',
  '#06b6d4', '#8b5cf6', '#eab308', '#ec4899',
]

const SEVERITY_CONFIG = {
  balanced: {
    label:    'Balanced',
    badge:    'bg-emerald-500/15 border-emerald-500/30 text-emerald-300',
    icon:     ShieldCheck,
    iconCls:  'text-emerald-400',
    banner:   'border-emerald-500/25 bg-emerald-500/8',
    range:    '≥ 0.5',
    desc:     'Dataset is well-balanced. Resampling is not recommended.',
  },
  slight: {
    label:    'Slight Imbalance',
    badge:    'bg-amber-500/15 border-amber-500/30 text-amber-300',
    icon:     AlertTriangle,
    iconCls:  'text-amber-400',
    banner:   'border-amber-500/25 bg-amber-500/8',
    range:    '0.3 – 0.5',
    desc:     'Slight imbalance. Class weighting is usually sufficient.',
  },
  moderate: {
    label:    'Moderate Imbalance',
    badge:    'bg-orange-500/15 border-orange-500/30 text-orange-300',
    icon:     AlertTriangle,
    iconCls:  'text-orange-400',
    banner:   'border-orange-500/25 bg-orange-500/8',
    range:    '0.1 – 0.3',
    desc:     'Moderate imbalance. SMOTE or class weighting recommended.',
  },
  extreme: {
    label:    'Extreme Imbalance',
    badge:    'bg-red-500/15 border-red-500/30 text-red-300',
    icon:     XCircle,
    iconCls:  'text-red-400',
    banner:   'border-red-500/25 bg-red-500/8',
    range:    '< 0.1',
    desc:     'Extreme imbalance. Hybrid resampling (SMOTE+Tomek) strongly recommended.',
  },
}

const CATEGORY_COLORS = {
  none:           'text-slate-400 bg-slate-500/10 border-slate-500/20',
  no_resampling:  'text-blue-400 bg-blue-500/10 border-blue-500/20',
  oversampling:   'text-brand-400 bg-brand-500/10 border-brand-500/20',
  undersampling:  'text-amber-400 bg-amber-500/10 border-amber-500/20',
  hybrid:         'text-purple-400 bg-purple-500/10 border-purple-500/20',
}

const CATEGORY_LABELS = {
  none:          'No Action',
  no_resampling: 'No Resampling',
  oversampling:  'Oversampling',
  undersampling: 'Undersampling',
  hybrid:        'Hybrid',
}

// ─── Tiny helpers ─────────────────────────────────────────────────────────────
const clsx = (...a) => a.filter(Boolean).join(' ')

function Banner({ type = 'info', title, children, className = '' }) {
  const s = {
    info:    'bg-blue-500/8 border-blue-500/20 text-blue-300',
    warn:    'bg-amber-500/10 border-amber-500/25 text-amber-300',
    error:   'bg-red-500/10 border-red-500/25 text-red-400',
    success: 'bg-emerald-500/10 border-emerald-500/25 text-emerald-400',
    brand:   'bg-brand-500/8 border-brand-500/20 text-brand-300',
  }
  const icons = {
    info: <Info size={13} className="shrink-0 mt-0.5" />,
    warn: <AlertTriangle size={13} className="shrink-0 mt-0.5" />,
    error: <AlertCircle size={13} className="shrink-0 mt-0.5" />,
    success: <CheckCircle size={13} className="shrink-0 mt-0.5" />,
    brand: <Zap size={13} className="shrink-0 mt-0.5" />,
  }
  return (
    <div className={clsx(`flex gap-2.5 rounded-xl border px-4 py-3 text-xs leading-relaxed ${s[type]}`, className)}>
      {icons[type]}
      <div>
        {title && <p className="font-semibold mb-0.5">{title}</p>}
        {children}
      </div>
    </div>
  )
}

function Toggle({ checked, onChange, label, sub, disabled }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <p className={clsx('text-sm font-medium', disabled ? 'text-slate-500' : 'text-slate-200')}>{label}</p>
        {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      </div>
      <button
        onClick={() => !disabled && onChange(!checked)}
        disabled={disabled}
        className={clsx(
          'relative w-10 h-5 rounded-full transition-colors shrink-0',
          checked && !disabled ? 'bg-brand-600' : 'bg-surface-600',
          disabled && 'opacity-40 cursor-not-allowed'
        )}
      >
        <span className={clsx(
          'absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform',
          checked ? 'translate-x-5' : ''
        )} />
      </button>
    </div>
  )
}

function StatBox({ label, value, color = 'text-white', sub }) {
  return (
    <div className="card text-center py-3">
      <p className={clsx('text-xl font-bold tabular-nums', color)}>{value ?? '—'}</p>
      {sub && <p className="text-[10px] text-slate-500 mt-0.5">{sub}</p>}
      <p className="text-xs text-slate-400 mt-1">{label}</p>
    </div>
  )
}

// ─── Distribution bar chart ─────────────────────────────────────────────────
function DistributionChart({ dist, title, subtitle }) {
  if (!dist) return null
  const data = Object.entries(dist).map(([cls, d], i) => ({
    class: cls, count: d.count, pct: d.pct, fill: CLASS_COLORS[i % CLASS_COLORS.length],
  }))
  return (
    <div>
      <p className="text-xs font-semibold text-slate-400 mb-0.5">{title}</p>
      {subtitle && <p className="text-[10px] text-slate-600 mb-2">{subtitle}</p>}
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="class" tick={{ fill: '#94a3b8', fontSize: 10 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, fontSize: 11 }}
            formatter={(v, n, p) => [`${v.toLocaleString()} (${p.payload.pct}%)`, 'Count']}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Strategy card ───────────────────────────────────────────────────────────
function StrategyCard({ s, selected, onSelect, recommended }) {
  const active = selected === s.id
  return (
    <label className={clsx(
      'flex items-start gap-3 p-3.5 rounded-xl border cursor-pointer transition-all select-none',
      s.disabled && 'opacity-40 cursor-not-allowed',
      active && !s.disabled
        ? 'border-brand-500/60 bg-brand-600/15'
        : 'border-surface-600 hover:border-surface-500 hover:bg-surface-700/40',
    )}>
      <input
        type="radio" name="strategy" value={s.id}
        checked={active} disabled={s.disabled}
        onChange={() => !s.disabled && onSelect(s.id)}
        className="mt-0.5 accent-brand-500 shrink-0"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold text-white">{s.label}</span>
          {recommended && (
            <span className="text-[9px] font-bold tracking-wide bg-brand-600 text-white px-1.5 py-0.5 rounded-full">
              RECOMMENDED
            </span>
          )}
          <span className={clsx('text-[10px] border px-1.5 py-0.5 rounded-md font-medium', CATEGORY_COLORS[s.category])}>
            {CATEGORY_LABELS[s.category]}
          </span>
          {s.imblearn && (
            <span className="text-[10px] text-purple-400 bg-purple-500/10 border border-purple-500/20 px-1.5 py-0.5 rounded-md">
              imblearn
            </span>
          )}
        </div>
        <p className="text-xs text-slate-400 mt-1 leading-relaxed">{s.description}</p>
        {s.disabled && s.disable_reason && (
          <p className="text-[10px] text-red-400 mt-1">🚫 {s.disable_reason}</p>
        )}
      </div>
    </label>
  )
}

// ─── Encoding Detection Panel (Point 19 & 20) ─────────────────────────────
function EncodingDetectionPanel({ enc }) {
  if (!enc) return null
  const isOhe      = enc.is_ohe
  const hasIndices = enc.n_cat_features > 0
  const borderCls  = isOhe
    ? 'border-blue-500/20 bg-blue-500/8'
    : hasIndices
    ? 'border-purple-500/20 bg-purple-500/8'
    : 'border-surface-600 bg-surface-700/30'
  return (
    <div className={clsx('rounded-xl border px-4 py-3 space-y-2', borderCls)}>
      <p className="text-xs font-semibold text-slate-300 flex items-center gap-2">
        <Sparkles size={11} className="text-brand-400" />
        Encoding Detection
        <span className="text-[10px] text-slate-500 font-normal">— source: {enc.source || 'unknown'}</span>
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {[
          { label: 'Low-card enc',   value: enc.low_card_enc  || '—' },
          { label: 'High-card enc',  value: enc.high_card_enc || '—' },
          { label: 'SMOTE variant',  value: enc.smote_variant  || 'smote', em: true },
          { label: 'Cat indices',    value: enc.n_cat_features > 0 ? `${enc.n_cat_features} found` : 'none' },
        ].map(({ label, value, em }) => (
          <div key={label} className="bg-surface-800 rounded-lg px-2.5 py-2">
            <p className="text-[10px] text-slate-500">{label}</p>
            <p className={clsx('text-xs font-semibold mt-0.5', em ? 'text-brand-300' : 'text-slate-200')}>{value}</p>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-500 leading-relaxed">{enc.note}</p>
      {isOhe && (
        <p className="text-[10px] text-blue-400">
          ℹ OneHot/Frequency/Target encoding → <strong>SMOTE</strong> will be used.
          SMOTENC is not applicable after OHE (expanded binary columns ≠ categorical indices).
        </p>
      )}
      {!isOhe && hasIndices && (
        <p className="text-[10px] text-purple-400">
          ✓ Ordinal/Label encoding + {enc.n_cat_features} categorical indices found →
          requesting SMOTE will auto-promote to <strong>SMOTENC</strong>.
        </p>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
export default function ClassImbalance({ onNavigate }) {
  const [analysis,     setAnalysis]     = useState(null)
  const [preview,      setPreview]      = useState(null)
  const [statusData,   setStatusData]   = useState(null)

  const [loading,      setLoading]      = useState(true)
  const [previewing,   setPreviewing]   = useState(false)
  const [saving,       setSaving]       = useState(false)
  const [showPreview,  setShowPreview]  = useState(false)

  const [enabled,      setEnabled]      = useState(false)
  const [autoMode,     setAutoMode]     = useState(true)
  const [strategy,     setStrategy]     = useState('none')
  const [forceBalance, setForceBalance] = useState(false)
  const [forceLarge,   setForceLarge]   = useState(false)   // Point 21 override

  const [flash,        setFlash]        = useState(null)
  const [saved,        setSaved]        = useState(false)

  // ── Load data ─────────────────────────────────────────────────────
  const load = useCallback(async () => {
    setLoading(true); setFlash(null)
    try {
      const [anaR, stR] = await Promise.allSettled([
        api.get('/imbalance/analyze'),
        api.get('/imbalance/status'),
      ])

      if (anaR.status === 'fulfilled') {
        const d = anaR.value.data
        setAnalysis(d)
        // If already confirmed from a previous session, load those settings
        const stored = stR.status === 'fulfilled' ? stR.value.data : null
        if (stored?.confirmed) {
          const strat = stored.technique || 'none'
          setStrategy(strat)
          setEnabled(strat !== 'none')
          setSaved(true)
        } else if (d.is_applicable !== false) {
          // Auto-preset from recommendation
          const rec = d.recommendation
          if (rec) {
            setEnabled(rec.enable_balancing)
            setStrategy(rec.strategy)
          }
        }
      } else {
        setFlash({ type: 'error', msg: anaR.reason?.response?.data?.detail || 'Analysis failed.' })
      }
      if (stR.status === 'fulfilled') setStatusData(stR.value.data)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  // ── When autoMode toggles, reset strategy to recommendation ──────
  useEffect(() => {
    if (autoMode && analysis?.recommendation) {
      setStrategy(analysis.recommendation.strategy)
      setEnabled(analysis.recommendation.enable_balancing)
    }
  }, [autoMode, analysis])

  // ── Preview ───────────────────────────────────────────────────────
  const runPreview = async () => {
    if (!strategy || !enabled) return
    setPreviewing(true); setFlash(null); setPreview(null)
    try {
      const res = await api.post('/imbalance/preview', { strategy })
      setPreview(res.data)
      setShowPreview(true)
    } catch (e) {
      setFlash({ type: 'error', msg: e.response?.data?.detail || 'Preview failed.' })
    } finally {
      setPreviewing(false) }
  }

  // ── Confirm ───────────────────────────────────────────────────────
  const handleConfirm = async () => {
    setSaving(true); setFlash(null); setSaved(false)
    try {
      await api.post('/imbalance/confirm', {
        strategy:           enabled ? strategy : 'none',
        enabled,
        force_on_balanced:  forceBalance,
        force_large:        forceLarge,
      })
      setSaved(true)
      const stR = await api.get('/imbalance/status')
      setStatusData(stR.data)
      const stratLabel = analysis?.available_strategies?.find(s => s.id === strategy)?.label || strategy
      setFlash({ type: 'success', msg: `Strategy confirmed: ${enabled ? stratLabel : 'No Balancing'}. Training will apply it automatically.` })
    } catch (e) {
      const detail = e.response?.data?.detail || ''
      if (detail.startsWith('STRATEGY_BLOCKED')) {
        setFlash({ type: 'error', msg: detail.replace('STRATEGY_BLOCKED: ', '') })
      } else {
        setFlash({ type: 'error', msg: detail || 'Failed to save.' })
      }
    } finally { setSaving(false) }
  }

  // ── Derived state ────────────────────────────────────────────────
  const effectiveStrategy = enabled ? strategy : 'none'
  const sev      = analysis?.severity
  const sevMeta  = SEVERITY_CONFIG[sev] || SEVERITY_CONFIG.balanced
  const SevIcon  = sevMeta.icon
  const isBalanced = sev === 'balanced'
  const isLarge    = analysis?.large_dataset || false
  const enc        = analysis?.encoding_detection || null
  const priority   = analysis?.priority_selection || null

  // Warnings (Points 19-25)
  const warnings = useMemo(() => {
    if (!analysis) return []
    const w = []
    if (isBalanced && enabled && !forceBalance)
      w.push({ type: 'warn', msg: '⚠ Dataset is already balanced. Applying resampling may degrade model performance.' })
    if (analysis.smote_blocked && ['smote','adasyn','smoteenn','smotetomek','smotenc'].includes(strategy) && enabled)
      w.push({ type: 'error', msg: `🚫 ${analysis.smote_block_reason}` })
    // Point 24: Hard error — SMOTENC + OHE
    if (strategy === 'smotenc' && enc?.is_ohe && enabled)
      w.push({ type: 'error', msg: '🚫 SMOTENC cannot be used with OneHot encoding. OHE expands categoricals to binary dummies — use SMOTE instead.' })
    // Point 21: Large dataset block
    if (isLarge && ['smote','adasyn','smoteenn','smotetomek','smotenc'].includes(strategy) && enabled && !forceLarge)
      w.push({ type: 'error', msg: `🚫 ${analysis.large_dataset_warning} Enable "Force resampling" to override.` })
    if (isLarge && forceLarge && enabled)
      w.push({ type: 'warn', msg: `⚠ Forced resampling on ${analysis.n_train?.toLocaleString()} rows — may be very slow and memory-intensive.` })
    // Point 22: High-dim safety
    if (analysis.high_dim_vs_samples && ['smote','adasyn'].includes(strategy) && enabled)
      w.push({ type: 'error', msg: `🚫 n_features (${analysis.n_features}) > n_train (${analysis.n_train}) — SMOTE creates meaningless samples in sparse space. Use class_weight.` })
    if (analysis.high_dim_warning && ['smote','adasyn'].includes(strategy) && enabled)
      w.push({ type: 'warn', msg: `⚠ High-dimensional data (${analysis.n_features} features) — SMOTE may produce noisy synthetic samples.` })
    if (!analysis.imblearn_available && IMBLEARN_STRATEGIES.has(strategy) && enabled)
      w.push({ type: 'error', msg: '🚫 imbalanced-learn is not installed. Run: pip install imbalanced-learn' })
    if (strategy === 'undersample' && enabled)
      w.push({ type: 'warn', msg: 'ℹ Undersampling removes majority rows — reduces total training data.' })
    if (strategy === 'class_weight')
      w.push({ type: 'info', msg: 'ℹ class_weight="balanced" is set on the model — no data is created or removed.' })
    if (enabled && ['smote','adasyn','smoteenn','smotetomek','smotenc'].includes(strategy))
      w.push({ type: 'info', msg: 'ℹ Resampling runs inside imblearn.Pipeline.fit() only — skipped automatically at predict()/inference.' })
    return w
  }, [analysis, strategy, enabled, isBalanced, forceBalance, forceLarge, enc])

  // ── Loading ───────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="space-y-5 animate-fade-in">
        <div><h2 className="section-title">Class Imbalance</h2></div>
        <div className="card flex items-center gap-3 text-brand-300">
          <div className="spinner" /> Analysing y_train distribution…
        </div>
      </div>
    )
  }

  // ── No split gate ─────────────────────────────────────────────────
  if (flash?.msg?.includes('Split Data')) {
    return (
      <div className="space-y-5 animate-fade-in">
        <div><h2 className="section-title">Class Imbalance</h2></div>
        <div className="card border-amber-500/30 bg-amber-500/8 flex items-center gap-3">
          <AlertCircle size={18} className="text-amber-400 shrink-0" />
          <div>
            <p className="font-semibold text-white text-sm">Split Data step required</p>
            <p className="text-xs text-slate-400 mt-0.5">Complete the Split Data step first to access y_train labels.</p>
          </div>
        </div>
      </div>
    )
  }

  // ── Regression N/A ────────────────────────────────────────────────
  if (analysis && analysis.is_applicable === false) {
    return (
      <div className="space-y-5 animate-fade-in">
        <div><h2 className="section-title">Class Imbalance</h2></div>
        <div className="card border-slate-600 flex items-center gap-3 py-4 px-5">
          <Info size={18} className="text-slate-400 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-white">Not applicable for regression</p>
            <p className="text-xs text-slate-400 mt-0.5">Class imbalance handling applies to classification tasks only.</p>
          </div>
          <button onClick={() => onNavigate?.('models')} className="btn-primary text-sm flex items-center gap-1.5 shrink-0">
            Model Selection <ChevronRight size={13} />
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-5 animate-fade-in">

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="section-title">Class Imbalance</h2>
          <p className="section-subtitle">y_train analysis · leakage-safe · imblearn.Pipeline integration</p>
        </div>
        <button onClick={load} className="btn-secondary text-sm flex items-center gap-2 shrink-0">
          <RefreshCw size={13} /> Re-analyse
        </button>
      </div>

      {/* ── Leakage rule ── */}
      <div className="flex items-start gap-3 rounded-xl border bg-brand-500/8 border-brand-500/20 px-4 py-3">
        <Scale size={13} className="text-brand-400 mt-0.5 shrink-0" />
        <div className="text-xs text-brand-300/80 space-y-0.5">
          <p className="font-semibold text-brand-200 text-sm">Pipeline Integration</p>
          <p>Resampling lives inside <code>imblearn.Pipeline([('preprocess',…), ('resample', handler), ('model', clf)])</code></p>
          <p>fit() applies resampling on X_train/y_train only · predict() skips resampling automatically · X_test is <strong>never modified</strong></p>
        </div>
      </div>

      {/* ── Flash ── */}
      {flash && (
        <div className={clsx('flex items-center gap-3 rounded-xl border px-4 py-3 text-sm',
          flash.type === 'success' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
          : flash.type === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-400'
          : 'bg-amber-500/10 border-amber-500/25 text-amber-300'
        )}>
          {flash.type === 'success' ? <CheckCircle size={14} className="shrink-0" />
          : flash.type === 'error' ? <AlertCircle size={14} className="shrink-0" />
          : <AlertTriangle size={14} className="shrink-0" />}
          {flash.msg}
        </div>
      )}

      {/* ── Severity banner ── */}
      {analysis && sev && (
        <div className={clsx('card border flex items-start gap-3 py-3 px-4', sevMeta.banner)}>
          <SevIcon size={18} className={clsx('mt-0.5 shrink-0', sevMeta.iconCls)} />
          <div className="flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <p className="font-semibold text-white text-sm">{sevMeta.label}</p>
              <span className={clsx('text-[10px] font-semibold border px-2 py-0.5 rounded-md', sevMeta.badge)}>
                ratio {sevMeta.range}
              </span>
              {statusData?.confirmed && (
                <span className="text-[10px] font-semibold border px-2 py-0.5 rounded-md bg-brand-500/10 border-brand-500/20 text-brand-300">
                  ✓ Confirmed: {statusData.label}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">{sevMeta.desc}</p>
          </div>
          <div className="text-right shrink-0">
            <p className="text-lg font-bold text-white tabular-nums">{analysis.minority_ratio?.toFixed(3)}</p>
            <p className="text-[10px] text-slate-500">minority ratio</p>
          </div>
        </div>
      )}

      {/* ── Stats ── */}
      {analysis && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatBox label="Train Samples"    value={analysis.n_train?.toLocaleString()} color="text-brand-400" />
          <StatBox label="Classes"          value={analysis.n_classes} color="text-slate-300" />
          <StatBox label="Minority Count"   value={analysis.minority_count?.toLocaleString()} color={analysis.smote_blocked ? 'text-red-400' : 'text-amber-400'} sub={`class '${analysis.minority_class || ''}'`} />
          <StatBox label="Majority Count"   value={analysis.majority_count?.toLocaleString()} color="text-blue-400" sub={`class '${analysis.majority_class || ''}'`} />
        </div>
      )}

      {/* ── Point 19-20: Encoding Detection Panel ── */}
      {enc && <EncodingDetectionPanel enc={enc} />}

      {/* ── Point 21: Large dataset warning ── */}
      {isLarge && (
        <Banner type="warn" title="Large Dataset Detected">
          {analysis.large_dataset_warning}
          {' '}SMOTE/ADASYN disabled by default. Use Class Weighting or enable "Force resampling" (not recommended).
        </Banner>
      )}

      {/* ── Distribution chart + class table ── */}
      {analysis?.distribution && (
        <div className="grid md:grid-cols-2 gap-4">
          <div className="card">
            <DistributionChart
              dist={analysis.distribution}
              title="Training Class Distribution (y_train)"
              subtitle="Test labels are never inspected"
            />
          </div>
          <div className="card">
            <p className="text-xs font-semibold text-slate-400 mb-3 flex items-center gap-2">
              <BarChart2 size={12} className="text-brand-400" /> Class Breakdown
            </p>
            <div className="space-y-2">
              {Object.entries(analysis.distribution).map(([cls, d], i) => (
                <div key={cls} className="flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: CLASS_COLORS[i % CLASS_COLORS.length] }} />
                  <span className="font-mono text-sm text-slate-300 flex-1">{cls}</span>
                  <div className="flex-1 bg-surface-700 rounded-full h-1.5 overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${d.pct}%`, background: CLASS_COLORS[i % CLASS_COLORS.length] }} />
                  </div>
                  <span className="text-sm font-bold text-white tabular-nums w-14 text-right">{d.count.toLocaleString()}</span>
                  <span className="text-xs text-slate-500 w-10 text-right tabular-nums">{d.pct}%</span>
                </div>
              ))}
            </div>
            {/* Severity thresholds legend */}
            <div className="mt-4 pt-3 border-t border-surface-700 grid grid-cols-2 gap-1.5">
              {Object.entries(SEVERITY_CONFIG).map(([k, v]) => (
                <div key={k} className={clsx('flex items-center gap-2 px-2 py-1 rounded-lg border text-[10px]', k === sev ? v.badge : 'border-surface-700 text-slate-600')}>
                  <span className="font-medium">{v.range}</span>
                  <span>{v.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── SMOTE blocked warning ── */}
      {analysis?.smote_blocked && (
        <Banner type="error" title="SMOTE Unavailable">
          {analysis.smote_block_reason} — synthetic generation requires KNN (k≥5 neighbours).
          Use <strong>Class Weighting</strong> or collect more minority class data.
        </Banner>
      )}

      {/* ── imblearn not installed ── */}
      {analysis && !analysis.imblearn_available && (
        <Banner type="warn" title="imbalanced-learn not installed">
          Run <code className="bg-surface-800 px-1 rounded">pip install imbalanced-learn</code> to enable SMOTE, ADASYN, and hybrid strategies.
          Class Weighting is always available.
        </Banner>
      )}

      {/* ══════════════════════════════════════════════
          CONFIGURATION PANEL
      ══════════════════════════════════════════════ */}
      {analysis?.is_applicable !== false && (
        <div className="card space-y-5">
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <Activity size={14} className="text-brand-400" /> Balancing Configuration
          </h3>

          {/* Enable toggle */}
          <div className={clsx('rounded-xl border p-4', enabled ? 'border-brand-500/30 bg-brand-500/5' : 'border-surface-600')}>
            <Toggle
              checked={enabled}
              onChange={(v) => { setEnabled(v); setSaved(false); if (!v) setStrategy('none') }}
              label="Enable Class Balancing"
              sub={isBalanced && !enabled ? 'Not recommended — data is already balanced' : 'Apply balancing strategy to X_train/y_train during training'}
              disabled={false}
            />
            {isBalanced && enabled && (
              <div className="mt-3 pt-3 border-t border-surface-700">
                <Toggle
                  checked={forceBalance}
                  onChange={setForceBalance}
                  label="Force resampling on balanced data"
                  sub="⚠ This may degrade model performance"
                />
              </div>
            )}
          </div>

          {enabled && (
            <>
              {/* Auto vs Manual mode */}
              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={() => setAutoMode(true)}
                  className={clsx('px-3 py-1.5 rounded-xl border text-xs font-medium transition-all flex items-center gap-1.5',
                    autoMode ? 'border-brand-500/60 bg-brand-600/20 text-brand-300' : 'border-surface-600 text-slate-400 hover:border-surface-500')}
                >
                  <Sparkles size={11} /> Auto (Recommended)
                </button>
                <button
                  onClick={() => setAutoMode(false)}
                  className={clsx('px-3 py-1.5 rounded-xl border text-xs font-medium transition-all',
                    !autoMode ? 'border-brand-500/60 bg-brand-600/20 text-brand-300' : 'border-surface-600 text-slate-400 hover:border-surface-500')}
                >
                  Manual Select
                </button>
              </div>

              {/* Recommendation box (auto mode) */}
              {autoMode && analysis?.recommendation && (
                <div className="bg-brand-600/10 border border-brand-600/20 rounded-xl px-4 py-3 space-y-1">
                  <p className="text-xs font-semibold text-brand-300 flex items-center gap-1.5">
                    <Zap size={11} /> Auto-selected: {analysis.recommendation.label}
                  </p>
                  <p className="text-xs text-slate-400 leading-relaxed">{analysis.recommendation.reason}</p>
                </div>
              )}

              {/* Priority override notice (Point 23) */}
              {priority?.overridden && (
                <div className="flex items-start gap-2.5 bg-amber-500/8 border border-amber-500/20 rounded-xl px-3 py-2.5">
                  <AlertTriangle size={12} className="text-amber-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-xs font-semibold text-amber-300">Auto-override applied</p>
                    <p className="text-[10px] text-slate-400 mt-0.5">{priority.override_reason}</p>
                  </div>
                </div>
              )}

              {/* Strategy selector */}
              <div className="space-y-2">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  {autoMode ? 'Selected Strategy' : 'Choose Strategy'}
                </p>
                <div className="space-y-2">
                  {(analysis?.available_strategies || [])
                    .filter(s => !s.hidden)
                    .map(s => (
                      <StrategyCard
                        key={s.id}
                        s={s}
                        selected={strategy}
                        onSelect={(id) => { setStrategy(id); setSaved(false) }}
                        recommended={s.id === analysis?.recommendation?.strategy}
                      />
                    ))}
                </div>
              </div>

              {/* Point 21: Force large override */}
              {isLarge && ['smote','adasyn','smoteenn','smotetomek','smotenc'].includes(strategy) && (
                <div className="rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 space-y-2">
                  <Toggle
                    checked={forceLarge}
                    onChange={setForceLarge}
                    label="Force resampling on large dataset"
                    sub={`⚠ Override memory safety block for ${analysis.n_train?.toLocaleString()} rows — may be slow or cause OOM errors`}
                  />
                </div>
              )}

              {/* Preview button */}
              {strategy && !['none', 'class_weight'].includes(strategy) && (
                <div className="flex items-center gap-3 pt-1">
                  <button
                    onClick={runPreview}
                    disabled={previewing || analysis?.smote_blocked}
                    className="btn-secondary text-sm flex items-center gap-2"
                  >
                    {previewing ? <><div className="spinner" /> Simulating…</> : <><Eye size={13} /> Preview Distribution Change</>}
                  </button>
                  {preview && (
                    <button onClick={() => setShowPreview(v => !v)} className="text-xs text-slate-400 hover:text-slate-300 flex items-center gap-1">
                      {showPreview ? <EyeOff size={11} /> : <Eye size={11} />}
                      {showPreview ? 'Hide' : 'Show'} Preview
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── Before/after preview ── */}
      {showPreview && preview && (
        <div className="card animate-slide-up space-y-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Distribution Preview — Simulation Only (session data unchanged)
          </p>
          {preview.resampled ? (
            <div className="grid md:grid-cols-2 gap-4">
              <DistributionChart dist={preview.before?.distribution} title="Before Resampling" subtitle="Original y_train" />
              <div>
                <DistributionChart dist={preview.after?.distribution} title="After Resampling" subtitle={`${preview.strategy} applied`} />
                {preview.delta_samples != null && (
                  <p className={clsx('text-xs mt-2 font-semibold', preview.delta_samples > 0 ? 'text-brand-400' : 'text-amber-400')}>
                    {preview.delta_samples > 0 ? '+' : ''}{preview.delta_samples.toLocaleString()} samples
                    {preview.delta_samples > 0 ? ' (oversampled)' : ' (undersampled)'}
                  </p>
                )}
              </div>
            </div>
          ) : (
            <Banner type="info">{preview.note}</Banner>
          )}
        </div>
      )}

      {/* ── Dynamic warnings ── */}
      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((w, i) => (
            <Banner key={i} type={w.type}>{w.msg}</Banner>
          ))}
        </div>
      )}

      {/* ── Model-aware hints ── */}
      <div className="card border-surface-600 space-y-2">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Model-Aware Hints</p>
        <div className="text-xs space-y-1.5 text-slate-400 leading-relaxed">
          <p>🌲 <span className="text-slate-300 font-medium">Tree-based models</span> (RF, XGBoost, LightGBM) — prefer <code>class_weight='balanced'</code> or <code>scale_pos_weight</code>. Often don't need resampling.</p>
          <p>📈 <span className="text-slate-300 font-medium">Linear models</span> (LR, SVM, KNN) — benefit more from resampling (SMOTE) or class weighting.</p>
          <p>⏱ <span className="text-slate-300 font-medium">Time series data</span> — <strong className="text-red-400">never apply resampling</strong>. Use class_weight only.</p>
          <p>📐 <span className="text-slate-300 font-medium">High-dimensional data</span> ({analysis?.n_features || '—'} features) — avoid SMOTE; prefer class_weight.</p>
        </div>
      </div>

      {/* ── Confirm ── */}
      <div className="flex items-center gap-4 flex-wrap">
        <button
          onClick={handleConfirm}
          disabled={saving}
          className="btn-primary flex items-center gap-2 py-2.5 px-5"
        >
          {saving
            ? <><div className="spinner" /> Saving…</>
            : <><CheckCircle size={15} /> Confirm &amp; Continue</>
          }
        </button>

        {saved && (
          <div className="flex items-center gap-2 text-sm text-emerald-400">
            <CheckCircle size={14} />
            <span>
              <strong>{effectiveStrategy === 'none' ? 'No Balancing' : (analysis?.available_strategies?.find(s => s.id === effectiveStrategy)?.label || effectiveStrategy)}</strong> saved —
              training will apply it inside imblearn.Pipeline automatically
            </span>
          </div>
        )}

        {saved && (
          <button
            onClick={() => onNavigate?.('models')}
            className="btn-secondary ml-auto flex items-center gap-1.5 text-sm"
          >
            Model Selection <ChevronRight size={13} />
          </button>
        )}
      </div>

      {/* ── Evaluation reminder ── */}
      {saved && (
        <Banner type="info" title="Evaluation Metrics Reminder">
          With imbalanced data, use <strong>F1 Score, Precision, Recall, PR-AUC, and Confusion Matrix</strong>.
          Accuracy alone is misleading — a model predicting the majority class always gets high accuracy.
          The Evaluation step provides threshold tuning (default 0.5 is not always optimal).
        </Banner>
      )}
    </div>
  )
}
