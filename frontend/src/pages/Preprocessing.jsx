import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  CheckCircle, AlertCircle, AlertTriangle, Info,
  RefreshCw, Play, Eye, Lock, Filter,
  BarChart2, Layers, ChevronDown, ChevronUp,
  X, Database, Zap, Shield, Activity,
} from 'lucide-react'
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 300_000 })

// ─── Tiny helpers ─────────────────────────────────────────────────────────────
const clsx = (...a) => a.filter(Boolean).join(' ')

function Banner({ type = 'info', children, onDismiss, className = '' }) {
  const s = {
    info:    'bg-blue-500/8 border-blue-500/25 text-blue-300',
    warn:    'bg-amber-500/10 border-amber-500/30 text-amber-300',
    error:   'bg-red-500/10 border-red-500/30 text-red-400',
    success: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
    brand:   'bg-brand-500/8 border-brand-500/20 text-brand-300',
  }
  const ic = {
    info: <Info size={13} className="shrink-0 mt-0.5" />,
    warn: <AlertTriangle size={13} className="shrink-0 mt-0.5" />,
    error: <AlertCircle size={13} className="shrink-0 mt-0.5" />,
    success: <CheckCircle size={13} className="shrink-0 mt-0.5" />,
    brand: <Lock size={13} className="shrink-0 mt-0.5" />,
  }
  return (
    <div className={clsx(`flex gap-2 rounded-xl border px-4 py-3 text-sm ${s[type]}`, className)}>
      {ic[type]}
      <span className="flex-1">{children}</span>
      {onDismiss && <button onClick={onDismiss}><X size={12} className="opacity-60 hover:opacity-100" /></button>}
    </div>
  )
}

function StatBox({ label, value, color = 'text-white', sub }) {
  return (
    <div className="card text-center py-4">
      <p className={clsx('text-2xl font-bold tabular-nums', color)}>{value ?? '—'}</p>
      {sub && <p className="text-[10px] text-slate-500 mt-0.5">{sub}</p>}
      <p className="text-xs text-slate-400 mt-1">{label}</p>
    </div>
  )
}

function OptionPill({ active, onClick, children, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'px-3 py-2 rounded-xl border text-sm font-medium transition-all',
        active
          ? 'border-brand-500/60 bg-brand-600/20 text-brand-300'
          : 'border-surface-600 text-slate-400 hover:border-surface-500 hover:text-slate-300',
        disabled && 'opacity-40 cursor-not-allowed'
      )}
    >
      {children}
    </button>
  )
}

function Toggle({ checked, onChange, label, sub }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="min-w-0">
        <p className="text-sm text-slate-300 font-medium">{label}</p>
        {sub && <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{sub}</p>}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={clsx(
          'relative w-10 h-5 rounded-full transition-colors shrink-0',
          checked ? 'bg-brand-600' : 'bg-surface-600'
        )}
      >
        <span className={clsx('absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform',
          checked ? 'translate-x-5' : '')} />
      </button>
    </div>
  )
}

function Card({ children, className = '', ...p }) {
  return <div className={clsx('card', className)} {...p}>{children}</div>
}

function CollapseCard({ title, icon: Icon, iconColor = 'text-brand-400', badge, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <Card>
      <button onClick={() => setOpen(v => !v)} className="flex items-center gap-2 w-full text-left">
        <Icon size={14} className={iconColor} />
        <span className="font-semibold text-white text-sm flex-1">{title}</span>
        {badge != null && (
          <span className="px-2 py-0.5 rounded-lg bg-surface-700 border border-surface-600 text-xs text-slate-400">{badge}</span>
        )}
        {open ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
      </button>
      {open && <div className="mt-4 animate-slide-up">{children}</div>}
    </Card>
  )
}

// ─── Encoding config for UI ───────────────────────────────────────────────────
const LOW_ENC_OPTIONS = [
  { id: 'ohe',       label: 'OneHot',    desc: 'Best for nominal. Expands columns (drop=first).' },
  { id: 'frequency', label: 'Frequency', desc: 'Replaces with category frequency. 1 col per feature.' },
  { id: 'label',     label: 'Label ⚠',  desc: 'Ordinal/tree models only. Introduces order.' },
]

const HIGH_ENC_OPTIONS = [
  { id: 'frequency', label: 'Frequency', desc: 'Recommended. Reduces to 1 col per feature.' },
  { id: 'target',    label: 'Target ⚠',  desc: 'Smoothed target encoding. Use with caution.' },
  { id: 'label',     label: 'Label ⚠',  desc: 'Ordinal/tree models only. Maps unknown → -1.' },
]


// ─── Dynamic warnings ─────────────────────────────────────────────────────────
function useConfigWarnings(config, analysis) {
  return useMemo(() => {
    const w = []
    if (config.lowCardEnc === 'label') w.push({ type: 'warn', msg: "Label Encoding introduces artificial order — only valid for ordinal features or tree-based models (RF, XGBoost)." })
    if (config.highCardEnc === 'label') w.push({ type: 'warn', msg: "Label Encoding on high-cardinality features is not recommended. Consider Frequency or Target encoding." })
    if (config.highCardEnc === 'target') w.push({ type: 'warn', msg: "Target Encoding can leak target information without cross-fitting. Use with caution in production." })
    if (config.numImputer === 'knn' && analysis?.auto_hints?.warn_knn_large) w.push({ type: 'warn', msg: "KNN Imputer is slow on large datasets (>10,000 rows). Consider Mean or Median." })
    if (config.lowCardEnc === 'ohe') {
      const estExplosion = Object.entries(analysis?.explosion_per_col || {})
        .filter(([c]) => analysis?.cat_cols?.includes(c))
        .reduce((s, [, v]) => s + v, 0)
      if (estExplosion > 50) w.push({ type: 'warn', msg: `OHE will create ~${estExplosion} columns from categorical features. Consider Frequency encoding for high-dimensional data.` })
    }
    return w
  }, [config, analysis])
}

// ─── Feature explosion estimator ──────────────────────────────────────────────
function useEstimatedFeatures(config, analysis) {
  return useMemo(() => {
    if (!analysis) return null
    let total = (analysis.num_cols?.length || 0)
    const catCols = analysis.cat_cols || []
    const explosion = analysis.explosion_per_col || {}
    for (const col of catCols) {
      const nu = analysis.column_info?.find(c => c.name === col)?.n_unique || 1
      const isHigh = nu > config.highCardThresh
      if (isHigh) {
        // target/frequency/label → 1 each
        total += 1
      } else {
        // ohe: n_unique - 1; others: 1
        total += config.lowCardEnc === 'ohe' ? Math.max(1, (explosion[col] || 1)) : 1
      }
    }
    return total
  }, [config, analysis])
}

// ═══════════════════════════════════════════════════════════════════════════════
export default function Preprocessing() {
  // ── Server state ──────────────────────────────────────────────────
  const [analysis,  setAnalysis]  = useState(null)
  const [preview,   setPreview]   = useState(null)
  const [status,    setStatus]    = useState(null)
  const [loading,   setLoading]   = useState(true)
  const [flash,     setFlash]     = useState(null)

  // ── Config ────────────────────────────────────────────────────────
  const [numImputer,    setNumImputer]    = useState('median')
  const [catImputer,    setCatImputer]    = useState('most_frequent')
  const [outlier,       setOutlier]       = useState('iqr')
  const [rareThresh,    setRareThresh]    = useState(5)
  const [lowCardEnc,    setLowCardEnc]    = useState('ohe')
  const [highCardEnc,   setHighCardEnc]   = useState('frequency')
  const [highCardThresh,setHighCardThresh]= useState(15)
  const [dropCorr,      setDropCorr]      = useState(true)
  const [corrThresh,    setCorrThresh]    = useState(95)

  // ── UI state ─────────────────────────────────────────────────────
  const [applying,           setApplying]           = useState(false)
  const [result,             setResult]             = useState(null)
  const [showPreview,        setShowPreview]        = useState(false)
  const [showMapping,        setShowMapping]        = useState(false)
  const [confirmRerun,       setConfirmRerun]       = useState(false)
  const [droppedHighMissing, setDroppedHighMissing] = useState(new Set()) // cols >50% missing to drop before pipeline

  const config = { numImputer, catImputer, outlier, rareThresh, lowCardEnc, highCardEnc, highCardThresh, dropCorr, corrThresh }
  const configWarnings = useConfigWarnings(config, analysis)
  const estimatedFeatures = useEstimatedFeatures(config, analysis)

  // Helper: re-apply all recommendations from current analysis
  const applyRecommendations = useCallback(() => {
    const h = analysis?.auto_hints
    if (!h) return
    if (h.recommended_num_imputer)   setNumImputer(h.recommended_num_imputer)
    if (h.recommended_cat_imputer)   setCatImputer(h.recommended_cat_imputer)
    if (h.recommended_outlier)       setOutlier(h.recommended_outlier)
    if (h.recommended_low_card_enc)  setLowCardEnc(h.recommended_low_card_enc)
    if (h.recommended_high_card_enc) setHighCardEnc(h.recommended_high_card_enc)
    if (typeof h.recommended_drop_corr === 'boolean') setDropCorr(h.recommended_drop_corr)
  }, [analysis])

  const [recDismissed, setRecDismissed] = useState(false)

  // ── Load ─────────────────────────────────────────────────────────
  const loadAll = useCallback(async () => {
    setLoading(true)
    setRecDismissed(false)   // always show recommendations panel on fresh load
    try {
      const [anaR, pvR, stR] = await Promise.allSettled([
        api.get('/preprocess/analyze'),
        api.get('/preprocess/preview'),
        api.get('/preprocess/status'),
      ])
      if (anaR.status === 'fulfilled') {
        const d = anaR.value.data
        setAnalysis(d)
        // Auto-apply ALL backend recommendations — no user action required
        if (d.auto_hints) {
          const h = d.auto_hints
          if (h.recommended_num_imputer)                    setNumImputer(h.recommended_num_imputer)
          if (h.recommended_cat_imputer)                    setCatImputer(h.recommended_cat_imputer)
          if (h.recommended_outlier)                        setOutlier(h.recommended_outlier)
          if (h.recommended_low_card_enc)                   setLowCardEnc(h.recommended_low_card_enc)
          if (h.recommended_high_card_enc)                  setHighCardEnc(h.recommended_high_card_enc)
          if (typeof h.recommended_drop_corr === 'boolean') setDropCorr(h.recommended_drop_corr)
        }
      }
      if (pvR.status === 'fulfilled') setPreview(pvR.value.data)
      if (stR.status === 'fulfilled') setStatus(stR.value.data)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadAll() }, [loadAll])

  // ── Apply ─────────────────────────────────────────────────────────
  const doApply = async (force = false) => {
    setApplying(true)
    setFlash(null)
    setResult(null)
    setConfirmRerun(false)
    try {
      const res = await api.post('/preprocess/apply', {
        num_imputer:              numImputer,
        cat_imputer:              catImputer,
        outlier_handling:         outlier,
        // NOTE: handle_skewness and scaler are NOT sent here.
        // They belong in the Training pipeline (post-split, fitted on X_train only).
        rare_threshold:           rareThresh / 100,
        low_card_encoding:        lowCardEnc,
        high_card_encoding:       highCardEnc,
        high_card_threshold:      highCardThresh,
        drop_correlated:          dropCorr,
        correlation_threshold:    corrThresh / 100,
        force_reapply:            force,
        drop_cols_before_pipeline: [...droppedHighMissing],
      })
      setResult(res.data)
      setFlash({ type: 'success', msg: res.data.message })
      const [pvR, stR] = await Promise.all([
        api.get('/preprocess/preview'),
        api.get('/preprocess/status'),
      ])
      setPreview(pvR.data)
      setStatus(stR.data)
    } catch (e) {
      const detail = e.response?.data?.detail || ''
      if (detail.startsWith('PIPELINE_EXISTS')) {
        setConfirmRerun(true)
      } else {
        setFlash({ type: 'error', msg: detail || 'Preprocessing failed.' })
      }
    } finally {
      setApplying(false)
    }
  }

  // ── Gating — only block if no dataset loaded at all ──────────────────────
  // Preprocessing runs on the FULL processed_df (before any split).
  // It does NOT require X_train or X_test to exist.
  const noDataset = !loading && analysis === null && preview === null
  const ppDone    = status?.done || preview?.preprocessed || false

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div><h2 className="section-title">Preprocessing</h2></div>
        <Card className="flex items-center gap-3 text-brand-300"><div className="spinner" /> Analyzing dataset…</Card>
      </div>
    )
  }

  if (noDataset) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div><h2 className="section-title">Preprocessing</h2></div>
        <Card className="border-amber-500/30 bg-amber-500/8 flex items-center gap-3">
          <AlertTriangle size={20} className="text-amber-400 shrink-0" />
          <div>
            <p className="font-semibold text-white text-sm">No dataset loaded</p>
            <p className="text-xs text-slate-400 mt-0.5">Upload a CSV and complete Data Cleaning first, then return here.</p>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-5 animate-fade-in">

      {/* ─── Header ─── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="section-title">Preprocessing</h2>
          <p className="section-subtitle">sklearn Pipeline — fit on full dataset before split · no scaling/skewness (configured in Training)</p>
        </div>
        <div className="flex gap-2 shrink-0">
          <button onClick={loadAll} className="btn-secondary text-sm flex items-center gap-2">
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Re-analyze
          </button>
        </div>
      </div>

      {/* ─── Leakage rule ─── */}
      <Banner type="brand">
        <div className="space-y-0.5">
          <p className="font-semibold text-brand-200">Full-Dataset Pipeline — No Leakage by Design</p>
          <p className="text-xs opacity-80">All steps live inside one <code>sklearn.Pipeline</code>: ColumnTransformer → VarianceThreshold → CorrelationDropper</p>
          <p className="text-xs opacity-80">Fitted on the <strong>full dataset</strong> before split · Skewness correction &amp; scaling configured in the Training step</p>
        </div>
      </Banner>

      {/* ─── Pipeline status ─── */}
      {ppDone && !result && (
        <div className="flex items-center gap-3 rounded-xl border border-brand-500/30 bg-brand-500/8 px-4 py-3">
          <Shield size={16} className="text-brand-400 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-white">Pipeline already applied — {status?.n_features} features · {status?.total_rows?.toLocaleString()} rows</p>
            <p className="text-xs text-slate-400">Re-applying will overwrite the existing pipeline. You will need to re-run Split Data afterwards.</p>
          </div>
          {status?.pipeline_saved && (
            <span className="badge text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 text-xs shrink-0">✓ Saved</span>
          )}
          {status?.version_mismatch?.length > 0 && (
            <span className="badge text-amber-400 bg-amber-500/10 border border-amber-500/20 text-xs shrink-0">⚠ Version mismatch</span>
          )}
        </div>
      )}
      {status?.version_mismatch?.map((w, i) => (
        <Banner key={i} type="warn">Library version mismatch: {w}</Banner>
      ))}

      {/* ─── Stats ─── */}
      {analysis && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatBox label="Numeric Cols"   value={analysis.num_cols?.length}  color="text-blue-400" />
          <StatBox label="Categorical"    value={analysis.cat_cols?.length}  color="text-amber-400" />
          <StatBox label="Skewed Cols"    value={analysis.skewed_cols?.length} color={analysis.skewed_cols?.length > 0 ? 'text-orange-400' : 'text-slate-500'} />
          <StatBox label="Est. Features"  value={estimatedFeatures}          color="text-brand-400" sub="after encoding" />
        </div>
      )}

      {/* ─── Flash + analysis warnings ─── */}
      {flash && <Banner type={flash.type} onDismiss={() => setFlash(null)}>{flash.msg}</Banner>}
      {analysis?.version_warnings?.map((w, i) => <Banner key={i} type="warn">{w}</Banner>)}
      {analysis?.warnings?.map((w, i) => <Banner key={i} type="warn">{w}</Banner>)}

      {/* ─── Smart Recommendations panel ─── */}
      {analysis?.auto_hints && !recDismissed && (() => {
        const h = analysis.auto_hints
        const taskLabel = h.task_type === 'regression' ? 'Regression'
                        : h.task_type === 'classification' ? 'Classification'
                        : null

        // Build recommendation rows — include "None" rows (they are valid recommendations)
        const recs = [
          {
            label: 'Numeric Imputer',
            value: h.recommended_num_imputer,
            current: numImputer,
            reason: h.reason_num_imputer,
            isNone: h.recommended_num_imputer === 'none',
          },
          {
            label: 'Categorical Imputer',
            value: h.recommended_cat_imputer,
            current: catImputer,
            reason: h.reason_cat_imputer,
            isNone: h.recommended_cat_imputer === 'none',
          },
          {
            label: 'Outlier Handling',
            value: h.recommended_outlier,
            current: outlier,
            reason: h.reason_outlier,
            isNone: h.recommended_outlier === 'none',
          },
          {
            label: 'Low-card Encoding',
            value: h.recommended_low_card_enc,
            current: lowCardEnc,
            reason: h.reason_low_card_enc,
            isNone: false,
          },
          {
            label: 'High-card Encoding',
            value: h.recommended_high_card_enc,
            current: highCardEnc,
            reason: h.reason_high_card_enc,
            isNone: false,
          },
          {
            label: 'Drop Correlated',
            value: h.recommended_drop_corr ? 'yes' : 'no',
            current: dropCorr ? 'yes' : 'no',
            reason: h.reason_drop_corr,
            isNone: !h.recommended_drop_corr,
          },
        ].filter(r => r.value != null && r.reason)

        if (!recs.length) return null
        const allActive = recs.every(r => String(r.current) === String(r.value))

        return (
          <Card className="border-brand-500/30 bg-brand-500/5 space-y-3">
            {/* Header */}
            <div className="flex items-center gap-2 flex-wrap">
              <Zap size={15} className="text-brand-400 shrink-0" />
              <p className="font-semibold text-white text-sm flex-1">Smart Recommendations</p>
              {taskLabel && (
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                  h.task_type === 'regression'
                    ? 'text-blue-400 bg-blue-500/15 border-blue-500/30'
                    : 'text-purple-400 bg-purple-500/15 border-purple-500/30'
                }`}>
                  {taskLabel}
                </span>
              )}
              {allActive
                ? <span className="text-[10px] text-emerald-400 font-semibold">✓ All applied</span>
                : (
                  <button
                    onClick={applyRecommendations}
                    className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-brand-500/50 bg-brand-600/20 text-brand-300 hover:bg-brand-600/30 transition-all"
                  >
                    Apply All
                  </button>
                )
              }
              <button onClick={() => setRecDismissed(true)}>
                <X size={13} className="text-slate-500 hover:text-slate-300" />
              </button>
            </div>

            {/* Context line */}
            <p className="text-xs text-slate-500 leading-relaxed">
              Auto-detected from your dataset ({analysis.n_rows?.toLocaleString()} rows,
              {' '}{analysis.num_cols?.length} numeric, {analysis.cat_cols?.length} categorical).
              Settings below are already pre-selected — override any manually.
              {h.handle_skewness_suggested && (
                <span className="text-orange-400 ml-1">
                  ⚡ {h.skewed_cols?.length} skewed column(s) detected — enable PowerTransformer in Training.
                </span>
              )}
            </p>

            {/* Recommendation rows */}
            <div className="divide-y divide-surface-700/60">
              {recs.map(r => {
                const isActive = String(r.current) === String(r.value)
                return (
                  <div key={r.label} className="flex items-start gap-3 py-2.5 first:pt-0 last:pb-0">
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold text-slate-300">{r.label}</p>
                      <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">{r.reason}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0 mt-0.5">
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                        r.isNone
                          ? 'text-slate-400 bg-surface-700 border-surface-600'   // muted for None
                          : 'text-brand-300 bg-brand-600/15 border-brand-500/40'  // accent for active
                      }`}>
                        {String(r.value)}
                      </span>
                      {isActive
                        ? <span className="text-[10px] text-emerald-400">✓</span>
                        : <span className="text-[10px] text-amber-400">≠</span>
                      }
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Evidence footer */}
            {(h.outlier_cols?.length > 0 || h.skewed_cols?.length > 0) && (
              <div className="border-t border-surface-700/60 pt-2.5 flex flex-wrap gap-3">
                {h.outlier_cols?.length > 0 && (
                  <div className="text-[11px] text-slate-500">
                    <span className="text-amber-400 font-semibold">Outlier cols:</span>
                    {' '}{h.outlier_cols.slice(0, 4).join(', ')}{h.outlier_cols.length > 4 ? ` +${h.outlier_cols.length - 4} more` : ''}
                  </div>
                )}
                {h.skewed_cols?.length > 0 && (
                  <div className="text-[11px] text-slate-500">
                    <span className="text-orange-400 font-semibold">Skewed cols:</span>
                    {' '}{h.skewed_cols.slice(0, 4).join(', ')}{h.skewed_cols.length > 4 ? ` +${h.skewed_cols.length - 4} more` : ''}
                  </div>
                )}
              </div>
            )}
          </Card>
        )
      })()}

      {/* ─── Dynamic config warnings ─── */}
      {configWarnings.length > 0 && (
        <div className="space-y-2">
          {configWarnings.map((w, i) => <Banner key={i} type={w.type}>{w.msg}</Banner>)}
        </div>
      )}


      {/* ─── HIGH MISSING COLUMNS PANEL (>50%) ─── */}
      {(() => {
        const highMissCols = (analysis?.column_info || []).filter(c => c.missing_pct > 50)
        if (highMissCols.length === 0) return null
        return (
          <Card className="space-y-3 border-red-500/30 bg-red-500/5">
            <div className="flex items-center gap-2">
              <AlertCircle size={15} className="text-red-400 shrink-0" />
              <p className="font-semibold text-white text-sm">
                {highMissCols.length} Column{highMissCols.length > 1 ? 's' : ''} with &gt;50% Missing Values
              </p>
              <span className="ml-auto text-[10px] text-red-400 bg-red-500/15 border border-red-500/25 rounded-full px-2 py-0.5">
                Action required
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Columns with more than 50% missing values are unreliable for imputation and can degrade model quality.
              Drop them before preprocessing, or proceed and impute (not recommended).
            </p>
            <div className="space-y-2">
              {highMissCols.map(col => {
                const isDropped = droppedHighMissing.has(col.name)
                return (
                  <div
                    key={col.name}
                    className={clsx(
                      'flex items-center gap-3 rounded-xl border px-3 py-2.5 transition-all',
                      isDropped
                        ? 'border-red-500/40 bg-red-500/10'
                        : 'border-surface-600 bg-surface-700/40'
                    )}
                  >
                    {/* Bar visualization */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-mono text-xs text-slate-200 truncate">{col.name}</span>
                        <span className={clsx(
                          'text-[10px] font-bold px-1.5 py-0.5 rounded border',
                          col.missing_pct > 80
                            ? 'text-red-400 bg-red-500/15 border-red-500/30'
                            : 'text-orange-400 bg-orange-500/15 border-orange-500/30'
                        )}>
                          {col.missing_pct}% missing
                        </span>
                        <span className="text-[10px] text-slate-500">{col.type}</span>
                      </div>
                      {/* Missing bar */}
                      <div className="h-1.5 rounded-full bg-surface-600 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-red-500/70 transition-all"
                          style={{ width: `${col.missing_pct}%` }}
                        />
                      </div>
                    </div>
                    {/* Drop toggle button */}
                    <button
                      onClick={() => setDroppedHighMissing(prev => {
                        const next = new Set(prev)
                        next.has(col.name) ? next.delete(col.name) : next.add(col.name)
                        return next
                      })}
                      className={clsx(
                        'shrink-0 text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all',
                        isDropped
                          ? 'border-red-500/60 bg-red-600/20 text-red-300 hover:bg-red-600/30'
                          : 'border-surface-500 text-slate-400 hover:border-red-500/40 hover:text-red-400 hover:bg-red-500/8'
                      )}
                    >
                      {isDropped ? '✓ Will Drop' : 'Drop Column'}
                    </button>
                  </div>
                )
              })}
            </div>
            {droppedHighMissing.size > 0 && (
              <Banner type="warn">
                {droppedHighMissing.size} column{droppedHighMissing.size > 1 ? 's' : ''} ({[...droppedHighMissing].join(', ')}) will be
                dropped from both X_train and X_test before the pipeline is built.
              </Banner>
            )}
          </Card>
        )
      })()}

      {/* ─── Re-run confirm dialog ─── */}
      {confirmRerun && (
        <Card className="border-amber-500/30 bg-amber-500/8 space-y-3">
          <div className="flex items-center gap-2">
            <AlertTriangle size={16} className="text-amber-400" />
            <p className="font-semibold text-white text-sm">Overwrite existing pipeline?</p>
          </div>
          <p className="text-xs text-slate-400">A pipeline is already applied. Re-applying will overwrite the saved <code>preprocessing_pipeline.pkl</code> and <code>preprocessing_metadata.json</code>.</p>
          <div className="flex gap-3">
            <button onClick={() => doApply(true)} className="btn-primary text-sm">Yes, overwrite</button>
            <button onClick={() => setConfirmRerun(false)} className="btn-secondary text-sm">Cancel</button>
          </div>
        </Card>
      )}

      {/* ══════════════════════════════════════════════
          CONFIG PANELS
      ══════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* ── Numeric pipeline ── */}
        <Card className="space-y-5">
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <Activity size={14} className="text-blue-400" />
            Numeric Pipeline
            <span className="text-xs font-normal text-slate-500">({analysis?.num_cols?.length || 0} columns)</span>
          </h3>

          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-slate-400">Missing Value Strategy</p>
              {(() => {
                const numMissing = (analysis?.column_info || [])
                  .filter(c => c.type === 'numeric' && c.missing > 0).length
                return numMissing === 0 ? (
                  <span className="text-[10px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-2 py-0.5">
                    No missing values
                  </span>
                ) : (
                  <span className="text-[10px] text-amber-400">{numMissing} col(s) have missing values</span>
                )
              })()}
            </div>
            {(() => {
              const numMissing = (analysis?.column_info || [])
                .filter(c => c.type === 'numeric' && c.missing > 0).length
              const hasMissing = numMissing > 0
              return (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {[['median','Median'],['mean','Mean'],['knn','KNN'],['none','None']].map(([v, l]) => (
                    <OptionPill
                      key={v}
                      active={numImputer === v}
                      onClick={() => setNumImputer(v)}
                      disabled={v === 'none' && hasMissing}  // None invalid when missing values exist
                    >
                      {l}
                    </OptionPill>
                  ))}
                </div>
              )
            })()}
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              {numImputer === 'knn'    && 'KNN uses 5 nearest neighbours — accurate but slow on large data.'}
              {numImputer === 'median' && 'Median is robust to outliers — recommended for skewed numeric data.'}
              {numImputer === 'mean'   && 'Mean is fast but pulled by outliers.'}
              {numImputer === 'none'   && '✓ No missing values detected — imputation skipped.'}
            </p>
          </div>

          <div>
            <p className="text-xs text-slate-400 mb-2">Outlier Handling</p>
            <div className="grid grid-cols-2 gap-2">
              <OptionPill active={outlier === 'iqr'}  onClick={() => setOutlier('iqr')}>IQR Capping</OptionPill>
              <OptionPill active={outlier === 'none'} onClick={() => setOutlier('none')}>None</OptionPill>
            </div>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              {outlier === 'iqr'
                ? (analysis?.auto_hints?.reason_outlier || 'IQR bounds fitted on dataset — caps extreme outliers.')
                : (analysis?.auto_hints?.outlier_cols?.length === 0 && !analysis?.auto_hints?.handle_skewness_suggested
                    ? '✓ No outliers or heavy skewness detected — capping skipped.'
                    : 'Outlier capping disabled. Enable if you see model instability.')
              }
            </p>
          </div>

          {/* Scaling/Skewness note */}
          <div className="bg-blue-500/8 rounded-xl border border-blue-500/20 p-3 space-y-1">
            <p className="text-xs font-semibold text-blue-300">⚡ Skewness Correction &amp; Scaling → Training Step</p>
            <p className="text-xs text-slate-500 leading-relaxed">
              PowerTransformer and StandardScaler/MinMaxScaler are configured in the <strong className="text-slate-300">Training</strong> page.
              They are fitted on X_train only (inside the post-split pipeline) — guaranteeing zero leakage.
            </p>
          </div>
        </Card>

        {/* ── Categorical pipeline ── */}
        <Card className="space-y-5">
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <Filter size={14} className="text-amber-400" />
            Categorical Pipeline
            <span className="text-xs font-normal text-slate-500">({analysis?.cat_cols?.length || 0} columns)</span>
            {analysis && (analysis.cat_cols?.length || 0) === 0 && (
              <span className="ml-auto text-[10px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-2 py-0.5">N/A</span>
            )}
          </h3>

          {/* No categorical columns — show N/A state */}
          {analysis && (analysis.cat_cols?.length || 0) === 0 ? (
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-5 text-center space-y-1">
              <p className="text-sm font-semibold text-emerald-400">✓ No categorical features detected</p>
              <p className="text-xs text-slate-500">
                All columns are numeric. Categorical imputation and encoding steps will be skipped automatically.
              </p>
            </div>
          ) : (
            <>
          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-slate-400">Missing Value Strategy</p>
              {(() => {
                const catMissing = (analysis?.column_info || [])
                  .filter(c => c.type === 'categorical' && c.missing > 0).length
                return catMissing === 0 ? (
                  <span className="text-[10px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-2 py-0.5">
                    No missing values
                  </span>
                ) : (
                  <span className="text-[10px] text-amber-400">{catMissing} col(s) have missing values</span>
                )
              })()}
            </div>
            {(() => {
              const catMissing = (analysis?.column_info || [])
                .filter(c => c.type === 'categorical' && c.missing > 0).length
              const hasMissing = catMissing > 0
              return (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <OptionPill active={catImputer === 'most_frequent'} onClick={() => setCatImputer('most_frequent')}>Most Frequent</OptionPill>
                  <OptionPill active={catImputer === 'constant'}      onClick={() => setCatImputer('constant')}>Fill "Missing"</OptionPill>
                  <OptionPill
                    active={catImputer === 'none'}
                    onClick={() => setCatImputer('none')}
                    disabled={hasMissing}
                  >
                    None
                  </OptionPill>
                </div>
              )
            })()}
            {catImputer === 'none'
              ? <p className="text-xs text-slate-500 mt-1.5">✓ No missing values in categorical columns — imputation skipped.</p>
              : catImputer === 'constant'
              ? <p className="text-xs text-slate-500 mt-1.5">Fills missing with the literal string "Missing" — preserves absence as a category signal.</p>
              : <p className="text-xs text-slate-500 mt-1.5">Replaces missing with the most common value per column.</p>
            }
          </div>

          <div>
            <p className="text-xs text-slate-400 mb-1.5">
              Low-cardinality Encoding
              <span className="text-slate-600 ml-1">(≤ {highCardThresh} unique)</span>
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {LOW_ENC_OPTIONS.map(({ id, label }) => (
                <OptionPill key={id} active={lowCardEnc === id} onClick={() => setLowCardEnc(id)}>{label}</OptionPill>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-1.5">{LOW_ENC_OPTIONS.find(o => o.id === lowCardEnc)?.desc}</p>
          </div>

          <div>
            <p className="text-xs text-slate-400 mb-1.5">
              High-cardinality Encoding
              <span className="text-slate-600 ml-1">(&gt; {highCardThresh} unique)</span>
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {HIGH_ENC_OPTIONS.map(({ id, label }) => (
                <OptionPill key={id} active={highCardEnc === id} onClick={() => setHighCardEnc(id)}>{label}</OptionPill>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-1.5">{HIGH_ENC_OPTIONS.find(o => o.id === highCardEnc)?.desc}</p>
          </div>

          <div>
            <p className="text-xs text-slate-400 mb-1">
              High-card Threshold — <span className="text-brand-300 font-semibold">{highCardThresh}</span>
            </p>
            <input type="range" min={5} max={50} step={1} value={highCardThresh}
              onChange={e => setHighCardThresh(Number(e.target.value))} className="w-full accent-brand-500" />
          </div>

          <div>
            <p className="text-xs text-slate-400 mb-1">
              Rare Category Threshold — <span className="text-brand-300 font-semibold">{rareThresh}%</span>
            </p>
            <input type="range" min={1} max={20} step={1} value={rareThresh}
              onChange={e => setRareThresh(Number(e.target.value))} className="w-full accent-brand-500" />
            <p className="text-xs text-slate-500 mt-1">Categories below {rareThresh}% frequency → grouped as "Other". Unseen test categories do NOT crash.</p>
          </div>

          <div className="border-t border-surface-700 pt-4 space-y-4">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Post-Pipeline Filters</p>
            <Toggle
              checked={dropCorr} onChange={setDropCorr}
              label="Drop Highly Correlated"
              sub={analysis?.auto_hints?.reason_drop_corr || 'CorrelationDropper — fitted on full dataset, drops one col from each correlated pair (>0.85)'}
            />
            {dropCorr && (
              <div>
                <p className="text-xs text-slate-400 mb-1">
                  Threshold — <span className="text-brand-300 font-semibold">{corrThresh}%</span>
                </p>
                <input type="range" min={80} max={99} step={1} value={corrThresh}
                  onChange={e => setCorrThresh(Number(e.target.value))} className="w-full accent-brand-500" />
                {(() => {
                  const n = (analysis?.corr_pairs || []).filter(p => p.corr > corrThresh / 100).length
                  return n > 0
                    ? <p className="text-xs text-amber-400 mt-1">{n} pair(s) will be dropped at this threshold.</p>
                    : <p className="text-xs text-slate-500 mt-1">No pairs exceed this threshold — no columns will be dropped.</p>
                })()}
              </div>
            )}
          </div>
            </>
          )}
        </Card>
      </div>

      {/* ── Feature explosion estimator ── */}
      {estimatedFeatures != null && (
        <Card className="flex items-center gap-4">
          <Zap size={16} className="text-brand-400 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-white">Estimated Feature Count After Encoding</p>
            <p className="text-xs text-slate-400 mt-0.5">
              {analysis?.num_cols?.length || 0} numeric + ~{estimatedFeatures - (analysis?.num_cols?.length || 0)} from categorical = <span className="text-brand-300 font-bold">{estimatedFeatures} total</span>
              {' '}<span className="text-slate-500">(before VarianceThreshold + CorrelationDropper)</span>
            </p>
          </div>
          <div className="text-3xl font-bold text-brand-400 tabular-nums">{estimatedFeatures}</div>
        </Card>
      )}

      {/* ── Correlation pairs ── */}
      {(analysis?.corr_pairs?.length || 0) > 0 && (
        <CollapseCard
          title={`Correlated Feature Pairs — computed on X_train`}
          icon={BarChart2} iconColor="text-amber-400"
          badge={analysis.corr_pairs.length}
        >
          <p className="text-xs text-slate-500 mb-3">Correlation computed on transformed features after pipeline (train-only). Values &gt;0.95 are severe.</p>
          <div className="table-wrapper">
            <table className="data-table">
              <thead><tr><th>Feature A</th><th>Feature B</th><th>Correlation</th><th>Severity</th><th>Action</th></tr></thead>
              <tbody>
                {analysis.corr_pairs.map((p, i) => (
                  <tr key={i}>
                    <td className="font-mono text-brand-300 text-xs">{p.col_a}</td>
                    <td className="font-mono text-brand-300 text-xs">{p.col_b}</td>
                    <td className={clsx('font-bold text-xs tabular-nums', p.severe ? 'text-red-400' : 'text-amber-400')}>{p.corr}</td>
                    <td><span className={clsx('badge border text-xs', p.severe ? 'text-red-400 bg-red-500/10 border-red-500/20' : 'text-amber-400 bg-amber-500/10 border-amber-500/20')}>{p.severe ? 'Severe >0.95' : 'High >0.90'}</span></td>
                    <td>{dropCorr && p.corr > corrThresh / 100
                      ? <span className="badge text-xs text-red-400 bg-red-500/10 border border-red-500/20">Drop one</span>
                      : <span className="text-xs text-slate-500">Keep</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapseCard>
      )}

      {/* ── Column analysis ── */}
      {analysis?.column_info && (
        <CollapseCard title="X_train Column Analysis" icon={Layers} iconColor="text-brand-400" badge={analysis.column_info.length}>
          <div className="table-wrapper">
            <table className="data-table">
              <thead><tr><th>Feature</th><th>Type</th><th>Unique</th><th>Missing %</th><th>Skewness</th><th>Flags</th></tr></thead>
              <tbody>
                {analysis.column_info.map(col => (
                  <tr key={col.name}>
                    <td className="font-mono text-slate-300 text-xs">{col.name}</td>
                    <td><span className={clsx('badge border text-xs', col.type === 'numeric' ? 'text-blue-400 bg-blue-500/10 border-blue-500/20' : 'text-amber-400 bg-amber-500/10 border-amber-500/20')}>{col.type}</span></td>
                    <td className="text-slate-400 text-xs tabular-nums">{col.n_unique}</td>
                    <td className={clsx('text-xs tabular-nums font-medium', col.warn_missing ? 'text-red-400' : col.missing > 0 ? 'text-amber-400' : 'text-slate-500')}>{col.missing_pct}%</td>
                    <td className={clsx('text-xs font-mono tabular-nums', col.needs_transform ? 'text-orange-400' : 'text-slate-500')}>{col.skewness ?? '—'}</td>
                    <td className="flex flex-wrap gap-1">
                      {col.warn_missing    && <span className="badge text-red-400 bg-red-500/10 border border-red-500/20 text-[10px]">High Missing</span>}
                      {col.needs_transform && <span className="badge text-orange-400 bg-orange-500/10 border border-orange-500/20 text-[10px]">Skewed</span>}
                      {col.warn_card       && <span className="badge text-amber-400 bg-amber-500/10 border border-amber-500/20 text-[10px]">High Card</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapseCard>
      )}

      {/* ── Action buttons ── */}
      <div className="flex gap-3 flex-wrap">
        <button
          onClick={() => doApply(false)}
          disabled={applying}
          className="btn-primary flex-1 justify-center py-3 text-base"
        >
          {applying
            ? <><div className="spinner" /> Building & Fitting Pipeline…</>
            : <><Play size={16} /> Apply Preprocessing</>
          }
        </button>
        {preview && (
          <button onClick={() => setShowPreview(v => !v)} className="btn-secondary flex items-center gap-2">
            <Eye size={14} /> {showPreview ? 'Hide' : 'Preview'}
          </button>
        )}
      </div>

      {/* ═══════════════════════════════════════════════
          RESULTS
      ═══════════════════════════════════════════════ */}
      {result && (
        <div className="space-y-4 animate-slide-up">
          <div className="flex items-center gap-2 flex-wrap">
            <CheckCircle size={18} className="text-emerald-400 shrink-0" />
            <h3 className="font-semibold text-white">Preprocessing Complete</h3>
            <div className="ml-auto flex gap-2">
              {result.pipeline_saved  && <span className="badge text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20">✓ Pipeline saved</span>}
              {result.metadata_saved  && <span className="badge text-xs text-brand-400 bg-brand-500/10 border border-brand-500/20">✓ Metadata saved</span>}
              <span className="badge text-xs text-slate-400 bg-surface-700 border border-surface-600">Ready for training</span>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatBox label="Train Rows"      value={result.train_rows?.toLocaleString()} color="text-emerald-400" />
            <StatBox label="Test Rows"       value={result.test_rows?.toLocaleString()}  color="text-rose-400" />
            <StatBox label="Features Before" value={result.n_features_before}            color="text-slate-400" />
            <StatBox label="Features After"  value={result.n_features_after}             color="text-brand-400" />
          </div>

          {result.dropped_variance?.length > 0 && (
            <Banner type="info">VarianceThreshold removed {result.dropped_variance.length} constant column(s): {result.dropped_variance.join(', ')}</Banner>
          )}
          {result.dropped_correlation?.length > 0 && (
            <Banner type="info">CorrelationDropper removed {result.dropped_correlation.length} correlated column(s): {result.dropped_correlation.join(', ')}</Banner>
          )}

          {/* Transformation summary */}
          <Card>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Transformation Summary per Feature</p>
            <div className="table-wrapper">
              <table className="data-table">
                <thead><tr><th>Feature</th><th>Type</th><th>Transformations Applied</th></tr></thead>
                <tbody>
                  {result.transformation_summary?.map((s, i) => (
                    <tr key={i}>
                      <td className="font-mono text-slate-300 text-xs">{s.feature}</td>
                      <td><span className={clsx('badge border text-[10px]',
                        s.type === 'numeric' ? 'text-blue-400 bg-blue-500/10 border-blue-500/20' :
                        s.type === 'categorical_high' ? 'text-orange-400 bg-orange-500/10 border-orange-500/20' :
                        'text-amber-400 bg-amber-500/10 border-amber-500/20')}>
                        {s.type === 'numeric' ? 'Numeric' : s.type === 'categorical_high' ? 'Cat (High)' : 'Cat (Low)'}
                      </span></td>
                      <td className="text-xs text-slate-400">{s.ops?.join(' → ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Feature mapping (original → transformed) */}
          {result.feature_mapping?.length > 0 && (
            <Card>
              <button onClick={() => setShowMapping(v => !v)} className="flex items-center gap-2 w-full text-left">
                <Database size={14} className="text-brand-400" />
                <span className="font-semibold text-white text-sm flex-1">
                  Feature Mapping — original → transformed
                </span>
                {showMapping ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
              </button>
              {showMapping && (
                <div className="mt-4 table-wrapper animate-slide-up">
                  <table className="data-table">
                    <thead><tr><th>Original Feature</th><th>Type</th><th>Transformed As</th><th>Output Cols</th></tr></thead>
                    <tbody>
                      {result.feature_mapping.map((m, i) => (
                        <tr key={i}>
                          <td className="font-mono text-slate-300 text-xs">{m.original}</td>
                          <td><span className="badge text-xs text-slate-400">{m.type}</span></td>
                          <td>
                            <div className="flex flex-wrap gap-1">
                              {m.transformed?.slice(0, 6).map(f => (
                                <span key={f} className="px-1.5 py-0.5 bg-brand-600/15 border border-brand-500/20 rounded text-[10px] font-mono text-brand-300">{f}</span>
                              ))}
                              {m.transformed?.length > 6 && <span className="text-xs text-slate-500">+{m.transformed.length - 6} more</span>}
                            </div>
                          </td>
                          <td className="text-slate-400 text-xs tabular-nums">{m.transformed?.length}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {/* Final feature list */}
          <Card>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Final Features in X_train ({result.feature_names?.length})
            </p>
            <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
              {result.feature_names?.map(f => (
                <span key={f} className="px-2 py-0.5 bg-surface-700 rounded-lg text-xs font-mono text-slate-300 border border-surface-600">{f}</span>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ── Data preview ── */}
      {showPreview && preview?.head?.length > 0 && (
        <Card className="animate-slide-up">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            X_train Preview — {preview.n_features} features · {preview.train_rows} rows
            {preview.preprocessed && <span className="text-emerald-400 ml-2">✓ Preprocessed</span>}
          </p>
          <div className="table-wrapper">
            <table className="data-table">
              <thead><tr>{Object.keys(preview.head[0]).map(k => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>
                {preview.head.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => <td key={j} className="font-mono text-xs text-slate-300 tabular-nums">{v}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Next step ── */}
      {(result || ppDone) && (
        <Banner type="success">
          Preprocessing pipeline saved and ready — proceed to <strong>Model Selection</strong>.
          Call <code>pipeline.transform(new_data)</code> for inference without any manual preprocessing.
        </Banner>
      )}
    </div>
  )
}
