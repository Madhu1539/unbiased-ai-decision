import React, { useState, useEffect, useCallback } from 'react'
import {
  Layers, Play, CheckCircle, AlertCircle, AlertTriangle,
  RefreshCw, RotateCcw, Trash2, Info, GitFork,
  BarChart2, FlaskConical, ChevronDown, ChevronUp,
  Lock, TrendingUp, Zap, X, Eye,
} from 'lucide-react'
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 120_000 })

// ─── Small UI helpers ─────────────────────────────────────────────────────────
function Alert({ type, children, onDismiss }) {
  const cfg = {
    info:    { bg: 'bg-blue-500/8 border-blue-500/25 text-blue-300',     icon: <Info size={14} className="shrink-0 mt-0.5" /> },
    warn:    { bg: 'bg-warn-500/10 border-warn-500/30 text-warn-300',    icon: <AlertTriangle size={14} className="shrink-0 mt-0.5" /> },
    error:   { bg: 'bg-danger-500/10 border-danger-500/30 text-danger-400', icon: <AlertCircle size={14} className="shrink-0 mt-0.5" /> },
    success: { bg: 'bg-accent-500/10 border-accent-500/30 text-accent-400', icon: <CheckCircle size={14} className="shrink-0 mt-0.5" /> },
  }
  const { bg, icon } = cfg[type] || cfg.info
  return (
    <div className={`flex gap-2 rounded-xl border px-4 py-3 text-sm ${bg}`}>
      {icon}
      <span className="flex-1">{children}</span>
      {onDismiss && <button onClick={onDismiss}><X size={13} /></button>}
    </div>
  )
}

function StatBox({ label, value, color = 'text-white', sub }) {
  return (
    <div className="card text-center py-4">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      <p className="text-xs text-slate-400 mt-1">{label}</p>
    </div>
  )
}

function SectionCard({ title, icon: Icon, iconColor = 'text-brand-400', children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card">
      <button onClick={() => setOpen(v => !v)} className="flex items-center gap-2 w-full text-left mb-0">
        <Icon size={15} className={iconColor} />
        <span className="font-semibold text-white text-sm flex-1">{title}</span>
        {open ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
      </button>
      {open && <div className="mt-4 space-y-4">{children}</div>}
    </div>
  )
}

const OP_OPTIONS = [
  { id: 'ratio',    label: '÷  Ratio',       desc: 'A ÷ B  — captures relative scale',     needs2: true  },
  { id: 'multiply', label: '×  Multiply',    desc: 'A × B  — interaction term',            needs2: true  },
  { id: 'add',      label: '+  Add',         desc: 'A + B  — combined magnitude',           needs2: true  },
  { id: 'subtract', label: '−  Subtract',   desc: 'A − B  — difference',                   needs2: true  },
  { id: 'log',      label: 'log  Log(1+x)', desc: 'log(1 + A)  — reduces right skew',     needs2: false },
  { id: 'square',   label: 'x²  Square',    desc: 'A²  — amplifies magnitude',             needs2: false },
]

// ─── Main page ────────────────────────────────────────────────────────────────
export default function FeatureEngineering() {
  // ── Global state ───────────────────────────────────────────────────
  const [status,      setStatus]      = useState(null)   // /status response
  const [analysis,    setAnalysis]    = useState(null)   // /analyze response
  const [preview,     setPreview]     = useState(null)   // /preview response
  const [loading,     setLoading]     = useState(true)
  const [flash,       setFlash]       = useState(null)
  const [actionBusy,  setActionBusy]  = useState(false)

  // ── Formula panel state ────────────────────────────────────────────
  const [fName, setFName]         = useState('')
  const [fOp,   setFOp]           = useState('ratio')
  const [fColA, setFColA]         = useState('')
  const [fColB, setFColB]         = useState('')

  // ── Selection panel state ──────────────────────────────────────────
  const [selK,  setSelK]          = useState(10)

  // ── Correlation panel ──────────────────────────────────────────────
  const [corrData, setCorrData]   = useState(null)
  const [corrLoading, setCorrLoading] = useState(false)
  const [showCorr, setShowCorr]   = useState(false)

  // ── Preview panel ──────────────────────────────────────────────────
  const [showPreview, setShowPreview] = useState(false)

  // ─────────────────────────────────────────────────────────────────
  const loadAll = useCallback(async () => {
    setLoading(true)
    try {
      const [statusRes, analyzeRes, previewRes] = await Promise.allSettled([
        api.get('/features/status'),
        api.get('/features/analyze'),
        api.get('/features/preview'),
      ])
      if (statusRes.status  === 'fulfilled') setStatus(statusRes.value.data)
      if (analyzeRes.status === 'fulfilled') {
        setAnalysis(analyzeRes.value.data)
        const nc = analyzeRes.value.data.numeric_cols || []
        if (nc.length > 0) { setFColA(nc[0]); setFColB(nc[1] || nc[0]) }
      }
      if (previewRes.status === 'fulfilled') setPreview(previewRes.value.data)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadAll() }, [loadAll])

  const showFlash = (type, msg) => setFlash({ type, msg })

  // ── Actions ──────────────────────────────────────────────────────
  const handleFormula = async () => {
    if (!fName.trim()) return showFlash('error', 'Feature name is required.')
    const op = OP_OPTIONS.find(o => o.id === fOp)
    if (op?.needs2 && !fColB) return showFlash('error', 'This operation requires a second column.')
    setActionBusy(true)
    setFlash(null)
    try {
      const res = await api.post('/features/formula', {
        name: fName.trim().replace(/\s+/g, '_'),
        col_a: fColA,
        col_b: op?.needs2 ? fColB : null,
        operation: fOp,
      })
      showFlash('success', res.data.message)
      if (res.data.warnings?.length) {
        setTimeout(() => showFlash('warn', res.data.warnings.join(' ')), 1000)
      }
      setFName('')
      await loadAll()
    } catch (e) {
      showFlash('error', e.response?.data?.detail || 'Formula feature failed.')
    } finally { setActionBusy(false) }
  }

  const handleSelect = async () => {
    setActionBusy(true)
    setFlash(null)
    try {
      const res = await api.post('/features/select', { method: 'kbest', k: selK })
      showFlash('success', res.data.message)
      await loadAll()
    } catch (e) {
      showFlash('error', e.response?.data?.detail || 'Selection failed.')
    } finally { setActionBusy(false) }
  }

  const handleUndo = async () => {
    setActionBusy(true)
    setFlash(null)
    try {
      const res = await api.post('/features/undo')
      showFlash('success', res.data.message)
      await loadAll()
    } catch (e) {
      showFlash('error', e.response?.data?.detail || 'Undo failed.')
    } finally { setActionBusy(false) }
  }

  const handleReset = async () => {
    if (!window.confirm('Reset all feature engineering? This cannot be undone.')) return
    setActionBusy(true)
    setFlash(null)
    try {
      const res = await api.post('/features/reset')
      showFlash('warn', res.data.message)
      await loadAll()
    } catch (e) {
      showFlash('error', e.response?.data?.detail || 'Reset failed.')
    } finally { setActionBusy(false) }
  }

  const loadCorr = async () => {
    setCorrLoading(true)
    try {
      const res = await api.get('/features/correlation')
      setCorrData(res.data)
      setShowCorr(true)
    } catch (e) {
      showFlash('error', 'Correlation calculation failed.')
    } finally { setCorrLoading(false) }
  }

  // ── Derived ──────────────────────────────────────────────────────
  const selectedOp      = OP_OPTIONS.find(o => o.id === fOp)
  const needsSecondCol  = selectedOp?.needs2 ?? false
  const numCols         = analysis?.numeric_cols || []
  const actionCount     = preview?.action_log?.length || 0
  const hasSplit        = status?.split_done !== false && status !== null
  const hasActions      = actionCount > 0

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div><h2 className="section-title">Feature Engineering</h2></div>
        <div className="card flex items-center gap-3 text-brand-300">
          <div className="spinner" /> Loading post-split data…
        </div>
      </div>
    )
  }

  if (!hasSplit) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div><h2 className="section-title">Feature Engineering</h2></div>
        <div className="card border-warn-500/30 bg-warn-500/8 flex items-center gap-3">
          <AlertTriangle size={20} className="text-warn-400 shrink-0" />
          <div>
            <p className="font-semibold text-white text-sm">Split Data step required</p>
            <p className="text-xs text-slate-400 mt-0.5">
              Please complete the <strong>Split Data</strong> step first.
              Feature Engineering runs post-split and fits selection transformers on X_train only.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="section-title">Feature Engineering</h2>
          <p className="section-subtitle">
            Create and select features using <strong>train data only</strong>. No test data is ever used for fitting.
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button onClick={loadAll} disabled={loading} className="btn-secondary text-sm flex items-center gap-2">
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
          {hasActions && (
            <button onClick={handleUndo} disabled={actionBusy} className="btn-secondary text-sm flex items-center gap-2">
              <RotateCcw size={13} /> Undo Last
            </button>
          )}
          <button onClick={handleReset} disabled={actionBusy} className="btn-secondary text-sm flex items-center gap-2 text-danger-400 border-danger-500/30">
            <Trash2 size={13} /> Reset
          </button>
        </div>
      </div>

      {/* ── Golden Rule Banner ── */}
      <div className="flex items-start gap-3 rounded-xl border bg-brand-500/8 border-brand-500/20 px-4 py-3">
        <Lock size={15} className="text-brand-400 mt-0.5 shrink-0" />
        <div className="text-sm text-brand-300 space-y-1">
          <p className="font-semibold text-brand-200">Golden Rule — No Test Data Leakage</p>
          <ul className="text-xs text-brand-300/80 list-disc list-inside space-y-0.5">
            <li>Any transformation that <strong>learns</strong> from data → fit ONLY on X_train</li>
            <li>Formula features (ratio, log, etc.) apply identically to both sets — no fitting needed</li>
            <li>Feature selection scores computed on (X_train, y_train) → mask applied to both</li>
            <li>X_test is never used to compute statistics, mappings, or importance scores</li>
          </ul>
        </div>
      </div>

      {/* ── Dataset stats ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatBox label="Train Rows"   value={status?.train_rows?.toLocaleString()} color="text-emerald-400" />
        <StatBox label="Test Rows"    value={status?.test_rows?.toLocaleString()}  color="text-rose-400" />
        <StatBox label="Features Now" value={status?.n_features}                   color="text-brand-400" />
        <StatBox label="Actions Done" value={actionCount}                          color={actionCount > 0 ? 'text-accent-400' : 'text-slate-500'} />
      </div>

      {/* ── Flash ── */}
      {flash && <Alert type={flash.type} onDismiss={() => setFlash(null)}>{flash.msg}</Alert>}

      {/* ── Pre-scan warnings ── */}
      {analysis?.corr_warnings?.filter(w => w.severe).map((w, i) => (
        <Alert key={i} type="warn">
          <strong>Highly correlated features detected:</strong> '{w.col_a}' and '{w.col_b}' have correlation {w.correlation} (&gt;0.95). Consider dropping one — this may cause redundancy and instability.
        </Alert>
      ))}
      {analysis?.target_corr?.map((t, i) => (
        <Alert key={i} type="error">
          <strong>Possible target leakage:</strong> Feature '{t.col}' is almost perfectly correlated with the target (r={t.correlation}). This may cause unrealistic model performance.
        </Alert>
      ))}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ══════════════════════════════════════════
            SECTION 1 — Formula Features
        ══════════════════════════════════════════ */}
        <SectionCard title="Formula Features (Leakage-Safe)" icon={FlaskConical} iconColor="text-brand-400">
          <div className="flex items-start gap-2 text-xs text-blue-300 bg-blue-500/8 border border-blue-500/20 rounded-xl px-3 py-2">
            <Info size={12} className="shrink-0 mt-0.5" />
            Formula features are deterministic — they don't learn from data, so they can safely be computed identically on both train and test without any fitting step.
          </div>

          {/* Operation selector */}
          <div>
            <label className="block text-xs text-slate-400 mb-2">Operation</label>
            <div className="grid grid-cols-2 gap-2">
              {OP_OPTIONS.map(op => (
                <button
                  key={op.id}
                  onClick={() => setFOp(op.id)}
                  className={`text-left rounded-xl border px-3 py-2.5 transition-all
                    ${fOp === op.id
                      ? 'border-brand-500/60 bg-brand-600/15 text-white'
                      : 'border-surface-600 text-slate-400 hover:border-surface-500'}`}
                >
                  <p className="text-xs font-semibold">{op.label}</p>
                  <p className="text-[10px] text-slate-500 mt-0.5">{op.desc}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Column selectors */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Column A</label>
              <select value={fColA} onChange={e => setFColA(e.target.value)} className="input w-full text-sm">
                {numCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            {needsSecondCol && (
              <div>
                <label className="block text-xs text-slate-400 mb-1">Column B</label>
                <select value={fColB} onChange={e => setFColB(e.target.value)} className="input w-full text-sm">
                  {numCols.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            )}
          </div>

          {/* Feature name */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">New Feature Name</label>
            <input
              className="input w-full"
              placeholder={`e.g. ${fColA}_${fOp}_${fColB || ''}`}
              value={fName}
              onChange={e => setFName(e.target.value.replace(/\s+/g, '_'))}
            />
          </div>

          <button
            onClick={handleFormula}
            disabled={actionBusy || !fName.trim() || !fColA}
            className="btn-primary w-full justify-center"
          >
            {actionBusy ? <><div className="spinner" /> Applying…</> : <><Zap size={14} /> Apply Feature Engineering</>}
          </button>
        </SectionCard>

        {/* ══════════════════════════════════════════
            SECTION 2 — Feature Selection
        ══════════════════════════════════════════ */}
        <SectionCard title="Feature Selection (Train-Only)" icon={TrendingUp} iconColor="text-accent-400">
          <div className="flex items-start gap-2 text-xs text-blue-300 bg-blue-500/8 border border-blue-500/20 rounded-xl px-3 py-2">
            <Info size={12} className="shrink-0 mt-0.5" />
            SelectKBest is fitted on (X_train, y_train) only. The selected feature mask is then applied identically to X_test — X_test never influences which features are kept.
          </div>

          {/* Current features list */}
          {numCols.length > 0 && (
            <div>
              <label className="block text-xs text-slate-400 mb-2">Current Numeric Features ({numCols.length})</label>
              <div className="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto p-2 bg-surface-800 rounded-xl border border-surface-700">
                {numCols.map(c => (
                  <span key={c} className="px-2 py-0.5 bg-surface-700 text-slate-300 rounded-lg text-xs font-mono border border-surface-600">{c}</span>
                ))}
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs text-slate-400 mb-1">Top K features to keep</label>
            <div className="flex items-center gap-3">
              <input
                type="range" min={1} max={Math.max(numCols.length, 20)} step={1}
                value={selK} onChange={e => setSelK(Number(e.target.value))}
                className="flex-1 accent-brand-500"
              />
              <span className="text-white font-bold w-8 text-center">{selK}</span>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Keeping top {selK} of {numCols.length} numeric features by F-score (SelectKBest).
            </p>
          </div>

          <button
            onClick={handleSelect}
            disabled={actionBusy || numCols.length === 0}
            className="btn-primary w-full justify-center"
          >
            {actionBusy ? <><div className="spinner" /> Selecting…</> : <><Play size={14} /> Apply Feature Engineering</>}
          </button>
        </SectionCard>
      </div>

      {/* ══════════════════════════════════════════
          SECTION 3 — Correlation / Redundancy Check
      ══════════════════════════════════════════ */}
      <div className="card">
        <div className="flex items-center gap-2 mb-0">
          <BarChart2 size={15} className="text-purple-400" />
          <span className="font-semibold text-white text-sm flex-1">Collinearity / Redundancy Check (Train-Only)</span>
          <button
            onClick={loadCorr}
            disabled={corrLoading}
            className="btn-secondary text-xs flex items-center gap-1.5"
          >
            <RefreshCw size={12} className={corrLoading ? 'animate-spin' : ''} />
            {corrLoading ? 'Computing…' : 'Run Correlation Check'}
          </button>
          {corrData && (
            <button onClick={() => setShowCorr(v => !v)} className="btn-secondary text-xs">
              {showCorr ? 'Hide' : 'Show'} Results
            </button>
          )}
        </div>

        {showCorr && corrData && (
          <div className="mt-4 space-y-3 animate-slide-up">
            <p className="text-xs text-slate-500">
              Computed on X_train only. Only pairs with |correlation| &gt; 0.90 are shown.
            </p>
            {corrData.pairs?.length === 0 && (
              <Alert type="success">No highly correlated feature pairs detected (threshold: 0.90).</Alert>
            )}
            {corrData.pairs?.length > 0 && (
              <>
                <Alert type="warn">
                  {corrData.pairs.filter(p => p.severe).length} pair(s) with correlation &gt; 0.95 detected.
                  Consider dropping one column from each high-correlation pair using the Data Cleaning step or manually.
                  <strong> No auto-drop is performed — you decide.</strong>
                </Alert>
                <div className="table-wrapper">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature A</th>
                        <th>Feature B</th>
                        <th>Correlation</th>
                        <th>Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {corrData.pairs.map((p, i) => (
                        <tr key={i}>
                          <td className="font-mono text-brand-300 text-xs">{p.col_a}</td>
                          <td className="font-mono text-brand-300 text-xs">{p.col_b}</td>
                          <td className="font-bold text-xs"
                            style={{ color: p.corr > 0.95 ? '#f87171' : '#fb923c' }}>
                            {p.corr}
                          </td>
                          <td>
                            <span className={`badge text-xs border ${p.severe
                              ? 'text-danger-400 bg-danger-500/15 border-danger-500/30'
                              : 'text-warn-400 bg-warn-500/10 border-warn-500/30'}`}>
                              {p.severe ? 'High (>0.95)' : 'Medium (>0.90)'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* ══════════════════════════════════════════
          SECTION 4 — Column Analysis
      ══════════════════════════════════════════ */}
      {analysis && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Layers size={15} className="text-brand-400" />
            <span className="font-semibold text-white text-sm">X_train Column Summary ({analysis.n_features} features)</span>
          </div>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Type</th>
                  <th>Unique</th>
                  <th>Missing</th>
                  <th>Mean</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {(analysis.col_summary || []).map(col => (
                  <tr key={col.name}>
                    <td className="font-mono text-slate-300 text-xs">{col.name}</td>
                    <td>
                      <span className={`badge text-xs border ${col.numeric
                        ? 'text-blue-400 bg-blue-500/10 border-blue-500/20'
                        : 'text-amber-400 bg-amber-500/10 border-amber-500/20'}`}>
                        {col.numeric ? 'Numeric' : 'Object'}
                      </span>
                    </td>
                    <td className="text-slate-400 text-xs">{col.n_unique}</td>
                    <td className="text-xs" style={{ color: col.missing > 0 ? '#fb923c' : '#64748b' }}>
                      {col.missing}
                    </td>
                    <td className="text-slate-400 text-xs">{col.mean ?? '—'}</td>
                    <td>
                      {col.constant && (
                        <span className="badge text-xs border text-danger-400 bg-danger-500/10 border-danger-500/20">Constant</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════
          SECTION 5 — Action Log + Preview
      ══════════════════════════════════════════ */}
      {preview && (
        <div className="card">
          <div className="flex items-center gap-2 mb-0">
            <Eye size={15} className="text-brand-400" />
            <span className="font-semibold text-white text-sm flex-1">
              Action Log & X_train Preview
            </span>
            <button onClick={() => setShowPreview(v => !v)} className="btn-secondary text-xs">
              {showPreview ? 'Hide' : 'Show'} Preview
            </button>
          </div>

          {showPreview && (
            <div className="mt-4 space-y-4 animate-slide-up">
              {/* Action log */}
              {preview.action_log?.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wider">Actions Applied</p>
                  <div className="space-y-1">
                    {preview.action_log.map((a, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs text-slate-300 bg-surface-700/50 rounded-lg px-3 py-2 border border-surface-600">
                        <CheckCircle size={11} className="text-accent-400 shrink-0" />
                        <span className="font-mono text-brand-300">[{a.type}]</span>
                        {a.name && <span>Created '{a.name}'</span>}
                        {a.op && <span className="text-slate-500">({a.op})</span>}
                        {a.method && <span>SelectKBest k={a.k}</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Current features */}
              <div>
                <p className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wider">
                  Current Features in X_train ({preview.n_features})
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {preview.features?.map(f => (
                    <span key={f} className="px-2 py-0.5 bg-surface-700 rounded-lg text-xs font-mono text-slate-300 border border-surface-600">{f}</span>
                  ))}
                </div>
              </div>

              {/* Head preview */}
              {preview.head?.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wider">X_train Sample (5 rows)</p>
                  <div className="table-wrapper">
                    <table className="data-table">
                      <thead>
                        <tr>{Object.keys(preview.head[0]).map(k => <th key={k}>{k}</th>)}</tr>
                      </thead>
                      <tbody>
                        {preview.head.map((row, i) => (
                          <tr key={i}>
                            {Object.values(row).map((v, j) => (
                              <td key={j} className="text-xs font-mono text-slate-300">{v}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Suggested formula features ── */}
      {analysis?.suggestions?.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Zap size={15} className="text-yellow-400" />
            <span className="font-semibold text-white text-sm">Suggested Formula Features</span>
            <span className="text-xs text-slate-500 font-normal">(click to pre-fill the formula builder above)</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {analysis.suggestions.slice(0, 8).map((s, i) => (
              <button
                key={i}
                onClick={() => {
                  setFColA(s.col_a)
                  if (s.col_b) setFColB(s.col_b)
                  setFOp(s.operation)
                  setFName(s.suggested_name)
                  window.scrollTo({ top: 0, behavior: 'smooth' })
                }}
                className="text-left rounded-xl border border-surface-600 bg-surface-700/30 hover:border-brand-500/40 hover:bg-brand-500/5 px-3 py-2.5 transition-all"
              >
                <p className="text-xs font-semibold text-white font-mono">{s.suggested_name}</p>
                <p className="text-[10px] text-slate-500 mt-0.5">{s.reason}</p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Next step ── */}
      <div className="flex items-center gap-2 text-xs text-accent-400 bg-accent-500/8 border border-accent-500/20 rounded-xl px-4 py-3">
        <CheckCircle size={13} />
        When done, proceed to <strong>Class Imbalance</strong> — resampling is applied to X_train only inside the training pipeline.
      </div>
    </div>
  )
}
