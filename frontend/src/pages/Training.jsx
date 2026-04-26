import React, { useState, useEffect, useCallback } from 'react'
import {
  Play, CheckCircle, AlertCircle, Terminal,
  ChevronRight, Info, Zap, Scale, Trophy,
  Layers, Database, BarChart2, TrendingUp,
  AlertTriangle, Cpu, Clock, Hash, RotateCcw,
} from 'lucide-react'
import {
  getAvailableModels, trainModel, trainMultiModel,
} from '../services/api'
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 60_000 })

// ── Constants ─────────────────────────────────────────────────────────────────
const TECHNIQUE_LABEL = {
  smote:        '⬆ SMOTE',
  smotenc:      '⬆ SMOTENC',
  adasyn:       '⬆ ADASYN',
  undersample:  '⬇ Random Undersampling',
  smoteenn:     '⚡ SMOTE + ENN',
  smotetomek:   '⚡ SMOTE + Tomek',
  class_weight: '⚖ Class Weighting',
  none:         '➖ No Balancing',
}

const CLS_METRICS = [
  { key: 'accuracy',  label: 'Accuracy',  fmt: v => `${(v * 100).toFixed(1)}%`, good: v => v >= 0.75 },
  { key: 'precision', label: 'Precision', fmt: v => `${(v * 100).toFixed(1)}%`, good: v => v >= 0.65 },
  { key: 'recall',    label: 'Recall',    fmt: v => `${(v * 100).toFixed(1)}%`, good: v => v >= 0.65 },
  { key: 'f1',        label: 'F1',        fmt: v => `${(v * 100).toFixed(1)}%`, good: v => v >= 0.65 },
  { key: 'roc_auc',   label: 'ROC-AUC',  fmt: v => v != null ? `${(v * 100).toFixed(1)}%` : '—', good: v => v != null && v >= 0.70 },
]

const REG_METRICS = [
  { key: 'r2',   label: 'R²',   fmt: v => v.toFixed(4), good: v => v >= 0.7 },
  { key: 'rmse', label: 'RMSE', fmt: v => v.toFixed(4), good: v => v <= 0.5, lower_is_better: true },
  { key: 'mae',  label: 'MAE',  fmt: v => v.toFixed(4), good: v => v <= 0.5, lower_is_better: true },
]

const STATUS_COLOR = {
  idle:     'bg-slate-600',
  training: 'bg-brand-400 animate-pulse',
  done:     'bg-accent-400',
  error:    'bg-danger-400',
}

// ── Small helpers ─────────────────────────────────────────────────────────────
function MetricPill({ label, value, good, lower_is_better }) {
  const isGood = good ? good(value) : null
  const color  = isGood === null ? '#94a3b8'
               : isGood ? '#34d399' : '#f97316'
  return (
    <div className="flex flex-col items-center gap-0.5 min-w-0">
      <span className="text-[10px] text-slate-500 font-medium">{label}</span>
      <span className="text-sm font-bold" style={{ color }}>{value}</span>
    </div>
  )
}

function PredDistBar({ dist }) {
  if (!dist) return null
  const total = Object.values(dist).reduce((a, b) => a + b, 0)
  const colors = ['#6366f1', '#34d399', '#f59e0b', '#f97316', '#c084fc']
  return (
    <div className="mt-2">
      <p className="text-[10px] text-slate-500 mb-1">Prediction Distribution</p>
      <div className="flex rounded-full overflow-hidden h-2">
        {Object.entries(dist).map(([cls, cnt], i) => (
          <div key={cls} title={`${cls}: ${cnt}`}
            style={{ width: `${(cnt / total) * 100}%`, background: colors[i % colors.length] }} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
        {Object.entries(dist).map(([cls, cnt], i) => (
          <span key={cls} className="text-[10px] text-slate-500">
            <span style={{ color: colors[i % colors.length] }}>■</span> {cls}: {cnt}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Balancing Distribution Card ───────────────────────────────────────────────
function BalancingDistCard({ datasetInfo, techniqueLabel }) {
  const before = datasetInfo?.before_class_dist || {}
  const after  = datasetInfo?.after_class_dist  || {}
  const applied = datasetInfo?.balancing_applied
  const technique = datasetInfo?.balancing_used

  if (!applied || Object.keys(before).length === 0) return null

  const COLORS = ['#6366f1','#34d399','#f59e0b','#f97316','#c084fc','#38bdf8']
  const classes  = [...new Set([...Object.keys(before), ...Object.keys(after)])]
  const clsColor = Object.fromEntries(classes.map((c, i) => [c, COLORS[i % COLORS.length]]))

  const totalBefore = Object.values(before).reduce((a, b) => a + b, 0)
  const totalAfter  = Object.values(after).reduce((a, b) => a + b, 0)
  const deltaRows   = totalAfter - totalBefore

  const DistBar = ({ dist, total, label }) => (
    <div>
      <p className="text-[10px] text-slate-500 mb-1 font-medium uppercase tracking-wider">{label}</p>
      <div className="flex rounded-full overflow-hidden h-3 mb-1.5">
        {classes.map(cls => {
          const cnt = dist[cls] || 0
          return (
            <div
              key={cls}
              title={`${cls}: ${cnt} (${total > 0 ? ((cnt/total)*100).toFixed(1) : 0}%)`}
              style={{ width: `${total > 0 ? (cnt/total)*100 : 0}%`, background: clsColor[cls] }}
            />
          )
        })}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5">
        {classes.map(cls => {
          const cnt = dist[cls] || 0
          const pct = total > 0 ? ((cnt / total) * 100).toFixed(1) : '0.0'
          return (
            <span key={cls} className="text-[11px] text-slate-400">
              <span style={{ color: clsColor[cls] }}>■</span>{' '}
              <span className="text-slate-300 font-medium">{cls}</span>:{' '}
              {cnt.toLocaleString()} <span className="text-slate-600">({pct}%)</span>
            </span>
          )
        })}
        <span className="text-[11px] text-slate-600 ml-auto">{total.toLocaleString()} rows</span>
      </div>
    </div>
  )

  return (
    <div className="card border-emerald-500/25 bg-emerald-500/5 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <p className="text-sm font-bold text-white">Balancing Applied — Class Distribution</p>
        <span className="ml-auto text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
          {techniqueLabel || technique}
        </span>
      </div>

      {/* Before / After bars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface-700/40 rounded-xl border border-surface-600 p-3">
          <DistBar dist={before} total={totalBefore} label="Before (Original Training Set)" />
        </div>
        <div className="bg-emerald-500/8 rounded-xl border border-emerald-500/20 p-3">
          <DistBar dist={after} total={totalAfter} label="After (Resampled Training Set)" />
        </div>
      </div>

      {/* Delta summary */}
      <div className="flex items-center gap-4 text-xs flex-wrap">
        <span className="text-slate-500">
          Training rows: <span className="text-slate-300 font-semibold">{totalBefore.toLocaleString()}</span>
          {' → '}
          <span className="text-emerald-400 font-semibold">{totalAfter.toLocaleString()}</span>
        </span>
        <span className={deltaRows > 0 ? 'text-emerald-400 font-semibold' : deltaRows < 0 ? 'text-orange-400 font-semibold' : 'text-slate-500'}>
          {deltaRows > 0 ? `+${deltaRows.toLocaleString()} rows added (oversampling)` :
           deltaRows < 0 ? `${deltaRows.toLocaleString()} rows removed (undersampling)` :
           'Row count unchanged (class weighting)'}
        </span>
        <span className="ml-auto text-slate-600 text-[10px]">X_train only · X_test unchanged</span>
      </div>
    </div>
  )
}


function ModelResultCard({ result, isBest, taskType }) {
  const metrics    = CLS_METRICS
  const regMetrics = REG_METRICS
  const isSuccess  = result.status === 'success'
  const isCls      = taskType === 'classification'
  const metricSet  = isCls ? metrics : regMetrics

  return (
    <div className="rounded-xl border transition-all duration-200"
      style={{
        borderColor: isBest ? '#f59e0b' : isSuccess ? 'rgba(99,102,241,0.3)' : 'rgba(239,68,68,0.3)',
        background:  isBest ? 'rgba(245,158,11,0.05)' : 'rgba(255,255,255,0.02)',
        boxShadow:   isBest ? '0 0 0 2px rgba(245,158,11,0.2)' : undefined,
      }}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-4 pb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center"
            style={{ background: isBest ? 'rgba(245,158,11,0.2)' : 'rgba(99,102,241,0.15)' }}>
            <Cpu size={14} style={{ color: isBest ? '#f59e0b' : '#818cf8' }} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <p className="font-bold text-white text-sm">{result.name}</p>
              {isBest && (
                <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full"
                  style={{ background: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)' }}>
                  <Trophy size={9} /> Best Model
                </span>
              )}
            </div>
            {isSuccess && (
              <p className="text-[10px] text-slate-500 mt-0.5">
                Train: {result.train_samples?.toLocaleString()} · Test: {result.test_samples?.toLocaleString()}
              </p>
            )}
          </div>
        </div>
        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
          isSuccess ? 'text-accent-300 bg-accent-500/10' : 'text-danger-300 bg-danger-500/10'
        }`}>
          {isSuccess ? 'Success' : result.status === 'failed' ? 'Failed' : 'Error'}
        </span>
      </div>

      {/* Failed / error */}
      {!isSuccess && result.error && (
        <div className="px-4 pb-4 text-xs text-danger-400 flex items-start gap-2">
          <AlertCircle size={12} className="mt-0.5 shrink-0" />
          {result.error}
        </div>
      )}

      {/* Metrics */}
      {isSuccess && (
        <div className="px-4 pb-4 space-y-3">
          <div className="flex justify-between gap-2 flex-wrap">
            {metricSet.map(m => {
              const val = result.metrics?.[m.key]
              if (val == null) return null
              return (
                <MetricPill key={m.key} label={m.label}
                  value={m.fmt(val)} good={m.good}
                  lower_is_better={m.lower_is_better} />
              )
            })}
          </div>

          {/* Degenerate warning */}
          {result.warning && (
            <div className="flex items-start gap-2 rounded-lg border border-warn-500/25 bg-warn-500/8 px-3 py-2">
              <AlertTriangle size={13} className="text-warn-400 mt-0.5 shrink-0" />
              <p className="text-[11px] text-warn-300">{result.warning}</p>
            </div>
          )}

          {/* Prediction distribution (classification) */}
          {isCls && result.pred_distribution && (
            <PredDistBar dist={result.pred_distribution} />
          )}
        </div>
      )}
    </div>
  )
}

// ── Comparison table (multi-model) ────────────────────────────────────────────
function ComparisonTable({ results, bestName, taskType }) {
  const isCls     = taskType === 'classification'
  const colKeys   = isCls
    ? ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    : ['r2', 'rmse', 'mae']
  const colLabels = isCls
    ? ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    : ['R²', 'RMSE', 'MAE']
  const success   = results.filter(r => r.status === 'success')
  if (success.length < 2) return null

  // Best value per column
  const bestVal = {}
  colKeys.forEach((k, i) => {
    const vals = success.map(r => r.metrics?.[k]).filter(v => v != null)
    if (!vals.length) return
    const lowerBetter = ['rmse', 'mae'].includes(k)
    bestVal[k] = lowerBetter ? Math.min(...vals) : Math.max(...vals)
  })

  const fmt = (k, v) => {
    if (v == null) return '—'
    if (['accuracy','precision','recall','f1','roc_auc'].includes(k)) return `${(v*100).toFixed(1)}%`
    return v.toFixed(4)
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <BarChart2 size={15} className="text-brand-400" />
        <h4 className="text-sm font-bold text-white">Model Comparison</h4>
      </div>
      <div className="overflow-x-auto rounded-xl border border-surface-700">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-surface-700 bg-surface-800/60">
              <th className="px-4 py-2.5 text-left text-slate-400 font-semibold">Model</th>
              {colLabels.map(l => (
                <th key={l} className="px-3 py-2.5 text-center text-slate-400 font-semibold">{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {success.map((r, i) => {
              const isBest = r.name === bestName
              return (
                <tr key={r.name}
                  className={`border-b border-surface-700/50 transition-colors ${isBest ? 'bg-amber-500/5' : i % 2 === 0 ? 'bg-surface-800/20' : ''}`}>
                  <td className="px-4 py-2.5 font-semibold text-white whitespace-nowrap">
                    <span className="flex items-center gap-1.5">
                      {isBest && <Trophy size={11} className="text-amber-400 shrink-0" />}
                      {r.name}
                    </span>
                  </td>
                  {colKeys.map(k => {
                    const val   = r.metrics?.[k]
                    const isTop = val != null && val === bestVal[k]
                    return (
                      <td key={k} className="px-3 py-2.5 text-center">
                        <span className="font-mono font-bold"
                          style={{ color: isTop ? '#f59e0b' : val != null ? '#e2e8f0' : '#475569' }}>
                          {fmt(k, val)}
                        </span>
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Dataset + config summary banner ──────────────────────────────────────────
function ConfigSummary({ models, taskType, datasetInfo, technique, balancingConfig }) {
  const items = [
    { icon: Database,   label: 'Dataset',  value: datasetInfo ? `${datasetInfo.total_rows?.toLocaleString()} rows` : '—' },
    { icon: Cpu,        label: 'Models',   value: models?.length ? models.join(', ') : '—' },
    { icon: TrendingUp, label: 'Task',     value: taskType === 'classification' ? '🏷️ Classification' : taskType === 'regression' ? '📈 Regression' : '—' },
    { icon: Scale,      label: 'Balancing',value: TECHNIQUE_LABEL[technique] || 'None' },
  ]
  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800/50 px-5 py-4 flex flex-wrap gap-6">
      {items.map(({ icon: Icon, label, value }) => (
        <div key={label} className="flex items-center gap-2 min-w-0">
          <Icon size={13} className="text-slate-500 shrink-0" />
          <span className="text-xs text-slate-500">{label}:</span>
          <span className="text-xs font-semibold text-slate-200 truncate">{value}</span>
        </div>
      ))}
    </div>
  )
}

// ── Best-model banner ─────────────────────────────────────────────────────────
function BestModelBanner({ multiResult, taskType, onNavigate }) {
  if (!multiResult?.best_model) return null
  const isCls    = taskType === 'classification'
  const metricLbl = multiResult.best_metric?.name?.toUpperCase() || (isCls ? 'F1' : 'RMSE')
  const metricVal = multiResult.best_metric?.value
  return (
    <div className="rounded-xl border border-accent-500/30 bg-accent-500/8 px-5 py-4 flex items-center justify-between gap-4 flex-wrap">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-amber-500/20">
          <Trophy size={20} className="text-amber-400" />
        </div>
        <div>
          <p className="font-bold text-white">
            Best Model: <span className="text-amber-300">{multiResult.best_model}</span>
          </p>
          <p className="text-xs text-slate-400 mt-0.5">
            {multiResult.selection_criterion} · {metricLbl}:{' '}
            <span className="text-accent-300 font-mono font-semibold">
              {metricVal != null ? `${(metricVal * 100).toFixed(1)}%` : '—'}
            </span>
            &nbsp;· Saved to session for Evaluation →
          </p>
        </div>
      </div>
      <button onClick={() => onNavigate?.('evaluation')} className="btn-primary text-sm shrink-0">
        Go to Evaluation <ChevronRight size={14} />
      </button>
    </div>
  )
}

/* ─── Main Training Page ─────────────────────────────────────────────────── */
export default function Training({ onNavigate, selectedModels }) {

  // ── Core state ───────────────────────────────────────────────
  const [availableModels, setAvailableModels] = useState([])
  const [taskType, setTaskType]               = useState('')
  const [config, setConfig]                   = useState({ model_name: '', test_size: 0.2, random_state: 42 })

  // Post-split transformation config (sent to training route)
  const [scalerType,     setScalerType]     = useState('standard')  // standard | minmax | none
  const [applySkewness,  setApplySkewness]  = useState(true)

  // Models to train — prop from Model Selection (single string or array)
  // Falls back to manual select-one if prop is absent
  const propModels = selectedModels
    ? (Array.isArray(selectedModels) ? selectedModels : [selectedModels]).filter(Boolean)
    : []
  const [manualModel, setManualModel] = useState('')   // used only when no prop

  // Balancing from Class Imbalance step
  const [technique,       setTechnique]       = useState('none')
  const [techLoading,     setTechLoading]     = useState(true)
  const [balancingConfig, setBalancingConfig] = useState(null)

  // Training state
  const [status,      setStatus]      = useState('idle')
  const [progress,    setProgress]    = useState(0)
  // Multi-model result
  const [multiResult, setMultiResult] = useState(null)
  // Single-model legacy result
  const [singleResult, setSingleResult] = useState(null)
  const [error,  setError]   = useState(null)
  const [logs,   setLogs]    = useState([])
  const [currentModel, setCurrentModel] = useState('')

  // ── Restore last training result from sessionStorage (survives navigation) ──
  const CACHE_KEY = 'training_result_cache'
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(CACHE_KEY)
      if (!raw) return
      const cached = JSON.parse(raw)
      if (cached.multiResult)  setMultiResult(cached.multiResult)
      if (cached.singleResult) setSingleResult(cached.singleResult)
      if (cached.logs?.length) setLogs(cached.logs)
      if (cached.status === 'done') setStatus('done')
    } catch (_) { /* corrupt cache — ignore */ }
  }, []) // run once on mount

  // ── Persist training result on change ────────────────────────────────
  useEffect(() => {
    if (status !== 'done') return
    try {
      sessionStorage.setItem(CACHE_KEY, JSON.stringify({
        multiResult, singleResult, logs, status,
      }))
    } catch (_) { /* storage full — ignore */ }
  }, [multiResult, singleResult, logs, status])

  const clearCache = () => {
    sessionStorage.removeItem(CACHE_KEY)
    setMultiResult(null)
    setSingleResult(null)
    setLogs([])
    setStatus('idle')
    setProgress(0)
    setError(null)
  }

  // ── Load available models + task type ───────────────────────────────
  useEffect(() => {
    getAvailableModels()
      .then(res => {
        const names = Object.keys(res.data.models)
        setAvailableModels(names)
        setTaskType(res.data.task_type || '')
        setManualModel(prev => prev || names[0] || '')
        setConfig(c => ({ ...c, model_name: names[0] || '' }))
      })
      .catch(() => {})
  }, [])

  // ── Load balancing config ────────────────────────────────────────────
  useEffect(() => {
    api.get('/imbalance/status')
      .then(res => {
        const d = res.data
        setTechnique(d.technique || 'none')
        setBalancingConfig(d.config || null)
      })
      .catch(() => setTechnique('none'))
      .finally(() => setTechLoading(false))
  }, [])

  // ── Derived: which models will be trained ────────────────────────────
  const modelsToTrain = propModels.length > 0
    ? propModels
    : manualModel ? [manualModel] : []

  const isMulti = modelsToTrain.length > 1

  // ── Progress ticker ──────────────────────────────────────────────────
  const startTicker = useCallback(() => {
    setProgress(0)
    const timer = setInterval(() => {
      setProgress(p => {
        if (p >= 90) { clearInterval(timer); return 90 }
        return p + Math.random() * (isMulti ? 5 : 12)
      })
    }, 400)
    return timer
  }, [isMulti])

  // ── Log builder ──────────────────────────────────────────────────────
  const addLog = useCallback((msg) => setLogs(l => [...l, msg]), [])

  // ── handleTrain ──────────────────────────────────────────────────────
  const handleTrain = async () => {
    if (!modelsToTrain.length) return
    setStatus('training')
    setError(null)
    setMultiResult(null)
    setSingleResult(null)
    setLogs([])
    setCurrentModel('')

    const timer = startTicker()

    const logHeader = [
      `🔬 Task type     : ${taskType || '—'}`,
      `📦 Dataset       : rows loading…`,
      `📋 Models        : ${modelsToTrain.join(', ')}`,
      `⚖️  Balancing    : ${TECHNIQUE_LABEL[technique] || 'none'} (train only)`,
      `🎲 Random seed   : ${config.random_state}`,
      `📐 Skewness corr : ${applySkewness ? 'Yes (Yeo-Johnson)' : 'No'}`,
      `📐 Scaler        : ${scalerType}`,
      ``,
    ]
    logHeader.forEach(addLog)

    try {
      if (isMulti) {
        // ── Multi-model run ──────────────────────────────────────────
        addLog(`🚀 Launching multi-model training (${modelsToTrain.length} models)…`)
        const res = await trainMultiModel({
          model_names:         modelsToTrain,
          random_state:        config.random_state,
          balancing_technique: technique === 'none' ? null : technique,
          scaler:              scalerType,
          apply_skewness:      applySkewness,
        })
        clearInterval(timer)
        setProgress(100)
        const d = res.data
        setMultiResult(d)
        setStatus('done')

        // Build logs from results
        d.models.forEach(m => {
          if (m.status === 'success') {
            const mets = m.metrics
            const metStr = Object.entries(mets)
              .map(([k, v]) => `${k}=${v != null ? v.toFixed ? v.toFixed(3) : v : '—'}`)
              .join('  ')
            addLog(`✅ ${m.name}: ${metStr}`)
          } else {
            addLog(`❌ ${m.name}: ${m.error || m.status}`)
          }
        })
        addLog(``)
        addLog(`🏆 Best model: ${d.best_model} (${d.selection_criterion})`)
        addLog(`💾 Saved to session — proceed to Evaluation →`)

      } else {
        // ── Single-model run (existing path, backward compat) ────────
        const mName = modelsToTrain[0]
        setCurrentModel(mName)
        addLog(`🚀 Training ${mName}…`)
        const res = await trainMultiModel({
          model_names:         [mName],
          random_state:        config.random_state,
          balancing_technique: technique === 'none' ? null : technique,
          scaler:              scalerType,
          apply_skewness:      applySkewness,
        })
        clearInterval(timer)
        setProgress(100)
        const d = res.data
        setMultiResult(d)
        setSingleResult(d.models?.[0] || null)
        setStatus('done')

        const m = d.models?.[0]
        if (m?.status === 'success') {
          addLog(`✅ Training complete!`)
          addLog(`📊 Train: ${m.train_samples?.toLocaleString()} · Test: ${m.test_samples?.toLocaleString()}`)
          if (m.metrics) {
            Object.entries(m.metrics).forEach(([k, v]) => {
              if (v != null) addLog(`   ${k}: ${v.toFixed ? v.toFixed(4) : v}`)
            })
          }
          addLog(`💾 Model saved — proceed to Evaluation →`)
        }
      }

      // Auto-navigate after 2.5 s (single model only)
      if (!isMulti) {
        setTimeout(() => { if (onNavigate) onNavigate('evaluation') }, 2500)
      }

    } catch (e) {
      clearInterval(timer)
      setProgress(0)
      const msg = e.response?.data?.detail || e.message || 'Training failed.'
      setError(msg)
      setStatus('error')
      addLog(`❌ Error: ${msg}`)
    }
  }

  // ── Status label ─────────────────────────────────────────────────────
  const statusLabel =
    status === 'idle'     ? 'Waiting to start…'
    : status === 'training' ? isMulti
        ? `Training ${modelsToTrain.length} models…`
        : `Training ${currentModel || modelsToTrain[0]}…`
    : status === 'done'     ? isMulti
        ? `All ${multiResult?.models?.filter(m => m.status === 'success').length || 0} models trained!`
        : 'Training complete!'
    : 'Training failed'

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── Page header ── */}
      <div>
        <h2 className="section-title">Model Training</h2>
        <p className="section-subtitle">
          {modelsToTrain.length > 1
            ? `Training ${modelsToTrain.length} models with identical split — best selected automatically`
            : 'Configure and launch model training'}
        </p>
      </div>

      {/* ── Config summary banner ── */}
      {(status !== 'idle' || modelsToTrain.length > 0) && (
        <ConfigSummary
          models={modelsToTrain}
          taskType={taskType}
          datasetInfo={multiResult?.dataset_info || null}
          technique={technique}
          balancingConfig={balancingConfig}
        />
      )}

      {/* ── Balancing info strip ── */}
      <div className={`flex items-start gap-3 rounded-xl border px-4 py-3 ${
        technique !== 'none'
          ? 'border-brand-500/30 bg-brand-500/8'
          : 'border-surface-600 bg-surface-700/30'
      }`}>
        <Scale size={15} className={technique !== 'none' ? 'text-brand-400' : 'text-slate-500'} />
        <div className="flex-1">
          <p className="text-sm font-semibold text-white">
            Balancing: {TECHNIQUE_LABEL[technique] || technique}
          </p>
          <p className="text-xs text-slate-400 mt-0.5">
            {techLoading
              ? 'Loading…'
              : technique === 'none'
                ? 'No balancing selected — configure in the Class Imbalance step.'
                : 'Applied to X_train / y_train only. X_test is NEVER modified.'}
          </p>
          {balancingConfig?.overridden && balancingConfig?.override_reason && (
            <p className="text-[11px] text-amber-400 mt-1 flex items-center gap-1">
              <span>⚠</span> Auto-override: {balancingConfig.override_reason}
            </p>
          )}
          {balancingConfig?.use_smotenc && (
            <p className="text-[11px] text-purple-400 mt-0.5">
              ✓ SMOTENC with {balancingConfig.cat_indices?.length || 0} categorical indices
            </p>
          )}
        </div>
        <button onClick={() => onNavigate?.('imbalance')}
          className="btn-secondary text-xs flex items-center gap-1 shrink-0">
          Change <ChevronRight size={12} />
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* ── Left: Config + Monitor ── */}
        <div className="space-y-5">

            {/* Post-split transformation config */}
            <div className="card space-y-4 border-indigo-500/20 bg-indigo-500/5">
              <h3 className="font-semibold text-white flex items-center gap-2 text-sm">
                <Scale size={15} className="text-indigo-400" /> Post-Split Transformations
                <span className="text-[10px] font-normal text-slate-500">— fitted on X_train only</span>
              </h3>

              {/* Skewness correction */}
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-white">Skewness Correction</p>
                  <p className="text-xs text-slate-500 mt-0.5">PowerTransformer (Yeo-Johnson) — reduces right/left skew before scaling</p>
                </div>
                <button
                  onClick={() => setApplySkewness(v => !v)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors shrink-0 ${
                    applySkewness ? 'bg-indigo-500' : 'bg-surface-600'
                  }`}
                >
                  <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                    applySkewness ? 'translate-x-4' : 'translate-x-1'
                  }`} />
                </button>
              </div>

              {/* Scaler type */}
              <div>
                <label className="block text-sm font-medium text-white mb-2">Scaler</label>
                <div className="grid grid-cols-3 gap-2">
                  {[['standard', 'StandardScaler', 'Zero mean, unit variance'], ['minmax', 'MinMaxScaler', 'Scale to [0, 1] range'], ['none', 'No Scaling', 'Pass raw values through']].map(
                    ([val, name, desc]) => (
                      <button
                        key={val}
                        onClick={() => setScalerType(val)}
                        className={`text-left rounded-xl border px-3 py-2.5 transition-all ${
                          scalerType === val
                            ? 'border-indigo-500/60 bg-indigo-600/15 text-white'
                            : 'border-surface-600 text-slate-400 hover:border-surface-500'
                        }`}
                      >
                        <p className="text-xs font-semibold">{name}</p>
                        <p className="text-[10px] text-slate-500 mt-0.5">{desc}</p>
                      </button>
                    )
                  )}
                </div>
              </div>

              {/* Pipeline preview */}
              <div className="flex items-center gap-1.5 flex-wrap text-[10px] font-mono">
                <span className="text-slate-500">Pipeline:</span>
                {technique !== 'none' && (
                  <><span className="px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-400 border border-cyan-500/20">{technique}</span>
                  <span className="text-slate-600">→</span></>
                )}
                {applySkewness && (
                  <><span className="px-1.5 py-0.5 rounded bg-indigo-500/15 text-indigo-400 border border-indigo-500/20">skewness</span>
                  <span className="text-slate-600">→</span></>
                )}
                {scalerType !== 'none' && (
                  <><span className="px-1.5 py-0.5 rounded bg-purple-500/15 text-purple-400 border border-purple-500/20">{scalerType}Scaler</span>
                  <span className="text-slate-600">→</span></>
                )}
                <span className="px-1.5 py-0.5 rounded bg-brand-500/15 text-brand-400 border border-brand-500/20">model</span>
              </div>
            </div>

            {/* Training config card */}
          <div className="card space-y-5">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Play size={16} className="text-brand-400" /> Training Configuration
            </h3>

            {/* Model selector — shown only when no prop */}
            {propModels.length === 0 && (
              <div>
                <label className="block text-sm text-slate-300 mb-2 font-medium">Select Model</label>
                <select value={manualModel}
                  onChange={e => { setManualModel(e.target.value); setConfig(c => ({ ...c, model_name: e.target.value })) }}
                  className="select">
                  {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            )}

            {/* Show selected models from Selection step */}
            {propModels.length > 0 && (
              <div>
                <label className="block text-sm text-slate-300 mb-2 font-medium flex items-center gap-2">
                  <Layers size={13} className="text-brand-400" />
                  {propModels.length === 1 ? 'Model to Train' : `Models to Train (${propModels.length})`}
                </label>
                <div className="flex flex-wrap gap-2">
                  {propModels.map(m => (
                    <span key={m} className="text-xs font-semibold px-3 py-1.5 rounded-xl"
                      style={{ background: 'rgba(99,102,241,0.12)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.25)' }}>
                      <Cpu size={10} className="inline mr-1" />{m}
                    </span>
                  ))}
                </div>
                <button onClick={() => onNavigate?.('model-selection')}
                  className="mt-2 text-xs text-brand-400 hover:text-brand-300 flex items-center gap-1 transition-colors">
                  Change models <ChevronRight size={11} />
                </button>
              </div>
            )}

            {/* Preprocessing summary */}
            <div className="rounded-xl border border-surface-700 bg-surface-800/40 px-4 py-3 space-y-1.5">
              <p className="text-xs font-semibold text-slate-400 mb-2">Pipeline Architecture</p>
              {[
                ['Pre-split',  'Full data: cleaned, encoded, feature-engineered', '#34d399'],
                ['Split',      'Train/test split (from Split Data step)', '#818cf8'],
                ['Post-split', applySkewness && scalerType !== 'none'
                  ? 'Skewness → ' + scalerType + 'Scaler → model (X_train only)'
                  : applySkewness ? 'Skewness → model (X_train only)'
                  : scalerType !== 'none' ? scalerType + 'Scaler → model (X_train only)'
                  : 'model only (no transforms)', '#f59e0b'],
                ['Leakage',   'Zero — X_test only receives .transform(), never .fit()', '#34d399'],
              ].map(([k, v, c]) => (
                <div key={k} className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: c }} />
                  <span className="text-xs text-slate-500 w-20 shrink-0">{k}:</span>
                  <span className="text-xs text-slate-300 font-medium">{v}</span>
                </div>
              ))}
            </div>

            {/* Random seed */}
            <div>
              <label className="block text-sm text-slate-300 mb-2 font-medium flex items-center gap-2">
                <Hash size={13} className="text-slate-500" /> Random Seed
                <span className="text-xs text-slate-500 font-normal">— ensures reproducibility</span>
              </label>
              <input type="number" value={config.random_state}
                onChange={e => setConfig(c => ({ ...c, random_state: +e.target.value }))}
                className="input" />
            </div>

            {technique !== 'none' && (
              <div className="flex items-center gap-2 text-xs text-brand-300 bg-brand-600/10 rounded-lg px-3 py-2 border border-brand-600/20">
                <Zap size={12} />
                <span>Will apply <strong>{TECHNIQUE_LABEL[technique]}</strong> to X_train only before fitting</span>
              </div>
            )}

            {/* Start Training button + Clear when results exist */}
            <div className="flex gap-2">
              <button onClick={() => { clearCache(); handleTrain() }}
                disabled={status === 'training' || modelsToTrain.length === 0}
                className="btn-primary flex-1 justify-center gap-2">
                {status === 'training'
                  ? <><div className="spinner" />
                      {isMulti ? `Training ${modelsToTrain.length} models…` : 'Training…'}
                    </>
                  : <><Play size={16} />
                      {isMulti ? `Train ${modelsToTrain.length} Models` : 'Start Training'}
                    </>}
              </button>
              {status === 'done' && (
                <button
                  onClick={clearCache}
                  title="Clear results and reset to idle"
                  className="btn-secondary flex items-center gap-1.5 text-sm shrink-0"
                >
                  <RotateCcw size={13} /> Clear
                </button>
              )}
            </div>
          </div>

          {/* ── Training Monitor ── */}
          <div className="card space-y-4">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Terminal size={16} className="text-brand-400" /> Training Monitor
            </h3>

            {/* Progress bar */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">Progress</span>
                <span className={`font-semibold ${status === 'done' ? 'text-accent-400' : 'text-brand-300'}`}>
                  {progress.toFixed(0)}%
                </span>
              </div>
              <div className="bg-surface-700 rounded-full h-2 overflow-hidden">
                <div className="progress-bar h-full transition-all duration-300" style={{ width: `${progress}%` }} />
              </div>
            </div>

            {/* Status indicator */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full shrink-0 ${STATUS_COLOR[status]}`} />
              <span className="text-sm text-slate-400">{statusLabel}</span>
            </div>

            {/* Log terminal */}
            <div className="bg-surface-900 rounded-xl p-4 font-mono text-xs min-h-[180px] space-y-1.5 overflow-y-auto max-h-64">
              {status === 'idle' && !logs.length && (
                <p className="text-slate-600">▎ Waiting for training to start…</p>
              )}
              {status === 'training' && logs.length === 0 && (
                <p className="text-brand-300 animate-pulse">▎ Initialising…</p>
              )}
              {logs.map((log, i) => (
                <p key={i} className={`animate-fade-in ${
                  log.includes('✅') || log.includes('🏆') || log.includes('💾') ? 'text-accent-400' :
                  log.includes('❌') ? 'text-danger-400' :
                  log.includes('🚀') ? 'text-brand-300' : 'text-slate-300'
                }`}>
                  {log && <><span className="text-slate-600">[{String(i + 1).padStart(2, '0')}]</span> {log}</>}
                </p>
              ))}
              {status === 'training' && logs.length > 0 && (
                <p className="text-brand-300 animate-pulse">▎</p>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="flex items-center gap-2 text-danger-400 text-sm">
                <AlertCircle size={16} />{error}
              </div>
            )}

            {/* Done: single model shortcut */}
            {status === 'done' && !isMulti && (
              <div className="bg-accent-500/10 border border-accent-500/30 rounded-xl p-3 flex items-center gap-3">
                <CheckCircle size={18} className="text-accent-400" />
                <p className="text-sm text-accent-300 font-medium">Model trained! Proceed to Evaluation →</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Right panel ── */}
        <div className="space-y-4">

          {/* Before training: pipeline info card */}
          {status === 'idle' && (
            <div className="card border border-surface-600 p-4 text-sm text-slate-400 flex items-start gap-3">
              <Info size={15} className="text-brand-400 mt-0.5 shrink-0" />
              <div className="space-y-2">
                <p className="font-semibold text-white">ML Pipeline Guarantees</p>
                {[
                  ['✅ Zero leakage',   'Train-test split happens first, before any transformation.'],
                  ['✅ Scaler re-fit',  'Scaler is fitted exclusively on X_train, then applied to X_test.'],
                  ['✅ Balancing order','SMOTE/oversampling applied after split — training data only.'],
                  ['✅ Reproducible',   'Controlled by random_state seed for identical results.'],
                  ...(isMulti ? [['✅ Identical splits', 'All models trained on the exact same X_train / X_test.']] : []),
                ].map(([t, d]) => (
                  <div key={t} className="flex items-start gap-2">
                    <span className="text-accent-400 shrink-0 text-xs font-bold">{t}</span>
                    <span className="text-xs text-slate-500">{d}</span>
                  </div>
                ))}
                <button onClick={() => onNavigate?.('imbalance')}
                  className="mt-1 text-xs text-brand-400 hover:text-brand-300 flex items-center gap-1 transition-colors">
                  Review Class Imbalance <ChevronRight size={11} />
                </button>
              </div>
            </div>
          )}

          {/* Best model banner (multi-model) */}
          {status === 'done' && isMulti && multiResult && (
            <BestModelBanner multiResult={multiResult} taskType={taskType} onNavigate={onNavigate} />
          )}

          {/* ── Balancing distribution card ── */}
          {status === 'done' && multiResult?.dataset_info?.balancing_applied && taskType === 'classification' && (
            <BalancingDistCard
              datasetInfo={multiResult.dataset_info}
              techniqueLabel={TECHNIQUE_LABEL[multiResult.dataset_info?.balancing_used] || multiResult.dataset_info?.balancing_used}
            />
          )}

          {/* Per-model result cards */}
          {status === 'done' && multiResult?.models?.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <CheckCircle size={15} className="text-accent-400" />
                <h3 className="text-sm font-bold text-white">
                  {isMulti ? 'Results per Model' : 'Training Results'}
                </h3>
              </div>
              {multiResult.models.map(r => (
                <ModelResultCard
                  key={r.name}
                  result={r}
                  isBest={r.name === multiResult.best_model}
                  taskType={taskType}
                />
              ))}
            </div>
          )}

          {/* Comparison table */}
          {status === 'done' && isMulti && multiResult && (
            <ComparisonTable
              results={multiResult.models}
              bestName={multiResult.best_model}
              taskType={taskType}
            />
          )}

          {/* Single model: navigate hint */}
          {status === 'done' && !isMulti && (
            <div className="card border-brand-500/30 bg-brand-500/5 flex items-center gap-3 flex-wrap">
              <Trophy size={18} className="text-amber-400 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-white">
                  {multiResult?.best_model} trained successfully
                </p>
                <p className="text-xs text-slate-400 mt-0.5">
                  Saved to session — Evaluation, Bias Detection and Predictions are ready.
                </p>
              </div>
              <button onClick={() => onNavigate?.('evaluation')} className="btn-primary text-xs shrink-0">
                Evaluate <ChevronRight size={12} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
