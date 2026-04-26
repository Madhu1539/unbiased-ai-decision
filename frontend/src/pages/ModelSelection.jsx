import React, { useState, useEffect, useMemo } from 'react'
import {
  Cpu, Star, Zap, Brain, Scale, CheckCircle, AlertCircle,
  ChevronRight, ChevronDown, Layers, Info, Database,
  Target, BarChart2, Lightbulb, Users, Eye, EyeOff,
} from 'lucide-react'
import { getModelRecommendations } from '../services/api'

// ── Badge config ──────────────────────────────────────────────────────────────
const BADGE_CONFIG = {
  'Recommended':      { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.3)',  icon: Star,   label: '⭐ Recommended'      },
  'Fast':             { color: '#34d399', bg: 'rgba(52,211,153,0.10)',  border: 'rgba(52,211,153,0.25)', icon: Zap,    label: '⚡ Fast'             },
  'Advanced':         { color: '#a78bfa', bg: 'rgba(167,139,250,0.10)', border: 'rgba(167,139,250,0.3)', icon: Brain,  label: '🧠 Advanced'         },
  'Handles Imbalance':{ color: '#38bdf8', bg: 'rgba(56,189,248,0.10)',  border: 'rgba(56,189,248,0.25)', icon: Scale,  label: '⚖️ Handles Imbalance' },
  'Interpretable':    { color: '#86efac', bg: 'rgba(134,239,172,0.10)', border: 'rgba(134,239,172,0.3)', icon: Eye,    label: '🔍 Interpretable'     },
  'High Accuracy':    { color: '#f97316', bg: 'rgba(249,115,22,0.10)',  border: 'rgba(249,115,22,0.25)', icon: Target, label: '🎯 High Accuracy'     },
  'Robust':           { color: '#fb923c', bg: 'rgba(251,146,60,0.10)',  border: 'rgba(251,146,60,0.25)', icon: Layers, label: '🛡️ Robust'            },
  'High Dimensional': { color: '#c084fc', bg: 'rgba(192,132,252,0.10)', border: 'rgba(192,132,252,0.3)', icon: BarChart2, label: '📐 High Dimensional' },
  'Baseline':         { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)', border: 'rgba(148,163,184,0.3)', icon: CheckCircle, label: '✔ Baseline'     },
  'Simple':           { color: '#67e8f9', bg: 'rgba(103,232,249,0.10)', border: 'rgba(103,232,249,0.3)', icon: Zap,    label: '💡 Simple'           },
  'Feature Selection':{ color: '#4ade80', bg: 'rgba(74,222,128,0.10)',  border: 'rgba(74,222,128,0.3)',  icon: Target, label: '✂️ Feature Selection' },
}

const PRIORITY_LABEL = ['', '⭐ Top Pick', '✅ Good Match', '📊 Baseline']
const PRIORITY_COLOR = ['', '#f59e0b',    '#34d399',        '#94a3b8']

// ── Small sub-components ─────────────────────────────────────────────────────
function Badge({ name }) {
  const cfg = BADGE_CONFIG[name] || { color: '#94a3b8', bg: 'rgba(148,163,184,0.1)', border: 'rgba(148,163,184,0.3)', label: name }
  return (
    <span className="inline-flex items-center text-[10px] font-semibold px-2 py-0.5 rounded-full"
      style={{ color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.border}` }}>
      {cfg.label || name}
    </span>
  )
}

function ContextPanel({ ctx, imbalance }) {
  if (!ctx) return null
  const { task_type, n_rows, n_features, n_numeric, n_categorical } = ctx
  const imb = typeof imbalance === 'number' ? imbalance : 1.0
  const imbLabel = imb >= 0.7 ? 'Balanced' : imb >= 0.4 ? 'Moderate' : imb >= 0.2 ? 'Imbalanced' : 'Severe'
  const imbColor = imb >= 0.7 ? '#34d399' : imb >= 0.4 ? '#f59e0b' : '#f97316'
  const items = [
    { icon: Database, label: 'Dataset Size', value: `${n_rows?.toLocaleString()} rows` },
    { icon: Target,   label: 'Problem Type', value: task_type === 'classification' ? '🏷️ Classification' : '📈 Regression' },
    { icon: Layers,   label: 'Features',     value: `${n_features} (${n_numeric} numeric, ${n_categorical} categorical)` },
    { icon: Scale,    label: 'Imbalance',    value: imbLabel, color: imbColor },
  ]
  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800/60 px-5 py-4 flex flex-wrap gap-6">
      {items.map(({ icon: Icon, label, value, color }) => (
        <div key={label} className="flex items-center gap-2 min-w-0">
          <Icon size={14} className="text-slate-500 shrink-0" />
          <span className="text-xs text-slate-500">{label}:</span>
          <span className="text-xs font-semibold" style={{ color: color || '#e2e8f0' }}>{value}</span>
        </div>
      ))}
    </div>
  )
}

function RecommendedStrip({ recs, allModels, selected, onSelect, mode }) {
  if (!recs?.length) return null
  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb size={16} className="text-amber-400" />
        <h3 className="text-sm font-bold text-white">System Recommendations</h3>
        <span className="text-[10px] text-slate-500 ml-1">Based on your dataset characteristics</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {recs.slice(0, 3).map((r) => {
          const info       = allModels?.[r.name] || {}
          const isSelected = mode === 'single' ? selected === r.name : (selected || []).includes(r.name)
          const pc         = PRIORITY_COLOR[r.priority] || '#f59e0b'
          return (
            <div key={r.name}
              onClick={() => onSelect(r.name)}
              className="rounded-xl border-2 p-4 cursor-pointer transition-all duration-200 hover:scale-[1.01]"
              style={{
                borderColor: isSelected ? pc : `${pc}55`,
                background:  isSelected ? `${pc}14` : 'rgba(255,255,255,0.03)',
                boxShadow:   isSelected ? `0 0 0 3px ${pc}30` : undefined,
              }}
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ background: `${pc}22` }}>
                    <Star size={15} style={{ color: pc }} />
                  </div>
                  <div>
                    <p className="text-sm font-bold text-white">{r.name}</p>
                    <p className="text-[10px] font-semibold" style={{ color: pc }}>
                      {PRIORITY_LABEL[r.priority]}
                    </p>
                  </div>
                </div>
                {isSelected && <CheckCircle size={16} style={{ color: pc }} className="shrink-0 mt-1" />}
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">{r.reason}</p>
              <div className="flex flex-wrap gap-1 mt-3">
                {(info.badges || []).filter(b => b !== 'Recommended').slice(0, 3).map(b => (
                  <Badge key={b} name={b} />
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ModelCard({ name, info, isSelected, isExpanded, onSelect, onToggle, mode, isRecommended }) {
  const hasStrengths = info.strengths?.length > 0
  const hasPros      = info.pros?.length > 0

  return (
    <div
      className="rounded-xl border transition-all duration-200 cursor-pointer"
      style={{
        borderColor: isSelected
          ? (isRecommended ? '#f59e0b' : '#6366f1')
          : isRecommended
            ? 'rgba(245,158,11,0.3)'
            : 'rgba(71,85,105,0.6)',
        background: isSelected
          ? (isRecommended ? 'rgba(245,158,11,0.06)' : 'rgba(99,102,241,0.06)')
          : 'rgba(255,255,255,0.02)',
        boxShadow: isSelected
          ? `0 0 0 2px ${isRecommended ? 'rgba(245,158,11,0.2)' : 'rgba(99,102,241,0.2)'}`
          : undefined,
      }}
      onClick={() => onSelect(name)}
    >
      {/* Card header */}
      <div className="flex items-center justify-between p-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
            style={{
              background: isSelected
                ? (isRecommended ? 'rgba(245,158,11,0.2)' : '#4f46e5')
                : 'rgba(255,255,255,0.06)',
            }}>
            <Cpu size={16}
              style={{ color: isSelected ? (isRecommended ? '#f59e0b' : '#fff') : '#64748b' }} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <p className="font-semibold text-white text-sm">{name}</p>
              {isRecommended && (
                <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full"
                  style={{ background: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)' }}>
                  ⭐ Recommended
                </span>
              )}
            </div>
            {info.when_to_use && (
              <p className="text-[11px] text-slate-500 mt-0.5 leading-snug line-clamp-1">{info.when_to_use}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {isSelected && (
            <CheckCircle size={16}
              style={{ color: isRecommended ? '#f59e0b' : '#6366f1' }} />
          )}
          <button
            className="w-7 h-7 rounded-lg flex items-center justify-center text-slate-500 hover:text-slate-300 hover:bg-white/5 transition-all"
            onClick={e => { e.stopPropagation(); onToggle() }}>
            <ChevronDown size={14} className={isExpanded ? 'rotate-180 transition-transform' : 'transition-transform'} />
          </button>
        </div>
      </div>

      {/* Badges row */}
      {info.badges?.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-4 pb-3">
          {info.badges.filter(b => b !== 'Recommended').map(b => <Badge key={b} name={b} />)}
        </div>
      )}

      {/* Expandable details */}
      {isExpanded && (
        <div className="border-t border-white/5 px-4 pt-3 pb-4 space-y-3 animate-fade-in">
          {/* Strengths */}
          {(hasStrengths || hasPros) && (
            <div>
              <p className="text-[11px] font-semibold text-accent-400 mb-1.5">✅ Strengths</p>
              <ul className="space-y-0.5">
                {(info.strengths?.length ? info.strengths : info.pros).map(s => (
                  <li key={s} className="text-[11px] text-slate-400 flex items-start gap-1.5">
                    <ChevronRight size={10} className="mt-0.5 text-accent-400 shrink-0" />{s}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Limitations */}
          {(info.limitations?.length > 0 || info.cons?.length > 0) && (
            <div>
              <p className="text-[11px] font-semibold text-danger-400 mb-1.5">⚠️ Limitations</p>
              <ul className="space-y-0.5">
                {(info.limitations?.length ? info.limitations : info.cons).map(c => (
                  <li key={c} className="text-[11px] text-slate-400 flex items-start gap-1.5">
                    <ChevronRight size={10} className="mt-0.5 text-danger-400 shrink-0" />{c}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function ModelSelection({ onModelSelect }) {
  const [data,       setData]       = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)
  const [mode,       setMode]       = useState('single')      // 'single' | 'multi'
  const [selected,   setSelected]   = useState(null)          // string | string[]
  const [expanded,   setExpanded]   = useState(null)
  const [showAll,    setShowAll]    = useState(false)

  useEffect(() => {
    setLoading(true)
    getModelRecommendations()
      .then(res => setData(res.data))
      .catch(e  => setError(e.response?.data?.detail || 'Could not load recommendations. Set target column first.'))
      .finally(() => setLoading(false))
  }, [])

  const taskType         = data?.task_type         || ''
  const ctx              = data?.context           || null
  const recommended      = data?.recommended_models || []
  const allModels        = data?.all_models         || {}
  const notes            = data?.notes              || []
  const imbalanceRatio   = data?.imbalance_ratio    ?? 1.0

  const recNames = useMemo(() => new Set(recommended.map(r => r.name)), [recommended])
  const allNames = useMemo(() => Object.keys(allModels), [allModels])

  // Sort: recommended first, rest alphabetically
  const sortedNames = useMemo(() => {
    const recs  = allNames.filter(n => recNames.has(n))
    const rest  = allNames.filter(n => !recNames.has(n)).sort()
    return [...recs, ...rest]
  }, [allNames, recNames])

  const visibleNames = showAll ? sortedNames : sortedNames.slice(0, recommended.length + 3)

  const handleSelect = (name) => {
    if (mode === 'single') {
      const next = selected === name ? null : name
      setSelected(next)
      onModelSelect?.(next)
    } else {
      setSelected(prev => {
        const arr  = Array.isArray(prev) ? prev : []
        const next = arr.includes(name) ? arr.filter(n => n !== name) : [...arr, name]
        onModelSelect?.(next)
        return next
      })
    }
  }

  const isSelected = (name) =>
    mode === 'single' ? selected === name : (Array.isArray(selected) && selected.includes(name))

  const selectedCount = mode === 'single'
    ? (selected ? 1 : 0)
    : (Array.isArray(selected) ? selected.length : 0)

  const selectedLabel = mode === 'single'
    ? selected
    : (Array.isArray(selected) && selected.length ? selected.join(', ') : null)

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="section-title">Model Selection</h2>
          <p className="section-subtitle">
            {data
              ? `The system has analysed your dataset and ranked the best algorithms for you`
              : 'Guided, intelligent algorithm selection'}
          </p>
        </div>

        {/* Selection mode toggle */}
        <div className="flex items-center bg-surface-800 border border-surface-700 rounded-xl p-1 gap-1 self-start">
          {[['single', Users, 'Single Model'], ['multi', Layers, 'Multi-Model']].map(([m, Icon, lbl]) => (
            <button key={m}
              onClick={() => { setMode(m); setSelected(m === 'single' ? null : []) }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                mode === m ? 'bg-brand-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-300'
              }`}>
              <Icon size={12} />
              {lbl}
            </button>
          ))}
        </div>
      </div>

      {/* Error / loading */}
      {error   && <div className="card border-danger-500/30 flex items-center gap-3 text-danger-400"><AlertCircle size={18}/>{error}</div>}
      {loading && <div className="card flex items-center gap-3 text-brand-300"><div className="spinner"/><span>Analysing your dataset…</span></div>}

      {data && (
        <>
          {/* Context panel */}
          <ContextPanel ctx={ctx} imbalance={imbalanceRatio} />

          {/* System notes / warnings */}
          {notes.length > 0 && (
            <div className="space-y-2">
              {notes.map((n, i) => (
                <div key={i} className="flex items-start gap-3 rounded-xl border border-warn-500/25 bg-warn-500/8 px-4 py-3">
                  <Info size={15} className="text-warn-400 mt-0.5 shrink-0" />
                  <p className="text-xs text-slate-300 leading-relaxed">{n}</p>
                </div>
              ))}
            </div>
          )}

          {/* Recommended strip */}
          <RecommendedStrip
            recs={recommended}
            allModels={allModels}
            selected={selected}
            onSelect={handleSelect}
            mode={mode}
          />

          {/* Divider */}
          <div className="flex items-center gap-3">
            <div className="flex-1 border-t border-surface-700" />
            <span className="text-xs text-slate-600 shrink-0">All Models</span>
            <div className="flex-1 border-t border-surface-700" />
          </div>

          {/* Full model grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {visibleNames.map(name => (
              <ModelCard
                key={name}
                name={name}
                info={allModels[name] || {}}
                isSelected={isSelected(name)}
                isRecommended={recNames.has(name)}
                isExpanded={expanded === name}
                onSelect={handleSelect}
                onToggle={() => setExpanded(p => p === name ? null : name)}
                mode={mode}
              />
            ))}
          </div>

          {/* Show more/fewer */}
          {sortedNames.length > recommended.length + 3 && (
            <button
              onClick={() => setShowAll(s => !s)}
              className="w-full flex items-center justify-center gap-2 text-sm text-brand-400 hover:text-brand-300 py-2 rounded-xl border border-surface-700 hover:border-brand-500/40 transition-all">
              {showAll ? (
                <><EyeOff size={14} /> Show Fewer Models</>
              ) : (
                <><Eye size={14} /> Show All {sortedNames.length} Models ({sortedNames.length - (recommended.length + 3)} more)</>
              )}
            </button>
          )}

          {/* Selection confirmation bar */}
          {selectedCount > 0 && (
            <div className="card border-brand-500/30 bg-brand-600/8 flex items-center gap-3 flex-wrap">
              <CheckCircle size={20} className="text-brand-400 shrink-0" />
              <div className="min-w-0">
                <p className="font-semibold text-white text-sm">
                  {mode === 'single' ? 'Selected: ' : `${selectedCount} model${selectedCount > 1 ? 's' : ''} selected: `}
                  <span className="text-brand-300">{selectedLabel}</span>
                </p>
                <p className="text-xs text-slate-400 mt-0.5">
                  Click <span className="text-brand-300 font-medium">"Training"</span> in the sidebar to proceed.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
