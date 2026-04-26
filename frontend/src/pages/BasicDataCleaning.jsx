import React, { useState, useEffect, useCallback } from 'react'
import {
  Trash2, RefreshCw, Database, Tag, CheckCircle,
  AlertCircle, ChevronDown, ChevronUp, Terminal,
  BarChart2, Columns, Wand2, Eye, EyeOff, Info,
} from 'lucide-react'
import {
  analyzeDataset,
  removeDuplicates,
  fixDataTypes,
  dropColumns,
  getCleaningPreview,
} from '../services/api'

// ─── Sub-component: Section wrapper ───────────────────────────────────────────
function CleanSection({ icon: Icon, title, badge, color = 'text-brand-400', children }) {
  return (
    <div className="card space-y-4">
      <div className="flex items-center gap-3">
        <span className={`w-8 h-8 rounded-lg flex items-center justify-center bg-surface-700 ${color}`}>
          <Icon size={16} />
        </span>
        <h3 className="font-semibold text-white flex-1">{title}</h3>
        {badge}
      </div>
      {children}
    </div>
  )
}

// ─── Sub-component: Action log ────────────────────────────────────────────────
function ActionLog({ log }) {
  if (!log || log.length === 0) return null
  return (
    <div className="bg-surface-900 rounded-xl p-4 font-mono text-xs space-y-1 max-h-40 overflow-y-auto border border-surface-700">
      <div className="flex items-center gap-2 mb-2 text-slate-500">
        <Terminal size={12} /> <span>Action Log</span>
      </div>
      {log.map((entry, i) => (
        <p key={i} className="text-accent-400">
          <span className="text-slate-600">[{String(i + 1).padStart(2, '0')}]</span> {entry}
        </p>
      ))}
    </div>
  )
}

// ─── Sub-component: Success/Error flash ───────────────────────────────────────
function Flash({ type, message }) {
  if (!message) return null
  const isError = type === 'error'
  return (
    <div className={`flex items-start gap-2 rounded-xl px-4 py-3 text-sm border animate-slide-up
      ${isError
        ? 'bg-danger-500/10 border-danger-500/30 text-danger-400'
        : 'bg-accent-500/10 border-accent-500/30 text-accent-400'
      }`}>
      {isError ? <AlertCircle size={16} className="mt-0.5 shrink-0" /> : <CheckCircle size={16} className="mt-0.5 shrink-0" />}
      <span>{message}</span>
    </div>
  )
}

// ─── Badge ────────────────────────────────────────────────────────────────────
function StatusBadge({ count, label, variant = 'warn' }) {
  const cls = {
    warn:    'text-warn-400 bg-warn-500/10 border-warn-500/30',
    success: 'text-accent-400 bg-accent-500/10 border-accent-500/30',
    info:    'text-brand-400 bg-brand-500/10 border-brand-500/30',
    danger:  'text-danger-400 bg-danger-500/10 border-danger-500/30',
  }[variant]
  return (
    <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${cls}`}>
      {count} {label}
    </span>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function BasicDataCleaning() {
  // ── Analysis state ────────────────────────────────────────────────────
  const [analysis, setAnalysis]       = useState(null)
  const [analyzing, setAnalyzing]     = useState(false)
  const [analyzeErr, setAnalyzeErr]   = useState(null)

  // ── Duplicates state ─────────────────────────────────────────────────
  const [dupLoading, setDupLoading]   = useState(false)
  const [dupFlash, setDupFlash]       = useState(null)
  const [showDupPreview, setShowDupPreview] = useState(false)

  // ── Dtype state ───────────────────────────────────────────────────────
  const [selectedConversions, setSelectedConversions] = useState({}) // col → to
  const [dtypeLoading, setDtypeLoading] = useState(false)
  const [dtypeFlash, setDtypeFlash]     = useState(null)

  // ── Drop columns state ───────────────────────────────────────────────
  const [checkedCols, setCheckedCols]   = useState({}) // col → bool
  const [dropLoading, setDropLoading]   = useState(false)
  const [dropFlash, setDropFlash]       = useState(null)

  // ── Dataset stats (refreshed after each action) ──────────────────────
  const [datasetShape, setDatasetShape] = useState(null)
  const [cleaningLog, setCleaningLog]   = useState([])

  // ─────────────────────────────────────────────────────────────────────
  const loadAnalysis = useCallback(async () => {
    setAnalyzing(true)
    setAnalyzeErr(null)
    setDupFlash(null)
    setDtypeFlash(null)
    setDropFlash(null)
    try {
      const res = await analyzeDataset()
      const data = res.data
      setAnalysis(data)
      setDatasetShape(data.shape)
      // Initialise conversion toggles (all OFF by default)
      const convInit = {}
      ;(data.dtype_issues || []).forEach(issue => {
        convInit[issue.column] = issue.suggested_dtype // pre-fill suggestion but checkbox unchecked
      })
      setSelectedConversions(convInit)
      // Initialise column checkboxes (all unchecked)
      const colInit = {}
      ;(data.columns_info || []).forEach(col => {
        colInit[col.name] = false
      })
      setCheckedCols(colInit)
    } catch (e) {
      setAnalyzeErr(e.response?.data?.detail || 'Could not analyze dataset. Please upload a CSV first.')
    } finally {
      setAnalyzing(false)
    }
  }, [])

  useEffect(() => { loadAnalysis() }, [loadAnalysis])

  // ── Refresh preview after any action ────────────────────────────────
  const refreshAfterAction = async () => {
    try {
      const res = await getCleaningPreview()
      setDatasetShape(res.data.shape)
      setCleaningLog(res.data.cleaning_log || [])
    } catch (_) {}
    // Re-run analysis so panels update
    await loadAnalysis()
  }

  // ─────────────────────────────────────────────────────────────────────
  //  ACTION: Remove duplicates
  // ─────────────────────────────────────────────────────────────────────
  const handleRemoveDuplicates = async () => {
    setDupLoading(true)
    setDupFlash(null)
    try {
      const res = await removeDuplicates()
      setDupFlash({ type: 'success', message: res.data.message })
      await refreshAfterAction()
    } catch (e) {
      setDupFlash({ type: 'error', message: e.response?.data?.detail || 'Failed to remove duplicates.' })
    } finally {
      setDupLoading(false)
    }
  }

  // ─────────────────────────────────────────────────────────────────────
  //  ACTION: Fix data types
  // ─────────────────────────────────────────────────────────────────────
  const toggleConversion = (col) => {
    setSelectedConversions(prev => ({
      ...prev,
      [`${col}__checked`]: !prev[`${col}__checked`],
    }))
  }

  const getCheckedConversions = () => {
    if (!analysis?.dtype_issues) return []
    return analysis.dtype_issues
      .filter(issue => selectedConversions[`${issue.column}__checked`])
      .map(issue => ({ column: issue.column, to: issue.suggested_dtype }))
  }

  const handleFixDtypes = async () => {
    const conversions = getCheckedConversions()
    if (conversions.length === 0) {
      setDtypeFlash({ type: 'error', message: 'Select at least one column to convert.' })
      return
    }
    setDtypeLoading(true)
    setDtypeFlash(null)
    try {
      const res = await fixDataTypes(conversions)
      setDtypeFlash({ type: 'success', message: res.data.message })
      await refreshAfterAction()
    } catch (e) {
      setDtypeFlash({ type: 'error', message: e.response?.data?.detail || 'Failed to fix data types.' })
    } finally {
      setDtypeLoading(false)
    }
  }

  // ─────────────────────────────────────────────────────────────────────
  //  ACTION: Drop columns
  // ─────────────────────────────────────────────────────────────────────
  const toggleCol = (colName) => {
    setCheckedCols(prev => ({ ...prev, [colName]: !prev[colName] }))
  }

  const getSelectedCols = () =>
    Object.entries(checkedCols).filter(([, v]) => v).map(([k]) => k)

  const handleDropColumns = async () => {
    const cols = getSelectedCols()
    if (cols.length === 0) {
      setDropFlash({ type: 'error', message: 'Select at least one column to drop.' })
      return
    }
    setDropLoading(true)
    setDropFlash(null)
    try {
      const res = await dropColumns(cols)
      setDropFlash({ type: 'success', message: res.data.message })
      await refreshAfterAction()
    } catch (e) {
      setDropFlash({ type: 'error', message: e.response?.data?.detail || 'Failed to drop columns.' })
    } finally {
      setDropLoading(false)
    }
  }

  // ─────────────────────────────────────────────────────────────────────
  //  RENDER
  // ─────────────────────────────────────────────────────────────────────
  const dupCount       = analysis?.duplicates?.count ?? 0
  const dtypeIssues    = analysis?.dtype_issues ?? []
  const columnsInfo    = analysis?.columns_info ?? []
  const suggestedDrop  = new Set(analysis?.suggested_irrelevant ?? [])
  const checkedConvCount = getCheckedConversions().length
  const checkedDropCount = getSelectedCols().length

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── Page Header ── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="section-title">Basic Data Cleaning</h2>
          <p className="section-subtitle">
            Review your dataset and manually apply safe cleaning operations — nothing changes until you click
          </p>
        </div>
        <button
          onClick={loadAnalysis}
          disabled={analyzing}
          className="btn-secondary flex items-center gap-2 text-sm"
        >
          <RefreshCw size={14} className={analyzing ? 'animate-spin' : ''} />
          {analyzing ? 'Scanning…' : 'Re-scan Dataset'}
        </button>
      </div>

      {/* ── Loading / Error states ── */}
      {analyzing && (
        <div className="card flex items-center gap-3 text-brand-300">
          <div className="spinner" /> <span>Scanning dataset for issues…</span>
        </div>
      )}
      {analyzeErr && !analyzing && (
        <div className="card border-danger-500/30 flex items-center gap-3 text-danger-400">
          <AlertCircle size={18} />
          <span>{analyzeErr}</span>
        </div>
      )}

      {/* ── Dataset Shape Banner ── */}
      {datasetShape && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: 'Rows',    value: datasetShape.rows.toLocaleString() },
            { label: 'Columns', value: datasetShape.columns },
            { label: 'Duplicates',  value: dupCount,      highlight: dupCount > 0 },
            { label: 'Type Issues', value: dtypeIssues.length, highlight: dtypeIssues.length > 0 },
          ].map(({ label, value, highlight }) => (
            <div key={label} className="card text-center py-4">
              <p className={`text-2xl font-bold ${highlight ? 'text-warn-400' : 'text-white'}`}>
                {value}
              </p>
              <p className="text-xs text-slate-400 mt-1">{label}</p>
            </div>
          ))}
        </div>
      )}

      {analysis && (
        <>
          {/* ══════════════════════════════════════════════════════════════
              SECTION 1 — Duplicate Rows
          ══════════════════════════════════════════════════════════════ */}
          <CleanSection
            icon={Trash2}
            title="Duplicate Row Handling"
            color={dupCount > 0 ? 'text-warn-400' : 'text-accent-400'}
            badge={
              dupCount > 0
                ? <StatusBadge count={dupCount} label="duplicates found" variant="warn" />
                : <StatusBadge count="No" label="duplicates" variant="success" />
            }
          >
            <p className="text-sm text-slate-400">
              {dupCount > 0
                ? `Found ${dupCount} duplicate row(s). Removing them keeps only the first occurrence of each.`
                : 'No duplicate rows detected in the current dataset.'}
            </p>

            {/* Duplicate preview toggle */}
            {dupCount > 0 && analysis.duplicates.preview?.length > 0 && (
              <div>
                <button
                  onClick={() => setShowDupPreview(v => !v)}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors mb-2"
                >
                  {showDupPreview ? <EyeOff size={13} /> : <Eye size={13} />}
                  {showDupPreview ? 'Hide' : 'Show'} duplicate preview
                  {showDupPreview ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                </button>

                {showDupPreview && (
                  <div className="table-wrapper max-h-48 overflow-y-auto animate-slide-up">
                    <table className="data-table">
                      <thead>
                        <tr>
                          {Object.keys(analysis.duplicates.preview[0] || {}).map(k => (
                            <th key={k}>{k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {analysis.duplicates.preview.map((row, i) => (
                          <tr key={i}>
                            {Object.values(row).map((v, j) => (
                              <td key={j} className="font-mono text-xs">{v ?? '—'}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            <Flash type={dupFlash?.type} message={dupFlash?.message} />

            <button
              onClick={handleRemoveDuplicates}
              disabled={dupLoading || dupCount === 0}
              className={`btn-primary w-full justify-center ${dupCount === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}
            >
              {dupLoading
                ? <><div className="spinner" /> Removing…</>
                : <><Trash2 size={15} /> Remove {dupCount} Duplicate Row{dupCount !== 1 ? 's' : ''}</>
              }
            </button>
          </CleanSection>

          {/* ══════════════════════════════════════════════════════════════
              SECTION 2 — Data Type Correction
          ══════════════════════════════════════════════════════════════ */}
          <CleanSection
            icon={Tag}
            title="Data Type Correction"
            color={dtypeIssues.length > 0 ? 'text-orange-400' : 'text-accent-400'}
            badge={
              dtypeIssues.length > 0
                ? <StatusBadge count={dtypeIssues.length} label="type issues" variant="warn" />
                : <StatusBadge count="No" label="type issues" variant="success" />
            }
          >
            <p className="text-sm text-slate-400">
              {dtypeIssues.length > 0
                ? 'The following columns appear to have incorrect types. Tick each conversion you want to apply, then click the button.'
                : 'All column types look correct.'}
            </p>

            {dtypeIssues.length > 0 && (
              <div className="space-y-2">
                {dtypeIssues.map(issue => {
                  const key = `${issue.column}__checked`
                  const isChecked = !!selectedConversions[key]
                  return (
                    <label
                      key={issue.column}
                      className={`flex items-start gap-3 rounded-xl p-3 border cursor-pointer transition-all duration-150
                        ${isChecked
                          ? 'bg-brand-600/15 border-brand-500/50'
                          : 'bg-surface-700/40 border-surface-600 hover:border-surface-500'}`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleConversion(issue.column)}
                        className="mt-0.5 accent-brand-500"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-white">
                          {issue.suggested_label}
                        </p>
                        <div className="flex items-center gap-2 mt-1 flex-wrap">
                          <span className="badge badge-info font-mono text-xs">{issue.current_dtype}</span>
                          <span className="text-slate-600 text-xs">→</span>
                          <span className="badge badge-success text-xs">{issue.suggested_dtype}</span>
                          <span className="text-slate-600 text-xs">·</span>
                          <span className="text-xs text-slate-500">
                            e.g. {issue.sample_values?.slice(0, 2).join(', ')}
                          </span>
                        </div>
                      </div>
                    </label>
                  )
                })}
              </div>
            )}

            <Flash type={dtypeFlash?.type} message={dtypeFlash?.message} />

            {dtypeIssues.length > 0 && (
              <button
                onClick={handleFixDtypes}
                disabled={dtypeLoading || checkedConvCount === 0}
                className={`btn-primary w-full justify-center ${checkedConvCount === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}
              >
                {dtypeLoading
                  ? <><div className="spinner" /> Fixing…</>
                  : <><Wand2 size={15} /> Fix {checkedConvCount || 'Selected'} Data Type{checkedConvCount !== 1 ? 's' : ''}</>
                }
              </button>
            )}
          </CleanSection>

          {/* ══════════════════════════════════════════════════════════════
              SECTION 3 — Column Removal
          ══════════════════════════════════════════════════════════════ */}
          <CleanSection
            icon={Columns}
            title="Irrelevant Column Removal"
            color="text-violet-400"
            badge={
              checkedDropCount > 0
                ? <StatusBadge count={checkedDropCount} label="selected to drop" variant="danger" />
                : <StatusBadge count={columnsInfo.length} label="columns" variant="info" />
            }
          >
            <div className="flex items-start gap-2 rounded-xl bg-blue-500/8 border border-blue-500/20 px-3 py-2.5">
              <Info size={14} className="text-blue-400 mt-0.5 shrink-0" />
              <p className="text-xs text-blue-300">
                Columns marked <span className="font-semibold text-warn-400">⚠ Suggested</span> are likely
                irrelevant (ID-type or near-unique). They are <strong>not pre-selected</strong> — tick only
                what you want to drop. The target column is protected and cannot be dropped.
              </p>
            </div>

            {/* Column checklist */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-72 overflow-y-auto pr-1">
              {columnsInfo.map(col => {
                const isTarget    = col.is_target
                const isSuggested = suggestedDrop.has(col.name)
                const isChecked   = !!checkedCols[col.name]

                return (
                  <label
                    key={col.name}
                    className={`flex items-center gap-3 rounded-xl px-3 py-2.5 border text-sm transition-all duration-150
                      ${isTarget       ? 'border-accent-500/40 bg-accent-500/8 cursor-not-allowed opacity-70'
                      : isChecked      ? 'border-danger-500/50 bg-danger-500/10 cursor-pointer'
                      : isSuggested    ? 'border-warn-500/40 bg-warn-500/8 cursor-pointer hover:border-warn-400'
                      :                  'border-surface-600 bg-surface-700/30 cursor-pointer hover:border-surface-500'}`}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      disabled={isTarget}
                      onChange={() => !isTarget && toggleCol(col.name)}
                      className="accent-danger-500"
                    />
                    <div className="flex-1 min-w-0">
                      <span className="font-mono text-xs text-white truncate block">{col.name}</span>
                      <span className="text-xs text-slate-500">{col.dtype} · {col.n_unique} unique</span>
                    </div>
                    <div className="flex flex-col items-end gap-0.5">
                      {isTarget    && <span className="text-[10px] font-bold text-accent-400">🎯 Target</span>}
                      {isSuggested && !isTarget && (
                        <span className="text-[10px] font-semibold text-warn-400">⚠ Suggested</span>
                      )}
                      {col.missing > 0 && (
                        <span className="text-[10px] text-slate-500">{col.missing} missing</span>
                      )}
                    </div>
                  </label>
                )
              })}
            </div>

            {/* Select/deselect helpers */}
            <div className="flex gap-2 flex-wrap text-xs">
              <button
                onClick={() => {
                  const next = {}
                  columnsInfo.forEach(col => {
                    if (!col.is_target) next[col.name] = suggestedDrop.has(col.name)
                  })
                  setCheckedCols(next)
                }}
                className="text-warn-400 hover:underline"
              >
                Select suggested
              </button>
              <span className="text-slate-600">·</span>
              <button
                onClick={() => {
                  const next = {}
                  columnsInfo.forEach(col => { if (!col.is_target) next[col.name] = true })
                  setCheckedCols(next)
                }}
                className="text-slate-400 hover:text-white hover:underline"
              >
                Select all
              </button>
              <span className="text-slate-600">·</span>
              <button
                onClick={() => {
                  const next = {}
                  columnsInfo.forEach(col => { next[col.name] = false })
                  setCheckedCols(next)
                }}
                className="text-slate-400 hover:text-white hover:underline"
              >
                Deselect all
              </button>
            </div>

            <Flash type={dropFlash?.type} message={dropFlash?.message} />

            <button
              onClick={handleDropColumns}
              disabled={dropLoading || checkedDropCount === 0}
              className={`btn-primary w-full justify-center bg-danger-600 hover:bg-danger-500 border-danger-500
                ${checkedDropCount === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}
            >
              {dropLoading
                ? <><div className="spinner" /> Dropping…</>
                : <><Trash2 size={15} /> Drop {checkedDropCount || 'Selected'} Column{checkedDropCount !== 1 ? 's' : ''}</>
              }
            </button>
          </CleanSection>

          {/* ══════════════════════════════════════════════════════════════
              Dataset Preview (updated after each action)
          ══════════════════════════════════════════════════════════════ */}
          <CleanSection icon={Database} title="Updated Dataset Preview" color="text-brand-400">
            <div className="flex items-center gap-3 flex-wrap text-xs text-slate-400 mb-1">
              {datasetShape && (
                <>
                  <span className="badge badge-info">{datasetShape.rows.toLocaleString()} rows</span>
                  <span className="text-slate-600">×</span>
                  <span className="badge badge-info">{datasetShape.columns} columns</span>
                </>
              )}
            </div>

            {columnsInfo.length > 0 && (
              <div className="table-wrapper max-h-48 overflow-y-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Type</th>
                      <th>Missing</th>
                      <th>Unique</th>
                      <th>Role</th>
                    </tr>
                  </thead>
                  <tbody>
                    {columnsInfo.map(col => (
                      <tr key={col.name}>
                        <td className="font-mono text-brand-300 text-xs">{col.name}</td>
                        <td><span className="badge badge-info text-xs">{col.dtype}</span></td>
                        <td className={col.missing > 0 ? 'text-warn-400' : 'text-accent-400'}>{col.missing}</td>
                        <td className="text-slate-400 text-xs">{col.n_unique}</td>
                        <td>
                          {col.is_target
                            ? <span className="badge badge-success text-xs">🎯 Target</span>
                            : suggestedDrop.has(col.name)
                            ? <span className="badge badge-warn text-xs">⚠ Review</span>
                            : <span className="badge text-xs bg-surface-700 text-slate-400">Feature</span>
                          }
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CleanSection>

          {/* ── Action Log ── */}
          {cleaningLog.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                <BarChart2 size={16} className="text-brand-400" /> Cleaning Action Log
              </h3>
              <ActionLog log={cleaningLog} />
            </div>
          )}
        </>
      )}
    </div>
  )
}
