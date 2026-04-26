import React, { useState, useEffect, useCallback } from 'react'
import {
  GitFork, Play, CheckCircle, AlertCircle, AlertTriangle,
  Info, RefreshCw, Clock, Users, ShieldAlert, BarChart2,
  Layers, Bug, ChevronDown, ChevronUp, Lock,
} from 'lucide-react'
import { analyzeSplit, performSplit, getSplitStatus } from '../services/api'

// ─── Tiny UI helpers ──────────────────────────────────────────────────────────
function Alert({ type, children, className = '' }) {
  const styles = {
    info:    'bg-blue-500/8   border-blue-500/25   text-blue-300',
    warn:    'bg-warn-500/10  border-warn-500/30   text-warn-300',
    error:   'bg-danger-500/10 border-danger-500/30 text-danger-400',
    success: 'bg-accent-500/10 border-accent-500/30 text-accent-400',
    purple:  'bg-purple-500/10 border-purple-500/30 text-purple-300',
  }
  const icons = {
    info: <Info size={14} className="shrink-0 mt-0.5" />,
    warn: <AlertTriangle size={14} className="shrink-0 mt-0.5" />,
    error: <AlertCircle size={14} className="shrink-0 mt-0.5" />,
    success: <CheckCircle size={14} className="shrink-0 mt-0.5" />,
    purple: <ShieldAlert size={14} className="shrink-0 mt-0.5" />,
  }
  return (
    <div className={`flex gap-2 rounded-xl border px-4 py-3 text-sm ${styles[type] || styles.info} ${className}`}>
      {icons[type] || icons.info}
      <span>{children}</span>
    </div>
  )
}

function Toggle({ checked, onChange, label, sub }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-slate-300 font-medium">{label}</p>
        {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative w-11 h-6 rounded-full transition-colors shrink-0 ${checked ? 'bg-brand-600' : 'bg-surface-600'}`}
      >
        <span className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-5' : ''}`} />
      </button>
    </div>
  )
}

function ClassDistBar({ label, dist, colorClass }) {
  if (!dist || Object.keys(dist).length === 0) return null
  return (
    <div>
      <p className="text-xs font-semibold text-slate-400 mb-2">{label}</p>
      <div className="space-y-2">
        {Object.entries(dist).map(([cls, info]) => (
          <div key={cls}>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span className="font-mono">{cls}</span>
              <span>{info.count?.toLocaleString()} ({info.pct}%)</span>
            </div>
            <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${colorClass}`} style={{ width: `${info.pct}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function StatBox({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="card text-center py-5">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      <p className="text-xs text-slate-400 mt-1">{label}</p>
    </div>
  )
}

// ─── Method option card ───────────────────────────────────────────────────────
function MethodCard({ id, icon: Icon, label, desc, selected, disabled, disabledReason, onClick }) {
  return (
    <button
      onClick={() => !disabled && onClick(id)}
      disabled={disabled}
      title={disabled ? disabledReason : ''}
      className={`w-full text-left rounded-xl border p-3 transition-all duration-150
        ${disabled
          ? 'opacity-40 cursor-not-allowed border-surface-600 bg-surface-700/30'
          : selected
          ? 'border-brand-500/60 bg-brand-600/15 cursor-pointer'
          : 'border-surface-600 bg-surface-700/30 hover:border-surface-500 cursor-pointer'
        }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <Icon size={14} className={selected ? 'text-brand-400' : 'text-slate-400'} />
        <span className={`text-sm font-semibold ${selected ? 'text-white' : 'text-slate-300'}`}>{label}</span>
        {disabled && <span className="text-[10px] text-slate-500 ml-auto">Unavailable</span>}
      </div>
      <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
    </button>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function SplitData() {
  // ── Analysis state ─────────────────────────────────────────────────
  const [info,       setInfo]       = useState(null)
  const [infoLoading,setInfoLoading]= useState(true)
  const [infoError,  setInfoError]  = useState(null)

  // ── Config state ───────────────────────────────────────────────────
  const [testPct,       setTestPct]       = useState(20)      // slider as integer %
  const [randomState,   setRandomState]   = useState(42)
  const [shuffle,       setShuffle]       = useState(true)
  const [stratify,      setStratify]      = useState(true)
  const [splitMethod,   setSplitMethod]   = useState('stratified')
  const [datetimeCol,   setDatetimeCol]   = useState('')
  const [groupCol,      setGroupCol]      = useState('')

  // ── Result / status state ──────────────────────────────────────────
  const [loading,   setLoading]   = useState(false)
  const [flash,     setFlash]     = useState(null)
  const [result,    setResult]    = useState(null)
  const [existing,  setExisting]  = useState(null)
  const [showLeakage, setShowLeakage] = useState(false)

  // ─────────────────────────────────────────────────────────────────
  const loadAnalysis = useCallback(async () => {
    setInfoLoading(true)
    setInfoError(null)
    try {
      const [anaRes, statusRes] = await Promise.allSettled([
        analyzeSplit(),
        getSplitStatus(),
      ])
      if (anaRes.status === 'fulfilled') {
        const d = anaRes.value.data
        setInfo(d)
        // Auto-select best default method
        if (d.task_type === 'regression') setSplitMethod('random')
        else if (d.task_type === 'classification' && d.stratify_ok) setSplitMethod('stratified')
        // Pre-fill datetime/group if detected
        if (d.datetime_cols?.length > 0) setDatetimeCol(d.datetime_cols[0])
        if (d.group_cols?.length > 0)    setGroupCol(d.group_cols[0])
        // Dataset size guidance → suggest smaller test size for large datasets
        if (d.n_rows > 1_000_000) setTestPct(5)
      } else {
        setInfoError(anaRes.reason?.response?.data?.detail || 'Could not analyze dataset.')
      }
      if (statusRes.status === 'fulfilled' && statusRes.value.data.split_done) {
        setExisting(statusRes.value.data)
      }
    } finally {
      setInfoLoading(false)
    }
  }, [])

  useEffect(() => { loadAnalysis() }, [loadAnalysis])

  // Auto-switch method when splitting type isn't suitable
  useEffect(() => {
    if (info?.task_type === 'regression' && splitMethod === 'stratified') {
      setSplitMethod('random')
    }
  }, [info, splitMethod])

  // When chronological selected → disable shuffle
  useEffect(() => {
    if (splitMethod === 'chronological') setShuffle(false)
  }, [splitMethod])

  // ─────────────────────────────────────────────────────────────────
  const handleSplit = async () => {
    setLoading(true)
    setFlash(null)
    setResult(null)
    try {
      const res = await performSplit({
        test_size:       testPct / 100,
        random_state:    randomState,
        shuffle:         splitMethod === 'chronological' ? false : shuffle,
        stratify,
        split_method:    splitMethod,
        datetime_column: splitMethod === 'chronological' ? datetimeCol : null,
        group_column:    splitMethod === 'group'         ? groupCol     : null,
      })
      setResult(res.data)
      setExisting(null)
      const t = res.data.leakage_warning
        ? 'warn'
        : res.data.warning
        ? 'warn'
        : 'success'
      setFlash({ type: t, message: res.data.message })
    } catch (e) {
      setFlash({
        type: 'error',
        message: e.response?.data?.detail || 'Split failed. Check that your dataset is loaded and a target column is set.',
      })
    } finally {
      setLoading(false)
    }
  }

  // ─── Derived ──────────────────────────────────────────────────────
  const trainPct        = 100 - testPct
  const task            = info?.task_type
  const isClassification = task === 'classification'
  const canSplit        = !infoLoading && !infoError && !(info?.class_errors?.length > 0)
  const active          = result || existing

  const methodDisabled = {
    stratified:     !isClassification || !(info?.stratify_ok),
    chronological:  !(info?.datetime_cols?.length > 0),
    group:          !(info?.group_cols?.length > 0),
  }

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="section-title">Split Data (Train / Test)</h2>
          <p className="section-subtitle">
            Divide the cleaned dataset into leakage-free training and test sets — nothing is applied until you confirm.
          </p>
        </div>
        <button onClick={loadAnalysis} disabled={infoLoading} className="btn-secondary flex items-center gap-2 text-sm shrink-0">
          <RefreshCw size={14} className={infoLoading ? 'animate-spin' : ''} />
          {infoLoading ? 'Scanning…' : 'Re-analyze'}
        </button>
      </div>

      {/* ── Loading / error ── */}
      {infoLoading && (
        <div className="card flex items-center gap-3 text-brand-300">
          <div className="spinner" /> Analyzing dataset…
        </div>
      )}
      {infoError && !infoLoading && (
        <Alert type="error">{infoError}</Alert>
      )}

      {/* ── Existing split banner ── */}
      {!infoLoading && existing && !result && (
        <div className="card border-brand-500/30 bg-brand-500/8 flex items-center gap-3 flex-wrap">
          <CheckCircle size={18} className="text-brand-400 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-white">
              Existing split — {existing.train_rows?.toLocaleString()} train / {existing.test_rows?.toLocaleString()} test
            </p>
            <p className="text-xs text-slate-400">
              Method: {existing.config?.method_label || 'Unknown'} · You can re-split below.
            </p>
          </div>
          <span className="badge badge-success text-xs">{existing.features} features</span>
        </div>
      )}

      {info && !infoLoading && (
        <>
          {/* ── Dataset overview ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatBox label="Total Rows"   value={info.n_rows?.toLocaleString()} />
            <StatBox label="Features"     value={info.n_features} />
            <StatBox label="Problem Type" value={info.task_type}  color={isClassification ? 'text-brand-400' : 'text-violet-400'} />
            <StatBox label="Target"       value={info.target}     color="text-accent-400" />
          </div>

          {/* ── Pre-split validations ── */}
          {info.class_errors?.map((e, i) => <Alert key={i} type="error">{e}</Alert>)}
          {info.class_warnings?.map((w, i) => <Alert key={i} type="warn">{w}</Alert>)}
          {info.size_guidance && <Alert type="info">{info.size_guidance}</Alert>}
          {info.datetime_cols?.length > 0 && (
            <Alert type="warn">
              <strong>Time-based data detected</strong> (column{info.datetime_cols.length > 1 ? 's' : ''}: {info.datetime_cols.join(', ')}). Random shuffling may cause look-ahead bias. Consider using <strong>Chronological Split</strong>.
            </Alert>
          )}
          {info.group_cols?.length > 0 && (
            <Alert type="purple">
              <strong>Potential group/ID column{info.group_cols.length > 1 ? 's' : ''} detected</strong>: {info.group_cols.join(', ')}. Consider using <strong>Group-Based Split</strong> to prevent the same entity appearing in both sets.
            </Alert>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* ══════════════════════════════════════════════
                LEFT — Configuration
            ══════════════════════════════════════════════ */}
            <div className="space-y-5">

              {/* Split Method */}
              <div className="card space-y-3">
                <h3 className="font-semibold text-white flex items-center gap-2 text-sm">
                  <GitFork size={15} className="text-brand-400" /> Split Method
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  <MethodCard
                    id="random" icon={Play}
                    label="Standard Random" desc="Randomly shuffles then splits. Good for well-balanced datasets."
                    selected={splitMethod === 'random'} onClick={setSplitMethod}
                  />
                  <MethodCard
                    id="stratified" icon={BarChart2}
                    label="Stratified" desc="Preserves class ratio in both sets. Recommended for classification."
                    selected={splitMethod === 'stratified'}
                    disabled={methodDisabled.stratified}
                    disabledReason={!isClassification ? 'Only for classification tasks' : 'Stratify unavailable — a class has < 2 samples'}
                    onClick={setSplitMethod}
                  />
                  <MethodCard
                    id="chronological" icon={Clock}
                    label="Chronological" desc="Sorts by date → first rows = train, last rows = test. No look-ahead bias."
                    selected={splitMethod === 'chronological'}
                    disabled={methodDisabled.chronological}
                    disabledReason="No datetime columns detected"
                    onClick={setSplitMethod}
                  />
                  <MethodCard
                    id="group" icon={Users}
                    label="Group-Based" desc="Ensures no group (e.g. patient, user) spans both train and test."
                    selected={splitMethod === 'group'}
                    disabled={methodDisabled.group}
                    disabledReason="No group/ID columns detected"
                    onClick={setSplitMethod}
                  />
                </div>

                {/* Conditional sub-selects */}
                {splitMethod === 'chronological' && info.datetime_cols?.length > 0 && (
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Sort by datetime column</label>
                    <select
                      value={datetimeCol}
                      onChange={e => setDatetimeCol(e.target.value)}
                      className="input w-full text-sm"
                    >
                      {info.datetime_cols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                    <Alert type="warn" className="mt-2 text-xs">
                      Shuffle will be automatically disabled for chronological split.
                    </Alert>
                  </div>
                )}
                {splitMethod === 'group' && (
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Group / ID column</label>
                    <select
                      value={groupCol}
                      onChange={e => setGroupCol(e.target.value)}
                      className="input w-full text-sm"
                    >
                      {(info.group_cols?.length > 0 ? info.group_cols : Object.keys(info.class_dist || {})).map(c =>
                        <option key={c} value={c}>{c}</option>
                      )}
                      {!info.group_cols?.length && (
                        <option value="" disabled>No group columns detected</option>
                      )}
                    </select>
                    <p className="text-xs text-slate-500 mt-1">
                      All records of each group will remain in a single split to prevent leakage.
                    </p>
                  </div>
                )}
              </div>

              {/* Ratio + Seed */}
              <div className="card space-y-5">
                <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                  <Layers size={15} className="text-brand-400" /> Split Ratio & Seed
                </h3>

                {/* Visual bar */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-slate-300">Train / Test Ratio</label>
                    <div className="flex gap-2 text-xs">
                      <span className="font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-lg">{trainPct}% train</span>
                      <span className="text-slate-600">/</span>
                      <span className="font-mono text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-lg">{testPct}% test</span>
                    </div>
                  </div>
                  <div className="h-5 rounded-full overflow-hidden flex mb-2">
                    <div className="h-full bg-emerald-500/60 flex items-center justify-center text-[10px] font-bold text-white transition-all" style={{ width: `${trainPct}%` }}>
                      {trainPct >= 15 && 'Train'}
                    </div>
                    <div className="h-full bg-rose-500/60 flex items-center justify-center text-[10px] font-bold text-white transition-all" style={{ width: `${testPct}%` }}>
                      {testPct >= 10 && 'Test'}
                    </div>
                  </div>
                  <input type="range" min={5} max={50} step={5} value={testPct} onChange={e => setTestPct(Number(e.target.value))} className="w-full accent-brand-500 cursor-pointer" />
                  <div className="flex justify-between text-xs text-slate-600 mt-1">
                    <span>5% test</span><span>50% test</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm text-slate-300 mb-1.5">Random Seed <span className="text-xs text-slate-500">(reproducibility)</span></label>
                  <input type="number" value={randomState} min={0} max={99999} onChange={e => setRandomState(Number(e.target.value))} className="input w-full" />
                </div>
              </div>

              {/* Toggles */}
              <div className="card space-y-4">
                <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                  <ShieldAlert size={15} className="text-brand-400" /> Safety Options
                </h3>

                <Toggle
                  checked={shuffle}
                  onChange={splitMethod === 'chronological' ? () => {} : setShuffle}
                  label="Shuffle Data"
                  sub={splitMethod === 'chronological' ? 'Disabled for chronological splits' : 'Randomise row order before splitting'}
                />
                {!shuffle && splitMethod !== 'chronological' && (
                  <Alert type="warn" className="text-xs">
                    Shuffle is OFF — rows will be split in their current order. Ensure data is not sorted by class.
                  </Alert>
                )}

                {isClassification && (
                  <Toggle
                    checked={splitMethod === 'stratified'}
                    onChange={v => setSplitMethod(v ? 'stratified' : 'random')}
                    label="Use Stratified Split"
                    sub="Ensures class proportions match in both sets"
                  />
                )}
                {isClassification && splitMethod !== 'stratified' && (
                  <Alert type="warn" className="text-xs">
                    Stratification is OFF — class distribution may be uneven in train/test.
                  </Alert>
                )}
              </div>
            </div>

            {/* ══════════════════════════════════════════════
                RIGHT — Class dist + Info
            ══════════════════════════════════════════════ */}
            <div className="space-y-5">

              {/* Class distribution preview (classification) */}
              {isClassification && info.class_dist && (
                <div className="card space-y-3">
                  <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                    <BarChart2 size={15} className="text-brand-400" /> Current Class Distribution
                  </h3>
                  <ClassDistBar label="Full Dataset" dist={info.class_dist} colorClass="bg-brand-500" />
                  {info.minority_count < 10 && info.minority_count >= 2 && (
                    <Alert type="warn" className="text-xs">
                      Minority class '{info.minority_class}' has only {info.minority_count} samples. Results may be unstable.
                    </Alert>
                  )}
                </div>
              )}

              {/* Test Set Lock reminder */}
              <div className="card border-surface-600 space-y-3">
                <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                  <Lock size={15} className="text-brand-400" /> Data Integrity Rules
                </h3>
                {[
                  { icon: <CheckCircle size={12} className="text-accent-400 shrink-0 mt-0.5" />, text: 'Test set is locked after split — no fitting on test data in later steps' },
                  { icon: <CheckCircle size={12} className="text-accent-400 shrink-0 mt-0.5" />, text: 'Feature Engineering transformations fit on train only' },
                  { icon: <CheckCircle size={12} className="text-accent-400 shrink-0 mt-0.5" />, text: 'Preprocessing (imputation, scaling, encoding) fit on train only' },
                  { icon: <CheckCircle size={12} className="text-accent-400 shrink-0 mt-0.5" />, text: 'Class imbalance correction (SMOTE, undersampling) on train only' },
                ].map((r, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs text-slate-400">{r.icon} {r.text}</div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Flash ── */}
          {flash && (
            <Alert type={flash.type} className="animate-slide-up">{flash.message}</Alert>
          )}

          {/* ── Apply button ── */}
          <button
            onClick={handleSplit}
            disabled={loading || !canSplit}
            className={`btn-primary w-full justify-center py-3 ${!canSplit ? 'opacity-40 cursor-not-allowed' : ''}`}
          >
            {loading
              ? <><div className="spinner" /> Applying Split…</>
              : <><Play size={16} /> Confirm and Apply Split</>
            }
          </button>
          {!canSplit && info?.class_errors?.length > 0 && (
            <p className="text-xs text-center text-danger-400">
              Fix the errors above before splitting.
            </p>
          )}
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════════
          POST-SPLIT RESULTS
      ═══════════════════════════════════════════════════════════════ */}
      {active && (
        <div className="space-y-5 animate-slide-up">
          <div className="flex items-center gap-3">
            <CheckCircle size={18} className="text-accent-400" />
            <h3 className="font-semibold text-white">Split Results</h3>
            {(result?.method_label || active?.config?.method_label) && (
              <span className="badge badge-success text-xs">
                Applied: {result?.method_label || active?.config?.method_label}
              </span>
            )}
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatBox label="Total Rows"  value={(active.train_rows + active.test_rows).toLocaleString()} />
            <StatBox label="Train Rows"  value={active.train_rows?.toLocaleString()} sub={`${result?.train_pct ?? trainPct}%`} color="text-emerald-400" />
            <StatBox label="Test Rows"   value={active.test_rows?.toLocaleString()}  sub={`${result?.test_pct  ?? testPct}%`}  color="text-rose-400" />
            <StatBox label="Features"    value={active.features} color="text-brand-400" />
          </div>

          {/* Class distribution comparison (classification) */}
          {result?.class_dist_train && Object.keys(result.class_dist_train).length > 0 && (
            <div className="card grid grid-cols-1 md:grid-cols-2 gap-6">
              <ClassDistBar label="Train Set — Class Distribution" dist={result.class_dist_train} colorClass="bg-emerald-500" />
              <ClassDistBar label="Test Set — Class Distribution"  dist={result.class_dist_test}  colorClass="bg-rose-500"    />
            </div>
          )}

          {/* Config summary */}
          <div className="card">
            <p className="text-xs text-slate-500 mb-3 font-semibold uppercase tracking-wider">Split Configuration</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              {[
                { k: 'Method',       v: result?.method_label || active?.config?.method_label || '—' },
                { k: 'Test Size',    v: `${result?.test_pct  ?? testPct}%` },
                { k: 'Random Seed',  v: result?.random_state ?? randomState },
                { k: 'Task Type',    v: result?.task_type    ?? task ?? '—' },
              ].map(({ k, v }) => (
                <div key={k} className="bg-surface-700/50 rounded-xl p-3 border border-surface-600">
                  <p className="text-xs text-slate-500">{k}</p>
                  <p className="text-white font-semibold mt-0.5 capitalize text-sm truncate">{String(v)}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ── Leakage Audit ── */}
          {result?.leakage_findings !== undefined && (
            <div className="card">
              <button
                onClick={() => setShowLeakage(v => !v)}
                className="flex items-center gap-2 w-full text-left"
              >
                <Bug size={15} className={result.leakage_findings?.length > 0 ? 'text-warn-400' : 'text-accent-400'} />
                <span className="font-semibold text-white text-sm flex-1">Post-Split Data Leakage Audit</span>
                <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border
                  ${result.leakage_findings?.length > 0
                    ? 'text-warn-400 bg-warn-500/10 border-warn-500/30'
                    : 'text-accent-400 bg-accent-500/10 border-accent-500/30'}`}>
                  {result.leakage_findings?.length > 0 ? `${result.leakage_findings.length} risk(s) found` : 'No leakage detected'}
                </span>
                {showLeakage ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
              </button>

              {showLeakage && (
                <div className="mt-4 space-y-3 animate-slide-up">
                  <p className="text-xs text-slate-500">
                    Only high-cardinality columns (&gt;50% unique values) are audited to avoid false positives on common values like gender or city names.
                  </p>

                  {result.leakage_findings?.length === 0 && (
                    <Alert type="success">No high-cardinality entity overlap detected between train and test sets.</Alert>
                  )}

                  {result.leakage_warning && (
                    <Alert type="warn">{result.leakage_warning}</Alert>
                  )}

                  {result.leakage_findings?.length > 0 && (
                    <div className="table-wrapper">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Column</th>
                            <th>Unique Values</th>
                            <th>Overlap Count</th>
                            <th>Overlap %</th>
                            <th>Risk</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.leakage_findings.map(f => (
                            <tr key={f.column}>
                              <td className="font-mono text-brand-300 text-xs">{f.column}</td>
                              <td className="text-slate-400 text-xs">{f.unique_total}</td>
                              <td className="text-warn-400 font-semibold text-xs">{f.overlap_count}</td>
                              <td className="text-xs">{f.overlap_pct}%</td>
                              <td>
                                <span className={`badge text-xs ${f.overlap_pct > 50 ? 'bg-danger-500/20 text-danger-400 border-danger-500/30' : 'bg-warn-500/10 text-warn-400 border-warn-500/30'} border`}>
                                  {f.overlap_pct > 50 ? 'High' : 'Medium'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {result.leakage_findings?.length > 0 && (
                    <Alert type="info">
                      Consider switching to <strong>Group-Based Split</strong> and selecting the flagged column as the group column to eliminate entity overlap.
                    </Alert>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Next step prompt */}
          <div className="flex items-center gap-2 text-xs text-accent-400 bg-accent-500/8 border border-accent-500/20 rounded-xl px-4 py-3">
            <CheckCircle size={13} />
            Split complete — proceed to <strong>Feature Engineering</strong>. All transformations will fit on train data only.
          </div>
        </div>
      )}
    </div>
  )
}
