import React, { useState, useEffect, useCallback } from 'react'
import { Crosshair, Send, RefreshCw, AlertCircle, Info } from 'lucide-react'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer,
} from 'recharts'
import { getBatchPredictions, predictSingle, getFEStatus } from '../services/api'

export default function Predictions() {
  const [batch, setBatch]         = useState(null)
  const [features, setFeatures]   = useState([])   // required feature names from session
  const [formVals, setFormVals]   = useState({})   // feature → numeric string value
  const [singleResult, setSingle] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [predicting, setPred]     = useState(false)
  const [error, setError]         = useState(null)
  const [singleError, setSingleError] = useState(null)
  const [featuresLoaded, setFeaturesLoaded] = useState(false)

  // ── Load feature columns from session (for pre-filling the form) ────
  const loadFeatures = useCallback(async () => {
    try {
      const res = await getFEStatus()
      const cols = res.data?.features || []
      if (cols.length > 0) {
        setFeatures(cols)
        // Pre-fill formVals with correct keys, empty values
        setFormVals(Object.fromEntries(cols.map(c => [c, ''])))
        setFeaturesLoaded(true)
      }
    } catch {
      // Feature status unavailable — user fills manually
      setFeaturesLoaded(false)
    }
  }, [])

  const loadBatch = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const res = await getBatchPredictions()
      setBatch(res.data)
      // If features not yet loaded from FE status, try features_used from batch
      if (!featuresLoaded && res.data?.features_used?.length > 0) {
        const cols = res.data.features_used
        setFeatures(cols)
        setFormVals(Object.fromEntries(cols.map(c => [c, ''])))
        setFeaturesLoaded(true)
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'No predictions available. Train a model first.')
    } finally { setLoading(false) }
  }, [featuresLoaded])

  useEffect(() => {
    loadFeatures()
    loadBatch()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const scatterData = batch
    ? (batch.actual || []).map((a, i) => ({
        actual: a,
        predicted: batch.predicted?.[i],
        error: Math.abs(a - (batch.predicted?.[i] || 0)),
      }))
    : []

  // ── Validate and submit manual prediction ────────────────────────────
  const handlePredict = async (e) => {
    e.preventDefault()
    setSingleError(null)
    setSingle(null)

    // Guard: every key must be non-empty and every value must be a valid number
    const entries = Object.entries(formVals)
    if (entries.length === 0) {
      setSingleError('Enter at least one feature value before predicting.')
      return
    }
    for (const [key, val] of entries) {
      if (!key.trim()) {
        setSingleError(`Feature name cannot be empty. Fill all feature names or load them automatically.`)
        return
      }
      if (val === '' || val === null || val === undefined) {
        setSingleError(`Value for "${key}" is empty. Enter a numeric value for every feature.`)
        return
      }
      if (isNaN(Number(val))) {
        setSingleError(`Value for "${key}" is not a number: "${val}". All values must be numeric.`)
        return
      }
    }

    setPred(true)
    try {
      // Convert all values to numbers before sending
      const input_data = Object.fromEntries(
        entries.map(([k, v]) => [k, Number(v)])
      )
      const res = await predictSingle(input_data)
      setSingle(res.data)
    } catch (e) {
      setSingleError(e.response?.data?.detail || 'Prediction failed.')
    } finally { setPred(false) }
  }

  const allFilled = features.length > 0
    && features.every(f => formVals[f] !== '' && formVals[f] !== undefined)

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="section-title">Predictions</h2>
          <p className="section-subtitle">Explore model predictions on test data and make custom inputs</p>
        </div>
        <button onClick={() => { loadFeatures(); loadBatch() }} className="btn-secondary text-sm">
          <RefreshCw size={14}/> Refresh
        </button>
      </div>

      {error && <div className="card border-danger-500/30 flex items-center gap-3 text-danger-400"><AlertCircle size={18}/>{error}</div>}
      {loading && <div className="card flex items-center gap-3 text-brand-300"><div className="spinner"/><span>Loading predictions…</span></div>}

      {batch && (
        <>
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="card text-center">
              <p className="text-2xl font-bold text-slate-900 dark:text-white">{batch.count}</p>
              <p className="text-xs text-slate-400 mt-1">Test Samples Shown</p>
            </div>
            <div className="card text-center">
              <p className="text-2xl font-bold gradient-text">{batch.model}</p>
              <p className="text-xs text-slate-400 mt-1">Model Used</p>
            </div>
            <div className="card text-center">
              <p className="text-2xl font-bold text-accent-400">
                {scatterData.length > 0
                  ? (scatterData.reduce((s, d) => s + d.error, 0) / scatterData.length).toFixed(3)
                  : '—'}
              </p>
              <p className="text-xs text-slate-400 mt-1">Mean Absolute Error</p>
            </div>
          </div>

          {/* Scatter */}
          <div className="card">
            <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Crosshair size={16} className="text-brand-400"/> Actual vs Predicted (Test Set)
              </h3>
              {/* Legend — one dot per sample, axes explain the colours */}
              <div className="flex items-center gap-5 text-[11px] font-semibold">
                <span className="flex items-center gap-1.5 text-blue-400">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ background: '#60a5fa' }} />
                  X-axis = Actual (true value)
                </span>
                <span className="flex items-center gap-1.5 text-green-400">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ background: '#4ade80' }} />
                  Y-axis = Predicted (model output)
                </span>
                <span className="flex items-center gap-1.5 text-green-400">
                  <span className="inline-block w-3 h-3 rounded-full" style={{ background: '#4ade80', opacity: 0.8 }} />
                  Each dot = 1 sample
                </span>
              </div>
            </div>
            <p className="text-slate-500 text-xs mb-4">
              Each <span className="text-green-400 font-semibold">green dot</span> is one test sample plotted at{' '}
              (<span className="text-blue-400 font-semibold">actual</span>,{' '}
              <span className="text-green-400 font-semibold">predicted</span>).
              Dots on the diagonal = perfect predictions. Dots far off = high error.
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155"/>
                <XAxis
                  dataKey="actual"
                  type="number"
                  name="Actual (y_true)"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  label={{ value: 'Actual (y_true)', position: 'insideBottom', offset: -14, fill: '#60a5fa', fontSize: 12, fontWeight: 700 }}
                />
                <YAxis
                  dataKey="predicted"
                  type="number"
                  name="Predicted (y_pred)"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  label={{ value: 'Predicted (y_pred)', angle: -90, position: 'insideLeft', fill: '#4ade80', fontSize: 12, fontWeight: 700 }}
                />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 12, color: '#e2e8f0', fontSize: 12 }}
                  formatter={(v, name) => [
                    typeof v === 'number' ? v.toFixed(4) : v,
                    name === 'Actual (y_true)' ? '🔵 Actual' : '🟢 Predicted',
                  ]}
                  cursor={{ strokeDasharray: '3 3', stroke: '#475569' }}
                />
                <Scatter
                  data={scatterData}
                  fill="#4ade80"
                  fillOpacity={0.75}
                  r={4}
                  name="Prediction"
                />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-[10px] text-slate-600 mt-1 text-right">
              🔵 X = true label &nbsp;|&nbsp; 🟢 Y = model output &nbsp;|&nbsp; {scatterData.length} samples
            </p>
          </div>

          {/* Prediction table */}
          <div className="card">
            <h3 className="font-semibold text-white mb-4">Sample Predictions</h3>
            <div className="table-wrapper max-h-64 overflow-y-auto">
              <table className="data-table">
                <thead><tr><th>#</th><th>Actual</th><th>Predicted</th><th>Error</th><th>Status</th></tr></thead>
                <tbody>
                  {scatterData.slice(0, 50).map((row, i) => (
                    <tr key={i}>
                      <td className="text-slate-500">{i+1}</td>
                      <td className="font-mono" style={{ color: '#60a5fa' }}>{row.actual?.toFixed?.(3) ?? row.actual}</td>
                      <td className="font-mono" style={{ color: '#4ade80' }}>{row.predicted?.toFixed?.(3) ?? row.predicted}</td>
                      <td className={`font-mono ${row.error > 0.5 ? 'text-warn-400' : 'text-accent-400'}`}>{row.error?.toFixed?.(3)}</td>
                      <td>{row.error === 0 ? <span className="badge badge-success">Exact</span> : row.error > 0.5 ? <span className="badge badge-warn">High Error</span> : <span className="badge badge-success">Good</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── Manual Prediction Form ───────────────────────────────────── */}
      <div className="card">
        <h3 className="font-semibold text-white mb-1 flex items-center gap-2">
          <Send size={16} className="text-brand-400"/> Manual Prediction
          <span className="text-xs text-slate-500">(enter values for each feature)</span>
        </h3>

        {/* Feature source indicator */}
        {featuresLoaded ? (
          <p className="text-xs text-green-400 mb-4 flex items-center gap-1">
            <Info size={12}/>
            {features.length} features loaded from your training session — enter a numeric value for each.
          </p>
        ) : (
          <p className="text-xs text-yellow-400 mb-4 flex items-center gap-1">
            <AlertCircle size={12}/>
            Feature names not found. Train a model first, or enter feature names manually below.
          </p>
        )}

        <form onSubmit={handlePredict} className="space-y-4">
          <div className="bg-surface-700/50 rounded-xl p-4 border border-surface-600">

            {featuresLoaded ? (
              /* Auto-filled feature rows — user only types the value */
              <div className="space-y-2">
                {features.map((col) => (
                  <div key={col} className="flex gap-2 items-center">
                    <span className="input w-1/2 bg-surface-800/60 text-slate-400 text-sm select-none cursor-default truncate">
                      {col}
                    </span>
                    <input
                      id={`feat-${col}`}
                      className="input flex-1"
                      placeholder="numeric value"
                      type="number"
                      step="any"
                      value={formVals[col] ?? ''}
                      onChange={e => setFormVals(prev => ({ ...prev, [col]: e.target.value }))}
                    />
                  </div>
                ))}
              </div>
            ) : (
              /* Manual key-value rows as fallback */
              <div className="space-y-2">
                <p className="text-sm text-slate-400 mb-3">Enter feature name → value pairs (one per row):</p>
                {(Object.keys(formVals).length === 0 ? [''] : Object.keys(formVals)).map((key, i) => (
                  <div key={i} className="flex gap-2">
                    <input
                      className="input w-1/2"
                      placeholder="feature_name"
                      value={key}
                      onChange={e => {
                        const newKey = e.target.value
                        setFormVals(prev => {
                          const entries = Object.entries(prev)
                          entries[i] = [newKey, prev[key] || '']
                          return Object.fromEntries(entries)
                        })
                      }}
                    />
                    <input
                      className="input flex-1"
                      placeholder="value (numeric)"
                      type="number"
                      step="any"
                      value={formVals[key] || ''}
                      onChange={e => setFormVals(prev => ({ ...prev, [key]: e.target.value }))}
                    />
                  </div>
                ))}
                <div className="flex gap-2 mt-3">
                  <button type="button" onClick={() => setFormVals(prev => ({ ...prev, '': '' }))}
                    className="btn-secondary text-xs">+ Add Feature</button>
                  <button type="button" onClick={() => setFormVals({})}
                    className="btn-secondary text-xs text-danger-400">Clear</button>
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={predicting || (featuresLoaded && !allFilled)}
            className="btn-primary"
          >
            {predicting ? <><div className="spinner"/>Predicting…</> : <><Send size={15}/>Predict</>}
          </button>
        </form>

        {singleError && (
          <p className="text-danger-400 text-sm mt-3 flex items-center gap-2">
            <AlertCircle size={15}/>{singleError}
          </p>
        )}

        {singleResult && (
          <div className="mt-4 bg-brand-600/10 border border-brand-500/30 rounded-xl p-4 animate-slide-up">
            <p className="text-sm text-slate-400 mb-1">Prediction Result</p>
            <p className="text-3xl font-bold gradient-text">
              {singleResult.prediction?.toFixed?.(4) ?? singleResult.prediction}
            </p>
            {singleResult.probability && (
              <div className="mt-3 space-y-1">
                <p className="text-xs text-slate-500 mb-2">Class Probabilities:</p>
                {Object.entries(singleResult.probability).map(([cls, prob]) => (
                  <div key={cls} className="flex items-center gap-3">
                    <span className="text-xs text-slate-400 w-12">{cls}</span>
                    <div className="flex-1 bg-surface-700 rounded-full h-2 overflow-hidden">
                      <div className="bg-brand-500 h-full rounded-full" style={{ width: `${(prob * 100).toFixed(0)}%` }}/>
                    </div>
                    <span className="text-xs text-brand-300 w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
