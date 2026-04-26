import React, { useState, useCallback } from 'react'
import { Upload, Table, Target, CheckCircle, AlertCircle, FileText } from 'lucide-react'
import { uploadCSV, setTargetColumn } from '../services/api'

export default function DataUpload() {
  const [dragging, setDragging] = useState(false)
  const [data, setData] = useState(null)
  const [target, setTarget] = useState('')
  const [loading, setLoading] = useState(false)
  const [targetSet, setTargetSet] = useState(false)
  const [error, setError] = useState(null)

  const handleFile = async (file) => {
    if (!file) return
    setLoading(true); setError(null); setData(null); setTargetSet(false)
    try {
      const res = await uploadCSV(file)
      setData(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed.')
    } finally { setLoading(false) }
  }

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const handleSetTarget = async () => {
    if (!target) return
    setLoading(true); setError(null)
    try {
      await setTargetColumn(target)
      setTargetSet(true)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to set target.')
    } finally { setLoading(false) }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="section-title">Data Upload</h2>
        <p className="section-subtitle">Upload a CSV file to begin your ML pipeline</p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => document.getElementById('csv-input').click()}
        className={`border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center cursor-pointer transition-all duration-300
          ${dragging ? 'border-brand-400 bg-brand-500/10 scale-[1.01]' : 'border-surface-600 hover:border-brand-500/60 hover:bg-surface-700/30'}`}
      >
        <div className="w-16 h-16 rounded-2xl bg-brand-600/20 flex items-center justify-center mb-4">
          <Upload size={28} className="text-brand-400" />
        </div>
        <p className="text-white font-semibold text-lg">Drop your CSV file here</p>
        <p className="text-slate-400 text-sm mt-1">or click to browse</p>
        <p className="text-slate-500 text-xs mt-3 flex items-center gap-1">
          <FileText size={12} /> Supports .csv files only
        </p>
        <input id="csv-input" type="file" accept=".csv" className="hidden"
          onChange={(e) => handleFile(e.target.files[0])} />
      </div>

      {loading && (
        <div className="card flex items-center gap-3 text-brand-300">
          <div className="spinner" /><span>Processing dataset…</span>
        </div>
      )}

      {error && (
        <div className="card border-danger-500/30 flex items-center gap-3 text-danger-400">
          <AlertCircle size={18} /><span>{error}</span>
        </div>
      )}

      {/* Dataset info */}
      {data && (
        <div className="space-y-4 animate-slide-up">
          {/* Stats row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Rows', value: data.rows.toLocaleString() },
              { label: 'Columns', value: data.columns.length },
              { label: 'File', value: data.filename },
              { label: 'Missing Cells', value: Object.values(data.missing_counts).reduce((a,b)=>a+b,0) },
            ].map(({ label, value }) => (
              <div key={label} className="card text-center py-4">
                <p className="text-2xl font-bold text-white">{value}</p>
                <p className="text-xs text-slate-400 mt-1">{label}</p>
              </div>
            ))}
          </div>

          {/* Target column selector */}
          <div className="card space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Target size={18} className="text-brand-400" />
              <h3 className="font-semibold text-white">Select Target Column</h3>
            </div>
            <div className="flex gap-3">
              <select value={target} onChange={e => setTarget(e.target.value)} className="select flex-1">
                <option value="">-- Choose target column --</option>
                {data.columns.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
              <button onClick={handleSetTarget} disabled={!target || loading} className="btn-primary whitespace-nowrap">
                <CheckCircle size={16} /> Set Target
              </button>
            </div>
            {targetSet && (
              <div className="flex items-center gap-2 text-accent-400 text-sm">
                <CheckCircle size={16} />
                Target column set to <strong>"{target}"</strong>
              </div>
            )}
          </div>

          {/* Column info */}
          <div className="card">
            <div className="flex items-center gap-2 mb-4">
              <Table size={18} className="text-brand-400" />
              <h3 className="font-semibold text-white">Column Overview</h3>
            </div>
            <div className="table-wrapper">
              <table className="data-table">
                <thead><tr>
                  <th>Column</th><th>Type</th><th>Missing</th><th>Status</th>
                </tr></thead>
                <tbody>
                  {data.columns.map(col => {
                    const miss = data.missing_counts[col] || 0
                    return (
                      <tr key={col}>
                        <td className="font-mono text-brand-300">{col}</td>
                        <td><span className="badge badge-info">{data.dtypes[col]}</span></td>
                        <td className={miss > 0 ? 'text-warn-400' : 'text-accent-400'}>{miss}</td>
                        <td>
                          {col === target
                            ? <span className="badge badge-success">🎯 Target</span>
                            : miss > 0
                            ? <span className="badge badge-warn">Has Missing</span>
                            : <span className="badge badge-success">Clean</span>}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Data preview */}
          <div className="card">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Table size={18} className="text-brand-400" /> Data Preview
              <span className="text-xs text-slate-500">(first 100 rows)</span>
            </h3>
            <div className="table-wrapper max-h-72 overflow-y-auto">
              <table className="data-table">
                <thead><tr>
                  {data.columns.map(c => <th key={c}>{c}</th>)}
                </tr></thead>
                <tbody>
                  {data.preview.slice(0, 50).map((row, i) => (
                    <tr key={i}>
                      {data.columns.map(c => (
                        <td key={c} className="font-mono text-xs">{row[c] ?? '—'}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
